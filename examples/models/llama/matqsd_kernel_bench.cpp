/*
 * MatQSD kernel bench v3: tile-packed, 16-channel parallel, dotprod, MT.
 *
 * Key insight: process NR=16 output channels simultaneously per tile,
 * reusing the same activation data — matching XNNPACK's approach.
 *
 * Build:
 *   clang++ -std=c++17 -O3 -march=armv8.2-a+dotprod+fp16 \
 *     matqsd_kernel_bench.cpp -o matqsd_kernel_bench -lpthread
 */

#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include <algorithm>

static constexpr int NR = 16;
static constexpr int KR = 16;  // process 16 k-elements per inner iteration
static constexpr int GROUP_SIZE = 128;
static constexpr int REPEATS = 200;

// ================================================================
//  Tile packing: [n_out/NR, n_in/KR, NR, KR] contiguous
//  Within each tile: NR rows of KR elements, row-major
//  This means: for a tile at (ot, kt), data is:
//    channel[ot*NR + 0], k[kt*KR .. kt*KR+KR-1]
//    channel[ot*NR + 1], k[kt*KR .. kt*KR+KR-1]
//    ...
//    channel[ot*NR + 15], k[kt*KR .. kt*KR+KR-1]
// ================================================================

struct TilePacked {
  std::vector<int8_t> upper;    // pre-computed: rounded_upper*16-124
  std::vector<uint8_t> lower;   // lower nibble (0-15)
  std::vector<float> scales;    // [n_out, n_groups]
  int n_out, n_in, n_groups;
};

static TilePacked tile_pack(
    const uint8_t* w_split, const float* scales_raw,
    int n_out, int n_in, int group_size) {
  TilePacked tp;
  tp.n_out = n_out; tp.n_in = n_in;
  tp.n_groups = n_in / group_size;
  int half_in = n_in / 2;
  int n_ot = (n_out + NR - 1) / NR;
  int n_kt = n_in / KR;

  tp.upper.resize(n_ot * n_kt * NR * KR, 0);
  tp.lower.resize(n_ot * n_kt * NR * KR, 0);
  tp.scales.assign(scales_raw, scales_raw + n_out * tp.n_groups);

  for (int ot = 0; ot < n_ot; ot++) {
    for (int kt = 0; kt < n_kt; kt++) {
      int tile_base = (ot * n_kt + kt) * NR * KR;
      for (int ch = 0; ch < NR; ch++) {
        int row = ot * NR + ch;
        if (row >= n_out) break;
        const uint8_t* ur = w_split + row * n_in;
        const uint8_t* lr = ur + half_in;
        for (int ki = 0; ki < KR; ki++) {
          int k = kt * KR + ki;
          int pi = k / 2;
          int odd = k & 1;
          uint8_t ub = ur[pi], lb = lr[pi];
          uint8_t un = odd ? (ub & 0xF) : (ub >> 4);
          uint8_t ln = odd ? (lb & 0xF) : (lb >> 4);
          tp.upper[tile_base + ch * KR + ki] = (int8_t)((int)(un << 4) - 124);
          tp.lower[tile_base + ch * KR + ki] = ln;
        }
      }
    }
  }
  return tp;
}

// ================================================================
//  Activation quantization
// ================================================================
struct ActQ {
  std::vector<int8_t> data;
  std::vector<float> scales;
};

static ActQ quant_act(const float* x, int n_in, int gs) {
  ActQ aq;
  int ng = n_in / gs;
  aq.data.resize(n_in);
  aq.scales.resize(ng);
  for (int g = 0; g < ng; g++) {
    const float* xg = x + g * gs;
    float amax = 0;
    for (int k = 0; k < gs; k++) amax = std::max(amax, std::abs(xg[k]));
    float sc = (amax > 0) ? amax / 127.0f : 1.0f;
    float inv = (amax > 0) ? 127.0f / amax : 0.0f;
    aq.scales[g] = sc;
    for (int k = 0; k < gs; k++)
      aq.data[g*gs+k] = (int8_t)std::max(-127, std::min(127, (int)roundf(xg[k]*inv)));
  }
  return aq;
}

// ================================================================
//  Kernel: process 16 output channels per tile group
//
//  For each tile (16 channels × KR elements):
//    Load activation[KR] once
//    Load weight[ch, KR] for ch=0..15 → 16 vdotq_s32 calls
//    Accumulate 16 separate int32 accumulators
// ================================================================

static void gemv_tile16(
    float* y, const ActQ& aq, const TilePacked& tp, int mode,
    int ot_start, int ot_end) {

  int n_kt = tp.n_in / KR;
  int tiles_per_group = GROUP_SIZE / KR;

  for (int ot = ot_start; ot < ot_end; ot++) {
    // 16 output channels: accumulators
    float acc[NR] = {};
    int n_groups = tp.n_groups;

    for (int g = 0; g < n_groups; g++) {
      float x_sc = aq.scales[g];
      const int8_t* xg = aq.data.data() + g * GROUP_SIZE;

      // Int32 accumulators for this group, 16 channels
      int32x4_t dot[4] = {vdupq_n_s32(0), vdupq_n_s32(0),
                          vdupq_n_s32(0), vdupq_n_s32(0)};

      int kt_start = g * tiles_per_group;
      for (int ti = 0; ti < tiles_per_group; ti++) {
        int kt = kt_start + ti;
        int tile_base = (ot * n_kt + kt) * NR * KR;
        int k_off = ti * KR;

        // Load activation once for this tile
        int8x16_t xv = vld1q_s8(xg + k_off);

        if (mode == 0) {
          // 4-bit: upper tiles are pre-computed int8, load directly
          // 16 channels × KR=16 bytes each, but we use vdotq per channel
          // vdotq_s32 does 4 independent dot products of 4 int8 each
          // For KR=16: need to split into 4-element chunks

          for (int ch4 = 0; ch4 < NR; ch4 += 4) {
            // Load 4 channels' weights (4 × 16 bytes = 64 bytes)
            int8x16_t w0 = vld1q_s8(tp.upper.data() + tile_base + (ch4+0)*KR);
            int8x16_t w1 = vld1q_s8(tp.upper.data() + tile_base + (ch4+1)*KR);
            int8x16_t w2 = vld1q_s8(tp.upper.data() + tile_base + (ch4+2)*KR);
            int8x16_t w3 = vld1q_s8(tp.upper.data() + tile_base + (ch4+3)*KR);

            // Each vdotq_s32 produces 4 int32 from 4×4 dot products
            // But we need full 16-element dot product per channel
            // So we use vdotq_s32 with the SAME xv for each channel
            int32x4_t d0 = vdotq_s32(vdupq_n_s32(0), w0, xv);
            int32x4_t d1 = vdotq_s32(vdupq_n_s32(0), w1, xv);
            int32x4_t d2 = vdotq_s32(vdupq_n_s32(0), w2, xv);
            int32x4_t d3 = vdotq_s32(vdupq_n_s32(0), w3, xv);

            // Horizontal sum each: 4 int32 → 1 int32 per channel
            // Pack 4 channel sums into one int32x4
            int32x4_t sums = {vaddvq_s32(d0), vaddvq_s32(d1),
                              vaddvq_s32(d2), vaddvq_s32(d3)};
            dot[ch4/4] = vaddq_s32(dot[ch4/4], sums);
          }
        } else {
          // 8-bit: reconstruct from upper + lower
          for (int ch4 = 0; ch4 < NR; ch4 += 4) {
            int32_t sums[4];
            for (int c = 0; c < 4; c++) {
              int idx = tile_base + (ch4+c)*KR;
              int8x16_t up = vld1q_s8(tp.upper.data() + idx);
              uint8x16_t lo = vld1q_u8(tp.lower.data() + idx);
              uint8x16_t rb = vshlq_n_u8(vandq_u8(vshrq_n_u8(lo, 3), vdupq_n_u8(1)), 4);
              int8x16_t wv = vsubq_s8(vaddq_s8(up, vreinterpretq_s8_u8(lo)),
                                       vreinterpretq_s8_u8(rb));
              int32x4_t d = vdotq_s32(vdupq_n_s32(0), wv, xv);
              sums[c] = vaddvq_s32(d);
            }
            dot[ch4/4] = vaddq_s32(dot[ch4/4], vld1q_s32(sums));
          }
        }
      }

      // Apply scales for this group: acc += int32_sum * x_scale * w_scale
      for (int ch4 = 0; ch4 < NR; ch4 += 4) {
        int row_base = ot * NR + ch4;
        float32x4_t w_sc = vld1q_f32(tp.scales.data() + row_base * n_groups + g);
        // But scales are per-row-per-group, not contiguous for 4 channels
        // Load individually
        float ws[4];
        for (int c = 0; c < 4; c++)
          ws[c] = tp.scales[(row_base+c) * n_groups + g];
        float32x4_t w_sc4 = vld1q_f32(ws);

        float32x4_t fsum = vcvtq_f32_s32(dot[ch4/4]);
        float32x4_t contrib = vmulq_f32(vmulq_n_f32(fsum, x_sc), w_sc4);

        // Accumulate
        float32x4_t prev = vld1q_f32(acc + ch4);
        vst1q_f32(acc + ch4, vaddq_f32(prev, contrib));
      }

      // Reset dots for next group
      dot[0] = dot[1] = dot[2] = dot[3] = vdupq_n_s32(0);
    }

    // Store 16 outputs
    int row_base = ot * NR;
    int n_valid = std::min(NR, tp.n_out - row_base);
    for (int c = 0; c < n_valid; c++)
      y[row_base + c] = acc[c];
  }
}

static void gemv_tile16_mt(float* y, const ActQ& aq, const TilePacked& tp, int mode, int nt) {
  int n_ot = (tp.n_out + NR - 1) / NR;
  if (nt <= 1) { gemv_tile16(y, aq, tp, mode, 0, n_ot); return; }
  std::vector<std::thread> threads;
  int chunk = (n_ot + nt - 1) / nt;
  for (int t = 0; t < nt; t++) {
    int s = t*chunk, e = std::min(s+chunk, n_ot);
    if (s < e) threads.emplace_back(gemv_tile16, y, std::cref(aq), std::cref(tp), mode, s, e);
  }
  for (auto& th : threads) th.join();
}

// ================================================================
//  Reference
// ================================================================
static void ref(float* y, const float* x, const uint8_t* w, const float* sc,
                int no, int ni, int gs, int mode) {
  int ng=ni/gs, hi=ni/2, hgs=gs/2;
  for (int r=0;r<no;r++){
    float a=0; const uint8_t*ur=w+r*ni,*lr=ur+hi; const float*sr=sc+r*ng;
    for(int g=0;g<ng;g++){float s=sr[g];const uint8_t*ug=ur+g*hgs,*lg=lr+g*hgs;const float*xg=x+g*gs;
      for(int k=0;k<hgs;k++){uint8_t u=ug[k],l=lg[k];uint8_t re=u>>4,le=l>>4,ro=u&0xF,lo_=l&0xF;
        float ve,vo;if(!mode){ve=((float)re*16-124)*s;vo=((float)ro*16-124)*s;}
        else{ve=(float)((int)(((re-((le>>3)&1))<<4)|le)-124)*s;vo=(float)((int)(((ro-((lo_>>3)&1))<<4)|lo_)-124)*s;}
        a+=ve*xg[k*2]+vo*xg[k*2+1];}} y[r]=a;}
}

// ================================================================
int main() {
  const int NO=2048, NI=2048;
  printf("MatQSD v3: tile16 + dotprod + MT\nNO=%d NI=%d\n\n",NO,NI);

  srand(42);
  std::vector<float> x(NI), sc(NO*(NI/GROUP_SIZE));
  std::vector<uint8_t> w(NO*NI);
  for(auto&v:x)v=(float)rand()/RAND_MAX*2-1;
  for(auto&v:sc)v=(float)rand()/RAND_MAX*0.01f;
  for(auto&v:w)v=rand()%248;

  auto tp=tile_pack(w.data(),sc.data(),NO,NI,GROUP_SIZE);
  std::vector<float> yr(NO),yt(NO);

  for(int mode=0;mode<=1;mode++){
    printf("=== %s ===\n",mode?"8-bit":"4-bit");
    ref(yr.data(),x.data(),w.data(),sc.data(),NO,NI,GROUP_SIZE,mode);
    auto aq=quant_act(x.data(),NI,GROUP_SIZE);
    gemv_tile16_mt(yt.data(),aq,tp,mode,1);

    double cs=0,nr2=0,nt2=0;float me=0;
    for(int i=0;i<NO;i++){float e=std::abs(yt[i]-yr[i]);if(e>me)me=e;
      cs+=yt[i]*yr[i];nr2+=yr[i]*yr[i];nt2+=yt[i]*yt[i];}
    cs/=sqrt(nr2*nt2);
    printf("  cosine=%.6f max_err=%.4f\n",cs,me);

    for(int nt:{1,5}){
      auto aqb=quant_act(x.data(),NI,GROUP_SIZE);
      for(int r=0;r<10;r++)gemv_tile16_mt(yt.data(),aqb,tp,mode,nt);
      auto t0=std::chrono::high_resolution_clock::now();
      for(int r=0;r<REPEATS;r++)gemv_tile16_mt(yt.data(),aqb,tp,mode,nt);
      auto t1=std::chrono::high_resolution_clock::now();
      double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
      printf("  %dT: %.4f ms/call (%.1f ms total)\n",nt,ms/REPEATS,ms);
    }
    printf("\n");
  }
}
