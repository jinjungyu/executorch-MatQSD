# MatQSD: Matryoshka Quantized Speculative Decoding on XNNPACK

단일 weight에서 4-bit draft / 8-bit target inference를 runtime mode 전환으로 지원하는 XNNPACK 기반 구현.

## 성능 (Llama-3.2-1B-Instruct, M4 MacBook, 256 tokens)

| Mode | TPS | 비교 |
|------|-----|------|
| mqint8 4-bit (draft) | **87** | qb4w 90 TPS 대비 97% |
| mqint8 8-bit (target) | **55** | 8da8w 53 TPS와 동일 |
| Speculative Decoding (K=3) | **42** | α=80%, τ=3.4 |

## 아키텍처 개요

```
.pte 파일 (단일)
  └─ 각 FC layer: mqint8 packed weights
       ├─ 4-bit region: [vksum4 | {upper_tiled | scale}×G | bias]  ← NR block 순차 배치
       └─ 8-bit extra:  [vksum8 | {lower_tiled}×G]                 ← 전체 뒤에 분리

Runtime mode 전환:
  MQINT8_MODE=0  →  4-bit: upper nibble만 읽음 (87 TPS)
  MQINT8_MODE=1  →  8-bit: upper+lower 복원 (55 TPS)
```

### Packed Layout (Global Region Separation)

```
Packed buffer:
  ┌─ 4-bit NR blocks (순차, skip 0) ─────────────────┐
  │ NR0: [vksum4(64B) | {upper(1024B)|scale(32B)}×G | bias(64B)] │
  │ NR1: [vksum4(64B) | {upper(1024B)|scale(32B)}×G | bias(64B)] │
  │ ...                                                            │
  ├─ 8-bit extra region ──────────────────────────────┤
  │ NR0: [vksum8(64B) | {lower(1024B)}×G]                        │
  │ NR1: [vksum8(64B) | {lower(1024B)}×G]                        │
  │ ...                                                            │
  └────────────────────────────────────────────────────┘

4-bit: w pointer가 전체 NR block을 완전 순차 순회 (qb4w와 동일 패턴)
8-bit: w로 upper+scale, 별도 wl pointer로 lower region 접근
```

## 파일 구조

### XNNPACK 커널 & 인프라

| 파일 | 역할 |
|------|------|
| `XNNPACK/src/qd8-f32-mqint8-gemm/gen/qd8-f32-mqint8-gemm-1x16c8-minmax-neoni8mm.c` | **i8mm 커널** (M4 기본) |
| `XNNPACK/src/qd8-f32-mqint8-gemm/gen/qd8-f32-mqint8-gemm-1x16c4-minmax-neondot.c` | dotprod 커널 (fallback) |
| `XNNPACK/src/reference/packing.cc` | Two-pass packer (4-bit→lower 전역 분리) |
| `XNNPACK/src/configs/gemm-config.c` | i8mm/dotprod 커널 등록 |
| `XNNPACK/src/operators/fully-connected-nc.c` | Operator create/reshape (w_stride override, lower params) |
| `XNNPACK/src/xnnpack/microparams.h` | `xnn_f32_mqint8_minmax_params` struct |
| `XNNPACK/src/microparams-init.c` | Params 초기화 |
| `XNNPACK/src/xnnpack/gemm.h` | 커널 선언 |
| `XNNPACK/src/xnnpack/pack.h` | Packer 선언 |
| `XNNPACK/src/xnnpack/compute.h` | gemm_context params union |
| `XNNPACK/cmake/gen/neoni8mm_microkernels.cmake` | i8mm 커널 빌드 등록 |
| `XNNPACK/cmake/gen/neondot_microkernels.cmake` | dotprod 커널 빌드 등록 |

### ExecuTorch 직렬화 & 컴파일

| 파일 | 역할 |
|------|------|
| `backends/xnnpack/operators/node_visitor.py` | `convert_to_mqint8()` + `MQINT8_APPLY_ROUNDING` |
| `backends/xnnpack/serialization/schema.fbs` | `xnn_datatype_mqint8 = 15` |
| `backends/xnnpack/serialization/runtime_schema.fbs` | 동일 |
| `backends/xnnpack/serialization/xnnpack_graph_schema.py` | Python enum |
| `backends/xnnpack/runtime/XNNCompiler.cpp` | mqint8 tensor 정의 (zp=128) |

### Runner & Export

| 파일 | 역할 |
|------|------|
| `examples/models/llama/sd_runner.cpp` | SD runner (batch verification 지원) |
| `MatQSD/m4_deploy/matqsd_src/export_matqsd_8bit.py` | Export 스크립트 |

## Export

### 기본 export (decode 전용, AR inference)

```bash
cd /tmp
python export_matqsd_8bit.py \
  --matgptq_dir ~/weights/Llama-3.2-1B-Instruct-zp128 \
  -- \
  -c ~/.cache/.../original/consolidated.00.pth \
  -p ~/.cache/.../original/params.json \
  -kv --use_sdpa_with_kv_cache \
  -d fp32 \
  --metadata '{"get_bos_id":128000,"get_eos_ids":[128009,128001]}' \
  --max_seq_length 256 --max_context_length 256 \
  -X \
  --output_name matqsd_mqint8.pte
```

### SD용 export (batch verification, full logits)

```bash
# --generate_full_logits 추가: 모든 position의 logits 반환
cd /tmp
python export_matqsd_8bit.py \
  --matgptq_dir ~/weights/Llama-3.2-1B-Instruct-zp128 \
  -- \
  -c ~/.cache/.../original/consolidated.00.pth \
  -p ~/.cache/.../original/params.json \
  -kv --use_sdpa_with_kv_cache \
  -d fp32 \
  --metadata '{"get_bos_id":128000,"get_eos_ids":[128009,128001]}' \
  --max_seq_length 256 --max_context_length 256 \
  -X \
  --generate_full_logits \
  --output_name matqsd_mqint8_fulllogits.pte
```

## 빌드

```bash
# ExecuTorch + XNNPACK 빌드
cmake --build cmake-out -j10

# XNNPACK 라이브러리 복사 (incremental build 시)
cp cmake-out/backends/xnnpack/third-party/XNNPACK/lib*.a cmake-out/lib/

# llama_main 빌드
cmake --build cmake-out/examples/models/llama -j10 --target llama_main

# sd_runner 빌드
cmake --build cmake-out/examples/models/llama -j10 --target sd_runner
```

## 실험

### AR Inference (4-bit / 8-bit)

```bash
# 4-bit (draft mode)
MQINT8_MODE=0 cmake-out/examples/models/llama/llama_main \
  --model_path=matqsd_mqint8.pte \
  --tokenizer_path=tokenizer.bin \
  --prompt="The capital of France is" \
  --seq_len=256 --temperature=0

# 8-bit (target mode)
MQINT8_MODE=1 cmake-out/examples/models/llama/llama_main \
  --model_path=matqsd_mqint8.pte \
  --tokenizer_path=tokenizer.bin \
  --prompt="The capital of France is" \
  --seq_len=256 --temperature=0
```

### Speculative Decoding

```bash
# SD with batch verification (full logits model 필요)
cmake-out/examples/models/llama/sd_runner \
  -mqint8_model=matqsd_mqint8_fulllogits.pte \
  -tokenizer_path=tokenizer.bin \
  -prompt="The capital of France is" \
  -max_new_tokens=128 \
  -K=5 \
  -seq_len=256

# K값 비교 테스트
for K in 3 5 7; do
  echo "=== K=$K ==="
  cmake-out/examples/models/llama/sd_runner \
    -mqint8_model=matqsd_mqint8_fulllogits.pte \
    -tokenizer_path=tokenizer.bin \
    -prompt="The capital of France is" \
    -max_new_tokens=128 -K=$K -seq_len=256
done
```

### 참조 모델 속도 측정

```bash
# qb4w (4-bit reference)
cmake-out/examples/models/llama/llama_main \
  --model_path=llama_8da4w_emb4.pte \
  --tokenizer_path=tokenizer.bin \
  --prompt="The capital of France is" --seq_len=256 --temperature=0

# 8da8w (8-bit reference)
cmake-out/examples/models/llama/llama_main \
  --model_path=llama_8da8w_emb4.pte \
  --tokenizer_path=tokenizer.bin \
  --prompt="The capital of France is" --seq_len=256 --temperature=0
```

## 최적화 히스토리

| 단계 | 4-bit TPS | 변화 | 핵심 |
|------|----------|------|------|
| 초기 (debug 포함) | 49 | - | dotprod c4, per-group interleaved layout |
| Debug 제거 | 50 | +2% | fprintf, static vars 제거 |
| i8mm 커널 | 50 | ±0 | vmmlaq_s32, c8 tiling (compute bound 아님) |
| NR block 내 region 분리 | 63 | +26% | [upper\|scale\|lower] → 4-bit region + 8-bit region |
| **Global region 분리** | **88** | **+40%** | Lower를 전체 buffer 뒤로 이동, w_stride override |
| vksum8 분리 | 87 | ±0 | 4-bit region에 skip 0, 구조적 완성 |

## Scale 체인

```
Checkpoint: scale_8bit (per-group, float16)
Export:     w.scale = scale_8bit (zp128 convention)
Serializer: .to(bf16) → .pte
Packer:     stored_scale = scale_8bit / 16 (bf16)
Kernel:     effective_scale = stored_scale × 16 = scale_8bit
```

## Rounding Reversal

3곳 동기화 필요 (`#define MQINT8_ROUND_REVERSAL`):
1. `node_visitor.py`: `MQINT8_APPLY_ROUNDING = True`
2. `packing.cc`: `#define MQINT8_ROUND_REVERSAL`
3. kernel `.c` files: `#define MQINT8_ROUND_REVERSAL`

## 모델 파일

| 파일 | 설명 |
|------|------|
| `matqsd_mqint8_zp128_rounded.pte` | AR inference용 (decode only logits) |
| `matqsd_mqint8_zp128_fulllogits.pte` | SD용 (all position logits) |
| `llama_8da4w_emb4.pte` | qb4w 참조 모델 |
| `llama_8da8w_emb4.pte` | 8da8w 참조 모델 |

## 알려진 제한사항

1. **SD speedup < 1×**: draft/target 속도 비율이 87/55=1.6×로, full logits output projection 비용 때문에 batch verification이 AR보다 느림. Output projection 최적화 또는 hidden state 기반 비교 필요.
2. **모델 2개 로드**: SD runner가 draft/target을 별도 인스턴스로 로드 (메모리 2×). Single-instance mode switching 구현 가능하나 미구현.
3. **NEON fast packer 미구현**: Reference C packer 사용 중. 모델 로드 시간 ~2.5s.
4. **macOS i8mm detection**: `cpuinfo_has_arm_i8mm()`가 macOS에서 동작하지 않아 compile-time `#if XNN_ENABLE_ARM_I8MM`로 강제 선택.
