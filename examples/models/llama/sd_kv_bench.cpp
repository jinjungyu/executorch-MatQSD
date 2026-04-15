/* MatQSD SD benchmark — standard pipeline, dynamic shape.
 *
 * Draft (4-bit):  K × step1 (seq_len=1, autoregressive)
 * Target (8-bit): 1 × step_batch (seq_len=K+1, batch verify, full logits)
 * AR baseline:    8-bit step1 (seq_len=1, no full logits)
 */
#include <gflags/gflags.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/runtime.h>
#include <chrono>
#include <iostream>
#include <vector>

DEFINE_string(draft, "", "Draft 4-bit .pte");
DEFINE_string(target, "", "Target 8-bit .pte (generate_full_logits)");
DEFINE_string(ar_baseline, "", "AR baseline 8-bit .pte (no full logits)");
DEFINE_int32(max_new_tokens, 50, "Max new tokens");
DEFINE_int32(K, 5, "Draft lookahead");

using namespace executorch::extension;
using namespace executorch::runtime;

static int64_t argmax(const float* data, int64_t n) {
  int64_t best = 0;
  for (int64_t i = 1; i < n; i++)
    if (data[i] > data[best]) best = i;
  return best;
}

static int64_t step1(Module& m, int64_t token, int64_t pos) {
  auto tok = make_tensor_ptr({1, 1}, std::vector<int64_t>{token});
  auto ipos = make_tensor_ptr({1}, std::vector<int64_t>{pos});
  std::vector<EValue> in = {*tok, *ipos};
  auto out = m.execute("forward", in);
  if (!out.ok()) return -1;
  auto& l = out.get()[0].toTensor();
  int64_t v = l.size(l.dim() - 1);
  return argmax(l.const_data_ptr<float>() + (l.numel() - v), v);
}

// Batch forward, returns argmax for EACH position
static std::vector<int64_t> step_batch(Module& m,
    const std::vector<int64_t>& tokens, int64_t start_pos) {
  int n = tokens.size();
  auto tok = make_tensor_ptr({1, n}, std::vector<int64_t>(tokens));
  auto ipos = make_tensor_ptr({1}, std::vector<int64_t>{start_pos});
  std::vector<EValue> in = {*tok, *ipos};
  auto out = m.execute("forward", in);
  if (!out.ok()) { return {}; }
  auto& l = out.get()[0].toTensor();
  int64_t v = l.size(l.dim() - 1);
  auto* d = l.const_data_ptr<float>();
  std::vector<int64_t> result;
  if (l.dim() == 3) {
    for (int t = 0; t < l.size(1); t++)
      result.push_back(argmax(d + t * v, v));
  } else {
    result.push_back(argmax(d, v));
  }
  return result;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  runtime_init();
  int K = FLAGS_K;
  int64_t prompt[] = {128000, 791, 6864, 315, 9822, 374};
  int plen = 6;

  // ==================== AR 8-bit baseline ====================
  double ar8_tps = 0;
  {
    std::cout << "=== AR 8-bit baseline ===" << std::endl;
    Module m(FLAGS_ar_baseline);
    m.load_method("forward");
    int64_t cur = 0;
    for (int i = 0; i < plen; i++) cur = step1(m, prompt[i], i);
    std::cout << "First: " << cur << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    int gen = 0; int64_t pos = plen;
    while (gen < FLAGS_max_new_tokens) {
      cur = step1(m, cur, pos++);
      if (cur < 0 || cur == 128009 || cur == 128001) break;
      gen++;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    ar8_tps = gen / (ms / 1000.0);
    std::cout << "AR-8bit: " << gen << " tok, " << ms << " ms, " << ar8_tps << " TPS\n" << std::endl;
  }

  // ==================== AR 4-bit ====================
  {
    std::cout << "=== AR 4-bit ===" << std::endl;
    Module m(FLAGS_draft);
    m.load_method("forward");
    int64_t cur = 0;
    for (int i = 0; i < plen; i++) cur = step1(m, prompt[i], i);
    std::cout << "First: " << cur << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    int gen = 0; int64_t pos = plen;
    while (gen < FLAGS_max_new_tokens) {
      cur = step1(m, cur, pos++);
      if (cur < 0 || cur == 128009 || cur == 128001) break;
      gen++;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "AR-4bit: " << gen << " tok, " << ms << " ms, " << gen/(ms/1000.0) << " TPS\n" << std::endl;
  }

  // ==================== Speculative Decoding ====================
  {
    std::cout << "=== SD (K=" << K << ") ===" << std::endl;
    Module draft(FLAGS_draft);
    draft.load_method("forward");
    Module target(FLAGS_target);
    target.load_method("forward");

    // Prefill draft token-by-token
    int64_t cur_d = 0;
    for (int i = 0; i < plen; i++) cur_d = step1(draft, prompt[i], i);

    // Prefill target with batch (K+1 = 6 = plen)
    std::vector<int64_t> pvec(prompt, prompt + plen);
    auto tpreds = step_batch(target, pvec, 0);
    int64_t cur = tpreds.empty() ? -1 : tpreds.back();
    std::cout << "First (target): " << cur << std::endl;
    if (cur < 0) { std::cerr << "Target prefill failed" << std::endl; return 1; }

    int64_t pos = plen;
    int total_gen = 0, total_drafted = 0, total_accepted = 0, total_steps = 0;
    double draft_ms = 0, target_ms = 0;
    auto sd_start = std::chrono::high_resolution_clock::now();

    while (total_gen < FLAGS_max_new_tokens) {
      // Draft: K × step1
      auto d0 = std::chrono::high_resolution_clock::now();
      std::vector<int64_t> dtoks;
      int64_t dtok = cur;
      for (int d = 0; d < K; d++) {
        dtok = step1(draft, dtok, pos + d);
        if (dtok < 0) break;
        dtoks.push_back(dtok);
      }
      auto d1 = std::chrono::high_resolution_clock::now();
      draft_ms += std::chrono::duration<double, std::milli>(d1 - d0).count();
      total_drafted += dtoks.size();
      if (dtoks.empty()) break;

      // Target: 1 × step_batch(K+1)
      auto v0 = std::chrono::high_resolution_clock::now();
      std::vector<int64_t> verify_in = {cur};
      for (auto t : dtoks) verify_in.push_back(t);
      while ((int)verify_in.size() < K + 1) verify_in.push_back(verify_in.back());

      auto tpreds = step_batch(target, verify_in, pos - 1);
      auto v1 = std::chrono::high_resolution_clock::now();
      target_ms += std::chrono::duration<double, std::milli>(v1 - v0).count();

      // Accept/reject
      int n_acc = 0;
      for (int i = 0; i < (int)dtoks.size() && i < (int)tpreds.size(); i++) {
        if (tpreds[i] == dtoks[i]) n_acc++;
        else break;
      }
      total_accepted += n_acc;
      total_gen += n_acc;

      int64_t bonus = (n_acc < (int)tpreds.size()) ? tpreds[n_acc] : tpreds.back();
      total_gen++;
      cur = bonus;
      pos += n_acc + 1;
      total_steps++;
      if (bonus == 128009 || bonus == 128001) break;
    }

    auto sd_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(sd_end - sd_start).count();
    double alpha = (double)total_accepted / std::max(total_drafted, 1);
    double tau = (double)total_gen / std::max(total_steps, 1);
    double sd_tps = total_gen / elapsed;

    std::cout << "\n=== SD Results ===" << std::endl;
    std::cout << "Generated: " << total_gen << " tok / " << total_steps << " steps" << std::endl;
    std::cout << "Drafted: " << total_drafted << ", Accepted: " << total_accepted << std::endl;
    std::cout << "alpha: " << alpha << ", tau: " << tau << std::endl;
    std::cout << "Draft: " << draft_ms << " ms (" << draft_ms/total_steps << " ms/step)" << std::endl;
    std::cout << "Target: " << target_ms << " ms (" << target_ms/total_steps << " ms/step)" << std::endl;
    std::cout << "SD TPS: " << sd_tps << std::endl;
    std::cout << "AR-8bit TPS: " << ar8_tps << std::endl;
    std::cout << "SD Speedup: " << sd_tps / ar8_tps << "x" << std::endl;
  }
  return 0;
}
