/*
 * MatQSD SD Benchmark: measures actual draft-verify cycle timing.
 *
 * Without KV cache, we can't do autoregressive generation, but we CAN
 * measure the real wall-clock cost of:
 *   - K forward passes on draft (4-bit) model
 *   - 1 forward pass on target (8-bit) model
 *
 * Combined with α/τ from server measurements, this gives projected SD TPS.
 */

#include <gflags/gflags.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/runtime.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

DEFINE_string(draft_model, "", "Draft (4-bit) .pte");
DEFINE_string(target_model, "", "Target (8-bit) .pte");
DEFINE_int32(K, 5, "Draft tokens per SD step");
DEFINE_int32(cycles, 20, "Number of SD cycles to benchmark");
DEFINE_double(alpha, 0.653, "Acceptance rate (from server measurement)");
DEFINE_double(tau, 4.33, "Tokens per SD step (from server measurement)");
DEFINE_int32(seq_len, 1, "Sequence length per forward (1=single token)");

using namespace executorch::extension;
using namespace executorch::runtime;

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  runtime_init();

  // ---- Load models ----
  std::cout << "Loading draft model (4-bit)..." << std::endl;
  Module draft(FLAGS_draft_model);
  auto err = draft.load_method("forward");
  if (err != Error::Ok) { std::cerr << "Draft load failed: " << (int)err << std::endl; return 1; }

  std::cout << "Loading target model (8-bit)..." << std::endl;
  Module target(FLAGS_target_model);
  err = target.load_method("forward");
  if (err != Error::Ok) { std::cerr << "Target load failed: " << (int)err << std::endl; return 1; }

  // ---- Prepare input ----
  std::vector<int64_t> tokens(FLAGS_seq_len, 128000); // BOS token repeated
  auto ids = make_tensor_ptr({1, FLAGS_seq_len}, tokens);

  // ---- Warmup ----
  std::cout << "Warming up..." << std::endl;
  for (int i = 0; i < 3; i++) {
    std::vector<EValue> d_in = {*ids}, t_in = {*ids};
    draft.execute("forward", d_in);
    target.execute("forward", t_in);
  }

  // ---- Benchmark: AR baselines ----
  std::cout << "\n=== AR Baselines (seq_len=" << FLAGS_seq_len << ") ===" << std::endl;

  // AR 8-bit (target only)
  {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < FLAGS_cycles; i++) {
      std::vector<EValue> in = {*ids};
      target.execute("forward", in);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_per_fwd = ms / FLAGS_cycles;
    std::cout << "AR-8bit: " << ms_per_fwd << " ms/forward ("
              << 1000.0 / ms_per_fwd << " fwd/s)" << std::endl;
  }

  // AR 4-bit (draft only)
  {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < FLAGS_cycles; i++) {
      std::vector<EValue> in = {*ids};
      draft.execute("forward", in);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_per_fwd = ms / FLAGS_cycles;
    std::cout << "AR-4bit: " << ms_per_fwd << " ms/forward ("
              << 1000.0 / ms_per_fwd << " fwd/s)" << std::endl;
  }

  // ---- Benchmark: SD cycle (K draft + 1 target) ----
  std::cout << "\n=== SD Cycle (K=" << FLAGS_K << ") ===" << std::endl;

  double total_draft_ms = 0, total_target_ms = 0;
  auto sd_start = std::chrono::high_resolution_clock::now();

  for (int c = 0; c < FLAGS_cycles; c++) {
    // Draft phase: K forward passes
    auto d0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < FLAGS_K; k++) {
      std::vector<EValue> in = {*ids};
      draft.execute("forward", in);
    }
    auto d1 = std::chrono::high_resolution_clock::now();

    // Verify phase: 1 forward pass
    auto v0 = std::chrono::high_resolution_clock::now();
    {
      std::vector<EValue> in = {*ids};
      target.execute("forward", in);
    }
    auto v1 = std::chrono::high_resolution_clock::now();

    total_draft_ms += std::chrono::duration<double, std::milli>(d1 - d0).count();
    total_target_ms += std::chrono::duration<double, std::milli>(v1 - v0).count();
  }

  auto sd_end = std::chrono::high_resolution_clock::now();
  double total_sd_ms = std::chrono::duration<double, std::milli>(sd_end - sd_start).count();

  double draft_per_cycle = total_draft_ms / FLAGS_cycles;
  double target_per_cycle = total_target_ms / FLAGS_cycles;
  double cycle_ms = draft_per_cycle + target_per_cycle;

  double draft_per_fwd = draft_per_cycle / FLAGS_K;
  double target_per_fwd = target_per_cycle;

  double gamma = target_per_fwd / draft_per_fwd;

  // SD projected tokens per second
  double tokens_per_cycle = FLAGS_tau;
  double sd_tps = tokens_per_cycle / (cycle_ms / 1000.0);
  double ar_tps = 1.0 / (target_per_fwd / 1000.0);

  double speedup = sd_tps / ar_tps;

  // Theoretical optimal SD speedup
  double theoretical_speedup = FLAGS_tau / (FLAGS_K / gamma + 1);

  std::cout << "Draft phase (K=" << FLAGS_K << "): " << draft_per_cycle
            << " ms/cycle (" << draft_per_fwd << " ms/fwd)" << std::endl;
  std::cout << "Verify phase: " << target_per_fwd << " ms/fwd" << std::endl;
  std::cout << "SD cycle: " << cycle_ms << " ms" << std::endl;

  std::cout << "\n=== Results ===" << std::endl;
  std::cout << "γ (target/draft latency): " << gamma << "x" << std::endl;
  std::cout << "α (acceptance rate): " << FLAGS_alpha << std::endl;
  std::cout << "τ (tokens/step): " << FLAGS_tau << std::endl;
  std::cout << "AR-8bit TPS: " << ar_tps << std::endl;
  std::cout << "SD TPS: " << sd_tps << std::endl;
  std::cout << "SD Speedup: " << speedup << "x" << std::endl;
  std::cout << "Theoretical Speedup: " << theoretical_speedup << "x" << std::endl;

  // Also test with different K values
  std::cout << "\n=== K Sensitivity ===" << std::endl;
  for (int testK : {1, 2, 3, 4, 5, 6, 7, 8}) {
    double test_cycle = testK * draft_per_fwd + target_per_fwd;
    // Estimate tau for different K using: tau ≈ (1 - α^(K+1)) / (1 - α)
    double a = FLAGS_alpha;
    double test_tau = (1.0 - std::pow(a, testK + 1)) / (1.0 - a);
    double test_tps = test_tau / (test_cycle / 1000.0);
    double test_speedup = test_tps / ar_tps;
    printf("  K=%d: τ=%.2f, cycle=%.1fms, SD_TPS=%.1f, speedup=%.3fx\n",
           testK, test_tau, test_cycle, test_tps, test_speedup);
  }

  return 0;
}
