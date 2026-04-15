/* MatQSD test: correctness check + multi-forward latency benchmark. */
#include <gflags/gflags.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/runtime.h>
#include <chrono>
#include <iostream>
#include <vector>

DEFINE_string(model, "", ".pte path");
DEFINE_int32(repeat, 1, "Number of forward passes for benchmarking");
DEFINE_int32(batch, 0, "Override batch size (0 = use default 6-token prompt)");

using namespace executorch::extension;
using namespace executorch::runtime;

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  runtime_init();

  Module m(FLAGS_model);
  auto err = m.load_method("forward");
  if (err != Error::Ok) { std::cerr << "Load failed: " << (int)err << std::endl; return 1; }

  // Default: "The capital of France is" = [128000, 791, 6864, 315, 9822, 374]
  std::vector<int64_t> tokens = {128000, 791, 6864, 315, 9822, 374};

  int seq_len = (FLAGS_batch > 0) ? FLAGS_batch : static_cast<int>(tokens.size());
  if (FLAGS_batch > 0) {
    // Use only first token repeated for batch=1 test, or truncate
    tokens.resize(seq_len, tokens[0]);
  }

  auto ids = make_tensor_ptr({1, seq_len}, tokens);

  // Warmup
  {
    std::vector<EValue> inputs = {*ids};
    auto out = m.execute("forward", inputs);
    if (!out.ok()) { std::cerr << "Forward failed: " << (int)out.error() << std::endl; return 1; }

    // Print correctness on first run
    auto& logits = out.get()[0].toTensor();
    int64_t vocab = logits.size(logits.dim() - 1);
    auto* data = logits.const_data_ptr<float>();
    int64_t offset = logits.dim() == 3 ? (logits.size(1) - 1) * vocab : 0;

    std::vector<std::pair<float, int64_t>> top;
    for (int64_t i = 0; i < vocab; i++) top.push_back({data[offset + i], i});
    std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.first > b.first; });

    std::cout << "Output shape: [" << logits.size(0);
    if (logits.dim() >= 2) std::cout << ", " << logits.size(1);
    if (logits.dim() >= 3) std::cout << ", " << logits.size(2);
    std::cout << "]  seq_len=" << seq_len << std::endl;
    std::cout << "Top 5:";
    for (int i = 0; i < 5; i++)
      std::cout << " " << top[i].second << "(" << top[i].first << ")";
    std::cout << std::endl;
  }

  // Benchmark
  if (FLAGS_repeat > 1) {
    // Multiple repeats for timing
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < FLAGS_repeat; r++) {
      std::vector<EValue> inputs = {*ids};
      m.execute("forward", inputs);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Benchmark: " << FLAGS_repeat << " forwards, "
              << ms << " ms total, "
              << ms / FLAGS_repeat << " ms/forward" << std::endl;
  }

  return 0;
}
