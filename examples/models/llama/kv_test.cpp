/* KV cache model load + forward test */
#include <gflags/gflags.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/runtime.h>
#include <iostream>
#include <vector>

DEFINE_string(model, "", ".pte path");

using namespace executorch::extension;
using namespace executorch::runtime;

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  runtime_init();

  Module m(FLAGS_model);
  std::cout << "Loading..." << std::endl;
  auto err = m.load_method("forward");
  if (err != Error::Ok) {
    std::cerr << "Load failed: " << (int)err << std::endl;
    return 1;
  }

  auto meta = m.method_meta("forward");
  if (meta.ok()) {
    std::cout << "Num inputs: " << meta->num_inputs() << std::endl;
    std::cout << "Num outputs: " << meta->num_outputs() << std::endl;
    for (size_t i = 0; i < meta->num_inputs(); i++) {
      auto input_meta = meta->input_tensor_meta(i);
      if (input_meta.ok()) {
        std::cout << "  Input " << i << ": dtype=" << (int)input_meta->scalar_type()
                  << " ndim=" << input_meta->sizes().size();
        for (auto s : input_meta->sizes()) std::cout << " " << s;
        std::cout << std::endl;
      } else {
        std::cout << "  Input " << i << ": not a tensor" << std::endl;
      }
    }
  }

  // Try forward with 1 token + input_pos
  std::vector<int64_t> token_data = {128000};
  auto tokens = make_tensor_ptr({1, 1}, token_data);

  std::vector<int64_t> pos_data = {0};
  auto input_pos = make_tensor_ptr({1}, pos_data);

  std::vector<EValue> inputs = {*tokens, *input_pos};
  std::cout << "Executing forward..." << std::endl;
  auto out = m.execute("forward", inputs);
  if (!out.ok()) {
    std::cerr << "Forward failed: " << (int)out.error() << std::endl;
    return 1;
  }

  std::cout << "Forward OK! Outputs: " << out.get().size() << std::endl;
  auto& logits = out.get()[0].toTensor();
  std::cout << "Logits shape: [" << logits.size(0);
  if (logits.dim() >= 2) std::cout << ", " << logits.size(1);
  if (logits.dim() >= 3) std::cout << ", " << logits.size(2);
  std::cout << "]" << std::endl;

  // Top-5
  int64_t vocab = logits.size(logits.dim() - 1);
  auto* data = logits.const_data_ptr<float>();
  int64_t offset = logits.dim() == 3 ? (logits.size(1) - 1) * vocab : 0;
  std::vector<std::pair<float, int>> top;
  for (int i = 0; i < (int)vocab; i++) top.push_back({data[offset + i], i});
  std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.first > b.first; });
  std::cout << "Top 5:";
  for (int i = 0; i < 5; i++) std::cout << " " << top[i].second << "(" << top[i].first << ")";
  std::cout << std::endl;

  return 0;
}
