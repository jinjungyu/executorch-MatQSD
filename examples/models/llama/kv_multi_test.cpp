#include <gflags/gflags.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/runtime.h>
#include <iostream>
#include <vector>
DEFINE_string(model, "", ".pte");
using namespace executorch::extension;
using namespace executorch::runtime;
int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  runtime_init();
  Module m(FLAGS_model);
  m.load_method("forward");
  // Step 1: token=128000, pos=0
  auto t1 = make_tensor_ptr({1,1}, std::vector<int64_t>{128000});
  auto p1 = make_tensor_ptr({1}, std::vector<int64_t>{0});
  std::vector<EValue> in1 = {*t1, *p1};
  auto o1 = m.execute("forward", in1);
  std::cout << "Step 0: " << (o1.ok() ? "OK" : "FAIL") << " err=" << (o1.ok()?0:(int)o1.error()) << std::endl;
  // Step 2: token=791, pos=1
  auto t2 = make_tensor_ptr({1,1}, std::vector<int64_t>{791});
  auto p2 = make_tensor_ptr({1}, std::vector<int64_t>{1});
  std::vector<EValue> in2 = {*t2, *p2};
  auto o2 = m.execute("forward", in2);
  std::cout << "Step 1: " << (o2.ok() ? "OK" : "FAIL") << " err=" << (o2.ok()?0:(int)o2.error()) << std::endl;
  return 0;
}
