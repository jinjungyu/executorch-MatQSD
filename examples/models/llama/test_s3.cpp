#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/runtime.h>
#include <iostream>
#include <vector>
using namespace executorch::extension;
using namespace executorch::runtime;
int main() {
  runtime_init();
  Module m("/Users/jjg_nota/matqsd/models/matqsd_draft.pte");
  m.load_method("forward");
  // Try with seq_len=3 (same as export example)
  auto t = make_tensor_ptr({1,3}, std::vector<int64_t>{128000,791,6864});
  auto p = make_tensor_ptr({1}, std::vector<int64_t>{0});
  std::vector<EValue> in = {*t, *p};
  auto o = m.execute("forward", in);
  std::cout << "seq=3: " << (o.ok()?"OK":"FAIL") << " err=" << (o.ok()?0:(int)o.error()) << std::endl;
  // Try seq_len=1
  auto t1 = make_tensor_ptr({1,1}, std::vector<int64_t>{128000});
  auto p1 = make_tensor_ptr({1}, std::vector<int64_t>{0});
  std::vector<EValue> in1 = {*t1, *p1};
  auto o1 = m.execute("forward", in1);
  std::cout << "seq=1: " << (o1.ok()?"OK":"FAIL") << " err=" << (o1.ok()?0:(int)o1.error()) << std::endl;
  return 0;
}
