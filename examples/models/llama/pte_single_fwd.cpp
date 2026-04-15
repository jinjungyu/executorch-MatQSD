// Single forward pass through .pte and dump output logits
// Usage: pte_single_fwd <model.pte>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/extension/module/module.h>

using namespace executorch::runtime;
using namespace executorch::extension;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.pte>\n", argv[0]);
        return 1;
    }

    runtime_init();

    Module module(argv[1]);

    // BOS token = 128000
    int64_t token_data[] = {128000};
    auto token_tensor = from_blob(token_data, {1, 1}, ScalarType::Long);

    int64_t pos_data[] = {0};
    auto pos_tensor = from_blob(pos_data, {1}, ScalarType::Long);

    // Forward
    fprintf(stderr, "Running forward...\n");
    auto result = module.forward({*token_tensor, *pos_tensor});

    if (!result.ok()) {
        fprintf(stderr, "Forward failed: %d\n", (int)result.error());
        return 1;
    }

    auto& outputs = result.get();
    if (outputs.empty()) {
        fprintf(stderr, "No outputs\n");
        return 1;
    }

    auto& logits_evalue = outputs[0];
    if (!logits_evalue.isTensor()) {
        fprintf(stderr, "Output is not a tensor\n");
        return 1;
    }

    auto logits = logits_evalue.toTensor();
    fprintf(stderr, "Output shape: [");
    for (int i = 0; i < logits.dim(); i++) {
        fprintf(stderr, "%d%s", (int)logits.size(i), i < logits.dim()-1 ? ", " : "");
    }
    fprintf(stderr, "]\n");

    // Dump first 10 and top-5
    float* data = logits.data_ptr<float>();
    int numel = logits.numel();

    fprintf(stderr, "First 10 logits: ");
    for (int i = 0; i < 10 && i < numel; i++) {
        fprintf(stderr, "%.4f ", data[i]);
    }
    fprintf(stderr, "\n");

    // Top-5
    std::vector<std::pair<float, int>> vals(numel);
    for (int i = 0; i < numel; i++) vals[i] = {data[i], i};
    std::partial_sort(vals.begin(), vals.begin()+5, vals.end(),
        [](auto& a, auto& b) { return a.first > b.first; });

    fprintf(stderr, "Top-5: ");
    for (int i = 0; i < 5; i++) {
        fprintf(stderr, "[%d]=%.4f ", vals[i].second, vals[i].first);
    }
    fprintf(stderr, "\n");

    return 0;
}
