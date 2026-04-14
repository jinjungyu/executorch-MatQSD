/*
 * MatQSD Speculative Decoding runner.
 * Single model instance with runtime 4-bit/8-bit mode switching.
 * Single weight buffer + single KV cache.
 * Draft (4-bit) proposes K tokens, Target (8-bit) batch-verifies [1, K+1].
 */

#include <gflags/gflags.h>
#include <chrono>
#include <iostream>
#include <vector>

#include <executorch/examples/models/llama/runner/runner.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/tensor/tensor.h>
// MatQSD mode switching (declared in xnnpack.h)
extern "C" void xnn_set_mqint8_global_mode(int mode);

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

DEFINE_string(draft_model, "", "Draft .pte (separate model SD)");
DEFINE_string(target_model, "", "Target .pte (separate model SD)");
DEFINE_string(mqint8_model, "", "Single mqint8 .pte (shared weight+KV)");
DEFINE_string(lastlogit_model, "", "Non-full-logits .pte for timing comparison");
DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer path");
DEFINE_string(prompt, "The capital of France is", "Input prompt");
DEFINE_int32(max_new_tokens, 64, "Max tokens to generate");
DEFINE_int32(K, 5, "Draft tokens per SD step");
DEFINE_int32(cpu_threads, 4, "CPU threads");
DEFINE_int32(seq_len, 128, "Max sequence length");

namespace llm = ::executorch::extension::llm;
using ::executorch::extension::TensorPtr;
using ::executorch::extension::make_tensor_ptr;

// Extract argmax for each position from logits tensor
static std::vector<int64_t> all_argmax(const executorch::aten::Tensor& logits) {
  std::vector<int64_t> results;
  const float* data = logits.const_data_ptr<float>();
  if (logits.dim() == 3) {
    int seq_len = logits.size(1);
    int vocab = logits.size(2);
    for (int s = 0; s < seq_len; s++) {
      const float* row = data + s * vocab;
      int best = 0;
      for (int v = 1; v < vocab; v++) {
        if (row[v] > row[best]) best = v;
      }
      results.push_back(best);
    }
  } else {
    int vocab = logits.size(logits.dim() - 1);
    int best = 0;
    for (int v = 1; v < vocab; v++) {
      if (data[v] > data[best]) best = v;
    }
    results.push_back(best);
  }
  return results;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

#if defined(ET_USE_THREADPOOL)
  if (FLAGS_cpu_threads > 0) {
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(FLAGS_cpu_threads);
  }
#endif

  int K = FLAGS_K;
  int max_new = FLAGS_max_new_tokens;
  bool single_model = !FLAGS_mqint8_model.empty();

  // ==========================================
  // Load model(s)
  // ==========================================
  std::unique_ptr<llm::TextLLMRunner> model;
  std::unique_ptr<llm::TextLLMRunner> draft_model_separate;
  std::unique_ptr<llm::TextLLMRunner> target_model_separate;
  llm::TextDecoderRunner* decoder = nullptr;

  if (single_model) {
    // Single instance: shared weight + shared KV cache
    std::cout << "Loading single mqint8 model..." << std::endl;
    setenv("MQINT8_MODE", "1", 1);  // default 8-bit
    model = example::create_llama_runner(
        FLAGS_mqint8_model, FLAGS_tokenizer_path, std::nullopt);
    if (!model) { std::cerr << "Failed to load model" << std::endl; return 1; }
    model->load();
    decoder = model->get_decoder_runner();
    std::cout << "Single model loaded (shared weight + KV cache)" << std::endl;
  } else {
    // Separate models (legacy)
    std::cout << "Loading draft model..." << std::endl;
    if (!FLAGS_draft_model.empty())
      draft_model_separate = example::create_llama_runner(
          FLAGS_draft_model, FLAGS_tokenizer_path, std::nullopt);
    std::cout << "Loading target model..." << std::endl;
    if (!FLAGS_target_model.empty())
      target_model_separate = example::create_llama_runner(
          FLAGS_target_model, FLAGS_tokenizer_path, std::nullopt);
    if (!draft_model_separate || !target_model_separate) {
      std::cerr << "Failed to load models" << std::endl; return 1;
    }
    draft_model_separate->load();
    target_model_separate->load();
  }

  auto* tokenizer = single_model ? model->get_tokenizer()
                                 : draft_model_separate->get_tokenizer();

  // Helper: step one token
  auto step1 = [](llm::TextDecoderRunner* dec, int64_t token, int64_t pos) -> int64_t {
    auto input = make_tensor_ptr({1, 1}, std::vector<int64_t>{token});
    auto result = dec->step(input, pos);
    if (!result.ok()) return -1;
    return dec->logits_to_token(result.get(), 0.0f);
  };

  // Encode prompt
  auto enc = tokenizer->encode(FLAGS_prompt, 0, 0);
  if (enc.error() != ::tokenizers::Error::Ok) {
    std::cerr << "Encode failed" << std::endl; return 1;
  }
  auto prompt_tokens = enc.get();
  std::cout << "Prompt: \"" << FLAGS_prompt << "\" (" << prompt_tokens.size() << " tokens)" << std::endl;

  // ==========================================
  // Prefill (8-bit mode for best quality)
  // ==========================================
  std::cout << "Prefilling..." << std::endl;
  int64_t cur_token = 0;

  if (single_model) {
    xnn_set_mqint8_global_mode(1);  // 8-bit for prefill
    for (int i = 0; i < (int)prompt_tokens.size(); i++) {
      cur_token = step1(decoder, prompt_tokens[i], i);
    }
  } else {
    auto* draft_dec = draft_model_separate->get_decoder_runner();
    auto* target_dec = target_model_separate->get_decoder_runner();
    for (int i = 0; i < (int)prompt_tokens.size(); i++) {
      cur_token = step1(target_dec, prompt_tokens[i], i);
      step1(draft_dec, prompt_tokens[i], i);
    }
  }

  auto dec_res = tokenizer->decode(cur_token, cur_token);
  if (dec_res.error() == ::tokenizers::Error::Ok)
    std::cout << "Output:" << dec_res.get() << std::flush;

  std::vector<int64_t> generated = {cur_token};
  int64_t pos = prompt_tokens.size();

  // ==========================================
  // Speculative Decoding Loop
  // ==========================================
  int total_drafted = 0, total_accepted = 0, total_steps = 0;
  double total_draft_ms = 0, total_verify_ms = 0;
  double total_verify_step_ms = 0;
  auto sd_start = std::chrono::high_resolution_clock::now();

  // Get decoders
  llm::TextDecoderRunner* draft_dec = single_model
      ? decoder : draft_model_separate->get_decoder_runner();
  llm::TextDecoderRunner* target_dec = single_model
      ? decoder : target_model_separate->get_decoder_runner();

  while ((int)generated.size() < max_new) {
    // --- Draft K tokens (4-bit mode) ---
    if (single_model) xnn_set_mqint8_global_mode(0);

    auto t_draft_start = std::chrono::high_resolution_clock::now();
    std::vector<int64_t> draft_tokens;
    int64_t draft_tok = cur_token;
    for (int d = 0; d < K && (int)(generated.size() + draft_tokens.size()) < max_new; d++) {
      draft_tok = step1(draft_dec, draft_tok, pos + d);
      if (draft_tok < 0) break;
      draft_tokens.push_back(draft_tok);
    }
    auto t_draft_end = std::chrono::high_resolution_clock::now();
    total_draft_ms += std::chrono::duration<double, std::milli>(t_draft_end - t_draft_start).count();

    total_drafted += draft_tokens.size();
    int n_draft = draft_tokens.size();

    // --- Batch verify (8-bit mode) ---
    if (single_model) xnn_set_mqint8_global_mode(1);

    auto t_verify_start = std::chrono::high_resolution_clock::now();
    std::vector<int64_t> verify_input;
    verify_input.push_back(cur_token);
    for (auto& t : draft_tokens) verify_input.push_back(t);

    int verify_len = verify_input.size();
    auto verify_tensor = make_tensor_ptr(
        {1, verify_len}, std::vector<int64_t>(verify_input));

    auto t_step_start = std::chrono::high_resolution_clock::now();
    auto verify_result = target_dec->step(verify_tensor, pos);
    auto t_step_end = std::chrono::high_resolution_clock::now();
    total_verify_step_ms += std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();

    std::vector<int64_t> target_tokens;
    if (verify_result.ok()) {
      target_tokens = all_argmax(verify_result.get());
    }

    auto t_verify_end = std::chrono::high_resolution_clock::now();
    total_verify_ms += std::chrono::duration<double, std::milli>(t_verify_end - t_verify_start).count();

    // --- Accept/reject ---
    int n_acc = 0;
    for (int i = 0; i < n_draft && i < (int)target_tokens.size(); i++) {
      if (target_tokens[i] == draft_tokens[i]) n_acc++;
      else break;
    }
    total_accepted += n_acc;

    // Emit accepted tokens
    for (int i = 0; i < n_acc; i++) {
      generated.push_back(draft_tokens[i]);
      uint64_t prev = generated.size() > 1 ? generated[generated.size()-2] : generated.back();
      auto dr = tokenizer->decode(generated.back(), prev);
      if (dr.error() == ::tokenizers::Error::Ok) std::cout << dr.get() << std::flush;
    }

    // Bonus token
    int64_t bonus = (n_acc < (int)target_tokens.size()) ?
        target_tokens[n_acc] : target_tokens.back();
    generated.push_back(bonus);
    {
      uint64_t prev = generated.size() > 1 ? generated[generated.size()-2] : generated.back();
      auto dr = tokenizer->decode(bonus, prev);
      if (dr.error() == ::tokenizers::Error::Ok) std::cout << dr.get() << std::flush;
    }

    cur_token = bonus;
    pos += n_acc + 1;

    total_steps++;
    if (bonus == 128009 || bonus == 128001) break;
  }

  auto sd_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(sd_end - sd_start).count();
  std::cout << std::endl;

  double alpha = (double)total_accepted / std::max(total_drafted, 1);
  double tau = (double)generated.size() / std::max(total_steps, 1);
  double tps = generated.size() / elapsed;

  double avg_draft_ms = total_draft_ms / std::max(total_steps, 1);
  double avg_verify_ms = total_verify_ms / std::max(total_steps, 1);
  double avg_vstep = total_verify_step_ms / std::max(total_steps, 1);

  std::cout << "\n=== SD Results (K=" << K << ", "
            << (single_model ? "single model" : "separate models") << ") ===" << std::endl;
  std::cout << "Generated: " << generated.size() << " tokens" << std::endl;
  std::cout << "Steps: " << total_steps << std::endl;
  std::cout << "Drafted: " << total_drafted << ", Accepted: " << total_accepted << std::endl;
  std::cout << "α (acceptance rate): " << alpha << std::endl;
  std::cout << "τ (tokens/step): " << tau << std::endl;
  std::cout << "TPS: " << tps << std::endl;
  std::cout << "Elapsed: " << elapsed << "s" << std::endl;
  std::cout << "\n--- Cycle Timing ---" << std::endl;
  std::cout << "Avg draft (" << K << " tokens):     " << avg_draft_ms << " ms  (" << (avg_draft_ms/K) << " ms/tok)" << std::endl;
  std::cout << "Avg verify (" << K+1 << " tokens):    " << avg_verify_ms << " ms" << std::endl;
  std::cout << "  Step (embed+fwd+unembed):" << avg_vstep << " ms  (" << (avg_vstep/(K+1)) << " ms/tok)" << std::endl;
  std::cout << "Draft/Verify split:        " << (int)(avg_draft_ms/(avg_draft_ms+avg_verify_ms)*100)
            << "% / " << (int)(avg_verify_ms/(avg_draft_ms+avg_verify_ms)*100) << "%" << std::endl;

  // AR baselines (same model, same prompt, same max_new_tokens)
  auto run_ar_baseline = [&](int mode, const char* label) {
    if (single_model) xnn_set_mqint8_global_mode(mode);
    auto* ar_runner = single_model ? model.get() : target_model_separate.get();
    ar_runner->reset();
    std::vector<std::string> ar_tokens;
    llm::GenerationConfig ar_cfg{.temperature = 0.0f};
    ar_cfg.max_new_tokens = max_new;
    ar_cfg.seq_len = FLAGS_seq_len;
    auto ar_start = std::chrono::high_resolution_clock::now();
    ar_runner->generate(FLAGS_prompt, ar_cfg,
        [&](const std::string& p) { ar_tokens.push_back(p); });
    auto ar_end = std::chrono::high_resolution_clock::now();
    double ar_elapsed = std::chrono::duration<double>(ar_end - ar_start).count();
    double ar_tps = ar_tokens.size() / ar_elapsed;
    std::cout << "\n=== " << label << " ===" << std::endl;
    std::cout << ar_tokens.size() << " tokens, " << ar_tps << " TPS" << std::endl;
    return ar_tps;
  };

  double ar4_tps = run_ar_baseline(0, "AR-4bit baseline");
  double ar8_tps = run_ar_baseline(1, "AR-8bit baseline");

  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "AR-4bit: " << ar4_tps << " TPS" << std::endl;
  std::cout << "AR-8bit: " << ar8_tps << " TPS" << std::endl;
  std::cout << "4/8 ratio: " << ar4_tps / ar8_tps << "x" << std::endl;
  std::cout << "SD: " << tps << " TPS" << std::endl;
  std::cout << "SD speedup vs AR-8bit: " << tps / ar8_tps << "x" << std::endl;

  return 0;
}
