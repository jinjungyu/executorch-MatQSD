/*
 * PPL (Perplexity) evaluator for .pte models on ExecuTorch.
 *
 * Supports:
 *   - Text file or pre-tokenized token file input
 *   - Multi-chunk evaluation (reset KV cache between chunks)
 *   - MatQSD mqint8 mode switching (4-bit / 8-bit)
 *   - WikiText2 evaluation via Python wrapper script
 *
 * Usage:
 *   ./ppl_eval --model model.pte --tokenizer_path tok.bin --text_file wiki.txt
 *   ./ppl_eval --model model.pte --tokenizer_path tok.bin --token_file tokens.txt
 */

#include <gflags/gflags.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <executorch/examples/models/llama/runner/runner.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/tensor/tensor.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

// MatQSD mode switching
extern "C" void xnn_set_mqint8_global_mode(int mode);
extern "C" void xnn_set_mqint8_float_zp_mode(int enable);

DEFINE_string(model, "", ".pte model path");
DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer path");
DEFINE_string(text_file, "", "Text file for PPL evaluation");
DEFINE_string(text, "", "Inline text for PPL evaluation");
DEFINE_string(token_file, "", "Pre-tokenized file (one token ID per line)");
DEFINE_int32(max_tokens, 0, "Max tokens to evaluate (0 = all)");
DEFINE_int32(cpu_threads, 4, "CPU threads");
DEFINE_int32(seq_len, 512, "Chunk size (= model's KV cache length)");
DEFINE_int32(mode, -1, "mqint8 mode: 0=4bit, 1=8bit, -1=default");
DEFINE_bool(float_zp, false, "Enable float zero-point mode");

namespace llm = ::executorch::extension::llm;
using ::executorch::extension::TensorPtr;
using ::executorch::extension::make_tensor_ptr;

// Numerically stable log-softmax → log P(target)
static double log_prob_of_token(const float* logits, int vocab_size, int target) {
  float max_val = logits[0];
  for (int i = 1; i < vocab_size; i++) {
    if (logits[i] > max_val) max_val = logits[i];
  }
  double sum_exp = 0.0;
  for (int i = 0; i < vocab_size; i++) {
    sum_exp += std::exp((double)(logits[i] - max_val));
  }
  double log_sum_exp = (double)max_val + std::log(sum_exp);
  return (double)logits[target] - log_sum_exp;
}

// Read pre-tokenized file (one token ID per line)
static std::vector<uint64_t> read_token_file(const std::string& path) {
  std::vector<uint64_t> tokens;
  std::ifstream f(path);
  if (!f.is_open()) return tokens;
  uint64_t tok;
  while (f >> tok) {
    tokens.push_back(tok);
  }
  return tokens;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

#if defined(ET_USE_THREADPOOL)
  if (FLAGS_cpu_threads > 0) {
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(FLAGS_cpu_threads);
  }
#endif

  // Set mqint8 mode before model load
  if (FLAGS_mode >= 0) {
    xnn_set_mqint8_global_mode(FLAGS_mode);
    std::cout << "[mqint8] mode = " << FLAGS_mode
              << (FLAGS_mode == 0 ? " (4-bit)" : " (8-bit)") << std::endl;
  }
  if (FLAGS_float_zp) {
    xnn_set_mqint8_float_zp_mode(1);
    std::cout << "[mqint8] float zero-point ENABLED" << std::endl;
  }

  // Load model
  std::cout << "Loading model: " << FLAGS_model << std::endl;
  auto runner = example::create_llama_runner(
      FLAGS_model, FLAGS_tokenizer_path, std::nullopt);
  if (!runner) {
    std::cerr << "Failed to create runner" << std::endl;
    return 1;
  }
  runner->load();

  auto* tokenizer = runner->get_tokenizer();
  auto* decoder = runner->get_decoder_runner();

  // Get tokens
  std::vector<uint64_t> tokens;

  if (!FLAGS_token_file.empty()) {
    // Pre-tokenized input
    tokens = read_token_file(FLAGS_token_file);
    if (tokens.empty()) {
      std::cerr << "Cannot read or empty token file: " << FLAGS_token_file << std::endl;
      return 1;
    }
    std::cout << "Loaded " << tokens.size() << " tokens from file" << std::endl;
  } else {
    // Text input → tokenize
    std::string text;
    if (!FLAGS_text_file.empty()) {
      std::ifstream f(FLAGS_text_file);
      if (!f.is_open()) {
        std::cerr << "Cannot open text file: " << FLAGS_text_file << std::endl;
        return 1;
      }
      std::ostringstream ss;
      ss << f.rdbuf();
      text = ss.str();
    } else if (!FLAGS_text.empty()) {
      text = FLAGS_text;
    } else {
      text = "The tower is 324 metres (1,063 ft) tall, about the same height as "
             "an 81-storey building, and the tallest structure in Paris. Its base "
             "is square, measuring 125 metres (410 ft) on each side. During its "
             "construction, the Eiffel Tower surpassed the Washington Monument to "
             "become the tallest man-made structure in the world, a title it held "
             "for 41 years until the Chrysler Building in New York City was "
             "finished in 1930. It was the first structure to reach a height of "
             "300 metres. Due to the addition of a broadcasting aerial at the top "
             "of the tower in 1957, it is now taller than the Chrysler Building "
             "by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is "
             "the second tallest free-standing structure in France after the "
             "Millau Viaduct.";
    }

    auto enc = tokenizer->encode(text, 0, 0);
    if (enc.error() != ::tokenizers::Error::Ok) {
      std::cerr << "Tokenization failed" << std::endl;
      return 1;
    }
    tokens = enc.get();
  }

  int n_tokens = (int)tokens.size();
  if (FLAGS_max_tokens > 0 && FLAGS_max_tokens < n_tokens) {
    n_tokens = FLAGS_max_tokens;
  }

  int chunk_size = FLAGS_seq_len;

  // Split into chunks for multi-chunk evaluation
  int n_chunks = (n_tokens + chunk_size - 1) / chunk_size;
  std::cout << "Total tokens: " << n_tokens
            << ", chunk_size: " << chunk_size
            << ", chunks: " << n_chunks << std::endl;

  double total_nll = 0.0;
  int total_count = 0;
  auto t_start = std::chrono::high_resolution_clock::now();

  for (int chunk = 0; chunk < n_chunks; chunk++) {
    int chunk_start = chunk * chunk_size;
    int chunk_end = std::min(chunk_start + chunk_size, n_tokens);
    int chunk_len = chunk_end - chunk_start;

    if (chunk_len < 2) continue;  // Need at least 2 tokens for NLL

    // Process tokens in this chunk (position resets to 0 → KV cache reset)
    for (int i = 0; i < chunk_len - 1; i++) {
      int64_t tok = (int64_t)tokens[chunk_start + i];
      auto input = make_tensor_ptr({1, 1}, std::vector<int64_t>{tok});
      auto result = decoder->step(input, i);  // pos = i (resets each chunk)
      if (!result.ok()) {
        std::cerr << "Step failed at chunk " << chunk << " pos " << i << std::endl;
        return 1;
      }

      auto& logits_tensor = result.get();
      int vocab_size = logits_tensor.size(logits_tensor.dim() - 1);
      const float* logits_data;

      if (logits_tensor.dim() == 3) {
        logits_data = logits_tensor.const_data_ptr<float>() +
                      (logits_tensor.size(1) - 1) * vocab_size;
      } else {
        logits_data = logits_tensor.const_data_ptr<float>();
      }

      int next_token = (int)tokens[chunk_start + i + 1];
      double lp = log_prob_of_token(logits_data, vocab_size, next_token);
      total_nll -= lp;
      total_count++;
    }

    double running_ppl = std::exp(total_nll / total_count);
    std::cout << "  chunk " << (chunk + 1) << "/" << n_chunks
              << " (" << total_count << " tokens) PPL = " << running_ppl << std::endl;
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();

  double ppl = std::exp(total_nll / total_count);
  double avg_nll = total_nll / total_count;

  std::cout << "\n=== PPL Results ===" << std::endl;
  std::cout << "Model: " << FLAGS_model << std::endl;
  if (FLAGS_mode >= 0) {
    std::cout << "Mode: " << (FLAGS_mode == 0 ? "4-bit" : "8-bit") << std::endl;
  }
  std::cout << "Tokens evaluated: " << total_count << std::endl;
  std::cout << "Chunks: " << n_chunks << " x " << chunk_size << std::endl;
  std::cout << "Avg NLL: " << avg_nll << std::endl;
  std::cout << "PPL: " << ppl << std::endl;
  std::cout << "Eval speed: " << total_count / elapsed << " tok/s" << std::endl;
  std::cout << "Elapsed: " << elapsed << "s" << std::endl;

  return 0;
}
