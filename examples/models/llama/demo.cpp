/*
 * MatQSD Split-Screen Demo
 * ncurses TUI: Left panel = AR-8bit, Right panel = MatQSD SD
 * Runs AR-8bit first (streams to left), then SD (streams to right).
 * Same model, same prompt — fair speed comparison with visual impact.
 */

#include <gflags/gflags.h>
#include <fcntl.h>
#include <locale.h>
#include <mach/mach.h>
#include <ncurses.h>
#include <sys/resource.h>
#include <unistd.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/hidsystem/IOHIDEventSystemClient.h>
#include <CoreFoundation/CoreFoundation.h>

// IOHIDEvent thermal sensor API (private but stable)
extern "C" {
  extern IOHIDEventSystemClientRef IOHIDEventSystemClientCreate(CFAllocatorRef);
  extern CFArrayRef IOHIDEventSystemClientCopyServices(IOHIDEventSystemClientRef);
  extern CFStringRef IOHIDServiceClientCopyProperty(void*, CFStringRef);
  typedef struct __IOHIDEvent *IOHIDEventRef;
  extern IOHIDEventRef IOHIDServiceClientCopyEvent(void*, int64_t, int32_t, int64_t);
  extern double IOHIDEventGetFloatValue(IOHIDEventRef, int32_t);
}
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <executorch/examples/models/llama/runner/runner.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/tensor/tensor.h>

extern "C" void xnn_set_mqint8_global_mode(int mode);

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

DEFINE_string(model, "", "Path to mqint8 .pte model");
DEFINE_string(tokenizer, "tokenizer.bin", "Path to tokenizer");
DEFINE_int32(K, 3, "Draft tokens per SD step");
DEFINE_int32(seq_len, 4096, "Max sequence length");
DEFINE_int32(cpu_threads, 4, "CPU threads");
DEFINE_string(system_prompt, "You are a helpful assistant.", "System prompt");
DEFINE_double(repetition_penalty, 1.05, "Repetition penalty (>1.0 penalizes, lower=better SD alpha)");

namespace llm = ::executorch::extension::llm;
using ::executorch::extension::TensorPtr;
using ::executorch::extension::make_tensor_ptr;

static constexpr int64_t TOK_BOS = 128000, TOK_EOT = 128001,
                          TOK_EOS = 128009, TOK_START_HEADER = 128006,
                          TOK_END_HEADER = 128007;

// ── CPU temperature via IOHIDEventSystemClient (macOS Apple Silicon) ────
static IOHIDEventSystemClientRef g_hid_client = nullptr;

static void thermal_init() {
  g_hid_client = IOHIDEventSystemClientCreate(kCFAllocatorDefault);
}

static void thermal_close() {
  if (g_hid_client) { CFRelease(g_hid_client); g_hid_client = nullptr; }
}

static double get_cpu_temp() {
  // Returns max CPU die temperature across all "PMU tdie*" sensors
  if (!g_hid_client) return -1;
  CFArrayRef services = IOHIDEventSystemClientCopyServices(g_hid_client);
  if (!services) return -1;

  double max_temp = -1;
  long count = CFArrayGetCount(services);
  for (long i = 0; i < count; i++) {
    void* svc = (void*)CFArrayGetValueAtIndex(services, i);
    CFStringRef name = IOHIDServiceClientCopyProperty(svc, CFSTR("Product"));
    if (!name) continue;
    char buf[128] = {};
    CFStringGetCString(name, buf, sizeof(buf), kCFStringEncodingUTF8);
    CFRelease(name);

    // Read all PMU temperature sensors (die, dev, cal — full SoC thermal picture)
    if (strncmp(buf, "PMU", 3) != 0) continue;

    IOHIDEventRef ev = IOHIDServiceClientCopyEvent(svc, 15, 0, 0); // 15 = temperature
    if (ev) {
      double t = IOHIDEventGetFloatValue(ev, 15 << 16);
      if (t > max_temp) max_temp = t;
      CFRelease(ev);
    }
  }
  CFRelease(services);
  return max_temp;
}

// ── CPU throttle detection via micro-benchmark ─────────────────────────
static double cpu_bench_once() {
  // Tight FP multiply loop — measures raw CPU throughput
  volatile float x = 1.000001f;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 5000000; i++) x *= 1.000001f;
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double g_cpu_bench_baseline = 0; // cold CPU baseline (set once at startup)

static double cpu_throttle_pct() {
  // Returns 0-100: 0 = no throttle, 100 = fully throttled
  if (g_cpu_bench_baseline <= 0) return 0;
  double now = cpu_bench_once();
  double slowdown = (now - g_cpu_bench_baseline) / g_cpu_bench_baseline * 100;
  return std::max(0.0, slowdown);
}

// ── System monitoring ───────────────────────────────────────────────────
struct SysInfo {
  int rss_mb;       // current resident set size
  double cpu_pct;   // CPU % since last call
};

static SysInfo get_sys_info() {
  SysInfo info = {0, 0};
  // RSS via mach task info
  mach_task_basic_info_data_t ti;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                (task_info_t)&ti, &count) == KERN_SUCCESS) {
    info.rss_mb = (int)(ti.resident_size / (1024 * 1024));
  }
  // CPU usage via thread times
  thread_array_t threads;
  mach_msg_type_number_t thread_count;
  if (task_threads(mach_task_self(), &threads, &thread_count) == KERN_SUCCESS) {
    double total_cpu = 0;
    for (mach_msg_type_number_t i = 0; i < thread_count; i++) {
      thread_basic_info_data_t tbi;
      mach_msg_type_number_t tbi_count = THREAD_BASIC_INFO_COUNT;
      if (thread_info(threads[i], THREAD_BASIC_INFO,
                      (thread_info_t)&tbi, &tbi_count) == KERN_SUCCESS) {
        if (!(tbi.flags & TH_FLAGS_IDLE))
          total_cpu += tbi.cpu_usage / (double)TH_USAGE_SCALE * 100.0;
      }
      mach_port_deallocate(mach_task_self(), threads[i]);
    }
    vm_deallocate(mach_task_self(), (vm_address_t)threads,
                  sizeof(thread_t) * thread_count);
    info.cpu_pct = total_cpu;
  }
  return info;
}

static bool is_control_token(int64_t t) {
  return t == TOK_BOS || t == TOK_EOT || t == TOK_EOS
      || t == TOK_START_HEADER || t == TOK_END_HEADER;
}

// ── argmax helpers ──────────────────────────────────────────────────────────
static std::vector<int64_t> all_argmax(const executorch::aten::Tensor& logits) {
  std::vector<int64_t> r;
  const float* d = logits.const_data_ptr<float>();
  if (logits.dim() == 3) {
    int seq = logits.size(1), vocab = logits.size(2);
    for (int s = 0; s < seq; s++) {
      const float* row = d + s * vocab;
      int best = 0;
      for (int v = 1; v < vocab; v++) if (row[v] > row[best]) best = v;
      r.push_back(best);
    }
  } else {
    int vocab = logits.size(logits.dim() - 1);
    int best = 0;
    for (int v = 1; v < vocab; v++) if (d[v] > d[best]) best = v;
    r.push_back(best);
  }
  return r;
}

static void apply_rep_penalty(float* logits, int vocab,
                              const std::vector<int64_t>& past, float pen) {
  if (pen <= 1.0f) return;
  for (auto t : past) {
    if (t < 0 || t >= vocab) continue;
    if (logits[t] > 0) logits[t] /= pen; else logits[t] *= pen;
  }
}

static int64_t argmax(const float* d, int n) {
  int b = 0; for (int i = 1; i < n; i++) if (d[i] > d[b]) b = i; return b;
}

// ── step helpers ────────────────────────────────────────────────────────────
static int64_t step1(llm::TextDecoderRunner* dec, int64_t tok, int64_t pos) {
  auto in = make_tensor_ptr({1, 1}, std::vector<int64_t>{tok});
  auto r = dec->step(in, pos);
  if (!r.ok()) return -1;
  return dec->logits_to_token(r.get(), 0.0f);
}

static int64_t step1_pen(llm::TextDecoderRunner* dec, int64_t tok, int64_t pos,
                         const std::vector<int64_t>& past, float pen) {
  auto in = make_tensor_ptr({1, 1}, std::vector<int64_t>{tok});
  auto r = dec->step(in, pos);
  if (!r.ok()) return -1;
  if (pen <= 1.0f) return dec->logits_to_token(r.get(), 0.0f);
  auto& lt = r.get();
  float* ld = lt.mutable_data_ptr<float>();
  int vocab = lt.size(lt.dim() - 1);
  apply_rep_penalty(ld, vocab, past, pen);
  return argmax(ld, vocab);
}

// ── tokenization ────────────────────────────────────────────────────────────
static std::vector<int64_t> build_system(tokenizers::Tokenizer* tok,
                                         const std::string& prompt) {
  std::vector<int64_t> s;
  s.push_back(TOK_BOS); s.push_back(TOK_START_HEADER);
  auto e = tok->encode("system", 0, 0);
  if (e.error() == ::tokenizers::Error::Ok) for (auto t : e.get()) s.push_back(t);
  s.push_back(TOK_END_HEADER);
  auto b = tok->encode("\n\n" + prompt, 0, 0);
  if (b.error() == ::tokenizers::Error::Ok) for (auto t : b.get()) s.push_back(t);
  s.push_back(TOK_EOT);
  return s;
}

static std::vector<int64_t> build_user(tokenizers::Tokenizer* tok,
                                       const std::string& input) {
  std::vector<int64_t> s;
  s.push_back(TOK_START_HEADER);
  auto r = tok->encode("user", 0, 0);
  if (r.error() == ::tokenizers::Error::Ok) for (auto t : r.get()) s.push_back(t);
  s.push_back(TOK_END_HEADER);
  auto b = tok->encode("\n\n" + input, 0, 0);
  if (b.error() == ::tokenizers::Error::Ok) for (auto t : b.get()) s.push_back(t);
  s.push_back(TOK_EOT);
  s.push_back(TOK_START_HEADER);
  auto a = tok->encode("assistant", 0, 0);
  if (a.error() == ::tokenizers::Error::Ok) for (auto t : a.get()) s.push_back(t);
  s.push_back(TOK_END_HEADER);
  auto nl = tok->encode("\n\n", 0, 0);
  if (nl.error() == ::tokenizers::Error::Ok) for (auto t : nl.get()) s.push_back(t);
  return s;
}

static int64_t prefill(llm::TextDecoderRunner* dec,
                       const std::vector<int64_t>& toks, int64_t& pos) {
  if (toks.empty()) return 0;
  // Batch prefill: feed all tokens in one forward pass
  auto input = make_tensor_ptr(
      {1, (int)toks.size()}, std::vector<int64_t>(toks));
  auto result = dec->step(input, pos);
  pos += toks.size();
  if (!result.ok()) return -1;
  // Return argmax of last position
  return dec->logits_to_token(result.get(), 0.0f);
}

// ── ncurses window text helper ──────────────────────────────────────────────
static void win_add_text(WINDOW* win, const std::string& text, int max_y, int max_x) {
  for (size_t i = 0; i < text.size(); ) {
    int cy, cx;
    getyx(win, cy, cx);

    if (text[i] == '\n') {
      if (cy < max_y - 2) wmove(win, cy + 1, 1);
      i++;
      continue;
    }

    // Determine UTF-8 character length
    unsigned char c = text[i];
    int clen = 1;
    if ((c & 0xE0) == 0xC0) clen = 2;
    else if ((c & 0xF0) == 0xE0) clen = 3;  // Korean, CJK
    else if ((c & 0xF8) == 0xF0) clen = 4;
    // Display width: ASCII=1, CJK/Korean=2
    int cwidth = (clen >= 3) ? 2 : 1;

    if (cx + cwidth > max_x - 2) {
      if (cy < max_y - 2) wmove(win, cy + 1, 1);
      else { scroll(win); wmove(win, max_y - 2, 1); }
    }

    // Extract the character and print with waddnstr
    std::string ch = text.substr(i, clen);
    waddnstr(win, ch.c_str(), clen);
    i += clen;
  }
  wrefresh(win);
}

// ═══════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
  gflags::SetUsageMessage("MatQSD Split-Screen Demo");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model.empty()) {
    fprintf(stderr, "Error: --model required\n"); return 1;
  }

#if defined(ET_USE_THREADPOOL)
  if (FLAGS_cpu_threads > 0)
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(FLAGS_cpu_threads);
#endif

  // ── Load model (before ncurses, redirect stderr during load) ──────────
  fprintf(stdout, "Loading model: %s\n", FLAGS_model.c_str());
  fflush(stdout);
  // Suppress noisy ET/XNNPACK diagnostics during load
  int saved_stderr = dup(STDERR_FILENO);
  int devnull = open("/dev/null", O_WRONLY);
  dup2(devnull, STDERR_FILENO);
  close(devnull);

  setenv("MQINT8_MODE", "1", 1);
  auto model = example::create_llama_runner(
      FLAGS_model, FLAGS_tokenizer, std::nullopt);
  if (!model) {
    dup2(saved_stderr, STDERR_FILENO);
    fprintf(stderr, "Failed to load\n"); return 1;
  }
  model->load();
  auto* dec = model->get_decoder_runner();
  auto* tok = model->get_tokenizer();

  // Restore stderr
  dup2(saved_stderr, STDERR_FILENO);
  close(saved_stderr);
  fprintf(stdout, "Model loaded. Starting demo...\n");
  fflush(stdout);

  // Measure cold CPU baseline for throttle detection
  g_cpu_bench_baseline = cpu_bench_once();

  int K = FLAGS_K;
  int max_sl = FLAGS_seq_len;
  float rp = FLAGS_repetition_penalty;
  auto sys_toks = build_system(tok, FLAGS_system_prompt);

  // ── Init ncurses ──────────────────────────────────────────────────────
  setlocale(LC_ALL, "");
  initscr();
  clear();
  refresh();
  cbreak();
  noecho();
  curs_set(0);
  start_color();
  init_pair(1, COLOR_CYAN, COLOR_BLACK);    // AR-8bit
  init_pair(2, COLOR_GREEN, COLOR_BLACK);   // MatQSD
  init_pair(3, COLOR_YELLOW, COLOR_BLACK);  // prompt/stats
  init_pair(4, COLOR_WHITE, COLOR_BLACK);   // text

  int rows, cols;
  getmaxyx(stdscr, rows, cols);
  int half = cols / 2;

  // Create windows: status bar (top), left (AR), right (SD), input (bottom)
  WINDOW* status_win = newwin(1, cols, 0, 0);
  int text_rows = rows - 4; // 1 status + text + 3 input
  WINDOW* left_border  = newwin(text_rows, half, 1, 0);
  WINDOW* right_border = newwin(text_rows, cols - half, 1, half);
  WINDOW* left_win  = derwin(left_border, text_rows - 2, half - 2, 1, 1);
  WINDOW* right_win = derwin(right_border, text_rows - 2, cols - half - 2, 1, 1);
  WINDOW* input_win = newwin(3, cols, 1 + text_rows, 0);

  scrollok(left_win, TRUE);
  scrollok(right_win, TRUE);

  auto draw_borders = [&]() {
    werase(left_border); werase(right_border); werase(input_win);
    box(left_border, 0, 0);
    box(right_border, 0, 0);
    box(input_win, 0, 0);
    wattron(left_border, COLOR_PAIR(1) | A_BOLD);
    mvwprintw(left_border, 0, 2, " AR-8bit ");
    wattroff(left_border, COLOR_PAIR(1) | A_BOLD);
    wattron(right_border, COLOR_PAIR(2) | A_BOLD);
    mvwprintw(right_border, 0, 2, " MatQSD (SD K=%d) ", K);
    wattroff(right_border, COLOR_PAIR(2) | A_BOLD);
    wrefresh(left_border);
    wrefresh(right_border);
    wrefresh(input_win);
  };

  int64_t cur_pos = 0; // tracks latest SD pos for display
  thermal_init(); // open IOHID for temperature reading

  auto update_status = [&](const char* phase = "") {
    auto si = get_sys_info();
    double temp = get_cpu_temp();
    werase(status_win);
    wbkgd(status_win, COLOR_PAIR(3) | A_BOLD);
    if (temp > 0)
      mvwprintw(status_win, 0, 1, " RSS: %d MB | CPU: %.0f%% | Temp: %.0f C | KV: %lld/%d | %s ",
                si.rss_mb, si.cpu_pct, temp, cur_pos, max_sl, phase);
    else
      mvwprintw(status_win, 0, 1, " RSS: %d MB | CPU: %.0f%% | KV: %lld/%d | %s ",
                si.rss_mb, si.cpu_pct, cur_pos, max_sl, phase);
    wrefresh(status_win);
  };

  draw_borders();
  update_status("Ready");

  // ── Input helper (ncurses with UTF-8 wide char support) ───────────────
  scrollok(input_win, FALSE);  // prevent input window from scrolling

  auto redraw_input = [&](const std::string& line) {
    werase(input_win);
    box(input_win, 0, 0);
    wattron(input_win, COLOR_PAIR(3));
    mvwprintw(input_win, 1, 1, "You> ");
    wattroff(input_win, COLOR_PAIR(3));
    // Show tail of line if too long for window
    int iw = getmaxx(input_win) - 8; // usable width after "You> " and border
    if ((int)line.size() <= iw) {
      mvwaddnstr(input_win, 1, 6, line.c_str(), line.size());
    } else {
      // Show last iw bytes (approximate)
      mvwaddnstr(input_win, 1, 6, line.c_str() + line.size() - iw, iw);
    }
    wrefresh(input_win);
  };

  auto get_input = [&]() -> std::string {
    std::string line;
    redraw_input(line);
    keypad(input_win, TRUE);
    meta(input_win, TRUE);
    wtimeout(input_win, 500); // 500ms timeout for periodic status update
    curs_set(1);
    while (true) {
      int ch = wgetch(input_win);
      if (ch == ERR) {
        // Timeout — refresh status bar and continue waiting
        update_status("Idle");
        continue;
      }
      if (ch == '\n' || ch == KEY_ENTER) break;
      if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
        if (!line.empty()) {
          // Remove last UTF-8 char
          size_t i = line.size() - 1;
          while (i > 0 && (line[i] & 0xC0) == 0x80) i--;
          line.erase(i);
          redraw_input(line);
        }
        continue;
      }
      if (ch >= 0x80) {
        // UTF-8 multibyte
        unsigned char first = (unsigned char)ch;
        int need = 0;
        if ((first & 0xE0) == 0xC0) need = 1;
        else if ((first & 0xF0) == 0xE0) need = 2;
        else if ((first & 0xF8) == 0xF0) need = 3;
        line.push_back((char)first);
        for (int b = 0; b < need; b++) {
          int next = wgetch(input_win);
          line.push_back((char)next);
        }
        redraw_input(line);
      } else if (ch >= 32 && ch < 127) {
        line.push_back((char)ch);
        redraw_input(line);
      }
    }
    curs_set(0);
    wtimeout(input_win, -1); // restore blocking mode
    return line;
  };

  // ── Main loop ─────────────────────────────────────────────────────────
  // Track history for re-prefill
  std::vector<std::vector<int64_t>> ar_history;  // AR-8bit conversation
  std::vector<std::vector<int64_t>> sd_history;  // MatQSD SD conversation

  // ── Example multi-turn scenarios ────────────────────────────────────
  struct Example {
    const char* name;
    std::vector<const char*> turns;
  };
  std::vector<Example> examples = {
    {"CPU Architecture", {
      "What are the key differences between CPU and GPU architectures?",
      "How does pipelining improve CPU performance?",
      "Explain branch prediction and why it matters.",
    }},
    {"Fibonacci Code", {
      "Write a Python function to compute fibonacci numbers recursively.",
      "Now optimize it with memoization.",
      "What is the time complexity of each approach?",
    }},
    {"Thermodynamics", {
      "What are the three laws of thermodynamics? Explain each one briefly.",
      "How does the second law relate to entropy in everyday life?",
      "Give me a simple mathematical example of entropy calculation.",
    }},
    {"Travel Planning", {
      "I'm planning a 5-day trip to Tokyo. What areas should I visit?",
      "What's the best way to get around the city?",
      "Recommend some must-try local foods and where to find them.",
    }},
    {"Machine Learning", {
      "Explain the difference between supervised and unsupervised learning.",
      "What is gradient descent and how does it work?",
      "Why do neural networks need activation functions?",
    }},
  };
  int example_turn = -1;           // current turn index in active example
  int example_idx = -1;            // active example scenario index

  while (true) {
    // Auto-fill from example if active
    std::string input;
    if (example_idx >= 0 && example_turn < (int)examples[example_idx].turns.size()) {
      // Pre-fill the input with example prompt
      input = get_input();
      // If user just pressed enter with empty input, use the example
      if (input.empty()) {
        input = examples[example_idx].turns[example_turn];
        example_turn++;
      } else {
        // User typed something else, cancel example auto-fill
        example_idx = -1;
        example_turn = -1;
      }
    } else {
      example_idx = -1;  // example finished
      input = get_input();
    }

    if (input == "/quit" || input == "/exit") break;
    if (input == "/reset") {
      model->reset();
      ar_history.clear();
      sd_history.clear();
      werase(left_win); werase(right_win);
      draw_borders();
      continue;
    }
    if (input == "/example" || input == "/ex") {
      // Show example menu in input area
      werase(input_win); box(input_win, 0, 0);
      // Use left panel to show menu
      wattron(left_win, COLOR_PAIR(3) | A_BOLD);
      waddstr(left_win, "\n=== Example Scenarios ===\n");
      wattroff(left_win, A_BOLD);
      for (int e = 0; e < (int)examples.size(); e++) {
        char line[128];
        snprintf(line, sizeof(line), "  %d. %s (%d turns)\n",
                 e + 1, examples[e].name, (int)examples[e].turns.size());
        wattron(left_win, COLOR_PAIR(3));
        waddstr(left_win, line);
        wattroff(left_win, COLOR_PAIR(3));
      }
      wattron(left_win, COLOR_PAIR(3));
      waddstr(left_win, "\nSelect (1-5), press Enter for each turn:\n");
      wattroff(left_win, COLOR_PAIR(3));
      wrefresh(left_win);

      // Get selection
      std::string sel = get_input();
      int choice = 0;
      try { choice = std::stoi(sel); } catch (...) {}
      if (choice >= 1 && choice <= (int)examples.size()) {
        example_idx = choice - 1;
        example_turn = 0;
        // Reset conversation for clean example
        model->reset();
        ar_history.clear();
        sd_history.clear();
        werase(left_win); werase(right_win);
        draw_borders();
        // Show first prompt hint
        wattron(left_win, COLOR_PAIR(3));
        char hint[256];
        snprintf(hint, sizeof(hint), "[%s] Press Enter for: \"%s\"\n",
                 examples[example_idx].name,
                 examples[example_idx].turns[0]);
        waddstr(left_win, hint);
        wattroff(left_win, COLOR_PAIR(3));
        wrefresh(left_win);
        wattron(right_win, COLOR_PAIR(3));
        waddstr(right_win, hint);
        wattroff(right_win, COLOR_PAIR(3));
        wrefresh(right_win);
      }
      continue;
    }
    if (input.empty()) {
      // If example active, show hint for next turn
      if (example_idx >= 0 && example_turn < (int)examples[example_idx].turns.size()) {
        input = examples[example_idx].turns[example_turn];
        example_turn++;
        // Show next turn hint if available
        if (example_turn < (int)examples[example_idx].turns.size()) {
          wattron(left_win, COLOR_PAIR(3));
          char hint[256];
          snprintf(hint, sizeof(hint), "\n[Next: \"%s\"]\n",
                   examples[example_idx].turns[example_turn]);
          // Will show after generation
        }
      } else {
        continue;
      }
    }

    auto turn_toks = build_user(tok, input);

    int lh, lw, rh, rw;
    getmaxyx(left_win, lh, lw);
    getmaxyx(right_win, rh, rw);

    // Print user input in both panels
    auto print_user = [&](WINDOW* win, int w) {
      wattron(win, COLOR_PAIR(3) | A_BOLD);
      waddnstr(win, "\nYou> ", 6);
      wattroff(win, COLOR_PAIR(3) | A_BOLD);
      // Truncate input display to fit panel width
      std::string display = input;
      if ((int)display.size() > w - 8) display = display.substr(0, w - 11) + "...";
      waddnstr(win, display.c_str(), display.size());
      waddch(win, '\n');
      wrefresh(win);
    };
    print_user(left_win, lw);
    print_user(right_win, rw);

    // Last token of prompt (for correct decode prev context)
    int64_t last_prompt_tok = turn_toks.empty() ? TOK_BOS : turn_toks.back();

    double ar8_tps_ref = 0;
    double sd_tps = 0;
    std::vector<int64_t> sd_gen_tokens;

    // ── Phase 1: AR-8bit (left panel, runs first) ──────────────────────
    {
      model->reset();
      int64_t pos = 0;
      xnn_set_mqint8_global_mode(1);
      prefill(dec, sys_toks, pos);
      for (auto& h : ar_history) prefill(dec, h, pos);
      int64_t first = prefill(dec, turn_toks, pos);

      wattron(left_win, COLOR_PAIR(1) | A_BOLD);
      waddnstr(left_win, "Bot> ", 5);
      wattroff(left_win, A_BOLD);
      wattron(left_win, COLOR_PAIR(1));
      wrefresh(left_win);
      auto t0 = std::chrono::high_resolution_clock::now();
      int64_t cur = first;
      std::vector<int64_t> gen = {cur};
      if (!is_control_token(cur)) {
        auto dr = tok->decode(cur, last_prompt_tok);
        if (dr.error() == ::tokenizers::Error::Ok)
          win_add_text(left_win, dr.get(), lh, lw);
      }
      int max_gen = max_sl - (int)pos;
      while ((int)gen.size() < max_gen) {
        if (pos >= (int64_t)max_sl) break;
        int64_t next = step1_pen(dec, cur, pos, gen, rp);
        pos++;
        if (next < 0) break;
        gen.push_back(next);
        if (next == TOK_EOS || next == TOK_EOT) break;
        if (!is_control_token(next)) {
          uint64_t prev = gen.size() > 1 ? gen[gen.size()-2] : gen.back();
          auto dr = tok->decode(next, prev);
          if (dr.error() == ::tokenizers::Error::Ok)
            win_add_text(left_win, dr.get(), lh, lw);
        }
        cur = next;
        if (gen.size() % 10 == 0) {
          char buf[64];
          snprintf(buf, sizeof(buf), "AR-8bit | %d tokens", (int)gen.size());
          update_status(buf);
        }
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      double elapsed = std::chrono::duration<double>(t1 - t0).count();
      double tps = gen.size() / std::max(elapsed, 1e-9);
      wattroff(left_win, COLOR_PAIR(1));
      ar8_tps_ref = tps;

      wattron(left_win, COLOR_PAIR(3));
      char ar_stat[80];
      snprintf(ar_stat, sizeof(ar_stat), "\n[%d tok | %.1f TPS | %.2fs]\n",
               (int)gen.size(), tps, elapsed);
      waddstr(left_win, ar_stat);
      wattroff(left_win, COLOR_PAIR(3));
      wrefresh(left_win);

      // Save AR turn
      std::vector<int64_t> ar_full_turn = turn_toks;
      for (auto t : gen) ar_full_turn.push_back(t);
      if (ar_full_turn.empty() || ar_full_turn.back() != TOK_EOT)
        ar_full_turn.push_back(TOK_EOT);
      ar_history.push_back(ar_full_turn);
    }

    // ── Cooldown: wait until CPU temperature drops ─────────────────
    {
      double target_temp = 52.0; // target temperature in Celsius
      int max_wait = 9999;       // no max — wait until target temp
      for (int i = 0; i < max_wait; i++) {
        double temp = get_cpu_temp();
        char cd_buf[80];
        if (temp > 0) {
          snprintf(cd_buf, sizeof(cd_buf),
                   "Cooling down... %.0f C (target < %.0f C) [%ds]",
                   temp, target_temp, i);
          update_status(cd_buf);
          if (temp <= target_temp) break;
        } else {
          snprintf(cd_buf, sizeof(cd_buf), "Cooling down... %ds", max_wait - i);
          update_status(cd_buf);
        }
        sleep(1);
      }
      update_status("Ready for MatQSD");
    }

    // ── Phase 2: MatQSD SD (right panel, runs after cooldown) ─────────
    {
      model->reset();
      int64_t pos = 0;
      xnn_set_mqint8_global_mode(1);
      prefill(dec, sys_toks, pos);
      for (auto& h : sd_history) prefill(dec, h, pos);
      int64_t first = prefill(dec, turn_toks, pos);

      wattron(right_win, COLOR_PAIR(2) | A_BOLD);
      waddnstr(right_win, "Bot> ", 5);
      wattroff(right_win, A_BOLD);
      wattron(right_win, COLOR_PAIR(2));
      wrefresh(right_win);
      auto t0 = std::chrono::high_resolution_clock::now();
      int64_t cur = first;
      std::vector<int64_t> gen = {cur};
      int drafted = 0, accepted = 0, steps = 0;

      if (!is_control_token(cur)) {
        auto dr = tok->decode(cur, last_prompt_tok);
        if (dr.error() == ::tokenizers::Error::Ok)
          win_add_text(right_win, dr.get(), rh, rw);
      }

      int max_gen = max_sl - (int)pos;
      while ((int)gen.size() < max_gen) {
        // Draft
        xnn_set_mqint8_global_mode(0);
        std::vector<int64_t> dtoks;
        int64_t dt = cur;
        for (int d = 0; d < K && pos + (int)dtoks.size() < (int64_t)max_sl - 1; d++) {
          dt = step1_pen(dec, dt, pos + d, gen, rp);
          if (dt < 0) break;
          dtoks.push_back(dt);
          if (dt == TOK_EOS || dt == TOK_EOT) break;
        }
        drafted += dtoks.size();
        if (dtoks.empty()) break;

        // Verify
        xnn_set_mqint8_global_mode(1);
        std::vector<int64_t> vi;
        vi.push_back(cur);
        for (auto& t : dtoks) vi.push_back(t);
        auto vt = make_tensor_ptr({1, (int)vi.size()}, std::vector<int64_t>(vi));
        auto vr = dec->step(vt, pos);
        std::vector<int64_t> tt;
        if (vr.ok()) {
          auto& vlogits = vr.get();
          float* vdata = vlogits.mutable_data_ptr<float>();
          if (vlogits.dim() == 3) {
            int vseq = vlogits.size(1), vvocab = vlogits.size(2);
            for (int s = 0; s < vseq; s++)
              apply_rep_penalty(vdata + s * vvocab, vvocab, gen, rp);
          } else {
            int vvocab = vlogits.size(vlogits.dim() - 1);
            apply_rep_penalty(vdata, vvocab, gen, rp);
          }
          tt = all_argmax(vlogits);
        }

        int nacc = 0;
        for (int i = 0; i < (int)dtoks.size() && i < (int)tt.size(); i++) {
          if (tt[i] == dtoks[i]) nacc++; else break;
        }
        accepted += nacc;

        bool hit_eos = false;
        for (int i = 0; i < nacc; i++) {
          int64_t t = dtoks[i];
          gen.push_back(t);
          if (t == TOK_EOS || t == TOK_EOT) { hit_eos = true; break; }
          if (!is_control_token(t)) {
            uint64_t prev = gen.size() > 1 ? gen[gen.size()-2] : gen.back();
            auto dr = tok->decode(t, prev);
            if (dr.error() == ::tokenizers::Error::Ok)
              win_add_text(right_win, dr.get(), rh, rw);
          }
        }
        if (hit_eos) { pos += nacc; break; }

        int64_t bonus = (nacc < (int)tt.size()) ? tt[nacc] : tt.back();
        gen.push_back(bonus);
        if (bonus != TOK_EOS && bonus != TOK_EOT && !is_control_token(bonus)) {
          uint64_t prev = gen.size() > 1 ? gen[gen.size()-2] : gen.back();
          auto dr = tok->decode(bonus, prev);
          if (dr.error() == ::tokenizers::Error::Ok)
            win_add_text(right_win, dr.get(), rh, rw);
        }
        cur = bonus;
        pos += nacc + 1;
        cur_pos = pos;
        steps++;
        if (bonus == TOK_EOS || bonus == TOK_EOT) break;
        if (gen.size() % 10 == 0) {
          double a = drafted > 0 ? (double)accepted / drafted * 100 : 0;
          char buf[80];
          snprintf(buf, sizeof(buf), "MatQSD | %d tok | a=%.0f%%", (int)gen.size(), a);
          update_status(buf);
        }
      }

      auto t1 = std::chrono::high_resolution_clock::now();
      double elapsed = std::chrono::duration<double>(t1 - t0).count();
      sd_tps = gen.size() / std::max(elapsed, 1e-9);
      double alpha = drafted > 0 ? (double)accepted / drafted : 0;
      wattroff(right_win, COLOR_PAIR(2));

      sd_gen_tokens = gen;
      double speedup = ar8_tps_ref > 0 ? sd_tps / ar8_tps_ref : 0;
      wattron(right_win, COLOR_PAIR(3));
      char sd_stat[96];
      snprintf(sd_stat, sizeof(sd_stat), "\n[%d tok | %.1f TPS | %.2fs | a=%.0f%% | %.2fx]\n",
               (int)gen.size(), sd_tps, elapsed, alpha * 100, speedup);
      waddstr(right_win, sd_stat);
      wattroff(right_win, COLOR_PAIR(3));
      wrefresh(right_win);

      // Save SD turn
      std::vector<int64_t> sd_full_turn = turn_toks;
      for (auto t : gen) sd_full_turn.push_back(t);
      if (sd_full_turn.empty() || sd_full_turn.back() != TOK_EOT)
        sd_full_turn.push_back(TOK_EOT);
      sd_history.push_back(sd_full_turn);

      // Final status bar
      char final_buf[128];
      snprintf(final_buf, sizeof(final_buf),
               "Done | AR: %.1f TPS  MatQSD: %.1f TPS  Speedup: %.2fx",
               sd_tps, ar8_tps_ref, speedup);
      update_status(final_buf);
    }

    // Update bottom bar — show next example hint if available
    werase(input_win); box(input_win, 0, 0);
    wattron(input_win, COLOR_PAIR(3) | A_BOLD);
    if (example_idx >= 0 && example_turn < (int)examples[example_idx].turns.size()) {
      char next_hint[256];
      snprintf(next_hint, sizeof(next_hint), "Enter → \"%s\"",
               examples[example_idx].turns[example_turn]);
      mvwprintw(input_win, 1, 1, "%s", next_hint);
    } else {
      mvwprintw(input_win, 1, 1, "/example for scenarios, /quit to exit, /reset to clear");
    }
    wattroff(input_win, COLOR_PAIR(3) | A_BOLD);
    wrefresh(input_win);

    // Wait for keypress before next turn
    // Wait for keypress
    wgetch(input_win);
  }

  thermal_close();
  endwin();
  return 0;
}
