/*
 * Gemma-4 E2B MatQSD Split-Screen Demo
 * ncurses TUI: Left panel = AR-8bit, Right panel = MatQSD SD
 *
 * Key difference from Llama demo: processes tokens one-at-a-time (no batch
 * prefill) because the exported model uses scalar input_pos ({0:1}).
 * Gemma-4 chat template: <bos> user prompt <turn|> model response <turn|>
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

DEFINE_string(model, "", "Path to Gemma-4 .pte model");
DEFINE_string(tokenizer, "tokenizer.json", "Path to HF tokenizer.json");
DEFINE_int32(K, 3, "Draft tokens per SD step");
DEFINE_int32(seq_len, 512, "Max sequence length");
DEFINE_int32(cpu_threads, 4, "CPU threads");
DEFINE_double(repetition_penalty, 1.05, "Repetition penalty");
DEFINE_bool(sd_first, false, "Run MatQSD SD first, then AR-8bit (default: AR first)");

namespace llm = ::executorch::extension::llm;
using ::executorch::extension::TensorPtr;
using ::executorch::extension::make_tensor_ptr;

// Gemma-4 special tokens
static constexpr int64_t TOK_BOS = 2, TOK_EOS = 1, TOK_TURN = 106;

// ── CPU temp (same as Llama demo) ──────────────────────────────────────
static IOHIDEventSystemClientRef g_hid_client = nullptr;
static void thermal_init() { g_hid_client = IOHIDEventSystemClientCreate(kCFAllocatorDefault); }
static void thermal_close() { if (g_hid_client) { CFRelease(g_hid_client); g_hid_client = nullptr; } }
static double get_cpu_temp() {
  if (!g_hid_client) return -1;
  CFArrayRef svcs = IOHIDEventSystemClientCopyServices(g_hid_client);
  if (!svcs) return -1;
  double mx = -1;
  for (long i = 0; i < CFArrayGetCount(svcs); i++) {
    void* svc = (void*)CFArrayGetValueAtIndex(svcs, i);
    CFStringRef nm = IOHIDServiceClientCopyProperty(svc, CFSTR("Product"));
    if (!nm) continue;
    char buf[128] = {};
    CFStringGetCString(nm, buf, sizeof(buf), kCFStringEncodingUTF8);
    CFRelease(nm);
    if (strncmp(buf, "PMU", 3) != 0) continue;
    IOHIDEventRef ev = IOHIDServiceClientCopyEvent(svc, 15, 0, 0);
    if (ev) { double t = IOHIDEventGetFloatValue(ev, 15 << 16); if (t > mx) mx = t; CFRelease(ev); }
  }
  CFRelease(svcs);
  return mx;
}

// ── System info ────────────────────────────────────────────────────────
struct SysInfo { int rss_mb; double cpu_pct; };
static SysInfo get_sys_info() {
  SysInfo si = {0, 0};
  mach_task_basic_info_data_t ti;
  mach_msg_type_number_t cnt = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&ti, &cnt) == KERN_SUCCESS)
    si.rss_mb = (int)(ti.resident_size / (1024 * 1024));
  return si;
}

// ── Step: single token forward ─────────────────────────────────────────
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
  for (auto t : past) {
    if (t >= 0 && t < vocab) { if (ld[t] > 0) ld[t] /= pen; else ld[t] *= pen; }
  }
  int b = 0; for (int i = 1; i < vocab; i++) if (ld[i] > ld[b]) b = i;
  return b;
}

// ── Batch prefill (requires dynamic input_pos export) ──────────────────
static int64_t prefill(llm::TextDecoderRunner* dec,
                       const std::vector<int64_t>& toks, int64_t& pos) {
  if (toks.empty()) return -1;
  auto input = make_tensor_ptr({1, (int)toks.size()}, std::vector<int64_t>(toks));
  auto result = dec->step(input, pos);
  pos += toks.size();
  if (!result.ok()) return -1;
  return dec->logits_to_token(result.get(), 0.0f);
}

// ── Multi-token verify for SD (uses generate_full_logits) ──────────────
static std::vector<int64_t> all_argmax(const executorch::aten::Tensor& logits) {
  std::vector<int64_t> r;
  const float* d = logits.const_data_ptr<float>();
  if (logits.dim() == 3) {
    int seq = logits.size(1), vocab = logits.size(2);
    for (int s = 0; s < seq; s++) {
      const float* row = d + s * vocab;
      int best = 0; for (int v = 1; v < vocab; v++) if (row[v] > row[best]) best = v;
      r.push_back(best);
    }
  } else {
    int vocab = logits.size(logits.dim() - 1);
    int best = 0; for (int v = 1; v < vocab; v++) if (d[v] > d[best]) best = v;
    r.push_back(best);
  }
  return r;
}

static void apply_rep_penalty(float* logits, int vocab,
                              const std::vector<int64_t>& past, float pen) {
  if (pen <= 1.0f) return;
  for (auto t : past) {
    if (t >= 0 && t < vocab) { if (logits[t] > 0) logits[t] /= pen; else logits[t] *= pen; }
  }
}

static bool is_eos(int64_t t) { return t == TOK_EOS || t == TOK_TURN; }

// ── ncurses text helper ────────────────────────────────────────────────
static void win_add_text(WINDOW* win, const std::string& text, int max_y, int max_x) {
  for (size_t i = 0; i < text.size(); ) {
    int cy, cx; getyx(win, cy, cx);
    if (text[i] == '\n') { if (cy < max_y - 2) wmove(win, cy + 1, 1); i++; continue; }
    unsigned char c = text[i];
    int clen = 1;
    if ((c & 0xE0) == 0xC0) clen = 2;
    else if ((c & 0xF0) == 0xE0) clen = 3;
    else if ((c & 0xF8) == 0xF0) clen = 4;
    int cwidth = (clen >= 3) ? 2 : 1;
    if (cx + cwidth > max_x - 2) {
      if (cy < max_y - 2) wmove(win, cy + 1, 1);
      else { scroll(win); wmove(win, max_y - 2, 1); }
    }
    std::string ch = text.substr(i, clen);
    waddnstr(win, ch.c_str(), clen);
    i += clen;
  }
  wrefresh(win);
}

// ═══════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
  gflags::SetUsageMessage("Gemma-4 E2B MatQSD Split-Screen Demo");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model.empty()) { fprintf(stderr, "--model required\n"); return 1; }

#if defined(ET_USE_THREADPOOL)
  if (FLAGS_cpu_threads > 0)
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(FLAGS_cpu_threads);
#endif

  fprintf(stdout, "Loading model: %s\n", FLAGS_model.c_str()); fflush(stdout);
  int saved = dup(STDERR_FILENO); int devnull = open("/dev/null", O_WRONLY);
  dup2(devnull, STDERR_FILENO); close(devnull);
  setenv("MQINT8_MODE", "1", 1);
  auto model = example::create_llama_runner(FLAGS_model, FLAGS_tokenizer, std::nullopt);
  if (!model) { dup2(saved, STDERR_FILENO); fprintf(stderr, "Failed to load\n"); return 1; }
  model->load();
  auto* dec = model->get_decoder_runner();
  auto* tok = model->get_tokenizer();
  dup2(saved, STDERR_FILENO); close(saved);
  fprintf(stdout, "Model loaded.\n"); fflush(stdout);

  // Keep stderr suppressed during ncurses to prevent tokenizer log leaks
  int saved_stderr2 = dup(STDERR_FILENO);
  int devnull2 = open("/dev/null", O_WRONLY);
  dup2(devnull2, STDERR_FILENO); close(devnull2);

  int K = FLAGS_K;
  int max_sl = FLAGS_seq_len;
  float rp = FLAGS_repetition_penalty;

  // ── ncurses init ─────────────────────────────────────────────────────
  setlocale(LC_ALL, "");
  initscr(); clear(); refresh(); cbreak(); noecho(); curs_set(0); start_color();
  init_pair(1, COLOR_CYAN, COLOR_BLACK);
  init_pair(2, COLOR_GREEN, COLOR_BLACK);
  init_pair(3, COLOR_YELLOW, COLOR_BLACK);
  init_pair(4, COLOR_WHITE, COLOR_BLACK);

  int rows, cols; getmaxyx(stdscr, rows, cols);
  int half = cols / 2;
  WINDOW* status_win = newwin(1, cols, 0, 0);
  int text_rows = rows - 4;
  WINDOW* left_border  = newwin(text_rows, half, 1, 0);
  WINDOW* right_border = newwin(text_rows, cols - half, 1, half);
  WINDOW* left_win  = derwin(left_border, text_rows - 2, half - 2, 1, 1);
  WINDOW* right_win = derwin(right_border, text_rows - 2, cols - half - 2, 1, 1);
  WINDOW* input_win = newwin(3, cols, 1 + text_rows, 0);
  scrollok(left_win, TRUE); scrollok(right_win, TRUE);
  thermal_init();

  auto draw_borders = [&]() {
    werase(left_border); werase(right_border); werase(input_win);
    box(left_border, 0, 0); box(right_border, 0, 0); box(input_win, 0, 0);
    wattron(left_border, COLOR_PAIR(1) | A_BOLD);
    mvwprintw(left_border, 0, 2, " AR-8bit ");
    wattroff(left_border, COLOR_PAIR(1) | A_BOLD);
    wattron(right_border, COLOR_PAIR(2) | A_BOLD);
    mvwprintw(right_border, 0, 2, " MatQSD (SD K=%d) ", K);
    wattroff(right_border, COLOR_PAIR(2) | A_BOLD);
    wrefresh(left_border); wrefresh(right_border); wrefresh(input_win);
  };

  int64_t cur_pos = 0;
  auto update_status = [&](const char* phase = "") {
    auto si = get_sys_info(); double temp = get_cpu_temp();
    werase(status_win); wbkgd(status_win, COLOR_PAIR(3) | A_BOLD);
    if (temp > 0) mvwprintw(status_win, 0, 1, " RSS: %dMB | Temp: %.0fC | KV: %lld/%d | %s ",
                            si.rss_mb, temp, cur_pos, max_sl, phase);
    else mvwprintw(status_win, 0, 1, " RSS: %dMB | KV: %lld/%d | %s ", si.rss_mb, cur_pos, max_sl, phase);
    wrefresh(status_win);
  };

  draw_borders(); update_status("Ready — Gemma-4 E2B");

  // ── Input helper ─────────────────────────────────────────────────────
  scrollok(input_win, FALSE);
  auto redraw_input = [&](const std::string& line) {
    werase(input_win); box(input_win, 0, 0);
    wattron(input_win, COLOR_PAIR(3)); mvwprintw(input_win, 1, 1, "You> ");
    wattroff(input_win, COLOR_PAIR(3));
    int iw = getmaxx(input_win) - 8;
    if ((int)line.size() <= iw) mvwaddnstr(input_win, 1, 6, line.c_str(), line.size());
    else mvwaddnstr(input_win, 1, 6, line.c_str() + line.size() - iw, iw);
    wrefresh(input_win);
  };
  auto get_input = [&]() -> std::string {
    std::string line; redraw_input(line);
    keypad(input_win, TRUE); meta(input_win, TRUE);
    wtimeout(input_win, 500); curs_set(1);
    while (true) {
      int ch = wgetch(input_win);
      if (ch == ERR) { update_status("Idle"); continue; }
      if (ch == '\n' || ch == KEY_ENTER) break;
      if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
        if (!line.empty()) { size_t i = line.size()-1; while (i>0 && (line[i]&0xC0)==0x80) i--; line.erase(i); redraw_input(line); }
        continue;
      }
      if (ch >= 0x80) {
        unsigned char f = (unsigned char)ch; int need = 0;
        if ((f&0xE0)==0xC0) need=1; else if ((f&0xF0)==0xE0) need=2; else if ((f&0xF8)==0xF0) need=3;
        line.push_back((char)f); for (int b=0; b<need; b++) line.push_back((char)wgetch(input_win));
        redraw_input(line);
      } else if (ch >= 32 && ch < 127) { line.push_back((char)ch); redraw_input(line); }
    }
    curs_set(0); wtimeout(input_win, -1); return line;
  };

  // ── Example scenarios ─────────────────────────────────────────────────
  struct Example { const char* name; std::vector<const char*> turns; };
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
  int example_turn = -1, example_idx = -1;

  // ── Main loop ────────────────────────────────────────────────────────
  while (true) {
    std::string input;
    if (example_idx >= 0 && example_turn < (int)examples[example_idx].turns.size()) {
      input = get_input();
      if (input.empty()) { input = examples[example_idx].turns[example_turn]; example_turn++; }
      else { example_idx = -1; example_turn = -1; }
    } else {
      example_idx = -1;
      input = get_input();
    }
    if (input == "/quit" || input == "/exit") break;
    if (input == "/reset") {
      model->reset(); werase(left_win); werase(right_win);
      draw_borders(); continue;
    }
    if (input == "/example" || input == "/ex") {
      wattron(left_win, COLOR_PAIR(3) | A_BOLD);
      waddstr(left_win, "\n=== Example Scenarios ===\n");
      wattroff(left_win, A_BOLD);
      for (int e = 0; e < (int)examples.size(); e++) {
        char line[128]; snprintf(line, sizeof(line), "  %d. %s (%d turns)\n",
                 e+1, examples[e].name, (int)examples[e].turns.size());
        wattron(left_win, COLOR_PAIR(3)); waddstr(left_win, line); wattroff(left_win, COLOR_PAIR(3));
      }
      wattron(left_win, COLOR_PAIR(3));
      waddstr(left_win, "\nSelect (1-5), press Enter for each turn:\n");
      wattroff(left_win, COLOR_PAIR(3)); wrefresh(left_win);
      std::string sel = get_input();
      int choice = 0; try { choice = std::stoi(sel); } catch (...) {}
      if (choice >= 1 && choice <= (int)examples.size()) {
        example_idx = choice - 1; example_turn = 0;
        model->reset(); werase(left_win); werase(right_win); draw_borders();
        char hint[256]; snprintf(hint, sizeof(hint), "[%s] Press Enter for: \"%s\"\n",
                 examples[example_idx].name, examples[example_idx].turns[0]);
        for (auto* w : {left_win, right_win}) {
          wattron(w, COLOR_PAIR(3)); waddstr(w, hint); wattroff(w, COLOR_PAIR(3)); wrefresh(w);
        }
      }
      continue;
    }
    if (input.empty()) {
      if (example_idx >= 0 && example_turn < (int)examples[example_idx].turns.size()) {
        input = examples[example_idx].turns[example_turn]; example_turn++;
      } else continue;
    }

    // Tokenize: <bos> + user input
    std::vector<int64_t> prompt_toks = {TOK_BOS};
    auto enc = tok->encode(input, 0, 0);
    if (enc.error() == ::tokenizers::Error::Ok)
      for (auto t : enc.get()) prompt_toks.push_back(t);

    int lh, lw, rh, rw;
    getmaxyx(left_win, lh, lw); getmaxyx(right_win, rh, rw);

    // Print user input in both panels
    for (auto* w : {left_win, right_win}) {
      wattron(w, COLOR_PAIR(3) | A_BOLD); waddnstr(w, "\nYou> ", 6);
      wattroff(w, COLOR_PAIR(3) | A_BOLD);
      std::string display = input;
      int pw = (w == left_win) ? lw : rw;
      if ((int)display.size() > pw-8) display = display.substr(0, pw-11) + "...";
      waddnstr(w, display.c_str(), display.size()); waddch(w, '\n'); wrefresh(w);
    }

    int64_t last_prompt_tok = prompt_toks.back();
    double ar8_tps_ref = 0, sd_tps = 0;

    // ── AR-8bit phase ──────────────────────────────────────────────────
    auto run_ar = [&]() {
      model->reset(); int64_t pos = 0;
      xnn_set_mqint8_global_mode(1);
      int64_t first = prefill(dec, prompt_toks, pos);

      wattron(left_win, COLOR_PAIR(1) | A_BOLD); waddnstr(left_win, "Bot> ", 5);
      wattroff(left_win, A_BOLD); wattron(left_win, COLOR_PAIR(1)); wrefresh(left_win);
      auto t0 = std::chrono::high_resolution_clock::now();
      int64_t cur = first;
      std::vector<int64_t> gen = {cur};
      if (!is_eos(cur)) {
        auto dr = tok->decode(cur, last_prompt_tok);
        if (dr.error() == ::tokenizers::Error::Ok) win_add_text(left_win, dr.get(), lh, lw);
      }
      while ((int)gen.size() < max_sl - (int)prompt_toks.size()) {
        if (pos >= max_sl || is_eos(cur)) break;
        int64_t next = step1_pen(dec, cur, pos, gen, rp); pos++;
        if (next < 0) break;
        gen.push_back(next);
        if (is_eos(next)) break;
        auto dr = tok->decode(next, gen.size()>1 ? gen[gen.size()-2] : gen.back());
        if (dr.error() == ::tokenizers::Error::Ok) win_add_text(left_win, dr.get(), lh, lw);
        cur = next;
        if (gen.size() % 10 == 0) {
          char buf[64]; snprintf(buf, sizeof(buf), "AR-8bit | %d tok", (int)gen.size());
          update_status(buf);
        }
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      double el = std::chrono::duration<double>(t1-t0).count();
      ar8_tps_ref = gen.size() / std::max(el, 1e-9);
      wattroff(left_win, COLOR_PAIR(1));
      wattron(left_win, COLOR_PAIR(3));
      char st[80]; snprintf(st, sizeof(st), "\n[%d tok | %.1f TPS | %.2fs]\n", (int)gen.size(), ar8_tps_ref, el);
      waddstr(left_win, st); wattroff(left_win, COLOR_PAIR(3)); wrefresh(left_win);
    };

    // ── MatQSD SD phase ──────────────────────────────────────────────
    auto run_sd = [&]() {
      model->reset(); int64_t pos = 0;
      xnn_set_mqint8_global_mode(1);
      int64_t first = prefill(dec, prompt_toks, pos);

      wattron(right_win, COLOR_PAIR(2) | A_BOLD); waddnstr(right_win, "Bot> ", 5);
      wattroff(right_win, A_BOLD); wattron(right_win, COLOR_PAIR(2)); wrefresh(right_win);
      auto t0 = std::chrono::high_resolution_clock::now();
      int64_t cur = first;
      std::vector<int64_t> gen = {cur};
      int drafted = 0, accepted = 0;
      if (!is_eos(cur)) {
        auto dr = tok->decode(cur, last_prompt_tok);
        if (dr.error() == ::tokenizers::Error::Ok) win_add_text(right_win, dr.get(), rh, rw);
      }

      while ((int)gen.size() < max_sl - (int)prompt_toks.size()) {
        // Draft K tokens with 4-bit
        xnn_set_mqint8_global_mode(0);
        std::vector<int64_t> dtoks;
        int64_t dt = cur;
        for (int d = 0; d < K && pos + (int)dtoks.size() < max_sl - 1; d++) {
          dt = step1_pen(dec, dt, pos + d, gen, rp);
          if (dt < 0) break;
          dtoks.push_back(dt);
          if (is_eos(dt)) break;
        }
        drafted += dtoks.size();
        if (dtoks.empty()) break;

        // Verify with 8-bit
        xnn_set_mqint8_global_mode(1);
        std::vector<int64_t> vi = {cur};
        for (auto& t : dtoks) vi.push_back(t);
        auto vt = make_tensor_ptr({1, (int)vi.size()}, std::vector<int64_t>(vi));
        auto vr = dec->step(vt, pos);
        std::vector<int64_t> tt;
        if (vr.ok()) {
          auto& vl = vr.get();
          float* vd = vl.mutable_data_ptr<float>();
          if (vl.dim() == 3) {
            int vs = vl.size(1), vv = vl.size(2);
            for (int s = 0; s < vs; s++) apply_rep_penalty(vd + s*vv, vv, gen, rp);
          } else {
            int vv = vl.size(vl.dim()-1);
            apply_rep_penalty(vd, vv, gen, rp);
          }
          tt = all_argmax(vl);
        }

        int nacc = 0;
        for (int i = 0; i < (int)dtoks.size() && i < (int)tt.size(); i++) {
          if (tt[i] == dtoks[i]) nacc++; else break;
        }
        accepted += nacc;

        bool hit_eos = false;
        for (int i = 0; i < nacc; i++) {
          int64_t t = dtoks[i]; gen.push_back(t);
          if (is_eos(t)) { hit_eos = true; break; }
          auto dr = tok->decode(t, gen.size()>1 ? gen[gen.size()-2] : gen.back());
          if (dr.error() == ::tokenizers::Error::Ok) win_add_text(right_win, dr.get(), rh, rw);
        }
        if (hit_eos) { pos += nacc; break; }

        int64_t bonus = (nacc < (int)tt.size()) ? tt[nacc] : tt.back();
        gen.push_back(bonus);
        if (!is_eos(bonus)) {
          auto dr = tok->decode(bonus, gen.size()>1 ? gen[gen.size()-2] : gen.back());
          if (dr.error() == ::tokenizers::Error::Ok) win_add_text(right_win, dr.get(), rh, rw);
        }
        cur = bonus; pos += nacc + 1; cur_pos = pos;
        if (is_eos(bonus)) break;
        if (gen.size() % 10 == 0) {
          double a = drafted > 0 ? (double)accepted / drafted * 100 : 0;
          char buf[80]; snprintf(buf, sizeof(buf), "MatQSD | %d tok | a=%.0f%%", (int)gen.size(), a);
          update_status(buf);
        }
      }

      auto t1 = std::chrono::high_resolution_clock::now();
      double el = std::chrono::duration<double>(t1-t0).count();
      sd_tps = gen.size() / std::max(el, 1e-9);
      double alpha = drafted > 0 ? (double)accepted / drafted : 0;
      double speedup = ar8_tps_ref > 0 ? sd_tps / ar8_tps_ref : 0;
      wattroff(right_win, COLOR_PAIR(2));
      wattron(right_win, COLOR_PAIR(3));
      char st[96]; snprintf(st, sizeof(st), "\n[%d tok | %.1f TPS | %.2fs | α=%.0f%% | %.2fx]\n",
               (int)gen.size(), sd_tps, el, alpha*100, speedup);
      waddstr(right_win, st); wattroff(right_win, COLOR_PAIR(3)); wrefresh(right_win);
    };

    // ── Cooldown between phases ──────────────────────────────────────
    auto cooldown = [&]() {
      double target = 52.0;
      for (int i = 0; i < 9999; i++) {
        double t = get_cpu_temp();
        char cd[80];
        if (t > 0) { snprintf(cd, sizeof(cd), "Cooling... %.0fC (< %.0fC) [%ds]", t, target, i);
          update_status(cd); if (t <= target) break;
        } else { snprintf(cd, sizeof(cd), "Cooling... %ds", i); update_status(cd); }
        sleep(1);
      }
    };

    // ── Execute phases in order (--sd_first swaps) ───────────────────
    if (FLAGS_sd_first) {
      run_sd(); cooldown(); run_ar();
    } else {
      run_ar(); cooldown(); run_sd();
    }

    // Final status
    {
      double speedup = ar8_tps_ref > 0 ? sd_tps / ar8_tps_ref : 0;
      char fb[128]; snprintf(fb, sizeof(fb),
        "Done | AR: %.1f TPS  MatQSD: %.1f TPS  Speedup: %.2fx", ar8_tps_ref, sd_tps, speedup);
      update_status(fb);
    }

    werase(input_win); box(input_win, 0, 0);
    wattron(input_win, COLOR_PAIR(3) | A_BOLD);
    if (example_idx >= 0 && example_turn < (int)examples[example_idx].turns.size()) {
      char nh[256]; snprintf(nh, sizeof(nh), "Enter → \"%s\"", examples[example_idx].turns[example_turn]);
      mvwprintw(input_win, 1, 1, "%s", nh);
    } else {
      mvwprintw(input_win, 1, 1, "/example for scenarios, /quit to exit, /reset to clear");
    }
    wattroff(input_win, COLOR_PAIR(3) | A_BOLD); wrefresh(input_win);
    wgetch(input_win);
  }

  thermal_close();
  dup2(saved_stderr2, STDERR_FILENO); close(saved_stderr2);
  endwin(); return 0;
}
