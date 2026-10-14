// PanParasite-Edge Phase 5 unit test: FiLM micro-net + tracker.
// Covers:
//   (a) Boot init + Platt T=1.38 sum-to-1.
//   (b) FiLM modulation impact (fbsc vs zeros; bright vs dark ROI).
//   (c) Taxon routing: ascaris/trichuris/hookworm/mansoni/haematobium/
//       debris via real topology fbsc + unknown via hand vector.
//   (d) INT8/FP32 agreement + forced-fallback recovery + audit counter.
//   (e) Hungarian assignment + IoU>=0.30 gate + empty/new cases.
//   (f) Tracker log-odds: +2.197 VERIFIED / -2.197 REJECTED / -0.10 decay.
//   (g) Slide aggregation: dedup by id, conf >= 0.70.
//   (h) ZERO heap new/delete across hot classify+track (tripwire).

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

static std::atomic<long long> g_newCalls{0};
static std::atomic<long long> g_deleteCalls{0};
static std::atomic<bool> g_countHeap{false};

void* operator new(std::size_t n) {
  if (g_countHeap.load(std::memory_order_relaxed)) g_newCalls.fetch_add(1);
  void* p = std::malloc(n);
  if (!p) throw std::bad_alloc();
  return p;
}
void* operator new[](std::size_t n) {
  if (g_countHeap.load(std::memory_order_relaxed)) g_newCalls.fetch_add(1);
  void* p = std::malloc(n);
  if (!p) throw std::bad_alloc();
  return p;
}
void operator delete(void* p) noexcept {
  if (p == nullptr) return;
  if (g_countHeap.load(std::memory_order_relaxed)) g_deleteCalls.fetch_add(1);
  std::free(p);
}
void operator delete[](void* p) noexcept {
  if (p == nullptr) return;
  if (g_countHeap.load(std::memory_order_relaxed)) g_deleteCalls.fetch_add(1);
  std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
  if (p == nullptr) return;
  if (g_countHeap.load(std::memory_order_relaxed)) g_deleteCalls.fetch_add(1);
  std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
  if (p == nullptr) return;
  if (g_countHeap.load(std::memory_order_relaxed)) g_deleteCalls.fetch_add(1);
  std::free(p);
}
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  if (g_countHeap.load(std::memory_order_relaxed)) g_newCalls.fetch_add(1);
  return std::malloc(n);
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
  if (g_countHeap.load(std::memory_order_relaxed)) g_newCalls.fetch_add(1);
  return std::malloc(n);
}
void operator delete(void* p, const std::nothrow_t&) noexcept {
  if (p == nullptr) return;
  if (g_countHeap.load(std::memory_order_relaxed)) g_deleteCalls.fetch_add(1);
  std::free(p);
}
void operator delete[](void* p, const std::nothrow_t&) noexcept {
  if (p == nullptr) return;
  if (g_countHeap.load(std::memory_order_relaxed)) g_deleteCalls.fetch_add(1);
  std::free(p);
}

#include "inference_tracker.hpp"
#include "memory_arena.hpp"
#include "topology_engine.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond, ...)                                              \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::printf("[FAIL] %s:%d: ", __FILE__, __LINE__);              \
      std::printf(__VA_ARGS__);                                       \
      std::printf("\n");                                              \
      ++g_failures;                                                   \
    } else {                                                          \
      std::printf("[PASS] ");                                         \
      std::printf(__VA_ARGS__);                                       \
      std::printf("\n");                                              \
    }                                                                 \
  } while (0)

inline void doNotOptimizeSink(volatile uint64_t v) {
#if defined(_MSC_VER)
  volatile uint64_t sink = v;
  (void)sink;
#else
  asm volatile("" : "+r"(v) : : "memory");
  (void)v;
#endif
}

constexpr int W = 1920;
constexpr int H = 1080;

void clearMask(uint8_t* m) { std::memset(m, 0, 2073600u); }

void fillEllipse(uint8_t* m, int cx, int cy, int rx, int ry, uint8_t v) {
  for (int y = cy - ry; y <= cy + ry; ++y) {
    if (y < 0 || y >= H) continue;
    const double t = 1.0 - (static_cast<double>(y - cy) * (y - cy)) /
                               (static_cast<double>(ry) * ry);
    if (t < 0.0) continue;
    const int dx = static_cast<int>(rx * std::sqrt(t));
    for (int x = cx - dx; x <= cx + dx; ++x) {
      if (x < 0 || x >= W) continue;
      m[static_cast<std::size_t>(y) * W + x] = v;
    }
  }
}

void fillCircle(uint8_t* m, int cx, int cy, int r, uint8_t v) {
  for (int y = cy - r; y <= cy + r; ++y) {
    if (y < 0 || y >= H) continue;
    for (int x = cx - r; x <= cx + r; ++x) {
      if (x < 0 || x >= W) continue;
      const int dx = x - cx, dy = y - cy;
      if (dx * dx + dy * dy <= r * r) m[static_cast<std::size_t>(y) * W + x] = v;
    }
  }
}

void fillRect(uint8_t* m, int x0, int y0, int x1, int y1, uint8_t v) {
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 >= W) x1 = W - 1;
  if (y1 >= H) y1 = H - 1;
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x) m[static_cast<std::size_t>(y) * W + x] = v;
}

void fillTriangle(uint8_t* m, int x0, int y0, int x1, int y1, int x2, int y2,
                 uint8_t v) {
  int xa = x0 < x1 ? (x0 < x2 ? x0 : x2) : (x1 < x2 ? x1 : x2);
  int xb = x0 > x1 ? (x0 > x2 ? x0 : x2) : (x1 > x2 ? x1 : x2);
  int ya = y0 < y1 ? (y0 < y2 ? y0 : y2) : (y1 < y2 ? y1 : y2);
  int yb = y0 > y1 ? (y0 > y2 ? y0 : y2) : (y1 > y2 ? y1 : y2);
  if (xa < 0) xa = 0;
  if (ya < 0) ya = 0;
  if (xb >= W) xb = W - 1;
  if (yb >= H) yb = H - 1;
  const int e0x = x1 - x0, e0y = y1 - y0;
  const int e1x = x2 - x1, e1y = y2 - y1;
  const int e2x = x0 - x2, e2y = y0 - y2;
  const int cx = (x0 + x1 + x2) / 3, cy = (y0 + y1 + y2) / 3;
  const int s0 = (cx - x0) * e0y - (cy - y0) * e0x;
  const int s1 = (cx - x1) * e1y - (cy - y1) * e1x;
  const int s2 = (cx - x2) * e2y - (cy - y2) * e2x;
  for (int y = ya; y <= yb; ++y) {
    for (int x = xa; x <= xb; ++x) {
      const int d0 = (x - x0) * e0y - (y - y0) * e0x;
      const int d1 = (x - x1) * e1y - (y - y1) * e1x;
      const int d2 = (x - x2) * e2y - (y - y2) * e2x;
      const bool in0 = (s0 >= 0) ? (d0 >= 0) : (d0 <= 0);
      const bool in1 = (s1 >= 0) ? (d1 >= 0) : (d1 <= 0);
      const bool in2 = (s2 >= 0) ? (d2 >= 0) : (d2 <= 0);
      if (in0 && in1 && in2) m[static_cast<std::size_t>(y) * W + x] = v;
    }
  }
}

double probL1(const double a[7], const double b[7]) {
  double s = 0.0;
  for (int i = 0; i < 7; ++i) {
    const double d = a[i] >= b[i] ? a[i] - b[i] : b[i] - a[i];
    s += d;
  }
  return s;
}

}  // namespace

int main() {
  std::printf("=== PanParasite-Edge Phase 5: inference+tracker test ===\n");

  static panparasite::StaticMemoryArena arena;
  CHECK(arena.isInitialized(), "arena initialized at boot");
  static panparasite::TopoResult topo;
  static panparasite::MicroNetAudit audit;
  static panparasite::TrackTable ttab;
  static panparasite::SlideDB sdb;

  // BOOT-ONLY init (uncounted).
  CHECK(panparasite::microNetInitBoot(arena), "micro-net boot init");
  std::printf("  [info] dnn ENGINE_NEW hook: %d\n",
              (int)panparasite::kDnnEngineNewAvailable);

  // ---- (a) Platt scaling ----
  {
    const double logits[7] = {2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0};
    double p[7] = {};
    panparasite::plattSoftmax(logits, p);
    double sum = 0.0;
    for (int i = 0; i < 7; ++i) sum += p[i];
    CHECK(sum > 1.0 - 1e-12 && sum < 1.0 + 1e-12, "platt sums to 1 (%.12f)",
          sum);
    bool mono = true;
    for (int i = 1; i < 7; ++i) {
      if (p[i] >= p[i - 1]) mono = false;
    }
    CHECK(mono && p[0] > p[6], "platt monotone in logits");
    CHECK(p[0] < 1.0 && p[0] > 1.0 / 7.0, "T=1.38 softens max (%.3f)", p[0]);
  }

  // ---- (b) FiLM modulation impact ----
  {
    std::memset(arena.scratchRoi(), 128, 16384u);
    const double egg[16] = {1, 0, 0, 3.66, 0.99, 0.02, 1.0, 0.0,
                            1600, 756, 0, 0, 0, 0, 0, 1.05};
    const double zero[16] = {};
    auto rEgg = panparasite::microNetClassify(arena, egg, audit);
    auto rZero = panparasite::microNetClassify(arena, zero, audit);
    CHECK(probL1(rEgg.p, rZero.p) > 0.05, "fbsc vs zeros changes probs (L1=%.3f)",
          probL1(rEgg.p, rZero.p));
    std::memset(arena.scratchRoi(), 64, 16384u);
    auto rDark = panparasite::microNetClassify(arena, egg, audit);
    std::memset(arena.scratchRoi(), 192, 16384u);
    auto rBright = panparasite::microNetClassify(arena, egg, audit);
    CHECK(probL1(rDark.p, rBright.p) > 1e-4, "dark vs bright ROI changes probs");
    std::printf("  [info] egg->cls%d (%.2f) zero->cls%d dark/br L1=%.4f\n",
                rEgg.argmax, rEgg.p[rEgg.argmax < 0 ? 0 : rEgg.argmax],
                rZero.argmax, probL1(rDark.p, rBright.p));
  }

  // ---- (c) Taxon routing via real topology ----
  {
    struct Case {
      const char* name;
      int expect;
    };
    // Shapes built inline per case below; expectations:
    // 0 ascaris, 1 trichuris, 2 hookworm, 3 mansoni, 4 haematobium, 5 debris.
    std::memset(arena.scratchRoi(), 128, 16384u);
    auto classifyFirst = [&](int& outCls, double& outConf) {
      const int n = panparasite::analyzeTopology(arena, topo);
      if (n != 1) return n;
      auto r = panparasite::microNetClassify(arena, topo.candidates[0].fbsc, audit);
      outCls = r.argmax;
      outConf = (r.argmax >= 0) ? r.p[r.argmax] : 0.0;
      return n;
    };
    int cls = -1;
    double conf = 0.0;
    int n = 0;
    clearMask(arena.mask());
    fillEllipse(arena.mask(), 960, 540, 80, 55, 255);
    n = classifyFirst(cls, conf);
    CHECK(n == 1 && cls == 0, "ascaris routes 0 (n=%d cls=%d %.2f)", n, cls, conf);
    clearMask(arena.mask());
    fillEllipse(arena.mask(), 960, 540, 80, 55, 255);
    fillCircle(arena.mask(), 935, 540, 12, 0);
    fillCircle(arena.mask(), 985, 540, 12, 0);
    n = classifyFirst(cls, conf);
    CHECK(n == 1 && cls == 1, "trichuris routes 1 (n=%d cls=%d %.2f)", n, cls, conf);
    clearMask(arena.mask());
    fillEllipse(arena.mask(), 960, 540, 40, 27, 255);
    n = classifyFirst(cls, conf);
    CHECK(n == 1 && cls == 2, "hookworm routes 2 (n=%d cls=%d %.2f)", n, cls, conf);
    clearMask(arena.mask());
    fillEllipse(arena.mask(), 960, 540, 80, 55, 255);
    fillTriangle(arena.mask(), 945, 495, 975, 495, 960, 440, 255);
    n = classifyFirst(cls, conf);
    CHECK(n == 1 && cls == 3, "mansoni routes 3 (n=%d cls=%d %.2f)", n, cls, conf);
    clearMask(arena.mask());
    fillEllipse(arena.mask(), 960, 540, 80, 55, 255);
    fillTriangle(arena.mask(), 1030, 525, 1030, 555, 1085, 540, 255);
    n = classifyFirst(cls, conf);
    CHECK(n == 1 && cls == 4, "haematobium routes 4 (n=%d cls=%d %.2f)", n, cls,
          conf);
    clearMask(arena.mask());
    fillRect(arena.mask(), 860, 465, 1059, 614, 255);
    fillRect(arena.mask(), 870, 475, 1049, 604, 0);
    n = classifyFirst(cls, conf);
    CHECK(n == 1 && cls == 5, "debris ring routes 5 (n=%d cls=%d %.2f)", n, cls,
          conf);
    const double unk[16] = {1, 1, 0.25, 4.5, 0.75, 0.1, 0.7, 0.7,
                            1500, 1200, 300, 500, -400, 300, -500, 1.3};
    auto rU = panparasite::microNetClassify(arena, unk, audit);
    CHECK(rU.argmax == 6, "unknown hand vector routes 6 (got %d)", rU.argmax);
    (void)sizeof(Case);
  }

  // ---- (d) INT8/FP32 agreement + fallback ----
  {
    std::memset(arena.scratchRoi(), 128, 16384u);
    const double egg[16] = {1, 0, 0, 3.66, 0.99, 0.02, 1.0, 0.0,
                            1600, 756, 0, 0, 0, 0, 0, 1.05};
    const int runs0 = audit.int8_runs;
    const int fb0 = audit.fp32_fallback_count;
    auto rInt8 = panparasite::microNetClassify(arena, egg, audit);
    CHECK(!rInt8.usedFp32Fallback, "clean INT8 path, no fallback");
    CHECK(audit.int8_runs == runs0 + 1 && audit.fp32_fallback_count == fb0,
          "audit counts INT8 run");
    panparasite::microNetSetForcedFallback(true);
    auto rFb = panparasite::microNetClassify(arena, egg, audit);
    panparasite::microNetSetForcedFallback(false);
    CHECK(rFb.usedFp32Fallback, "forced fault takes FP32 fallback");
    CHECK(audit.fp32_fallback_count == fb0 + 1, "fallback counter increments");
    double sum = 0.0;
    for (int i = 0; i < 7; ++i) sum += rFb.p[i];
    CHECK(sum > 1.0 - 1e-12 && sum < 1.0 + 1e-12, "fallback probs sum to 1");
    CHECK(rFb.argmax == rInt8.argmax, "fallback preserves argmax (%d)",
          rFb.argmax);
  }

  // ---- (e) Hungarian ----
  {
    // cost = 1 - IoU. tracks x dets.
    const double cost[4] = {0.32, 0.95, 0.90, 0.20};
    int assign[64];
    int fresh[8];
    int nFresh = 0;
    panparasite::hungarianMatch(cost, 2, 2, assign, fresh, &nFresh);
    CHECK(assign[0] == 0 && assign[1] == 1 && nFresh == 0,
          "2x2 optimal assignment (got %d,%d new=%d)", assign[0], assign[1],
          nFresh);
    const double gated[2] = {0.95, 0.96};  // both below IoU 0.30
    panparasite::hungarianMatch(gated, 1, 2, assign, fresh, &nFresh);
    CHECK(assign[0] == -1 && nFresh == 2, "gated pairs unmatched (new=%d)",
          nFresh);
    panparasite::hungarianMatch(nullptr, 0, 3, assign, fresh, &nFresh);
    CHECK(nFresh == 3 && fresh[0] == 0 && fresh[2] == 2, "empty tracks: all new");
    CHECK(panparasite::bboxIoU(0, 0, 10, 10, 1, 1, 11, 11) > 0.30, "IoU sane");
    CHECK(panparasite::bboxIoU(0, 0, 10, 10, 50, 50, 60, 60) == 0.0,
          "disjoint IoU=0");
  }

  // ---- (f) Tracker log-odds ----
  {
    for (int i = 0; i < 64; ++i) ttab.tracks[i] = panparasite::Track{};
    ttab.nextId = 0;
    panparasite::Detection d0{100.0f, 100.0f, 180.0f, 160.0f, 0, 0.9};
    panparasite::trackerStep(ttab, &d0, 1);
    CHECK(ttab.tracks[0].active && ttab.tracks[0].id == 0, "spawn id 0");
    CHECK(ttab.tracks[0].status == panparasite::kTrackVerified,
          "p=0.9 -> VERIFIED (L=%.4f)", ttab.tracks[0].logOdds);
    panparasite::Detection d1{500.0f, 500.0f, 560.0f, 550.0f, 5, 0.1};
    panparasite::trackerStep(ttab, &d1, 1);
    // Track 0 unmatched this field (far det): decays; track 1 spawned+rejected.
    CHECK(ttab.tracks[1].status == panparasite::kTrackRejected &&
              !ttab.tracks[1].active,
          "p=0.1 -> REJECTED+inactive (L=%.4f)", ttab.tracks[1].logOdds);
    // Decay check on a mid-confidence track.
    for (int i = 0; i < 64; ++i) ttab.tracks[i] = panparasite::Track{};
    ttab.nextId = 0;
    panparasite::Detection d2{100.0f, 100.0f, 180.0f, 160.0f, 0, 0.7};
    panparasite::trackerStep(ttab, &d2, 1);
    const double l0 = ttab.tracks[0].logOdds;  // ln(0.7/0.3) ~= 0.8473
    panparasite::trackerStep(ttab, nullptr, 0);
    CHECK(ttab.tracks[0].unseen == 1, "unseen increments");
    const double dl = ttab.tracks[0].logOdds - l0;
    CHECK(dl > -0.101 && dl < -0.099, "unobserved decay -0.10 (got %.4f)", dl);
    // IoU continuity: slight shift keeps id, hits grow.
    panparasite::Detection d3{104.0f, 102.0f, 184.0f, 162.0f, 0, 0.8};
    panparasite::trackerStep(ttab, &d3, 1);
    CHECK(ttab.tracks[0].hits == 2 && ttab.tracks[0].id == 0,
          "IoU match keeps id (hits=%d)", ttab.tracks[0].hits);
  }

  // ---- (g) Slide aggregation ----
  {
    for (int i = 0; i < 256; ++i) sdb.slots[i] = panparasite::SlideTrack{};
    for (int i = 0; i < 7; ++i) sdb.tally[i] = 0;
    panparasite::slideCommit(sdb, ttab);
    // ttab currently: track0 active (L~0.847-0.1+ln(0.8/0.2)=~2.03? verify below)
    std::printf("  [info] track0 L=%.3f status=%d\n", ttab.tracks[0].logOdds,
                (int)ttab.tracks[0].status);
    // Force a verified commit path deterministically:
    for (int i = 0; i < 64; ++i) ttab.tracks[i] = panparasite::Track{};
    ttab.nextId = 0;
    panparasite::Detection dv{100.0f, 100.0f, 180.0f, 160.0f, 3, 0.9};
    panparasite::trackerStep(ttab, &dv, 1);
    panparasite::slideCommit(sdb, ttab);
    CHECK(sdb.tally[3] == 1, "verified taxon tallied once");
    panparasite::slideCommit(sdb, ttab);
    CHECK(sdb.tally[3] == 1, "re-commit dedups by id");
    // Rejected debris never tallies.
    panparasite::Detection dd{400.0f, 400.0f, 450.0f, 440.0f, 5, 0.05};
    panparasite::trackerStep(ttab, &dd, 1);
    panparasite::slideCommit(sdb, ttab);
    CHECK(sdb.tally[5] == 0, "rejected debris not tallied");
  }

  // ---- (h) Zero-heap hot region ----
  {
    std::memset(arena.scratchRoi(), 128, 16384u);
    const double egg[16] = {1, 0, 0, 3.66, 0.99, 0.02, 1.0, 0.0,
                            1600, 756, 0, 0, 0, 0, 0, 1.05};
    g_newCalls.store(0);
    g_deleteCalls.store(0);
    g_countHeap.store(true);
    volatile uint64_t sink = 0;
    {
      auto r = panparasite::microNetClassify(arena, egg, audit);
      sink += static_cast<uint64_t>(r.argmax + 1);
      for (int i = 0; i < 7; ++i) sink += static_cast<uint64_t>(r.p[i] * 1000.0);
      double lg[7] = {0.5, 0.2, -0.1, -0.4, -0.2, -1.0, -0.5};
      double pp[7] = {};
      panparasite::plattSoftmax(lg, pp);
      sink += static_cast<uint64_t>(pp[0] * 1000.0);
      panparasite::Detection dd{10.0f, 10.0f, 50.0f, 40.0f, 0, 0.8};
      panparasite::trackerStep(ttab, &dd, 1);
      sink += static_cast<uint64_t>(ttab.tracks[0].hits);
      panparasite::slideCommit(sdb, ttab);
      sink += static_cast<uint64_t>(sdb.tally[0]);
      const double c2[4] = {0.1, 0.9, 0.8, 0.2};
      int as[64];
      int fr[8];
      int nf = 0;
      panparasite::hungarianMatch(c2, 2, 2, as, fr, &nf);
      sink += static_cast<uint64_t>(as[0] + 2 * as[1] + 4 * nf);
    }
    doNotOptimizeSink(sink);
    g_countHeap.store(false);
    CHECK(g_newCalls.load() == 0, "zero heap new in Stage 5+6 path (got %lld)",
          g_newCalls.load());
    CHECK(g_deleteCalls.load() == 0, "zero heap delete in Stage 5+6 path (got %lld)",
          g_deleteCalls.load());
  }

  if (g_failures == 0) {
    std::printf("=== ALL INFERENCE+TRACKER TESTS PASSED ===\n");
    return 0;
  }
  std::printf("=== %d INFERENCE+TRACKER TEST(S) FAILED ===\n", g_failures);
  return 1;
}
