// PanParasite-Edge Phase 4 unit test: gate + occlusion + diffusion.
// Covers:
//   (a) Calibration pinning: hand egg vector scores high, debris ~0.
//   (b) End-to-end gate on 3 eggs: all pass, sorted desc, no cap.
//   (c) Top-K cap: 10 viable eggs -> n==8, CANDIDATE_CAP_EXCEEDED set.
//   (d) Debris rejection: ring (R~0.78) + 4-hole ellipse rejected, egg kept.
//   (e) Occlusion: clean egg O low; star (plus) candidate O > 0.65 direct.
//   (f) ROI diffusion: flat-noise variance down, step edge preserved.
//   (g) No-trigger path leaves scratch untouched.
//   (h) Zero heap new/delete across the hot path (tripwire).

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

#include "active_perception.hpp"
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

// Mean/variance over a scratch rect (test-side stats, no heap).
void rectStats(const uint8_t* img, int stride, int x0, int y0, int x1, int y1,
               double& mean, double& var) {
  double s = 0.0, s2 = 0.0;
  int64_t n = 0;
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x) {
      const double v = img[static_cast<std::size_t>(y) * stride + x];
      s += v;
      s2 += v * v;
      ++n;
    }
  mean = s / n;
  var = s2 / n - mean * mean;
}

}  // namespace

int main() {
  std::printf("=== PanParasite-Edge Phase 4: active perception test ===\n");

  static panparasite::StaticMemoryArena arena;
  CHECK(arena.isInitialized(), "arena initialized at boot");
  static panparasite::TopoResult topo;
  CHECK(std::strcmp(panparasite::kCapAuditFlag, "CANDIDATE_CAP_EXCEEDED") == 0,
        "audit flag name CANDIDATE_CAP_EXCEEDED");

  // ---- (a) Calibration pinning with hand vectors ----
  {
    // Egg-like: Beta0=1, Beta1=0, R=0, smooth hull, moderate moments.
    const double egg[16] = {1, 0, 0, 3.66, 0.99, 0.02, 0.9, 0.44,
                            1600, 756, 0, 0, 0, 0, 0, 1.05};
    // Chaotic debris: Beta1=4, R=0.72, ragged hull, wild moments.
    const double deb[16] = {1, 4, 0.72, 6.5, 0.45, 0.55, 0.3, 0.95,
                            3200, 2900, 800, 3000, -2500, 1800, -3200, 2.3};
    const double sEgg = panparasite::evalLogisticScore(egg);
    const double sDeb = panparasite::evalLogisticScore(deb);
    CHECK(sEgg > 0.60, "hand egg score %.3f > 0.60", sEgg);
    CHECK(sDeb < 0.10, "hand debris score %.4f < 0.10", sDeb);
    CHECK(sEgg >= panparasite::kGateTau && sDeb < panparasite::kGateTau,
          "tau=0.38 separates (%.3f vs %.4f)", sEgg, sDeb);
  }

  // ---- (b) End-to-end gate on 3 eggs ----
  {
    clearMask(arena.mask());
    fillEllipse(arena.mask(), 480, 300, 60, 42, 255);
    fillEllipse(arena.mask(), 960, 540, 60, 42, 255);
    fillEllipse(arena.mask(), 1440, 800, 60, 42, 255);
    const int n = panparasite::analyzeTopology(arena, topo);
    CHECK(n == 3, "3-egg field: 3 candidates (got %d)", n);
    panparasite::GatedField gated;
    panparasite::runLogisticGate(topo, gated);
    CHECK(gated.n == 3 && !gated.capExceeded, "gate: 3 pass, no cap (n=%d)",
          gated.n);
    bool sorted = true;
    for (int i = 1; i < gated.n; ++i)
      if (gated.scores[i] > gated.scores[i - 1]) sorted = false;
    CHECK(sorted, "gate: scores sorted desc (%.3f,%.3f,%.3f)", gated.scores[0],
          gated.scores[1], gated.scores[2]);
    for (int i = 0; i < gated.n; ++i)
      CHECK(gated.scores[i] >= panparasite::kGateTau, "egg %d score %.3f >= tau",
            gated.idx[i], gated.scores[i]);
  }

  // ---- (c) Top-K cap with 10 viable eggs ----
  {
    clearMask(arena.mask());
    for (int r = 0; r < 2; ++r)
      for (int c = 0; c < 5; ++c)
        fillEllipse(arena.mask(), 240 + c * 360, 270 + r * 540, 60, 42, 255);
    const int n = panparasite::analyzeTopology(arena, topo);
    CHECK(n == 10, "10-egg field: 10 candidates (got %d)", n);
    panparasite::GatedField gated;
    panparasite::runLogisticGate(topo, gated);
    CHECK(gated.n == 8 && gated.capExceeded,
          "gate: capped at 8 + flag (n=%d flag=%d)", gated.n, gated.capExceeded);
    bool sorted = true;
    for (int i = 1; i < gated.n; ++i)
      if (gated.scores[i] > gated.scores[i - 1]) sorted = false;
    CHECK(sorted, "cap: top-8 sorted desc");
  }

  // ---- (d) Debris rejection end-to-end ----
  {
    clearMask(arena.mask());
    fillEllipse(arena.mask(), 1500, 800, 80, 55, 255);  // valid egg
    fillRect(arena.mask(), 200, 200, 399, 349, 255);    // ring outer 200x150
    fillRect(arena.mask(), 210, 210, 389, 339, 0);      // ring hole 180x130
    fillEllipse(arena.mask(), 500, 750, 80, 55, 255);   // holed body
    fillCircle(arena.mask(), 480, 750, 10, 0);
    fillCircle(arena.mask(), 500, 735, 10, 0);
    fillCircle(arena.mask(), 520, 750, 10, 0);
    fillCircle(arena.mask(), 500, 765, 10, 0);
    const int n = panparasite::analyzeTopology(arena, topo);
    CHECK(n == 3, "debris field: 3 candidates (got %d)", n);
    panparasite::GatedField gated;
    panparasite::runLogisticGate(topo, gated);
    CHECK(gated.n == 1, "gate: only egg survives (n=%d)", gated.n);
    if (gated.n == 1) {
      const auto& c = topo.candidates[gated.idx[0]];
      CHECK(c.xbar > 1400.0 && c.ybar > 700.0, "survivor is the egg (xb=%.0f yb=%.0f)",
            c.xbar, c.ybar);
    }
    for (int i = 0; i < topo.numCandidates; ++i) {
      const double s =
          panparasite::evalLogisticScore(topo.candidates[i].fbsc);
      const bool isEgg =
          (topo.candidates[i].xbar > 1400.0 && topo.candidates[i].ybar > 700.0);
      std::printf("  [info] cand x=%.0f y=%.0f B1=%d R=%.3f score=%.3f %s\n",
                  topo.candidates[i].xbar, topo.candidates[i].ybar,
                  topo.candidates[i].beta1, topo.candidates[i].rHole, s,
                  isEgg ? "EGG" : "debris");
      CHECK(isEgg ? (s >= 0.38) : (s < 0.38), "cand %d score %.3f %s tau", i, s,
            isEgg ? "passes" : "rejected");
    }
  }

  // ---- (e) Occlusion: clean vs star ----
  double oClean = 0.0, oStar = 0.0;
  {
    clearMask(arena.mask());
    fillEllipse(arena.mask(), 960, 540, 80, 55, 255);
    panparasite::analyzeTopology(arena, topo);
    CHECK(topo.numCandidates == 1, "clean: 1 candidate");
    oClean = panparasite::computeOcclusion(topo.candidates[0], 0, arena);
    CHECK(oClean < 0.30, "clean O=%.4f < 0.30", oClean);
    CHECK(!panparasite::needsActivePerception(oClean), "clean: no trigger");

    clearMask(arena.mask());  // sparse plus: solidity ~0.2 -> O high
    fillRect(arena.mask(), 760, 528, 1160, 552, 255);
    fillRect(arena.mask(), 948, 340, 972, 740, 255);
    const int n = panparasite::analyzeTopology(arena, topo);
    CHECK(n == 1, "star: 1 candidate (got %d)", n);
    oStar = panparasite::computeOcclusion(topo.candidates[0], 0, arena);
    CHECK(oStar > 0.65, "star O=%.4f > 0.65", oStar);
    CHECK(panparasite::needsActivePerception(oStar), "star: trigger fires");
    CHECK(oStar > oClean + 0.30, "O_star - O_clean > 0.30 (%.3f vs %.3f)", oStar,
          oClean);
    std::printf("  [info] O_clean=%.4f O_star=%.4f\n", oClean, oStar);
  }

  // ---- (f) ROI diffusion: denoise + edge preservation ----
  {
    // frameB = vertical step (100|200 at x=960) + deterministic +/-4 noise.
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        const int base = (x < 960) ? 100 : 200;
        const int nz = ((x * 37 + y * 91) % 9) - 4;
        arena.frameB()[static_cast<std::size_t>(y) * W + x] =
            static_cast<uint8_t>(base + nz);
      }
    clearMask(arena.mask());
    fillEllipse(arena.mask(), 960, 540, 80, 55, 255);
    panparasite::analyzeTopology(arena, topo);
    panparasite::BBox box;
    CHECK(panparasite::candidateBBox(1u, arena.equivMap(), W, H, box),
          "egg bbox found");
    panparasite::extractCandidateROI(arena, box);
    double mL0, vL0, mR0, vR0;
    rectStats(arena.scratchRoi(), 128, 0, 0, 60, 127, mL0, vL0);
    rectStats(arena.scratchRoi(), 128, 67, 0, 127, 127, mR0, vR0);
    const double edge0 = mR0 - mL0;
    // Save pre-diffusion ROI in frameA head (test-only save, 16 KB).
    std::memcpy(arena.frameA(), arena.scratchRoi(), 16384u);
    panparasite::diffuseScratchROI(arena, 3);
    double mL1, vL1, mR1, vR1;
    rectStats(arena.scratchRoi(), 128, 0, 0, 60, 127, mL1, vL1);
    rectStats(arena.scratchRoi(), 128, 67, 0, 127, 127, mR1, vR1);
    const double edge1 = mR1 - mL1;
    CHECK(std::memcmp(arena.frameA(), arena.scratchRoi(), 16384u) != 0,
          "diffusion modifies ROI");
    CHECK(vL1 < 0.75 * vL0, "flat noise smoothed %.2f -> %.2f", vL0, vL1);
    CHECK(edge1 > 0.70 * edge0 && edge0 > 80.0,
          "step edge preserved %.1f -> %.1f", edge0, edge1);
    std::printf("  [info] var %.2f->%.2f edge %.1f->%.1f\n", vL0, vL1, edge0,
                edge1);
  }

  // ---- (g) No-trigger path leaves scratch untouched ----
  {
    clearMask(arena.mask());
    fillEllipse(arena.mask(), 960, 540, 80, 55, 255);
    for (int y = 0; y < H; ++y)  // flat frameB: diffusion would be no-op-ish
      for (int x = 0; x < W; ++x)
        arena.frameB()[static_cast<std::size_t>(y) * W + x] = 128;
    panparasite::analyzeTopology(arena, topo);
    panparasite::GatedField gated;
    panparasite::runLogisticGate(topo, gated);
    std::memset(arena.scratchRoi(), 0xA5, 16384u);
    panparasite::PerceptionRecord recs[8];
    const int nRec = panparasite::runActivePerception(arena, topo, gated, recs, 8);
    CHECK(nRec == gated.n, "records per gated candidate (%d)", nRec);
    bool anyTrig = false;
    for (int i = 0; i < nRec; ++i) anyTrig = anyTrig || recs[i].triggered;
    CHECK(!anyTrig, "clean field: no triggers");
    bool intact = true;
    for (int i = 0; i < 16384; ++i)
      if (arena.scratchRoi()[i] != 0xA5u) {
        intact = false;
        break;
      }
    CHECK(intact, "scratch untouched without trigger");
  }

  // ---- (h) Zero-heap hot region ----
  {
    clearMask(arena.mask());
    fillEllipse(arena.mask(), 960, 540, 80, 55, 255);
    panparasite::analyzeTopology(arena, topo);  // setup, uncounted
    g_newCalls.store(0);
    g_deleteCalls.store(0);
    g_countHeap.store(true);
    volatile uint64_t sink = 0;
    {
      panparasite::GatedField gated;
      panparasite::runLogisticGate(topo, gated);
      sink += static_cast<uint64_t>(gated.n);
      for (int i = 0; i < gated.n; ++i)
        sink += static_cast<uint64_t>(gated.scores[i] * 1000.0);
      for (int c = 0; c < topo.numCandidates; ++c) {
        const double o = panparasite::computeOcclusion(topo.candidates[c], c, arena);
        sink += static_cast<uint64_t>(o * 1000.0);
      }
      panparasite::BBox box;
      if (panparasite::candidateBBox(1u, arena.equivMap(), W, H, box)) {
        panparasite::extractCandidateROI(arena, box);
        panparasite::diffuseScratchROI(arena, 3);
        sink += arena.scratchRoi()[0] + arena.scratchRoi()[16383];
      }
      panparasite::PerceptionRecord recs[8];
      sink += static_cast<uint64_t>(
          panparasite::runActivePerception(arena, topo, gated, recs, 8));
    }
    doNotOptimizeSink(sink);
    g_countHeap.store(false);
    CHECK(g_newCalls.load() == 0, "zero heap new in Stage 4 hot path (got %lld)",
          g_newCalls.load());
    CHECK(g_deleteCalls.load() == 0, "zero heap delete in Stage 4 hot path (got %lld)",
          g_deleteCalls.load());
  }

  if (g_failures == 0) {
    std::printf("=== ALL ACTIVE PERCEPTION TESTS PASSED ===\n");
    return 0;
  }
  std::printf("=== %d ACTIVE PERCEPTION TEST(S) FAILED ===\n", g_failures);
  return 1;
}
