// PanParasite-Edge Phase 3 unit test: Stage 3 topology engine.
// Covers:
//   (a) DSU find/union sanity (path compression + rank).
//   (b) Ascaris ellipse: Beta_1 = 0, R_hole = 0, small d_max, f_BSC ranges.
//   (c) Trichuris (ellipse + 2 holes): Beta_1 = 2, chi = -1, R_hole > 0.
//   (d) S. mansoni (lateral spine): d_max >= 19.4 px, theta in [65,90].
//   (e) S. haematobium (terminal spine): d_max >= 22.6 px, theta in [0,20].
//   (f) Area filter: 314 px dot + 160K block excluded, valid egg kept.
//   (g) ZERO heap new/delete across analyzeTopology (tripwire).
//
// Synthetic eggs (all deterministic, no heap, no rand):
//   body: filled ellipse c=(960,540) rx=80 ry=55 (area ~= 13823 px).
//   mansoni spine: triangle (945,495),(975,495),(960,440) — lateral.
//   haematobium spine: triangle (1030,525),(1030,555),(1085,540) — terminal.

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

// Half-space triangle fill (integer, no heap). Winding-independent.
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
  // Reference sign at centroid (strictly inside for non-degenerate).
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

void buildAscaris(uint8_t* m) {
  clearMask(m);
  fillEllipse(m, 960, 540, 80, 55, 255);
}
void buildTrichuris(uint8_t* m) {
  buildAscaris(m);
  fillCircle(m, 935, 540, 12, 0);
  fillCircle(m, 985, 540, 12, 0);
}
void buildMansoni(uint8_t* m) {
  buildAscaris(m);
  fillTriangle(m, 945, 495, 975, 495, 960, 440, 255);
}
void buildHaematobium(uint8_t* m) {
  buildAscaris(m);
  fillTriangle(m, 1030, 525, 1030, 555, 1085, 540, 255);
}

}  // namespace

int main() {
  std::printf("=== PanParasite-Edge Phase 3: topology test ===\n");

  static panparasite::StaticMemoryArena arena;
  CHECK(arena.isInitialized(), "arena initialized at boot");
  // TopoResult ~25 KB: function-static (BSS), never stack, never heap.
  static panparasite::TopoResult result;

  // ---- (a) DSU sanity ----
  {
    arena.resetDsuParents();
    auto* dsu = arena.dsu();
    for (uint32_t i = 1; i <= 5; ++i) {
      dsu[i].parent = i;
      dsu[i].rank = 0;
      dsu[i].size = 1;
    }
    CHECK(panparasite::unionLabels(dsu, 1, 2) != 0, "union(1,2) ok");
    panparasite::unionLabels(dsu, 3, 4);
    panparasite::unionLabels(dsu, 2, 3);
    const uint32_t r1 = panparasite::findRoot(dsu, 1);
    CHECK(r1 == panparasite::findRoot(dsu, 4), "1..4 share root after unions");
    CHECK(dsu[r1].size == 4, "root size=4 (got %u)", dsu[r1].size);
    CHECK(panparasite::findRoot(dsu, 5) == 5, "5 stays singleton");
  }

  // ---- (b) Ascaris: solid ellipse ----
  {
    buildAscaris(arena.mask());
    const int n = panparasite::analyzeTopology(arena, result);
    CHECK(n == 1 && result.numCandidates == 1, "ascaris: 1 candidate (got %d)", n);
    CHECK(result.globalBeta0 == 1 && result.globalBeta1 == 0 && result.globalChi == 1,
          "ascaris: global (1,0,1) (got %d,%d,%d)", result.globalBeta0,
          result.globalBeta1, result.globalChi);
    const auto& c = result.candidates[0];
    CHECK(c.beta0 == 1 && c.beta1 == 0 && c.chi == 1, "ascaris: per-cand (1,0,1)");
    CHECK(c.rHole == 0.0 && c.holeArea == 0u, "ascaris: no holes");
    CHECK(c.areaExt >= 1000u && c.areaExt <= 35000u, "ascaris: area %u in bounds",
          c.areaExt);
    CHECK(c.dmaxPx < 10.0, "ascaris: near-convex dmax=%.2fpx", c.dmaxPx);
    CHECK(!c.isMansoniLike() && !c.isHaematobiumLike() && !c.isTrichurisLike(),
          "ascaris: no species-spine trigger");
    CHECK(c.fbsc[0] == 1.0 && c.fbsc[1] == 0.0 && c.fbsc[2] == 0.0,
          "ascaris: fbsc[0..2]=(1,0,0)");
    CHECK(c.fbsc[4] > 0.5 && c.fbsc[4] <= 1.0, "ascaris: Aext/Ahull=%.3f",
          c.fbsc[4]);
    CHECK(c.fbsc[15] >= 1.0 && c.fbsc[15] < 3.0, "ascaris: Pext/Phull=%.3f",
          c.fbsc[15]);
    CHECK(c.fbsc[3] > 2.5 && c.fbsc[3] < 6.0, "ascaris: P/sqrtA=%.3f", c.fbsc[3]);
    const double unit = c.fbsc[6] * c.fbsc[6] + c.fbsc[7] * c.fbsc[7];
    CHECK(unit > 0.999 && unit < 1.001, "ascaris: cos^2+sin^2=1 (%.5f)", unit);
    std::printf("  [info] ascaris area=%u dmax=%.2fpx theta=%.1fdeg\n", c.areaExt,
                c.dmaxPx, c.thetaFoldedDeg);
  }

  // ---- (c) Trichuris: two vacuoles -> Beta_1 = 2 ----
  {
    buildTrichuris(arena.mask());
    const int n = panparasite::analyzeTopology(arena, result);
    CHECK(n == 1, "trichuris: 1 candidate (got %d)", n);
    const auto& c = result.candidates[0];
    CHECK(c.beta1 == 2, "trichuris: Beta_1=2 (got %d)", c.beta1);
    CHECK(c.chi == -1, "trichuris: chi=-1 (got %d)", c.chi);
    CHECK(c.rHole > 0.03 && c.rHole < 0.12, "trichuris: R_hole=%.4f", c.rHole);
    CHECK(c.isTrichurisLike(), "trichuris: isTrichurisLike");
    CHECK(result.globalBeta1 == 2 && result.globalChi == -1,
          "trichuris: global (B1=2,chi=-1)");
    CHECK(c.fbsc[1] == 2.0, "trichuris: fbsc[1]=2");
    std::printf("  [info] trichuris holes=%d Rhole=%.4f\n", c.beta1, c.rHole);
  }

  // ---- (d) S. mansoni: lateral spine ----
  {
    buildMansoni(arena.mask());
    const int n = panparasite::analyzeTopology(arena, result);
    CHECK(n == 1, "mansoni: 1 candidate (got %d)", n);
    const auto& c = result.candidates[0];
    CHECK(c.beta1 == 0, "mansoni: Beta_1=0 (got %d)", c.beta1);
    CHECK(c.dmaxPx >= 19.4, "mansoni: dmax>=19.4px (got %.2fpx=%.2fum)", c.dmaxPx,
          c.dmaxUm);
    CHECK(c.thetaFoldedDeg >= 65.0 && c.thetaFoldedDeg <= 90.0,
          "mansoni: theta in [65,90] (got %.1f)", c.thetaFoldedDeg);
    CHECK(c.isMansoniLike(), "mansoni: isMansoniLike");
    CHECK(!c.isHaematobiumLike(), "mansoni: not haematobium-like");
    std::printf("  [info] mansoni dmax=%.2fpx theta=%.1fdeg\n", c.dmaxPx,
                c.thetaFoldedDeg);
  }

  // ---- (e) S. haematobium: terminal spine ----
  {
    buildHaematobium(arena.mask());
    const int n = panparasite::analyzeTopology(arena, result);
    CHECK(n == 1, "haematobium: 1 candidate (got %d)", n);
    const auto& c = result.candidates[0];
    CHECK(c.beta1 == 0, "haematobium: Beta_1=0 (got %d)", c.beta1);
    CHECK(c.dmaxPx >= 22.6, "haematobium: dmax>=22.6px (got %.2fpx=%.2fum)",
          c.dmaxPx, c.dmaxUm);
    CHECK(c.thetaFoldedDeg >= 0.0 && c.thetaFoldedDeg <= 20.0,
          "haematobium: theta in [0,20] (got %.1f)", c.thetaFoldedDeg);
    CHECK(c.isHaematobiumLike(), "haematobium: isHaematobiumLike");
    CHECK(!c.isMansoniLike(), "haematobium: not mansoni-like");
    std::printf("  [info] haematobium dmax=%.2fpx theta=%.1fdeg\n", c.dmaxPx,
                c.thetaFoldedDeg);
  }

  // ---- (f) Area filter ----
  {
    buildAscaris(arena.mask());
    fillCircle(arena.mask(), 100, 900, 10, 255);  // ~314 px < 1000: debris
    for (int y = 0; y < 400; ++y)                 // 160K px > 35000: sheet
      for (int x = 0; x < 400; ++x)
        arena.mask()[static_cast<std::size_t>(y) * W + x] = 255;
    const int n = panparasite::analyzeTopology(arena, result);
    CHECK(n == 1, "area-filter: only egg survives (got %d)", n);
    if (n == 1) {
      CHECK(result.candidates[0].areaExt >= 1000u &&
                result.candidates[0].areaExt <= 35000u,
            "area-filter: survivor in bounds (%u)", result.candidates[0].areaExt);
    }
  }

  // ---- (g) Zero-heap hot region ----
  buildMansoni(arena.mask());
  g_newCalls.store(0);
  g_deleteCalls.store(0);
  g_countHeap.store(true);
  {
    volatile uint64_t sink = 0;
    const int n = panparasite::analyzeTopology(arena, result);
    sink += static_cast<uint64_t>(n);
    for (int c = 0; c < n && c < 4; ++c) {
      sink += static_cast<uint64_t>(result.candidates[c].areaExt);
      sink += static_cast<uint64_t>(result.candidates[c].beta1 * 7u);
      sink += static_cast<uint64_t>(result.candidates[c].dmaxPx * 100.0);
    }
    doNotOptimizeSink(sink);
  }
  g_countHeap.store(false);
  CHECK(g_newCalls.load() == 0, "zero heap new in Stage 3 hot path (got %lld)",
        g_newCalls.load());
  CHECK(g_deleteCalls.load() == 0, "zero heap delete in Stage 3 hot path (got %lld)",
        g_deleteCalls.load());
  CHECK(!result.overflow, "no overflow flags on synthetic shapes");

  if (g_failures == 0) {
    std::printf("=== ALL TOPOLOGY TESTS PASSED ===\n");
    return 0;
  }
  std::printf("=== %d TOPOLOGY TEST(S) FAILED ===\n", g_failures);
  return 1;
}
