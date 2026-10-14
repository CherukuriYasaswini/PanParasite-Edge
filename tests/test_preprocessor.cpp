// PanParasite-Edge Phase 2 unit test: Stage 1 + Stage 2 preprocessor.
// Covers:
//   (a) Flicker helper T_int = k/(2*f): 50 Hz/20.0 ms/k=2, 60 Hz/33.33 ms/k=4.
//   (b) Acquisition gate: Delta I_t < 1.80, Phi_focus >= 125.0, 3-frame lock.
//   (c) Integral image: small brute-force + full-frame spot checks.
//   (d) Adaptive binarization R_w=31/C=0.88: dark-square fg, uniform bg.
//   (e) 11x11 elliptical closing: hole-fill + gap-bridge on stack images.
//   (f) Dual-path equivalence: Full-Frame vs 32 px Tiled bit-identical.
//   (g) ZERO heap new/delete across hot Stage 1+2 calls (tripwire).
//
// Tripwire: same global new/delete counters as test_arena. Boot + setup
// run with counting OFF; the hot region runs with counting ON and performs
// no printing/STL/vector/Mat::create.

#include <atomic>
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
#include "preprocessor.hpp"

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

// FNV-1a 64 over bytes. No heap.
uint64_t fnv1a(const uint8_t* p, std::size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (std::size_t i = 0; i < n; ++i) {
    h ^= static_cast<uint64_t>(p[i]);
    h *= 1099511628211ull;
  }
  return h;
}

inline bool approxEq(double a, double b, double tol) {
  const double d = a >= b ? a - b : b - a;
  return d <= tol;
}

}  // namespace

int main() {
  std::printf("=== PanParasite-Edge Phase 2: preprocessor test ===\n");

  static panparasite::StaticMemoryArena arena;
  CHECK(arena.isInitialized(), "arena initialized at boot");
  CHECK(panparasite::kEllipse11x11.count >= 70 &&
            panparasite::kEllipse11x11.count <= 121,
        "ellipse kernel taps=%d in [70,121]", panparasite::kEllipse11x11.count);
  std::printf("  [info] ellipse taps=%d tile=%dx%d overlap=%d\n",
              panparasite::kEllipse11x11.count, panparasite::kTileWidth,
              panparasite::kTileHeight, panparasite::kTileOverlap);

  // ---- (a) Flicker helper ----
  CHECK(approxEq(panparasite::flickerTintMs(50.0, 2), 20.0, 1e-9),
        "50Hz k=2 -> 20.0ms (got %.6f)", panparasite::flickerTintMs(50.0, 2));
  CHECK(approxEq(panparasite::flickerTintMs(60.0, 4), 100.0 / 3.0, 1e-6),
        "60Hz k=4 -> 33.33ms (got %.6f)", panparasite::flickerTintMs(60.0, 4));
  {
    const auto r50 = panparasite::recommendedFlickerExposure(50.0);
    const auto r60 = panparasite::recommendedFlickerExposure(60.0);
    CHECK(r50.k == 2 && approxEq(r50.t_int_ms, 20.0, 1e-9), "rec 50Hz k=2 20ms");
    CHECK(r60.k == 4 && approxEq(r60.t_int_ms, 100.0 / 3.0, 1e-6),
          "rec 60Hz k=4 33.33ms");
    CHECK(panparasite::isFlickerCompliant(20.0, 50.0), "20ms compliant at 50Hz");
    CHECK(!panparasite::isFlickerCompliant(15.0, 50.0), "15ms rejected at 50Hz");
    CHECK(panparasite::isFlickerCompliant(100.0 / 3.0, 60.0),
          "33.33ms compliant at 60Hz");
  }

  // ---- (b) Delta ----
  {
    std::memset(arena.frameA(), 100, panparasite::kFramePixels);
    std::memset(arena.frameB(), 100, panparasite::kFramePixels);
    const double d0 = panparasite::meanInterFrameDelta(arena);
    CHECK(approxEq(d0, 0.0, 1e-9), "identical frames delta=0 (got %.4f)", d0);

    std::memset(arena.frameB(), 110, panparasite::kFramePixels);
    const double d10 = panparasite::meanInterFrameDelta(arena);
    CHECK(approxEq(d10, 10.0, 1e-9) && d10 >= panparasite::kDeltaIThreshold,
          "offset+10 delta=10 fails gate (got %.4f)", d10);

    std::memset(arena.frameB(), 101, panparasite::kFramePixels);
    const double d1 = panparasite::meanInterFrameDelta(arena);
    CHECK(approxEq(d1, 1.0, 1e-9) && d1 < panparasite::kDeltaIThreshold,
          "offset+1 delta=1 passes gate (got %.4f)", d1);
  }

  // ---- (b) Focus: uniform vs checkerboard ----
  {
    std::memset(arena.frameB(), 128, panparasite::kFramePixels);
    const double vFlat = panparasite::laplacianFocusVariance(arena);
    CHECK(vFlat < panparasite::kFocusPhiThreshold,
          "flat frame focus %.2f fails gate (<125)", vFlat);

    uint8_t* fb = arena.frameB();
    for (int y = 0; y < panparasite::kFrameHeight; ++y) {
      for (int x = 0; x < panparasite::kFrameWidth; ++x) {
        fb[static_cast<std::size_t>(y) * 1920u + static_cast<std::size_t>(x)] =
            static_cast<uint8_t>(((x + y) & 1) ? 255u : 0u);
      }
    }
    const double vCheck = panparasite::laplacianFocusVariance(arena);
    CHECK(vCheck >= panparasite::kFocusPhiThreshold,
          "checkerboard focus %.1f passes gate (>=125)", vCheck);

    // Small-kernel sanity on stack (no heap).
    uint8_t flat8[64];
    uint8_t check8[64];
    std::memset(flat8, 128, sizeof(flat8));
    for (int i = 0; i < 64; ++i) check8[i] = static_cast<uint8_t>((i & 1) ? 255u : 0u);
    CHECK(panparasite::laplacianFocusVariance(flat8, 8, 8) < 125.0,
          "8x8 flat focus fails");
    CHECK(panparasite::laplacianFocusVariance(check8, 8, 8) >= 125.0,
          "8x8 checker focus passes");
  }

  // ---- (b) 3-frame lock ----
  {
    panparasite::AcquisitionGate gate;
    // Passing pair: identical checkerboards (delta 0, focus high).
    uint8_t* fa = arena.frameA();
    uint8_t* fb = arena.frameB();
    for (std::size_t i = 0; i < panparasite::kFramePixels; ++i) {
      const int x = static_cast<int>(i % 1920u);
      const int y = static_cast<int>(i / 1920u);
      const uint8_t v = static_cast<uint8_t>(((x + y) & 1) ? 255u : 0u);
      fa[i] = v;
      fb[i] = v;
    }
    for (int f = 0; f < 2; ++f) {
      const auto m = panparasite::checkAcquisitionGate(arena, gate);
      CHECK(m.pair_pass && !gate.locked && gate.consecutive_passes == f + 1,
            "pass %d/3: locked=0 count=%d", f + 1, gate.consecutive_passes);
    }
    {
      const auto m = panparasite::checkAcquisitionGate(arena, gate);
      CHECK(m.pair_pass && gate.locked && gate.consecutive_passes == 3,
            "pass 3/3: locked=1 count=3");
    }
    // Failing pair: flat frames reset the counter.
    std::memset(arena.frameA(), 128, panparasite::kFramePixels);
    std::memset(arena.frameB(), 128, panparasite::kFramePixels);
    {
      const auto m = panparasite::checkAcquisitionGate(arena, gate);
      CHECK(!m.pair_pass && !gate.locked && gate.consecutive_passes == 0,
            "flat pair resets lock (delta=%.2f phi=%.1f)", m.delta_it, m.phi_focus);
    }
    gate.reset();
    CHECK(gate.consecutive_passes == 0 && !gate.locked, "gate.reset() clears");
  }

  // ---- (c) Integral: small brute-force ----
  {
    uint8_t s[64];
    uint32_t ii[81];
    for (int i = 0; i < 64; ++i) s[i] = 1;
    panparasite::computeIntegralImage(s, ii, 8, 8);
    const int stride = 9;
    CHECK(ii[0] == 0u && ii[8] == 0u && ii[8 * stride] == 0u, "integral zero border");
    CHECK(ii[9 * stride - 1] == 64u, "8x8 ones total=64 (got %u)",
          ii[9 * stride - 1]);
    CHECK(panparasite::rectSum(ii, stride, 0, 0, 7, 7) == 64u, "rectSum full=64");
    CHECK(panparasite::rectSum(ii, stride, 2, 3, 5, 6) == 16u, "rectSum 4x4=16");
    // Non-uniform: s = x+y pattern vs brute force.
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) s[static_cast<std::size_t>(y) * 8u + x] =
          static_cast<uint8_t>(x + y);
    panparasite::computeIntegralImage(s, ii, 8, 8);
    uint32_t brute = 0;
    for (int v : s) brute += static_cast<uint32_t>(v);
    CHECK(panparasite::rectSum(ii, stride, 0, 0, 7, 7) == brute,
          "rectSum matches brute (%u)", brute);
  }

  // ---- (c) Integral: full-frame constant spot ----
  {
    std::memset(arena.frameB(), 7, panparasite::kFramePixels);
    panparasite::computeIntegralImage(arena);
    const uint32_t* ii = arena.integral();
    const int S = 1921;
    CHECK(ii[(1 * S) + 1] == 7u, "integral (0,0)=7");
    CHECK(ii[(1 * S) + 1920] == 7u * 1920u, "integral row0 sum=%u",
          ii[S + 1920]);
    CHECK(ii[(1080 * S) + 1920] == 7u * 2073600u, "integral total=%u (expect %u)",
          ii[1080 * S + 1920], 7u * 2073600u);
    CHECK(panparasite::rectSum(ii, S, 0, 0, 1919, 1079) == 7u * 2073600u,
          "rectSum full-frame exact");
    CHECK(panparasite::rectSum(ii, S, 100, 100, 162, 162) == 7u * 63u * 63u,
          "rectSum 63x63 exact");
  }

  // ---- (d) Binarize: small dark-square + uniform ----
  {
    uint8_t s[4096];
    uint32_t ii[4225];
    uint8_t mk[4096];
    std::memset(s, 200, sizeof(s));
    for (int y = 24; y < 40; ++y)
      for (int x = 24; x < 40; ++x) s[static_cast<std::size_t>(y) * 64u + x] = 50;
    panparasite::computeIntegralImage(s, ii, 64, 64);
    panparasite::adaptiveBinarize(s, ii, mk, 64, 64);
    CHECK(mk[32 * 64 + 32] == 255u, "dark square center fg=255");
    CHECK(mk[0] == 0u, "bright corner bg=0");
    // Uniform -> all background (pix == mean, never < 0.88*mean).
    std::memset(s, 200, sizeof(s));
    panparasite::computeIntegralImage(s, ii, 64, 64);
    panparasite::adaptiveBinarize(s, ii, mk, 64, 64);
    std::size_t fg = 0;
    for (uint8_t v : mk) fg += (v == 255u);
    CHECK(fg == 0, "uniform 64x64 all bg (fg=%zu)", fg);
  }

  // ---- (d) Binarize: full-frame uniform smoke ----
  {
    std::memset(arena.frameB(), 200, panparasite::kFramePixels);
    panparasite::computeIntegralImage(arena);
    panparasite::adaptiveBinarize(arena);
    std::size_t fg = 0;
    const uint8_t* mk = arena.mask();
    for (std::size_t i = 0; i < panparasite::kFramePixels; ++i) fg += (mk[i] == 255u);
    CHECK(fg == 0, "uniform 1920x1080 all bg (fg=%zu)", fg);
  }

  // ---- (e) Closing: hole-fill + gap-bridge on stack ----
  {
    uint8_t m[1024];
    uint32_t t[1024];
    uint8_t out[1024];
    // Hole fill: 16x16 block at [8..23]^2 with 3x3 hole at [14..16]^2.
    std::memset(m, 0, sizeof(m));
    for (int y = 8; y < 24; ++y)
      for (int x = 8; x < 24; ++x) m[static_cast<std::size_t>(y) * 32u + x] = 255u;
    for (int y = 14; y < 17; ++y)
      for (int x = 14; x < 17; ++x) m[static_cast<std::size_t>(y) * 32u + x] = 0u;
    panparasite::dilateMask11x11(m, t, 32, 32);
    panparasite::erodeMask11x11(t, out, 32, 32);
    CHECK(out[15 * 32 + 15] == 255u, "3x3 hole filled by closing");
    CHECK(out[0] == 0u, "far background stays 0");
    // Gap bridge: two 8x8 blocks with 4 px gap.
    std::memset(m, 0, sizeof(m));
    for (int y = 12; y < 20; ++y)
      for (int x = 4; x < 12; ++x) m[static_cast<std::size_t>(y) * 32u + x] = 255u;
    for (int y = 12; y < 20; ++y)
      for (int x = 16; x < 24; ++x) m[static_cast<std::size_t>(y) * 32u + x] = 255u;
    panparasite::dilateMask11x11(m, t, 32, 32);
    panparasite::erodeMask11x11(t, out, 32, 32);
    CHECK(out[15 * 32 + 14] == 255u, "4px gap bridged by closing");
  }

  // ---- (f) Dual-path equivalence (full vs tiled) ----
  {
    uint8_t* fb = arena.frameB();
    for (int y = 0; y < 1080; ++y) {
      for (int x = 0; x < 1920; ++x) {
        uint8_t v = static_cast<uint8_t>((x * 13 + y * 29 + (x * y) % 97) & 0xFF);
        fb[static_cast<std::size_t>(y) * 1920u + x] = v;
      }
    }
    for (int y = 400; y < 600; ++y)
      for (int x = 800; x < 1000; ++x)
        fb[static_cast<std::size_t>(y) * 1920u + x] = 40;  // dark body
    panparasite::runStage2FullFrame(arena);
    const uint64_t hFull = fnv1a(arena.mask(), panparasite::kFramePixels);
    std::size_t fgFull = 0;
    for (std::size_t i = 0; i < panparasite::kFramePixels; ++i)
      fgFull += (arena.mask()[i] == 255u);
    std::memcpy(arena.frameA(), arena.mask(), panparasite::kFramePixels);  // save
    panparasite::runStage2Tiled32(arena);
    const uint64_t hTiled = fnv1a(arena.mask(), panparasite::kFramePixels);
    const int cmp = std::memcmp(arena.frameA(), arena.mask(), panparasite::kFramePixels);
    CHECK(hFull == hTiled && cmp == 0, "tiled bit-identical (fnv %llx vs %llx)",
          (unsigned long long)hFull, (unsigned long long)hTiled);
    CHECK(fgFull > 0 && fgFull < panparasite::kFramePixels / 2,
          "sane fg count %zu", fgFull);
    std::printf("  [info] stage2 fg=%zu fnv=%llx\n", fgFull, (unsigned long long)hFull);
  }

  // ---- (g) Zero-heap hot region ----
  g_newCalls.store(0);
  g_deleteCalls.store(0);
  g_countHeap.store(true);
  {
    // Representative hot traffic: gate + integral + binarize + close.
    // No printing/STL/Mat inside this block.
    panparasite::AcquisitionGate gate;
    volatile uint64_t sink = 0;
    panparasite::AcquisitionMetrics m = panparasite::checkAcquisitionGate(arena, gate);
    sink += static_cast<uint64_t>(m.delta_it * 1000.0) + static_cast<uint64_t>(m.phi_focus);
    panparasite::computeIntegralImage(arena);
    sink += arena.integral()[1000];
    panparasite::adaptiveBinarize(arena);
    sink += arena.mask()[1000];
    panparasite::closeMask11x11Elliptical(arena);
    sink += arena.mask()[2000] + arena.equivMap()[2000];
    panparasite::AcquisitionMetrics m2 = panparasite::checkAcquisitionGate(arena, gate);
    sink += static_cast<uint64_t>(m2.delta_it * 1000.0);
    doNotOptimizeSink(sink);
  }
  g_countHeap.store(false);
  CHECK(g_newCalls.load() == 0, "zero heap new in Stage1+2 hot path (got %lld)",
        g_newCalls.load());
  CHECK(g_deleteCalls.load() == 0, "zero heap delete in Stage1+2 hot path (got %lld)",
        g_deleteCalls.load());

  if (g_failures == 0) {
    std::printf("=== ALL PREPROCESSOR TESTS PASSED ===\n");
    return 0;
  }
  std::printf("=== %d PREPROCESSOR TEST(S) FAILED ===\n", g_failures);
  return 1;
}
