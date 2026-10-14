// PanParasite-Edge Phase 1 unit test: StaticMemoryArena.
// Verifies:
//   (a) byte-accurate 9-buffer map totalling 25,108,516 B (~23.9459 MiB),
//   (b) DisjointSetNode == 24 B, DSU arena == 786,432 B,
//   (c) DIO sector 4096-byte aligned,
//   (d) non-owning cv::Mat wrappers (data ptr == arena ptr, u == nullptr,
//       correct size/type, no Mat::create called anywhere in the hot path),
//   (e) ZERO heap allocations (global new/delete) across a simulated
//       Stages 2-5 hot loop after boot initialization.
//
// Method for (e): global operator new/delete overloads count allocations
// only while g_countHeap is true. Boot (arena construction, OpenCV static
// init, iostream setup) runs with counting OFF. The hot loop then performs
// only raw-pointer + Mat-header work with counting ON and zero printing /
// STL / vector::push_back inside the counted region. Any hidden new/delete
// (e.g. an accidental Mat::create cloning data) trips the assertion.
//
// NOTE: the overloads below intentionally route to malloc/free so the test
// harness itself never recurses into counted new. malloc/calloc use inside
// the counted region is ALSO forbidden by Rule 2; this test catches new/
// delete directly and catches malloc-style misuse indirectly because every
// wrapper invariant (u == nullptr, data == arena ptr) would break if a copy
// had been heap-allocated. A companion CI grep audit
// (forbidden: "Mat::create", "push_back", "malloc(", "calloc(", " new " in
//  src/stage*.cpp) covers the lexical side; see the comment at the bottom.
//
// Build: add_executable(test_arena tests/test_arena.cpp) + add_test(...).
// Run:   ctest -R arena_zero_alloc --output-on-failure

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

// ---- Heap-allocation tripwire (must precede all other includes that
// ---- might emit sized-delete references; definitions are below) ----
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
// C++14 sized deallocation (some toolchains emit these).
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
// Nothrow variants (route through malloc, counted the same way).
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

#include <opencv2/core.hpp>

#include "memory_arena.hpp"

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

// Prevent the optimizer from deleting the hot-loop workload while still
// emitting zero heap calls: compiler barrier over a sunk value.
inline void doNotOptimizeSink(volatile uint64_t v) {
#if defined(_MSC_VER)
  volatile uint64_t sink = v;
  (void)sink;
#else
  asm volatile("" : "+r"(v) : : "memory");
  (void)v;
#endif
}

}  // namespace

int main() {
  std::printf("=== PanParasite-Edge Phase 1: StaticMemoryArena test ===\n");

  // ---- Boot phase (counting OFF): static storage => zero heap by ----
  // ---- construction. Function-static keeps 25 MB out of the stack. --
  static panparasite::StaticMemoryArena arena;
  CHECK(arena.isInitialized(), "arena initialized at boot");
  CHECK(panparasite::kArenaTotalBytes == 25108516u, "total bytes == 25108516");
  CHECK(panparasite::StaticMemoryArena::totalBytes() == 25108516u,
        "StaticMemoryArena::totalBytes() == 25108516");
  {
    const double mib = panparasite::StaticMemoryArena::totalMiB();
    CHECK(mib > 23.9 && mib < 24.0, "total MiB ~23.94 (got %.4f)", mib);
    std::printf("  [info] arena total = 25108516 B = %.4f MiB\n", mib);
    std::printf("  [info] sizeof(DisjointSetNode)=%zu sizeof(arena)=%zu\n",
                sizeof(panparasite::DisjointSetNode), sizeof(arena));
  }
  CHECK(sizeof(panparasite::DisjointSetNode) == 24u, "DisjointSetNode == 24 B");
  CHECK(panparasite::kDsuBytes == 786432u, "DSU arena == 786432 B");
  CHECK(panparasite::kFrameBytes == 2073600u, "frame buffer == 2073600 B");
  CHECK(panparasite::kIntegralBytes == 8306404u, "integral buffer == 8306404 B");
  CHECK(panparasite::kEquivMapBytes == 8294400u, "equiv map == 8294400 B");
  CHECK(panparasite::kFilmParamBytes == 1480000u, "FiLM blob == 1480000 B");

  // DIO alignment (hard requirement for Stage 7 O_DIRECT).
  CHECK(arena.dioSector() != nullptr, "DIO sector non-null");
  CHECK(arena.isDioAligned(), "DIO sector 4096-aligned (ptr=%p)",
        (const void*)arena.dioSector());

  // mlock/VirtualLock is best-effort at boot (stock RLIMIT_MEMLOCK or a
  // small Windows working set may refuse). Record, do not fail: the
  // ledger (Stage 7) persists the degraded mode instead.
  std::printf("  [info] FiLM page lock: %s\n",
              arena.isFilmLocked()
                  ? "LOCKED (mlock/VirtualLock ok)"
                  : "UNLOCKED (sim fallback; raise ulimit -l / working set for prod)");

  // ---- Non-owning Mat wrappers (still boot phase, printing allowed) --
  {
    cv::Mat a = arena.frameAMat();
    cv::Mat b = arena.frameBMat();
    cv::Mat m = arena.maskMat();
    cv::Mat ii = arena.integralMat();
    cv::Mat em = arena.equivMapMat();
    cv::Mat sc = arena.scratchMat();

    CHECK(!a.empty() && a.rows == 1080 && a.cols == 1920 && a.type() == CV_8UC1,
          "frameA Mat 1080x1920 CV_8UC1");
    CHECK(!b.empty() && b.rows == 1080 && b.cols == 1920 && b.type() == CV_8UC1,
          "frameB Mat 1080x1920 CV_8UC1");
    CHECK(!m.empty() && m.rows == 1080 && m.cols == 1920 && m.type() == CV_8UC1,
          "mask Mat 1080x1920 CV_8UC1");
    CHECK(!ii.empty() && ii.rows == 1081 && ii.cols == 1921 && ii.type() == CV_32SC1,
          "integral Mat 1081x1921 CV_32SC1");
    CHECK(!em.empty() && em.rows == 1080 && em.cols == 1920 && em.type() == CV_32SC1,
          "equivMap Mat 1080x1920 CV_32SC1");
    CHECK(!sc.empty() && sc.rows == 128 && sc.cols == 128 && sc.type() == CV_8UC1,
          "scratch Mat 128x128 CV_8UC1");

    CHECK(a.data == arena.frameA(), "frameA Mat wraps arena ptr (no copy)");
    CHECK(b.data == arena.frameB(), "frameB Mat wraps arena ptr (no copy)");
    CHECK(m.data == arena.mask(), "mask Mat wraps arena ptr (no copy)");
    CHECK(ii.data == reinterpret_cast<uint8_t*>(arena.integral()),
          "integral Mat wraps arena ptr (no copy)");
    CHECK(em.data == reinterpret_cast<uint8_t*>(arena.equivMap()),
          "equivMap Mat wraps arena ptr (no copy)");
    CHECK(sc.data == arena.scratchRoi(), "scratch Mat wraps arena ptr (no copy)");

    // Non-owning proof: external-data Mats carry no refcount (u == null).
    CHECK(a.u == nullptr, "frameA Mat non-owning (u==nullptr)");
    CHECK(b.u == nullptr, "frameB Mat non-owning (u==nullptr)");
    CHECK(m.u == nullptr, "mask Mat non-owning (u==nullptr)");
    CHECK(ii.u == nullptr, "integral Mat non-owning (u==nullptr)");
    CHECK(em.u == nullptr, "equivMap Mat non-owning (u==nullptr)");
    CHECK(sc.u == nullptr, "scratch Mat non-owning (u==nullptr)");
    CHECK(a.isContinuous() && m.isContinuous(), "frame/mask Mats continuous");

    // Const-view overloads must alias the same storage.
    const auto& carena = arena;
    CHECK(carena.frameAMat().data == arena.frameA(), "const frameA view aliases");
    CHECK(carena.integralMat().data == reinterpret_cast<const uint8_t*>(arena.integral()),
          "const integral view aliases");
  }

  // ---- Hot-path simulation with heap counting ON --------------------
  // No printing, no STL, no vector, no Mat::create in this region.
  // Workload mimics Stages 2-5 memory traffic at small scale:
  //   * Mat-header acquisition (headers only, no alloc),
  //   * raw ping-pong + mask writes,
  //   * integral/equiv-map word writes,
  //   * DSU parent-touch with path halving (bounded, no growth),
  //   * 128x128 scratch stencil.
  g_newCalls.store(0);
  g_deleteCalls.store(0);
  g_countHeap.store(true);

  volatile uint64_t sink = 0;
  for (int iter = 0; iter < 200; ++iter) {
    // Header-only views (must not allocate; verified by counters + u==null).
    cv::Mat fa = arena.frameAMat();
    cv::Mat fb = arena.frameBMat();
    cv::Mat mk = arena.maskMat();
    cv::Mat ii = arena.integralMat();
    cv::Mat em = arena.equivMapMat();
    cv::Mat sc = arena.scratchMat();

    uint8_t* faP = fa.data;
    uint8_t* fbP = fb.data;
    uint8_t* mkP = mk.data;
    uint32_t* iiP = reinterpret_cast<uint32_t*>(ii.data);
    uint32_t* emP = reinterpret_cast<uint32_t*>(em.data);
    uint8_t* scP = sc.data;

    // Strided touch (touches ~8K of the 25 MB per iter; enough to prove
    // write access without a 200*25MB memset slowing the test).
    for (std::size_t k = 0; k < 2048; ++k) {
      const std::size_t fi = (static_cast<std::size_t>(iter) * 7919u + k * 1024u) %
                             panparasite::kFramePixels;
      const uint8_t v = static_cast<uint8_t>((iter + k) & 0xFF);
      faP[fi] = v;
      fbP[fi] = static_cast<uint8_t>(v ^ 0xA5u);
      mkP[fi] = (v > 127) ? 255u : 0u;
      sink += static_cast<uint64_t>(faP[fi] + fbP[fi] + mkP[fi]);
    }
    for (std::size_t k = 0; k < 512; ++k) {
      const std::size_t ix =
          (static_cast<std::size_t>(iter) * 104729u + k * 4096u) % panparasite::kIntegralElems;
      iiP[ix] = static_cast<uint32_t>(iter + k);
      emP[(ix * 7u) % panparasite::kFramePixels] = static_cast<uint32_t>(k);
      sink += iiP[ix];
    }
    // DSU: bounded parent-path touch over the fixed 32768 arena.
    // No push_back / growth: indices wrap modulo capacity.
    panparasite::DisjointSetNode* dsu = arena.dsu();
    for (std::size_t k = 0; k < 512; ++k) {
      const uint32_t lbl =
          static_cast<uint32_t>((static_cast<std::size_t>(iter) * 131u + k) % panparasite::kMaxLabels);
      uint32_t p = dsu[lbl].parent;
      if (p >= panparasite::kMaxLabels) p = lbl;  // clamp, never grow
      dsu[lbl].parent = dsu[p].parent;           // one halving step
      sink += dsu[lbl].parent;
    }
    // 128x128 scratch stencil (Stage 4 Perona-Malik neighbourhood shape).
    for (int y = 1; y < 127; ++y) {
      uint8_t* row = scP + static_cast<std::size_t>(y) * 128u;
      for (int x = 1; x < 127; x += 8) {
        row[x] = static_cast<uint8_t>((row[x - 1] + row[x + 1] + iter) & 0xFF);
        sink += row[x];
      }
    }
    // FiLM blob + DIO sector: single-word touch proves mapping, no copy.
    arena.filmParams()[(static_cast<std::size_t>(iter) * 65537u) % panparasite::kFilmParamBytes] =
        static_cast<uint8_t>(iter);
    arena.dioSector()[iter % panparasite::kDioSectorBytes] = static_cast<uint8_t>(iter);
    sink += arena.filmParams()[0] + arena.dioSector()[0];
  }
  doNotOptimizeSink(sink);

  g_countHeap.store(false);
  const long long heapNew = g_newCalls.load();
  const long long heapDel = g_deleteCalls.load();

  CHECK(heapNew == 0, "zero heap new/new[] in hot loop (got %lld)", heapNew);
  CHECK(heapDel == 0, "zero heap delete/delete[] in hot loop (got %lld)", heapDel);

  // Post-loop integrity: wrappers still alias arena (no silent realloc).
  CHECK(arena.frameAMat().data == arena.frameA(), "post-loop frameA alias intact");
  CHECK(arena.maskMat().data == arena.mask(), "post-loop mask alias intact");
  CHECK(arena.integralMat().u == nullptr, "post-loop integral still non-owning");

  if (g_failures == 0) {
    std::printf("=== ALL ARENA TESTS PASSED ===\n");
    return 0;
  }
  std::printf("=== %d ARENA TEST(S) FAILED ===\n", g_failures);
  return 1;
}

// CI lexical audit (companion to the runtime counters above; run as:
//   grep -rnE "Mat::create|push_back|malloc\(|calloc\(|[^:]new " \
//     src/ include/ --include=*.cpp --include=*.hpp \
//   | grep -v test_arena | grep -v BOOT-ONLY
// and require empty output for src/stage*.cpp hot paths).
