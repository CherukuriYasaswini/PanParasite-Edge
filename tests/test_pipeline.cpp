// PanParasite-Edge Phase 6 (Final) unit test: end-to-end pipeline.
// Covers:
//   (a) Gate locks by the 3rd identical textured field; chain runs only
//       when locked (early fields report zero detections).
//   (b) Drifting egg: detections every locked field, stable track id,
//       slide tally[ascaris] >= 1, snapshot coherent.
//   (c) Ring: one sector per locked field, boot scan validates all.
//   (d) Audit: boot + per-field rows, hash chain verifies.
//   (e) Telemetry builders on live results: topic exact, payload carries
//       node/slide/tally; status page renders metrics; review queue works;
//       drift alarm boundaries.
//   (f) ZERO heap new/delete across processField + render (tripwire).

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
#include "persistence.hpp"
#include "pipeline.hpp"
#include "telemetry_ui.hpp"

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

// Identical textured field in A and B (Delta=0, focus≫125 from noise).
// int64 ellipse math: |dx|<=1920 keeps products (~1.1e10) exact.
// NOTE (load-bearing): the R_w=31 window sits fully inside the large egg,
// so flat interior pixels sit ABOVE mu*0.88 and only the rim + ~6% noise
// speckle (value 52 < 52.8 threshold) binarize as fg. The Stage-2 11x11
// closing bridges that dense speckle (~4px spacing < r=5) into the solid
// M_body Stage 3 expects. Do NOT "clean up" the noise: a noiseless egg
// would binarize as a hollow rim and route to debris.
void drawField(uint8_t* a, uint8_t* b, int field) {
  const int cx = 700 + field * 40, cy = 540, rx = 80, ry = 55;
  for (int y = 0; y < 1080; ++y) {
    for (int x = 0; x < 1920; ++x) {
      const int nz = ((x * 73 + y * 149) % 17) - 8;
      int v = 200 + nz;
      const int64_t dx = static_cast<int64_t>(x - cx);
      const int64_t dy = static_cast<int64_t>(y - cy);
      if (dx * dx * ry * ry + dy * dy * rx * rx <=
          static_cast<int64_t>(rx) * rx * ry * ry) {
        v = 60 + nz;
      }
      if (v < 0) v = 0;
      if (v > 255) v = 255;
      const std::size_t i = static_cast<std::size_t>(y) * 1920u + x;
      a[i] = static_cast<uint8_t>(v);
      b[i] = static_cast<uint8_t>(v);
    }
  }
}

}  // namespace

int main() {
  std::printf("=== PanParasite-Edge Phase 6: pipeline test ===\n");

  static panparasite::StaticMemoryArena arena;
  CHECK(arena.isInitialized(), "arena initialized at boot");
  static panparasite::PanParasitePipeline pipe;

  const char* kRing = "test_pipe_ring.bin";
  const char* kDb = "test_pipe_audit.db";
  std::remove(kRing);
  std::remove(kDb);

  panparasite::PipelineConfig cfg;
  cfg.stage2 = panparasite::Stage2Mode::kFullFrame;
  std::snprintf(cfg.nodeId, sizeof(cfg.nodeId), "%s", "test-node-01");
  std::snprintf(cfg.ringPath, sizeof(cfg.ringPath), "%s", kRing);
  std::snprintf(cfg.dbPath, sizeof(cfg.dbPath), "%s", kDb);
  CHECK(pipe.boot(arena, cfg), "pipeline boots");

  // ---- (a) gate lock sequencing ----
  panparasite::FieldReport r0, r1, r2;
  drawField(arena.frameA(), arena.frameB(), 0);
  CHECK(pipe.processField(arena, r0), "field 0 processes");
  drawField(arena.frameA(), arena.frameB(), 0);
  CHECK(pipe.processField(arena, r1), "field 1 processes");
  drawField(arena.frameA(), arena.frameB(), 0);
  CHECK(pipe.processField(arena, r2), "field 2 processes");
  CHECK(!r0.gateLocked && !r1.gateLocked, "fields 0-1 unlocked (chain gated)");
  CHECK(r0.numDetections == 0 && r1.numDetections == 0, "no dets pre-lock");
  CHECK(r2.gateLocked, "field 2 locked (phi=%.1f)", r2.phiFocus);
  CHECK(r2.numCandidates >= 1 && r2.numDetections >= 1,
        "locked field detects (cand=%d det=%d)", r2.numCandidates,
        r2.numDetections);

  // ---- (b) drifting slide: stable track + ascaris tally ----
  for (int f = 3; f < 8; ++f) {
    drawField(arena.frameA(), arena.frameB(), f);
    panparasite::FieldReport r;
    if (!pipe.processField(arena, r)) {
      CHECK(false, "field %d processes", f);
      break;
    }
    CHECK(r.gateLocked && r.numDetections >= 1, "field %d tracks egg", f);
  }
  int tally[7] = {};
  pipe.slideTally(tally);
  CHECK(pipe.verifiedTotal() >= 1, "slide verified >= 1 (got %d)",
        pipe.verifiedTotal());
  CHECK(tally[0] >= 1, "ascaris tally >= 1");
  CHECK(pipe.snapshot().activeTracks == 1, "one stable track (got %d)",
        pipe.snapshot().activeTracks);
  CHECK(pipe.snapshot().fields == 8, "snapshot fields == 8 (got %d)",
        pipe.snapshot().fields);
  CHECK(pipe.snapshot().slideUuid[0] != '\0', "snapshot carries slide UUID");

  // ---- (c) ring: one sector per locked field (fields 2..7 = 6) ----
  {
    panparasite::RingScanStats st;
    CHECK(panparasite::ringScan(arena, kRing, 256, st), "ring rescan runs");
    CHECK(st.valid == 6 && st.valid + st.corrupt == 256 && st.maxSeq == 6u,
          "6 field sectors, rest virgin (v=%d c=%d m=%u)", st.valid, st.corrupt,
          st.maxSeq);
  }

  // ---- (d) audit: boot + 6 field rows, chain verifies ----
  {
    panparasite::AuditDB a;
    CHECK(panparasite::auditOpen(a, kDb), "audit reopens persisted ledger");
    CHECK(panparasite::auditRowCount(a) == 7, "7 ledger rows (got %d)",
          panparasite::auditRowCount(a));
    int fail = -99;
    CHECK(panparasite::auditVerifyChain(a, &fail) && fail == -1,
          "persisted chain verifies");
    panparasite::auditClose(a);
  }

  // ---- (e) telemetry on live results ----
  {
    char topic[128];
    CHECK(panparasite::mqttBuildTopic("test-node-01", topic, sizeof(topic)) &&
              std::strcmp(topic,
                          "telemetry/sth/test-node-01/slide_summary") == 0,
          "MQTT topic exact (%s)", topic);
    char payload[512];
    CHECK(panparasite::mqttBuildSlideSummary(
              "test-node-01", pipe.snapshot().slideUuid, tally,
              pipe.verifiedTotal(), pipe.fieldIndex(), 0.02, payload,
              sizeof(payload)),
          "payload builds");
    CHECK(std::strstr(payload, "\"verified\"") != nullptr &&
              std::strstr(payload, "test-node-01") != nullptr,
          "payload carries node + counts");
    std::printf("  [info] %s\n  %s\n", topic, payload);
    char html[16384];
    const int n = panparasite::uiRenderPage(pipe.snapshot(), html, sizeof(html));
    CHECK(n > 512 && std::strstr(html, "#21918C") != nullptr,
          "page renders viridis metrics (%d B)", n);
    CHECK(std::strstr(html, "min-height:56px") != nullptr &&
              std::strstr(html, "min-width:56px") != nullptr,
          "touch targets >= 56px");
    CHECK(std::strstr(html, "Status") != nullptr, "localized labels present");
    static panparasite::UiReviewQueue rq;
    CHECK(panparasite::uiReviewPush(rq, "CONFIRM", 2), "review CONFIRM queued");
    CHECK(!panparasite::uiReviewPush(rq, "MAYBE", 2), "bad action refused");
    CHECK(!panparasite::cloudwatchDriftAlarm(0.10), "10%% abstain: no alarm");
    CHECK(panparasite::cloudwatchDriftAlarm(0.11), "11%% abstain: alarm");
    CHECK(panparasite::slideAbstainRate(1, 1, 20) == 0.10, "abstain math");
  }

  // ---- (f) Zero-heap hot region (locked-field full chain + render) ----
  {
    drawField(arena.frameA(), arena.frameB(), 9);
    g_newCalls.store(0);
    g_deleteCalls.store(0);
    g_countHeap.store(true);
    volatile uint64_t sink = 0;
    {
      panparasite::FieldReport r;
      if (pipe.processField(arena, r)) {
        sink += static_cast<uint64_t>(r.numDetections + 1);
        sink += static_cast<uint64_t>(r.slideVerifiedTotal);
      }
      char html[4096];
      sink += static_cast<uint64_t>(
          panparasite::uiRenderPage(pipe.snapshot(), html, sizeof(html)));
      char tp[128], js[512];
      int t[7] = {};
      pipe.slideTally(t);
      if (panparasite::mqttBuildTopic("n", tp, sizeof(tp))) sink += tp[0];
      if (panparasite::mqttBuildSlideSummary("n", pipe.snapshot().slideUuid, t,
                                            1, 9, 0.0, js, sizeof(js))) {
        sink += js[10];
      }
    }
    doNotOptimizeSink(sink);
    g_countHeap.store(false);
    CHECK(g_newCalls.load() == 0, "zero heap new end-to-end (got %lld)",
          g_newCalls.load());
    CHECK(g_deleteCalls.load() == 0, "zero heap delete end-to-end (got %lld)",
          g_deleteCalls.load());
  }

  pipe.shutdown();
  std::remove(kRing);
  std::remove(kDb);

  if (g_failures == 0) {
    std::printf("=== ALL PIPELINE TESTS PASSED ===\n");
    return 0;
  }
  std::printf("=== %d PIPELINE TEST(S) FAILED ===\n", g_failures);
  return 1;
}
