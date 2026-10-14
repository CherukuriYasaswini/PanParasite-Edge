// PanParasite-Edge Phase 6 (Final): CLI benchmark + slide runner.
// Builds the full Stages 1-8 pipeline on static storage (no heap):
// synthetic drifting-egg fields -> gate -> inference -> track -> persist.
// Usage:
//   panparasite_edge [--simulate N] [--ring FILE] [--db FILE] [--node ID]
//                    [--tiled] [--bench M] [--serve PORT SECS] [--help]

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "memory_arena.hpp"
#include "persistence.hpp"
#include "pipeline.hpp"
#include "telemetry_ui.hpp"

#if defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <time.h>
#endif

namespace {

void usage() {
  std::printf(
      "panparasite_edge - PanParasite field slide runner\n"
      "  --simulate N   synthetic fields to process (default 5)\n"
      "  --ring FILE    ring buffer path (default slide_ring.bin)\n"
      "  --db FILE      audit ledger path (default slide_audit.db)\n"
      "  --node ID      telemetry node id (default pi5-edge-01)\n"
      "  --tiled        Stage 2 tiled-32 benchmark path\n"
      "  --bench M      extra timed fields for ms/field benchmark\n"
      "  --serve P S    status server on port P for S seconds (0 = off)\n"
      "  --help         this text\n");
}

int parseInt(const char* s, int dflt) {
  if (s == nullptr || *s == '\0') return dflt;
  int v = 0;
  for (const char* p = s; *p != '\0'; ++p) {
    if (*p < '0' || *p > '9') return dflt;
    v = v * 10 + (*p - '0');
    if (v > 1000000) return dflt;
  }
  return v;
}

// Deterministic textured field: bright noisy matrix + dark drifting egg.
// Identical content in A and B => Delta I_t = 0; sensor noise keeps focus
// variance far above the 125.0 gate floor.
void drawField(panparasite::StaticMemoryArena& arena, int field) {
  uint8_t* a = arena.frameA();
  uint8_t* b = arena.frameB();
  const int cx = 700 + (field * 40) % 900;  // drift, wraps on-screen
  const int cy = 540, rx = 80, ry = 55;
  for (int y = 0; y < 1080; ++y) {
    for (int x = 0; x < 1920; ++x) {
      const int nz = ((x * 73 + y * 149) % 17) - 8;
      int v = 200 + nz;
      const int64_t dx = static_cast<int64_t>(x - cx);
      const int64_t dy = static_cast<int64_t>(y - cy);
      // int64: |dx|<=1920 keeps dx^2*ry^2 (~1.1e10) far below 2^63.
      if (dx * dx * ry * ry + dy * dy * rx * rx <=
          static_cast<int64_t>(rx) * rx * ry * ry) {
        v = 60 + nz;  // egg body (adaptive threshold catches it)
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

int main(int argc, char** argv) {
  const char* ringPath = "slide_ring.bin";
  const char* dbPath = "slide_audit.db";
  const char* nodeId = "pi5-edge-01";
  int nFields = 5;
  int benchExtra = 0;
  int servePort = 0, serveSecs = 0;
  bool tiled = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0) {
      usage();
      return 0;
    } else if (std::strcmp(argv[i], "--simulate") == 0 && i + 1 < argc) {
      nFields = parseInt(argv[++i], 5);
    } else if (std::strcmp(argv[i], "--ring") == 0 && i + 1 < argc) {
      ringPath = argv[++i];
    } else if (std::strcmp(argv[i], "--db") == 0 && i + 1 < argc) {
      dbPath = argv[++i];
    } else if (std::strcmp(argv[i], "--node") == 0 && i + 1 < argc) {
      nodeId = argv[++i];
    } else if (std::strcmp(argv[i], "--tiled") == 0) {
      tiled = true;
    } else if (std::strcmp(argv[i], "--bench") == 0 && i + 1 < argc) {
      benchExtra = parseInt(argv[++i], 0);
    } else if (std::strcmp(argv[i], "--serve") == 0 && i + 2 < argc) {
      servePort = parseInt(argv[++i], 0);
      serveSecs = parseInt(argv[++i], 0);
    } else {
      std::printf("unknown arg: %s\n", argv[i]);
      usage();
      return 1;
    }
  }
  if (nFields < 1) nFields = 1;

  static panparasite::StaticMemoryArena arena;
  if (!arena.isInitialized()) {
    std::printf("arena init failed\n");
    return 1;
  }
  static panparasite::PanParasitePipeline pipe;
  panparasite::PipelineConfig cfg;
  cfg.stage2 = tiled ? panparasite::Stage2Mode::kTiled32
                     : panparasite::Stage2Mode::kFullFrame;
  std::snprintf(cfg.nodeId, sizeof(cfg.nodeId), "%s", nodeId);
  std::snprintf(cfg.ringPath, sizeof(cfg.ringPath), "%s", ringPath);
  std::snprintf(cfg.dbPath, sizeof(cfg.dbPath), "%s", dbPath);
  if (!pipe.boot(arena, cfg)) {
    std::printf("pipeline boot failed\n");
    return 1;
  }

  static panparasite::UiServer server;
  static panparasite::UiReviewQueue reviewQ;
  const bool serving = servePort > 0 && serveSecs > 0;
  if (serving && !panparasite::uiStart(server, reviewQ, servePort)) {
    std::printf("ui server on %d failed to start (continuing headless)\n",
                servePort);
  }

  int totalDet = 0;
  for (int f = 0; f < nFields; ++f) {
    drawField(arena, f);
    panparasite::FieldReport rep;
    if (!pipe.processField(arena, rep)) {
      std::printf("field %d failed\n", f);
      pipe.shutdown();
      if (serving) panparasite::uiStop(server);
      return 1;
    }
    totalDet += rep.numDetections;
    std::printf(
        "field %d lock=%d dI=%.2f phi=%.1f cand=%d gated=%d det=%d trig=%d "
        "ver=%d cap=%d fb=%d\n",
        rep.fieldIndex, (int)rep.gateLocked, rep.deltaIt, rep.phiFocus,
        rep.numCandidates, rep.numGated, rep.numDetections, rep.numTriggered,
        rep.slideVerifiedTotal, (int)rep.capExceeded, rep.fp32Fallbacks);
  }

  if (benchExtra > 0) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int f = 0; f < benchExtra; ++f) {
      drawField(arena, nFields + (f % 8));
      panparasite::FieldReport rep;
      if (!pipe.processField(arena, rep)) {
        std::printf("bench field %d failed\n", f);
        break;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("bench: %d fields in %.1f ms (%.2f ms/field)\n", benchExtra, ms,
                benchExtra > 0 ? ms / benchExtra : 0.0);
  }

  int tally[7] = {};
  pipe.slideTally(tally);
  std::printf("slide %s verified=%d det_total=%d tally=[%d,%d,%d,%d,%d,%d,%d]\n",
              "done", pipe.verifiedTotal(), totalDet, tally[0], tally[1],
              tally[2], tally[3], tally[4], tally[5], tally[6]);

  char topic[128], payload[512];
  if (panparasite::mqttBuildTopic(nodeId, topic, sizeof(topic)) &&
      panparasite::mqttBuildSlideSummary(nodeId, pipe.snapshot().slideUuid,
                                         tally, pipe.verifiedTotal(),
                                         pipe.fieldIndex(), 0.0, payload,
                                         sizeof(payload))) {
    std::printf("mqtt %s\n  %s\n", topic, payload);
  }

  if (serving) {
    panparasite::uiPublish(server, pipe.snapshot());
    std::printf("serving status on :%d for %ds ...\n", servePort, serveSecs);
    for (int s = 0; s < serveSecs * 5; ++s) {
#if defined(_WIN32) || defined(_WIN64)
      Sleep(200);
#else
      struct timespec ts;
      ts.tv_sec = 0;
      ts.tv_nsec = 200000000L;
      nanosleep(&ts, nullptr);
#endif
      panparasite::uiPublish(server, pipe.snapshot());
    }
    panparasite::uiStop(server);
  }

  pipe.shutdown();
  return 0;
}
