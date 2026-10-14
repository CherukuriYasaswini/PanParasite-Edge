#pragma once

// PanParasite-Edge Phase 6 (Final): End-to-End Pipeline Orchestrator.
// Wires Stages 1 -> 8 into PanParasitePipeline with strictly ZERO heap
// allocations across Stages 2-6 (all stage primitives are already
// heap-free; this orchestrator adds only fixed-size locals + snprintf).
//
// Per-field flow (frameB must hold the newest frame on entry):
//   S1 gate -> S2 binarize -> S3 topology -> S4A gate -> S4 occlusion +
//     per-candidate ROI extract/diffuse -> S5 classify -> S6 track/commit
//     -> S7 ring+audit persist -> S8 snapshot publish.
// Stage 2 destroys the integral image; Stage 3 reborrows it as the bg
// label plane; Stage 5 reborrows it again as activation planes. The
// documented order above is therefore load-bearing — do not reorder.

#include "active_perception.hpp"
#include "inference_tracker.hpp"
#include "memory_arena.hpp"
#include "persistence.hpp"
#include "preprocessor.hpp"
#include "telemetry_ui.hpp"
#include "topology_engine.hpp"

namespace panparasite {

// ------------------------------------------------------------------
// Pipeline configuration (plain data, caller-owned).
// ------------------------------------------------------------------
struct PipelineConfig {
  Stage2Mode stage2 = Stage2Mode::kFullFrame;
  char nodeId[64] = {};
  char ringPath[256] = {};
  char dbPath[256] = {};
};

// Per-field outcome (plain data for CLI/tests/benchmarks).
struct FieldReport {
  int fieldIndex = -1;
  bool gateLocked = false;
  double deltaIt = 0.0;
  double phiFocus = 0.0;
  int numCandidates = 0;
  int numGated = 0;
  int numDetections = 0;
  int numTriggered = 0;
  bool capExceeded = false;
  int slideVerifiedTotal = 0;
  int fp32Fallbacks = 0;
};

// Unified pipeline. ~45 KB object: declare static/global (never on small
// stacks, never with new). All methods noexcept; false = I/O or boot
// failure (pipeline left in a safe, retryable state).
class PanParasitePipeline {
 public:
  PanParasitePipeline() noexcept = default;
  PanParasitePipeline(const PanParasitePipeline&) = delete;
  PanParasitePipeline& operator=(const PanParasitePipeline&) = delete;

  // BOOT: film weights, slide UUID, ring open (format-on-first-miss),
  // audit open, counters reset. arena must be initialized already.
  bool boot(StaticMemoryArena& arena, const PipelineConfig& cfg) noexcept;
  void shutdown() noexcept;

  // Run one field end-to-end. HOT-PATH SAFE across Stages 2-6.
  bool processField(StaticMemoryArena& arena, FieldReport& rep) noexcept;

  // Whole-slide accessors for CLI/tests (plain reads, no heap).
  int verifiedTotal() const noexcept;
  void slideTally(int out[7]) const noexcept;
  const UiSnapshot& snapshot() const noexcept { return snap_; }
  int fieldIndex() const noexcept { return fieldIndex_; }

 private:
  PipelineConfig cfg_;
  AcquisitionGate gate_;
  TopoResult topo_;
  GatedField gated_;
  MicroNetAudit audit_;
  TrackTable tracks_;
  SlideDB slide_;
  RingWriter ring_;
  AuditDB adb_;
  UiSnapshot snap_;
  char slideUuid_[37] = {};
  int fieldIndex_ = 0;
  bool booted_ = false;
};

}  // namespace panparasite
