#pragma once

// PanParasite-Edge Phase 5: Stage B FiLM-Conditioned Depthwise Micro-Net
// (5.2-5.4) + Stage 6 Bayesian Log-Odds Spatial Tracker & Slide Aggregator.
//
// OPENCODE.md Rules enforced:
//   Rule 2: HOT-PATH SAFE only — raw arena pointers, single-pass loops,
//           no malloc/calloc/new/vector::push_back/cv::Mat::create,
//           no temporary cv::Mat. Every function below is noexcept.
//   Stage 5: FiLM micro-net (INT8 native + FP32 fallback) + Platt T = 1.38.
//   Stage 6: Hungarian IoU >= 0.30 + log-odds +/-2.197.
//
// Buffer discipline (all inside StaticMemoryArena, no heap):
//   filmParams (1.48 MB mlock'd): INT8 tables + FP32 mirrors + scales.
//     Layout (bytes): INT8 blob ~7.9 KB, FP32 blob ~30.1 KB, total < 40 KB.
//   integral (borrowed): activation planes A/B as float32 (2 x 65536).
//     Stage 3 bg labels are dead by Stage 5; Stage 2 must rerun before its
//     integral values are needed again.
//   scratch (128x128): network input patch (read-only during inference).
//   frameA/B, mask, equivMap, dsu, DIO untouched by inference.
// caller storage:
//   MicroNetAudit (8 B), TrackTable (~4 KB), SlideDB (~7 KB) live in
//   static/global or generous-thread stack memory. No heap either way.
//
// OpenCV 5 hook: when built against OpenCV >= 5 the macro below reports
// the ENGINE_NEW dispatch as available; the deterministic zero-alloc
// native path below REMAINS authoritative (no cv::Mat is constructed in
// the hot path on any backend).

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "active_perception.hpp"
#include "memory_arena.hpp"
#include "topology_engine.hpp"

namespace panparasite {

// ------------------------------------------------------------------
// OpenCV 5 cv::dnn::ENGINE_NEW hook (compile-time dispatch flag only).
// ------------------------------------------------------------------
#if defined(CV_VERSION_MAJOR) && (CV_VERSION_MAJOR >= 5)
inline constexpr bool kDnnEngineNewAvailable = true;
#else
inline constexpr bool kDnnEngineNewAvailable = false;
#endif

// ------------------------------------------------------------------
// Stage 5.2: 7 diagnostic classes.
// ------------------------------------------------------------------
inline constexpr int kNumTaxa = 7;
inline constexpr int kTaxonAscaris = 0;
inline constexpr int kTaxonTrichuris = 1;
inline constexpr int kTaxonHookworm = 2;
inline constexpr int kTaxonMansoni = 3;
inline constexpr int kTaxonHaematobium = 4;
inline constexpr int kTaxonDebris = 5;
inline constexpr int kTaxonUnknown = 6;

// ------------------------------------------------------------------
// Stage 5.2: network geometry (fixed).
// ------------------------------------------------------------------
inline constexpr int kNetInHW = 128;
inline constexpr int kNetL1HW = 64;
inline constexpr int kNetL1C = 16;
inline constexpr int kNetL2HW = 32;
inline constexpr int kNetL2C = 32;
inline constexpr int kNetL3HW = 16;
inline constexpr int kNetL3C = 64;
inline constexpr int kNetActCap = 65536;  // max plane floats (64*64*16)

// ------------------------------------------------------------------
// Stage 5.2: parameter arena layout inside filmParams (1,480,000 B).
// INT8 section (int8 weights, int32 biases), then FP32 mirrors.
// Total ~38 KB << 1.48 MB; remainder reserved for future growth.
// ------------------------------------------------------------------
inline constexpr std::size_t kMiL1W = 144;          // 16*1*3*3 int8
inline constexpr std::size_t kMiL1B = 64;           // 16 int32
inline constexpr std::size_t kMiDW1W = 144;         // 16*3*3 int8
inline constexpr std::size_t kMiPW1W = 512;         // 32*16 int8
inline constexpr std::size_t kMiB1 = 128;           // 32 int32
inline constexpr std::size_t kMiF1W = 1024;         // 64*16 int8
inline constexpr std::size_t kMiF1B = 256;          // 64 int32
inline constexpr std::size_t kMiDW2W = 288;         // 32*3*3 int8
inline constexpr std::size_t kMiPW2W = 2048;        // 64*32 int8
inline constexpr std::size_t kMiB2 = 256;           // 64 int32
inline constexpr std::size_t kMiF2W = 2048;         // 128*16 int8
inline constexpr std::size_t kMiF2B = 512;          // 128 int32
inline constexpr std::size_t kMiHW = 448;           // 7*64 int8
inline constexpr std::size_t kMiHB = 28;            // 7 int32
inline constexpr std::size_t kMiInt8Total =
    kMiL1W + kMiL1B + kMiDW1W + kMiPW1W + kMiB1 + kMiF1W + kMiF1B +
    kMiDW2W + kMiPW2W + kMiB2 + kMiF2W + kMiF2B + kMiHW + kMiHB;  // 7900
inline constexpr std::size_t kMiL1Wf = 576;         // 144 fp32
inline constexpr std::size_t kMiL1Bf = 64;
inline constexpr std::size_t kMiDW1Wf = 576;
inline constexpr std::size_t kMiPW1Wf = 2048;
inline constexpr std::size_t kMiB1f = 128;
inline constexpr std::size_t kMiF1Wf = 4096;
inline constexpr std::size_t kMiF1Bf = 256;
inline constexpr std::size_t kMiDW2Wf = 1152;
inline constexpr std::size_t kMiPW2Wf = 8192;
inline constexpr std::size_t kMiB2f = 256;
inline constexpr std::size_t kMiF2Wf = 8192;
inline constexpr std::size_t kMiF2Bf = 512;
inline constexpr std::size_t kMiHWf = 1792;
inline constexpr std::size_t kMiHBf = 28;
inline constexpr std::size_t kMiFp32Total =
    kMiL1Wf + kMiL1Bf + kMiDW1Wf + kMiPW1Wf + kMiB1f + kMiF1Wf + kMiF1Bf +
    kMiDW2Wf + kMiPW2Wf + kMiB2f + kMiF2Wf + kMiF2Bf + kMiHWf + kMiHBf;
inline constexpr std::size_t kMiScales = 32;  // 8 fp32 requant scales
inline constexpr std::size_t kMiUsedTotal =
    kMiInt8Total + kMiFp32Total + kMiScales;
static_assert(kMiUsedTotal <= 1480000u, "micro-net must fit FiLM arena");
static_assert(kMiUsedTotal < 60000u, "micro-net footprint sanity (<60 KB)");

// Byte offsets of each table from filmParams()[0].
inline constexpr std::size_t kOffL1W = 0;
inline constexpr std::size_t kOffL1B = kOffL1W + kMiL1W;
inline constexpr std::size_t kOffDW1W = kOffL1B + kMiL1B;
inline constexpr std::size_t kOffPW1W = kOffDW1W + kMiDW1W;
inline constexpr std::size_t kOffB1 = kOffPW1W + kMiPW1W;
inline constexpr std::size_t kOffF1W = kOffB1 + kMiB1;
inline constexpr std::size_t kOffF1B = kOffF1W + kMiF1W;
inline constexpr std::size_t kOffDW2W = kOffF1B + kMiF1B;
inline constexpr std::size_t kOffPW2W = kOffDW2W + kMiDW2W;
inline constexpr std::size_t kOffB2 = kOffPW2W + kMiPW2W;
inline constexpr std::size_t kOffF2W = kOffB2 + kMiB2;
inline constexpr std::size_t kOffF2B = kOffF2W + kMiF2W;
inline constexpr std::size_t kOffHW = kOffF2B + kMiF2B;
inline constexpr std::size_t kOffHB = kOffHW + kMiHW;
inline constexpr std::size_t kOffFp32 = kOffHB + kMiHB;
inline constexpr std::size_t kOffScales = kOffFp32 + kMiFp32Total;

// Fixed-point FiLM quantization with PER-FEATURE scales (exact powers of
// two relationship preserved per feature: W_q[j] = round(coef*WS[j]),
// fq[j] = round(f[j]*Q[j]), WS[j]*Q[j] = 512 = 2^9 for every j).
// Per-feature Q keeps tiny moment coefficients exact: mu20' ~1600 needs
// Q=1/16 (fq ~100) so coef -0.004 maps to -33, not zero. Uniform x64
// would round |coef| < 1/16 to zero and destroy hookworm/ascaris sizing.
inline constexpr double kFeatQ[16] = {
    64.0, 32.0, 128.0, 16.0, 64.0, 256.0, 64.0, 64.0,
    0.0625, 0.125, 0.125, 0.015625, 0.015625, 0.015625, 0.015625, 64.0,
};
inline constexpr double kFilmWScale[16] = {
    8.0, 16.0, 4.0, 32.0, 8.0, 2.0, 8.0, 8.0,
    8192.0, 4096.0, 4096.0, 32768.0, 32768.0, 32768.0, 32768.0, 8.0,
};
inline constexpr int kFilmQShift = 9;  // 2^9 = 512 product scale, all j
// FiLM output int8 counts carry beta at x16 (out = gamma*x + beta*16).

// Per-layer INT8 requant right-shifts (tuned; acc magnitudes stay tiny).
inline constexpr int kShiftL1 = 5;
inline constexpr int kShiftDW1 = 3;
inline constexpr int kShiftPW1 = 4;
inline constexpr int kShiftDW2 = 3;
inline constexpr int kShiftPW2 = 4;

// ------------------------------------------------------------------
// Stage 5.4: Platt temperature.
// ------------------------------------------------------------------
inline constexpr double kPlattT = 1.38;

// FP32 overflow sentinel that forces the fallback path.
inline constexpr double kNonFiniteSentinel = 1e300;

// ------------------------------------------------------------------
// Inference I/O (caller storage, no heap).
// ------------------------------------------------------------------
struct MicroNetProbs {
  double p[kNumTaxa] = {};
  int argmax = -1;
  bool usedFp32Fallback = false;
};

struct MicroNetAudit {
  int fp32_fallback_count = 0;
  int int8_runs = 0;
};

// BOOT-ONLY: deterministic biologically-aligned weights (fixed LCG seed).
// May touch filmParams only. Never throws; returns false on null arena.
bool microNetInitBoot(StaticMemoryArena& arena) noexcept;

// Test-only fault-injection hook (heap-free static flag): forces the INT8
// path to report non-finite so the FP32 fallback can be exercised.
void microNetSetForcedFallback(bool on) noexcept;

// HOT-PATH SAFE: full classify (INT8 -> finite-check -> FP32 fallback).
// Reads scratch ROI + fbsc; borrows integral as activation planes;
// reads filmParams tables. Writes audit counters.
MicroNetProbs microNetClassify(StaticMemoryArena& arena, const double fbsc[16],
                               MicroNetAudit& audit) noexcept;

// HOT-PATH SAFE: Platt-scaled softmax, T = 1.38. probs sum to 1.
void plattSoftmax(const double logits[kNumTaxa], double probs[kNumTaxa]) noexcept;

// ------------------------------------------------------------------
// Stage 6: fixed-capacity tracker (no heap).
// ------------------------------------------------------------------
inline constexpr int kMaxActiveTracks = 64;
inline constexpr int kMaxSlideTracks = 256;
inline constexpr int kMaxDetections = 8;  // Top-K gate width
inline constexpr double kIouThreshold = 0.30;
inline constexpr double kLogOddsPos = 2.197;  // ln(0.9/0.1), P >= 0.90
inline constexpr double kLogOddsNeg = -2.197;
inline constexpr double kUnobservedDecay = 0.10;
inline constexpr double kSlideConfMin = 0.70;

enum TrackStatus : int {
  kTrackActive = 0,
  kTrackVerified = 1,  // VERIFIED_POSITIVE_EGG
  kTrackRejected = 2,  // REJECTED_DEBRIS_ARTIFACT
};

struct Detection {
  float x0 = 0.0f, y0 = 0.0f, x1 = -1.0f, y1 = -1.0f;
  int cls = -1;
  double conf = 0.0;
};

struct Track {
  bool active = false;
  int id = -1;
  int cls = -1;
  float x0 = 0.0f, y0 = 0.0f, x1 = -1.0f, y1 = -1.0f;
  float vx = 0.0f, vy = 0.0f;
  double logOdds = 0.0;
  int unseen = 0;
  int hits = 0;
  TrackStatus status = kTrackActive;
};

struct TrackTable {
  Track tracks[kMaxActiveTracks];
  int nextId = 0;
};

struct SlideTrack {
  bool used = false;
  int id = -1;
  int cls = -1;
  double conf = 0.0;
};

struct SlideDB {
  SlideTrack slots[kMaxSlideTracks];
  int tally[kNumTaxa] = {};
};

// IoU of two bboxes. HOT-PATH SAFE.
double bboxIoU(float ax0, float ay0, float ax1, float ay1, float bx0,
               float by0, float bx1, float by1) noexcept;

// Rectangular Hungarian assignment on a static cost matrix
// (nTracks x nDets, nTracks <= 64, nDets <= 8). assign[t] = detection
// index or -1. Unmatched detections listed in newDet[] (*nNew <= 8).
// HOT-PATH SAFE (O(n^2*m) worst case on tiny inputs, no heap).
void hungarianMatch(const double* cost, int nTracks, int nDets, int* assign,
                    int* newDet, int* nNew) noexcept;

// One tracker step: match -> log-odds update/decay -> status. HOT-PATH SAFE.
void trackerStep(TrackTable& tt, const Detection* dets, int nDets) noexcept;

// Log-odds helpers. HOT-PATH SAFE.
inline double clampConf(double p) noexcept {
  if (p < 0.01) return 0.01;
  if (p > 0.99) return 0.99;
  return p;
}
inline double logOddsOf(double p) noexcept {
  const double c = clampConf(p);
  // ln via standard lib (never allocates).
  return std::log(c / (1.0 - c));
}

// Whole-slide: commit verified tracks (P >= 0.70), dedup by id. HOT-PATH.
void slideCommit(SlideDB& db, const TrackTable& tt) noexcept;

}  // namespace panparasite
