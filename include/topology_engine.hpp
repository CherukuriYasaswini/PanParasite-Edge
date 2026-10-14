#pragma once

// PanParasite-Edge Phase 3: Stage 3 Dual-Connectivity Union-Find Homology
// Engine (3.1) + Directional Convexity Defect & Folded Spine Angle (3.2) +
// Scale-Invariant Moments & 16-D Betti Signature Vector (3.3).
//
// OPENCODE.md Rules enforced:
//   Rule 2: HOT-PATH SAFE only — raw arena pointers, single-pass loops,
//           no malloc/calloc/new/vector::push_back/cv::Mat::create,
//           no temporary cv::Mat. Every function below is noexcept.
//   Rule 3: Fixed 1920x1080 uint8 grayscale, S = 0.620 um/pixel.
//   Stage 3: 8-connected foreground / 4-connected background Union-Find
//            homology (Beta_0, Beta_1, R_hole, theta_folded, f_BSC[16]).
//
// Buffer discipline (all inside StaticMemoryArena, no heap):
//   mask      : read-only binarized input (M_body from Stage 2).
//   equivMap  : foreground label plane (W stride), remapped in place to
//               compact ids (0 = background, 1..F = candidate index+1).
//   integral  : BORROWED as background label plane (stride 1921, rows
//               0..1079, cols 0..1919). Stage 2 must rerun before its
//               integral values are needed again.
//   dsu[32768]: Union-Find arena, reset between fg and bg passes
//               (provisional labels 1..32767, 0 = background).
//   scratch   : hull workspace as packed int16 (x,y) pairs:
//               slots [0,2048) boundary input, [2048,4096) hull output.
//   frameA/B, FiLM blob, DIO sector untouched.
// caller storage:
//   TopoResult (~25 KB) must live in static/global or generous-thread
//   stack memory — never on a small embedded stack. No heap either way.

#include <cstddef>
#include <cstdint>

#include "memory_arena.hpp"

namespace panparasite {

// ------------------------------------------------------------------
// Stage 3.1 biological area bounds (px^2 at 0.620 um/px).
// 1000 px = 384 um^2, 35000 px = 13454 um^2.
// ------------------------------------------------------------------
inline constexpr uint32_t kMinBioAreaPx = 1000u;
inline constexpr uint32_t kMaxBioAreaPx = 35000u;
inline constexpr double kBioAreaUm2PerPx =
    kMicronsPerPixel * kMicronsPerPixel;  // 0.3844

// Candidate / hull capacity (fixed, no heap).
inline constexpr int kMaxTopoCandidates = 64;
inline constexpr int kMaxHullInputPoints = 2048;
inline constexpr int kScratchPointCap = 4096;  // 16384 B / 4 B per (x,y)

// Species spine thresholds.
inline constexpr double kHaemDmaxUm = 14.0;
inline constexpr double kHaemDmaxPx = 14.0 / kMicronsPerPixel;  // 22.58
inline constexpr double kHaemThetaLoDeg = 0.0;
inline constexpr double kHaemThetaHiDeg = 20.0;
inline constexpr double kMansDmaxUm = 12.0;
inline constexpr double kMansDmaxPx = 12.0 / kMicronsPerPixel;  // 19.35
inline constexpr double kMansThetaLoDeg = 65.0;
inline constexpr double kMansThetaHiDeg = 90.0;

// Background label plane stride (borrowed integral buffer).
inline constexpr int kBgStride = kIntegralWidth;  // 1921

// ------------------------------------------------------------------
// Union-Find primitives over DisjointSetNode[32768]. HOT-PATH SAFE.
// findRoot uses iterative path halving; unionLabels uses rank + size.
// ------------------------------------------------------------------
uint32_t findRoot(DisjointSetNode* dsu, uint32_t x) noexcept;
uint32_t unionLabels(DisjointSetNode* dsu, uint32_t a, uint32_t b) noexcept;

// Single-pass raster labeling. Returns provisional label high-water mark
// (labels live in 1..32767); sets *overflow if capacity exhausts (extra
// pixels degrade to background 0 — never crashes, never allocates).
// HOT-PATH SAFE.
uint32_t labelForeground8(const uint8_t* mask, uint32_t* labels,
                          DisjointSetNode* dsu, int w, int h,
                          bool* overflow) noexcept;
uint32_t labelBackground4(const uint8_t* mask, uint32_t* bgPlane, int bgStride,
                          DisjointSetNode* dsu, int w, int h,
                          bool* overflow) noexcept;

// ------------------------------------------------------------------
// Per-candidate topology + morphometrics.
// f_BSC layout:
//  [0]=Beta_0(1) [1]=Beta_1 [2]=R_hole [3]=P_ext/sqrt(A_ext)
//  [4]=A_ext/A_hull [5]=d_max/sqrt(A_ext) [6]=cos(theta_folded)
//  [7]=sin(theta_folded) [8]=mu20' [9]=mu02' [10]=mu11' [11]=mu30'
//  [12]=mu12' [13]=mu21' [14]=mu03' [15]=P_ext/P_hull
// Normalized moments use sigma = sqrt(mu00):
//  mu20'=mu20/mu00, mu02'=mu02/mu00, mu11'=mu11/mu00,
//  mu30'=mu30/mu00^1.5, mu12'=mu12/mu00^1.5, mu21'=mu21/mu00^1.5,
//  mu03'=mu03/mu00^1.5.
// ------------------------------------------------------------------
struct TopoCandidate {
  uint32_t rootLabel = 0;   // fg provisional root (pre-compaction id)
  uint32_t areaExt = 0;     // A_ext, px
  double areaUm2 = 0.0;
  int beta0 = 1;
  int beta1 = 0;
  int chi = 1;              // 1 - beta1
  double rHole = 0.0;
  uint32_t holeArea = 0;

  double xbar = 0.0, ybar = 0.0, mu00 = 0.0;
  double mu20 = 0.0, mu02 = 0.0, mu11 = 0.0;
  double mu30 = 0.0, mu12 = 0.0, mu21 = 0.0, mu03 = 0.0;
  double mu20n = 0.0, mu02n = 0.0, mu11n = 0.0;
  double mu30n = 0.0, mu12n = 0.0, mu21n = 0.0, mu03n = 0.0;

  double thetaMajor = 0.0;  // rad, inertial axis
  double vmajX = 1.0, vmajY = 0.0;
  double dmaxPx = 0.0, dmaxUm = 0.0;
  double xd = 0.0, yd = 0.0;  // deepest defect location
  double vdefX = 0.0, vdefY = 0.0;
  double thetaFoldedDeg = 0.0;  // [0,90]

  double pExt = 0.0;    // boundary pixel count
  double aHull = 0.0;   // shoelace hull area, px^2
  double pHull = 0.0;   // hull perimeter, px
  bool hullTruncated = false;

  double fbsc[16] = {};

  // Morphological differentiators (spec §3.2).
  [[nodiscard]] bool isTrichurisLike() const noexcept { return beta1 == 2; }
  [[nodiscard]] bool isMansoniLike() const noexcept {
    return dmaxPx >= kMansDmaxPx && thetaFoldedDeg >= kMansThetaLoDeg &&
           thetaFoldedDeg <= kMansThetaHiDeg;
  }
  [[nodiscard]] bool isHaematobiumLike() const noexcept {
    return dmaxPx >= kHaemDmaxPx && thetaFoldedDeg >= kHaemThetaLoDeg &&
           thetaFoldedDeg <= kHaemThetaHiDeg;
  }
};

struct TopoResult {
  TopoCandidate candidates[kMaxTopoCandidates];
  int numCandidates = 0;
  int globalBeta0 = 0;
  int globalBeta1 = 0;
  int globalChi = 0;  // Beta_0 - Beta_1
  bool overflow = false;      // label exhaustion or candidate/hull cap hit
  uint32_t fgProvisional = 0;
  uint32_t bgProvisional = 0;
};

// Full Stage 3 pipeline on arena buffers. HOT-PATH SAFE.
// Reads mask; writes equivMap (compact fg ids), integral (bg labels),
// dsu (bg pass state — fg state is materialized into equivMap first),
// scratch (per-candidate hull workspace). Returns candidate count.
int analyzeTopology(StaticMemoryArena& arena, TopoResult& out) noexcept;

}  // namespace panparasite
