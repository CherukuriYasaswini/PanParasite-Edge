#pragma once

// PanParasite-Edge Phase 4: Stage A Micro-Logistic Gate + Occlusion Gating
// (4.1) + Edge-Preserving Perona-Malik Anisotropic Diffusion PDE (4.2).
//
// OPENCODE.md Rules enforced:
//   Rule 2: HOT-PATH SAFE only — raw arena pointers, single-pass loops,
//           no malloc/calloc/new/vector::push_back/cv::Mat::create,
//           no temporary cv::Mat. Every function below is noexcept.
//   Stage A (gate): Score_egg = sigma(w_0 + sum_{i=1..16} w_i*f_BSC[i]),
//            tau = 0.38, Top K <= 8, audit flag CANDIDATE_CAP_EXCEEDED.
//   Stage 4.1: O_score = 1 - A_contour/A_hull + 0.25*(Debris_1.5x/A_ROI),
//            Active Perception iff O_score > 0.65.
//   Stage 4.2: 3x 4-neighbor Perona-Malik, Cauchy g = 1/(1+(|n|/k)^2),
//            kappa = 18.0, lambda = 0.20 (<= 0.25 stability bound).
//
// Buffer discipline (all inside StaticMemoryArena, no heap):
//   Gate reads TopoCandidate.fbsc (caller storage, no arena traffic).
//   Occlusion reads mask + equivMap (compact fg ids from Stage 3).
//   ROI extract reads frameB (locked grayscale frame).
//   Diffusion runs IN-PLACE (Gauss-Seidel) in the 128x128 scratch ROI —
//   no second plane is needed, so equivMap labels stay intact.
//   NOTE: one shared scratch plane => process one candidate at a time
//   (extract -> diffuse -> classify) in production; the batch driver below
//   is a test-friendly form and leaves the LAST triggered ROI in scratch.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "memory_arena.hpp"
#include "topology_engine.hpp"

namespace panparasite {

// ------------------------------------------------------------------
// Stage A gate constants.
// ------------------------------------------------------------------
inline constexpr double kGateTau = 0.38;
inline constexpr int kTopK = 8;
inline constexpr const char* kCapAuditFlag = "CANDIDATE_CAP_EXCEEDED";

// ------------------------------------------------------------------
// Calibrated default weights w_0..w_16 (w_0 bias, w_{j+1} for fbsc[j]).
// Calibration rationale (logit(tau=0.38) = -0.49):
//   Prior favors area-prefiltered Stage-3 survivors (w_0 = +4.0); strong
//   negatives on Beta_1 (w_2, trichuris Beta_1=2 costs -0.8 yet still
//   passes) and R_hole (w_3 = -5.0: R=0.7 costs -3.5); solidity rewarded
//   (w_5 = +2.5), spikiness penalized (w_6 = -3.0, w_16 = -1.2 on P/P_hull).
//   Normalized-moment weights are near-zero neutrals (those features span
//   1e2..1e4 px powers and must not dominate topology terms).
// Spot-checks: plain egg z ~= +2.1 (0.89); trichuris z ~= +0.3 (0.57);
//   chaotic debris (Beta_1=4, R=0.72) z ~= -9.7 (~0.0). All verified in
//   tests/test_active_perception.cpp.
// ------------------------------------------------------------------
inline constexpr double kDefaultLogisticW[17] = {
    4.0,     // w_0 bias
    -1.5,    // w_1  Beta_0 (==1 favored; >=2 penalized extra)
    -0.4,    // w_2  Beta_1
    -5.0,    // w_3  R_hole
    -0.2,    // w_4  P_ext/sqrt(A_ext)
    2.5,     // w_5  A_ext/A_hull (solidity rewarded)
    -3.0,    // w_6  d_max/sqrt(A_ext)
    0.1,     // w_7  cos(theta_folded)
    0.1,     // w_8  sin(theta_folded)
    -0.0004,   // w_9  mu20'
    -0.0004,   // w_10 mu02'
    -0.0002,   // w_11 mu11'
    -0.00002,  // w_12 mu30'
    -0.00002,  // w_13 mu12'
    -0.00002,  // w_14 mu21'
    -0.00002,  // w_15 mu03'
    -1.2,    // w_16 P_ext/P_hull
};

inline double sigmoid(double z) noexcept {
  // std::exp never allocates; branches guard overflow (|z| <= ~30 here).
  if (z >= 0.0) {
    const double e = std::exp(-z);
    return 1.0 / (1.0 + e);
  }
  const double e = std::exp(z);
  return e / (1.0 + e);
}

// Score_egg for one 16-D signature. HOT-PATH SAFE.
inline double evalLogisticScore(const double fbsc[16],
                               const double w[17] = kDefaultLogisticW) noexcept {
  double z = w[0];
  for (int j = 0; j < 16; ++j) z += w[j + 1] * fbsc[j];
  return sigmoid(z);
}

struct GateDecision {
  double score = 0.0;
  bool pass = false;  // score >= tau
};

// In-place Top-K field progation: passing indices sorted by score desc,
// capped at 8. HOT-PATH SAFE (insertion sort over <= 64, no alloc).
struct GatedField {
  int idx[kTopK] = {};
  double scores[kTopK] = {};
  int n = 0;
  bool capExceeded = false;  // CANDIDATE_CAP_EXCEEDED audit flag
};

void runLogisticGate(const TopoResult& topo, GatedField& out,
                     const double w[17] = kDefaultLogisticW) noexcept;
inline GateDecision evalCandidateGate(const TopoCandidate& c,
                                      const double w[17] = kDefaultLogisticW) noexcept {
  const double s = evalLogisticScore(c.fbsc, w);
  return GateDecision{s, s >= kGateTau};
}

// ------------------------------------------------------------------
// Stage 4.1 occlusion gating.
// ROI = 1.5x expanded candidate bbox (clamped). Debris_1.5x counts ROI
// pixels with mask==255 that are NOT this candidate (other components +
// filtered debris, both visible in mask but carrying a foreign/zero
// compact id in equivMap). HOT-PATH SAFE.
// ------------------------------------------------------------------
inline constexpr double kOcclusionThreshold = 0.65;
inline constexpr double kOcclusionDebrisW = 0.25;

struct BBox {
  int x0 = 0, y0 = 0, x1 = -1, y1 = -1;
  bool valid = false;
};

bool candidateBBox(uint32_t compactId, const uint32_t* equivMap, int w, int h,
                   BBox& box) noexcept;
double computeOcclusion(const TopoCandidate& c, int candIdx,
                        const StaticMemoryArena& arena) noexcept;
inline bool needsActivePerception(double oScore) noexcept {
  return oScore > kOcclusionThreshold;
}

// ------------------------------------------------------------------
// Stage 4.2 Perona-Malik diffusion on the 128x128 scratch ROI.
// extractCandidateROI resamples the frameB bbox region (nearest neighbor,
// clamped) into scratch. diffuseScratchROI runs `iters` in-place
// Gauss-Seidel iterations (4-neighbor, Cauchy flux, lambda <= 0.25).
// HOT-PATH SAFE.
// ------------------------------------------------------------------
inline constexpr int kPMIters = 3;
inline constexpr double kPMKappa = 18.0;
inline constexpr double kPMLambda = 0.20;
static_assert(kPMLambda <= 0.25, "explicit 4-neighbor stability bound");

void extractCandidateROI(StaticMemoryArena& arena, const BBox& box) noexcept;
void diffuseScratchROI(StaticMemoryArena& arena, int iters = kPMIters) noexcept;

struct PerceptionRecord {
  int candIdx = -1;
  double oScore = 0.0;
  bool triggered = false;
};

// Batch driver (test-friendly): gates -> occlusion -> conditional diffuse.
// Returns record count (<= K). Scratch ends holding the LAST triggered
// candidate's diffused ROI (or is untouched if none triggered).
int runActivePerception(StaticMemoryArena& arena, const TopoResult& topo,
                        const GatedField& gated, PerceptionRecord* records,
                        int recordCap) noexcept;

}  // namespace panparasite
