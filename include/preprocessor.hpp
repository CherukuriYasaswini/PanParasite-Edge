#pragma once

// PanParasite-Edge Phase 2: Stage 1 (Acquisition & Focus Gate) + Stage 2
// (Full-Frame Integral Adaptive Preprocessing Engine).
//
// OPENCODE.md Rules enforced:
//   Rule 2: HOT-PATH SAFE only — raw arena pointers, single-pass loops,
//           no malloc/calloc/new/vector::push_back/cv::Mat::create,
//           no temporary cv::Mat. Every function below is noexcept.
//   Rule 3: Fixed 1920x1080 uint8 grayscale, S = 0.620 um/pixel.
//   Stage 1: Delta I_t < 1.80, Phi_focus (3x3 Laplacian variance) >= 125.0,
//            lock across 3 consecutive frames. Flicker helper gives valid
//            T_int = k/(2*f_line): 50 Hz -> 20.0 ms (k=2),
//            60 Hz -> 33.33 ms (k=4).
//   Stage 2: Single-pass integral (O(1)/px) into uint32_t[1921*1081],
//            adaptive binarization R_w = 31 px (63x63 window ~= 39 um
//            Ascaris shell at 0.620 um/px: diameter 63*0.620 = 39.06 um),
//            C_thresh = 0.88 (exact integer form 22/25):
//              B(x,y) = 255 if I(x,y)*area*25 < sum_w*22 else 0,
//            11x11 elliptical closing (circle r=5, dx^2+dy^2<=25) with
//            zero extra allocation (equivMap reused as 0/1 scratch),
//            plus Dual-Path 32 px overlap tiling benchmark toggle.
//
// Buffer discipline (all inside StaticMemoryArena, no heap):
//   Stage 1 reads  frameA (ping) + frameB (pong, latest).
//   Stage 2 reads  frameB (locked frame), writes integral, writes mask
//           (M_rim then overwritten in place by M_body), borrows equivMap
//           as 0/1 dilate scratch (Stage 3 overwrites it afterwards).
//   frameA is never touched by Stage 2 (free as test save area).
//   FiLM blob / DIO sector / DSU arena untouched.

#include <array>
#include <cstddef>
#include <cstdint>

#include "memory_arena.hpp"

namespace panparasite {

// ------------------------------------------------------------------
// Stage 1 thresholds.
// ------------------------------------------------------------------
inline constexpr double kDeltaIThreshold = 1.80;
inline constexpr double kFocusPhiThreshold = 125.0;
inline constexpr int kGateConsecutiveFrames = 3;

// ------------------------------------------------------------------
// Stage 2 constants.
// ------------------------------------------------------------------
inline constexpr int kAdaptiveRadiusRw = 31;
inline constexpr int kAdaptiveWindow = 2 * kAdaptiveRadiusRw + 1;  // 63
inline constexpr double kThreshSensitivity = 0.88;  // exact ratio 22/25
inline constexpr int kThreshNum = 22;
inline constexpr int kThreshDen = 25;
inline constexpr int kClosingDiameter = 11;
inline constexpr int kClosingRadius = 5;
inline constexpr int kTileOverlap = 32;  // halo budget (>= closing radius 5)
inline constexpr int kTileWidth = 480;   // 1920/480 = 4 columns
inline constexpr int kTileHeight = 270;  // 1080/270 = 4 rows
static_assert(kTileWidth * 4 == kFrameWidth, "tiles must cover width exactly");
static_assert(kTileHeight * 4 == kFrameHeight, "tiles must cover height exactly");

// ------------------------------------------------------------------
// AC mains flicker cancellation helper: T_int = k / (2 * f_line).
// Canonical: 50 Hz/k=2 -> 20.0 ms; 60 Hz/k=4 -> 33.333 ms.
// HOT-PATH SAFE (constexpr, no allocation).
// ------------------------------------------------------------------
constexpr double flickerTintSeconds(double f_line_hz, int k) noexcept {
  return static_cast<double>(k) / (2.0 * f_line_hz);
}
constexpr double flickerTintMs(double f_line_hz, int k) noexcept {
  return 1000.0 * flickerTintSeconds(f_line_hz, k);
}

struct FlickerExposure {
  double line_freq_hz = 50.0;
  int k = 2;
  double t_int_ms = 20.0;
};

constexpr FlickerExposure recommendedFlickerExposure(double f_line_hz) noexcept {
  // Nearest-mains selection: <=55 Hz -> 50 Hz family (k=2), else 60 Hz (k=4).
  if (f_line_hz <= 55.0) {
    return FlickerExposure{f_line_hz, 2, flickerTintMs(f_line_hz, 2)};
  }
  return FlickerExposure{f_line_hz, 4, flickerTintMs(f_line_hz, 4)};
}

constexpr bool isFlickerCompliant(double t_int_ms, double f_line_hz,
                                  double tol_ms = 0.5) noexcept {
  const FlickerExposure rec = recommendedFlickerExposure(f_line_hz);
  const double diff = t_int_ms >= rec.t_int_ms ? t_int_ms - rec.t_int_ms
                                               : rec.t_int_ms - t_int_ms;
  return diff <= tol_ms;
}

// ------------------------------------------------------------------
// Acquisition gate state + metrics. HOT-PATH SAFE.
// ------------------------------------------------------------------
struct AcquisitionMetrics {
  double delta_it = 0.0;   // mean |A-B|
  double phi_focus = 0.0;  // Laplacian variance on frameB
  bool delta_pass = false;
  bool focus_pass = false;
  bool pair_pass = false;
};

struct AcquisitionGate {
  int consecutive_passes = 0;
  bool locked = false;
  void reset() noexcept {
    consecutive_passes = 0;
    locked = false;
  }
};

// Mean absolute inter-frame difference over n pixels. HOT-PATH SAFE.
double meanInterFrameDelta(const uint8_t* a, const uint8_t* b,
                           std::size_t n) noexcept;
inline double meanInterFrameDelta(const StaticMemoryArena& arena) noexcept {
  return meanInterFrameDelta(arena.frameA(), arena.frameB(), kFramePixels);
}

// Variance of 3x3 cross Laplacian (L = N+S+E+W-4C) over interior
// [1,w-2]x[1,h-2] (1 px border skipped, no branches). Single pass with
// int64 sum/sumsq, no temp buffer. HOT-PATH SAFE.
double laplacianFocusVariance(const uint8_t* frame, int w, int h) noexcept;
inline double laplacianFocusVariance(const StaticMemoryArena& arena) noexcept {
  return laplacianFocusVariance(arena.frameB(), kFrameWidth, kFrameHeight);
}

// Full gate: updates consecutive counter, locks after 3 passes.
// Focus is measured on frameB (latest). HOT-PATH SAFE.
AcquisitionMetrics checkAcquisitionGate(StaticMemoryArena& arena,
                                        AcquisitionGate& gate) noexcept;

// ------------------------------------------------------------------
// Stage 2: integral image. src is WxH, integral is (W+1)x(H+1) with zero
// border. Row-running-sum form, O(1) amortized per pixel. HOT-PATH SAFE.
// Max sum 2073600*255 = 528,768,000 < 2^32: no overflow.
// ------------------------------------------------------------------
void computeIntegralImage(const uint8_t* src, uint32_t* integral, int w,
                          int h) noexcept;
inline void computeIntegralImage(StaticMemoryArena& arena) noexcept {
  computeIntegralImage(arena.frameB(), arena.integral(), kFrameWidth,
                       kFrameHeight);
}

// O(1) inclusive-rectangle sum over integral (stride = w+1).
// Caller clamps window; coordinates inclusive, 0 <= x0<=x1<w. HOT-PATH SAFE.
inline uint32_t rectSum(const uint32_t* integral, int stride, int x0, int y0,
                        int x1, int y1) noexcept {
  const std::size_t s = static_cast<std::size_t>(stride);
  const std::size_t a = static_cast<std::size_t>(y0) * s + static_cast<std::size_t>(x0);
  const std::size_t b = static_cast<std::size_t>(y0) * s + static_cast<std::size_t>(x1 + 1);
  const std::size_t c = static_cast<std::size_t>(y1 + 1) * s + static_cast<std::size_t>(x0);
  const std::size_t d = static_cast<std::size_t>(y1 + 1) * s + static_cast<std::size_t>(x1 + 1);
  return integral[d] - integral[b] - integral[c] + integral[a];
}

// ------------------------------------------------------------------
// Stage 2: adaptive binarization (M_rim).
// B=255 iff src*area*25 < sum_w*22 (exact 0.88, integer, no float).
// Window [x-31,x+31]x[y-31,y+31] clamped. HOT-PATH SAFE.
// ------------------------------------------------------------------
void adaptiveBinarize(const uint8_t* src, const uint32_t* integral,
                      uint8_t* mask, int w, int h) noexcept;
inline void adaptiveBinarize(StaticMemoryArena& arena) noexcept {
  adaptiveBinarize(arena.frameB(), arena.integral(), arena.mask(),
                   kFrameWidth, kFrameHeight);
}
// Tiled range variant: binarize half-open [x0,x1)x[y0,y1) using the GLOBAL
// integral (identical math, tile-major order for benchmark). HOT-PATH SAFE.
void adaptiveBinarizeRange(const uint8_t* src, const uint32_t* integral,
                           uint8_t* mask, int w, int h, int x0, int y0, int x1,
                           int y1) noexcept;

// ------------------------------------------------------------------
// 11x11 elliptical (r=5 disc) kernel, compile-time taps, no heap.
// ------------------------------------------------------------------
struct EllipseOffset {
  int8_t dx = 0;
  int8_t dy = 0;
};
struct EllipseKernel {
  std::array<EllipseOffset, 121> taps{};
  int count = 0;
};
consteval EllipseKernel buildEllipse11x11() {
  EllipseKernel k{};
  for (int dy = -kClosingRadius; dy <= kClosingRadius; ++dy) {
    for (int dx = -kClosingRadius; dx <= kClosingRadius; ++dx) {
      if (dx * dx + dy * dy <= kClosingRadius * kClosingRadius) {
        k.taps[static_cast<std::size_t>(k.count)] =
            EllipseOffset{static_cast<int8_t>(dx), static_cast<int8_t>(dy)};
        ++k.count;
      }
    }
  }
  return k;
}
inline constexpr EllipseKernel kEllipse11x11 = buildEllipse11x11();
static_assert(kEllipse11x11.count >= 70 && kEllipse11x11.count <= 121,
              "11x11 disc must hold ~81 taps");

// ------------------------------------------------------------------
// 11x11 elliptical closing: dilate (any 255 in disc -> 1) then erode
// (all disc 1 -> 255). Borders replicate (clamped). tmp is WxH uint32
// scratch holding 0/1 (arena.equivMap in production). HOT-PATH SAFE.
// ------------------------------------------------------------------
void dilateMask11x11(const uint8_t* srcMask, uint32_t* tmp, int w,
                     int h) noexcept;
void erodeMask11x11(const uint32_t* tmp, uint8_t* dstMask, int w,
                    int h) noexcept;
// Range (tiled) variants: read halo from GLOBAL planes, write interior
// [x0,x1)x[y0,y1) only. Halo budget kTileOverlap=32 >= radius 5. HOT-PATH SAFE.
void dilateMaskRange(const uint8_t* srcMask, uint32_t* tmp, int w, int h,
                     int x0, int y0, int x1, int y1) noexcept;
void erodeMaskRange(const uint32_t* tmp, uint8_t* dstMask, int w, int h,
                    int x0, int y0, int x1, int y1) noexcept;

// Arena convenience: mask (M_rim) -> mask (M_body) via equivMap scratch.
void closeMask11x11Elliptical(StaticMemoryArena& arena) noexcept;

// ------------------------------------------------------------------
// Dual-path Stage 2 driver for benchmarks.
// kFullFrame: single row-major sweeps. kTiled32: 4x4 tiles of 480x270
//   with 32 px halo budget (tile-major order). Both paths share the
//   global integral so outputs are bit-identical; only traversal differs.
// HOT-PATH SAFE.
// ------------------------------------------------------------------
enum class Stage2Mode : int { kFullFrame = 0, kTiled32 = 1 };

void runStage2FullFrame(StaticMemoryArena& arena) noexcept;
void runStage2Tiled32(StaticMemoryArena& arena) noexcept;
inline void runStage2(StaticMemoryArena& arena, Stage2Mode mode) noexcept {
  if (mode == Stage2Mode::kTiled32) {
    runStage2Tiled32(arena);
  } else {
    runStage2FullFrame(arena);
  }
}

}  // namespace panparasite
