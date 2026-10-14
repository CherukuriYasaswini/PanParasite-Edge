// PanParasite-Edge Phase 2 implementation: Stage 1 + Stage 2.
// HOT-PATH translation unit: raw loops over StaticMemoryArena buffers only.
// Forbidden here: malloc/calloc/new/vector::push_back/cv::Mat::create,
// temporary cv::Mat, <vector>/<string> (heap). Allowed: <cstdint>/<cstddef>
// scalars, int64 accumulators, arena pointers, constexpr kernel table.

#include "preprocessor.hpp"

#include <cstddef>
#include <cstdint>

namespace panparasite {

// ------------------------------------------------------------------
// Stage 1.
// ------------------------------------------------------------------
double meanInterFrameDelta(const uint8_t* a, const uint8_t* b,
                           std::size_t n) noexcept {
  if (a == nullptr || b == nullptr || n == 0) return 0.0;
  uint64_t sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const int d = static_cast<int>(a[i]) - static_cast<int>(b[i]);
    sum += static_cast<uint64_t>(d < 0 ? -d : d);
  }
  return static_cast<double>(sum) / static_cast<double>(n);
}

double laplacianFocusVariance(const uint8_t* frame, int w, int h) noexcept {
  if (frame == nullptr || w < 3 || h < 3) return 0.0;
  int64_t sum = 0;
  uint64_t sumsq = 0;
  uint64_t n = 0;
  for (int y = 1; y < h - 1; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(w);
    for (int x = 1; x < w - 1; ++x) {
      const std::size_t i = row + static_cast<std::size_t>(x);
      const int c = static_cast<int>(frame[i]);
      const int lap = static_cast<int>(frame[i - 1]) + static_cast<int>(frame[i + 1]) +
                      static_cast<int>(frame[i - w]) + static_cast<int>(frame[i + w]) - 4 * c;
      sum += static_cast<int64_t>(lap);
      sumsq += static_cast<uint64_t>(static_cast<int64_t>(lap) * static_cast<int64_t>(lap));
      ++n;
    }
  }
  if (n == 0) return 0.0;
  const double mean = static_cast<double>(sum) / static_cast<double>(n);
  const double e2 = static_cast<double>(sumsq) / static_cast<double>(n);
  const double var = e2 - mean * mean;
  return (var < 0.0 && var > -1e-9) ? 0.0 : var;
}

AcquisitionMetrics checkAcquisitionGate(StaticMemoryArena& arena,
                                        AcquisitionGate& gate) noexcept {
  AcquisitionMetrics m;
  m.delta_it = meanInterFrameDelta(arena);
  m.phi_focus = laplacianFocusVariance(arena);
  m.delta_pass = m.delta_it < kDeltaIThreshold;
  m.focus_pass = m.phi_focus >= kFocusPhiThreshold;
  m.pair_pass = m.delta_pass && m.focus_pass;
  if (m.pair_pass) {
    if (gate.consecutive_passes < kGateConsecutiveFrames) ++gate.consecutive_passes;
  } else {
    gate.consecutive_passes = 0;
  }
  gate.locked = gate.consecutive_passes >= kGateConsecutiveFrames;
  return m;
}

// ------------------------------------------------------------------
// Stage 2: integral image (row-running-sum, single pass).
// ------------------------------------------------------------------
void computeIntegralImage(const uint8_t* src, uint32_t* integral, int w,
                          int h) noexcept {
  if (src == nullptr || integral == nullptr || w <= 0 || h <= 0) return;
  const std::size_t stride = static_cast<std::size_t>(w) + 1u;
  // Zero border row + column.
  for (int x = 0; x <= w; ++x) integral[static_cast<std::size_t>(x)] = 0u;
  for (int y = 1; y <= h; ++y) integral[static_cast<std::size_t>(y) * stride] = 0u;
  for (int y = 0; y < h; ++y) {
    const std::size_t srcRow = static_cast<std::size_t>(y) * static_cast<std::size_t>(w);
    const std::size_t iiRow = (static_cast<std::size_t>(y) + 1u) * stride;
    const std::size_t iiPrev = static_cast<std::size_t>(y) * stride;
    uint32_t rowsum = 0;
    for (int x = 0; x < w; ++x) {
      rowsum += static_cast<uint32_t>(src[srcRow + static_cast<std::size_t>(x)]);
      integral[iiRow + static_cast<std::size_t>(x) + 1u] =
          integral[iiPrev + static_cast<std::size_t>(x) + 1u] + rowsum;
    }
  }
}

// ------------------------------------------------------------------
// Stage 2: adaptive binarization.
// ------------------------------------------------------------------
void adaptiveBinarizeRange(const uint8_t* src, const uint32_t* integral,
                           uint8_t* mask, int w, int h, int x0, int y0, int x1,
                           int y1) noexcept {
  if (src == nullptr || integral == nullptr || mask == nullptr) return;
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 > w) x1 = w;
  if (y1 > h) y1 = h;
  if (x0 >= x1 || y0 >= y1) return;
  const int stride = w + 1;
  for (int y = y0; y < y1; ++y) {
    const std::size_t srcRow = static_cast<std::size_t>(y) * static_cast<std::size_t>(w);
    int wx0 = 0, wx1 = 0, wy0 = 0, wy1 = 0;
    wy0 = y - kAdaptiveRadiusRw;
    if (wy0 < 0) wy0 = 0;
    wy1 = y + kAdaptiveRadiusRw;
    if (wy1 > h - 1) wy1 = h - 1;
    for (int x = x0; x < x1; ++x) {
      wx0 = x - kAdaptiveRadiusRw;
      if (wx0 < 0) wx0 = 0;
      wx1 = x + kAdaptiveRadiusRw;
      if (wx1 > w - 1) wx1 = w - 1;
      const uint32_t area =
          static_cast<uint32_t>(wx1 - wx0 + 1) * static_cast<uint32_t>(wy1 - wy0 + 1);
      const uint32_t sum = rectSum(integral, stride, wx0, wy0, wx1, wy1);
      const uint32_t pix = static_cast<uint32_t>(src[srcRow + static_cast<std::size_t>(x)]);
      // pix < (sum/area)*0.88  <=>  pix*area*25 < sum*22 (exact, no float).
      const uint32_t lhs = pix * area * static_cast<uint32_t>(kThreshDen);
      const uint32_t rhs = sum * static_cast<uint32_t>(kThreshNum);
      mask[srcRow + static_cast<std::size_t>(x)] = (lhs < rhs) ? 255u : 0u;
    }
  }
}

void adaptiveBinarize(const uint8_t* src, const uint32_t* integral,
                      uint8_t* mask, int w, int h) noexcept {
  adaptiveBinarizeRange(src, integral, mask, w, h, 0, 0, w, h);
}

// ------------------------------------------------------------------
// Stage 2: 11x11 elliptical closing via 0/1 scratch.
// ------------------------------------------------------------------
void dilateMaskRange(const uint8_t* srcMask, uint32_t* tmp, int w, int h,
                     int x0, int y0, int x1, int y1) noexcept {
  if (srcMask == nullptr || tmp == nullptr) return;
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 > w) x1 = w;
  if (y1 > h) y1 = h;
  if (x0 >= x1 || y0 >= y1) return;
  const int taps = kEllipse11x11.count;
  for (int y = y0; y < y1; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(w);
    for (int x = x0; x < x1; ++x) {
      uint32_t hit = 0;
      for (int t = 0; t < taps; ++t) {
        int nx = x + static_cast<int>(kEllipse11x11.taps[static_cast<std::size_t>(t)].dx);
        int ny = y + static_cast<int>(kEllipse11x11.taps[static_cast<std::size_t>(t)].dy);
        if (nx < 0) nx = 0;
        else if (nx > w - 1) nx = w - 1;
        if (ny < 0) ny = 0;
        else if (ny > h - 1) ny = h - 1;
        if (srcMask[static_cast<std::size_t>(ny) * static_cast<std::size_t>(w) +
                    static_cast<std::size_t>(nx)] == 255u) {
          hit = 1;
          break;
        }
      }
      tmp[row + static_cast<std::size_t>(x)] = hit;
    }
  }
}

void erodeMaskRange(const uint32_t* tmp, uint8_t* dstMask, int w, int h,
                    int x0, int y0, int x1, int y1) noexcept {
  if (tmp == nullptr || dstMask == nullptr) return;
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 > w) x1 = w;
  if (y1 > h) y1 = h;
  if (x0 >= x1 || y0 >= y1) return;
  const int taps = kEllipse11x11.count;
  for (int y = y0; y < y1; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(w);
    for (int x = x0; x < x1; ++x) {
      uint8_t out = 255u;
      for (int t = 0; t < taps; ++t) {
        int nx = x + static_cast<int>(kEllipse11x11.taps[static_cast<std::size_t>(t)].dx);
        int ny = y + static_cast<int>(kEllipse11x11.taps[static_cast<std::size_t>(t)].dy);
        if (nx < 0) nx = 0;
        else if (nx > w - 1) nx = w - 1;
        if (ny < 0) ny = 0;
        else if (ny > h - 1) ny = h - 1;
        if (tmp[static_cast<std::size_t>(ny) * static_cast<std::size_t>(w) +
                 static_cast<std::size_t>(nx)] == 0u) {
          out = 0u;
          break;
        }
      }
      dstMask[row + static_cast<std::size_t>(x)] = out;
    }
  }
}

void dilateMask11x11(const uint8_t* srcMask, uint32_t* tmp, int w,
                     int h) noexcept {
  dilateMaskRange(srcMask, tmp, w, h, 0, 0, w, h);
}

void erodeMask11x11(const uint32_t* tmp, uint8_t* dstMask, int w,
                    int h) noexcept {
  erodeMaskRange(tmp, dstMask, w, h, 0, 0, w, h);
}

void closeMask11x11Elliptical(StaticMemoryArena& arena) noexcept {
  dilateMask11x11(arena.mask(), arena.equivMap(), kFrameWidth, kFrameHeight);
  erodeMask11x11(arena.equivMap(), arena.mask(), kFrameWidth, kFrameHeight);
}

// ------------------------------------------------------------------
// Dual-path driver.
// ------------------------------------------------------------------
void runStage2FullFrame(StaticMemoryArena& arena) noexcept {
  computeIntegralImage(arena);
  adaptiveBinarize(arena);
  closeMask11x11Elliptical(arena);
}

void runStage2Tiled32(StaticMemoryArena& arena) noexcept {
  // Integral stays global (exact local means); traversal is tile-major.
  computeIntegralImage(arena.frameB(), arena.integral(), kFrameWidth, kFrameHeight);
  const uint8_t* src = arena.frameB();
  const uint32_t* ii = arena.integral();
  uint8_t* mask = arena.mask();
  uint32_t* tmp = arena.equivMap();
  // Pass 1: tiled binarize (M_rim).
  for (int ty = 0; ty < kFrameHeight; ty += kTileHeight) {
    int y1 = ty + kTileHeight;
    if (y1 > kFrameHeight) y1 = kFrameHeight;
    for (int tx = 0; tx < kFrameWidth; tx += kTileWidth) {
      int x1 = tx + kTileWidth;
      if (x1 > kFrameWidth) x1 = kFrameWidth;
      adaptiveBinarizeRange(src, ii, mask, kFrameWidth, kFrameHeight, tx, ty, x1, y1);
    }
  }
  // Pass 2: tiled dilate into tmp (halo reads are global + clamped, so
  // the 32 px budget covers the 5 px closing radius with margin).
  for (int ty = 0; ty < kFrameHeight; ty += kTileHeight) {
    int y1 = ty + kTileHeight;
    if (y1 > kFrameHeight) y1 = kFrameHeight;
    for (int tx = 0; tx < kFrameWidth; tx += kTileWidth) {
      int x1 = tx + kTileWidth;
      if (x1 > kFrameWidth) x1 = kFrameWidth;
      dilateMaskRange(mask, tmp, kFrameWidth, kFrameHeight, tx, ty, x1, y1);
    }
  }
  // Pass 3: tiled erode back into mask (M_body).
  for (int ty = 0; ty < kFrameHeight; ty += kTileHeight) {
    int y1 = ty + kTileHeight;
    if (y1 > kFrameHeight) y1 = kFrameHeight;
    for (int tx = 0; tx < kFrameWidth; tx += kTileWidth) {
      int x1 = tx + kTileWidth;
      if (x1 > kFrameWidth) x1 = kFrameWidth;
      erodeMaskRange(tmp, mask, kFrameWidth, kFrameHeight, tx, ty, x1, y1);
    }
  }
}

}  // namespace panparasite
