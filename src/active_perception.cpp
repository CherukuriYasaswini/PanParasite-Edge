// PanParasite-Edge Phase 4 implementation: gate + occlusion + diffusion.
// HOT-PATH translation unit: raw loops over StaticMemoryArena buffers only.
// Forbidden here: malloc/calloc/new/vector::push_back/cv::Mat::create,
// temporary cv::Mat, <vector>/<string>. Allowed: fixed stack scalars,
// <cmath> (exp/sqrt, never allocates), arena pointers.

#include "active_perception.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace panparasite {

void runLogisticGate(const TopoResult& topo, GatedField& out,
                     const double w[17]) noexcept {
  out.n = 0;
  out.capExceeded = false;
  for (int i = 0; i < kTopK; ++i) {
    out.idx[i] = -1;
    out.scores[i] = 0.0;
  }
  if (w == nullptr) return;
  // Collect passers (insertion into descending order, in place).
  int passTotal = 0;
  const int n = (topo.numCandidates < 0) ? 0 : topo.numCandidates;
  const int lim = (n < kMaxTopoCandidates) ? n : kMaxTopoCandidates;
  for (int i = 0; i < lim; ++i) {
    const double s = evalLogisticScore(topo.candidates[i].fbsc, w);
    if (s < kGateTau) continue;
    ++passTotal;
    if (out.n < kTopK) {
      // Insert at sorted position (descending).
      int pos = out.n;
      while (pos > 0 && out.scores[pos - 1] < s) {
        out.idx[pos] = out.idx[pos - 1];
        out.scores[pos] = out.scores[pos - 1];
        --pos;
      }
      out.idx[pos] = i;
      out.scores[pos] = s;
      ++out.n;
    } else {
      // Field full: still counts toward the audit flag; keep Top-K by
      // replacing the tail when the newcomer outscores it.
      out.capExceeded = true;
      if (s > out.scores[kTopK - 1]) {
        int pos = kTopK - 1;
        while (pos > 0 && out.scores[pos - 1] < s) {
          out.idx[pos] = out.idx[pos - 1];
          out.scores[pos] = out.scores[pos - 1];
          --pos;
        }
        out.idx[pos] = i;
        out.scores[pos] = s;
      }
    }
  }
  if (passTotal > kTopK) out.capExceeded = true;
}

bool candidateBBox(uint32_t compactId, const uint32_t* equivMap, int w, int h,
                   BBox& box) noexcept {
  box.x0 = 0;
  box.y0 = 0;
  box.x1 = -1;
  box.y1 = -1;
  box.valid = false;
  if (compactId == 0u || equivMap == nullptr || w <= 0 || h <= 0) return false;
  int x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (int y = 0; y < h; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(w);
    for (int x = 0; x < w; ++x) {
      if (equivMap[row + static_cast<std::size_t>(x)] != compactId) continue;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
  }
  if (x1 < x0 || y1 < y0) return false;
  box.x0 = x0;
  box.y0 = y0;
  box.x1 = x1;
  box.y1 = y1;
  box.valid = true;
  return true;
}

double computeOcclusion(const TopoCandidate& c, int candIdx,
                        const StaticMemoryArena& arena) noexcept {
  const double aExt = static_cast<double>(c.areaExt);
  const double aHull = c.aHull;
  double solidityTerm = 1.0;
  if (aHull > 0.0 && aExt > 0.0) {
    double r = aExt / aHull;
    if (r > 1.0) r = 1.0;  // discrete-hull numerical guard
    solidityTerm = 1.0 - r;
  }
  // 1.5x expanded bbox ROI (clamped), debris = foreign fg in ROI.
  const uint32_t want = static_cast<uint32_t>(candIdx) + 1u;
  BBox box;
  double debrisTerm = 0.0;
  if (candidateBBox(want, arena.equivMap(), kFrameWidth, kFrameHeight, box)) {
    const int bw = box.x1 - box.x0 + 1;
    const int bh = box.y1 - box.y0 + 1;
    const int cx = box.x0 + bw / 2;
    const int cy = box.y0 + bh / 2;
    int hw = (bw * 3) / 4;  // 1.5x linear => half-extent 0.75*side
    int hh = (bh * 3) / 4;
    if (hw < 1) hw = 1;
    if (hh < 1) hh = 1;
    int rx0 = cx - hw, rx1 = cx + hw;
    int ry0 = cy - hh, ry1 = cy + hh;
    if (rx0 < 0) rx0 = 0;
    if (ry0 < 0) ry0 = 0;
    if (rx1 >= kFrameWidth) rx1 = kFrameWidth - 1;
    if (ry1 >= kFrameHeight) ry1 = kFrameHeight - 1;
    const uint8_t* mask = arena.mask();
    const uint32_t* fg = arena.equivMap();
    int64_t debris = 0;
    const int64_t roiArea =
        static_cast<int64_t>(rx1 - rx0 + 1) * static_cast<int64_t>(ry1 - ry0 + 1);
    for (int y = ry0; y <= ry1; ++y) {
      const std::size_t mrow = static_cast<std::size_t>(y) * kFrameWidth;
      for (int x = rx0; x <= rx1; ++x) {
        const std::size_t i = mrow + static_cast<std::size_t>(x);
        if (mask[i] == 255u && fg[i] != want) ++debris;
      }
    }
    if (roiArea > 0) {
      debrisTerm =
          kOcclusionDebrisW * static_cast<double>(debris) / static_cast<double>(roiArea);
    }
  }
  return solidityTerm + debrisTerm;
}

void extractCandidateROI(StaticMemoryArena& arena, const BBox& box) noexcept {
  uint8_t* dst = arena.scratchRoi();
  const uint8_t* src = arena.frameB();
  if (!box.valid) {
    for (int i = 0; i < 128 * 128; ++i) dst[i] = 0u;
    return;
  }
  const int bw = box.x1 - box.x0 + 1;
  const int bh = box.y1 - box.y0 + 1;
  for (int dy = 0; dy < 128; ++dy) {
    int sy = box.y0 + (dy * bh) / 128;
    if (sy < 0) sy = 0;
    else if (sy >= kFrameHeight) sy = kFrameHeight - 1;
    const std::size_t srow = static_cast<std::size_t>(sy) * kFrameWidth;
    uint8_t* drow = dst + static_cast<std::size_t>(dy) * 128u;
    for (int dx = 0; dx < 128; ++dx) {
      int sx = box.x0 + (dx * bw) / 128;
      if (sx < 0) sx = 0;
      else if (sx >= kFrameWidth) sx = kFrameWidth - 1;
      drow[dx] = src[srow + static_cast<std::size_t>(sx)];
    }
  }
}

void diffuseScratchROI(StaticMemoryArena& arena, int iters) noexcept {
  if (iters <= 0) return;
  if (iters > 16) iters = 16;  // runaway guard, still heap-free
  uint8_t* img = arena.scratchRoi();
  const double kappa = kPMKappa;
  const double lambda = kPMLambda;
  const double k2 = kappa * kappa;
  for (int t = 0; t < iters; ++t) {
    // In-place Gauss-Seidel sweep (row-major). Clamped (replicate) borders.
    for (int y = 0; y < 128; ++y) {
      const int yu = (y > 0) ? y - 1 : 0;
      const int yd = (y < 127) ? y + 1 : 127;
      uint8_t* row = img + static_cast<std::size_t>(y) * 128u;
      const uint8_t* rowU = img + static_cast<std::size_t>(yu) * 128u;
      const uint8_t* rowD = img + static_cast<std::size_t>(yd) * 128u;
      for (int x = 0; x < 128; ++x) {
        const int xl = (x > 0) ? x - 1 : 0;
        const int xr = (x < 127) ? x + 1 : 127;
        const double c = static_cast<double>(row[x]);
        const double nN = static_cast<double>(rowU[x]) - c;
        const double nS = static_cast<double>(rowD[x]) - c;
        const double nW = static_cast<double>(row[xl]) - c;
        const double nE = static_cast<double>(row[xr]) - c;
        // Cauchy flux limiter g = 1/(1+(|n|/k)^2), applied per arm.
        const double fN = nN / (1.0 + (nN * nN) / k2);
        const double fS = nS / (1.0 + (nS * nS) / k2);
        const double fW = nW / (1.0 + (nW * nW) / k2);
        const double fE = nE / (1.0 + (nE * nE) / k2);
        double v = c + lambda * (fN + fS + fW + fE);
        if (v < 0.0) v = 0.0;
        else if (v > 255.0) v = 255.0;
        row[x] = static_cast<uint8_t>(v + 0.5);
      }
    }
  }
}

int runActivePerception(StaticMemoryArena& arena, const TopoResult& topo,
                        const GatedField& gated, PerceptionRecord* records,
                        int recordCap) noexcept {
  if (records == nullptr || recordCap <= 0) return 0;
  int nRec = 0;
  const int n = (gated.n < 0) ? 0 : gated.n;
  const int lim = (n < kTopK) ? n : kTopK;
  for (int k = 0; k < lim; ++k) {
    const int ci = gated.idx[k];
    if (ci < 0 || ci >= topo.numCandidates || ci >= kMaxTopoCandidates) continue;
    const double o = computeOcclusion(topo.candidates[ci], ci, arena);
    const bool trig = needsActivePerception(o);
    if (trig) {
      BBox box;
      if (candidateBBox(static_cast<uint32_t>(ci) + 1u, arena.equivMap(),
                        kFrameWidth, kFrameHeight, box)) {
        extractCandidateROI(arena, box);
        diffuseScratchROI(arena, kPMIters);
      }
    }
    if (nRec < recordCap) {
      records[nRec].candIdx = ci;
      records[nRec].oScore = o;
      records[nRec].triggered = trig;
      ++nRec;
    }
  }
  return nRec;
}

}  // namespace panparasite
