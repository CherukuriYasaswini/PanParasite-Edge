// PanParasite-Edge Phase 3 implementation: Stage 3 homology engine.
// HOT-PATH translation unit: raw loops over StaticMemoryArena buffers only.
// Forbidden here: malloc/calloc/new/vector::push_back/cv::Mat::create,
// temporary cv::Mat, <vector>/<string>. Allowed: fixed stack scalars and
// small fixed stack tables ([64] accumulators), <cmath>, arena pointers.
//
// DSU field reuse: DisjointSetNode.min_x doubles as the background
// border-touch flag (0/1) during the bg pass only. resetDsuParents()
// zeroes it; the fg pass never reads it.

#include "topology_engine.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace panparasite {
namespace {

// Pi + deg/rad helpers (constexpr, no heap).
inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kRad2Deg = 180.0 / kPi;

// Packed hull point (int16 holds 0..1919 / 0..1079).
struct Pt {
  int16_t x = 0;
  int16_t y = 0;
};

// Linear search over small root tables. HOT-PATH SAFE.
int findIdx(const uint32_t* roots, int n, uint32_t r) noexcept {
  for (int i = 0; i < n; ++i) {
    if (roots[i] == r) return i;
  }
  return -1;
}

double ptSegDist(double px, double py, double ax, double ay, double bx,
                 double by) noexcept {
  const double dx = bx - ax;
  const double dy = by - ay;
  const double len2 = dx * dx + dy * dy;
  if (len2 <= 0.0) {
    const double ex = px - ax;
    const double ey = py - ay;
    return std::sqrt(ex * ex + ey * ey);
  }
  double t = ((px - ax) * dx + (py - ay) * dy) / len2;
  if (t < 0.0) t = 0.0;
  else if (t > 1.0) t = 1.0;
  const double cx = ax + t * dx - px;
  const double cy = ay + t * dy - py;
  return std::sqrt(cx * cx + cy * cy);
}

}  // namespace

// ------------------------------------------------------------------
// Union-Find primitives.
// ------------------------------------------------------------------
uint32_t findRoot(DisjointSetNode* dsu, uint32_t x) noexcept {
  if (dsu == nullptr || x >= static_cast<uint32_t>(kMaxLabels)) return 0u;
  uint32_t root = x;
  while (dsu[root].parent != root) {
    root = dsu[root].parent;
    if (root >= static_cast<uint32_t>(kMaxLabels)) return 0u;
  }
  while (dsu[x].parent != root) {
    const uint32_t nxt = dsu[x].parent;
    dsu[x].parent = root;
    x = nxt;
  }
  return root;
}

uint32_t unionLabels(DisjointSetNode* dsu, uint32_t a, uint32_t b) noexcept {
  if (a == 0u) return b;
  if (b == 0u) return a;
  if (a >= static_cast<uint32_t>(kMaxLabels) ||
      b >= static_cast<uint32_t>(kMaxLabels))
    return a < b ? a : b;
  uint32_t ra = findRoot(dsu, a);
  uint32_t rb = findRoot(dsu, b);
  if (ra == rb) return ra;
  if (dsu[ra].rank < dsu[rb].rank) {
    dsu[ra].parent = rb;
    dsu[rb].size += dsu[ra].size;
    return rb;
  }
  if (dsu[ra].rank > dsu[rb].rank) {
    dsu[rb].parent = ra;
    dsu[ra].size += dsu[rb].size;
    return ra;
  }
  dsu[rb].parent = ra;
  ++dsu[ra].rank;
  dsu[ra].size += dsu[rb].size;
  return ra;
}

// ------------------------------------------------------------------
// Raster labeling.
// ------------------------------------------------------------------
uint32_t labelForeground8(const uint8_t* mask, uint32_t* labels,
                          DisjointSetNode* dsu, int w, int h,
                          bool* overflow) noexcept {
  if (mask == nullptr || labels == nullptr || dsu == nullptr || w <= 0 ||
      h <= 0) {
    if (overflow != nullptr) *overflow = true;
    return 0u;
  }
  bool ov = false;
  uint32_t next = 1u;
  const uint32_t cap = static_cast<uint32_t>(kMaxLabels);
  for (int y = 0; y < h; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(w);
    for (int x = 0; x < w; ++x) {
      const std::size_t i = row + static_cast<std::size_t>(x);
      if (mask[i] == 0u) {
        labels[i] = 0u;
        continue;
      }
      // 8-connected predecessors: W, NW, N, NE.
      uint32_t nb[4];
      int n = 0;
      if (x > 0) {
        const uint32_t l = labels[i - 1u];
        if (l != 0u) nb[n++] = l;
      }
      if (x > 0 && y > 0) {
        const uint32_t l = labels[i - static_cast<std::size_t>(w) - 1u];
        if (l != 0u) nb[n++] = l;
      }
      if (y > 0) {
        const uint32_t l = labels[i - static_cast<std::size_t>(w)];
        if (l != 0u) nb[n++] = l;
      }
      if (x + 1 < w && y > 0) {
        const uint32_t l = labels[i - static_cast<std::size_t>(w) + 1u];
        if (l != 0u) nb[n++] = l;
      }
      if (n == 0) {
        if (next >= cap) {
          ov = true;
          labels[i] = 0u;  // degrade to background, never crash
          continue;
        }
        dsu[next].parent = next;
        dsu[next].rank = 0u;
        dsu[next].size = 1u;
        dsu[next].area = 0u;
        dsu[next].min_x = 0;
        dsu[next].min_y = 0;
        labels[i] = next;
        ++next;
      } else {
        uint32_t m = nb[0];
        for (int k = 1; k < n; ++k) {
          if (nb[k] < m) m = nb[k];
        }
        uint32_t r = m;
        for (int k = 0; k < n; ++k) {
          r = unionLabels(dsu, r, nb[k]);
        }
        labels[i] = r;
        const uint32_t root = findRoot(dsu, r);
        dsu[root].size += 1u;  // count current pixel
      }
    }
  }
  if (overflow != nullptr) *overflow = ov;
  return next - 1u;
}

uint32_t labelBackground4(const uint8_t* mask, uint32_t* bgPlane, int bgStride,
                          DisjointSetNode* dsu, int w, int h,
                          bool* overflow) noexcept {
  if (mask == nullptr || bgPlane == nullptr || dsu == nullptr || w <= 0 ||
      h <= 0 || bgStride <= w) {
    if (overflow != nullptr) *overflow = true;
    return 0u;
  }
  bool ov = false;
  uint32_t next = 1u;
  const uint32_t cap = static_cast<uint32_t>(kMaxLabels);
  const std::size_t s = static_cast<std::size_t>(bgStride);
  for (int y = 0; y < h; ++y) {
    const std::size_t mrow = static_cast<std::size_t>(y) * static_cast<std::size_t>(w);
    const std::size_t brow = static_cast<std::size_t>(y) * s;
    for (int x = 0; x < w; ++x) {
      const std::size_t bi = brow + static_cast<std::size_t>(x);
      if (mask[mrow + static_cast<std::size_t>(x)] != 0u) {
        bgPlane[bi] = 0u;  // foreground hole in bg plane
        continue;
      }
      uint32_t west = 0u, north = 0u;
      if (x > 0) west = bgPlane[bi - 1u];
      if (y > 0) north = bgPlane[bi - s];
      if (west == 0u && north == 0u) {
        if (next >= cap) {
          ov = true;
          bgPlane[bi] = 0u;
          continue;
        }
        dsu[next].parent = next;
        dsu[next].rank = 0u;
        dsu[next].size = 1u;
        dsu[next].area = 0u;
        dsu[next].min_x = 0;
        dsu[next].min_y = 0;
        bgPlane[bi] = next;
        ++next;
      } else if (west != 0u && north != 0u) {
        const uint32_t r = unionLabels(dsu, west, north);
        bgPlane[bi] = r;
        dsu[findRoot(dsu, r)].size += 1u;
      } else {
        const uint32_t r = (west != 0u) ? west : north;
        bgPlane[bi] = r;
        dsu[findRoot(dsu, r)].size += 1u;
      }
    }
  }
  if (overflow != nullptr) *overflow = ov;
  return next - 1u;
}

// ------------------------------------------------------------------
// Full pipeline.
// ------------------------------------------------------------------
int analyzeTopology(StaticMemoryArena& arena, TopoResult& out) noexcept {
  out.numCandidates = 0;
  out.globalBeta0 = 0;
  out.globalBeta1 = 0;
  out.globalChi = 0;
  out.overflow = false;
  out.fgProvisional = 0;
  out.bgProvisional = 0;
  for (int c = 0; c < kMaxTopoCandidates; ++c) out.candidates[c] = TopoCandidate{};

  const int W = kFrameWidth;
  const int H = kFrameHeight;
  const uint8_t* mask = arena.mask();
  uint32_t* fg = arena.equivMap();
  uint32_t* bg = arena.integral();  // borrowed bg plane
  DisjointSetNode* dsu = arena.dsu();
  uint8_t* scratch = arena.scratchRoi();

  // ---- fg pass ----
  arena.resetDsuParents();
  bool fgOv = false;
  const uint32_t fgMax = labelForeground8(mask, fg, dsu, W, H, &fgOv);
  out.fgProvisional = fgMax;
  out.overflow = out.overflow || fgOv;

  // Flatten provisional -> root (single loop, no per-pixel find later).
  for (uint32_t l = 1u; l <= fgMax; ++l) {
    dsu[l].parent = findRoot(dsu, l);
  }

  // Collect valid fg roots (area filter), dedup via small table.
  uint32_t fgRoots[kMaxTopoCandidates];
  int nFg = 0;
  bool candOv = false;
  for (uint32_t l = 1u; l <= fgMax; ++l) {
    const uint32_t r = dsu[l].parent;
    if (r == 0u || r >= static_cast<uint32_t>(kMaxLabels)) continue;
    if (dsu[r].size < kMinBioAreaPx || dsu[r].size > kMaxBioAreaPx) continue;
    if (findIdx(fgRoots, nFg, r) >= 0) continue;
    if (nFg >= kMaxTopoCandidates) {
      candOv = true;
      continue;
    }
    fgRoots[nFg++] = r;
  }
  out.overflow = out.overflow || candOv;
  out.numCandidates = nFg;
  out.globalBeta0 = nFg;

  // ---- remap + raw moment sums (single image pass) ----
  int64_t sumX[kMaxTopoCandidates] = {};
  int64_t sumY[kMaxTopoCandidates] = {};
  int64_t sumXX[kMaxTopoCandidates] = {};
  int64_t sumXY[kMaxTopoCandidates] = {};
  int64_t sumYY[kMaxTopoCandidates] = {};
  int64_t sumXXX[kMaxTopoCandidates] = {};
  int64_t sumXXY[kMaxTopoCandidates] = {};
  int64_t sumXYY[kMaxTopoCandidates] = {};
  int64_t sumYYY[kMaxTopoCandidates] = {};
  {
    const std::size_t n = kFramePixels;
    for (std::size_t i = 0; i < n; ++i) {
      if (mask[i] == 0u) {
        fg[i] = 0u;
        continue;
      }
      const uint32_t prov = fg[i];
      const uint32_t r = (prov < static_cast<uint32_t>(kMaxLabels)) ? dsu[prov].parent : 0u;
      const int idx = (r == 0u) ? -1 : findIdx(fgRoots, nFg, r);
      if (idx < 0) {
        fg[i] = 0u;  // filtered debris
        continue;
      }
      fg[i] = static_cast<uint32_t>(idx) + 1u;
      const int x = static_cast<int>(i % static_cast<std::size_t>(W));
      const int y = static_cast<int>(i / static_cast<std::size_t>(W));
      sumX[idx] += x;
      sumY[idx] += y;
      sumXX[idx] += static_cast<int64_t>(x) * x;
      sumXY[idx] += static_cast<int64_t>(x) * y;
      sumYY[idx] += static_cast<int64_t>(y) * y;
      sumXXX[idx] += static_cast<int64_t>(x) * x * x;
      sumXXY[idx] += static_cast<int64_t>(x) * x * y;
      sumXYY[idx] += static_cast<int64_t>(x) * y * y;
      sumYYY[idx] += static_cast<int64_t>(y) * y * y;
    }
  }

  // ---- per-candidate moments + inertial axis ----
  for (int c = 0; c < nFg; ++c) {
    TopoCandidate& cd = out.candidates[c];
    cd.rootLabel = fgRoots[c];
    cd.areaExt = dsu[fgRoots[c]].size;
    cd.areaUm2 = static_cast<double>(cd.areaExt) * kBioAreaUm2PerPx;
    const double a = static_cast<double>(cd.areaExt);
    cd.mu00 = a;
    cd.xbar = static_cast<double>(sumX[c]) / a;
    cd.ybar = static_cast<double>(sumY[c]) / a;
    const double xb = cd.xbar;
    const double yb = cd.ybar;
    cd.mu20 = static_cast<double>(sumXX[c]) - 2.0 * xb * static_cast<double>(sumX[c]) + a * xb * xb;
    cd.mu02 = static_cast<double>(sumYY[c]) - 2.0 * yb * static_cast<double>(sumY[c]) + a * yb * yb;
    cd.mu11 = static_cast<double>(sumXY[c]) - xb * static_cast<double>(sumY[c]) -
              yb * static_cast<double>(sumX[c]) + a * xb * yb;
    cd.mu30 = static_cast<double>(sumXXX[c]) - 3.0 * xb * static_cast<double>(sumXX[c]) +
              3.0 * xb * xb * static_cast<double>(sumX[c]) - a * xb * xb * xb;
    cd.mu21 = static_cast<double>(sumXXY[c]) - 2.0 * xb * static_cast<double>(sumXY[c]) -
              yb * static_cast<double>(sumXX[c]) + 2.0 * xb * xb * static_cast<double>(sumY[c]) +
              2.0 * xb * yb * static_cast<double>(sumX[c]) - a * xb * xb * yb;
    cd.mu12 = static_cast<double>(sumXYY[c]) - 2.0 * yb * static_cast<double>(sumXY[c]) -
              xb * static_cast<double>(sumYY[c]) + 2.0 * yb * yb * static_cast<double>(sumX[c]) +
              2.0 * xb * yb * static_cast<double>(sumY[c]) - a * xb * yb * yb;
    cd.mu03 = static_cast<double>(sumYYY[c]) - 3.0 * yb * static_cast<double>(sumYY[c]) +
              3.0 * yb * yb * static_cast<double>(sumY[c]) - a * yb * yb * yb;
    if (cd.mu20 < 0.0 && cd.mu20 > -1e-6) cd.mu20 = 0.0;
    if (cd.mu02 < 0.0 && cd.mu02 > -1e-6) cd.mu02 = 0.0;
    // Normalized by sigma = sqrt(mu00).
    const double s1 = a;              // sigma^2
    const double s15 = a * std::sqrt(a);  // sigma^3
    cd.mu20n = (s1 > 0.0) ? cd.mu20 / s1 : 0.0;
    cd.mu02n = (s1 > 0.0) ? cd.mu02 / s1 : 0.0;
    cd.mu11n = (s1 > 0.0) ? cd.mu11 / s1 : 0.0;
    cd.mu30n = (s15 > 0.0) ? cd.mu30 / s15 : 0.0;
    cd.mu21n = (s15 > 0.0) ? cd.mu21 / s15 : 0.0;
    cd.mu12n = (s15 > 0.0) ? cd.mu12 / s15 : 0.0;
    cd.mu03n = (s15 > 0.0) ? cd.mu03 / s15 : 0.0;
    // Principal inertial axis.
    cd.thetaMajor = 0.5 * std::atan2(2.0 * cd.mu11, cd.mu20 - cd.mu02);
    cd.vmajX = std::cos(cd.thetaMajor);
    cd.vmajY = std::sin(cd.thetaMajor);
    cd.beta0 = 1;
    cd.beta1 = 0;
    cd.chi = 1;
    cd.rHole = 0.0;
    cd.holeArea = 0u;
  }

  if (nFg == 0) {
    out.globalBeta1 = 0;
    out.globalChi = 0;
    return 0;
  }

  // ---- bg pass (DSU reused; fg state lives on in equivMap compact ids) ----
  arena.resetDsuParents();
  bool bgOv = false;
  const uint32_t bgMax =
      labelBackground4(mask, bg, kBgStride, dsu, W, H, &bgOv);
  out.bgProvisional = bgMax;
  out.overflow = out.overflow || bgOv;

  // Border-touch flags: scan image borders only (6K finds, cheap).
  for (int x = 0; x < W; ++x) {
    std::size_t mi = static_cast<std::size_t>(x);
    if (mask[mi] == 0u) {
      const uint32_t p = bg[static_cast<std::size_t>(0) * kBgStride + x];
      if (p != 0u) dsu[findRoot(dsu, p)].min_x = 1;
    }
    mi = static_cast<std::size_t>(H - 1) * W + x;
    if (mask[mi] == 0u) {
      const uint32_t p = bg[static_cast<std::size_t>(H - 1) * kBgStride + x];
      if (p != 0u) dsu[findRoot(dsu, p)].min_x = 1;
    }
  }
  for (int y = 0; y < H; ++y) {
    std::size_t mi = static_cast<std::size_t>(y) * W;
    if (mask[mi] == 0u) {
      const uint32_t p = bg[static_cast<std::size_t>(y) * kBgStride + 0];
      if (p != 0u) dsu[findRoot(dsu, p)].min_x = 1;
    }
    mi = static_cast<std::size_t>(y) * W + (W - 1);
    if (mask[mi] == 0u) {
      const uint32_t p = bg[static_cast<std::size_t>(y) * kBgStride + (W - 1)];
      if (p != 0u) dsu[findRoot(dsu, p)].min_x = 1;
    }
  }
  // Flatten bg labels.
  for (uint32_t l = 1u; l <= bgMax; ++l) {
    dsu[l].parent = findRoot(dsu, l);
  }

  // Collect holes (interior bg components).
  uint32_t holeRoots[kMaxTopoCandidates];
  uint32_t holeAreas[kMaxTopoCandidates];
  int nHoles = 0;
  bool holeOv = false;
  for (uint32_t l = 1u; l <= bgMax; ++l) {
    const uint32_t r = dsu[l].parent;
    if (r == 0u || r >= static_cast<uint32_t>(kMaxLabels)) continue;
    if (dsu[r].size == 0u) continue;
    if (dsu[r].min_x != 0) continue;  // exterior fluid
    if (findIdx(holeRoots, nHoles, r) >= 0) continue;
    if (nHoles >= kMaxTopoCandidates) {
      holeOv = true;
      continue;
    }
    holeRoots[nHoles] = r;
    holeAreas[nHoles] = dsu[r].size;
    ++nHoles;
  }
  out.overflow = out.overflow || holeOv;

  // Enclosure: single image pass; first fg neighbor wins.
  int holeOwner[kMaxTopoCandidates];
  for (int i = 0; i < nHoles; ++i) holeOwner[i] = -1;
  if (nHoles > 0) {
    for (int y = 0; y < H; ++y) {
      const std::size_t mrow = static_cast<std::size_t>(y) * W;
      const std::size_t brow = static_cast<std::size_t>(y) * kBgStride;
      for (int x = 0; x < W; ++x) {
        if (mask[mrow + x] != 0u) continue;
        const uint32_t p = bg[brow + x];
        if (p == 0u) continue;
        const uint32_t r = dsu[p].parent;
        const int hi = findIdx(holeRoots, nHoles, r);
        if (hi < 0 || holeOwner[hi] >= 0) continue;
        int owner = -1;
        if (x > 0 && fg[mrow + x - 1u] != 0u) owner = static_cast<int>(fg[mrow + x - 1u]) - 1;
        else if (x + 1 < W && fg[mrow + x + 1u] != 0u)
          owner = static_cast<int>(fg[mrow + x + 1u]) - 1;
        else if (y > 0 && fg[mrow + x - W] != 0u)
          owner = static_cast<int>(fg[mrow + x - W]) - 1;
        else if (y + 1 < H && fg[mrow + x + W] != 0u)
          owner = static_cast<int>(fg[mrow + x + W]) - 1;
        if (owner >= 0 && owner < nFg) holeOwner[hi] = owner;
      }
    }
  }
  int totalBeta1 = 0;
  for (int i = 0; i < nHoles; ++i) {
    const int o = holeOwner[i];
    if (o < 0 || o >= nFg) continue;  // unattributed (debris gap)
    out.candidates[o].beta1 += 1;
    out.candidates[o].holeArea += holeAreas[i];
    ++totalBeta1;
  }
  for (int c = 0; c < nFg; ++c) {
    TopoCandidate& cd = out.candidates[c];
    const double denom = static_cast<double>(cd.areaExt + cd.holeArea);
    cd.rHole = (denom > 0.0) ? static_cast<double>(cd.holeArea) / denom : 0.0;
    cd.chi = cd.beta0 - cd.beta1;
  }
  out.globalBeta1 = totalBeta1;
  out.globalChi = out.globalBeta0 - out.globalBeta1;

  // ---- per-candidate hull / defect / f_BSC (sequential, scratch) ----
  // Alignment note: scratch_ sits after only 4-multiple-sized members, and
  // alignof(StaticMemoryArena) >= 4 (uint32_t + pointer members), so the
  // int16 pair view below is naturally 2-byte aligned on all supported
  // ABIs (verified for GCC/Clang/MSVC, ARM64/x86_64).
  Pt* inPts = reinterpret_cast<Pt*>(scratch);
  Pt* hull = reinterpret_cast<Pt*>(scratch + 2048u * sizeof(Pt));
  for (int c = 0; c < nFg; ++c) {
    TopoCandidate& cd = out.candidates[c];
    const uint32_t want = static_cast<uint32_t>(c) + 1u;
    // Boundary extraction: full scan, count all, store up to cap.
    int64_t bcount = 0;
    int nPts = 0;
    bool trunc = false;
    for (int y = 0; y < H; ++y) {
      const std::size_t row = static_cast<std::size_t>(y) * W;
      for (int x = 0; x < W; ++x) {
        if (fg[row + x] != want) continue;
        const bool edge = (x == 0 || fg[row + x - 1u] != want || x + 1 == W ||
                           fg[row + x + 1u] != want || y == 0 || fg[row + x - W] != want ||
                           y + 1 == H || fg[row + x + W] != want);
        if (!edge) continue;
        ++bcount;
        if (nPts < kMaxHullInputPoints) {
          inPts[nPts].x = static_cast<int16_t>(x);
          inPts[nPts].y = static_cast<int16_t>(y);
          ++nPts;
        } else {
          trunc = true;
        }
      }
    }
    cd.pExt = static_cast<double>(bcount);
    cd.hullTruncated = trunc;
    if (trunc) out.overflow = true;

    // Insertion sort lexicographically (x, then y). nPts <= 2048.
    for (int i = 1; i < nPts; ++i) {
      Pt key = inPts[i];
      int j = i - 1;
      while (j >= 0 && (inPts[j].x > key.x || (inPts[j].x == key.x && inPts[j].y > key.y))) {
        inPts[j + 1] = inPts[j];
        --j;
      }
      inPts[j + 1] = key;
    }
    // Andrew monotone chain (cross fits int32).
    int hn = 0;
    for (int i = 0; i < nPts; ++i) {
      while (hn >= 2) {
        const int32_t x1 = hull[hn - 2].x, y1 = hull[hn - 2].y;
        const int32_t x2 = hull[hn - 1].x, y2 = hull[hn - 1].y;
        const int32_t x3 = inPts[i].x, y3 = inPts[i].y;
        const int32_t cr = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
        if (cr > 0) break;
        --hn;
      }
      hull[hn++] = inPts[i];
    }
    const int lower = hn;
    for (int i = nPts - 2; i >= 0; --i) {
      while (hn > lower) {
        const int32_t x1 = hull[hn - 2].x, y1 = hull[hn - 2].y;
        const int32_t x2 = hull[hn - 1].x, y2 = hull[hn - 1].y;
        const int32_t x3 = inPts[i].x, y3 = inPts[i].y;
        const int32_t cr = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
        if (cr > 0) break;
        --hn;
      }
      hull[hn++] = inPts[i];
    }
    if (hn > 1) --hn;  // drop duplicated start point
    // Hull area (shoelace) + perimeter.
    double ah = 0.0, ph = 0.0;
    if (hn >= 3) {
      double acc = 0.0;
      for (int i = 0; i < hn; ++i) {
        const int j = (i + 1) % hn;
        acc += static_cast<double>(hull[i].x) * hull[j].y -
               static_cast<double>(hull[j].x) * hull[i].y;
        const double dx = static_cast<double>(hull[j].x) - hull[i].x;
        const double dy = static_cast<double>(hull[j].y) - hull[i].y;
        ph += std::sqrt(dx * dx + dy * dy);
      }
      ah = (acc < 0.0 ? -acc : acc) * 0.5;
    } else if (hn == 2) {
      const double dx = static_cast<double>(hull[1].x) - hull[0].x;
      const double dy = static_cast<double>(hull[1].y) - hull[0].y;
      ph = 2.0 * std::sqrt(dx * dx + dy * dy);
      ah = 0.0;
    }
    cd.aHull = ah;
    cd.pHull = ph;
    // Max defect: farthest stored boundary point from hull polygon.
    double dmax = 0.0, dxAt = cd.xbar, dyAt = cd.ybar;
    if (hn >= 3 && nPts > 0) {
      for (int i = 0; i < nPts; ++i) {
        const double px = inPts[i].x, py = inPts[i].y;
        double dmin = 1e30;
        for (int e = 0; e < hn; ++e) {
          const int f = (e + 1) % hn;
          const double d = ptSegDist(px, py, hull[e].x, hull[e].y, hull[f].x, hull[f].y);
          if (d < dmin) dmin = d;
        }
        if (dmin > dmax) {
          dmax = dmin;
          dxAt = px;
          dyAt = py;
        }
      }
    } else if (hn == 2 && nPts > 0) {
      for (int i = 0; i < nPts; ++i) {
        const double d = ptSegDist(inPts[i].x, inPts[i].y, hull[0].x, hull[0].y,
                                   hull[1].x, hull[1].y);
        if (d > dmax) {
          dmax = d;
          dxAt = inPts[i].x;
          dyAt = inPts[i].y;
        }
      }
    }
    cd.dmaxPx = dmax;
    cd.dmaxUm = dmax * kMicronsPerPixel;
    cd.xd = dxAt;
    cd.yd = dyAt;
    cd.vdefX = dxAt - cd.xbar;
    cd.vdefY = dyAt - cd.ybar;
    const double vlen = std::sqrt(cd.vdefX * cd.vdefX + cd.vdefY * cd.vdefY);
    const double mlen = std::sqrt(cd.vmajX * cd.vmajX + cd.vmajY * cd.vmajY);
    double folded = 0.0;
    if (vlen > 1e-9 && mlen > 1e-9) {
      double cosv = (cd.vdefX * cd.vmajX + cd.vdefY * cd.vmajY) / (vlen * mlen);
      if (cosv < 0.0) cosv = -cosv;
      if (cosv > 1.0) cosv = 1.0;
      folded = std::acos(cosv) * kRad2Deg;  // [0,90]
      if (folded < 0.0) folded = 0.0;
      if (folded > 90.0) folded = 90.0;
    }
    cd.thetaFoldedDeg = folded;
    // ---- 16-D Betti signature ----
    const double ae = static_cast<double>(cd.areaExt);
    const double sq = (ae > 0.0) ? std::sqrt(ae) : 0.0;
    const double th = folded / kRad2Deg;  // rad for cos/sin
    cd.fbsc[0] = 1.0;
    cd.fbsc[1] = static_cast<double>(cd.beta1);
    cd.fbsc[2] = cd.rHole;
    cd.fbsc[3] = (sq > 0.0) ? cd.pExt / sq : 0.0;
    cd.fbsc[4] = (ah > 0.0) ? ae / ah : 0.0;
    cd.fbsc[5] = (sq > 0.0) ? dmax / sq : 0.0;
    cd.fbsc[6] = std::cos(th);
    cd.fbsc[7] = std::sin(th);
    cd.fbsc[8] = cd.mu20n;
    cd.fbsc[9] = cd.mu02n;
    cd.fbsc[10] = cd.mu11n;
    cd.fbsc[11] = cd.mu30n;
    cd.fbsc[12] = cd.mu12n;
    cd.fbsc[13] = cd.mu21n;
    cd.fbsc[14] = cd.mu03n;
    cd.fbsc[15] = (ph > 0.0) ? cd.pExt / ph : 0.0;
  }

  return nFg;
}

}  // namespace panparasite
