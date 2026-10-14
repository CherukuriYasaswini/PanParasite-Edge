// PanParasite-Edge Phase 5 implementation: FiLM micro-net + tracker.
// HOT-PATH translation unit: raw loops over StaticMemoryArena buffers only.
// Forbidden here: malloc/calloc/new/vector::push_back/cv::Mat::create,
// temporary cv::Mat, <vector>/<string>. Allowed: fixed stack scalars and
// small fixed stack tables, <cmath> (never allocates), arena pointers.
//
// Numerical design (verified by integer-exact Python mirror):
//   INT8 conv: uint8/int8 input, int32 accumulators (peak < 2^15, guard
//     trips at 2^28), arithmetic right-shift requant, clamp [0,127].
//   INT8 FiLM: out = (g_num*x + b_num*16) >> 9 (fbsc_q x64, W x8).
//   FP32 path: same topology in float; head gain matched approximately.
//   Routing is FiLM-beta dominated by construction: all conv kernels are
//   mean-preserving balanced patterns, so inter-block conv variance is
//   small next to template beta margins (min margin 0.8 beta units).

#include "inference_tracker.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace panparasite {
namespace {

// Deterministic boot LCG (fixed seed, no heap, no <random>).
inline uint32_t lcgNext(uint32_t& s) noexcept {
  s = s * 1664525u + 1013904223u;
  return s;
}

// Beta templates: B_k(fbsc) = base[k] + sum_j coef[k][j]*fbsc[j].
// Tuned so each taxon signature wins its block (see header rationale).
// Rows: 0 ascaris, 1 trichuris, 2 hookworm, 3 mansoni, 4 haematobium,
//       5 debris, 6 unknown.
// NOTE: the Beta_0 column (j=0) carries 0.0 because per-candidate Beta_0
// is identically 1 (the Beta_0 penalty from the Stage-A gate lives in
// active_perception.hpp; here the constant bases absorb it).

const double kBetaBase[7] = {0.5, 0.2, 5.0, 0.3, 0.3, -1.0, 0.9};
const double kBetaCoef[7][16] = {
    // ascaris: solidity+, R/B1/P-sparse-, size+
    {0.0, -1.5, -0.8, -0.15, 2.0, -0.3, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5},
    // trichuris: B1 rewards (2 -> +1.8), mild R toll
    {0.0, 0.0, 0.9, -0.15, 1.6, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5},
    // hookworm: small-egg bias via -f8, high const
    {0.0, -1.5, -0.8, -0.15, 1.5, -0.3, 0.0, 0.0, -0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5},
    // mansoni: lateral spine (dmax+, sin+, cos-)
    {0.0, -1.0, -0.5, 0.0, 1.2, 1.2, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4},
    // haematobium: terminal spine (dmax+, sin-, cos+)
    {0.0, -1.0, -0.5, 0.0, 1.2, 1.2, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4},
    // debris: holes/raggedness rewarded
    {0.0, 0.0, 0.6, 0.05, -1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3},
    // unknown: mild const, tiny tolls
    {0.0, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
};
// NOTE: kBetaCoef rows encode per-feature weights; the Beta_0 column (j=0)
// carries 0.0 because per-candidate Beta_0 is identically 1 (the -1.5
// Beta_0 penalty from the Stage-A gate lives in active_perception.hpp;
// here the constant bases absorb it).

bool g_forcedFallback = false;  // test-only fault-injection hook

// Activation plane views over the borrowed integral buffer (2 x 65536
// floats = 524,288 B of the 8,306,404 B plane; Stage 3 bg labels dead).
inline float* actPlaneA(StaticMemoryArena& a) noexcept {
  return reinterpret_cast<float*>(a.integral());
}

// INT8 helpers -------------------------------------------------------
inline int int8Layer(const int8_t* w, const int32_t* b, const uint8_t* in,
                     int inHW, int inC, int8_t* out, int outHW, int outC,
                     int k, int stride, int shift, int64_t& sat) noexcept {
  // Generic strided conv (k=3 stride 2, or k=1 stride 1 pointwise with
  // inHW==outHW). Returns peak |acc| for the overflow guard.
  int peak = 0;
  for (int oy = 0; oy < outHW; ++oy) {
    for (int ox = 0; ox < outHW; ++ox) {
      for (int oc = 0; oc < outC; ++oc) {
        int32_t acc = b ? b[oc] : 0;
        if (k == 1) {
          const std::size_t ibase =
              (static_cast<std::size_t>(oy) * inHW + ox) * inC;
          const std::size_t wbase = static_cast<std::size_t>(oc) * inC;
          for (int ic = 0; ic < inC; ++ic) {
            acc += static_cast<int32_t>(w[wbase + ic]) *
                   static_cast<int32_t>(in[ibase + ic]);
          }
        } else {
          for (int ky = 0; ky < k; ++ky) {
            const int iy = oy * stride + ky - 1;  // same-pad for k=3,s=2
            for (int kx = 0; kx < k; ++kx) {
              const int ix = ox * stride + kx - 1;
              const int py = (iy < 0) ? 0 : ((iy >= inHW) ? inHW - 1 : iy);
              const int px = (ix < 0) ? 0 : ((ix >= inHW) ? inHW - 1 : ix);
              const std::size_t ibase =
                  (static_cast<std::size_t>(py) * inHW + px) * inC;
              for (int ic = 0; ic < inC; ++ic) {
                const std::size_t wi =
                    ((static_cast<std::size_t>(oc) * inC + ic) * k + ky) * k +
                    kx;
                acc += static_cast<int32_t>(w[wi]) *
                       static_cast<int32_t>(in[ibase + ic]);
              }
            }
          }
        }
        const int32_t a = (acc < 0) ? -acc : acc;
        if (a > peak) peak = a;
        int32_t v = acc >> shift;  // arithmetic (C++20), power-of-two scale
        if (v < 0) v = 0;          // ReLU (ReLU6 bound >= 127: no-op cap)
        if (v > 127) {
          v = 127;
          ++sat;
        }
        out[(static_cast<std::size_t>(oy) * outHW + ox) * outC + oc] =
            static_cast<int8_t>(v);
      }
    }
  }
  return peak;
}

// Depthwise 3x3 stride 2 INT8 (channelwise).
inline int int8Depthwise(const int8_t* w, const uint8_t* in, int inHW, int ch,
                         int8_t* out, int outHW, int shift,
                         int64_t& sat) noexcept {
  int peak = 0;
  for (int oy = 0; oy < outHW; ++oy) {
    for (int ox = 0; ox < outHW; ++ox) {
      for (int c = 0; c < ch; ++c) {
        int32_t acc = 0;
        for (int ky = 0; ky < 3; ++ky) {
          const int iy = oy * 2 + ky - 1;
          const int py = (iy < 0) ? 0 : ((iy >= inHW) ? inHW - 1 : iy);
          for (int kx = 0; kx < 3; ++kx) {
            const int ix = ox * 2 + kx - 1;
            const int px = (ix < 0) ? 0 : ((ix >= inHW) ? inHW - 1 : ix);
            acc += static_cast<int32_t>(w[c * 9 + ky * 3 + kx]) *
                   static_cast<int32_t>(
                       in[(static_cast<std::size_t>(py) * inHW + px) * ch + c]);
          }
        }
        const int32_t a = (acc < 0) ? -acc : acc;
        if (a > peak) peak = a;
        int32_t v = acc >> shift;
        if (v < 0) v = 0;
        if (v > 127) {
          v = 127;
          ++sat;
        }
        out[(static_cast<std::size_t>(oy) * outHW + ox) * ch + c] =
            static_cast<int8_t>(v);
      }
    }
  }
  return peak;
}

// FP32 helpers --------------------------------------------------------
inline void fp32Layer(const float* w, const float* b, const float* in,
                      int inHW, int inC, float* out, int outHW, int outC,
                      int k, int stride) noexcept {
  for (int oy = 0; oy < outHW; ++oy) {
    for (int ox = 0; ox < outHW; ++ox) {
      for (int oc = 0; oc < outC; ++oc) {
        float acc = b ? b[oc] : 0.0f;
        if (k == 1) {
          const std::size_t ibase =
              (static_cast<std::size_t>(oy) * inHW + ox) * inC;
          const std::size_t wbase = static_cast<std::size_t>(oc) * inC;
          for (int ic = 0; ic < inC; ++ic)
            acc += w[wbase + ic] * in[ibase + ic];
        } else {
          for (int ky = 0; ky < k; ++ky) {
            const int iy = oy * stride + ky - 1;
            for (int kx = 0; kx < k; ++kx) {
              const int ix = ox * stride + kx - 1;
              const int py = (iy < 0) ? 0 : ((iy >= inHW) ? inHW - 1 : iy);
              const int px = (ix < 0) ? 0 : ((ix >= inHW) ? inHW - 1 : ix);
              const std::size_t ibase =
                  (static_cast<std::size_t>(py) * inHW + px) * inC;
              for (int ic = 0; ic < inC; ++ic) {
                const std::size_t wi =
                    ((static_cast<std::size_t>(oc) * inC + ic) * k + ky) * k +
                    kx;
                acc += w[wi] * in[ibase + ic];
              }
            }
          }
        }
        if (acc < 0.0f) acc = 0.0f;  // ReLU
        if (acc > 6.0f) acc = 6.0f;  // ReLU6
        out[(static_cast<std::size_t>(oy) * outHW + ox) * outC + oc] = acc;
      }
    }
  }
}

inline void fp32Depthwise(const float* w, const float* in, int inHW, int ch,
                          float* out, int outHW) noexcept {
  for (int oy = 0; oy < outHW; ++oy) {
    for (int ox = 0; ox < outHW; ++ox) {
      for (int c = 0; c < ch; ++c) {
        float acc = 0.0f;
        for (int ky = 0; ky < 3; ++ky) {
          const int iy = oy * 2 + ky - 1;
          const int py = (iy < 0) ? 0 : ((iy >= inHW) ? inHW - 1 : iy);
          for (int kx = 0; kx < 3; ++kx) {
            const int ix = ox * 2 + kx - 1;
            const int px = (ix < 0) ? 0 : ((ix >= inHW) ? inHW - 1 : ix);
            acc += w[c * 9 + ky * 3 + kx] *
                   in[(static_cast<std::size_t>(py) * inHW + px) * ch + c];
          }
        }
        if (acc < 0.0f) acc = 0.0f;
        if (acc > 6.0f) acc = 6.0f;
        out[(static_cast<std::size_t>(oy) * outHW + ox) * ch + c] = acc;
      }
    }
  }
}

}  // namespace

void microNetSetForcedFallback(bool on) noexcept { g_forcedFallback = on; }

bool microNetInitBoot(StaticMemoryArena& arena) noexcept {
  uint8_t* base = arena.filmParams();
  if (base == nullptr) return false;
  // FP32 table pointers (mirror section).
  auto fp32At = [&](std::size_t off) noexcept -> float* {
    return reinterpret_cast<float*>(base + kOffFp32 + off);
  };
  uint32_t s = 0xC0FFEEu;
  // L1 front end: per-filter gains {1,2} (LCG order matters — the Python
  // mirror replicates this exact sequence). Gain variation keeps the
  // image path alive through later balanced stages.
  int8_t* l1w = reinterpret_cast<int8_t*>(base + kOffL1W);
  for (int f = 0; f < 16; ++f) {
    lcgNext(s);
    const int8_t g = static_cast<int8_t>(1 + (s & 1u));
    for (int t = 0; t < 9; ++t) l1w[f * 9 + t] = g;
  }
  auto fillBalanced = [&](int8_t* w, std::size_t n) noexcept {
    // Exactly half +1, half -1 (n even), LCG-permuted positions.
    for (std::size_t i = 0; i < n; ++i) w[i] = (i < n / 2) ? 1 : -1;
    for (std::size_t i = n - 1; i > 0; --i) {
      lcgNext(s);
      const std::size_t j = s % (i + 1);
      const int8_t t = w[i];
      w[i] = w[j];
      w[j] = t;
    }
  };
  // Per-channel 3x3 DW: five +1 / four -1 (sum +1), permuted per channel.
  auto fillDw = [&](int8_t* w, int ch) noexcept {
    for (int c = 0; c < ch; ++c) {
      int8_t k[9] = {1, 1, 1, 1, 1, -1, -1, -1, -1};
      for (int i = 8; i > 0; --i) {
        lcgNext(s);
        const int j = static_cast<int>(s % static_cast<uint32_t>(i + 1));
        const int8_t t = k[i];
        k[i] = k[j];
        k[j] = t;
      }
      for (int i = 0; i < 9; ++i) w[c * 9 + i] = k[i];
    }
  };
  fillDw(reinterpret_cast<int8_t*>(base + kOffDW1W), 16);
  fillBalanced(reinterpret_cast<int8_t*>(base + kOffPW1W), kMiPW1W);
  fillDw(reinterpret_cast<int8_t*>(base + kOffDW2W), 32);
  fillBalanced(reinterpret_cast<int8_t*>(base + kOffPW2W), kMiPW2W);
  // Head: +6 on own 8-ch block, -1 elsewhere.
  int8_t* hw = reinterpret_cast<int8_t*>(base + kOffHW);
  for (int k = 0; k < 7; ++k) {
    for (int c = 0; c < 64; ++c) {
      hw[k * 64 + c] = ((c / 8) == k) ? 6 : -1;
    }
  }
  // INT8 biases: conv/head zero; FiLM gamma base 1.0, beta from templates.
  auto setB32 = [&](std::size_t off, int idx, int32_t v) noexcept {
    reinterpret_cast<int32_t*>(base + off)[idx] = v;
  };
  for (int i = 0; i < 16; ++i) setB32(kOffL1B, i, 0);
  for (int i = 0; i < 32; ++i) setB32(kOffB1, i, 0);
  for (int i = 0; i < 64; ++i) setB32(kOffB2, i, 0);
  for (int i = 0; i < 7; ++i) setB32(kOffHB, i, 0);
  // FiLM1: mild shared egg-ness (0.3 x ascaris row), gamma = 1.
  for (int c = 0; c < 32; ++c) {
    int8_t* wg = reinterpret_cast<int8_t*>(base + kOffF1W) + c * 16;
    int8_t* wb = reinterpret_cast<int8_t*>(base + kOffF1W) + 512 + c * 16;
    for (int j = 0; j < 16; ++j) {
      wg[j] = 0;
      double coef = 0.3 * kBetaCoef[0][j];
      double q = coef * kFilmWScale[j];
      if (q > 127.0) q = 127.0;
      if (q < -128.0) q = -128.0;
      wb[j] = static_cast<int8_t>(q);
    }
    setB32(kOffF1B, c, 512);            // gamma base 1.0 (x512)
    setB32(kOffF1B, 32 + c, 77);        // beta base 0.3*B0base(0.5) x512
  }
  // FiLM2: per-block class templates (8 channels per block).
  for (int c = 0; c < 64; ++c) {
    const int blk = c / 8;
    const int k = (blk < 7) ? blk : 6;
    int8_t* wg = reinterpret_cast<int8_t*>(base + kOffF2W) + c * 16;
    int8_t* wb = reinterpret_cast<int8_t*>(base + kOffF2W) + 1024 + c * 16;
    for (int j = 0; j < 16; ++j) {
      wg[j] = 0;  // gamma topology nudge: 0 (gamma == 1.0 exactly)
      double q = kBetaCoef[k][j] * kFilmWScale[j];
      if (q > 127.0) q = 127.0;
      if (q < -128.0) q = -128.0;
      wb[j] = static_cast<int8_t>(q);
    }
    setB32(kOffF2B, c, 512);  // gamma base 1.0 (x512)
    double bb = kBetaBase[k] * 512.0;
    int32_t bi = (bb > 0.0) ? static_cast<int32_t>(bb + 0.5)
                            : static_cast<int32_t>(bb - 0.5);
    setB32(kOffF2B, 64 + c, bi);
  }
  // FP32 mirrors: conv = int8 * 0.25; head same integers; FiLM exact.
  auto mirror8 = [&](std::size_t ioff, std::size_t foff, std::size_t n,
                     double g) noexcept {
    const int8_t* si = reinterpret_cast<const int8_t*>(base + ioff);
    float* d = fp32At(foff);
    for (std::size_t i = 0; i < n; ++i) d[i] = static_cast<float>(si[i] * g);
  };
  auto mirror32 = [&](std::size_t ioff, std::size_t foff, std::size_t n,
                      double g) noexcept {
    const int32_t* si = reinterpret_cast<const int32_t*>(base + ioff);
    float* d = fp32At(foff);
    for (std::size_t i = 0; i < n; ++i) d[i] = static_cast<float>(si[i] * g);
  };
  mirror8(kOffL1W, 0, kMiL1W, 0.25);
  mirror32(kOffL1B, kMiL1Wf, 16, 0.25);
  mirror8(kOffDW1W, kMiL1Wf + kMiL1Bf, kMiDW1W, 0.25);
  mirror8(kOffPW1W, kMiL1Wf + kMiL1Bf + kMiDW1Wf, kMiPW1W, 0.25);
  mirror32(kOffB1, kMiL1Wf + kMiL1Bf + kMiDW1Wf + kMiPW1Wf, 32, 0.25);
  // FiLM FP32 mirrors are exact (W/8, B/512).
  // FiLM FP32 mirrors are exact per feature (W/WScale[j], B/512).
  {
    const int8_t* si =
        reinterpret_cast<const int8_t*>(base + kOffF1W);
    float* d = fp32At(kMiL1Wf + kMiL1Bf + kMiDW1Wf + kMiPW1Wf + kMiB1f);
    for (int r = 0; r < 64; ++r) {
      for (int j = 0; j < 16; ++j) {
        d[r * 16 + j] =
            static_cast<float>(si[r * 16 + j] / kFilmWScale[j]);
      }
    }
  }
  mirror32(kOffF1B, kMiL1Wf + kMiL1Bf + kMiDW1Wf + kMiPW1Wf + kMiB1f + kMiF1Wf,
           64, 1.0 / 512.0);
  mirror8(kOffDW2W, kMiL1Wf + kMiL1Bf + kMiDW1Wf + kMiPW1Wf + kMiB1f + kMiF1Wf +
                        kMiF1Bf,
          kMiDW2W, 0.25);
  mirror8(kOffPW2W, kMiL1Wf + kMiL1Bf + kMiDW1Wf + kMiPW1Wf + kMiB1f + kMiF1Wf +
                        kMiF1Bf + kMiDW2Wf,
          kMiPW2W, 0.25);
  mirror32(kOffB2, kMiL1Wf + kMiL1Bf + kMiDW1Wf + kMiPW1Wf + kMiB1f + kMiF1Wf +
                       kMiF1Bf + kMiDW2Wf + kMiPW2Wf,
           64, 0.25);
  {
    const int8_t* si =
        reinterpret_cast<const int8_t*>(base + kOffF2W);
    float* d = fp32At(kMiL1Wf + kMiL1Bf + kMiDW1Wf + kMiPW1Wf + kMiB1f + kMiF1Wf +
                      kMiF1Bf + kMiDW2Wf + kMiPW2Wf + kMiB2f);
    for (int r = 0; r < 128; ++r) {
      for (int j = 0; j < 16; ++j) {
        d[r * 16 + j] =
            static_cast<float>(si[r * 16 + j] / kFilmWScale[j]);
      }
    }
  }
  mirror32(kOffF2B, kMiL1Wf + kMiL1Bf + kMiDW1Wf + kMiPW1Wf + kMiB1f + kMiF1Wf +
                        kMiF1Bf + kMiDW2Wf + kMiPW2Wf + kMiB2f + kMiF2Wf,
           128, 1.0 / 512.0);
  mirror8(kOffHW, kMiL1Wf + kMiL1Bf + kMiDW1Wf + kMiPW1Wf + kMiB1f + kMiF1Wf +
                      kMiF1Bf + kMiDW2Wf + kMiPW2Wf + kMiB2f + kMiF2Wf +
                      kMiF2Bf,
          kMiHW, 1.0);
  mirror32(kOffHB, kMiL1Wf + kMiL1Bf + kMiDW1Wf + kMiPW1Wf + kMiB1f + kMiF1Wf +
                       kMiF1Bf + kMiDW2Wf + kMiPW2Wf + kMiB2f + kMiF2Wf +
                       kMiF2Bf + kMiHWf,
           7, 1.0);
  // Requant scales (informational; shifts are authoritative in INT8).
  float* sc = reinterpret_cast<float*>(base + kOffScales);
  sc[0] = 1.0f / 32.0f;
  sc[1] = 1.0f / 16.0f;
  sc[2] = 1.0f / 32.0f;
  sc[3] = 1.0f / 16.0f;
  sc[4] = 1.0f / 64.0f;
  sc[5] = 0.02f;  // head dequant
  sc[6] = 0.0f;
  sc[7] = 0.0f;
  return true;
}

void plattSoftmax(const double logits[kNumTaxa], double probs[kNumTaxa]) noexcept {
  if (logits == nullptr || probs == nullptr) return;
  double mx = logits[0];
  for (int i = 1; i < kNumTaxa; ++i) {
    if (logits[i] > mx) mx = logits[i];
  }
  double sum = 0.0;
  for (int i = 0; i < kNumTaxa; ++i) {
    const double e = std::exp((logits[i] - mx) / kPlattT);
    probs[i] = e;
    sum += e;
  }
  if (!(sum > 0.0)) {
    for (int i = 0; i < kNumTaxa; ++i) probs[i] = 1.0 / kNumTaxa;
    return;
  }
  for (int i = 0; i < kNumTaxa; ++i) probs[i] /= sum;
}

MicroNetProbs microNetClassify(StaticMemoryArena& arena, const double fbsc[16],
                               MicroNetAudit& audit) noexcept {
  MicroNetProbs out;
  if (fbsc == nullptr) return out;
  uint8_t* base = arena.filmParams();
  if (base == nullptr) return out;
  float* plane = actPlaneA(arena);
  // Activation planes: A @0, B @65536 floats (524,288 B of integral).
  float* pA = plane;
  float* pB = plane + 65536;
  // ---- INT8 path (native fixed-point) ----
  // Scratch int8 planes borrowed from plane memory (same bytes, no heap).
  // L1 needs 65536 B; L2 32768 B; L3 16384 B — all fit plane A/B.
  uint8_t* iA = reinterpret_cast<uint8_t*>(pA);
  uint8_t* iB = reinterpret_cast<uint8_t*>(pB);
  int8_t* l1 = reinterpret_cast<int8_t*>(iA);
  int8_t* dw1 = reinterpret_cast<int8_t*>(iB);
  int8_t* pw1 = reinterpret_cast<int8_t*>(iA);
  int8_t* dw2 = reinterpret_cast<int8_t*>(iB);
  int8_t* pw2 = reinterpret_cast<int8_t*>(iA);
  const uint8_t* patch = arena.scratchRoi();
  int64_t sat = 0;
  int peak = 0;
  {
    const int8_t* w = reinterpret_cast<const int8_t*>(base + kOffL1W);
    const int32_t* b = reinterpret_cast<const int32_t*>(base + kOffL1B);
    // L1: 3x3 s2 1->16 over uint8 patch (bias int32, shift 5).
    for (int oy = 0; oy < 64; ++oy) {
      for (int ox = 0; ox < 64; ++ox) {
        for (int oc = 0; oc < 16; ++oc) {
          int32_t acc = b[oc];
          for (int ky = 0; ky < 3; ++ky) {
            const int iy = oy * 2 + ky - 1;
            const int py = (iy < 0) ? 0 : ((iy > 127) ? 127 : iy);
            for (int kx = 0; kx < 3; ++kx) {
              const int ix = ox * 2 + kx - 1;
              const int px = (ix < 0) ? 0 : ((ix > 127) ? 127 : ix);
              acc += static_cast<int32_t>(w[oc * 9 + ky * 3 + kx]) *
                     static_cast<int32_t>(
                         patch[static_cast<std::size_t>(py) * 128u + px]);
            }
          }
          const int32_t a = (acc < 0) ? -acc : acc;
          if (a > peak) peak = a;
          int32_t v = acc >> kShiftL1;
          if (v < 0) v = 0;
          if (v > 127) {
            v = 127;
            ++sat;
          }
          l1[(static_cast<std::size_t>(oy) * 64u + ox) * 16u + oc] =
              static_cast<int8_t>(v);
        }
      }
    }
  }
  {
    const int8_t* w = reinterpret_cast<const int8_t*>(base + kOffDW1W);
    int pdw = int8Depthwise(w, reinterpret_cast<const uint8_t*>(l1), 64, 16, dw1,
                            32, kShiftDW1, sat);
    if (pdw > peak) peak = pdw;
  }
  {
    const int8_t* w = reinterpret_cast<const int8_t*>(base + kOffPW1W);
    const int32_t* b = reinterpret_cast<const int32_t*>(base + kOffB1);
    int p = int8Layer(w, b, reinterpret_cast<const uint8_t*>(dw1), 32, 16, pw1,
                      32, 32, 1, 1, kShiftPW1, sat);
    if (p > peak) peak = p;
  }
  // FiLM INT8: quantize fbsc per-feature (stack, 16 int32 = 64 B).
  int32_t fq[16];
  for (int j = 0; j < 16; ++j) {
    double q = fbsc[j] * kFeatQ[j];
    if (q > 2147483647.0) q = 2147483647.0;
    if (q < -2147483648.0) q = -2147483648.0;
    fq[j] = static_cast<int32_t>(q);
  }
  {
    const int8_t* wf = reinterpret_cast<const int8_t*>(base + kOffF1W);
    const int32_t* bf = reinterpret_cast<const int32_t*>(base + kOffF1B);
    uint8_t* src = reinterpret_cast<uint8_t*>(pw1);
    for (int c = 0; c < 32; ++c) {
      int64_t gn = bf[c];
      int64_t bn = bf[32 + c];
      for (int j = 0; j < 16; ++j) {
        gn += static_cast<int64_t>(wf[c * 16 + j]) * fq[j];
        bn += static_cast<int64_t>(wf[512 + c * 16 + j]) * fq[j];
      }
      // Clamp FiLM numerators to int32 range before use.
      int32_t g32 = (gn > 2147483647LL) ? 2147483647 : ((gn < -2147483648LL) ? -2147483648 : static_cast<int32_t>(gn));
      int32_t b32 = (bn > 2147483647LL) ? 2147483647 : ((bn < -2147483648LL) ? -2147483648 : static_cast<int32_t>(bn));
      for (int i = 0; i < 1024; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i) * 32u + c;
        const int32_t x = static_cast<int32_t>(src[idx]);
        const int64_t v =
            (static_cast<int64_t>(g32) * x + static_cast<int64_t>(b32) * 16) >>
            kFilmQShift;
        int32_t c8 = (v > 2147483647LL) ? 2147483647 : ((v < -2147483648LL) ? -2147483648 : static_cast<int32_t>(v));
        if (c8 < 0) c8 = 0;
        if (c8 > 127) {
          c8 = 127;
          ++sat;
        }
        src[idx] = static_cast<uint8_t>(c8);
      }
    }
  }
  {
    const int8_t* w = reinterpret_cast<const int8_t*>(base + kOffDW2W);
    int pdw2 = int8Depthwise(w, reinterpret_cast<const uint8_t*>(pw1), 32, 32, dw2,
                             16, kShiftDW2, sat);
    if (pdw2 > peak) peak = pdw2;
  }
  {
    const int8_t* w = reinterpret_cast<const int8_t*>(base + kOffPW2W);
    const int32_t* b = reinterpret_cast<const int32_t*>(base + kOffB2);
    int p = int8Layer(w, b, reinterpret_cast<const uint8_t*>(dw2), 16, 32, pw2,
                      16, 64, 1, 1, kShiftPW2, sat);
    if (p > peak) peak = p;
  }
  {
    const int8_t* wf = reinterpret_cast<const int8_t*>(base + kOffF2W);
    const int32_t* bf = reinterpret_cast<const int32_t*>(base + kOffF2B);
    uint8_t* src = reinterpret_cast<uint8_t*>(pw2);
    for (int c = 0; c < 64; ++c) {
      int64_t gn = bf[c];
      int64_t bn = bf[64 + c];
      for (int j = 0; j < 16; ++j) {
        gn += static_cast<int64_t>(wf[c * 16 + j]) * fq[j];
        bn += static_cast<int64_t>(wf[1024 + c * 16 + j]) * fq[j];
      }
      int32_t g32 = (gn > 2147483647LL) ? 2147483647 : ((gn < -2147483648LL) ? -2147483648 : static_cast<int32_t>(gn));
      int32_t b32 = (bn > 2147483647LL) ? 2147483647 : ((bn < -2147483648LL) ? -2147483648 : static_cast<int32_t>(bn));
      for (int i = 0; i < 256; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i) * 64u + c;
        const int32_t x = static_cast<int32_t>(src[idx]);
        const int64_t v =
            (static_cast<int64_t>(g32) * x + static_cast<int64_t>(b32) * 16) >>
            kFilmQShift;
        int32_t c8 = (v > 2147483647LL) ? 2147483647 : ((v < -2147483648LL) ? -2147483648 : static_cast<int32_t>(v));
        if (c8 < 0) c8 = 0;
        if (c8 > 127) {
          c8 = 127;
          ++sat;
        }
        src[idx] = static_cast<uint8_t>(c8);
      }
    }
  }
  // GAP + head (int32), dequant to float logits.
  double logits[kNumTaxa] = {};
  {
    const int8_t* w = reinterpret_cast<const int8_t*>(base + kOffHW);
    const int32_t* b = reinterpret_cast<const int32_t*>(base + kOffHB);
    const uint8_t* feat = reinterpret_cast<const uint8_t*>(pw2);
    int32_t mean[64];
    for (int c = 0; c < 64; ++c) {
      int32_t s = 0;
      for (int i = 0; i < 256; ++i)
        s += static_cast<int32_t>(feat[static_cast<std::size_t>(i) * 64u + c]);
      mean[c] = (s + 128) >> 8;
    }
    const float* sc = reinterpret_cast<const float*>(base + kOffScales);
    const float hs = sc[5];
    for (int k = 0; k < kNumTaxa; ++k) {
      int32_t acc = b[k];
      for (int c = 0; c < 64; ++c) {
        acc += static_cast<int32_t>(w[k * 64 + c]) * mean[c];
      }
      logits[k] = static_cast<double>(acc) * hs;
    }
  }
  ++audit.int8_runs;
  // Finite/range guard: non-finite logits, int32 overflow risk, or the
  // test-only fault-injection hook forces the FP32 fallback path.
  bool bad = g_forcedFallback;
  if (!bad) {
    if (peak > (1 << 28)) bad = true;
    for (int k = 0; k < kNumTaxa && !bad; ++k) {
      if (!std::isfinite(logits[k])) bad = true;
    }
  } else {
    for (int k = 0; k < kNumTaxa; ++k) logits[k] = kNonFiniteSentinel;
  }
  if (bad) {
    ++audit.fp32_fallback_count;
    // ---- Embedded FP32 fallback engine (same graph, float) ----
    const float* L1W = reinterpret_cast<const float*>(base + kOffFp32);
    const float* L1B = L1W + 144;
    const float* DW1W = L1B + 16;
    const float* PW1W = DW1W + 144;
    const float* B1 = PW1W + 512;
    const float* F1W = B1 + 32;
    const float* F1B = F1W + 1024;
    const float* DW2W = F1B + 64;
    const float* PW2W = DW2W + 288;
    const float* B2 = PW2W + 2048;
    const float* F2W = B2 + 64;
    const float* F2B = F2W + 2048;
    const float* HW = F2B + 128;
    const float* HB = HW + 448;
    // Normalize patch to [-1,1).
    for (int i = 0; i < 16384; ++i)
      pA[i] = static_cast<float>(patch[i]) / 128.0f - 1.0f;
    fp32Layer(L1W, L1B, pA, 128, 1, pB, 64, 16, 3, 2);
    fp32Depthwise(DW1W, pB, 64, 16, pA, 32);
    fp32Layer(PW1W, B1, pA, 32, 16, pB, 32, 32, 1, 1);
    for (int c = 0; c < 32; ++c) {  // FiLM1
      float g = F1B[c], bb = F1B[32 + c];
      for (int j = 0; j < 16; ++j) {
        g += F1W[c * 16 + j] * static_cast<float>(fbsc[j]);
        bb += F1W[512 + c * 16 + j] * static_cast<float>(fbsc[j]);
      }
      for (int i = 0; i < 1024; ++i) {
        float v = g * pB[static_cast<std::size_t>(i) * 32u + c] + bb;
        if (v < 0.0f) v = 0.0f;
        if (v > 6.0f) v = 6.0f;
        pB[static_cast<std::size_t>(i) * 32u + c] = v;
      }
    }
    fp32Depthwise(DW2W, pB, 32, 32, pA, 16);
    fp32Layer(PW2W, B2, pA, 16, 32, pB, 16, 64, 1, 1);
    for (int c = 0; c < 64; ++c) {  // FiLM2
      float g = F2B[c], bb = F2B[64 + c];
      for (int j = 0; j < 16; ++j) {
        g += F2W[c * 16 + j] * static_cast<float>(fbsc[j]);
        bb += F2W[1024 + c * 16 + j] * static_cast<float>(fbsc[j]);
      }
      for (int i = 0; i < 256; ++i) {
        float v = g * pB[static_cast<std::size_t>(i) * 64u + c] + bb;
        if (v < 0.0f) v = 0.0f;
        if (v > 6.0f) v = 6.0f;
        pB[static_cast<std::size_t>(i) * 64u + c] = v;
      }
    }
    for (int k = 0; k < kNumTaxa; ++k) {  // GAP + head
      double acc = HB[k];
      for (int c = 0; c < 64; ++c) {
        double s = 0.0;
        for (int i = 0; i < 256; ++i)
          s += pB[static_cast<std::size_t>(i) * 64u + c];
        acc += HW[k * 64 + c] * (s / 256.0);
      }
      logits[k] = acc;
      if (!std::isfinite(logits[k])) logits[k] = 0.0;
    }
    out.usedFp32Fallback = true;
  }
  plattSoftmax(logits, out.p);
  int best = 0;
  for (int k = 1; k < kNumTaxa; ++k) {
    if (out.p[k] > out.p[best]) best = k;
  }
  out.argmax = best;
  (void)sat;
  return out;
}

// ------------------------------------------------------------------
// Stage 6.
// ------------------------------------------------------------------
double bboxIoU(float ax0, float ay0, float ax1, float ay1, float bx0,
               float by0, float bx1, float by1) noexcept {
  float ix0 = (ax0 > bx0) ? ax0 : bx0;
  float iy0 = (ay0 > by0) ? ay0 : by0;
  float ix1 = (ax1 < bx1) ? ax1 : bx1;
  float iy1 = (ay1 < by1) ? ay1 : by1;
  if (ix1 < ix0 || iy1 < iy0) return 0.0;
  const double inter =
      static_cast<double>(ix1 - ix0 + 1) * (iy1 - iy0 + 1);
  const double aa =
      static_cast<double>(ax1 - ax0 + 1) * (ay1 - ay0 + 1);
  const double ab =
      static_cast<double>(bx1 - bx0 + 1) * (by1 - by0 + 1);
  if (!(aa > 0.0) || !(ab > 0.0)) return 0.0;
  return inter / (aa + ab - inter);
}

void hungarianMatch(const double* cost, int nTracks, int nDets, int* assign,
                    int* newDet, int* nNew) noexcept {
  if (assign == nullptr) return;
  for (int i = 0; i < kMaxActiveTracks; ++i) assign[i] = -1;
  if (nNew != nullptr) *nNew = 0;
  if (cost == nullptr || nTracks <= 0 || nDets <= 0) {
    // Everything unmatched: all detections are new.
    if (newDet != nullptr && nNew != nullptr && nDets > 0) {
      int lim = (nDets < kMaxDetections) ? nDets : kMaxDetections;
      for (int j = 0; j < lim; ++j) newDet[j] = j;
      *nNew = lim;
    }
    return;
  }
  int nT = (nTracks < kMaxActiveTracks) ? nTracks : kMaxActiveTracks;
  int nD = (nDets < kMaxDetections) ? nDets : kMaxDetections;
  int N = (nT > nD) ? nT : nD;
  if (N > kMaxActiveTracks) N = kMaxActiveTracks;
  if (N < 1) N = 1;
  // Square workspace on stack (N<=64: <=32 KB; documented requirement).
  static double a[64][64];
  static double u[65], v[65];
  static int p[65], way[65];
  static double minv[65];
  static bool used[65];
  const double kSkip = 1.0 - kIouThreshold;  // 0.70: match iff IoU > 0.30
  const double kInf = 1e9;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (i < nT && j < nD) {
        const double c = cost[static_cast<std::size_t>(i) * nD + j];
        a[i][j] = (c >= kInf / 2.0) ? kInf : c;
      } else {
        a[i][j] = kSkip;
      }
    }
  }
  for (int i = 0; i <= N; ++i) {
    u[i] = 0.0;
    v[i] = 0.0;
    p[i] = 0;
    way[i] = 0;
  }
  for (int i = 1; i <= N; ++i) {
    p[0] = i;
    int j0 = 0;
    for (int j = 0; j <= N; ++j) {
      minv[j] = kInf;
      used[j] = false;
    }
    do {
      used[j0] = true;
      const int i0 = p[j0];
      double delta = kInf;
      int j1 = 0;
      for (int j = 1; j <= N; ++j) {
        if (used[j]) continue;
        const double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (int j = 0; j <= N; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }
  // Decode: row i (1-indexed p) -> column Assign.
  bool detUsed[kMaxDetections] = {};
  for (int j = 1; j <= N; ++j) {
    const int i = p[j];
    if (i >= 1 && i <= nT) {
      const int col = j - 1;
      if (col >= 0 && col < nD && a[i - 1][col] < kInf / 2.0 &&
          a[i - 1][col] < kSkip - 1e-12) {
        assign[i - 1] = col;
        detUsed[col] = true;
      }
    }
  }
  if (newDet != nullptr && nNew != nullptr) {
    int c = 0;
    for (int j = 0; j < nD && c < kMaxDetections; ++j) {
      if (!detUsed[j]) newDet[c++] = j;
    }
    *nNew = c;
  }
}

void trackerStep(TrackTable& tt, const Detection* dets, int nDets) noexcept {
  if (dets == nullptr) nDets = 0;
  if (nDets < 0) nDets = 0;
  if (nDets > kMaxDetections) nDets = kMaxDetections;
  // Collect live track indices.
  int tIdx[kMaxActiveTracks];
  int nT = 0;
  for (int i = 0; i < kMaxActiveTracks; ++i) {
    if (tt.tracks[i].active) tIdx[nT++] = i;
  }
  // Cost = 1 - IoU on velocity-predicted boxes; gate at IoU >= 0.30.
  double cost[kMaxActiveTracks * kMaxDetections];
  for (int i = 0; i < nT; ++i) {
    const Track& t = tt.tracks[tIdx[i]];
    const float px0 = t.x0 + t.vx, py0 = t.y0 + t.vy;
    const float px1 = t.x1 + t.vx, py1 = t.y1 + t.vy;
    for (int j = 0; j < nDets; ++j) {
      const double iou = bboxIoU(px0, py0, px1, py1, dets[j].x0, dets[j].y0,
                                dets[j].x1, dets[j].y1);
      cost[static_cast<std::size_t>(i) * nDets + j] =
          (iou >= kIouThreshold) ? (1.0 - iou) : 1e9;
    }
  }
  int assign[kMaxActiveTracks];
  int fresh[kMaxDetections];
  int nFresh = 0;
  hungarianMatch(nT > 0 && nDets > 0 ? cost : nullptr, nT, nDets, assign,
                 fresh, &nFresh);
  bool detHit[kMaxDetections] = {};
  for (int i = 0; i < nT; ++i) {
    Track& t = tt.tracks[tIdx[i]];
    const int j = assign[i];
    if (j >= 0 && j < nDets) {
      detHit[j] = true;
      const float dx = (dets[j].x0 + dets[j].x1) * 0.5f -
                       (t.x0 + t.x1) * 0.5f;
      const float dy = (dets[j].y0 + dets[j].y1) * 0.5f -
                       (t.y0 + t.y1) * 0.5f;
      t.vx = 0.5f * t.vx + 0.5f * dx;
      t.vy = 0.5f * t.vy + 0.5f * dy;
      t.x0 = dets[j].x0;
      t.y0 = dets[j].y0;
      t.x1 = dets[j].x1;
      t.y1 = dets[j].y1;
      t.cls = dets[j].cls;
      t.logOdds += logOddsOf(dets[j].conf);
      t.unseen = 0;
      ++t.hits;
    } else {
      ++t.unseen;
      t.logOdds -= kUnobservedDecay;  // unobserved frame step decay
    }
    if (t.logOdds >= kLogOddsPos) {
      t.status = kTrackVerified;
    } else if (t.logOdds <= kLogOddsNeg) {
      t.status = kTrackRejected;
      t.active = false;  // free capacity; terminal artifact
    } else {
      t.status = kTrackActive;
    }
  }
  // Spawn tracks for unmatched detections.
  for (int f = 0; f < nFresh; ++f) {
    const int j = fresh[f];
    if (j < 0 || j >= nDets || detHit[j]) continue;
    int slot = -1;
    for (int i = 0; i < kMaxActiveTracks; ++i) {
      if (!tt.tracks[i].active) {
        slot = i;
        break;
      }
    }
    if (slot < 0) continue;  // table full: drop (documented)
    Track& t = tt.tracks[slot];
    t.active = true;
    t.id = tt.nextId++;
    t.cls = dets[j].cls;
    t.x0 = dets[j].x0;
    t.y0 = dets[j].y0;
    t.x1 = dets[j].x1;
    t.y1 = dets[j].y1;
    t.vx = 0.0f;
    t.vy = 0.0f;
    t.logOdds = logOddsOf(dets[j].conf);
    t.unseen = 0;
    t.hits = 1;
    if (t.logOdds >= kLogOddsPos) {
      t.status = kTrackVerified;
    } else if (t.logOdds <= kLogOddsNeg) {
      t.status = kTrackRejected;
      t.active = false;
    } else {
      t.status = kTrackActive;
    }
  }
}

void slideCommit(SlideDB& db, const TrackTable& tt) noexcept {
  for (int i = 0; i < kMaxActiveTracks; ++i) {
    const Track& t = tt.tracks[i];
    if (t.status != kTrackVerified) continue;
    // P(t) from log-odds; require >= 0.70.
    const double p = 1.0 / (1.0 + std::exp(-t.logOdds));
    if (p < kSlideConfMin) continue;
    if (t.cls < 0 || t.cls >= kNumTaxa) continue;
    int slot = -1, free = -1;
    for (int s = 0; s < kMaxSlideTracks; ++s) {
      if (db.slots[s].used && db.slots[s].id == t.id) {
        slot = s;
        break;
      }
      if (!db.slots[s].used && free < 0) free = s;
    }
    if (slot >= 0) {
      db.slots[slot].cls = t.cls;  // refresh, no double count
      db.slots[slot].conf = p;
      continue;
    }
    if (free < 0) continue;  // slide DB full: drop (documented)
    db.slots[free].used = true;
    db.slots[free].id = t.id;
    db.slots[free].cls = t.cls;
    db.slots[free].conf = p;
    db.tally[t.cls] += 1;
  }
}

}  // namespace panparasite
