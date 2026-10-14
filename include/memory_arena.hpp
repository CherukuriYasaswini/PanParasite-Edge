#pragma once

// PanParasite-Edge Phase 1: StaticMemoryArena.
// OPENCODE.md Rules enforced:
//   Rule 1: C++20, cross-platform (Linux ARM64 Pi5 + Windows/x86_64 sim).
//   Rule 2: Zero heap invariant. Everything below is pre-allocated at boot.
//           HOT-PATH SAFE  = inline raw-pointer / cv::Mat-header only,
//                            never malloc/new/vector::push_back/Mat::create.
//           BOOT-ONLY      = may use aligned alloc + mlock once at init.
//   Rule 3: Fixed 1920x1080 uint8 grayscale, S = 0.620 um/pixel.
//
// Memory map (~23.94 MiB total, 25,108,516 bytes):
//   [0] Frame A (ping)            uint8_t [1920*1080]        2,073,600 B
//   [1] Frame B (pong)            uint8_t [1920*1080]        2,073,600 B
//   [2] Full-frame integral       uint32_t[1921*1081]        8,306,404 B
//   [3] Binarized mask            uint8_t [1920*1080]        2,073,600 B
//   [4] Union-Find disjoint arena DisjointSetNode[32768]       786,432 B (24 B/node)
//   [5] Raster equivalence map    uint32_t[1920*1080]        8,294,400 B
//   [6] Anisotropic scratch ROI   uint8_t [128*128]             16,384 B
//   [7] FiLM micro-net params     uint8_t [1480000]          1,480,000 B (mlock/VirtualLock)
//   [8] Direct-I/O sector buffer  uint8_t [4096]                 4,096 B (4KB-aligned)
//   TOTAL                                                   25,108,516 B = 23.9459 MiB
//
// Usage contract:
//   - Declare the arena in STATIC storage (global / function-static) or
//     allocate ONCE at boot with std::make_unique. NEVER on the stack
//     (25 MB overflows the typical 1-8 MB stack).
//   - After init(), Stages 2-5 must only touch raw pointers and the
//     non-owning cv::Mat wrappers below. Those wrappers never allocate:
//     cv::Mat(h, w, type, data_ptr) sets datastart/dataend to the arena
//     pointer with u == nullptr (no refcount, no ownership).

#include <cstddef>
#include <cstdint>

#include <opencv2/core.hpp>

namespace panparasite {

// ------------------------------------------------------------------
// Frame geometry (Rule 3).
// ------------------------------------------------------------------
inline constexpr int kFrameWidth = 1920;
inline constexpr int kFrameHeight = 1080;
inline constexpr std::size_t kFramePixels = 1920u * 1080u;  // 2,073,600
inline constexpr double kMicronsPerPixel = 0.620;           // S

inline constexpr int kIntegralWidth = kFrameWidth + 1;    // 1921
inline constexpr int kIntegralHeight = kFrameHeight + 1;  // 1081
inline constexpr std::size_t kIntegralElems =
    static_cast<std::size_t>(kIntegralWidth) * static_cast<std::size_t>(kIntegralHeight);

// Stage 3 Union-Find capacity.
inline constexpr std::size_t kMaxLabels = 32768u;

// Stage 4 anisotropic scratch ROI (128x128 hot crop).
inline constexpr int kScratchWidth = 128;
inline constexpr int kScratchHeight = 128;
inline constexpr std::size_t kScratchPixels = 128u * 128u;  // 16,384

// Stage 5 FiLM micro-net parameter blob.
inline constexpr std::size_t kFilmParamBytes = 1480000u;

// Stage 7 direct-I/O sector (one 4K sector, 4096-byte aligned).
inline constexpr std::size_t kDioSectorBytes = 4096u;
inline constexpr std::size_t kDioAlignment = 4096u;

// ------------------------------------------------------------------
// Byte accounting (must stay ~23.94 MiB).
// ------------------------------------------------------------------
inline constexpr std::size_t kFrameBytes = kFramePixels * sizeof(uint8_t);
inline constexpr std::size_t kIntegralBytes = kIntegralElems * sizeof(uint32_t);
inline constexpr std::size_t kMaskBytes = kFramePixels * sizeof(uint8_t);
inline constexpr std::size_t kEquivMapBytes = kFramePixels * sizeof(uint32_t);
inline constexpr std::size_t kScratchBytes = kScratchPixels * sizeof(uint8_t);

inline constexpr std::size_t kArenaTotalBytes = 2u * kFrameBytes + kIntegralBytes +
                                                kMaskBytes + 786432u +  // DSU, see below
                                                kEquivMapBytes + kScratchBytes +
                                                kFilmParamBytes + kDioSectorBytes;
static_assert(kArenaTotalBytes == 25108516u,
              "Arena map changed: expected 25,108,516 bytes (~23.9459 MiB)");

// ------------------------------------------------------------------
// Disjoint-set node for the Stage 3 dual-connectivity Union-Find
// homology engine. Must be exactly 24 bytes.
// Layout: parent | rank | size | area | min_x | min_y (6 x 4 B).
// ------------------------------------------------------------------
struct DisjointSetNode {
  uint32_t parent;  // DSU parent label
  uint32_t rank;    // union-by-rank
  uint32_t size;    // component pixel count
  uint32_t area;    // duplicate of size kept for homology stats (keeps 24 B)
  int32_t min_x;    // bounding-box seed x (folded-stat scratch)
  int32_t min_y;    // bounding-box seed y
};
static_assert(sizeof(DisjointSetNode) == 24u, "DisjointSetNode must be 24 bytes");
static_assert(kMaxLabels * sizeof(DisjointSetNode) == 786432u,
              "DSU arena must be 786,432 bytes");

inline constexpr std::size_t kDsuBytes = kMaxLabels * sizeof(DisjointSetNode);

// ------------------------------------------------------------------
// StaticMemoryArena: single pre-allocated owner of all 9 buffers.
// ------------------------------------------------------------------
class StaticMemoryArena {
 public:
  // BOOT-ONLY: allocates the 4K-aligned DIO sector and attempts to
  // mlock/VirtualLock the FiLM blob. Never throws; check isInitialized().
  StaticMemoryArena() noexcept;
  // BOOT-ONLY (shutdown path): unlocks + frees the DIO sector.
  ~StaticMemoryArena() noexcept;

  StaticMemoryArena(const StaticMemoryArena&) = delete;
  StaticMemoryArena& operator=(const StaticMemoryArena&) = delete;
  StaticMemoryArena(StaticMemoryArena&&) = delete;
  StaticMemoryArena& operator=(StaticMemoryArena&&) = delete;

  // BOOT-ONLY: idempotent re-init (frees + reallocates DIO, relocks).
  // Returns false only if the aligned DIO allocation fails.
  // NOTE: the constructor already calls init(), so most call sites do
  // not need to call this. It exists for explicit boot sequencing/tests.
  bool init() noexcept;
  // BOOT-ONLY: unlock + free DIO. After shutdown(), hot-path accessors
  // must not be used until init() succeeds again.
  void shutdown() noexcept;

  [[nodiscard]] bool isInitialized() const noexcept { return initialized_; }
  [[nodiscard]] bool isFilmLocked() const noexcept { return filmLocked_; }
  // HOT-PATH SAFE: pure pointer-arithmetic check, no syscall.
  [[nodiscard]] bool isDioAligned() const noexcept;

  static constexpr std::size_t totalBytes() noexcept { return kArenaTotalBytes; }
  static constexpr double totalMiB() noexcept {
    return static_cast<double>(kArenaTotalBytes) / (1024.0 * 1024.0);
  }

  // ---- Raw accessors: HOT-PATH SAFE (inline, no allocation) ----
  [[nodiscard]] uint8_t* frameA() noexcept { return frame_a_; }
  [[nodiscard]] const uint8_t* frameA() const noexcept { return frame_a_; }
  [[nodiscard]] uint8_t* frameB() noexcept { return frame_b_; }
  [[nodiscard]] const uint8_t* frameB() const noexcept { return frame_b_; }
  [[nodiscard]] uint32_t* integral() noexcept { return integral_; }
  [[nodiscard]] const uint32_t* integral() const noexcept { return integral_; }
  [[nodiscard]] uint8_t* mask() noexcept { return mask_; }
  [[nodiscard]] const uint8_t* mask() const noexcept { return mask_; }
  [[nodiscard]] DisjointSetNode* dsu() noexcept { return dsu_; }
  [[nodiscard]] const DisjointSetNode* dsu() const noexcept { return dsu_; }
  [[nodiscard]] uint32_t* equivMap() noexcept { return equiv_map_; }
  [[nodiscard]] const uint32_t* equivMap() const noexcept { return equiv_map_; }
  [[nodiscard]] uint8_t* scratchRoi() noexcept { return scratch_; }
  [[nodiscard]] const uint8_t* scratchRoi() const noexcept { return scratch_; }
  [[nodiscard]] uint8_t* filmParams() noexcept { return film_params_; }
  [[nodiscard]] const uint8_t* filmParams() const noexcept { return film_params_; }
  [[nodiscard]] uint8_t* dioSector() noexcept { return dio_sector_; }
  [[nodiscard]] const uint8_t* dioSector() const noexcept { return dio_sector_; }

  // ---- Non-owning cv::Mat wrappers: HOT-PATH SAFE ----
  // Each constructs only a Mat header over arena memory (u == nullptr,
  // no data copy, no heap). NEVER call .create() on the result.
  [[nodiscard]] cv::Mat frameAMat() noexcept {
    return cv::Mat(kFrameHeight, kFrameWidth, CV_8UC1, frame_a_);
  }
  [[nodiscard]] cv::Mat frameBMat() noexcept {
    return cv::Mat(kFrameHeight, kFrameWidth, CV_8UC1, frame_b_);
  }
  [[nodiscard]] cv::Mat maskMat() noexcept {
    return cv::Mat(kFrameHeight, kFrameWidth, CV_8UC1, mask_);
  }
  // Full-frame integral image incl. zero border: (H+1)x(W+1), CV_32SC1.
  [[nodiscard]] cv::Mat integralMat() noexcept {
    return cv::Mat(kIntegralHeight, kIntegralWidth, CV_32SC1, integral_);
  }
  // Raster equivalence map viewed as signed int (labels fit in int32).
  [[nodiscard]] cv::Mat equivMapMat() noexcept {
    return cv::Mat(kFrameHeight, kFrameWidth, CV_32SC1, equiv_map_);
  }
  [[nodiscard]] cv::Mat scratchMat() noexcept {
    return cv::Mat(kScratchHeight, kScratchWidth, CV_8UC1, scratch_);
  }
  // Const overloads: view is still non-owning; const_cast is safe here
  // because cv::Mat has no const-data constructor and we never write
  // through a const-view in the hot path.
  [[nodiscard]] cv::Mat frameAMat() const noexcept {
    return cv::Mat(kFrameHeight, kFrameWidth, CV_8UC1,
                   const_cast<uint8_t*>(frame_a_));
  }
  [[nodiscard]] cv::Mat frameBMat() const noexcept {
    return cv::Mat(kFrameHeight, kFrameWidth, CV_8UC1,
                   const_cast<uint8_t*>(frame_b_));
  }
  [[nodiscard]] cv::Mat maskMat() const noexcept {
    return cv::Mat(kFrameHeight, kFrameWidth, CV_8UC1,
                   const_cast<uint8_t*>(mask_));
  }
  [[nodiscard]] cv::Mat integralMat() const noexcept {
    return cv::Mat(kIntegralHeight, kIntegralWidth, CV_32SC1,
                   const_cast<uint32_t*>(integral_));
  }
  [[nodiscard]] cv::Mat equivMapMat() const noexcept {
    return cv::Mat(kFrameHeight, kFrameWidth, CV_32SC1,
                   const_cast<uint32_t*>(equiv_map_));
  }
  [[nodiscard]] cv::Mat scratchMat() const noexcept {
    return cv::Mat(kScratchHeight, kScratchWidth, CV_8UC1,
                   const_cast<uint8_t*>(scratch_));
  }

  // ---- Tiny hot-path helpers (all inline, no allocation) ----
  // Reset DSU parent chain: parent[i] = i. Call once per frame (Stage 3).
  void resetDsuParents() noexcept {
    for (std::size_t i = 0; i < kMaxLabels; ++i) {
      dsu_[i].parent = static_cast<uint32_t>(i);
      dsu_[i].rank = 0;
      dsu_[i].size = 0;
      dsu_[i].area = 0;
      dsu_[i].min_x = 0;
      dsu_[i].min_y = 0;
    }
  }

 private:
  // BOOT-ONLY helpers (may syscall, may touch page tables).
  bool allocateDioSectorBootOnly() noexcept;
  void freeDioSectorBootOnly() noexcept;
  bool lockFilmBootOnly() noexcept;
  void unlockFilmBootOnly() noexcept;

  // Buffers [0]-[7]: inline storage => part of the arena object (BSS
  // when the arena itself has static storage duration). ~25.10 MB.
  uint8_t frame_a_[kFramePixels];      // [0] ping
  uint8_t frame_b_[kFramePixels];      // [1] pong
  uint32_t integral_[kIntegralElems];  // [2] integral (with zero border)
  uint8_t mask_[kFramePixels];         // [3] binarized mask
  DisjointSetNode dsu_[kMaxLabels];    // [4] union-find arena
  uint32_t equiv_map_[kFramePixels];   // [5] raster equivalence map
  uint8_t scratch_[kScratchPixels];    // [6] anisotropic scratch ROI
  uint8_t film_params_[kFilmParamBytes];  // [7] FiLM params (mlocked)

  // Buffer [8]: separately allocated 4K-aligned sector. Pointer only;
  // the 4096 payload lives outside sizeof(*this).
  uint8_t* dio_sector_;  // [8]

  bool initialized_;
  bool filmLocked_;
};

// Compile-time guard: inline storage must hold buffers [0]-[7]; only the
// 4K DIO sector lives outside sizeof (aligned allocation at boot).
// Lower bound is exact; upper bound allows for inter-member padding and
// trailing alignment on any ABI (GCC/Clang/MSVC, ARM64/x86_64).
static_assert(sizeof(StaticMemoryArena) >= kArenaTotalBytes - kDioSectorBytes,
              "StaticMemoryArena inline storage smaller than buffers [0]-[7]");
static_assert(sizeof(StaticMemoryArena) <=
                  kArenaTotalBytes - kDioSectorBytes + 128u,
              "StaticMemoryArena has unexpected padding/bloat");

}  // namespace panparasite
