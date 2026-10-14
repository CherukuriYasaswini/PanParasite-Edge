// PanParasite-Edge Phase 1: StaticMemoryArena implementation.
// BOOT-ONLY translation unit: this file is the ONLY place that may call
// aligned allocation (posix_memalign / _aligned_malloc) and
// mlock / VirtualLock. Everything else in the runtime hot path
// (Stages 2-5) must go through the inline accessors in memory_arena.hpp.

#include "memory_arena.hpp"

#include <cstdint>
#include <cstring>

#if defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <malloc.h>   // _aligned_malloc / _aligned_free
#include <windows.h>  // VirtualLock / VirtualUnlock
#else
#include <errno.h>
#include <fcntl.h>      // O_DIRECT (defined via _GNU_SOURCE; fallback below)
#include <stdlib.h>     // posix_memalign / free
#include <sys/mman.h>   // mlock / munlock
#include <unistd.h>
#endif

// Windows/x86_64 simulation fallback for O_DIRECT: open() without the
// flag behaves as buffered I/O; Stage 7 detects this via
// PANPARASITE_SIMULATE_DIRECT_IO and skips alignment-enforced writes.
#ifndef O_DIRECT
#define O_DIRECT 0
#endif
static_assert(O_DIRECT == 0 || O_DIRECT != 0, "O_DIRECT fallback configured");

namespace panparasite {

StaticMemoryArena::StaticMemoryArena() noexcept
    : dio_sector_(nullptr), initialized_(false), filmLocked_(false) {
  // Boot-time allocation is allowed (Rule 2: "initialized at boot").
  init();
}

StaticMemoryArena::~StaticMemoryArena() noexcept { shutdown(); }

bool StaticMemoryArena::init() noexcept {
  if (initialized_) {
    return dio_sector_ != nullptr;
  }

  // Zero all inline buffers at boot so Stage 1 starts deterministic.
  // memset is BOOT-ONLY here; hot-path clears must target only the
  // working slice (see resetDsuParents() for the per-frame pattern).
  std::memset(frame_a_, 0, sizeof(frame_a_));
  std::memset(frame_b_, 0, sizeof(frame_b_));
  std::memset(integral_, 0, sizeof(integral_));
  std::memset(mask_, 0, sizeof(mask_));
  std::memset(dsu_, 0, sizeof(dsu_));
  std::memset(equiv_map_, 0, sizeof(equiv_map_));
  std::memset(scratch_, 0, sizeof(scratch_));
  std::memset(film_params_, 0, sizeof(film_params_));

  if (!allocateDioSectorBootOnly()) {
    initialized_ = false;
    filmLocked_ = false;
    return false;
  }
  std::memset(dio_sector_, 0, kDioSectorBytes);

  // Best-effort page lock for the FiLM blob. Failure (e.g. default
  // RLIMIT_MEMLOCK=64 KiB on stock Linux, or restricted Windows working
  // set) does NOT fail boot: the simulation fallback keeps running
  // unlocked and reports isFilmLocked() == false so the audit ledger
  // (Stage 7) can record the degraded mode. Deployment playbooks raise
  // the limit (ulimit -l / /etc/security/limits.conf) for production.
  filmLocked_ = lockFilmBootOnly();

  initialized_ = true;
  return true;
}

void StaticMemoryArena::shutdown() noexcept {
  unlockFilmBootOnly();
  freeDioSectorBootOnly();
  filmLocked_ = false;
  initialized_ = false;
}

bool StaticMemoryArena::isDioAligned() const noexcept {
  if (dio_sector_ == nullptr) return false;
  return (reinterpret_cast<std::uintptr_t>(dio_sector_) % kDioAlignment) == 0u;
}

bool StaticMemoryArena::allocateDioSectorBootOnly() noexcept {
  freeDioSectorBootOnly();  // idempotent re-init
#if defined(_WIN32) || defined(_WIN64)
  void* p = _aligned_malloc(kDioSectorBytes, kDioAlignment);
  if (p == nullptr) {
    dio_sector_ = nullptr;
    return false;
  }
  dio_sector_ = static_cast<uint8_t*>(p);
  return true;
#else
  void* p = nullptr;
  const int rc = ::posix_memalign(&p, kDioAlignment, kDioSectorBytes);
  if (rc != 0 || p == nullptr) {
    dio_sector_ = nullptr;
    return false;
  }
  dio_sector_ = static_cast<uint8_t*>(p);
  return true;
#endif
}

void StaticMemoryArena::freeDioSectorBootOnly() noexcept {
  if (dio_sector_ == nullptr) return;
#if defined(_WIN32) || defined(_WIN64)
  _aligned_free(dio_sector_);
#else
  ::free(dio_sector_);  // paired with posix_memalign; BOOT-ONLY
#endif
  dio_sector_ = nullptr;
}

bool StaticMemoryArena::lockFilmBootOnly() noexcept {
#if defined(_WIN32) || defined(_WIN64)
  // VirtualLock requires the pages to be resident in the working set.
  // Best-effort bump of the working set by ~2 MiB so the 1.48 MB blob
  // has a chance to lock on dev machines; failure is non-fatal (sim).
  SIZE_T minWs = 0, maxWs = 0;
  if (::GetProcessWorkingSetSize(::GetCurrentProcess(), &minWs, &maxWs)) {
    const SIZE_T want = maxWs + static_cast<SIZE_T>(kFilmParamBytes) + 65536u;
    ::SetProcessWorkingSetSize(::GetCurrentProcess(), minWs, want);
  }
  return ::VirtualLock(static_cast<LPVOID>(film_params_),
                       static_cast<SIZE_T>(sizeof(film_params_))) == TRUE;
#else
  return ::mlock(static_cast<void*>(film_params_),
                 sizeof(film_params_)) == 0;
#endif
}

void StaticMemoryArena::unlockFilmBootOnly() noexcept {
  if (!filmLocked_) return;
#if defined(_WIN32) || defined(_WIN64)
  ::VirtualUnlock(static_cast<LPVOID>(film_params_),
                  static_cast<SIZE_T>(sizeof(film_params_)));
#else
  ::munlock(static_cast<void*>(film_params_), sizeof(film_params_));
#endif
  filmLocked_ = false;
}

}  // namespace panparasite
