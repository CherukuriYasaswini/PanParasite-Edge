#pragma once

// PanParasite-Edge Phase 6 (Stage 7): Direct-I/O Ring Storage & Tamper-Proof
// Audit Ledger.
//
// OPENCODE.md Rules enforced:
//   Rule 1: C++20, Linux ARM64 (Pi 5) + Windows/x86_64 simulation fallback.
//   Rule 2: OUR code below performs no malloc/calloc/new/vector/
//           cv::Mat::create (fixed-size caller storage + arena staging).
//           NOTE: sqlite3/OpenSSL internals may allocate privately; that is
//           outside Stages 2-5 and outside this translation unit's control.
//   Stage 7: 4096-byte aligned Ring Buffer with CRC32-C + SHA-256 ledger.
//
// Buffer discipline (all inside StaticMemoryArena, no heap):
//   dioSector (uint8_t[4096], 4K-aligned): THE staging buffer for every
//     sector read/write. Ring I/O never touches another arena buffer.
//   FiLM blob / frames / mask / DSU / integral / equivMap untouched.
//
// Sector layout (4096 B, little-endian):
//   [0..3]    magic   = 0x50415241 ("PARA")
//   [4..7]    seq     = monotonic uint32
//   [8..15]   ts_us   = wall-clock microseconds, uint64
//   [16..19]  payload_len (<= kRingPayloadMax)
//   [20..21]  event_type (RingEvent)
//   [22..23]  reserved (zero)
//   [24..4091] payload bytes (kRingPayloadMax = 4068)
//   [4092..4095] CRC32-C (Castagnoli) over bytes [0..4091]

#include <cstddef>
#include <cstdint>

#include "memory_arena.hpp"

namespace panparasite {

// ------------------------------------------------------------------
// Ring geometry + sector map.
// ------------------------------------------------------------------
inline constexpr std::size_t kSectorBytes = 4096u;
inline constexpr std::size_t kSectorCrcOff = 4092u;
inline constexpr std::size_t kSectorCrcLen = 4u;
inline constexpr std::size_t kRingHeaderLen = 24u;
inline constexpr std::size_t kRingPayloadMax = kSectorCrcOff - kRingHeaderLen;
inline constexpr uint32_t kRingMagic = 0x50415241u;  // "PARA"
inline constexpr int kRingSectors = 256;             // 1 MiB ring file
static_assert(kRingPayloadMax == 4068u, "payload budget");

// Castagnoli polynomial (reflected) for the software fallback path.
inline constexpr uint32_t kCrc32cPoly = 0x82F63B78u;
inline constexpr uint32_t kCrc32cCheck = 0xE3069283u;  // "123456789"

// Ring event types (fixed vocabulary for the ledger too).
enum RingEvent : uint16_t {
  kEvFieldSummary = 1,
  kEvTrackVerified = 2,
  kEvCapExceeded = 3,
  kEvReview = 4,
  kEvBoot = 5,
};

// Writer state: caller storage (static/global), no heap. fd is a raw OS
// handle (-1 closed). directIo reports whether O_DIRECT is in effect.
struct RingWriter {
  int fd = -1;
  int sectors = 0;
  int head = 0;          // next sector index to write
  uint32_t nextSeq = 1;  // next sector sequence number
  bool directIo = false;
  char path[256] = {};
};

// Boot-scan statistics (fixed, no heap).
struct RingScanStats {
  int sectors = 0;
  int valid = 0;
  int corrupt = 0;
  uint32_t maxSeq = 0;
};

// CRC32-C over [0,n). Uses ARMv8 __crc32cd when __ARM_FEATURE_CRC32 is
// defined, else the exact software table (same digest either way).
// HOT-PATH SAFE w.r.t. heap (pure arithmetic over caller buffers).
uint32_t crc32c(const uint8_t* data, std::size_t n) noexcept;

// Sector codec over the 4096 B staging buffer (arena.dioSector()).
// packSector: fills [0..4095] incl. CRC. verifySector: checks magic+CRC.
bool packSector(uint8_t* sector, uint32_t seq, uint64_t tsUs,
                RingEvent ev, const uint8_t* payload,
                std::size_t payloadLen) noexcept;
bool verifySector(const uint8_t* sector) noexcept;

// Ring file lifecycle (BOOT-ONLY-ish; syscalls, no heap in our code).
// format: (re)creates + preallocates sectors*4096 zero bytes.
// open: opens with O_DIRECT|O_SYNC on Linux, falls back cleanly when the
//   filesystem rejects O_DIRECT (tmpfs/overlayfs) or on Win32 simulation.
bool ringFormat(const char* path, int sectors) noexcept;
bool ringOpen(RingWriter& w, const char* path, int sectors) noexcept;
void ringClose(RingWriter& w) noexcept;
// Append: stages into arena.dioSector(), pwrite()s one sector, advances.
// Returns false on I/O error (writer left consistent; caller may retry).
bool ringAppend(StaticMemoryArena& arena, RingWriter& w, RingEvent ev,
                const uint8_t* payload, std::size_t payloadLen,
                uint64_t tsUs) noexcept;
// Boot scanner: preads every sector through the staging buffer, verifies,
// fills stats. Never throws; corrupt sectors are counted and skipped.
bool ringScan(StaticMemoryArena& arena, const char* path, int sectors,
              RingScanStats& stats) noexcept;

// Wall-clock microseconds (CLOCK_REALTIME / Win32 precise time).
uint64_t nowUs() noexcept;

// ------------------------------------------------------------------
// Cryptographic hash-chained audit ledger (SQLite3).
//   H_k = SHA-256(H_{k-1} || Timestamp_k || EventType_k || Payload_k)
// BEFORE UPDATE / BEFORE DELETE triggers RAISE(ABORT): tamper-evident.
// ------------------------------------------------------------------
struct AuditDB {
  void* db = nullptr;  // sqlite3* (opaque here; persistence.cpp casts)
  bool open = false;
  char path[256] = {};
  uint8_t prevHash[32] = {};  // H_{k-1}, zero genesis
};

// Zero-PHI slide identifier: 128-bit cryptographic random UUID
// (8-4-4-4-12 hex into out[37]). No patient bits, ever.
bool makeSlideUuid(char out[37]) noexcept;

// Ledger lifecycle. Event payloads capped at 512 chars (truncated safely).
bool auditOpen(AuditDB& a, const char* path) noexcept;
void auditClose(AuditDB& a) noexcept;
bool auditAppend(AuditDB& a, RingEvent ev, const char* payload,
                 uint64_t tsUs) noexcept;
// Full-chain re-verification: recomputes every H_k from genesis.
// Returns false on first mismatch (and fills failSeq when non-null).
bool auditVerifyChain(AuditDB& a, int* failSeq) noexcept;
// Row count helper for tests/benchmarks (-1 on error).
int auditRowCount(AuditDB& a) noexcept;

}  // namespace panparasite
