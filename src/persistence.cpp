// PanParasite-Edge Phase 6 (Stage 7) implementation: ring + ledger.
// OUR code below performs no malloc/calloc/new/vector/cv::Mat::create:
// fixed-size caller storage, arena staging, stack buffers, snprintf.
// sqlite3/OpenSSL internals may allocate privately (outside Stages 2-5).

#include "persistence.hpp"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>

#include <openssl/evp.h>
#include <openssl/rand.h>
#include <sqlite3.h>

#if defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#endif

#if defined(__ARM_FEATURE_CRC32) && !defined(_MSC_VER)
#include <arm_acle.h>
#endif

namespace panparasite {
namespace {

// ------------------------------------------------------------------
// Little-endian codec (memcpy: no alignment UB, no heap).
// ------------------------------------------------------------------
inline void putU32(uint8_t* p, uint32_t v) noexcept {
  p[0] = static_cast<uint8_t>(v);
  p[1] = static_cast<uint8_t>(v >> 8);
  p[2] = static_cast<uint8_t>(v >> 16);
  p[3] = static_cast<uint8_t>(v >> 24);
}
inline void putU64(uint8_t* p, uint64_t v) noexcept {
  for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
}
inline uint32_t getU32(const uint8_t* p) noexcept {
  // Explicit little-endian assembly: portable, no alignment UB.
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

// ------------------------------------------------------------------
// CRC32-C software table (consteval: zero runtime cost, no heap).
// ------------------------------------------------------------------
consteval std::array<uint32_t, 256> buildCrc32cTable() {
  std::array<uint32_t, 256> t{};
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int k = 0; k < 8; ++k) {
      c = (c & 1u) ? (kCrc32cPoly ^ (c >> 1)) : (c >> 1);
    }
    t[i] = c;
  }
  return t;
}
inline constexpr std::array<uint32_t, 256> kCrcTable = buildCrc32cTable();

uint32_t crc32cSoftware(const uint8_t* data, std::size_t n) noexcept {
  uint32_t crc = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < n; ++i) {
    crc = kCrcTable[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

// ------------------------------------------------------------------
// OS file shims (heap-free). Serialized use; no concurrent fd sharing.
// ------------------------------------------------------------------
#if defined(_WIN32) || defined(_WIN64)
inline int platOpen(const char* path, bool truncate) noexcept {
  int flags = _O_RDWR | _O_CREAT | _O_BINARY;
  if (truncate) flags |= _O_TRUNC;
  int fd = 0;
  if (_sopen_s(&fd, path, flags, _SH_DENYNO, _S_IREAD | _S_IWRITE) != 0) {
    return -1;
  }
  return fd;
}
inline void platClose(int fd) noexcept {
  if (fd >= 0) _close(fd);
}
inline bool platTruncate(int fd, int64_t len) noexcept {
  return _chsize_s(fd, len) == 0;
}
inline bool platWriteAt(int fd, int64_t off, const uint8_t* buf,
                        std::size_t n) noexcept {
  std::size_t done = 0;
  while (done < n) {
    if (_lseeki64(fd, off + static_cast<int64_t>(done), SEEK_SET) < 0) {
      return false;
    }
    const unsigned chunk =
        (n - done > 65536u) ? 65536u : static_cast<unsigned>(n - done);
    const int w = _write(fd, buf + done, chunk);
    if (w <= 0) return false;
    done += static_cast<std::size_t>(w);
  }
  return true;
}
inline bool platReadAt(int fd, int64_t off, uint8_t* buf,
                       std::size_t n) noexcept {
  std::size_t done = 0;
  while (done < n) {
    if (_lseeki64(fd, off + static_cast<int64_t>(done), SEEK_SET) < 0) {
      return false;
    }
    const unsigned chunk =
        (n - done > 65536u) ? 65536u : static_cast<unsigned>(n - done);
    const int r = _read(fd, buf + done, chunk);
    if (r <= 0) return false;
    done += static_cast<std::size_t>(r);
  }
  return true;
}
#else
inline int platOpenDirect(const char* path, bool truncate) noexcept {
#if defined(PANPARASITE_SIMULATE_DIRECT_IO)
  (void)truncate;
  return -1;  // simulation build: force the clean fallback path
#else
  int flags = O_RDWR | O_CREAT | O_DIRECT | O_SYNC;
  if (truncate) flags |= O_TRUNC;
  return ::open(path, flags, 0644);
#endif
}
inline int platOpenPlain(const char* path, bool truncate) noexcept {
  int flags = O_RDWR | O_CREAT;
  if (truncate) flags |= O_TRUNC;
  return ::open(path, flags, 0644);
}
inline void platClose(int fd) noexcept {
  if (fd >= 0) ::close(fd);
}
inline bool platTruncate(int fd, int64_t len) noexcept {
  return ::ftruncate(fd, static_cast<off_t>(len)) == 0;
}
inline bool platWriteAt(int fd, int64_t off, const uint8_t* buf,
                        std::size_t n) noexcept {
  std::size_t done = 0;
  while (done < n) {
    const ssize_t w = ::pwrite(fd, buf + done, n - done,
                               static_cast<off_t>(off + done));
    if (w < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (w == 0) return false;
    done += static_cast<std::size_t>(w);
  }
  return true;
}
inline bool platReadAt(int fd, int64_t off, uint8_t* buf,
                       std::size_t n) noexcept {
  std::size_t done = 0;
  while (done < n) {
    const ssize_t r =
        ::pread(fd, buf + done, n - done, static_cast<off_t>(off + done));
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (r == 0) return false;
    done += static_cast<std::size_t>(r);
  }
  return true;
}
#endif

inline sqlite3* adb(AuditDB& a) noexcept {
  return static_cast<sqlite3*>(a.db);
}

inline void hex32(const uint8_t h[32], char out[65]) noexcept {
  static const char* kHex = "0123456789abcdef";
  for (int i = 0; i < 32; ++i) {
    out[2 * i] = kHex[h[i] >> 4];
    out[2 * i + 1] = kHex[h[i] & 0xF];
  }
  out[64] = '\0';
}

inline bool unhex32(const char* hex, uint8_t out[32]) noexcept {
  if (hex == nullptr) return false;
  for (int i = 0; i < 32; ++i) {
    unsigned v = 0;
    for (int k = 0; k < 2; ++k) {
      const char c = hex[2 * i + k];
      unsigned d = 0;
      if (c >= '0' && c <= '9') d = static_cast<unsigned>(c - '0');
      else if (c >= 'a' && c <= 'f') d = static_cast<unsigned>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') d = static_cast<unsigned>(c - 'A' + 10);
      else return false;
      v = (v << 4) | d;
    }
    out[i] = static_cast<uint8_t>(v);
  }
  return true;
}

// H = SHA-256(prev || ts_le64 || ev_le16 || payload). EVP API: works on
// OpenSSL 1.1.1 and 3.x without deprecation warnings. No heap of ours.
inline bool chainHash(const uint8_t prev[32], uint64_t tsUs, uint16_t ev,
                      const uint8_t* payload, std::size_t n,
                      uint8_t out[32]) noexcept {
  EVP_MD_CTX* ctx = EVP_MD_CTX_new();
  if (ctx == nullptr) return false;
  uint8_t tsb[8];
  putU64(tsb, tsUs);
  const uint8_t evb[2] = {static_cast<uint8_t>(ev),
                          static_cast<uint8_t>(ev >> 8)};
  bool ok = EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1 &&
            EVP_DigestUpdate(ctx, prev, 32) == 1 &&
            EVP_DigestUpdate(ctx, tsb, 8) == 1 &&
            EVP_DigestUpdate(ctx, evb, 2) == 1 &&
            (n == 0 || payload == nullptr ||
             EVP_DigestUpdate(ctx, payload, n) == 1);
  unsigned int len = 0;
  if (ok) ok = EVP_DigestFinal_ex(ctx, out, &len) == 1 && len == 32u;
  EVP_MD_CTX_free(ctx);
  return ok;
}

}  // namespace

uint32_t crc32c(const uint8_t* data, std::size_t n) noexcept {
  if (data == nullptr) return 0u;
#if defined(__ARM_FEATURE_CRC32) && !defined(_MSC_VER)
  // ARMv8 CRC32-C (Castagnoli): __crc32cd folds 8 bytes at a time.
  uint64_t crc = 0xFFFFFFFFu;
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    uint64_t w = 0;
    std::memcpy(&w, data + i, 8);
    crc = __crc32cd(static_cast<uint32_t>(crc), w);
  }
  uint32_t c32 = static_cast<uint32_t>(crc);
  for (; i < n; ++i) c32 = __crc32cb(c32, data[i]);
  return c32 ^ 0xFFFFFFFFu;
#else
  return crc32cSoftware(data, n);
#endif
}

bool packSector(uint8_t* sector, uint32_t seq, uint64_t tsUs, RingEvent ev,
                const uint8_t* payload, std::size_t payloadLen) noexcept {
  if (sector == nullptr) return false;
  if (payloadLen > kRingPayloadMax) return false;
  if (payloadLen > 0 && payload == nullptr) return false;
  putU32(sector + 0, kRingMagic);
  putU32(sector + 4, seq);
  putU64(sector + 8, tsUs);
  putU32(sector + 16, static_cast<uint32_t>(payloadLen));
  sector[20] = static_cast<uint8_t>(ev);
  sector[21] = static_cast<uint8_t>(ev >> 8);
  sector[22] = 0;
  sector[23] = 0;
  for (std::size_t i = 0; i < kRingPayloadMax; ++i) sector[24 + i] = 0;
  for (std::size_t i = 0; i < payloadLen; ++i) sector[24 + i] = payload[i];
  putU32(sector + kSectorCrcOff, crc32c(sector, kSectorCrcOff));
  return true;
}

bool verifySector(const uint8_t* sector) noexcept {
  if (sector == nullptr) return false;
  if (getU32(sector + 0) != kRingMagic) return false;
  const uint32_t want = getU32(sector + kSectorCrcOff);
  if (crc32c(sector, kSectorCrcOff) != want) return false;
  if (getU32(sector + 16) > kRingPayloadMax) return false;
  return true;
}

bool ringFormat(const char* path, int sectors) noexcept {
  if (path == nullptr || sectors <= 0) return false;
#if defined(_WIN32) || defined(_WIN64)
  const int fd = platOpen(path, true);
  if (fd < 0) return false;
  const bool ok = platTruncate(fd, static_cast<int64_t>(sectors) * 4096);
  platClose(fd);
  return ok;
#else
  int fd = platOpenDirect(path, true);
  if (fd < 0) fd = platOpenPlain(path, true);
  if (fd < 0) return false;
  const bool ok = platTruncate(fd, static_cast<int64_t>(sectors) * 4096);
  platClose(fd);
  return ok;
#endif
}

bool ringOpen(RingWriter& w, const char* path, int sectors) noexcept {
  if (path == nullptr || sectors <= 0) return false;
  ringClose(w);
  std::snprintf(w.path, sizeof(w.path), "%s", path);
#if defined(_WIN32) || defined(_WIN64)
  const int fd = platOpen(path, false);
  if (fd < 0) return false;
  w.fd = fd;
  w.directIo = false;  // Win32 simulation: buffered I/O, same layout
#else
  int fd = platOpenDirect(path, false);
  w.directIo = (fd >= 0);
  if (fd < 0) {
    fd = platOpenPlain(path, false);  // clean fallback (tmpfs/overlayfs)
    if (fd < 0) return false;
  }
  w.fd = fd;
#endif
  w.sectors = sectors;
  w.head = 0;
  w.nextSeq = 1;
  return true;
}

void ringClose(RingWriter& w) noexcept {
  platClose(w.fd);
  w.fd = -1;
  w.sectors = 0;
  w.head = 0;
}

bool ringAppend(StaticMemoryArena& arena, RingWriter& w, RingEvent ev,
                const uint8_t* payload, std::size_t payloadLen,
                uint64_t tsUs) noexcept {
  if (w.fd < 0 || w.sectors <= 0) return false;
  uint8_t* staging = arena.dioSector();  // THE 4K-aligned staging buffer
  if (staging == nullptr) return false;
  if (!packSector(staging, w.nextSeq, tsUs, ev, payload, payloadLen)) {
    return false;
  }
  const int64_t off = static_cast<int64_t>(w.head) * 4096;
  if (!platWriteAt(w.fd, off, staging, kSectorBytes)) return false;
  w.head = (w.head + 1) % w.sectors;
  ++w.nextSeq;
  return true;
}

bool ringScan(StaticMemoryArena& arena, const char* path, int sectors,
              RingScanStats& stats) noexcept {
  stats.sectors = sectors;
  stats.valid = 0;
  stats.corrupt = 0;
  stats.maxSeq = 0;
  if (path == nullptr || sectors <= 0) return false;
  RingWriter w;
  if (!ringOpen(w, path, sectors)) return false;
  uint8_t* staging = arena.dioSector();
  if (staging == nullptr) {
    ringClose(w);
    return false;
  }
  for (int i = 0; i < sectors; ++i) {
    const int64_t off = static_cast<int64_t>(i) * 4096;
    if (!platReadAt(w.fd, off, staging, kSectorBytes)) {
      ringClose(w);
      return false;
    }
    if (verifySector(staging)) {
      ++stats.valid;
      const uint32_t seq = getU32(staging + 4);
      if (seq > stats.maxSeq) stats.maxSeq = seq;
    } else {
      ++stats.corrupt;  // power-loss torn write: counted and skipped
    }
  }
  ringClose(w);
  return true;
}

uint64_t nowUs() noexcept {
#if defined(_WIN32) || defined(_WIN64)
  FILETIME ft;
  GetSystemTimePreciseAsFileTime(&ft);
  ULARGE_INTEGER u;
  u.LowPart = ft.dwLowDateTime;
  u.HighPart = ft.dwHighDateTime;
  // 100-ns ticks since 1601-01-01 -> us since 1970-01-01.
  return u.QuadPart / 10u - 11644473600000000u;
#else
  struct timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) != 0) return 0u;
  return static_cast<uint64_t>(ts.tv_sec) * 1000000u +
         static_cast<uint64_t>(ts.tv_nsec) / 1000u;
#endif
}

bool makeSlideUuid(char out[37]) noexcept {
  if (out == nullptr) return false;
  uint8_t b[16];
  if (RAND_bytes(b, 16) != 1) return false;
  b[6] = static_cast<uint8_t>((b[6] & 0x0F) | 0x40);  // version 4
  b[8] = static_cast<uint8_t>((b[8] & 0x3F) | 0x80);  // variant 10
  std::snprintf(out, 37,
                "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9],
                b[10], b[11], b[12], b[13], b[14], b[15]);
  return true;
}

namespace {

int execNoRows(sqlite3* db, const char* sql) noexcept {
  char* err = nullptr;
  const int rc = sqlite3_exec(db, sql, nullptr, nullptr, &err);
  if (err != nullptr) sqlite3_free(err);  // sqlite-owned heap, freed by API
  return rc;
}

}  // namespace

bool auditOpen(AuditDB& a, const char* path) noexcept {
  if (path == nullptr) return false;
  auditClose(a);
  std::snprintf(a.path, sizeof(a.path), "%s", path);
  sqlite3* db = nullptr;
  if (sqlite3_open(path, &db) != SQLITE_OK) {
    if (db != nullptr) sqlite3_close(db);
    return false;
  }
  a.db = db;
  static const char* kSchema =
      "CREATE TABLE IF NOT EXISTS ledger("
      "seq INTEGER PRIMARY KEY AUTOINCREMENT,"
      "ts INTEGER NOT NULL,etype INTEGER NOT NULL,"
      "payload TEXT NOT NULL,hash TEXT NOT NULL);"
      "CREATE TRIGGER IF NOT EXISTS no_update BEFORE UPDATE ON ledger "
      "BEGIN SELECT RAISE(ABORT,'ledger immutable'); END;"
      "CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON ledger "
      "BEGIN SELECT RAISE(ABORT,'ledger immutable'); END;";
  if (execNoRows(db, kSchema) != SQLITE_OK) {
    auditClose(a);
    return false;
  }
  // Resume hash chain from the last committed row (reboot-safe).
  for (int i = 0; i < 32; ++i) a.prevHash[i] = 0;
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(db,
                         "SELECT hash FROM ledger ORDER BY seq DESC LIMIT 1;",
                         -1, &st, nullptr) == SQLITE_OK) {
    if (sqlite3_step(st) == SQLITE_ROW) {
      const unsigned char* hx = sqlite3_column_text(st, 0);
      if (hx != nullptr) {
        unhex32(reinterpret_cast<const char*>(hx), a.prevHash);
      }
    }
    sqlite3_finalize(st);
  }
  a.open = true;
  return true;
}

void auditClose(AuditDB& a) noexcept {
  if (a.db != nullptr) {
    sqlite3_close(static_cast<sqlite3*>(a.db));
    a.db = nullptr;
  }
  a.open = false;
}

bool auditAppend(AuditDB& a, RingEvent ev, const char* payload,
                 uint64_t tsUs) noexcept {
  sqlite3* db = adb(a);
  if (db == nullptr || !a.open) return false;
  char pbuf[512];
  if (payload == nullptr) {
    pbuf[0] = '\0';
  } else {
    std::snprintf(pbuf, sizeof(pbuf), "%s", payload);
  }
  std::size_t n = 0;  // bounded strlen (heap-free, MSVC-portable)
  while (n < sizeof(pbuf) && pbuf[n] != '\0') ++n;
  uint8_t h[32];
  if (!chainHash(a.prevHash, tsUs, static_cast<uint16_t>(ev),
                 reinterpret_cast<const uint8_t*>(pbuf), n, h)) {
    return false;
  }
  char hex[65];
  hex32(h, hex);
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(db,
                         "INSERT INTO ledger(ts,etype,payload,hash)"
                         " VALUES(?,?,?,?);",
                         -1, &st, nullptr) != SQLITE_OK) {
    return false;
  }
  bool ok = sqlite3_bind_int64(st, 1, static_cast<sqlite3_int64>(tsUs)) ==
                SQLITE_OK &&
            sqlite3_bind_int(st, 2, static_cast<int>(ev)) == SQLITE_OK &&
            sqlite3_bind_text(st, 3, pbuf, static_cast<int>(n),
                              SQLITE_TRANSIENT) == SQLITE_OK &&
            sqlite3_bind_text(st, 4, hex, 64, SQLITE_TRANSIENT) == SQLITE_OK &&
            sqlite3_step(st) == SQLITE_DONE;
  sqlite3_finalize(st);
  if (!ok) return false;
  std::memcpy(a.prevHash, h, 32);
  return true;
}

bool auditVerifyChain(AuditDB& a, int* failSeq) noexcept {
  sqlite3* db = adb(a);
  if (db == nullptr || !a.open) return false;
  if (failSeq != nullptr) *failSeq = -1;
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(db,
                         "SELECT seq,ts,etype,payload,hash FROM ledger"
                         " ORDER BY seq ASC;",
                         -1, &st, nullptr) != SQLITE_OK) {
    return false;
  }
  uint8_t prev[32] = {};
  bool ok = true;
  for (;;) {
    const int rc = sqlite3_step(st);
    if (rc == SQLITE_DONE) break;
    if (rc != SQLITE_ROW) {
      ok = false;
      break;
    }
    const int seq = sqlite3_column_int(st, 0);
    const auto ts = static_cast<uint64_t>(sqlite3_column_int64(st, 1));
    const int et = sqlite3_column_int(st, 2);
    const unsigned char* pt =
        static_cast<const unsigned char*>(sqlite3_column_text(st, 3));
    const int pn = sqlite3_column_bytes(st, 3);
    const unsigned char* hx =
        static_cast<const unsigned char*>(sqlite3_column_text(st, 4));
    uint8_t h[32];
    if (!chainHash(prev, ts, static_cast<uint16_t>(et), pt,
                   pn < 0 ? 0u : static_cast<std::size_t>(pn), h)) {
      ok = false;
      if (failSeq != nullptr) *failSeq = seq;
      break;
    }
    char hex[65];
    hex32(h, hex);
    if (hx == nullptr ||
        std::strncmp(hex, reinterpret_cast<const char*>(hx), 64) != 0) {
      ok = false;  // hash mismatch: tamper evidence
      if (failSeq != nullptr) *failSeq = seq;
      break;
    }
    std::memcpy(prev, h, 32);
  }
  sqlite3_finalize(st);
  return ok;
}

int auditRowCount(AuditDB& a) noexcept {
  sqlite3* db = adb(a);
  if (db == nullptr || !a.open) return -1;
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM ledger;", -1, &st,
                         nullptr) != SQLITE_OK) {
    return -1;
  }
  int n = -1;
  if (sqlite3_step(st) == SQLITE_ROW) n = sqlite3_column_int(st, 0);
  sqlite3_finalize(st);
  return n;
}

}  // namespace panparasite
