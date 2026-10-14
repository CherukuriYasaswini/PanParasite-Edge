// PanParasite-Edge Phase 6 (Stage 7) unit test: ring + CRC32-C + ledger.
// Covers:
//   (a) CRC32-C check vector ("123456789" -> 0xE3069283, Castagnoli).
//   (b) Sector pack/verify round-trip + corruption rejection.
//   (c) Ring append/scan round-trip, wrap-around, power-loss skip.
//   (d) Audit ledger: append, row count, chain verify, immutability
//       triggers (UPDATE/DELETE abort), forgery detection via failSeq.
//   (e) Zero-PHI UUID format + uniqueness.
//   (f) ZERO heap new/delete across the persistence hot path (tripwire).

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

static std::atomic<long long> g_newCalls{0};
static std::atomic<long long> g_deleteCalls{0};
static std::atomic<bool> g_countHeap{false};

void* operator new(std::size_t n) {
  if (g_countHeap.load(std::memory_order_relaxed)) g_newCalls.fetch_add(1);
  void* p = std::malloc(n);
  if (!p) throw std::bad_alloc();
  return p;
}
void* operator new[](std::size_t n) {
  if (g_countHeap.load(std::memory_order_relaxed)) g_newCalls.fetch_add(1);
  void* p = std::malloc(n);
  if (!p) throw std::bad_alloc();
  return p;
}
void operator delete(void* p) noexcept {
  if (p == nullptr) return;
  if (g_countHeap.load(std::memory_order_relaxed)) g_deleteCalls.fetch_add(1);
  std::free(p);
}
void operator delete[](void* p) noexcept {
  if (p == nullptr) return;
  if (g_countHeap.load(std::memory_order_relaxed)) g_deleteCalls.fetch_add(1);
  std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
  if (p == nullptr) return;
  if (g_countHeap.load(std::memory_order_relaxed)) g_deleteCalls.fetch_add(1);
  std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
  if (p == nullptr) return;
  if (g_countHeap.load(std::memory_order_relaxed)) g_deleteCalls.fetch_add(1);
  std::free(p);
}
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  if (g_countHeap.load(std::memory_order_relaxed)) g_newCalls.fetch_add(1);
  return std::malloc(n);
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
  if (g_countHeap.load(std::memory_order_relaxed)) g_newCalls.fetch_add(1);
  return std::malloc(n);
}
void operator delete(void* p, const std::nothrow_t&) noexcept {
  if (p == nullptr) return;
  if (g_countHeap.load(std::memory_order_relaxed)) g_deleteCalls.fetch_add(1);
  std::free(p);
}
void operator delete[](void* p, const std::nothrow_t&) noexcept {
  if (p == nullptr) return;
  if (g_countHeap.load(std::memory_order_relaxed)) g_deleteCalls.fetch_add(1);
  std::free(p);
}

#include <sqlite3.h>

#include "memory_arena.hpp"
#include "persistence.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond, ...)                                              \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::printf("[FAIL] %s:%d: ", __FILE__, __LINE__);              \
      std::printf(__VA_ARGS__);                                       \
      std::printf("\n");                                              \
      ++g_failures;                                                   \
    } else {                                                          \
      std::printf("[PASS] ");                                         \
      std::printf(__VA_ARGS__);                                       \
      std::printf("\n");                                              \
    }                                                                 \
  } while (0)

inline void doNotOptimizeSink(volatile uint64_t v) {
#if defined(_MSC_VER)
  volatile uint64_t sink = v;
  (void)sink;
#else
  asm volatile("" : "+r"(v) : : "memory");
  (void)v;
#endif
}

}  // namespace

int main() {
  std::printf("=== PanParasite-Edge Phase 6: persistence test ===\n");

  static panparasite::StaticMemoryArena arena;
  CHECK(arena.isInitialized(), "arena initialized at boot");
  CHECK(arena.dioSector() != nullptr && arena.isDioAligned(),
        "DIO staging buffer 4K-aligned");

  // ---- (a) CRC32-C check vector ----
  {
    const uint8_t v[9] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
    const uint32_t c = panparasite::crc32c(v, 9);
    CHECK(c == panparasite::kCrc32cCheck, "CRC32C('123456789')=%08X", c);
    CHECK(panparasite::crc32c(v, 0) == 0u, "CRC32C(empty)=0");
  }
  {
    const uint64_t t0 = panparasite::nowUs();
    const uint64_t t1 = panparasite::nowUs();
    CHECK(t1 >= t0 && t0 > 0, "nowUs monotonic");
  }

  // ---- (b) Sector codec ----
  {
    uint8_t* s = arena.dioSector();
    const uint8_t pl[5] = {'h', 'e', 'l', 'l', 'o'};
    CHECK(panparasite::packSector(s, 7, 123456789u, panparasite::kEvFieldSummary,
                                 pl, 5),
          "packSector ok");
    CHECK(panparasite::verifySector(s), "verifySector accepts own packing");
    s[30] ^= 0xFFu;  // payload corruption (torn write simulation)
    CHECK(!panparasite::verifySector(s), "payload corruption rejected");
    s[30] ^= 0xFFu;
    CHECK(panparasite::verifySector(s), "restored sector valid again");
    s[4095] ^= 0x01u;  // CRC word corruption
    CHECK(!panparasite::verifySector(s), "CRC corruption rejected");
    s[0] = 0x00u;  // magic corruption (virgin/power-loss sector)
    CHECK(!panparasite::verifySector(s), "bad magic rejected");
    CHECK(!panparasite::packSector(s, 1, 0, panparasite::kEvBoot, pl, 5000),
          "oversize payload refused");
  }

  const char* kRing = "test_ring.bin";
  const char* kDb = "test_audit.db";
  std::remove(kRing);
  std::remove(kDb);

  // ---- (c) Ring round-trip + wrap + power-loss skip ----
  {
    CHECK(panparasite::ringFormat(kRing, 8), "ring format 8 sectors");
    panparasite::RingWriter w;
    CHECK(panparasite::ringOpen(w, kRing, 8), "ring open");
    std::printf("  [info] O_DIRECT in effect: %d\n", (int)w.directIo);
    const uint8_t p1[3] = {'a', 'b', 'c'};
    CHECK(panparasite::ringAppend(arena, w, panparasite::kEvFieldSummary, p1, 3,
                                 1000u),
          "append 1");
    CHECK(panparasite::ringAppend(arena, w, panparasite::kEvTrackVerified,
                                 nullptr, 0, 2000u),
          "append 2 (empty payload)");
    uint8_t big[100];
    for (int i = 0; i < 100; ++i) big[i] = static_cast<uint8_t>(i);
    CHECK(panparasite::ringAppend(arena, w, panparasite::kEvBoot, big, 100,
                                 3000u),
          "append 3");
    CHECK(w.nextSeq == 4u && w.head == 3, "writer state advances");
    panparasite::ringClose(w);
    panparasite::RingScanStats st;
    CHECK(panparasite::ringScan(arena, kRing, 8, st), "boot scan runs");
    CHECK(st.valid == 3 && st.valid + st.corrupt == 8 && st.maxSeq == 3u,
          "scan: 3 valid / 5 virgin-skipped / maxSeq 3 (v=%d c=%d m=%u)",
          st.valid, st.corrupt, st.maxSeq);
    // Wrap-around: 10 more appends into the 8-sector ring.
    CHECK(panparasite::ringOpen(w, kRing, 8), "ring reopen");
    for (int i = 0; i < 10; ++i) {
      if (!panparasite::ringAppend(arena, w, panparasite::kEvFieldSummary, p1, 3,
                                  4000u + static_cast<uint64_t>(i))) {
        CHECK(false, "wrap append %d", i);
        break;
      }
    }
    panparasite::ringClose(w);
    CHECK(panparasite::ringScan(arena, kRing, 8, st), "rescan after wrap");
    CHECK(st.valid == 8 && st.maxSeq == 13u, "wrap: 8 valid, maxSeq 13 (v=%d)",
          st.valid);
  }

  // ---- (d) Audit ledger ----
  {
    panparasite::AuditDB a;
    CHECK(panparasite::auditOpen(a, kDb), "audit open");
    CHECK(panparasite::auditAppend(a, panparasite::kEvBoot, "boot:node=pi5", 1u),
          "ledger append 1");
    CHECK(panparasite::auditAppend(a, panparasite::kEvFieldSummary,
                                  "field=0 verified=2", 2u),
          "ledger append 2");
    CHECK(panparasite::auditAppend(a, panparasite::kEvTrackVerified,
                                  "track=7 taxon=0 p=0.93", 3u),
          "ledger append 3");
    CHECK(panparasite::auditRowCount(a) == 3, "row count 3");
    int fail = -99;
    CHECK(panparasite::auditVerifyChain(a, &fail) && fail == -1,
          "chain verifies (fail=%d)", fail);
    // Immutability triggers (raw handle, test-only).
    sqlite3* db = static_cast<sqlite3*>(a.db);
    char* err = nullptr;
    const int ur = sqlite3_exec(db, "UPDATE ledger SET payload='forged';",
                                nullptr, nullptr, &err);
    if (err != nullptr) sqlite3_free(err);
    CHECK(ur != SQLITE_OK, "UPDATE trigger aborts (%d)", ur);
    const int dr = sqlite3_exec(db, "DELETE FROM ledger;", nullptr, nullptr,
                                &err);
    if (err != nullptr) sqlite3_free(err);
    CHECK(dr != SQLITE_OK, "DELETE trigger aborts (%d)", dr);
    CHECK(panparasite::auditRowCount(a) == 3, "rows intact after attacks");
    CHECK(panparasite::auditVerifyChain(a, nullptr), "chain intact");
    // Forgery: bypass the API with a bogus hash row, then detect it.
    const int ir = sqlite3_exec(db,
                                "INSERT INTO ledger(ts,etype,payload,hash)"
                                " VALUES(4,1,'forged',"
                                "'0000000000000000000000000000000000000000000000000000000000000000');",
                                nullptr, nullptr, &err);
    if (err != nullptr) sqlite3_free(err);
    CHECK(ir == SQLITE_OK, "forgery row inserted (setup)");
    int fseq = -1;
    CHECK(!panparasite::auditVerifyChain(a, &fseq) && fseq == 4,
          "forgery detected at seq 4 (fail=%d)", fseq);
    panparasite::auditClose(a);
  }

  // ---- (e) UUID ----
  {
    char u1[37], u2[37];
    CHECK(panparasite::makeSlideUuid(u1) && panparasite::makeSlideUuid(u2),
          "UUIDs generated");
    bool fmt = std::strlen(u1) == 36 && u1[8] == '-' && u1[13] == '-' &&
               u1[18] == '-' && u1[23] == '-' && u1[14] == '4' &&
               (u1[19] == '8' || u1[19] == '9' || u1[19] == 'a' ||
                u1[19] == 'b');
    CHECK(fmt, "UUID v4 format (%s)", u1);
    CHECK(std::strcmp(u1, u2) != 0, "UUIDs unique");
  }

  // ---- (f) Zero-heap hot region ----
  {
    std::remove(kRing);
    g_newCalls.store(0);
    g_deleteCalls.store(0);
    g_countHeap.store(true);
    volatile uint64_t sink = 0;
    {
      sink += panparasite::crc32c(arena.dioSector(), 4096);
      panparasite::packSector(arena.dioSector(), 1, 9u,
                             panparasite::kEvFieldSummary, nullptr, 0);
      sink += panparasite::verifySector(arena.dioSector()) ? 1u : 0u;
      panparasite::RingWriter w;
      if (panparasite::ringOpen(w, kRing, 4)) {
        const uint8_t p[2] = {'x', 'y'};
        if (panparasite::ringAppend(arena, w, panparasite::kEvBoot, p, 2, 5u)) {
          sink += w.nextSeq;
        }
        panparasite::ringClose(w);
      }
      panparasite::RingScanStats st;
      if (panparasite::ringScan(arena, kRing, 4, st)) sink += st.valid;
      panparasite::AuditDB a;
      if (panparasite::auditOpen(a, kDb)) {
        if (panparasite::auditAppend(a, panparasite::kEvFieldSummary, "hot", 6u)) {
          sink += 1u;
        }
        int fs = -1;
        if (panparasite::auditVerifyChain(a, &fs)) sink += 2u;
        sink += static_cast<uint64_t>(panparasite::auditRowCount(a));
        panparasite::auditClose(a);
      }
      char uu[37];
      if (panparasite::makeSlideUuid(uu)) sink += static_cast<uint64_t>(uu[0]);
      sink += panparasite::nowUs() != 0u ? 1u : 0u;
    }
    doNotOptimizeSink(sink);
    g_countHeap.store(false);
    CHECK(g_newCalls.load() == 0, "zero heap new in Stage 7 path (got %lld)",
          g_newCalls.load());
    CHECK(g_deleteCalls.load() == 0, "zero heap delete in Stage 7 path (got %lld)",
          g_deleteCalls.load());
  }

  std::remove(kRing);
  // NOTE: kDb intentionally retained for manual inspection; harmless.

  if (g_failures == 0) {
    std::printf("=== ALL PERSISTENCE TESTS PASSED ===\n");
    return 0;
  }
  std::printf("=== %d PERSISTENCE TEST(S) FAILED ===\n", g_failures);
  return 1;
}
