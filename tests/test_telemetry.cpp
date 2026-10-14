// PanParasite-Edge Phase 6 (Stage 8) unit test: telemetry + UI.
// Covers:
//   (a) Status page: viridis palette, >=56px targets, EN/TE/HI labels,
//       metrics + tally + review form.
//   (b) Review queue: push/confirm/reject, bad input refused, full=false.
//   (c) MQTT topic exact + payload JSON + truncation safety.
//   (d) Drift alarm boundaries (0.10 false / >0.10 true).
//   (e) ECDSA-SHA256 OTA canary: valid signature verifies, bit-flip fails.
//   (f) Embedded server smoke on loopback (bind failure => SKIP, not FAIL):
//       GET / serves the page, POST /review enqueues.
//   (g) ZERO heap new/delete across render/builders/queue (tripwire).

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

#include <openssl/ec.h>
#include <openssl/ecdsa.h>
#include <openssl/obj_mac.h>
#include <openssl/sha.h>

#include "telemetry_ui.hpp"

#if defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

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

void demoSnapshot(panparasite::UiSnapshot& s) {
  std::snprintf(s.slideUuid, sizeof(s.slideUuid), "%s",
                "123e4567-e89b-42d3-a456-426614174000");
  std::snprintf(s.nodeId, sizeof(s.nodeId), "%s", "pi5-edge-01");
  s.fields = 8;
  s.verified = 5;
  s.rejected = 2;
  s.activeTracks = 1;
  for (int k = 0; k < 7; ++k) s.tally[k] = k + 1;
  s.focusPhi = 312.5;
  s.deltaIt = 0.42;
  s.gateLocked = true;
  s.fp32Fallbacks = 1;
  s.capExceeded = true;
}

}  // namespace

int main() {
  std::printf("=== PanParasite-Edge Phase 6: telemetry test ===\n");

  // ---- (a) status page ----
  {
    panparasite::UiSnapshot s;
    demoSnapshot(s);
    char html[16384];
    const int n = panparasite::uiRenderPage(s, html, sizeof(html));
    CHECK(n > 1024 && n < 16384, "page renders (%d B)", n);
    CHECK(std::strstr(html, "#440154") != nullptr &&
              std::strstr(html, "#FDE725") != nullptr &&
              std::strstr(html, "#21918C") != nullptr,
          "viridis palette present");
    CHECK(std::strstr(html, "min-height:56px") != nullptr &&
              std::strstr(html, "min-width:56px") != nullptr,
          "touch targets >= 56px");
    CHECK(std::strstr(html, "lang=\"te\"") != nullptr &&
              std::strstr(html, "lang=\"hi\"") != nullptr,
          "Telugu + Hindi spans present");
    CHECK(std::strstr(html, "CONFIRM") != nullptr &&
              std::strstr(html, "REJECT") != nullptr,
          "CONFIRM/REJECT controls present");
    CHECK(std::strstr(html, "action=\"/review\"") != nullptr,
          "review form posts to /review");
    CHECK(std::strstr(html, "Ascaris lumbricoides") != nullptr &&
              std::strstr(html, "123e4567-e89b-42d3-a456-426614174000") !=
                  nullptr,
          "taxon + slide id rendered");
    CHECK(std::strstr(html, ">5<") != nullptr, "verified count rendered");
    char tiny[64];
    const int nt = panparasite::uiRenderPage(s, tiny, sizeof(tiny));
    CHECK(nt == 63 && tiny[63] == '\0', "truncation stays NUL-safe");
    CHECK(panparasite::uiRenderPage(s, nullptr, 0) == 0, "null render safe");
  }

  // ---- (b) review queue ----
  {
    static panparasite::UiReviewQueue q;
    CHECK(panparasite::uiReviewPush(q, "CONFIRM", 0), "confirm queued");
    CHECK(panparasite::uiReviewPush(q, "REJECT", 7), "reject queued");
    CHECK(!panparasite::uiReviewPush(q, "MAYBE", 1), "bad action refused");
    CHECK(!panparasite::uiReviewPush(q, "CONFIRM", -1), "bad id refused");
    CHECK(!panparasite::uiReviewPush(q, nullptr, 1), "null action refused");
    for (int i = 2; i < 32; ++i) {
      if (!panparasite::uiReviewPush(q, "CONFIRM", i)) {
        CHECK(false, "fill slot %d", i);
        break;
      }
    }
    CHECK(!panparasite::uiReviewPush(q, "CONFIRM", 99), "full queue refuses");
    CHECK(q.count == 32, "queue holds 32");
  }

  // ---- (c) MQTT builders ----
  {
    char topic[128];
    CHECK(panparasite::mqttBuildTopic("pi5-edge-01", topic, sizeof(topic)) &&
              std::strcmp(topic, "telemetry/sth/pi5-edge-01/slide_summary") ==
                  0,
          "topic exact");
    CHECK(!panparasite::mqttBuildTopic("", topic, sizeof(topic)),
          "empty node refused");
    CHECK(!panparasite::mqttBuildTopic("n", topic, 10), "small cap refused");
    int tally[7] = {3, 1, 0, 2, 1, 4, 0};
    char js[512];
    CHECK(panparasite::mqttBuildSlideSummary(
              "pi5-edge-01", "uuid-9", tally, 7, 11, 0.05, js, sizeof(js)),
          "payload builds");
    CHECK(std::strstr(js, "\"tally\":[3,1,0,2,1,4,0]") != nullptr &&
              std::strstr(js, "\"abstain\":0.0500") != nullptr,
          "payload JSON exact (%s)", js);
    CHECK(!panparasite::mqttBuildSlideSummary("n", "u", tally, 1, 1, 2.0, js,
                                             sizeof(js)) ||
              std::strstr(js, "\"abstain\":0.0000") != nullptr,
          "out-of-range abstain clamped");
  }

  // ---- (d) drift alarm ----
  {
    CHECK(!panparasite::cloudwatchDriftAlarm(0.10), "0.10: no alarm");
    CHECK(panparasite::cloudwatchDriftAlarm(0.1001), ">0.10: alarm");
    CHECK(panparasite::slideAbstainRate(0, 0, 0) == 0.0, "zero total safe");
    CHECK(!panparasite::cloudwatchDriftAlarm(
              panparasite::slideAbstainRate(1, 0, 20)),
          "5%% abstain: no alarm");
    CHECK(panparasite::cloudwatchDriftAlarm(
              panparasite::slideAbstainRate(2, 1, 20)),
          "15%% abstain: alarm");
  }

  // ---- (e) OTA canary ----
  {
    EC_KEY* key = EC_KEY_new_by_curve_name(NID_X9_62_prime256v1);
    CHECK(key != nullptr, "P-256 key generated");
    bool okGen = key != nullptr && EC_KEY_generate_key(key) == 1;
    CHECK(okGen, "keypair ready");
    uint8_t digest[32];
    const char* fw = "canary-fw-v1.2.3";
    SHA256(reinterpret_cast<const uint8_t*>(fw), std::strlen(fw), digest);
    uint8_t sig[256];
    unsigned int sigLen = 0;
    const int sr =
        ECDSA_sign(0, digest, 32, sig, &sigLen, key);
    CHECK(sr == 1 && sigLen > 0, "canary signed");
    CHECK(panparasite::otaVerifyCanary(digest, sig, sigLen, key),
          "valid canary verifies");
    sig[sigLen / 2] ^= 0x01u;  // tamper
    CHECK(!panparasite::otaVerifyCanary(digest, sig, sigLen, key),
          "tampered canary fails");
    sig[sigLen / 2] ^= 0x01u;
    uint8_t other[32];
    std::memset(other, 0xAA, sizeof(other));
    CHECK(!panparasite::otaVerifyCanary(other, sig, sigLen, key),
          "wrong digest fails");
    CHECK(!panparasite::otaVerifyCanary(nullptr, sig, sigLen, key),
          "null hash safe");
    CHECK(!panparasite::otaVerifyCanary(digest, nullptr, 0, key),
          "null sig safe");
    CHECK(!panparasite::otaVerifyCanary(digest, sig, sigLen, nullptr),
          "null key safe");
    EC_KEY_free(key);
  }

  // ---- (f) server smoke (loopback; bind failure => SKIP) ----
  {
    static panparasite::UiServer srv;
    static panparasite::UiReviewQueue rq;
    panparasite::UiSnapshot s;
    demoSnapshot(s);
    if (!panparasite::uiStart(srv, rq, 18080)) {
      std::printf("[SKIP] loopback bind unavailable; server paths untested\n");
    } else {
      panparasite::uiPublish(srv, s);
      bool gotPage = false, posted = false;
      for (int attempt = 0; attempt < 2 && !(gotPage && posted); ++attempt) {
#if defined(_WIN32) || defined(_WIN64)
        WSADATA wd;
        const bool wsa = WSAStartup(MAKEWORD(2, 2), &wd) == 0;
        SOCKET c = INVALID_SOCKET;
        if (wsa) c = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        bool conn = false;
        if (c != INVALID_SOCKET) {
          sockaddr_in ad;
          std::memset(&ad, 0, sizeof(ad));
          ad.sin_family = AF_INET;
          ad.sin_port = htons(18080);
          inet_pton(AF_INET, "127.0.0.1", &ad.sin_addr);
          conn = connect(c, reinterpret_cast<sockaddr*>(&ad), sizeof(ad)) == 0;
        }
#define T_CLOSEitiv(c) \
  do {                 \
    if ((c) != INVALID_SOCKET) closesocket(c); \
    WSACleanup();      \
  } while (0)
#else
        int c = socket(AF_INET, SOCK_STREAM, 0);
        bool conn = false;
        if (c >= 0) {
          sockaddr_in ad;
          std::memset(&ad, 0, sizeof(ad));
          ad.sin_family = AF_INET;
          ad.sin_port = htons(18080);
          inet_pton(AF_INET, "127.0.0.1", &ad.sin_addr);
          conn = connect(c, reinterpret_cast<sockaddr*>(&ad), sizeof(ad)) == 0;
        }
#define T_CLOSEitiv(c) \
  do {                 \
    if ((c) >= 0) ::close(c); \
  } while (0)
#endif
        if (conn && !gotPage) {
          const char* get = "GET / HTTP/1.0\r\n\r\n";
#if defined(_WIN32) || defined(_WIN64)
          send(c, get, (int)std::strlen(get), 0);
          char rb[8192];
          const int n = recv(c, rb, (int)sizeof(rb) - 1, 0);
#else
          send(c, get, std::strlen(get), 0);
          char rb[8192];
          const ssize_t n = recv(c, rb, sizeof(rb) - 1, 0);
#endif
          if (n > 0) {
            rb[n] = '\0';
            gotPage = std::strstr(rb, "200 OK") != nullptr &&
                      std::strstr(rb, "PanParasite-Edge Status") != nullptr;
          }
        }
        T_CLOSEitiv(c);
#if defined(_WIN32) || defined(_WIN64)
        const bool wsa2 = WSAStartup(MAKEWORD(2, 2), &wd) == 0;
        SOCKET c2 = INVALID_SOCKET;
        if (wsa2) c2 = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        bool conn2 = false;
        if (c2 != INVALID_SOCKET) {
          sockaddr_in ad;
          std::memset(&ad, 0, sizeof(ad));
          ad.sin_family = AF_INET;
          ad.sin_port = htons(18080);
          inet_pton(AF_INET, "127.0.0.1", &ad.sin_addr);
          conn2 =
              connect(c2, reinterpret_cast<sockaddr*>(&ad), sizeof(ad)) == 0;
        }
        if (conn2 && !posted) {
          const char* body = "action=CONFIRM&id=5";
          char req[256];
          const int rl = std::snprintf(
              req, sizeof(req),
              "POST /review HTTP/1.0\r\nContent-Length: %d\r\n\r\n%s",
              (int)std::strlen(body), body);
          send(c2, req, rl, 0);
          char rb[1024];
          const int n = recv(c2, rb, (int)sizeof(rb) - 1, 0);
          if (n > 0) {
            rb[n] = '\0';
            posted = std::strstr(rb, "303") != nullptr && rq.count == 1 &&
                     rq.slots[0].used && rq.slots[0].candId == 5;
          }
        }
        if (c2 != INVALID_SOCKET) closesocket(c2);
        WSACleanup();
#else
        int c2 = socket(AF_INET, SOCK_STREAM, 0);
        bool conn2 = false;
        if (c2 >= 0) {
          sockaddr_in ad;
          std::memset(&ad, 0, sizeof(ad));
          ad.sin_family = AF_INET;
          ad.sin_port = htons(18080);
          inet_pton(AF_INET, "127.0.0.1", &ad.sin_addr);
          conn2 =
              connect(c2, reinterpret_cast<sockaddr*>(&ad), sizeof(ad)) == 0;
        }
        if (conn2 && !posted) {
          const char* body = "action=CONFIRM&id=5";
          char req[256];
          const int rl = std::snprintf(
              req, sizeof(req),
              "POST /review HTTP/1.0\r\nContent-Length: %d\r\n\r\n%s",
              (int)std::strlen(body), body);
          send(c2, req, static_cast<std::size_t>(rl), 0);
          char rb[1024];
          const ssize_t n = recv(c2, rb, sizeof(rb) - 1, 0);
          if (n > 0) {
            rb[n] = '\0';
            posted = std::strstr(rb, "303") != nullptr && rq.count == 1 &&
                     rq.slots[0].used && rq.slots[0].candId == 5;
          }
        }
        if (c2 >= 0) ::close(c2);
#endif
#undef T_CLOSEitiv
      }
      CHECK(gotPage, "GET / serves status page");
      CHECK(posted, "POST /review enqueues CONFIRM id=5");
      panparasite::uiStop(srv);
      panparasite::uiStop(srv);  // idempotent stop
    }
  }

  // ---- (g) Zero-heap hot region ----
  {
    panparasite::UiSnapshot s;
    demoSnapshot(s);
    static panparasite::UiReviewQueue q2;
    g_newCalls.store(0);
    g_deleteCalls.store(0);
    g_countHeap.store(true);
    volatile uint64_t sink = 0;
    {
      char html[4096];
      sink += static_cast<uint64_t>(
          panparasite::uiRenderPage(s, html, sizeof(html)));
      if (panparasite::uiReviewPush(q2, "REJECT", 4)) sink += 1u;
      char tp[128], js[512];
      int t[7] = {1, 0, 0, 0, 0, 0, 0};
      if (panparasite::mqttBuildTopic("n", tp, sizeof(tp))) sink += tp[0];
      if (panparasite::mqttBuildSlideSummary("n", "u", t, 1, 2, 0.0, js,
                                            sizeof(js))) {
        sink += js[10];
      }
      sink += panparasite::cloudwatchDriftAlarm(0.2) ? 4u : 8u;
      sink += static_cast<uint64_t>(
          panparasite::slideAbstainRate(1, 1, 10) * 1000.0);
    }
    doNotOptimizeSink(sink);
    g_countHeap.store(false);
    CHECK(g_newCalls.load() == 0, "zero heap new in UI path (got %lld)",
          g_newCalls.load());
    CHECK(g_deleteCalls.load() == 0, "zero heap delete in UI path (got %lld)",
          g_deleteCalls.load());
  }

  if (g_failures == 0) {
    std::printf("=== ALL TELEMETRY TESTS PASSED ===\n");
    return 0;
  }
  std::printf("=== %d TELEMETRY TEST(S) FAILED ===\n", g_failures);
  return 1;
}
