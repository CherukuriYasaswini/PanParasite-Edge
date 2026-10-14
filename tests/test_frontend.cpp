// MICROPARA frontend test: static blueprint markers, JSON bridge pure
// functions, and live server routes (/app.js, /api/status, /api/review).
// Covers Revision 8.2 modifications #1-#11 structurally (markers), the
// backend bridge contract, and the zero-heap tripwire for the new code.
// Server/file sections SKIP (not fail) when loopback or web/ is missing.

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

// Read a web/ source file through candidate roots (heap-free stdio is
// avoided: fopen's FILE uses libc malloc, invisible to the new/delete
// tripwire, and this runs outside counted regions anyway).
bool readWebFile(const char* name, char* out, std::size_t cap,
                 std::size_t& len) {
  static const char* kRoots[3] = {"web/", "../web/", "../../web/"};
  len = 0;
  for (int r = 0; r < 3; ++r) {
    char path[64];
    std::snprintf(path, sizeof(path), "%s%s", kRoots[r], name);
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) continue;
    len = std::fread(out, 1, cap - 1, f);
    std::fclose(f);
    out[len] = '\0';
    return true;
  }
  return false;
}

bool hasAll(const char* hay, const char* const* needles, int n) {
  for (int i = 0; i < n; ++i) {
    if (std::strstr(hay, needles[i]) == nullptr) {
      std::printf("  [info] missing marker: %s\n", needles[i]);
      return false;
    }
  }
  return true;
}

// Minimal blocking HTTP exchange over loopback. Returns reply bytes or -1.
int httpExchange(int port, const char* req, char* rep, std::size_t cap) {
#if defined(_WIN32) || defined(_WIN64)
  WSADATA wd;
  if (WSAStartup(MAKEWORD(2, 2), &wd) != 0) return -1;
  SOCKET c = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  int out = -1;
  if (c != INVALID_SOCKET) {
    sockaddr_in ad;
    std::memset(&ad, 0, sizeof(ad));
    ad.sin_family = AF_INET;
    ad.sin_port = htons(static_cast<uint16_t>(port));
    inet_pton(AF_INET, "127.0.0.1", &ad.sin_addr);
    if (connect(c, reinterpret_cast<sockaddr*>(&ad), sizeof(ad)) == 0) {
      send(c, req, (int)std::strlen(req), 0);
      const int n = recv(c, rep, (int)cap - 1, 0);
      if (n > 0) {
        rep[n] = '\0';
        out = n;
      }
    }
    closesocket(c);
  }
  WSACleanup();
  return out;
#else
  int c = socket(AF_INET, SOCK_STREAM, 0);
  int out = -1;
  if (c >= 0) {
    sockaddr_in ad;
    std::memset(&ad, 0, sizeof(ad));
    ad.sin_family = AF_INET;
    ad.sin_port = htons(static_cast<uint16_t>(port));
    inet_pton(AF_INET, "127.0.0.1", &ad.sin_addr);
    if (connect(c, reinterpret_cast<sockaddr*>(&ad), sizeof(ad)) == 0) {
      send(c, req, std::strlen(req), 0);
      const ssize_t n = recv(c, rep, cap - 1, 0);
      if (n > 0) {
        rep[n] = '\0';
        out = (int)n;
      }
    }
    ::close(c);
  }
  return out;
#endif
}

}  // namespace

int main() {
  std::printf("=== MICROPARA frontend test ===\n");

  // ---- (a) static blueprint markers (#1-#11) ----
  {
    char buf[65536];
    std::size_t len = 0;
    if (!readWebFile("index.html", buf, sizeof(buf), len)) {
      std::printf("[SKIP] web/ checkout not found; static markers untested\n");
    } else {
      static const char* kHtml[] = {
          "MICROPARA", "SUSPEND ACTIVE SESSION", "OFFLINE | SYSTEM READY",
          "densityBanner", "minimap", "contrastBadge", "techSheet",
          "verifyCheck", "wf-step", "historyTable", "diagBody", "setLang",
          "PanParasite-Edge Status"};
      CHECK(hasAll(buf, kHtml, 13), "index.html carries blueprint markers");
      if (!readWebFile("styles.css", buf, sizeof(buf), len)) {
        CHECK(false, "styles.css present");
      } else {
        static const char* kCss[] = {"#D97706", "min-height:56px",
                                     "min-width:56px", "16%", "84%", "8vh",
                                     "10vh", "#techSheet"};
        CHECK(hasAll(buf, kCss, 8), "styles.css carries theme tokens");
      }
      if (!readWebFile("app.js", buf, sizeof(buf), len)) {
        CHECK(false, "app.js present");
      } else {
        static const char* kJs[] = {
            "/api/status", "/api/review", "Target Contrast Enhanced",
            "High candidate density", "localStorage", "minimap",
            "Technical Details", "FINALIZE", "Other Helminth",
            "Dismiss as Artifact", "taxon-override"};
        CHECK(hasAll(buf, kJs, 11), "app.js carries bridge + mods");
      }
    }
  }

  // ---- (b) JSON bridge pure functions ----
  {
    panparasite::UiSnapshot s;
    std::snprintf(s.slideUuid, sizeof(s.slideUuid), "%s", "slide-1");
    std::snprintf(s.nodeId, sizeof(s.nodeId), "%s", "n1");
    s.fields = 3;
    s.verified = 2;
    s.tally[0] = 2;
    s.gateLocked = true;
    s.capExceeded = true;
    s.ringDirectIo = true;
    s.numCands = 2;
    s.cands[0].x0 = 10;
    s.cands[0].y0 = 20;
    s.cands[0].x1 = 30;
    s.cands[0].y1 = 40;
    s.cands[0].cls = 3;
    s.cands[0].conf = 0.63;
    s.cands[0].trig = true;
    s.cands[1].cls = 5;
    s.cands[1].conf = 0.2;
    char js[4096];
    const int n = panparasite::uiRenderStatusJson(s, js, sizeof(js));
    CHECK(n > 100 && n < 4096, "status JSON renders (%d B)", n);
    CHECK(std::strstr(js, "\"slide\":\"slide-1\"") != nullptr &&
              std::strstr(js, "\"tally\":[2,0,0,0,0,0,0]") != nullptr &&
              std::strstr(js, "\"locked\":1") != nullptr &&
              std::strstr(js, "\"cap\":1") != nullptr &&
              std::strstr(js, "\"ringDirectIo\":1") != nullptr,
          "status JSON keys");
    CHECK(std::strstr(js, "\"cls\":3") != nullptr &&
              std::strstr(js, "\"conf\":0.6300") != nullptr &&
              std::strstr(js, "\"trig\":1") != nullptr,
          "candidate array serializes");
    char tiny[32];
    const int nt = panparasite::uiRenderStatusJson(s, tiny, sizeof(tiny));
    CHECK(nt == 31 && tiny[31] == '\0', "JSON truncation NUL-safe");
    CHECK(panparasite::uiRenderStatusJson(s, nullptr, 0) == 0, "null safe");
    char act[8] = {};
    int id = -1;
    CHECK(panparasite::uiParseReviewJson("{\"action\":\"CONFIRM\",\"id\":5}",
                                        act, id) &&
              std::strcmp(act, "CONFIRM") == 0 && id == 5,
          "JSON review parses");
    CHECK(panparasite::uiParseReviewJson("{\"id\":2,\"action\":\"REJECT\"}",
                                        act, id) &&
              std::strcmp(act, "REJECT") == 0 && id == 2,
          "JSON review order-independent");
    CHECK(!panparasite::uiParseReviewJson("{\"action\":\"MAYBE\",\"id\":1}",
                                         act, id),
          "bad action refused");
    CHECK(!panparasite::uiParseReviewJson("{\"action\":\"CONFIRM\"}", act, id),
          "missing id refused");
    CHECK(!panparasite::uiParseReviewJson(nullptr, act, id), "null safe");
  }

  // ---- (c) live server routes (SKIP when loopback unavailable) ----
  {
    static panparasite::UiServer srv;
    static panparasite::UiReviewQueue rq;
    if (!panparasite::uiStart(srv, rq, 18081)) {
      std::printf("[SKIP] loopback bind unavailable; routes untested\n");
    } else {
      panparasite::UiSnapshot s;
      std::snprintf(s.slideUuid, sizeof(s.slideUuid), "%s", "slide-live");
      std::snprintf(s.nodeId, sizeof(s.nodeId), "%s", "n-live");
      s.fields = 4;
      s.verified = 1;
      s.numCands = 1;
      s.cands[0].cls = 0;
      s.cands[0].conf = 0.9;
      panparasite::uiPublish(srv, s);
      char rep[8192];
      int n = httpExchange(18081, "GET /api/status HTTP/1.0\r\n\r\n", rep,
                           sizeof(rep));
      CHECK(n > 0 && std::strstr(rep, "200 OK") != nullptr &&
                std::strstr(rep, "\"slide\":\"slide-live\"") != nullptr,
            "GET /api/status serves snapshot JSON");
      n = httpExchange(18081, "GET /app.js HTTP/1.0\r\n\r\n", rep, sizeof(rep));
      if (n < 0 || std::strstr(rep, "404") != nullptr) {
        std::printf("[SKIP] web/ not beside test CWD; file route untested\n");
      } else {
        CHECK(std::strstr(rep, "200 OK") != nullptr &&
                  std::strstr(rep, "application/javascript") != nullptr &&
                  std::strstr(rep, "/api/status") != nullptr,
              "GET /app.js serves frontend JS");
      }
      n = httpExchange(18081, "GET / HTTP/1.0\r\n\r\n", rep, sizeof(rep));
      CHECK(n > 0 && (std::strstr(rep, "MICROPARA") != nullptr ||
                                 std::strstr(rep, "PanParasite-Edge Status") !=
                                     nullptr),
            "GET / serves UI or legacy page");
      const char* body = "{\"action\":\"CONFIRM\",\"id\":3}";
      char req[256];
      std::snprintf(req, sizeof(req),
                    "POST /api/review HTTP/1.0\r\nContent-Length: %d\r\n\r\n%s",
                    (int)std::strlen(body), body);
      n = httpExchange(18081, req, rep, sizeof(rep));
      CHECK(n > 0 && std::strstr(rep, "200 OK") != nullptr &&
                std::strstr(rep, "{\"ok\":true") != nullptr && rq.count == 1 &&
                rq.slots[0].candId == 3,
            "POST /api/review enqueues CONFIRM id=3");
      const char* bad = "{\"action\":\"MAYBE\",\"id\":1}";
      std::snprintf(req, sizeof(req),
                    "POST /api/review HTTP/1.0\r\nContent-Length: %d\r\n\r\n%s",
                    (int)std::strlen(bad), bad);
      n = httpExchange(18081, req, rep, sizeof(rep));
      CHECK(n > 0 && std::strstr(rep, "400") != nullptr, "bad review -> 400");
      n = httpExchange(18081, "GET /nope HTTP/1.0\r\n\r\n", rep, sizeof(rep));
      CHECK(n > 0 && std::strstr(rep, "404") != nullptr, "unknown -> 404");
      panparasite::uiStop(srv);
    }
  }

  // ---- (d) Zero-heap hot region (new bridge code) ----
  {
    panparasite::UiSnapshot s;
    std::snprintf(s.nodeId, sizeof(s.nodeId), "%s", "n");
    s.numCands = 1;
    s.cands[0].cls = 0;
    s.cands[0].conf = 0.8;
    static panparasite::UiReviewQueue q2;
    g_newCalls.store(0);
    g_deleteCalls.store(0);
    g_countHeap.store(true);
    volatile uint64_t sink = 0;
    {
      char js[2048];
      sink += static_cast<uint64_t>(
          panparasite::uiRenderStatusJson(s, js, sizeof(js)));
      char act[8] = {};
      int id = -1;
      if (panparasite::uiParseReviewJson("{\"action\":\"REJECT\",\"id\":2}",
                                        act, id)) {
        sink += static_cast<uint64_t>(id);
      }
      if (panparasite::uiReviewPush(q2, act, id)) sink += 1u;
      char tp[96], pl[256];
      int t[7] = {};
      if (panparasite::mqttBuildTopic("n", tp, sizeof(tp))) sink += tp[0];
      if (panparasite::mqttBuildSlideSummary("n", "u", t, 0, 1, 0.0, pl,
                                            sizeof(pl))) {
        sink += pl[10];
      }
    }
    doNotOptimizeSink(sink);
    g_countHeap.store(false);
    CHECK(g_newCalls.load() == 0, "zero heap new in bridge path (got %lld)",
          g_newCalls.load());
    CHECK(g_deleteCalls.load() == 0, "zero heap delete in bridge path (got %lld)",
          g_deleteCalls.load());
  }

  if (g_failures == 0) {
    std::printf("=== ALL FRONTEND TESTS PASSED ===\n");
    return 0;
  }
  std::printf("=== %d FRONTEND TEST(S) FAILED ===\n", g_failures);
  return 1;
}
