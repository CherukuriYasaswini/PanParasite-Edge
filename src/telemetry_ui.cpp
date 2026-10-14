// PanParasite-Edge Phase 6 (Stage 8) implementation: UI server + telemetry.
// OUR code below performs no malloc/calloc/new/vector/cv::Mat::create:
// fixed-size caller storage, stack request/response buffers, snprintf.
// (std::thread construction and socket syscalls may use private OS/libc
// allocations; server start/stop is BOOT-lifecycle, never hot path.)

#include "telemetry_ui.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstring>

#include <openssl/ecdsa.h>

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
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace panparasite {
namespace {

// Viridis color-blind-safe palette (5 stops) for tally bars + accents.
inline const char* viridis(int k) noexcept {
  static const char* kPal[7] = {"#440154", "#482878", "#3B528B", "#21918C",
                                "#5DC863", "#FDE725", "#8FD744"};
  if (k < 0) k = 0;
  if (k > 6) k = 6;
  return kPal[k];
}

inline const char* taxonName(int k) noexcept {
  static const char* kNames[7] = {
      "Ascaris lumbricoides", "Trichuris trichiura",
      "Hookworm (A. duodenale / N. americanus)", "Schistosoma mansoni",
      "Schistosoma haematobium", "Non-Biological Matrix Debris",
      "Unknown / Other Helminth"};
  if (k < 0 || k > 6) return "?";
  return kNames[k];
}

// Append formatted text into a bounded buffer (truncation-safe).
inline void emit(char* out, std::size_t cap, std::size_t& off,
                 const char* fmt, ...) noexcept {
  if (off >= cap || out == nullptr || fmt == nullptr) return;
  va_list ap;
  va_start(ap, fmt);
  // vsnprintf is heap-free (stack buffer target).
  const int n = std::vsnprintf(out + off, cap - off, fmt, ap);
  va_end(ap);
  if (n <= 0) return;
  off += static_cast<std::size_t>(n);
  if (off > cap) off = cap;
}

inline bool actionOk(const char* a) noexcept {
  return a != nullptr &&
         (std::strcmp(a, "CONFIRM") == 0 || std::strcmp(a, "REJECT") == 0);
}

#if defined(_WIN32) || defined(_WIN64)
using SockT = SOCKET;
inline bool sockValid(SockT s) noexcept { return s != INVALID_SOCKET; }
inline void sockClose(SockT s) noexcept {
  if (sockValid(s)) closesocket(s);
}
#else
using SockT = int;
inline bool sockValid(SockT s) noexcept { return s >= 0; }
inline void sockClose(SockT s) noexcept {
  if (sockValid(s)) ::close(s);
}
#endif

// Read one HTTP request (headers + optional small body) into buf.
// Returns total bytes, or -1 on error/empty.
inline int readRequest(SockT c, char* buf, std::size_t cap) noexcept {
  std::size_t total = 0;
  for (int tries = 0; tries < 8; ++tries) {
    fd_set rf;
    FD_ZERO(&rf);
#if defined(_WIN32) || defined(_WIN64)
    // select() on Windows ignores the first arg; socket must be in the set.
    FD_SET(c, &rf);
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 200000;
    const int r = select(0, &rf, nullptr, nullptr, &tv);
#else
    FD_SET(c, &rf);
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 200000;
    const int r = select(c + 1, &rf, nullptr, nullptr, &tv);
#endif
    if (r <= 0) break;
#if defined(_WIN32) || defined(_WIN64)
    const int n = recv(c, buf + total, static_cast<int>(cap - total - 1), 0);
#else
    const ssize_t n = recv(c, buf + total, cap - total - 1, 0);
#endif
    if (n <= 0) break;
    total += static_cast<std::size_t>(n);
    buf[total] = '\0';
    if (std::strstr(buf, "\r\n\r\n") != nullptr) break;
    if (total + 512 >= cap) break;
  }
  if (total == 0) return -1;
  buf[total] = '\0';
  return static_cast<int>(total);
}

inline bool sendAll(SockT c, const char* buf, std::size_t n) noexcept {
  std::size_t done = 0;
  while (done < n) {
#if defined(_WIN32) || defined(_WIN64)
    const int w = send(c, buf + done, static_cast<int>(n - done), 0);
#else
    const ssize_t w = send(c, buf + done, n - done, 0);
#endif
    if (w <= 0) return false;
    done += static_cast<std::size_t>(w);
  }
  return true;
}

// Minimal x-www-form-urlencoded parse: action=CONFIRM|REJECT & id=<int>.
inline bool parseReview(const char* body, char action[8], int& id) noexcept {
  if (body == nullptr || action == nullptr) return false;
  const char* pa = std::strstr(body, "action=");
  const char* pi = std::strstr(body, "id=");
  if (pa == nullptr || pi == nullptr) return false;
  pa += 7;
  int i = 0;
  while (i < 7 && pa[i] != '\0' && pa[i] != '&' && pa[i] != ' ' &&
         pa[i] != '\r' && pa[i] != '\n') {
    action[i] = pa[i];
    ++i;
  }
  action[i] = '\0';
  if (!actionOk(action)) return false;
  int v = 0, neg = 0;
  pi += 3;
  if (*pi == '-') {
    neg = 1;
    ++pi;
  }
  if (*pi < '0' || *pi > '9') return false;
  while (*pi >= '0' && *pi <= '9') {
    v = v * 10 + (*pi - '0');
    ++pi;
  }
  id = neg ? -v : v;
  return true;
}

// ---- Static web-frontend files (MICROPARA instrument UI) ----
// Only exact route->filename matches (no user-controlled paths, no
// traversal risk). Raw syscalls + fixed 4KB chunk buffer: heap-free.
inline bool webOpen(const char* name, int& fdOut, long long& sizeOut) noexcept {
  static const char* kRoots[3] = {"web/", "../web/", "./web/"};
  for (int r = 0; r < 3; ++r) {
    char path[64];
    const int n =
        std::snprintf(path, sizeof(path), "%s%s", kRoots[r], name);
    if (n <= 0 || static_cast<std::size_t>(n) >= sizeof(path)) continue;
#if defined(_WIN32) || defined(_WIN64)
    const int fd = _open(path, _O_RDONLY | _O_BINARY);
    if (fd < 0) continue;
    struct _stat st;
    if (_fstat(fd, &st) != 0 || st.st_size < 0) {
      _close(fd);
      continue;
    }
    fdOut = fd;
    sizeOut = static_cast<long long>(st.st_size);
    return true;
#else
    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) continue;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) {
      ::close(fd);
      continue;
    }
    fdOut = fd;
    sizeOut = static_cast<long long>(st.st_size);
    return true;
#endif
  }
  return false;
}

inline void webClose(int fd) noexcept {
  if (fd < 0) return;
#if defined(_WIN32) || defined(_WIN64)
  _close(fd);
#else
  ::close(fd);
#endif
}

// Streams a frontend file with MIME headers. Returns true when the file
// was found (served or attempted); false when absent on all roots.
inline bool sendWebFile(SockT conn, const char* name,
                        const char* mime) noexcept {
  int fd = -1;
  long long size = 0;
  if (!webOpen(name, fd, size)) return false;
  char hdr[256];
  int hlen = std::snprintf(hdr, sizeof(hdr),
                           "HTTP/1.0 200 OK\r\nContent-Type: %s\r\n"
                           "Content-Length: %lld\r\n\r\n",
                           mime, size);
  if (hlen < 0) hlen = 0;
  if (hlen > static_cast<int>(sizeof(hdr))) hlen = static_cast<int>(sizeof(hdr));
  bool ok = sendAll(conn, hdr, static_cast<std::size_t>(hlen));
  char chunk[4096];
  long long left = size;
  while (ok && left > 0) {
    const std::size_t want =
        (left > 4096LL) ? 4096u : static_cast<std::size_t>(left);
#if defined(_WIN32) || defined(_WIN64)
    const int got = _read(fd, chunk, static_cast<unsigned>(want));
#else
    const ssize_t got = ::read(fd, chunk, want);
#endif
    if (got <= 0) {
      ok = false;
      break;
    }
    ok = sendAll(conn, chunk, static_cast<std::size_t>(got));
    left -= got;
  }
  webClose(fd);
  return true;
}

inline void sendJson(SockT conn, int status, const char* js,
                     std::size_t len) noexcept {
  char hdr[256];
  int hlen = std::snprintf(hdr, sizeof(hdr),
                           "HTTP/1.0 %d %s\r\nContent-Type: "
                           "application/json\r\nContent-Length: %d\r\n\r\n",
                           status, (status == 200) ? "OK" : "Bad Request",
                           static_cast<int>(len));
  if (hlen < 0) hlen = 0;
  if (hlen > static_cast<int>(sizeof(hdr))) hlen = static_cast<int>(sizeof(hdr));
  sendAll(conn, hdr, static_cast<std::size_t>(hlen));
  if (js != nullptr && len > 0) sendAll(conn, js, len);
}

void serveOne(UiServer& srv, SockT conn) noexcept {
  char req[4096];
  int n = readRequest(conn, req, sizeof(req));
  if (n < 0) return;
  // POST bodies can trail the headers: honor Content-Length (bounded).
  char* hdrEnd = std::strstr(req, "\r\n\r\n");
  std::size_t have = (hdrEnd != nullptr)
                         ? static_cast<std::size_t>(n) -
                               static_cast<std::size_t>(hdrEnd + 4 - req)
                         : 0u;
  std::size_t want = 0u;
  const char* pcl = std::strstr(req, "Content-Length:");
  if (pcl != nullptr) {
    int v = 0;
    if (std::sscanf(pcl + 15, "%d", &v) == 1 && v > 0 &&
        v < 2048) {
      want = static_cast<std::size_t>(v);
    }
  }
  int guard = 0;
  while (have < want && n + 256 < static_cast<int>(sizeof(req)) && guard < 4) {
#if defined(_WIN32) || defined(_WIN64)
    const int m = recv(conn, req + n, 256, 0);
#else
    const ssize_t m = recv(conn, req + n, 256, 0);
#endif
    if (m <= 0) break;
    n += m;
    have += static_cast<std::size_t>(m);
    req[n] = '\0';
    ++guard;
  }
  char page[16384];
  // ---- MICROPARA frontend + JSON bridge (Revision 8.2 wiring) ----
  if (std::strncmp(req, "POST /api/review", 16) == 0) {
    const char* body = std::strstr(req, "\r\n\r\n");
    body = (body != nullptr) ? body + 4 : "";
    char action[8] = {};
    int id = -1;
    bool ok = false;
    if (uiParseReviewJson(body, action, id) && srv.queue != nullptr) {
      ok = uiReviewPush(*srv.queue, action, id);
    }
    char js[128];
    int jl = std::snprintf(js, sizeof(js), "{\"ok\":%s,\"id\":%d}",
                           ok ? "true" : "false", id);
    if (jl < 0) jl = 0;
    if (jl > static_cast<int>(sizeof(js))) jl = static_cast<int>(sizeof(js));
    sendJson(conn, ok ? 200 : 400, js, static_cast<std::size_t>(jl));
    return;
  }
  if (std::strncmp(req, "GET /api/status", 15) == 0 &&
      (req[15] == ' ' || req[15] == '\r' || req[15] == '\n' ||
       req[15] == '\0')) {
    UiSnapshot snap;
    {
      std::lock_guard<std::mutex> lk(srv.snapMu);
      snap = srv.snap;
    }
    char js[4096];
    const int jl = uiRenderStatusJson(snap, js, sizeof(js));
    sendJson(conn, 200, js, static_cast<std::size_t>(jl));
    return;
  }
  if (std::strncmp(req, "GET /styles.css", 15) == 0 && req[15] == ' ') {
    if (sendWebFile(conn, "styles.css", "text/css; charset=utf-8")) return;
  }
  if (std::strncmp(req, "GET /app.js", 11) == 0 && req[11] == ' ') {
    if (sendWebFile(conn, "app.js", "application/javascript; charset=utf-8")) {
      return;
    }
  }
  if ((std::strncmp(req, "GET / ", 6) == 0 ||
       (std::strncmp(req, "GET /index.html", 15) == 0 && req[15] == ' '))) {
    if (sendWebFile(conn, "index.html", "text/html; charset=utf-8")) return;
    // No web/ checkout alongside the binary: legacy rendered status page
    // (keeps headless builds and build-dir test runs functional).
    UiSnapshot snap;
    {
      std::lock_guard<std::mutex> lk(srv.snapMu);
      snap = srv.snap;
    }
    char html[16384];
    const int plen = uiRenderPage(snap, html, sizeof(html));
    char hdr[256];
    int hlen = std::snprintf(hdr, sizeof(hdr),
                             "HTTP/1.0 200 OK\r\nContent-Type: text/html; "
                             "charset=utf-8\r\nContent-Length: %d\r\n\r\n",
                             plen);
    if (hlen < 0) hlen = 0;
    if (hlen > static_cast<int>(sizeof(hdr))) hlen = static_cast<int>(sizeof(hdr));
    sendAll(conn, hdr, static_cast<std::size_t>(hlen));
    sendAll(conn, html, static_cast<std::size_t>(plen));
    return;
  }
  if (std::strncmp(req, "POST /review", 12) == 0) {
    const char* body = std::strstr(req, "\r\n\r\n");
    body = (body != nullptr) ? body + 4 : "";
    char action[8] = {};
    int id = -1;
    bool ok = false;
    if (parseReview(body, action, id) && srv.queue != nullptr) {
      ok = uiReviewPush(*srv.queue, action, id);
    }
    const char* msg = ok ? "queued" : "rejected";
    int plen = std::snprintf(page, sizeof(page),
                             "<html><body><p>review %s</p>"
                             "<a href=\"/\">back</a></body></html>",
                             msg);
    if (plen < 0) plen = 0;
    if (plen > static_cast<int>(sizeof(page))) plen = static_cast<int>(sizeof(page));
    char hdr[256];
    int hlen = std::snprintf(hdr, sizeof(hdr),
                             "HTTP/1.0 303 See Other\r\nLocation: /\r\n"
                             "Content-Length: %d\r\n\r\n",
                             plen);
    if (hlen < 0) hlen = 0;
    if (hlen > static_cast<int>(sizeof(hdr))) hlen = static_cast<int>(sizeof(hdr));
    sendAll(conn, hdr, static_cast<std::size_t>(hlen));
    sendAll(conn, page, static_cast<std::size_t>(plen));
    return;
  }
  // Anything else: 404 (keeps /favicon.ico and probes quiet; the legacy
  // rendered page remains the GET / fallback when no web/ checkout exists).
  {
    const char* nf =
        "<html><body><p>not found</p><a href=\"/\">back</a></body></html>";
    const int nflen = static_cast<int>(std::strlen(nf));
    char hdr[256];
    int hlen = std::snprintf(hdr, sizeof(hdr),
                             "HTTP/1.0 404 Not Found\r\nContent-Type: "
                             "text/html; charset=utf-8\r\nContent-Length: %d"
                             "\r\n\r\n",
                             nflen);
    if (hlen < 0) hlen = 0;
    if (hlen > static_cast<int>(sizeof(hdr))) hlen = static_cast<int>(sizeof(hdr));
    sendAll(conn, hdr, static_cast<std::size_t>(hlen));
    sendAll(conn, nf, static_cast<std::size_t>(nflen));
  }
}

void uiWorker(UiServer* srv) noexcept {
  if (srv == nullptr) return;
  while (srv->running.load(std::memory_order_relaxed)) {
#if defined(_WIN32) || defined(_WIN64)
    SockT lf = static_cast<SockT>(srv->listenFd);
#else
    const int lf = static_cast<int>(srv->listenFd);
#endif
    if (!sockValid(lf)) break;
    fd_set rf;
    FD_ZERO(&rf);
#if defined(_WIN32) || defined(_WIN64)
    FD_SET(lf, &rf);
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 200000;
    const int r = select(0, &rf, nullptr, nullptr, &tv);
#else
    FD_SET(lf, &rf);
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 200000;
    const int r = select(lf + 1, &rf, nullptr, nullptr, &tv);
#endif
    if (r <= 0) continue;
    if (!srv->running.load(std::memory_order_relaxed)) break;
#if defined(_WIN32) || defined(_WIN64)
    SockT conn = accept(lf, nullptr, nullptr);
#else
    SockT conn = accept(lf, nullptr, nullptr);
#endif
    if (!sockValid(conn)) continue;
    serveOne(*srv, conn);
    sockClose(conn);
  }
}

}  // namespace

bool uiReviewPush(UiReviewQueue& q, const char* action, int candId) noexcept {
  if (!actionOk(action) || candId < 0) return false;
  std::lock_guard<std::mutex> lk(q.mu);
  if (q.count >= kReviewSlots) return false;
  for (int i = 0; i < kReviewSlots; ++i) {
    if (!q.slots[i].used) {
      for (int k = 0; k < 7; ++k) q.slots[i].action[k] = action[k];
      q.slots[i].action[7] = '\0';
      q.slots[i].candId = candId;
      q.slots[i].used = true;
      ++q.count;
      return true;
    }
  }
  return false;
}

int uiRenderPage(const UiSnapshot& s, char* out, std::size_t cap) noexcept {
  if (out == nullptr || cap == 0) return 0;
  out[0] = '\0';
  std::size_t off = 0;
  emit(out, cap, off,
       "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
       "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
       "<title>PanParasite-Edge Status</title><style>"
       "body{font-family:sans-serif;background:#f4f4f4;margin:0;padding:16px;}"
       ".card{background:#fff;border-radius:12px;padding:16px;margin-bottom:16px;}"
       ".btn{display:inline-block;min-height:56px;min-width:56px;padding:12px "
       "24px;margin:8px;border-radius:12px;border:0;font-size:20px;"
       "color:#fff;text-decoration:none;text-align:center;}"
       ".confirm{background:%s;}.reject{background:%s;}"
       "table{border-collapse:collapse;}td,th{padding:8px 12px;border:1px solid "
       "#ddd;text-align:left;}"
       ".bar{height:14px;border-radius:7px;}</style></head><body>",
       viridis(3), viridis(0));
  emit(out, cap, off,
       "<div class=\"card\"><h1>PanParasite-Edge Status</h1>"
       "<p>Status &#8226; <span lang=\"te\">\xe0\xb0\xb8\xe0\xb1\x8d\xe0\xb0\xa5"
       "\xe0\xb0\xbf\xe0\xb0\xa4\xe0\xb0\xbf</span> &#8226; <span lang=\"hi\">\xe0\xa4\xb8\xe0\xa5\x8d"
       "\xe0\xa4\xa5\xe0\xa4\xbf\xe0\xa4\xa4\xe0\xa4\xbf</span></p>"
       "<table><tr><th>Metric</th><th>Value</th></tr>");
  emit(out, cap, off, "<tr><td>Slide</td><td>%s</td></tr>", s.slideUuid);
  emit(out, cap, off, "<tr><td>Node</td><td>%s</td></tr>", s.nodeId);
  emit(out, cap, off,
       "<tr><td>Fields</td><td>%d</td></tr>"
       "<tr><td>Verified eggs</td><td>%d</td></tr>"
       "<tr><td>Rejected debris</td><td>%d</td></tr>"
       "<tr><td>Active tracks</td><td>%d</td></tr>"
       "<tr><td>Focus Phi</td><td>%.1f</td></tr>"
       "<tr><td>Delta I</td><td>%.3f</td></tr>"
       "<tr><td>Gate locked</td><td>%d</td></tr>"
       "<tr><td>FP32 fallbacks</td><td>%d</td></tr>"
       "<tr><td>Cap exceeded</td><td>%d</td></tr>",
       s.fields, s.verified, s.rejected, s.activeTracks, s.focusPhi, s.deltaIt,
       (int)s.gateLocked, s.fp32Fallbacks, (int)s.capExceeded);
  emit(out, cap, off, "</table></div><div class=\"card\"><h2>Taxon tally</h2>");
  for (int k = 0; k < 7; ++k) {
    emit(out, cap, off,
         "<p>%s: <b>%d</b></p>"
         "<div class=\"bar\" style=\"width:%dpx;background:%s;\"></div>",
         taxonName(k), s.tally[k], s.tally[k] > 200 ? 200 : s.tally[k] * 20,
         viridis(k));
  }
  emit(out, cap, off,
       "</div><div class=\"card\"><h2>Clinician review</h2>"
       "<p>Confirm &#8226; <span lang=\"te\">\xe0\xb0\xa8\xe0\xb0\xbf\xe0\xb0\xb0"
       "\xe0\xb1\x8d\xe0\xb0\xa7\xe0\xb0\xbe\xe0\xb0\xb0\xe0\xb0\xbf\xe0\xb0\x82"
       "\xe0\xb0\x9a\xe0\xb1\x81</span> &#8226; <span lang=\"hi\">\xe0\xa4\xaa"
       "\xe0\xa5\x81\xe0\xa4\xb7\xe0\xa5\x8d\xe0\xa4\x9f\xe0\xa4\xbf "
       "\xe0\xa4\x95\xe0\xa4\xb0\xe0\xa5\x87\xe0\xa4\x82</span> / Reject "
       "&#8226; <span lang=\"te\">\xe0\xb0\xa4\xe0\xb0\xbf\xe0\xb0\xb0\xe0\xb1\x8d"
       "\xe0\xb0\xb8\xe0\xb1\x8d\xe0\xb0\x95\xe0\xb0\xb0\xe0\xb0\xbf\xe0\xb0\x82"
       "\xe0\xb0\x9a\xe0\xb1\x81</span> &#8226; <span lang=\"hi\">\xe0\xa4\x85"
       "\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xb5\xe0\xa5\x80\xe0\xa4\x95\xe0\xa4\xbe"
       "\xe0\xa4\xb0 \xe0\xa4\x95\xe0\xa4\xb0\xe0\xa5\x87\xe0\xa4\x82</span></p>"
       "<form method=\"POST\" action=\"/review\">"
       "<label>Candidate <input type=\"number\" name=\"id\" value=\"0\" "
       "style=\"min-height:56px;font-size:20px;\"></label><br>"
       "<button class=\"btn confirm\" type=\"submit\" name=\"action\" "
       "value=\"CONFIRM\">CONFIRM</button>"
       "<button class=\"btn reject\" type=\"submit\" name=\"action\" "
       "value=\"REJECT\">REJECT</button>"
       "</form></div></body></html>");
  if (off >= cap) off = cap - 1;
  out[off] = '\0';
  return static_cast<int>(off);
}

bool uiStart(UiServer& srv, UiReviewQueue& q, int port) noexcept {
  if (port <= 0 || port > 65535) return false;
  uiStop(srv);
#if defined(_WIN32) || defined(_WIN64)
  WSADATA wd;
  if (WSAStartup(MAKEWORD(2, 2), &wd) != 0) return false;
#endif
  SockT lf =
#if defined(_WIN32) || defined(_WIN64)
      socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
#else
      ::socket(AF_INET, SOCK_STREAM, 0);
#endif
  if (!sockValid(lf)) {
#if defined(_WIN32) || defined(_WIN64)
    WSACleanup();
#endif
    return false;
  }
  int one = 1;
#if defined(_WIN32) || defined(_WIN64)
  setsockopt(lf, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one),
             sizeof(one));
#else
  setsockopt(lf, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // loopback only (embedded)
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (bind(lf, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
      listen(lf, 4) != 0) {
    sockClose(lf);
#if defined(_WIN32) || defined(_WIN64)
    WSACleanup();
#endif
    return false;
  }
  srv.listenFd = static_cast<std::intptr_t>(lf);
  srv.port = port;
  srv.queue = &q;
  srv.running.store(true, std::memory_order_relaxed);
  try {
    srv.worker = std::thread(uiWorker, &srv);
  } catch (...) {
    srv.running.store(false, std::memory_order_relaxed);
    sockClose(lf);
    srv.listenFd = -1;
#if defined(_WIN32) || defined(_WIN64)
    WSACleanup();
#endif
    return false;
  }
  return true;
}

void uiStop(UiServer& srv) noexcept {
  srv.running.store(false, std::memory_order_relaxed);
#if defined(_WIN32) || defined(_WIN64)
  if (srv.listenFd != -1) {
    sockClose(static_cast<SockT>(srv.listenFd));
    srv.listenFd = -1;
  }
#else
  if (srv.listenFd != -1) {
    // Shutdown unblocks a blocking accept(); the worker also polls.
    ::shutdown(static_cast<int>(srv.listenFd), SHUT_RDWR);
    ::close(static_cast<int>(srv.listenFd));
    srv.listenFd = -1;
  }
#endif
  try {
    if (srv.worker.joinable()) srv.worker.join();
  } catch (...) {
  }
#if defined(_WIN32) || defined(_WIN64)
  WSACleanup();
#endif
  srv.queue = nullptr;
}

void uiPublish(UiServer& srv, const UiSnapshot& s) noexcept {
  std::lock_guard<std::mutex> lk(srv.snapMu);
  srv.snap = s;  // plain-data copy, heap-free
}

bool mqttBuildTopic(const char* nodeId, char* topicOut,
                    std::size_t cap) noexcept {
  if (nodeId == nullptr || nodeId[0] == '\0' || topicOut == nullptr ||
      cap == 0) {
    return false;
  }
  const int n = std::snprintf(topicOut, cap, "telemetry/sth/%s/slide_summary",
                              nodeId);
  return n > 0 && static_cast<std::size_t>(n) < cap;
}

bool mqttBuildSlideSummary(const char* nodeId, const char* slideUuid,
                           const int tally[7], int verified, int total,
                           double abstain, char* jsonOut,
                           std::size_t cap) noexcept {
  if (nodeId == nullptr || slideUuid == nullptr || tally == nullptr ||
      jsonOut == nullptr || cap == 0) {
    return false;
  }
  if (!(abstain >= 0.0) || abstain > 1.0) abstain = 0.0;
  const int n = std::snprintf(
      jsonOut, cap,
      "{\"node\":\"%s\",\"slide\":\"%s\",\"tally\":[%d,%d,%d,%d,%d,%d,%d],"
      "\"verified\":%d,\"total\":%d,\"abstain\":%.4f}",
      nodeId, slideUuid, tally[0], tally[1], tally[2], tally[3], tally[4],
      tally[5], tally[6], verified, total, abstain);
  return n > 0 && static_cast<std::size_t>(n) < cap;
}

bool otaVerifyCanary(const uint8_t hash32[32], const uint8_t* sigDer,
                     std::size_t sigLen, void* pubkey) noexcept {
  if (hash32 == nullptr || sigDer == nullptr || sigLen == 0 ||
      pubkey == nullptr) {
    return false;
  }
  EC_KEY* key = static_cast<EC_KEY*>(pubkey);
  const int rc = ECDSA_verify(0, hash32, 32, sigDer,
                              static_cast<int>(sigLen), key);
  return rc == 1;
}

int uiRenderStatusJson(const UiSnapshot& s, char* out,
                       std::size_t cap) noexcept {
  if (out == nullptr || cap == 0) return 0;
  out[0] = '\0';
  std::size_t off = 0;
  emit(out, cap, off,
       "{\"slide\":\"%s\",\"node\":\"%s\",\"fields\":%d,\"verified\":%d,"
       "\"rejected\":%d,\"tracks\":%d,\"tally\":[%d,%d,%d,%d,%d,%d,%d],"
       "\"focus\":%.1f,\"deltaIt\":%.3f,\"locked\":%d,\"fallbacks\":%d,"
       "\"cap\":%d,\"ringDirectIo\":%d,\"cands\":[",
       s.slideUuid, s.nodeId, s.fields, s.verified, s.rejected,
       s.activeTracks, s.tally[0], s.tally[1], s.tally[2], s.tally[3],
       s.tally[4], s.tally[5], s.tally[6], s.focusPhi, s.deltaIt,
       (int)s.gateLocked, s.fp32Fallbacks, (int)s.capExceeded,
       (int)s.ringDirectIo);
  const int n =
      (s.numCands < 0) ? 0 : ((s.numCands > 8) ? 8 : s.numCands);
  for (int i = 0; i < n; ++i) {
    const UiCandidate& c = s.cands[i];
    emit(out, cap, off,
         "%s{\"x0\":%.0f,\"y0\":%.0f,\"x1\":%.0f,\"y1\":%.0f,\"cls\":%d,"
         "\"conf\":%.4f,\"trig\":%d}",
         (i == 0) ? "" : ",", (double)c.x0, (double)c.y0, (double)c.x1,
         (double)c.y1, c.cls, c.conf, (int)c.trig);
  }
  emit(out, cap, off, "]}");
  if (off >= cap) off = cap - 1;
  out[off] = '\0';
  return static_cast<int>(off);
}

bool uiParseReviewJson(const char* body, char action[8], int& id) noexcept {
  if (body == nullptr || action == nullptr) return false;
  id = -1;
  // "action" value (order-independent, bounded copy).
  const char* pa = std::strstr(body, "\"action\"");
  if (pa == nullptr) return false;
  pa = std::strchr(pa + 8, '"');
  if (pa == nullptr) return false;
  ++pa;
  int i = 0;
  while (i < 7 && pa[i] != '\0' && pa[i] != '"') {
    action[i] = pa[i];
    ++i;
  }
  action[i] = '\0';
  if (!actionOk(action)) return false;
  // "id" integer (optional leading '-', digits required).
  const char* pi = std::strstr(body, "\"id\"");
  if (pi == nullptr) return false;
  pi = std::strchr(pi + 4, ':');
  if (pi == nullptr) return false;
  ++pi;
  while (*pi == ' ' || *pi == '\t') ++pi;
  int neg = 0;
  if (*pi == '-') {
    neg = 1;
    ++pi;
  }
  if (*pi < '0' || *pi > '9') return false;
  int v = 0;
  while (*pi >= '0' && *pi <= '9') {
    v = v * 10 + (*pi - '0');
    ++pi;
  }
  id = neg ? -v : v;
  return true;
}

}  // namespace panparasite
