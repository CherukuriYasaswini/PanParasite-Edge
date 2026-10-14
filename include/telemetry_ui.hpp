#pragma once

// PanParasite-Edge Phase 6 (Stage 8): Embedded Client UI & Asynchronous
// AWS IoT Core Bridge.
//
// OPENCODE.md Rules enforced:
//   Rule 1: C++20, Linux ARM64 + Windows/x86_64 simulation fallback
//     (Winsock2 server path, same wire protocol).
//   Rule 2: OUR code performs no malloc/calloc/new/vector/cv::Mat::create:
//     fixed-size caller storage, stack request/response buffers, snprintf.
//     (std::thread/mutex construction and socket syscalls may use private
//     OS/libc allocations; server start/stop is BOOT-lifecycle, never in
//     the Stages 2-5 hot path.)
//
// What this stage provides:
//   * Embedded HTTP status + human-in-the-loop review server (port 8080):
//     Viridis color-blind-safe palette, >=56x56px touch targets,
//     English/Telugu/Hindi labels, single-tap CONFIRM/REJECT queuing.
//   * Async AWS IoT Core MQTT/TLS telemetry payload builder:
//     topic telemetry/sth/<node_id>/slide_summary + JSON slide summary.
//   * CloudWatch drift alarm check (abstain rate > 10%).
//   * ECDSA-SHA256 OTA canary model verification hook.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>

namespace panparasite {

// ------------------------------------------------------------------
// Human-in-the-loop review queue (fixed, no heap).
// ------------------------------------------------------------------
inline constexpr int kReviewSlots = 32;

struct UiReviewItem {
  char action[8] = {};  // "CONFIRM" or "REJECT"
  int candId = -1;
  bool used = false;
};

struct UiReviewQueue {
  UiReviewItem slots[kReviewSlots];
  int count = 0;
  std::mutex mu;
};

// Live slide snapshot rendered by the status page (plain data).
// cands[] carries the current field's Top-K detections for the web
// minimap / findings panel (compact candidate index order).
struct UiCandidate {
  float x0 = 0.0f, y0 = 0.0f, x1 = -1.0f, y1 = -1.0f;
  int cls = -1;
  double conf = 0.0;
  bool trig = false;  // active-perception contrast enhancement ran
};

struct UiSnapshot {
  char slideUuid[37] = {};
  char nodeId[64] = {};
  int fields = 0;
  int verified = 0;
  int rejected = 0;
  int activeTracks = 0;
  int tally[7] = {};
  double focusPhi = 0.0;
  double deltaIt = 0.0;
  bool gateLocked = false;
  int fp32Fallbacks = 0;
  bool capExceeded = false;
  UiCandidate cands[8] = {};
  int numCands = 0;
  bool ringDirectIo = false;
};

struct UiServer {
  int port = 8080;
  std::intptr_t listenFd = -1;  // int fd (POSIX) / SOCKET (Win64): intptr-wide
  std::atomic<bool> running{false};
  std::thread worker;
  UiReviewQueue* queue = nullptr;
  UiSnapshot snap;
  std::mutex snapMu;
};

// Single-tap clinician action enqueue. Returns false when full/invalid.
bool uiReviewPush(UiReviewQueue& q, const char* action, int candId) noexcept;

// Render the status page into out[cap]; returns bytes written (<= cap).
// Always NUL-terminates when cap > 0. Heap-free.
int uiRenderPage(const UiSnapshot& s, char* out, std::size_t cap) noexcept;

// Render live device state as JSON for GET /api/status into out[cap];
// returns bytes written (<= cap), NUL-terminated when cap > 0. Heap-free.
// Schema: {slide,node,fields,verified,rejected,tracks,tally[7],focus,
// deltaIt,locked,fallbacks,cap,ringDirectIo,cands:[{x0,y0,x1,y1,cls,
// conf,trig}]}. Coordinates are full-frame pixels (1920x1080).
int uiRenderStatusJson(const UiSnapshot& s, char* out,
                       std::size_t cap) noexcept;

// Parse a JSON clinician override {"action":"CONFIRM"|"REJECT","id":N}
// (used by POST /api/review). Heap-free, order-independent. Returns false
// on malformed input; action is always NUL-terminated on true.
bool uiParseReviewJson(const char* body, char action[8], int& id) noexcept;

// Server lifecycle (BOOT-lifecycle: start once, stop at shutdown).
bool uiStart(UiServer& srv, UiReviewQueue& q, int port) noexcept;
void uiStop(UiServer& srv) noexcept;
// Publish a fresh snapshot for subsequent HTTP renders (mutex, memcpy).
void uiPublish(UiServer& srv, const UiSnapshot& s) noexcept;

// ------------------------------------------------------------------
// AWS IoT Core telemetry payloads (heap-free builders).
// ------------------------------------------------------------------
inline constexpr double kDriftAbstainLimit = 0.10;

// Topic: telemetry/sth/<node_id>/slide_summary -> topicOut (cap >= 128).
bool mqttBuildTopic(const char* nodeId, char* topicOut,
                    std::size_t cap) noexcept;
// JSON slide summary -> jsonOut (cap >= 512 recommended). abstain in [0,1].
bool mqttBuildSlideSummary(const char* nodeId, const char* slideUuid,
                           const int tally[7], int verified, int total,
                           double abstain, char* jsonOut,
                           std::size_t cap) noexcept;

// Abstain-rate helper + CloudWatch drift alarm predicate.
inline double slideAbstainRate(int unknownCount, int rejectedCount,
                               int total) noexcept {
  if (total <= 0) return 0.0;
  return static_cast<double>(unknownCount + rejectedCount) /
         static_cast<double>(total);
}
inline bool cloudwatchDriftAlarm(double abstainRate) noexcept {
  return abstainRate > kDriftAbstainLimit;
}

// ------------------------------------------------------------------
// ECDSA-SHA256 OTA canary verification hook.
// pubkey is an opaque EC_KEY* (test creates it via OpenSSL directly).
// Returns true iff sigDer is a valid DER signature of hash32.
// ------------------------------------------------------------------
bool otaVerifyCanary(const uint8_t hash32[32], const uint8_t* sigDer,
                     std::size_t sigLen, void* pubkey) noexcept;

}  // namespace panparasite
