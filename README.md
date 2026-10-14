# PanParasite-Edge & MICROPARA

Offline micro-parasitology screening instrument: a deterministic C++20 edge
pipeline (Raspberry Pi 5 / ARM64, Windows x86_64 simulation) plus the
MICROPARA Revision 8.2 offline screening-instrument web frontend, served by
the embedded C++ HTTP server on port 8080.

## Architecture

### Zero-allocation static memory arena (~23.94 MB)

All runtime buffers live in one pre-allocated `StaticMemoryArena`
(`include/memory_arena.hpp`, 25,108,516 bytes ≈ 23.94 MiB, initialized at
boot). **No `malloc`/`calloc`/`new`/`vector::push_back`/`cv::Mat::create`
in the Stages 2–5 hot path** — enforced by `new`/`delete` tripwire unit
tests. Image buffers are wrapped in non-owning
`cv::Mat(h, w, type, data_ptr)` headers only.

| Buffer | Layout | Bytes |
|---|---|---|
| Frame A / B (ping-pong) | 2 × 1920×1080 `uint8` | 2 × 2,073,600 |
| Integral image | 1921×1081 `uint32` | 8,306,404 |
| Binarized mask | 1920×1080 `uint8` | 2,073,600 |
| Union-Find arena | 32768 × 24 B nodes | 786,432 |
| Raster equivalence map | 1920×1080 `uint32` | 8,294,400 |
| Anisotropic scratch ROI | 128×128 `uint8` | 16,384 |
| FiLM micro-net params (`mlock`/`VirtualLock`) | 1,480,000 B | 1,480,000 |
| Direct-I/O sector (4K-aligned) | 4096 B | 4,096 |

Fixed geometry: 1920×1080 grayscale, spatial scale S = 0.620 µm/pixel.

### Pipeline stages

| Stage | Module | Key parameters |
|---|---|---|
| 1 Acquisition gate | `preprocessor.*` | ΔI < 1.80, Laplacian focus variance ≥ 125.0, 3 consecutive frames; AC-mains flicker exposure helper |
| 2 Integral preprocessor | `preprocessor.*` | Single-pass integral image, adaptive binarization R_w = 31 px / C = 0.88, 11×11 elliptical closing, 32 px overlap tiling benchmark path |
| 3 Betti homology engine | `topology_engine.*` | 8-connected foreground / 4-connected background Union-Find, area filter 1000–35000 px², β₀/β₁/χ/R_hole, folded spine angle, 16-D `f_BSC` vector |
| 4 Active perception | `active_perception.*` | Micro-logistic gate τ = 0.38, Top-K ≤ 8 (`CANDIDATE_CAP_EXCEEDED`), occlusion score > 0.65 → 3× Perona-Malik PDE (κ = 18.0, λ = 0.20) in the 128×128 scratch ROI |
| 5 FiLM micro-net | `inference_tracker.*` | Depthwise micro-net (INT8 fixed-point + FP32 fallback, `fp32_fallback_count` audit), 7 taxa, Platt scaling T = 1.38; OpenCV 5 `ENGINE_NEW` hook |
| 6 Tracker | `inference_tracker.*` | Hungarian matching IoU ≥ 0.30, recursive log-odds ±2.197, unobserved decay −0.10, whole-slide tally (conf ≥ 0.70) |
| 7 Persistence | `persistence.*` | 4096 B sector ring (`O_DIRECT\|O_SYNC`, Win32 fallback), CRC32-C (`__crc32cd` / Castagnoli software table), boot scanner, SHA-256 hash-chained SQLite ledger with immutable triggers, Zero-PHI UUIDs |
| 8 Telemetry/UI | `telemetry_ui.*`, `web/` | Embedded HTTP server, MQTT payload builders, drift alarm (>10 % abstain), ECDSA-SHA256 OTA hook |

End-to-end orchestration lives in `PanParasitePipeline`
(`include/pipeline.hpp`, `src/pipeline.cpp`); `src/main.cpp` builds the
`panparasite_edge` CLI benchmark.

### MICROPARA Revision 8.2 frontend (`web/`)

Calm light-mode instrument UI: 16 % nav / 84 % workspace / 8vh header /
10vh workflow bar, six-state screening workflow (setup → microscopy →
review → result → verified report), density-heatmap banner, slide minimap
with review anchors, contrast-enhancement badge, inline per-candidate
Confirm/Needs-Review toggles, dismiss-as-artifact, 4-layer evidence
(incl. taxonomy override + overlay technical sheet that never resizes the
canvas), mandatory label-verification checkbox, EN/TE/HI localization,
plus RESULTS / HISTORY / DEVICE / SETTINGS views.

## Requirements

- CMake ≥ 3.20, C++20 compiler (GCC 10+ / Clang 12+ / MSVC 19.29+)
- OpenCV (core, imgproc, dnn), SQLite3, OpenSSL, Threads
- Linux ARM64 (Raspberry Pi 5 target) or Windows x86_64 (simulation
  fallbacks for `O_DIRECT`/`mlock`)
- Any modern browser for the frontend (no build step, no dependencies)

## Build & test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Test suites: `arena_zero_alloc`, `preprocessor_stage1_stage2`,
`topology_stage3`, `active_perception_stage4`,
`inference_tracker_stage5_6`, `persistence_stage7`, `telemetry_stage8`,
`pipeline_end_to_end`, `frontend_micropara`.

Run a synthetic slide through the full pipeline:

```sh
./build/panparasite_edge --simulate 5 --bench 20
./build/panparasite_edge --help   # --ring/--db/--node/--tiled/--serve
```

## Launching the frontend

**Via the device (live data):** start the CLI with the server enabled,
then open `http://localhost:8080` (or the Pi's address). The C++ server
serves `web/index.html` + assets and bridges live state:

```sh
./build/panparasite_edge --simulate 5 --serve 8080 60
```

- `GET /`, `/index.html`, `/styles.css`, `/app.js` — instrument UI
  (falls back to a rendered status page if no `web/` checkout sits next
  to the binary)
- `GET /api/status` — live JSON snapshot (slide, tally, tracks, gate,
  candidates with bboxes/confidence/contrast flags)
- `POST /api/review` — JSON `{"action":"CONFIRM"|"REJECT","id":N}` →
  clinician queue (legacy form POST `/review` also kept)

**Directly in a browser (no device):** just open `web/index.html` from
this checkout — the app detects the `file://` context and seamlessly
falls back to rich interactive local demo state (simulated fields,
suspend/resume via localStorage, EN/TE/HI switching).

## Layout

```text
include/   memory_arena, preprocessor, topology_engine,
           active_perception, inference_tracker, persistence,
           telemetry_ui, pipeline           (headers, zero-heap contracts)
src/       implementations + main.cpp       (panparasite_edge CLI)
tests/     tripwire-enforced unit + end-to-end suites
web/       index.html / styles.css / app.js (MICROPARA instrument UI)
```

## Notes

- `OPENCODE.md` holds the frozen architectural rules; stage thresholds
  above are normative — change code and docs together.
- Ring (`*.bin`) and ledger (`*.db`) runtime files are git-ignored.
- Known load-bearing subtlety (see `tests/test_pipeline.cpp`): with the
  R_w = 31 window, large-egg interiors binarize as rim + noise speckle,
  and the 11×11 closing bridges that speckle into the solid body — keep
  the synthetic texture when editing feeders.
