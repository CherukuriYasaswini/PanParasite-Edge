# PanParasite-Edge Architectural Rules (STRICT ENFORCEMENT)

1. **Language & Build:** C++20, CMake, cross-platform compatibility (Linux ARM64 for Raspberry Pi 5 + Windows/x86_64 simulation fallback for `O_DIRECT` and `mlock`).
2. **Zero Heap Allocation Invariant:**
   - Strictly NO `malloc`, `calloc`, `new`, `std::vector::push_back`, or dynamic `cv::Mat::create` inside the runtime hot-path (Stages 2 through 5).
   - All memory buffers MUST reside in a single pre-allocated `StaticMemoryArena` (~23.94 MB total) initialized at boot.
   - Wrap pre-allocated raw arrays in non-owning `cv::Mat(height, width, type, data_ptr)` headers.
3. **Frame Resolution & Optical Scale:**
   - Fixed 1920x1080 grayscale (`uint8_t`), spatial scale S = 0.620 um/pixel.
4. **Pipeline Stages:**
   - Stage 1: Acquisition Gate (Delta I_t < 1.80, Laplacian Focus Variance >= 125.0 across 3 consecutive frames).
   - Stage 2: Full-Frame Single-Pass Integral Adaptive Binarization (R_w = 31 px, C_thresh = 0.88) + 11x11 elliptical closing (plus 32px overlap tiling benchmark toggle).
   - Stage 3: Dual-Connectivity Union-Find Homology Engine (8-connected foreground, 4-connected background) extracting Beta_0, Beta_1, R_hole, theta_folded, and 16-D Betti vector f_BSC.
   - Stage 4: Stage A Micro-Logistic Gate (tau = 0.38, cap K <= 8) + Active Perception Perona-Malik Anisotropic Diffusion PDE (O_score > 0.65, 3 iterations, kappa = 18.0, lambda = 0.20).
   - Stage 5: FiLM-Conditioned Depthwise Micro-Net via OpenCV `cv::dnn::ENGINE_NEW` (INT8 with FP32 fallback) + Platt Scaling (T = 1.38).
   - Stage 6: Hungarian Bipartite Spatial Tracker (IoU >= 0.30) + Recursive Bayesian Log-Odds updater (L_t +/- 2.197).
   - Stage 7: 4096-byte aligned Ring Buffer with CRC32-C checksums + SHA-256 SQLite Audit Ledger.
