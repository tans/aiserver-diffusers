# Optimization results — 2026-10-21

Reference quality status: Turbo4 is withheld after a full-size seed 20261030 failure (issue #2). Controlled same-input Turbo8 seeds 20261030/31 passed full-size frame review and strict raw-VAE re-decode; public one-image reference admission is restored only for Turbo8/TP6 at 512×320/124f. The corrected mixed workload is a new run; historical successful executions do not establish usable-video quality.

## Accepted measurements

- Native Diffusers TP6, 512×320/124f/20 actual forwards: 86.27 s denoise, 111.32 s DiT process. Separate native video/audio decode and MP4: 23.91 s.
- Turbo8 TP6, two fixed seeds: 28.08 s and 26.03 s denoise, 56.87 s and 54.31 s DiT process; approximately 24 s separate decode each. Both MP4s validate. Continuous frame samples were visually reviewed for character/scene stability. Audio stream validity is checked; no human lip-sync rating is claimed.
- Cache-warm dual TP3: two 20-forward videos and verified MP4s in 188.75 s. CPU staging is serialized; this historical pair preceded the final row-output fix. The subsequent final configuration passed two different-prompt cold tasks in 276.39 s pair wall time.
- Native sm70 H3 attention probe, same seed/input/20 forwards: 72.66 s denoise versus 86.27 s SDPA. One seed only; not a robust kernel speedup estimate or public default. Final latent video RMSE 0.0975 against baseline RMS 1.0235; arithmetic is not bit-identical. The decoded output was sampled visually.
- TP6 higher-resolution feasibility: 1024×576/124f one forward 15.52 s, peak CUDA allocation 10,436.90 MiB; 1280×768 one forward 33.48 s, 12,784.27 MiB. These are single-forward feasibility checks, not completed higher-resolution videos.
- Image conditioning: 139.49 s total, including 22.67 s TE setup and 48.57 s text/vision encoding. Native reference video VAE produces the original-resolution conditioning latent. Two complete reference Turbo4 seeds subsequently passed native decoding and MP4 checks, with approximately 81.4 s denoising each.

## Mechanisms

Read-only mmap slicing avoids full CPU checkpoint copies per TP rank. Native Diffusers transformer linears use actual tensor-parallel shards, FP32 reductions and bounded activation chunks. Sensitive context/modulation/norm projections use FP32 without NaN clamping.

TE, DiT and VAE processes release components between phases. Identical conditioning is cached by source, dependency, weight metadata, prompt, geometry and reference content. Complete-entry markers and per-entry locks protect publication.

On the local POWER9 stack, unsupported cuDNN convolution required process-local native convolution. Qwen's one-window vision patch convolution is replaced by an equivalent GEMM, without lowering input pixel count. Qwen attention computes full-context SDPA in bounded query chunks with the correct global causal offset; it avoids a 21.69 GiB single allocation observed on image input.

The original GPU lock prevents the old service and the new service from competing. A separate model-staging lock bounds concurrent CPU-heavy loading across disjoint candidate workers.

## Disabled candidates and remaining decisions

Turbo4 reference generation on six cards and cold three-card conditioning have passed. DiT residency across tasks, INT8 and block offload are not enabled. At the validated small geometry FP16 TP3 already holds resident DiT shards, so weight compression/offload is not required to establish that baseline. Their usefulness for higher-resolution TP3 remains unmeasured; no performance claim is made.

Public generation/download and ten consecutive dual-TP3 tasks passed. A separate final TP6 Turbo8 two-task window measured 34.16 videos/hour versus 61.37/hour for the dual ten-task window; mean inference 105.34 versus 113.15 s. Different short-window sample sizes limit this approximately 1.80x observed throughput comparison. The public twenty-job mixed workload and complete larger-geometry clips remain active/pending, followed by the final matched cold LightX comparison. Raw JSON is retained under reports/evidence.


The captured-forward numerical fix is documented in reports/numeric-fix.md. FP32 row GEMM output is now the default, before distributed reduction and rescaling. Earlier individual stage timings above remain historical probe measurements; the final API comparison is explicitly identified separately.
