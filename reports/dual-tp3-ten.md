# Ten-job dual-TP3 Turbo8 acceptance

2026-10-21. SDPA, FP16 weights, FP32 sensitive operations and row-projection outputs, 512×320, 124 frames, eight actual forwards. Two isolated GPU groups; one cached prompt and ten different seeds. Both workers reload DiT for each task. Every output passed native finite checks and FFmpeg decoding checks.

| Measurement | Observed |
|---|---:|
| Completed / submitted | 10 / 10 |
| Whole batch wall time | 878.44 s |
| Whole batch throughput | 40.98 videos/hour |
| Whole batch latency P50 / P95 | 112.27 / 431.24 s |
| Initial old-service GPU admission wait | about 290 s |
| Active window after initial admission | 586.61 s |
| Active-window throughput | 61.37 videos/hour |
| Inference mean / P50 / maximum | 113.15 / 110.65 / 139.48 s |
| Per-card observed peak GPU memory | at most 15,470 MiB |
| New host OOM kills during batch | 0 (counter remained 15) |

The active window starts at the earliest job creation time plus its recorded queue/admission wait and ends at the last terminal update. It includes model staging, reloads, decoding, scheduling gaps and polling overhead; it excludes the initial wait for the existing service. This is a short measured window, not a one-hour soak test or a measured human usable-video rate. The first pair has extra cold model-staging contention. A fair TP6 throughput comparison remains separate.

Raw requests, job IDs, per-stage timings, sampled GPU/PSS telemetry and output hashes are in `evidence/dual-tp3-turbo8-ten.json`; derived window calculations are in `evidence/dual-tp3-turbo8-ten-analysis.json`.

## Separate TP6 comparison

The final FP32-row-output TP6 configuration completed two cached-prompt Turbo8 tasks serially: 103.33 and 107.35 s inference time, mean 105.34 s. Its active window was 210.77 s, or 34.16 videos/hour. Comparing these short observed windows gives 61.37 / 34.16 ≈ 1.80× throughput for dual TP3, at an approximately 7.4% longer mean individual inference time. The sample sizes differ (ten versus two), and no statistical confidence or hour-long capacity is asserted. Raw TP6 batch wall 248.74 s includes initial wait for the independent cancellation probe. See evidence/tp6-turbo8-two.json and its analysis JSON.

Four-frame samples from the first and last ten-job seeds showed stable characters, living-room geometry and continuous pose/mouth movement without the old grid corruption. This is a limited visual sample, not a ten-video audio or lip-sync rating.
