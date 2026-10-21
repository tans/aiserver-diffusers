# Known limitations — 2026-10-21

- Native Diffusers T2VA, TP6, 512×320, 124 frames, 20 actual forwards has completed video/audio VAE decoding and MP4 verification. Only this geometry/task is initially admitted by the API.
- TP3 with the default CUDA allocator failed with OOM. `expandable_segments:True` passed a single forward and two concurrent 20-forward denoisers. Full dual MP4 acceptance has now passed for one cache-warm pair; cold-conditioning concurrency remains a separate gate.
- Root kernel logs later confirmed host memory-policy OOM during the first simultaneous-load attempt. Both original and new APIs, workers and relays were killed and restarted by systemd. Earlier post-event healthy responses proved recovery, not uninterrupted availability. Staged loading, allocator trimming, a separate bounded inference slice and high inference OOM eviction priority now reduce and isolate this pressure. See reports/evidence/initial-host-oom.json.
- Turbo8 text generation passed two seeds and is admitted explicitly via profile=turbo8. Turbo4 reference generation and the native sm70 attention kernel remain experimental CLI routes.
- INT8 and CPU group offload are not enabled. The working FP16 TP3 candidate does not require weight quantization, but its activation margin is small.
- DiT is reloaded per task. Conditioning is cached with a source/model/input version key and complete-entry publication. Resident DiT across tasks remains future work.
- Reference videos/audio are not admitted. Image-reference encoding and complete generation are still being validated. Original image pixel resolution is preserved by the patchify GEMM adaptation.
- Native VAE uses one GPU with spatial tiling; encoding and decoding use staged subprocesses. CPU FFmpeg export is cancellable.
- Original and new services coordinate through the existing GPU lock. Queue waiting for original jobs must not be reported as inference compute time.
- CPU PSS and GPU metrics are sampled. Exact per-forward H2D time and compute/idle decomposition are not instrumented.
- The independent venv reuses the audited POWER9 torch binaries read-only. It isolates package writes but still depends on those shared binary files.
- API listens on loopback 8190 and is published through a separate reverse SSH tunnel at http://minixm.minapp.xin:8190. External HTTP and browser API-documentation access passed; public generation/download is being followed through.
- Only the API and default TP6 worker are enabled. Cleanup units are supplied but retention is not enabled until generation acceptance. Worker B remains gated.
- No OS reboot has been performed. Process restart recovery does not establish real reboot recovery.
- Full high-resolution, ten-job and twenty-job mixed workloads remain outstanding. Issue #1 remains open.

- The first cold dual-worker pair failed one prompt at forward 17. A captured-forward replay reproduced overflow at block 45 FFN output projection. Requesting FP32 row-projection output before TP reduction fixed that exact forward and both complete 20-step cold tasks subsequently passed. Public dual admission remains gated on the sustained Turbo8 batch.
