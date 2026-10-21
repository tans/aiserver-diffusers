# Known limitations — 2026-10-21

- Native Diffusers T2VA, TP6, 512×320, 124 frames, 20 actual forwards has completed video/audio VAE decoding and MP4 verification. Only this geometry/task is initially admitted by the API.
- TP3 with the default CUDA allocator failed with OOM. `expandable_segments:True` passed a single forward and two concurrent 20-forward denoisers. Full dual MP4 acceptance has now passed for one cache-warm pair; cold-conditioning concurrency remains a separate gate.
- Simultaneous dual model loading suffered one SIGKILL. Kernel logs are unavailable to the service account, so host OOM is a hypothesis rather than a confirmed cause. Staged loading and CPU allocator trimming reduced sampled process-tree PSS on the subsequent attempt.
- Turbo4/8 and the native sm70 H3 attention kernel have experimental CLI routes. They are not public API profiles until complete multi-seed video validation.
- INT8 and CPU group offload are not enabled. The working FP16 TP3 candidate does not require weight quantization, but its activation margin is small.
- DiT is reloaded per task. Conditioning is cached with a source/model/input version key and complete-entry publication. Resident DiT across tasks remains future work.
- Reference videos/audio are not admitted. Image-reference encoding and complete generation are still being validated. Original image pixel resolution is preserved by the patchify GEMM adaptation.
- Native VAE uses one GPU with spatial tiling; encoding and decoding use staged subprocesses. CPU FFmpeg export is cancellable.
- Original and new services coordinate through the existing GPU lock. Queue waiting for original jobs must not be reported as inference compute time.
- CPU PSS and GPU metrics are sampled. Exact per-forward H2D time and compute/idle decomposition are not instrumented.
- The independent venv reuses the audited POWER9 torch binaries read-only. It isolates package writes but still depends on those shared binary files.
- API is loopback-only on port 8190. The server account cannot alter the root-managed cloud tunnel; no public endpoint is claimed.
- Only the API and default TP6 worker are enabled. Cleanup units are supplied but retention is not enabled until generation acceptance. Worker B remains gated.
- No OS reboot has been performed. Process restart recovery does not establish real reboot recovery.
- Full high-resolution, ten-job and twenty-job mixed workloads remain outstanding. Issue #1 remains open.
