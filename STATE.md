# Working state

Issue truth: https://github.com/tans/aiserver-diffusers/issues/1 (open).

Independent server directory: `/home/minixm/AIserver-diffusers`; loopback API 8190. Original service remains `/home/minixm/AIserver` and public relay 8189. Never edit its code or environment for this task.

Passed: full native Diffusers T2VA 512×320/124f/20 forwards, original video/audio VAEs, MP4 validation, strict TP6 loader, TP3 finite forward, dual TP3 complete two-MP4 cache-warm generation (188.75 s pair), upload, queued cancellation, nearest-frame API compatibility. Default worker remains TP6.

Active: captured forward 17 for a cold TP3 prompt fails at block 45 FFN row projection; numeric fix pending evidence. Public Turbo8 API generation and result hash passed. Two Turbo4 reference seeds passed MP4 checks. Cold three-card TE passed, but the full dual cold pair was 1/2 because of this numeric failure. Candidate modes remain disabled until their own evidence passes.

Remaining: image-reference complete generation, production dual cold admission, larger geometry feasibility, sustained workloads, matched LightX comparison, optional quantization/offload/residency evaluation, public routing and real reboot recovery.

Every source edit is committed/pushed on main. No test code is added. Benchmarks perform real local inference under the shared GPU lock and wait for existing service jobs.

Kernel journal confirmed first-load host OOM affected original and new APIs before recovery. Inference is now under a shared 24 GiB cgroup memory cap, 22 GiB soft threshold and OOM eviction priority 800. Do not describe the earlier API as continuously available.
