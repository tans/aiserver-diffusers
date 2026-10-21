# Working state

Issue truth: https://github.com/tans/aiserver-diffusers/issues/1 (open).

Independent server directory: `/home/minixm/AIserver-diffusers`; loopback API 8190. Original service remains `/home/minixm/AIserver` and public relay 8189. Never edit its code or environment for this task.

Passed: full native Diffusers T2VA 512×320/124f/20 forwards, original video/audio VAEs, MP4 validation, strict TP6 loader, TP3 finite forward, dual TP3 complete two-MP4 cache-warm generation (188.75 s pair), upload, queued cancellation, nearest-frame API compatibility. Default worker remains TP6.

Active: reference conditioning with equivalent patchify GEMM; API cold/warm and Turbo8/native sm70 attention probes. Candidate modes remain disabled until their own evidence passes.

Remaining: image-reference complete generation, production dual cold admission, larger geometry feasibility, sustained workloads, matched LightX comparison, optional quantization/offload/residency evaluation, public routing and real reboot recovery.

Every source edit is committed/pushed on main. No test code is added. Benchmarks perform real local inference under the shared GPU lock and wait for existing service jobs.
