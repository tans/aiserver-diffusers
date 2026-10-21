# Working state

Issue truth: https://github.com/tans/aiserver-diffusers/issues/1 (open).

Independent server directory: `/home/minixm/AIserver-diffusers`; loopback API 8190. Original service remains `/home/minixm/AIserver` and public relay 8189. Never edit its code or environment for this task.

Passed: full native Diffusers T2VA 512×320/124f/20 forwards, original video/audio VAEs, MP4 validation, strict TP6 loader, TP3 finite forward, dual TP3 complete two-MP4 cache-warm generation (188.75 s pair), upload, queued cancellation, nearest-frame API compatibility. Default worker remains TP6.

Passed: ten consecutive Turbo8 tasks, concurrency two, isolated TP3 API 8192; 10/10 succeeded. Wall throughput 40.98 videos/hour includes initial old-service GPU wait; active-window throughput 61.37/hour over 586.61 seconds. Public default remains TP6 pending mode switch and fault acceptance.

Numerical fix: captured forward 17 failed reproducibly at block 45 FFN row projection. FP32 row-projection outputs resolved the captured forward and both different-prompt full 20-forward cold jobs passed (276.39 s pair wall time). The original failed attempts are retained. Public Turbo8 submission/result hash and two reference Turbo4 seeds also passed.

Pending higher-resolution unit h3-highres-turbo8 is stopped before its first GPU task so the ten-job throughput run has an independent window. Restart it after the batch; it runs two seeds each at 1024×576 and 1280×768, Turbo8, under the inference memory slice.


Remaining: production mode switching, running cancellation and crash recovery, larger geometry full generation, twenty mixed tasks, final matched mode comparison and real reboot recovery. Reference Turbo4 generation and public Turbo8 generation/download have passed. INT8/offload/residency remain unimplemented candidates.

Every source edit is committed/pushed on main. No test code is added. Benchmarks perform real local inference under the shared GPU lock and wait for existing service jobs.

Kernel journal confirmed first-load host OOM affected original and new APIs before recovery. Inference is now under a shared 24 GiB cgroup memory cap, 22 GiB soft threshold and OOM eviction priority 800. Do not describe the earlier API as continuously available.

Runtime acceptance passed: actual-forward SIGKILL followed by WORKER_RESTARTED and separate actual-forward user cancellation; API remained available, no GPU subprocess remained at checkpoints. Raw evidence reports/evidence/runtime-acceptance.json. Candidate units were stopped afterwards. Public TP6 two-job Turbo8 comparison is active.
