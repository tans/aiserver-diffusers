# Working state — 2026-10-22

Issue truth: https://github.com/tans/aiserver-diffusers/issues/1 (closed 2026-10-22); Turbo4 containment: https://github.com/tans/aiserver-diffusers/issues/2 (closed 2026-10-22 with Turbo4 still disabled).

## Current deployment

- Independent server directory `/home/minixm/AIserver-diffusers`, own venv and SQLite/data. Public API http://minixm.minapp.xin:8190, docs `/docs`; the 8188 penetration alias is live as a loopback TCP proxy to the same API. Original service remains `/home/minixm/AIserver` on public 8189.
- White-theme web console (commit `ce9616f`) is live on 8188 and 8190: index page, per-job detail page, and `/llms.txt` / `/llms-full.txt` machine-readable documentation. Verified over the public 8188 entry on 2026-10-22; 8189 legacy health stayed 200. Only the API unit was restarted; workers and the proxy unit were untouched.
- API, worker A, worker B and the separate reverse tunnel are enabled. User linger is enabled. `scripts.set_mode dual-tp3` was exercised successfully; latest `/ready` reports A [0,1,2], B [3,4,5]. Text quality/Turbo8 uses dual TP3; reference Turbo8 obtains exclusive TP6 through A.
- Public geometry remains 512×320/124 frames. Current admission: text quality (20+ actual forwards), text Turbo8 (eight), and one-image reference Turbo8 (eight) on exclusive TP6. Turbo4 is withheld after the visual-quality failure recorded in issue #2.
- Daily cleanup timer is enabled; next scheduled run 2026-10-22 00:00 CST. It expires this service's seven-day terminal job artifacts/cache and preserves metadata; inputs are not automatically pruned. No old-service data is touched.
- Inference memory slice: MemoryHigh 22 GiB, MemoryMax 24 GiB, MemorySwapMax 1 GiB, inference OOM eviction priority 800. Subsequent max/oom/oom_kill counters remain zero; host historical oom_kill remains 15.

## Passed evidence

- Native Diffusers TP6 and TP3, original video/audio VAEs and verified MP4 output. Two different-prompt cold dual quality jobs completed twenty actual forwards each after the FP32 row-output fix, pair wall 276.39 s.
- Turbo8 text and reference fixed-seed complete outputs. Public browser submission and result download/hash passed. Turbo4 reference is withheld; its visual root cause is not known.
- Ten consecutive cached-prompt Turbo8 dual-TP3 tasks: 10/10 succeeded, five per group. Batch wall 878.44 s includes initial legacy GPU waiting; active window 586.61 s gives 61.37 videos/hour. Mean inference 113.15 s.
- Two serial TP6 cached-prompt Turbo8 tasks: 2/2 succeeded, mean inference 105.34 s, active-window rate 34.16/hour. Observed short-window ratio about 1.80x; sample sizes differ and no hour-long capacity claim is made.
- Upload, queued cancellation, nearest-frame compatibility (5.17 seconds maps to 124), isolated storage-admission 507.
- SIGKILL during actual denoising: API stayed available; restarted worker marked interrupted work failed/WORKER_RESTARTED. Subsequent actual-forward user cancellation reached cancelled; GPU process groups were reaped.

## Completed final acceptance and visual review

- The first mixed workload was aborted after reference Turbo4 seed 20261030 showed block artifacts. Fourteen accepted jobs finished technically; this is not twenty-job acceptance. Raw initial/final rows remain in reports/evidence/public-mixed20-turbo4-aborted.json.
- Controlled comparison passed: exact conditioning tensors match; disabling VAE tiling does not fix the bad Turbo4 latent; Turbo8 seeds 20261030/31 have clean full-size frame samples and strict raw-VAE finite checks with identical re-decode hashes. Reference admission is restored only for Turbo8/TP6. See reports/reference-quality.md and issue #2.
- The corrected public mixed workload completed 20/20: eighteen text Turbo8 and two one-image reference Turbo8, 512×320/124f. A mid-video frame from every output was reviewed; no obvious recurrence of the withheld Turbo4 block artifacts. Reference seeds were 20261030 and 20261040.
- The isolated candidate completed one cold quality20 output at 512×320 and two text Turbo8 outputs each at 1024×576 and 1280×768. Three frames (0/2.5/5 seconds) per video were reviewed; no obvious decoding or block artifacts in those samples. See `reports/final-visual-review-2026-10-22.md`.
- Candidate units `h3-final-geometry-api.service` and `h3-final-geometry-worker.service` were stopped after result collection. Public geometry remains 512×320; no larger mode is advertised. The 1280×768 candidate is close to the 16 GB device memory ceiling and has only two samples.
- A same-prompt/current-seed quality20 original LightX2V control task completed. This records functional completion only; no visual side-by-side or speed superiority claim was made.
- No additional performance tests were run in this continuation as requested. Earlier throughput records remain historical evidence.
- Issues #1 and #2 were closed with comments that state the remaining boundaries and make no claim that Turbo4 was repaired or that host reboot recovery was validated.

## Remaining boundaries

- Server Git metadata has not been reconciled with the deployed archive revision; runtime readiness and acceptance were checked separately. Reconcile metadata only if needed for later operations.
- No real host reboot has been performed. Worker restart and enabled units do not prove host reboot recovery. A reboot affects the original service and other active jobs and needs an agreed maintenance boundary.
- INT8, cross-task DiT residency and DiT group offload are unimplemented candidates. Native sm70 attention passed one historical complete seed and remains an explicit CLI candidate, not the public default. Exact H2D/compute/idle decomposition and human lip-sync ratings are unmeasured.
- The initial simultaneous-load attempt caused kernel-confirmed host memory-policy OOM that killed original and new APIs before systemd recovery. The later bounded configuration has not repeated it; do not describe the initial API as continuously available.

Every source/doc change is committed and pushed on main. No test code is added. Real inference benchmarks use GPU locks and yield to recent original-service work. The new venv reuses the original POWER9 binary packages read-only; package writes are isolated but the binaries remain a dependency.
