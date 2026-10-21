# Known limitations — active implementation

- No engine configuration has completed end-to-end acceptance yet; API generation admission remains disabled.
- TP6 weight loading is proven; TP3 and dual TP3 have not yet passed actual forward/memory checks.
- LoRA, INT8, custom V100 attention and CPU group offload are not enabled.
- The current design restarts the DiT subprocess for every task. Conditioning can be cached, but the DiT is not yet resident across jobs.
- Reference videos/audio are not admitted. Image-reference blocks require full encode/denoise/decode validation.
- Initial native VAE uses a single GPU with spatial tiling; this may be slower than the original six-card decode.
- The original service and new service share GPU admission. While original jobs run, new benchmarks wait and their wall-clock waiting time is not inference latency.
- CPU process-tree PSS and GPU telemetry are sampled. Per-forward transfer time and an exact CPU/GPU compute/idle decomposition are not instrumented yet; do not infer them from wall time.
- The new venv reuses the audited POWER9 torch stack read-only; removing or changing shared binary dependencies can invalidate the new runtime.
- The API currently listens on loopback 8190. Public routing is a later deployment step; an active unit and HTTP health response do not prove generation success.
- Result files expire after the configured retention period; metadata remains. No source weights or old-service media are subject to retention.
- No operating-system reboot has been performed. Recovery logic and enabled units are not evidence of a successful real reboot.
