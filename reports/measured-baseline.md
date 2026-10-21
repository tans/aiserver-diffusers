# Measured baseline — 2026-10-21

Native Diffusers 0.41.0, local H3 padded checkpoint, FP16 bulk weights with FP32 sensitive operations, SDPA, 512×320/124f/20 actual forwards.

- TP6 DiT process: 111.32 s including model load; denoise 86.27 s. Separate native VAE + MP4 process: 23.91 s.
- Dual TP3 with staged weight loading and precomputed native conditioning: two verified MP4s in **188.75 s**, DiT pair phase 152.55 s. This is one pair, not a sustained throughput run. Arithmetic pair rate: 38.15 videos/hour.
- Sampled pair PSS peak: 18.34 GiB. `nvidia-smi` peak used memory: up to 15,530 MiB/card. Rank-0 tensor allocation peak: 14,573.37 MiB per denoiser.
- No claim of cold end-to-end dual throughput: conditioning was already generated. No matched original-engine speedup claim.
- Earlier simultaneous load SIGKILL and shared MP4 temporary-file failure are retained as failures in benchmark documentation. No cgroup oom_kill increment was observed across the successful rerun (15 before and after).

Raw rank timings and telemetry are in `evidence/`. CSV exposes comparable DiT process measurements. Visual/audio usability remains distinct from decode and finite-tensor correctness.
