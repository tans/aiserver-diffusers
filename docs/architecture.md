# Architecture

The FastAPI process only validates requests and writes durable SQLite jobs. Independent systemd workers claim queued tasks transactionally, preserving priority and enqueue order. Every task runs conditioning, distributed denoising and native VAE decode in separate process groups; a failed CUDA context cannot poison the API or the next task.

The deployment can switch at an idle boundary between one six-card worker and two three-card workers; the current accepted small-geometry text mode is dual TP3. Its GPU admission takes the original service's existing all-GPU lock exclusively, without editing the original code. Candidate three-card workers take that same lock shared and each selected GPU's new lock exclusively. Thus original LightX2V jobs retain exclusive access and new groups can overlap only when disjoint. TP3 startup is blocked until dual-worker validation is recorded.

Diffusers 0.41.0 owns H3 modular layout, noise creation, scheduler stepping, transformer block structure and native VAEs. The independent adapter loads local AdaLN-curve weights and supplies column-sharded Q/K/V, paired SwiGLU shards and row-sharded output projections with real distributed reductions. This is actual tensor parallelism; the text encoder's balanced device map is explicitly model placement, not TP.

V100 mixed precision keeps residual/modulation/norm arithmetic in FP32 while large matrix weights use FP16. SDPA is the baseline. No LightX2V execution or import is allowed in the new engine. Original weight files are opened read-only.

An independent conditioning cache hashes the prompt, reference contents, target geometry, tokenizer/model-file version identity, source and dependency pins. TE weights are freed before the DiT stage. Initial caching avoids repeated TE work for identical conditioning. It does not make the DiT resident across the current per-job subprocess boundary.

The API rejects unverified workflows, geometries and few-step LoRA configurations. The capability record is generated only after successful measured runs. Worker B is enabled after the cold pair and ten-task Turbo8 gates passed. It only claims eligible text tasks; worker A expands to TP6 for reference tasks and other separately admitted non-TP3 geometries. An intent lock lets already running TP3 work finish before TP6 obtains the exclusive old-service GPU lock.

Recovery: a restarted worker explicitly fails its previously running jobs; queued jobs remain persisted. systemd reaps the worker's entire control group. Running cancellation stops the stage process group; terminal results are atomically published after FFmpeg validates the whole MP4. Storage admission rejects new work below a free-space reserve. Retention is a separate maintenance command scoped exclusively to this service's data directory.
