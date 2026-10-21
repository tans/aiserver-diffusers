# Implementation plan

Tracking issue: https://github.com/tans/aiserver-diffusers/issues/1

1. Audit binaries/topology/weights and pin Diffusers source; create an independent writable venv. Reuse installed POWER9 CUDA binaries through read-only package paths, never pip-upgrade the production environment.
2. Implement a strict local weight adapter only where Diffusers native loading is incompatible. Start with FP16 + SDPA, staged TE/DiT/VAE memory, one full audio/video output. Record real forward count.
3. Evaluate actual tensor parallel support, layer placement and offload separately. Label each mode accurately. Test one three-card group before two workers. Coordinate new group locks with the original all-GPU lock using shared/exclusive flock admission.
4. Reuse the public video API schema in a new application backed by a durable SQLite queue and worker subprocesses. A worker crash must not take down the API. Only advertise verified engine configurations.
5. Validate local Turbo LoRA formats, seeds and FP16 numeric safety; stage performance optimizations independently. Cache keys include dependency/weight identity and all conditioning inputs.
6. Benchmark cold/warm and queued/concurrent workloads at increasing feasible sizes. Keep failed attempts and reasons. Deploy on a separate free port, leave the original service available, document measured recommendation and fallback.

No prediction is a benchmark. A device_map or sequential layer placement is not TP. Dual TP3 is not the default until verified.
