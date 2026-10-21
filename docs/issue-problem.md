## Problem

Build a separate Diffusers MiniMax H3 service on IBM AC922 POWER9 / 6×V100 16 GB, preserving the running LightX2V service, repository and Python environment. Preserve the existing video API contract. Image generation and super resolution may be omitted.

The desired dual TP3 architecture is unproven. Existing weights include compressed AdaLN and LightX2V-specific layouts; package compatibility, true TP, per-card memory and host memory are feasibility constraints.

## Acceptance

- Independent repo, directory, venv, data store, service units and listening port.
- Real Diffusers T2VA and image-reference generation with correct video/audio; never silently call LightX2V.
- Document candidate TP6/TP3/offload/quantization configurations and enable only proven modes.
- Durable job state, resource admission shared with existing GPU lock, cancellation, failure isolation and recovery.
- Fixed dependency/source revisions and local checkpoint inventory; no duplicate full model download.
- Cold/warm/concurrent/mixed workload benchmarks with JSON/CSV/Markdown records, performance and quality limitations stated honestly.
- Commit and push on main, close milestone issues only after evidence satisfies their scope.
