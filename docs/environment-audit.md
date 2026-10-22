# Environment audit — 2026-10-21

## Live observations

- Host `minixm`: POWER9 / ppc64le, six Tesla V100 SXM2 16 GB.
- GPU topology: 0–2 and 3–5 each form NV2 groups; cross-group links are SYS.
- NUMA CPU nodes are 0 and 8; six additional memory-only nodes 250–255 exist. Live zone inspection reports Normal on CPU nodes 0/8 and Movable on nodes 250–255. Treat the 126 GiB total as NUMA memory inventory, not as a proven 126 GiB ordinary CPU model allocation budget.
- Linux reports 126 GiB total and approximately 80 GiB available at the initial audit; CPU-local Normal nodes total approximately 32 GiB; swap 1.5 GiB, partially occupied. Both workers retaining complete CPU model copies is unsafe without measurement.
- Free storage: `/home` on root filesystem 205 GiB, `/srv` 113 GiB. Reuse weight files read-only; do not copy a full model tree.
- Existing read-only Python stack: Python 3.12, torch 2.10.0, diffusers 0.41.0, transformers 5.20.1, accelerate 1.15.0, safetensors 0.8.0, FastAPI 0.142.2, uvicorn 0.54.0, PyAV 12.3.0, numpy 2.5.4, PEFT 0.21.2. New runtime must isolate package writes.
- Existing diffusers contains MiniMax H3 transformer, modular blocks, video/audio VAE and scheduler. Installed distribution has no direct_url commit metadata; source integrity must be pinned separately before deployment.
- Existing API `/home/minixm/AIserver`, framework `/srv/lightx2v`; old API remains on 8090 / relay 8189.
- Existing all-GPU lock `/home/minixm/AIserver/data/gpu0-5.lock` is an interoperability boundary. New GPU work must coordinate with it without changing old code.

## Weight compatibility findings

- Existing TE is original Qwen3VL 32B: 64 layers, hidden 5120, 64 attention heads and 8 KV heads. Tokenizer and processor are local.
- Existing video/audio VAE have Diffusers metadata. Native Diffusers video decode already ran on saved latents during the preceding quality diagnosis.
- Existing LightX2V DiT is converted, AdaLN-curve compressed, 60 heads padded from 56; FFN 14340. These are not directly loadable into an unmodified Diffusers transformer state dict.
- Existing original/pruned single files and multiple LoRA layouts require tensor-key inspection. Reuse only after a strict loader proves shapes and key coverage.
- Community reference pins Diffusers H3 PR #14355 at f37ab93e621d5ce206c9662e8291ca8b67d9c555. PR is merged; current local package is newer. Inspect both rather than assuming identical APIs.

## Feasibility gates

1. Strict local checkpoint loading and one Diffusers FP16 forward.
2. Complete T2VA and image-reference generation, including original VAE denormalization and synchronized audio.
3. Measure real multi-GPU implementation: device mapping is model parallel placement, not TP.
4. TP3 cannot be advertised solely by selecting three CUDA devices. Enable dual workers only after a memory and throughput result.

Audit is ongoing. Driver/CUDA details, P2P execution, package source pins, baseline model formats and performance will be added from measured evidence.

Live systemd inspection also confirmed the existing global service drop-in binds memory to CPU NUMA nodes 0 and 8, excluding memory-only HBM nodes 250–255. This existing guard was not modified. The approximately 32 GiB CPU-local budget is therefore a material constraint despite the 126 GiB inventory.
