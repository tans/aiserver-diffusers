# Compatibility and isolation

Reference quality status: Turbo4 is withheld after a full-size seed 20261030 failure (issue #2). Controlled same-input Turbo8 seeds 20261030/31 passed full-size frame review and strict raw-VAE re-decode; public one-image reference admission is restored only for Turbo8/TP6 at 512×320/124f. The corrected mixed workload is a new run; historical successful executions do not establish usable-video quality.

## Runtime pins

Diffusers 0.41.0 corresponds to upstream tag commit `086bf9578c0f4acbc66e48cd1e7cc26befd9e10f`. Transformers 5.20.1, Accelerate 1.15.0, Safetensors 0.8.0, FastAPI 0.142.2 and Uvicorn 0.54.0 are installed into the independent venv. See requirements.txt and the host's data/environment-packages.json for observed dependencies.

POWER9 torch 2.10.0 / CUDA binaries and remaining existing host dependencies are read-only imports via a `.pth` path. This is package-write isolation, not a fully copied binary environment. The existing venv must remain available; bootstrap must never run its pip. No sm_80+ binary is installed for V100. SDPA is the public default. A directly loaded sm70 H3 native extension is an explicit CLI candidate; one complete seed passed, so it remains outside public admission.

## Local DiT adapter

Local `/srv/models/lightx2v/h3/diT/bf16_tp6` stores 635 tensors, padded 60-head attention and FFN 14340. Each tensor is validated against a meta-initialized Diffusers H3 model before slicing. Shard header keys must match the declared index, with duplicate keys rejected. The historical index refers to missing temporary shard names; the adapter reconstructs file locations from real headers in memory without editing shared weight files.

AdaLN stores an already activated 1025×8 basis curve. Native time embedding is replaced by linear interpolation of that curve, and a second SiLU is intentionally omitted in the modulation adapters. This does not substitute a LightX2V forward. Diffusers' transformer and modular scheduler remain the computation path.

Q/K/V projections are column sharded by attention head. Both SwiGLU halves are partitioned with identical rank slices. Attention/FFN output weights are row sharded, with FP32 distributed sums. Norm/modulation/input/output/text projection arithmetic uses FP32 where necessary; bulk matrices use FP16. A diagnostic per-module finite gate identifies the first offending output and never clamps or replaces NaNs.

## Public contract

The new service implements the original model/task/input/parameters/options request envelope, file uploads, job submission/list/status/cancel, MP4 results and thumbnails. Job IDs and timestamps follow the existing shape. Duration maps to the nearest legal frame slot, preserving existing requests such as 5.17 seconds → 124 frames. Original data is not imported into the new SQLite database.

Initial admission only permits verified H3 video profiles and geometries. `minimax-h3-fast`, Krea image generation, SeedVR super resolution and unverified LoRA requests are rejected instead of silently mapping to a different engine. Video/audio references are withheld until separately verified; image references are the first reference milestone. Running cancellation is an extension over the original queued-only cancellation.

## Current verification boundaries

Six-card strict weight loading succeeded at approximately 6.6 GiB of local parameters per rank. The initial forward diagnostic found FP16 text-projection overflow; that projection was promoted to FP32. A complete native T2VA 20-forward generation and video/audio decode have now passed at 512×320/124f. This is the only initial API geometry. TP3 and dual TP3 completed full exports with the expandable allocator. Two different-prompt cold quality tasks and ten consecutive cached-prompt Turbo8 tasks passed after the FP32 row-output fix. Reference Turbo4 passed two seeds on TP6. Quantization, DiT group offload and cross-task residency are unimplemented; native-kernel acceleration remains a CLI candidate.
