# Optimization results

## Implemented mechanisms

- Local mmap/shard slicing avoids loading a full CPU model copy per rank.
- Native Diffusers transformer uses real tensor parallel linear layers and bounded activation chunks.
- FP32-sensitive residual/modulation/projection arithmetic addresses FP16 range limits without replacing non-finite outputs.
- Staged conditioning/denoising/VAE processes free components between phases.
- Identical conditioning requests have a versioned cache.

## Measured results

Only strict six-card weight loading is accepted so far. Throughput, quality and production defaults are not yet determined. The first numeric probe identified an overflow at the context projection; the adapter now keeps it in FP32 and verification is continuing.

## Decisions pending measurement

True TP3 feasibility, dual-worker hourly production, Turbo4/8 quality, INT8 performance and group-offload transfer cost. Candidate code or configuration is not a measured performance result.
