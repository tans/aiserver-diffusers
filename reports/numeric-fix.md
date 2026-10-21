# FP16 projection-output overflow — fixed, 2026-10-21

The first cold dual-worker pair completed the blue-hoodie prompt, while the red-raincoat/cat prompt (seed 20261022, quality 20, 512×320/124f) failed at forward 17. Finite checks prevented publication.

A complete diagnostic run captured forward 17 inputs. A one-forward replay with the original output dtype again failed first at `transformer_blocks.45.ff.net.2`. Its FP16 input-cast check did not fail. A replay of the identical tensors using `torch.mm(..., out_dtype=torch.float32)` for row projections completed with finite video and audio predictions.

Bulk weights and GEMM inputs remain FP16. FP32 projection outputs avoid FP16 packing/reduction range loss before the existing FP32 tensor-parallel reduction and rescaling. No NaN replacement or clipping was introduced. `H3_FP32_LINEAR_OUTPUT=0` retains the old diagnostic path; production defaults to FP32 output.

The subsequent two different-prompt cold jobs both completed 20 actual forwards, native video/audio VAE decode, FFmpeg verification and MP4 publication. Pair submission-to-completion wall time was 276.39 s, including approximately 11 s of admission wait. This is a single cold pair, not a sustained hourly production result. Raw task records are in evidence/dual-cold-v2.json.

The inference slice had no OOM events during this rerun. API generation and result checks are separate from subjective audio/lip-sync quality.
