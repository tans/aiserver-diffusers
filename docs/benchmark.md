# Benchmark results — active validation

## Native Diffusers baseline

| Probe | Result | Measured time | Memory |
| --- | --- | --- | --- |
| Native Qwen3-VL conditioning, text only | Passed; 34 tokens | 25.18 s, including 23.21 s TE load | Six-GPU balanced placement, not TP |
| TP6, 512×320, 124 frames, 20 actual forwards | Passed | 17.79 s model load; 86.27 s denoise; 111.32 s process | 8,131.68 MiB peak CUDA allocation on rank 0 |
| Native video/audio VAE + MP4 | Passed | 23.91 s process | 5,679.57 MiB peak CUDA allocation |
| TP3, standard allocator, one forward | Failed OOM | Not a throughput result | Allocation failed near capacity |
| TP3, expandable allocator, one forward | Passed | 18.60 s load; 4.28 s forward | 14,589.59 MiB rank 0 peak CUDA allocation |
| First simultaneous dual TP3 load | Failed SIGKILL | 35.53 s until failure | 25.19 GiB sampled process-tree PSS |
| Staggered dual TP3, 20 forwards each | Denoising passed; one MP4 failed due to shared temporary filename | Approximately 75 s per denoiser | 17.36 GiB sampled tree PSS; GPU used up to 15,530 MiB |

The shared temporary filename was corrected. A complete cache-warm pair subsequently passed: 188.75 s for two verified MP4s, including staged weight loading, 152.55 s DiT pair phase, 18.34 GiB sampled PSS and up to 15,530 MiB GPU used memory. See reports/measured-baseline.md and raw JSON. Failed attempts remain recorded and are not counted as successful videos. A sum of separately measured stages is not a measured end-to-end API latency.

## Benchmark tooling

`scripts/benchmark.sh` submits real jobs to the independent API and writes JSON, CSV and Markdown: wall latency, success rate, videos/hour and generated seconds/hour. Reports identify queue waiting via job timestamps and worker stage timing. Completion is not a quality rating; usable-video proportion needs visual/audio review.

Required remaining progression: API cold/warm, image reference, full dual TP3 with cold/warm conditioning, higher-resolution feasibility, continuous 10 jobs and 20 mixed jobs. Larger geometry is rejected by the API until passed.

Recent original LightX2V 864×480/124f/20-forward T2VA jobs took approximately 303–307 seconds (TE ~40 s, DiT ~214–216 s, decode ~46–53 s). Their geometry differs from this Diffusers baseline; these values do not establish an engine speedup. Match geometry, input, seed, forward count and cold/warm state for an A/B claim.
