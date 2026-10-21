# Reference quality gate — 2026-10-21

## Failure retained

Public reference Turbo4 seed 20261030 at 512×320/124 frames completed four actual forwards and valid audio/video streams, but full-size frames 0/40/80/120 showed conspicuous block texture. Output SHA256 `e11ce143b28080e8280ec7640eab0347c36d538c671836877ef23eaceb14ce9f`. The first mixed workload was aborted after fourteen accepted jobs completed technically; it is not twenty-task acceptance, and this reference output is not counted as usable video. Earlier small thumbnail review did not establish robust quality.

## Controlled comparisons

- The old CLI and public conditioning are bit-identical for prompt embeddings, token tags and reference video latent. Encoder/cache differences are excluded for this pair.
- Local Turbo4 metadata identifies the reference-specific ref2v v0.1 adapter. Its upstream reference recipe uses video/audio flow shifts 12/3 and alpha 8, matching the selected settings: [upstream recipe](https://github.com/ModelTC/LightX2V/blob/main/configs/minimax_h3/dmd/minimax_h3_ref2av_4step.json). This is configuration evidence, not a proof of model quality or checkpoint byte identity.
- Decoding the **same bad Turbo4 latent** with VAE spatial tiling disabled completed at 5,825.41 MiB peak CUDA allocation, but block artifacts persisted. The issue is not resolved by switching off decoder tiling.
- Strict default-tiling re-decode checks raw VAE output **before clamping**. The Turbo4 raw output was finite, range −3.053 to 4.035. Re-encoded MP4 hash exactly matched the original bad output. This failure is not a nonfinite value hidden by pixel clipping.
- The **same input and seed 20261030**, using the eight-step adapter/configuration, produced clean full-size frame samples. A second seed 20261031 also showed clean frame samples and retained the reference boy's blue/white hoodie, dark hair and backpack.
- Both Turbo8 latents passed strict raw-VAE re-decode; hashes exactly matched their initial outputs. Audio remains 32 kHz stereo and MP4 checks pass. No human speech or lip-sync quality score is claimed.

| Reference Turbo8 seed | Conditioning cache | DiT process | Decode process | Inference total |
|---|---|---:|---:|---:|
| 20261030 | Hit | 206.06 s | 35.08 s | 241.68 s |
| 20261031 | Hit | 202.50 s | 36.10 s | 239.13 s |

The two-job caller wall window includes admission waiting and the separate untiled decode diagnostic, so it is not an isolated reference-throughput estimate. Native reference preprocessing encodes this 234×542 input at the released checkpoint's independent short-edge rule (2048), yielding 9,538 conditioning tokens; it does not use the output video's 512×320 canvas as the reference encoding resolution. This helps explain its much higher per-forward cost than the 34-token text-only example.

## Deployment decision

Public reference generation is restored **only with Turbo8 on exclusive TP6**, at 512×320/124 frames and one image. Turbo4 is withheld from public admission. Its exact low-step failure mechanism has not been established; no repaired-weight or universal-quality claim is made. Text Turbo8/quality continues on dual TP3. Per-workflow geometry gates prevent future text-only high-resolution admission from admitting untested reference geometry.

The corrected public twenty-job workload uses eighteen text Turbo8 and two reference Turbo8 requests. It is a new run; the aborted batch remains separate.

Evidence: `evidence/reference-conditioning-comparison.json`, `evidence/reference-turbo8-quality-30-31.json`, `evidence/reference-raw-decode-checks.json`, `evidence/reference-turbo4-seed30-untiled.json`, `evidence/public-mixed20-turbo4-aborted.json`. Full-size contact sheets and MP4s remain generated artifacts outside Git.
