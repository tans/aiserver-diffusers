"""Native Diffusers VAE loading; local checkpoints are opened read-only."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from safetensors.torch import load_file

MODEL_ROOT = Path("/srv/models/lightx2v/h3/model_root")


def video_vae(device):
    model = AutoencoderKLMiniMaxH3.from_pretrained(
        str(MODEL_ROOT / "vae"), torch_dtype=torch.float16, local_files_only=True)
    model.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256,
                        tile_sample_min_overlap_height=64, tile_sample_min_overlap_width=64)
    return model.eval().to(device)


def audio_vae(device):
    root = MODEL_ROOT / "audio_vae"
    cfg = json.loads((root / "config.json").read_text())
    meta = json.loads((root / "metadata.json").read_text())["metadata"]["kwargs"]
    if meta["decoder_type"] != "bigvgan" or not meta["attn_proj"]:
        raise ValueError("Unsupported original audio VAE architecture")
    args = {k: meta[k] for k in ("encoder_dim", "encoder_rates", "latent_dim", "decoder_dim", "decoder_rates")}
    args.update(latent_channels=meta["vae_latent_channels"], sampling_rate=meta["sample_rate"],
                decoder_kernel_sizes=[2 * r - r % 2 for r in meta["decoder_rates"]],
                latents_mean=cfg["latents_mean"], latents_std=cfg["latents_std"])
    if cfg["latent_channels"] != args["latent_channels"] or cfg["sample_rate"] != args["sampling_rate"]:
        raise ValueError("Audio metadata disagrees with wrapper configuration")
    model = AutoencoderKLMiniMaxH3Audio(**args)
    model.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
    return model.eval().to(device=device, dtype=torch.float32)
