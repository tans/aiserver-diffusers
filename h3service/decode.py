"""Native Diffusers H3 video/audio VAE decoding and CPU FFmpeg export."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .components import video_vae, audio_vae


def decode(source, output, vae_tile_size=256):
    from .memory import prefer_inference_eviction
    prefer_inference_eviction()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    started = time.perf_counter()
    latent = load_file(source)
    device = "cuda:0"
    vae = video_vae(device, tile_size=vae_tile_size)
    times = {"video_load": time.perf_counter() - started}
    t = time.perf_counter()
    x = latent["video"].to(device)
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    raw = vae.decode((x * std + mean).to(vae.dtype), return_dict=False)[0].float().cpu()
    pixels = (raw * torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
              + torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)).clamp(0, 1)
    if not bool(torch.isfinite(pixels).all()):
        raise FloatingPointError("Non-finite video VAE output")
    frames = (pixels[0].permute(1, 2, 3, 0) * 255).round().byte().contiguous().numpy()
    times["video_decode"] = time.perf_counter() - t
    del vae, x, raw, pixels
    gc.collect()
    torch.cuda.empty_cache()
    t = time.perf_counter()
    vae = audio_vae(device)
    x = latent["audio"].to(device)
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1)
    samples = vae.decode(x * std + mean, return_dict=False)[0].float().cpu()
    if samples.shape[0:2] != (2, 1) or not bool(torch.isfinite(samples).all()):
        raise ValueError(f"Invalid stereo audio: {samples.shape}")
    times["audio_load_decode"] = time.perf_counter() - t
    output = Path(output)
    pcm = output.with_suffix(".wav")
    with wave.open(str(pcm), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(32000)
        w.writeframes((samples[:, 0].T.numpy().clip(-1, 1) * 32767).astype("<i2").tobytes())
    t = time.perf_counter()
    count, height, width, _ = frames.shape
    temp = output.with_name(output.stem + ".encoding.mp4")
    command = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", "24", "-i", "pipe:0", "-i", str(pcm), "-c:v", "libx264", "-crf", "17", "-preset", "veryfast",
               "-x264-params", "asm=0", "-threads", "8",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "32000", "-ac", "2", "-movflags", "+faststart", str(temp)]
    subprocess.run(command, input=frames.tobytes(), check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(temp), "-f", "null", "-"], check=True)
    temp.replace(output)
    pcm.unlink()
    times["encode_mp4"] = time.perf_counter() - t
    report = {"engine": "diffusers", "video_latents_denormalized": True, "vae_tile_size": vae_tile_size,
              "frames": count, "width": width, "height": height, "fps": 24,
              "audio_samples": samples.shape[-1], "sample_rate": 32000,
              "timing_seconds": times, "total_seconds": time.perf_counter() - started,
              "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
    output.with_suffix(".decode.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--latents", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--vae-tile-size", type=int, choices=[0, 256, 512], default=256)
    a = p.parse_args()
    decode(a.latents, a.output, a.vae_tile_size)
