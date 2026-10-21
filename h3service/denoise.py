"""Distributed Diffusers denoising entry point; writes unpacked diffusion-space latents."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from .tensor_parallel import load_transformer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", default="/srv/models/lightx2v/h3/diT/bf16_tp6")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--frames", type=int, default=124)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--attention", choices=["sdpa"], default="sdpa")
    args = p.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    device = torch.device("cuda", rank)
    started = time.perf_counter()
    model, report = load_transformer(args.checkpoint, device, args.attention)
    print(json.dumps({"event": "loaded", **report}), flush=True)
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3CoreDenoiseStep
    pipe = MiniMaxH3ModularPipeline(blocks=MiniMaxH3CoreDenoiseStep())
    pipe.update_components(transformer=model, scheduler=MiniMaxH3Scheduler(shift=12.0),
                           audio_scheduler=MiniMaxH3Scheduler(shift=3.0))
    condition = load_file(args.condition, device=str(device))
    prompt = condition["prompt_embeds"].to(torch.float16)
    if prompt.ndim == 2:
        prompt = prompt[None]
    tags = condition.get("text_token_tags", torch.ones(prompt.shape[1], device=device, dtype=torch.long))
    forwards = []
    current = [0.0]
    def before(module, inputs):
        torch.cuda.synchronize()
        current[0] = time.perf_counter()
    def after(module, inputs, output):
        torch.cuda.synchronize()
        values = (output.sample, output.audio_sample) if hasattr(output, "sample") else output
        if not all(bool(torch.isfinite(t).all()) for t in values):
            raise FloatingPointError("Non-finite Diffusers transformer prediction")
        forwards.append(time.perf_counter() - current[0])
        if rank == 0:
            print(json.dumps({"event": "forward", "number": len(forwards), "seconds": forwards[-1]}), flush=True)
    model.register_forward_pre_hook(before)
    model.register_forward_hook(after)
    denoise_started = time.perf_counter()
    # Native H3 scheduler uses N sigma points for N-1 forward evaluations.
    state = pipe(prompt_embeds=prompt, text_token_tags=tags, width=args.width, height=args.height,
                 num_frames=args.frames, num_inference_steps=args.steps + 1,
                 generator=torch.Generator(device="cpu").manual_seed(args.seed),
                 output=["latents", "audio_latents"])
    torch.cuda.synchronize()
    if len(forwards) != args.steps:
        raise RuntimeError(f"Expected {args.steps} actual forwards, observed {len(forwards)}")
    report.update({"engine": "diffusers", "actual_forwards": len(forwards), "requested_steps": args.steps,
                   "forward_seconds": forwards, "denoise_seconds": time.perf_counter() - denoise_started,
                   "process_seconds": time.perf_counter() - started,
                   "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})
    reports = [None] * dist.get_world_size()
    dist.all_gather_object(reports, report)
    if rank == 0:
        latents, audio = state["latents"], state["audio_latents"]
        if not bool(torch.isfinite(latents).all() and torch.isfinite(audio).all()):
            raise FloatingPointError("Non-finite final latents")
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file({"video": latents.float().cpu().contiguous(), "audio": audio.float().cpu().contiguous()}, str(out),
                  metadata={"engine": "diffusers", "actual_forwards": str(len(forwards))})
        out.with_suffix(".report.json").write_text(json.dumps({"ranks": reports}, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
