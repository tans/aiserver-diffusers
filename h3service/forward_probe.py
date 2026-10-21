"""Replay a captured native forward to isolate numerical failures."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .memory import prefer_inference_eviction
from .tensor_parallel import load_transformer


def main():
    prefer_inference_eviction()
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    model, report = load_transformer("/srv/models/lightx2v/h3/diT/bf16_tp6", device)
    with safe_open(a.capture, framework="pt", device="cpu") as f:
        meta = f.metadata()
    kwargs, devices = json.loads(meta["extra"]), json.loads(meta["devices"])
    kwargs.update({k: v.to(device if devices[k] == "cuda" else "cpu") for k, v in load_file(a.capture).items()})
    result = model(**kwargs)
    values = (result.sample, result.audio_sample) if hasattr(result, "sample") else result
    if not all(bool(torch.isfinite(t).all()) for t in values):
        raise FloatingPointError("Non-finite captured forward prediction")
    if rank == 0:
        save_file({"video": values[0].float().cpu(), "audio": values[1].float().cpu()}, a.output)
        Path(a.output).with_suffix(".json").write_text(json.dumps(report, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
