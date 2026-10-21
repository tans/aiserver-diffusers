"""Strict experimental Turbo folding into the existing TP shards."""
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open


TURBO_ROOT = Path("/srv/models/lightx2v/h3/lora/turbo_tp6")
FILES = {
    "turbo4": "minimax_h3_turbo_4step_v0.1_lightx2v.safetensors",
    "turbo8": "minimax_h3_turbo_8step_v1.0_lightx2v.safetensors",
}


def fold_turbo(model, profile):
    rank, world = dist.get_rank(), dist.get_world_size()
    path = TURBO_ROOT / FILES[profile]
    consumed, count = set(), 0
    with safe_open(path, framework="pt", device="cpu") as state:
        keys, meta = set(state.keys()), state.metadata() or {}
        suffix_a, suffix_b = ((".lora_down.weight", ".lora_up.weight") if profile == "turbo4"
                              else (".lora_A.default.weight", ".lora_B.default.weight"))
        for key in sorted(k for k in keys if k.endswith(suffix_a)):
            name = key[:-len(suffix_a)]
            other = name + suffix_b
            module = model.get_submodule(name)
            if not hasattr(module, "mode") or other not in keys:
                raise ValueError(f"Unsupported Turbo target {name}")
            down, up = state.get_tensor(key), state.get_tensor(other)
            rows, cols = module.weight.shape
            full_rows = rows * world if module.mode in {"column", "swiglu"} else rows
            full_cols = cols * world if module.mode == "row" else cols
            if down.ndim != 2 or up.shape != (full_rows, down.shape[0]) or down.shape[1] != full_cols:
                raise ValueError(f"Turbo shape mismatch at {name}")
            alpha_key = name + ".alpha"
            if alpha_key in keys:
                alpha = float(state.get_tensor(alpha_key))
                consumed.add(alpha_key)
            elif "alpha" in meta:
                alpha = float(meta["alpha"])
            else:
                raise ValueError("Turbo alpha must be explicit; refusing a guessed scale")
            scale = alpha / down.shape[0]
            if module.mode == "column":
                up = up[rank * rows:(rank + 1) * rows]
            elif module.mode == "swiglu":
                half, step = full_rows // 2, rows // 2
                up = torch.cat((up[rank * step:(rank + 1) * step],
                                up[half + rank * step:half + (rank + 1) * step]))
            elif module.mode == "row":
                down = down[:, rank * cols:(rank + 1) * cols]
            device = module.weight.device
            down = down.to(device=device, dtype=torch.float32)
            for start in range(0, rows, 256):
                delta = up[start:start + 256].to(device=device, dtype=torch.float32) @ down
                value = module.weight[start:start + 256].float() + delta * scale
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f"Non-finite folded Turbo weights at {name}")
                module.weight[start:start + 256].copy_(value)
            consumed.update((key, other))
            count += 1
            del down, up, delta, value
        if consumed != keys or count != 312:
            raise ValueError(f"Incomplete Turbo adapter: targets={count}, unused={sorted(keys-consumed)}")
    torch.cuda.empty_cache()
    return {"profile": profile, "path": str(path), "targets": count, "scale_convention": "alpha / rank"}
