"""TP adapter for Diffusers' H3 modules and the local padded AdaLN-curve checkpoint.

Diffusers owns the transformer and modular denoising. This adapter only supplies
checkpoint loading, real column/row tensor parallel linears and V100 numerics.
No LightX2V package is imported or executed.
"""
from __future__ import annotations

import json
import os
import time
import types
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import init_empty_weights
from safetensors import safe_open
from torch import nn


class CurveTime(nn.Module):
    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", table.float())

    def forward(self, timestep):
        position = timestep.float().clamp(0, 1) * (len(self.table) - 1)
        lower = position.floor().long().clamp(max=len(self.table) - 2)
        return torch.lerp(self.table[lower], self.table[lower + 1], (position - lower)[:, None])


class TPLinear(nn.Module):
    def __init__(self, source, mode="replica", promote=False, scale=1.0, chunk=2048):
        super().__init__()
        self.weight = source.weight
        self.bias = source.bias
        self.mode, self.promote, self.scale, self.chunk = mode, promote, scale, chunk

    def forward(self, x):
        # Bound FP32 FFN/projection intermediates along the token axis.
        if x.ndim == 3 and x.shape[1] > self.chunk:
            output = None
            for start in range(0, x.shape[1], self.chunk):
                part = self._forward(x[:, start:start + self.chunk])
                if output is None:
                    output = part.new_empty((x.shape[0], x.shape[1], part.shape[-1]))
                output[:, start:start + part.shape[1]] = part
            return output
        return self._forward(x)

    def _forward(self, x):
        x = (x / self.scale).to(self.weight.dtype) if self.scale != 1 else x.to(self.weight.dtype)
        output = F.linear(x, self.weight, None if self.mode == "row" else self.bias)
        if self.mode == "row" and dist.get_world_size() > 1:
            # FP32 collectives avoid FP16 overflow in a sum of rank partials.
            output = output.float()
            dist.all_reduce(output)
        if self.promote or self.scale != 1:
            output = output.float() * self.scale
        if self.mode == "row" and self.bias is not None:
            output = output + self.bias
        return output


class V100SDPA:
    """Public PyTorch SDPA with bounded query chunks, including the math fallback."""
    def __init__(self, chunk=512, backend="sdpa"):
        self.chunk, self.backend = chunk, backend

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

        q = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        k = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        v = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        q = attn.norm_q(q.float()).to(torch.float16)
        k = attn.norm_k(k.float()).to(torch.float16)
        if rotary_emb is not None:
            q = _apply_rotary_emb(q, *rotary_emb)
            k = _apply_rotary_emb(k, *rotary_emb)
        if self.backend == "h3_flashattn":
            if attention_mask is not None:
                raise ValueError("H3 flash attention does not support a mask")
            import h3kern
            # Verified separately before this experimental backend can be enabled.
            out = h3kern.flash_attn_func(q, k, v)
        else:
            q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))
            out = torch.empty_like(q)
            for start in range(0, q.shape[2], self.chunk):
                mask = attention_mask[..., start:start + self.chunk, :] if attention_mask is not None else None
                out[:, :, start:start + self.chunk] = F.scaled_dot_product_attention(
                    q[:, :, start:start + self.chunk], k, v, attn_mask=mask, dropout_p=0.0)
            out = out.transpose(1, 2)
        return attn.to_out[1](attn.to_out[0](out.flatten(2, 3)))


def curve_modulation(self, temb):
    # The stored basis is already the activated timestep curve: no second SiLU.
    return self.linear(temb.float()).reshape(-1, 6 * self.hidden_size).chunk(6, dim=-1)


def curve_final(self, hidden_states, temb, timestep_indices):
    shift, scale = self.linear(temb.float()).chunk(2, dim=-1)
    return self.norm(hidden_states.float()) * (1 + scale[timestep_indices]) + shift[timestep_indices]


def _replace(root, name, value):
    parent, _, leaf = name.rpartition(".")
    module = root.get_submodule(parent) if parent else root
    setattr(module, leaf, value)


def load_transformer(checkpoint: str, device: torch.device, attention="sdpa"):
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel, MiniMaxH3Attention,
    )

    started = time.perf_counter()
    root = Path(checkpoint)
    config = json.loads((root / "config.json").read_text())
    if not config.get("h3_adaln_curve"):
        raise ValueError("This strict loader accepts only the audited local AdaLN curve checkpoint")
    rank, world = dist.get_rank(), dist.get_world_size()
    if config["num_attention_heads"] % world or config["ffn_dim"] % world:
        raise ValueError("Checkpoint head and FFN dimensions must divide by the actual TP size")
    args = {k: v for k, v in config.items() if k in {
        "num_attention_heads", "attention_head_dim", "hidden_size", "num_layers", "num_refiner_layers",
        "ffn_dim", "in_channels", "audio_in_channels", "patch_size", "text_dim", "freq_dim",
        "time_embed_hidden_dim", "time_embed_dim", "rope_freq_dim", "rope_theta", "norm_eps",
        "qk_norm_eps", "final_norm_eps"}}
    with init_empty_weights():
        model = MiniMaxH3Transformer3DModel(**args)
    model.time_proj = nn.Identity()
    model.time_embedder = nn.Identity()
    model.time_embedder.register_buffer("dtype_reference", torch.zeros((), device=device, dtype=torch.float32))
    model.rope = type(model.rope)(config["rope_freq_dim"], config["rope_theta"]).to(device)
    modes = {}
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        mode = "replica"
        if name.endswith(("attn.to_q", "attn.to_k", "attn.to_v")):
            mode = "column"
        elif name.endswith("ff.net.0.proj"):
            mode = "swiglu"
        elif name.endswith(("attn.to_out.0", "ff.net.2")):
            mode = "row"
        modes[name] = mode
        promote = mode in {"row", "swiglu"} or name == "context_embedder"
        _replace(model, name, TPLinear(module, mode, promote, 64.0 if mode == "row" else 1.0))
    declared = json.loads((root / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    # Existing converted checkpoint index still names temporary shards. Inspect
    # the real files without modifying that shared production metadata.
    index = {}
    for file in sorted(root.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as state:
            for key in state.keys():
                if key in index:
                    raise ValueError(f"Duplicate checkpoint key {key}")
                index[key] = file.name
    if set(index) != set(declared):
        raise ValueError("Checkpoint header inventory differs from declared tensor keys")
    expected = dict(model.named_parameters())
    missing = set(expected) - set(index)
    unexpected = set(index) - set(expected) - {"adaln_t_table"}
    if missing or unexpected:
        raise ValueError(f"Checkpoint keys mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    from collections import defaultdict
    shards = defaultdict(list)
    for key, file in index.items():
        shards[file].append(key)
    for file, keys in sorted(shards.items()):
        with safe_open(root / file, framework="pt", device="cpu") as state:
            for key in keys:
                part = state.get_slice(key)
                shape = tuple(part.get_shape())
                if key == "adaln_t_table":
                    model.time_proj = CurveTime(state.get_tensor(key).to(device))
                    continue
                if shape != tuple(expected[key].shape):
                    raise ValueError(f"Shape mismatch {key}: {shape} != {tuple(expected[key].shape)}")
                module_name, _, param = key.rpartition(".")
                mode = modes.get(module_name, "replica") if param == "weight" else "replica"
                if mode == "column":
                    step = shape[0] // world
                    tensor = part[rank * step:(rank + 1) * step, :]
                elif mode == "swiglu":
                    half = shape[0] // 2
                    step = half // world
                    tensor = torch.cat((part[rank * step:(rank + 1) * step, :],
                                        part[half + rank * step:half + (rank + 1) * step, :]))
                elif mode == "row":
                    step = shape[1] // world
                    tensor = part[:, rank * step:(rank + 1) * step]
                else:
                    tensor = state.get_tensor(key)
                fp32 = ("adaln_proj" in key or key.startswith(("norm_out.", "proj_in.", "audio_proj_in.", "proj_out.", "audio_proj_out.")) or ".norm" in key or "final_norm" in key)
                dtype = torch.float32 if fp32 else torch.float16
                _replace(model, key, nn.Parameter(tensor.to(device=device, dtype=dtype), requires_grad=False))
    for module in model.modules():
        if isinstance(module, MiniMaxH3Attention):
            module.heads //= world
            module.inner_dim //= world
            module.set_processor(V100SDPA(backend=attention))
        if type(module).__name__ == "MiniMaxH3AdaLayerNormModulation":
            module.forward = types.MethodType(curve_modulation, module)
    model.norm_out.forward = types.MethodType(curve_final, model.norm_out)
    if any(p.is_meta for p in model.parameters()):
        raise ValueError("Unloaded meta parameters remain")
    model.eval()
    if os.getenv("H3_TRACE_FINITE") == "1":
        def trace(name):
            def hook(module, inputs, output):
                values = output if isinstance(output, tuple) else (output,)
                for value in values:
                    if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                        raise FloatingPointError(f"First non-finite output at {name} ({type(module).__name__})")
            return hook
        for name, module in model.named_modules():
            if name:
                module.register_forward_hook(trace(name))
    return model, {"load_seconds": time.perf_counter() - started,
                   "tp_size": world, "rank": rank, "checkpoint": str(root),
                   "local_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                   "attention": attention, "checkpoint_keys": len(index)}
