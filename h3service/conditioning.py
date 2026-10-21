"""Diffusers modular presentation/TE/reference encoding in its own process."""
from __future__ import annotations

import argparse
import gc
import faulthandler
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from diffusers.modular_pipelines.modular_pipeline import PipelineState, SequentialPipelineBlocks
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
    MiniMaxH3AutoBeforeEncodeStep, MiniMaxH3AutoTextEncoderStep,
)
from safetensors.torch import save_file
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

from .components import MODEL_ROOT, video_vae, audio_vae


def bounded_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """Full-context SDPA in query chunks, preserving global causal positions."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    if query.shape[-2] != key.shape[-2] or dropout:
        raise ValueError("Conditioner attention expects a full, inference-only prefill")
    output = torch.empty_like(query.transpose(1, 2))
    for start in range(0, query.shape[-2], 128):
        q = query[:, :, start:start + 128]
        mask = attention_mask[..., start:start + q.shape[-2], :] if attention_mask is not None else None
        if mask is None and getattr(module, "is_causal", True):
            positions = torch.arange(start, start + q.shape[-2], device=q.device)
            mask = torch.arange(key.shape[-2], device=q.device)[None, :] <= positions[:, None]
        part, _ = sdpa_attention_forward(module, q, key, value, mask, dropout=0.0, scaling=scaling,
                                         is_causal=False)
        output[:, start:start + q.shape[-2]] = part
    return output, None


class PatchifyLinear(torch.nn.Module):
    """Exactly one convolution window per Qwen vision patch, computed as GEMM."""
    def __init__(self, conv):
        super().__init__()
        if conv.kernel_size != conv.stride or any(conv.padding) or conv.groups != 1 or any(d != 1 for d in conv.dilation):
            raise ValueError("Qwen patch convolution does not have the audited one-window layout")
        if conv.weight.device.type != "cuda":
            raise ValueError("Vision patch weights must be GPU resident")
        self.weight, self.bias = conv.weight, conv.bias

    def forward(self, x):
        if tuple(x.shape[2:]) != tuple(self.weight.shape[2:]):
            raise ValueError("Expected exactly one vision patch per batch entry")
        return torch.nn.functional.linear(x.flatten(1), self.weight.flatten(1), self.bias)[:, :, None, None, None]


def encode(request, output):
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    started = time.perf_counter()
    inp, params = request["input"], request["parameters"]
    blocks = SequentialPipelineBlocks.from_blocks_dict({
        "before_encode": MiniMaxH3AutoBeforeEncodeStep(), "text_encoder": MiniMaxH3AutoTextEncoderStep()})
    pipe = MiniMaxH3ModularPipeline(blocks=blocks)
    config = AutoConfig.from_pretrained(MODEL_ROOT / "text_encoder", local_files_only=True)
    # hidden_states[50] must be an intermediate, pre-final-norm hidden state.
    # Loading 51 layers preserves that exact native Diffusers contract.
    config.text_config.num_hidden_layers = 51
    count = torch.cuda.device_count()
    memory = {i: "13GiB" if count == 3 else "10GiB" for i in range(count)}
    memory["cpu"] = "12GiB"
    loaded = time.perf_counter()
    te = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ROOT / "text_encoder", config=config, dtype=torch.float16,
        local_files_only=True, device_map="balanced", max_memory=memory,
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    # Native POWER9 Conv3d without cuDNN is pathological for this huge-batch,
    # single-window patchify operation. GEMM keeps every original input pixel.
    te.model.visual.patch_embed.proj = PatchifyLinear(te.model.visual.patch_embed.proj)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS.register("h3_bounded_sdpa", bounded_sdpa)
    te.model.language_model.config._attn_implementation = "h3_bounded_sdpa"
    te.model.visual.config._attn_implementation = "h3_bounded_sdpa"
    print(json.dumps({"event": "te_loaded", "seconds": time.perf_counter() - loaded}), flush=True)
    pipe.update_components(text_encoder=te,
                           tokenizer=AutoTokenizer.from_pretrained(MODEL_ROOT / "tokenizer", local_files_only=True),
                           processor=AutoProcessor.from_pretrained(MODEL_ROOT / "processor", local_files_only=True))
    refs = inp.get("_references", {})
    references = None
    if any(refs.values()):
        from diffusers.modular_pipelines.minimax_h3.references import (
            MiniMaxH3ImageReference, MiniMaxH3VideoReference, MiniMaxH3AudioReference)
        references = []
        for kind, cls in (("images", MiniMaxH3ImageReference), ("videos", MiniMaxH3VideoReference),
                          ("audios", MiniMaxH3AudioReference)):
            references.extend(cls.from_file(path) for path in refs.get(kind, []))
    report = {"te_load_seconds": time.perf_counter() - loaded,
              "te_device_map": {k: str(v) for k, v in te.hf_device_map.items()},
              "te_layers": 51, "engine": "diffusers"}
    encoding = time.perf_counter()
    faulthandler.dump_traceback_later(120, repeat=True)
    state = pipe(prompt=inp["prompt"], references=references, width=params["width"], height=params["height"],
                 num_frames=params["frames"], output=None)
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps({"event": "text_encoded", "seconds": time.perf_counter() - encoding}), flush=True)
    report["text_encode_seconds"] = time.perf_counter() - encoding
    tensors = {"prompt_embeds": state.get("prompt_embeds").detach().cpu().contiguous(),
               "text_token_tags": state.get("text_token_tags").detach().cpu().contiguous()}
    normalized = state.get("normalized_references")
    pipe.update_components(text_encoder=None)
    del te
    gc.collect()
    for i in range(count):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    if references:
        # A separate native reference block encodes after TE weights are released.
        from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VAReferenceEncoderStep
        refpipe = MiniMaxH3ModularPipeline(blocks=MiniMaxH3Ref2VAReferenceEncoderStep())
        vae = video_vae("cuda:0")
        refpipe.update_components(vae=vae, audio_vae=audio_vae("cpu"))
        refstate = refpipe(normalized_references=normalized, output=None)
        for name in ("condition_latents", "audio_condition_latents"):
            value = refstate.get(name)
            if value:
                for i, tensor in enumerate(value):
                    tensors[f"{name}.{i}"] = tensor.detach().cpu().contiguous()
        Path(output).with_suffix(".references.json").write_text(json.dumps(
            [{"kind": r.kind, "has_audio": r.has_audio} for r in normalized]))
    if not all(bool(torch.isfinite(v).all()) for v in tensors.values()):
        raise FloatingPointError("Non-finite conditioning")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(target), metadata={"engine": "diffusers", "te_layers": "51"})
    report["total_seconds"] = time.perf_counter() - started
    report["shapes"] = {k: list(v.shape) for k, v in tensors.items()}
    target.with_suffix(".report.json").write_text(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    from .memory import model_staging_lock
    with model_staging_lock():
        encode(json.loads(Path(args.request).read_text()), args.output)


if __name__ == "__main__":
    main()
