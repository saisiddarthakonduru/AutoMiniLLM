"""
Compression pipeline for AutoMiniLLM.

Creates real, persisted compressed models:
  - fp32 / fp16 / fp8 / int8 / int4 / nf4 quantization
  - magnitude, 2:4 structured, and channel pruning (sparse storage)
  - combined prune -> quantize (sparse quantized storage)

Each experiment produces measured metrics (size, RAM, sparsity, latency,
tokens/sec, perplexity, KL) computed on the actual resulting model, and
saves weights and metadata to disk.
"""
import copy
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .evaluator import evaluate_model
from .models import TinyGPT
from .pruning import (
    PRUNE_MODES,
    SparseLinear,
    apply_2_4_pruning,
    apply_channel_pruning,
    apply_magnitude_pruning,
    compute_sparsity,
    count_sparse_storage_bytes,
    model_parameter_count,
)
from .quantization import (
    NF4_QUANT_LEVELS,
    QUANT_MODES,
    QuantizedLinear,
    count_quantized_bytes,
    pack_int4,
    pack_uint4,
    quantize_fp8,
    quantize_group,
    quantize_model,
    quantize_nf4,
    quantize_per_channel,
    unpack_int4,
    unpack_uint4,
)
from .utils import (
    detect_device,
    get_file_size_mb,
    get_system_info,
    measure_model_ram_mb,
)

logger = logging.getLogger(__name__)

ALL_MODES = (
    "fp32",
    "fp16",
    "fp8",
    "int8",
    "int4",
    "nf4",
    "prune",
    "prune_2_4",
    "prune_channel",
    "combined",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class CompressionConfig:
    mode: str = "int8"             # fp32|fp16|fp8|int8|int4|nf4|prune|prune_2_4|prune_channel|combined
    prune_mode: str = "magnitude"  # magnitude|2_4|channel (for prune / combined)
    sparsity: float = 0.5          # for prune / combined
    group_size: int = 64           # int4/nf4 group size
    quant_mode: str = "int8"       # quant stage for combined (int8|fp8|int4|nf4)
    vocab_size: int = 50257
    gen_token_len: int = 24
    seq_len: int = 20
    tokenizer: Optional[object] = None
    system: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Combined sparse + quantized linear
# ---------------------------------------------------------------------------
class QuantizedSparseLinear(nn.Module):
    """
    Linear layer that is BOTH pruned (stores only non-zero values) and
    quantized (values stored as int8/int4/nf4/fp8).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mode: str = "int8",
        group_size: int = 64,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode.lower()
        self.group_size = group_size
        self.register_buffer("indices", torch.zeros(2, 0, dtype=torch.int32))

        if self.mode in ("int4", "nf4"):
            self.register_buffer("values_packed", torch.zeros(0, dtype=torch.uint8))
        elif self.mode == "fp8" and hasattr(torch, "float8_e4m3fn"):
            self.register_buffer("values", torch.zeros(0, dtype=torch.float8_e4m3fn))
        else:
            self.register_buffer("values", torch.zeros(0, dtype=torch.int8))

        self.register_buffer("scales", torch.zeros(0, dtype=torch.float32))
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(torch.float32))
        else:
            self.bias = None

    @property
    def sparsity(self) -> float:
        total = self.out_features * self.in_features
        if self.mode in ("int4", "nf4"):
            n = self.values_packed.numel() * 2
        else:
            n = self.values.numel()
        return 1.0 - (n / total) if total else 0.0

    def set_quantized_sparse(self, weight: torch.Tensor):
        """Take a pruned dense fp32 weight; store surviving entries in quantized buffers."""
        weight = weight.detach().float()
        nz_mask = weight != 0
        nz = nz_mask.nonzero(as_tuple=False)
        vals = weight[nz.t().unbind(0)] if nz.numel() > 0 else torch.zeros(0, dtype=torch.float32)
        self.indices = nz.to(torch.int32).t().contiguous() if nz.numel() > 0 else torch.zeros(2, 0, dtype=torch.int32)

        if vals.numel() == 0:
            if self.mode in ("int4", "nf4"):
                self.values_packed = torch.zeros(0, dtype=torch.uint8)
            elif self.mode == "fp8" and hasattr(torch, "float8_e4m3fn"):
                self.values = torch.zeros(0, dtype=torch.float8_e4m3fn)
            else:
                self.values = torch.zeros(0, dtype=torch.int8)
            self.scales = torch.zeros(0, dtype=torch.float32)
            return

        row_idx = self.indices[0]
        row_max = torch.zeros(self.out_features, dtype=torch.float32)
        row_max.index_reduce_(0, row_idx, vals.abs(), reduce="amax", include_self=True)
        row_max = row_max.clamp(min=1e-8)

        if self.mode == "int8":
            scales_2d = row_max[row_idx] / 127.0
            q = (vals / scales_2d).round().clamp(-127, 127).to(torch.int8)
            self.values = q
            self.scales = (row_max / 127.0).to(torch.float32)
        elif self.mode == "fp8":
            scales_2d = row_max[row_idx] / 448.0
            scaled = (vals / scales_2d).clamp(-448.0, 448.0)
            if hasattr(torch, "float8_e4m3fn"):
                self.values = scaled.to(torch.float8_e4m3fn)
            else:
                self.values = (scaled / (448.0 / 127.0)).round().clamp(-127, 127).to(torch.int8)
            self.scales = (row_max / 448.0).to(torch.float32)
        elif self.mode == "int4":
            scales_per_row = row_max / 7.0
            q = (vals / scales_per_row[row_idx]).round().clamp(-8, 7).to(torch.int8)
            self.values_packed = pack_int4(q.reshape(1, -1)).reshape(-1)
            self.scales = scales_per_row.to(torch.float32)
        elif self.mode == "nf4":
            scales_per_row = row_max
            norm_v = (vals / scales_per_row[row_idx]).clamp(-1.0, 1.0)
            levels = NF4_QUANT_LEVELS.to(vals.device).view(1, 16)
            best_idx = (norm_v.unsqueeze(1) - levels).abs().argmin(dim=-1).to(torch.uint8)
            self.values_packed = pack_uint4(best_idx.reshape(1, -1)).reshape(-1)
            self.scales = scales_per_row.to(torch.float32)

    @torch.no_grad()
    def dense_weight(self) -> torch.Tensor:
        w = torch.zeros(self.out_features, self.in_features, dtype=torch.float32)
        n = self.indices.shape[1]
        if n == 0:
            return w

        row_sel = self.indices[0].long()
        col_sel = self.indices[1].long()

        if self.mode in ("int8", "fp8"):
            vals = self.values.float() * self.scales[row_sel].float()
        elif self.mode == "int4":
            q = unpack_int4(self.values_packed, (1, self.values_packed.numel() * 2))
            vals = q.reshape(-1)[:n].float() * self.scales[row_sel].float()
        elif self.mode == "nf4":
            idx = unpack_uint4(self.values_packed, (1, self.values_packed.numel() * 2)).reshape(-1)[:n].long()
            levels = NF4_QUANT_LEVELS.to(idx.device)
            vals = levels[idx].float() * self.scales[row_sel].float()
        else:
            vals = self.values.float()

        w[row_sel, col_sel] = vals
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dense_weight().to(x.dtype)
        bias = getattr(self, "bias", None)
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(x.dtype)
        return torch.nn.functional.linear(x, w, bias)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _module_bytes(model: nn.Module) -> int:
    """Actual bytes of parameters + buffers (real footprint)."""
    total = 0
    seen = set()
    for p in list(model.parameters()) + list(model.buffers()):
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel() * p.element_size()
    return total


def _get_model_vocab_size(model: nn.Module) -> Optional[int]:
    """Return the model's ACTUAL vocabulary size."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        vs = getattr(cfg, "vocab_size", None)
        if vs:
            return int(vs)
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            return m.num_embeddings
    return None


def _save_compressed_model(model: nn.Module, save_dir: str, label: str, metadata: Optional[Dict] = None) -> str:
    """Persist the compressed model's state and companion metadata."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{label}.pt")
    torch.save(model.state_dict(), path)

    if metadata is not None:
        meta_path = os.path.join(save_dir, f"{label}_meta.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning("Could not save metadata: %s", e)

    return path


def _apply_pruning(model: nn.Module, prune_mode: str, sparsity: float) -> Tuple[nn.Module, float]:
    """Dispatch to chosen pruning technique."""
    if prune_mode == "2_4":
        return apply_2_4_pruning(model)
    elif prune_mode == "channel":
        return apply_channel_pruning(model, sparsity)
    else:
        return apply_magnitude_pruning(model, sparsity)


def _apply_combined(model: nn.Module, config: CompressionConfig) -> nn.Module:
    """prune -> quantize the surviving values into QuantizedSparseLinear."""
    pruned, _ = _apply_pruning(model, config.prune_mode, config.sparsity)

    def convert(module: nn.Module) -> nn.Module:
        for name, child in list(module.named_children()):
            if "lm_head" in name or name == "lm_head":
                continue
            if isinstance(child, SparseLinear):
                w = child.dense_weight()
                bias = getattr(child, "bias", None)
                qs = QuantizedSparseLinear(
                    child.in_features,
                    child.out_features,
                    mode=config.quant_mode,
                    group_size=config.group_size,
                    bias=bias,
                )
                qs.set_quantized_sparse(w)
                setattr(module, name, qs)
            else:
                convert(child)
        return module

    return convert(pruned)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compress_model(
    model: nn.Module,
    config: CompressionConfig,
    save_dir: str = "models",
    device: Optional[str] = None,
) -> Dict:
    device = device or detect_device()
    t_start = time.perf_counter()

    model = model.to(device).eval()
    mode = config.mode.lower()
    label = _build_label(mode, config.sparsity, config.group_size, config.quant_mode, config.prune_mode)

    BIG_MODEL_PARAMS = 350_000_000
    is_big = model_parameter_count(model) > BIG_MODEL_PARAMS

    if mode == "fp32":
        compressed = model
        base_model = model
        baseline_dtype = "fp32"
    else:
        if is_big:
            model = model.to(torch.float16)
            baseline_dtype = "fp16"
        else:
            baseline_dtype = "fp32"
        base_model = copy.deepcopy(model)
        compressed = copy.deepcopy(model)

        if mode in ("fp16", "fp8", "int8", "int4", "nf4"):
            compressed = quantize_model(compressed, mode, group_size=config.group_size)
        elif mode == "prune":
            compressed, _ = apply_magnitude_pruning(compressed, config.sparsity)
        elif mode == "prune_2_4":
            compressed, _ = apply_2_4_pruning(compressed)
        elif mode == "prune_channel":
            compressed, _ = apply_channel_pruning(compressed, config.sparsity)
        elif mode == "combined":
            compressed = _apply_combined(compressed, config)
        else:
            raise ValueError(f"Unknown mode {mode}")

    # Collect layer architecture metadata for reconstruction
    meta = {
        "mode": mode,
        "prune_mode": config.prune_mode,
        "quant_mode": config.quant_mode if mode == "combined" else mode,
        "sparsity": config.sparsity,
        "group_size": config.group_size,
        "linear_layers": {
            name: type(m).__name__
            for name, m in compressed.named_modules()
            if isinstance(m, (QuantizedLinear, SparseLinear, QuantizedSparseLinear))
        },
    }

    # Persist compressed model + metadata
    save_path = _save_compressed_model(compressed, os.path.join(save_dir, "models"), label, metadata=meta)
    file_size_mb = get_file_size_mb(save_path)
    base_path = _save_compressed_model(base_model, os.path.join(save_dir, "base"), "fp32")
    base_file_mb = get_file_size_mb(base_path)

    # Runtime measurements on real model
    actual_vocab = _get_model_vocab_size(compressed) or config.vocab_size
    ev_results = evaluate_model(
        compressed,
        baseline_model=base_model,
        vocab_size=actual_vocab,
        device=device,
        gen_token_len=config.gen_token_len,
        seq_len=config.seq_len,
        tokenizer=config.tokenizer,
    )
    ev_results["vocab_size_used"] = actual_vocab

    compressed_params = model_parameter_count(compressed)
    base_params = model_parameter_count(base_model)
    achieved_sparsity = round(compute_sparsity(compressed) * 100.0, 2)

    ram_mb = round(_module_bytes(compressed) / (1024 * 1024), 4)
    base_ram_mb = round(_module_bytes(base_model) / (1024 * 1024), 4)
    compress_seconds = round(time.perf_counter() - t_start, 3)

    file_ratio = round(base_file_mb / file_size_mb, 3) if base_file_mb and file_size_mb else None
    ram_ratio = round(base_ram_mb / ram_mb, 3) if ram_mb else None

    result = {
        "id": f"{time.strftime('%Y%m%d_%H%M%S')}_{label}",
        "label": label,
        "mode": mode,
        "prune_mode": config.prune_mode,
        "quant_mode": config.quant_mode if mode == "combined" else mode,
        "sparsity_target": config.sparsity,
        "group_size": config.group_size,
        "baseline_dtype": baseline_dtype,
        "achieved_sparsity": achieved_sparsity,
        "parameter_count": int(compressed_params),
        "base_parameter_count": int(base_params),
        "ram_mb": ram_mb,
        "base_ram_mb": base_ram_mb,
        "ram_compression_ratio": ram_ratio,
        "file_size_mb": file_size_mb,
        "base_file_size_mb": base_file_mb,
        "file_compression_ratio": file_ratio,
        "perplexity": ev_results["perplexity"],
        "loss": ev_results["loss"],
        "kl_vs_baseline": ev_results.get("kl_vs_baseline"),
        "tokens_per_sec": ev_results["tokens_per_sec"],
        "ms_per_token": ev_results["ms_per_token"],
        "ttft_ms": ev_results.get("ttft_ms"),
        "prefill_latency_ms": ev_results["prefill_latency_ms"],
        "prefill_p95_ms": ev_results["prefill_p95_ms"],
        "generated_token_count": ev_results["generated_token_count"],
        "model_ram_mb": ev_results.get("model_ram_mb"),
        "compress_seconds": compress_seconds,
        "saved_path": save_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": get_system_info(),
    }
    return result


def _build_label(mode: str, sparsity: float, group_size: int, quant_mode: str, prune_mode: str) -> str:
    if mode == "prune":
        return f"prune-{int(round(sparsity*100))}%"
    if mode == "prune_2_4":
        return "prune-2-4"
    if mode == "prune_channel":
        return f"prune-channel-{int(round(sparsity*100))}%"
    if mode == "combined":
        p = "2-4" if prune_mode == "2_4" else f"{int(round(sparsity*100))}%"
        return f"comb-{p}+{quant_mode}"
    return mode
