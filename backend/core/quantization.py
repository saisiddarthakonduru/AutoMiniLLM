"""
Weight-only quantization engine for AutoMiniLLM.

Implements real quantization (not simulation):
  - FP32 / FP16  : native dtype conversion (real lower-precision storage)
  - FP8          : native torch.float8_e4m3fn with per-channel scale
  - INT8         : per-channel symmetric (scale = max abs / 127)
  - INT4         : group-wise symmetric (group=64) with packed 4-bit uint8 storage
  - NF4          : group-wise NormalFloat4 (QLoRA quantile mapping) packed into 4-bit storage

Quantized weights are stored as compact int/byte/float8 tensors and dequantized per
layer during forward pass (standard "weight-only" serving on consumer hardware).
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from transformers.pytorch_utils import Conv1D as _HFConv1D
except Exception:
    try:
        from transformers.modeling_utils import Conv1D as _HFConv1D
    except Exception:
        _HFConv1D = None

QUANT_MODES = ("fp32", "fp16", "fp8", "int8", "int4", "nf4")

# NormalFloat4 (NF4) quantile levels (from QLoRA paper / bitsandbytes)
NF4_QUANT_LEVELS = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169300557,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.4407079815864563,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
)


# ---------------------------------------------------------------------------
# Low-level quantization helpers
# ---------------------------------------------------------------------------
def _quant_scale_1d(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-channel symmetric scale for a 2D weight -> shape [out, 1]."""
    qmax = float(2 ** (bits - 1) - 1)
    amax = tensor.abs().amax(dim=1, keepdim=True)
    amax = torch.clamp(amax, min=1e-8)
    return (amax / qmax).float()


def quantize_per_channel(weight: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel quantization. Returns (q_w int8, scales fp32)."""
    qmax = float(2 ** (bits - 1) - 1)
    scales = _quant_scale_1d(weight, bits).squeeze(1)  # [out]
    q_w = (weight / scales.unsqueeze(1)).round().clamp(-qmax, qmax - 1).to(torch.int8)
    return q_w, scales


def quantize_fp8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP8 (E4M3) quantization with per-channel scaling (max representable = 448.0)."""
    amax = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scales = (amax / 448.0).float().squeeze(1)  # [out]
    scaled = weight / scales.unsqueeze(1)
    if hasattr(torch, "float8_e4m3fn"):
        q_w = scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    else:
        # Fallback to int8 if device lacks float8 support
        q_w = (scaled / (448.0 / 127.0)).round().clamp(-127, 127).to(torch.int8)
    return q_w, scales


def quantize_group(
    weight: torch.Tensor, bits: int, group_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Group-wise symmetric quantization. Returns the PADDED quantized grid
    [out, num_groups*group_size] and scales [out, num_groups].
    """
    qmax = float(2 ** (bits - 1) - 1)
    out_rows, in_cols = weight.shape
    num_groups = math.ceil(in_cols / group_size)
    pad_cols = num_groups * group_size - in_cols
    if pad_cols > 0:
        padded = torch.nn.functional.pad(weight, (0, pad_cols, 0, 0))
    else:
        padded = weight
    padded = padded.view(out_rows, num_groups, group_size)

    amax = padded.abs().amax(dim=2, keepdim=True)
    amax = torch.clamp(amax, min=1e-8)
    qmax_t = torch.tensor(qmax, dtype=padded.dtype, device=weight.device)
    scales = (amax / qmax_t).float()  # [out, groups, 1]
    q = (padded / scales).round().clamp(-qmax, qmax - 1)
    q = q.reshape(out_rows, num_groups * group_size).contiguous().to(torch.int8)
    return q, scales.squeeze(-1)


def quantize_nf4(
    weight: torch.Tensor, group_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NormalFloat4 (NF4) group-wise quantization.
    Maps weights to the nearest of 16 optimal normal-distribution quantiles.
    Returns (packed_uint8 [out, ceil(padded_in/2)], scales [out, num_groups]).
    """
    out_rows, in_cols = weight.shape
    num_groups = math.ceil(in_cols / group_size)
    pad_cols = num_groups * group_size - in_cols
    if pad_cols > 0:
        padded = torch.nn.functional.pad(weight, (0, pad_cols, 0, 0))
    else:
        padded = weight
    padded = padded.view(out_rows, num_groups, group_size)

    # Per-group absolute max scale
    amax = padded.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
    norm_w = (padded / amax).float()  # in [-1.0, 1.0]

    # Map each value to closest NF4 quantile index [0..15]
    levels = NF4_QUANT_LEVELS.to(weight.device).view(1, 1, 1, 16)
    dist = (norm_w.unsqueeze(-1) - levels).abs()
    indices = dist.argmin(dim=-1).to(torch.uint8)  # [out, num_groups, group_size]

    grid = indices.reshape(out_rows, num_groups * group_size)
    packed = pack_uint4(grid)
    return packed, amax.squeeze(-1).float()


def pack_int4(q_w: torch.Tensor) -> torch.Tensor:
    """Pack [-8..7] signed int4 weights into uint8, row-wise."""
    out, in_cols = q_w.shape
    w = q_w.to(torch.int8).to(torch.uint8) & 0x0F
    if in_cols % 2 == 1:
        w = torch.nn.functional.pad(w, (0, 1))
    w = w.view(out, -1, 2)
    low = w[..., 0].to(torch.uint8)
    high = (w[..., 1].to(torch.uint8) << 4)
    return (low | high).to(torch.uint8)


def unpack_int4(packed: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Unpack uint8 nibbles back to signed int4 values (low nibble first)."""
    lo = (packed & 0x0F).to(torch.uint8)
    hi = ((packed >> 4) & 0x0F).to(torch.uint8)
    n = shape[0] * shape[1]
    v = torch.stack([lo, hi], dim=-1).reshape(-1)[:n]
    v = v.to(torch.int16)
    v = torch.where(v >= 8, v - 16, v).to(torch.int8)
    return v.reshape(shape)


def pack_uint4(idx: torch.Tensor) -> torch.Tensor:
    """Pack [0..15] unsigned 4-bit indices into uint8, row-wise."""
    out, in_cols = idx.shape
    w = idx.to(torch.uint8) & 0x0F
    if in_cols % 2 == 1:
        w = torch.nn.functional.pad(w, (0, 1))
    w = w.view(out, -1, 2)
    low = w[..., 0].to(torch.uint8)
    high = (w[..., 1].to(torch.uint8) << 4)
    return (low | high).to(torch.uint8)


def unpack_uint4(packed: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Unpack uint8 nibbles back to unsigned [0..15] values."""
    lo = (packed & 0x0F).to(torch.uint8)
    hi = ((packed >> 4) & 0x0F).to(torch.uint8)
    n = shape[0] * shape[1]
    v = torch.stack([lo, hi], dim=-1).reshape(-1)[:n]
    return v.reshape(shape)


# ---------------------------------------------------------------------------
# Quantized Linear layer
# ---------------------------------------------------------------------------
class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear storing weights at reduced precision."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mode: str = "int8",
        group_size: Optional[int] = None,
        orig_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode.lower()
        self.group_size = group_size if group_size else 64
        padded_in = math.ceil(in_features / self.group_size) * self.group_size
        self.packed_shape = (out_features, padded_in)
        self.packed_cols = padded_in

        if self.mode in ("fp32", "fp16"):
            dtype = torch.float16 if self.mode == "fp16" else torch.float32
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            if orig_bias is not None:
                self.bias = nn.Parameter(orig_bias.detach().to(dtype))
            else:
                self.register_parameter("bias", None)
        elif self.mode == "fp8":
            if hasattr(torch, "float8_e4m3fn"):
                self.register_buffer(
                    "weight", torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn)
                )
            else:
                self.register_buffer(
                    "weight", torch.zeros(out_features, in_features, dtype=torch.int8)
                )
            self.register_buffer("scale", torch.zeros(out_features, dtype=torch.float32))
            self._init_bias(orig_bias)
        elif self.mode == "int8":
            self.register_buffer(
                "weight", torch.zeros(out_features, in_features, dtype=torch.int8)
            )
            self.register_buffer("scale", torch.zeros(out_features, dtype=torch.float32))
            self._init_bias(orig_bias)
        elif self.mode in ("int4", "nf4"):
            self.register_buffer(
                "packed",
                torch.zeros(out_features, math.ceil(self.packed_cols / 2), dtype=torch.uint8),
            )
            self.register_buffer(
                "scales",
                torch.zeros(
                    out_features,
                    math.ceil(in_features / self.group_size),
                    dtype=torch.float32,
                ),
            )
            self._init_bias(orig_bias)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _init_bias(self, orig_bias: Optional[torch.Tensor]):
        if orig_bias is not None:
            self.register_buffer("bias", orig_bias.detach().to(torch.float32))
        else:
            self.bias = None

    def set_quant_weight(self, q_w: torch.Tensor, scales: torch.Tensor):
        if self.mode in ("int8", "fp8"):
            self.weight.copy_(q_w)
            self.scale.copy_(scales)
        elif self.mode == "int4":
            self.packed.copy_(pack_int4(q_w))
            self.scales.copy_(scales)
        elif self.mode == "nf4":
            # q_w is already packed uint4 from quantize_nf4
            self.packed.copy_(q_w)
            self.scales.copy_(scales)

    def _dequantized(self) -> torch.Tensor:
        """Reconstruct a float weight for forward pass (weight-only serving)."""
        if self.mode == "int8":
            return self.weight.to(torch.float32) * self.scale.unsqueeze(1)
        if self.mode == "fp8":
            return self.weight.to(torch.float32) * self.scale.unsqueeze(1)
        if self.mode == "int4":
            rows, cols = self.packed_shape
            gs = self.group_size
            num_groups = cols // gs
            w = unpack_int4(self.packed, self.packed_shape).to(torch.float32)
            w = w.view(rows, num_groups, gs)
            w = w * self.scales.view(rows, num_groups, 1)
            w = w.reshape(rows, cols)[:, : self.in_features]
            return w.contiguous()
        if self.mode == "nf4":
            rows, cols = self.packed_shape
            gs = self.group_size
            num_groups = cols // gs
            idx = unpack_uint4(self.packed, self.packed_shape).long()
            levels = NF4_QUANT_LEVELS.to(idx.device)
            w = levels[idx].view(rows, num_groups, gs)
            w = w * self.scales.view(rows, num_groups, 1)
            w = w.reshape(rows, cols)[:, : self.in_features]
            return w.contiguous()
        return self.weight.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantized()
        if x.dtype != w.dtype:
            w = w.to(x.dtype)
        bias = getattr(self, "bias", None)
        if bias is not None and bias.dtype != w.dtype:
            bias = bias.to(w.dtype)
        return torch.nn.functional.linear(x, w, bias)

    def format_dtype(self) -> str:
        return self.mode


# ---------------------------------------------------------------------------
# Model-level quantizer
# ---------------------------------------------------------------------------
def quantize_model(
    model: nn.Module,
    mode: str = "int8",
    group_size: int = 64,
) -> nn.Module:
    """Convert all Linear layers of a model to QuantizedLinear in tree structure."""
    mode = mode.lower()
    if mode not in QUANT_MODES:
        raise ValueError(f"Unsupported mode {mode}. Choose from {QUANT_MODES}.")

    def convert(module: nn.Module) -> nn.Module:
        for name, child in list(module.named_children()):
            is_linear = isinstance(child, nn.Linear)
            is_conv1d = _HFConv1D is not None and isinstance(child, _HFConv1D)
            if is_linear or is_conv1d:
                if is_conv1d:
                    orig_w = child.weight.detach().float().t()
                    orig_b = child.bias.detach() if child.bias is not None else None
                    in_f, out_f = child.nx, child.nf
                else:
                    orig_w = child.weight.detach().float()
                    orig_b = child.bias.detach() if child.bias is not None else None
                    in_f, out_f = child.in_features, child.out_features

                q_layer = QuantizedLinear(
                    in_f,
                    out_f,
                    mode=mode,
                    group_size=group_size if mode in ("int4", "nf4") else None,
                    orig_bias=orig_b,
                )

                if mode == "int8":
                    q_w, scales = quantize_per_channel(orig_w, 8)
                    q_layer.set_quant_weight(q_w, scales)
                elif mode == "fp8":
                    q_w, scales = quantize_fp8(orig_w)
                    q_layer.set_quant_weight(q_w, scales)
                elif mode == "int4":
                    q_w, scales = quantize_group(orig_w, 4, q_layer.group_size)
                    q_layer.set_quant_weight(q_w, scales)
                elif mode == "nf4":
                    packed, scales = quantize_nf4(orig_w, q_layer.group_size)
                    q_layer.set_quant_weight(packed, scales)
                else:  # fp32, fp16
                    with torch.no_grad():
                        q_layer.weight.copy_(orig_w.to(q_layer.weight.dtype))
                    if q_layer.bias is not None:
                        with torch.no_grad():
                            q_layer.bias.copy_(orig_b.to(q_layer.bias.dtype))
                setattr(module, name, q_layer)
            else:
                convert(module=child)
        return module

    model.eval()
    return convert(model)


def count_quantized_bytes(model: nn.Module) -> Tuple[int, int]:
    """Return (quant_weight_bytes, orig_weight_bytes)."""
    quant_bytes = 0
    orig_bytes = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            q = module
            total = q.in_features * q.out_features
            if q.mode in ("int8", "fp8"):
                quant_bytes += total + q.out_features * 4
            elif q.mode in ("int4", "nf4"):
                quant_bytes += total // 2 + math.ceil(total / q.group_size) * q.out_features * 4
            elif q.mode == "fp16":
                quant_bytes += total * 2
            else:
                quant_bytes += total * 4
            orig_bytes += total * 4
            if q.bias is not None:
                quant_bytes += q.out_features * 4
                orig_bytes += q.out_features * 4
    return quant_bytes, orig_bytes
