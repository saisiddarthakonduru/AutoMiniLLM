"""
Magnitude-based, 2:4 structured, and channel pruning with sparse storage for AutoMiniLLM.

Implements:
  - Global magnitude pruning (smallest |weights| -> 0) over all Linear layers.
  - Semi-structured 2:4 pruning (2 zeroes in every 4-element block for hardware tensor cores).
  - Structured channel/neuron pruning (prunes least-salient output rows by L1 norm).
  - SparseLinear wrapper storing non-zero weights sparsely (COO indices + values).
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

PRUNE_MODES = ("magnitude", "2_4", "channel")


class SparseLinear(nn.Module):
    """Linear layer storing only non-zero weights (COO-style indices + values)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("indices", torch.zeros(2, 0, dtype=torch.int32))
        self.register_buffer("values", torch.zeros(0, dtype=torch.float16))
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(torch.float32))
        else:
            self.bias = None

    @property
    def sparsity(self) -> float:
        total = self.out_features * self.in_features
        return 1.0 - (self.values.numel() / total) if total > 0 else 0.0

    def set_pruned_weight(self, weight: torch.Tensor):
        """Store only non-zero entries of a (pruned) dense weight tensor."""
        weight = weight.detach().float()
        nz = (weight != 0).nonzero(as_tuple=False).t().contiguous()
        self.indices = nz.to(torch.int32)
        self.values = weight[weight != 0].to(torch.float16).contiguous()

    def dense_weight(self) -> torch.Tensor:
        w = torch.zeros(self.out_features, self.in_features, dtype=torch.float32)
        if self.values.numel() > 0:
            w[self.indices[0], self.indices[1]] = self.values.float()
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dense_weight().to(x.dtype)
        bias = getattr(self, "bias", None)
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(x.dtype)
        return torch.nn.functional.linear(x, w, bias)


def _find_parent(root: nn.Module, dotted_name: str) -> nn.Module:
    """Return the parent module of a dotted attribute path like 'layers.0.fc1'."""
    parts = dotted_name.split(".")
    node = root
    for part in parts[:-1]:
        node = getattr(node, part)
    return node


def model_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_sparsity(model: nn.Module) -> float:
    """
    Fraction of zeroed weights over PRUNABLE weight tensors only (Linear layers
    and sparse wrappers). Embeddings / LayerNorm / biases are excluded so the
    number reflects the actual pruning sparsity of the weight matrices.
    """
    total = 0
    zeros = 0
    for m in model.modules():
        if isinstance(m, nn.Linear) and not hasattr(m, "indices"):
            total += m.weight.numel()
            try:
                zeros += int((m.weight == 0).sum().item())
            except RuntimeError:
                pass
        elif hasattr(m, "indices") and hasattr(m, "out_features") and hasattr(m, "in_features"):
            dense_total = m.out_features * m.in_features
            n = m.indices.shape[1] if m.indices.numel() else 0
            total += dense_total
            zeros += dense_total - n
    return zeros / total if total else 0.0


def count_sparse_storage_bytes(model: nn.Module) -> Tuple[int, int]:
    """Return (sparse_bytes, orig_dense_bytes) for pruned Linear layers."""
    sparse_bytes = 0
    orig_bytes = 0
    for module in model.modules():
        if isinstance(module, SparseLinear):
            nvals = module.values.numel()
            sparse_bytes += nvals * 2 + nvals * 2 * 4  # values fp16 + 2x int32 indices
            if module.bias is not None:
                sparse_bytes += module.out_features * 4
            orig_bytes += module.out_features * module.in_features * 4
            if module.bias is not None:
                orig_bytes += module.out_features * 4
    return sparse_bytes, orig_bytes


def _global_magnitude_threshold(models, sparsity: float, sample_budget: int = 8_000_000) -> Optional[float]:
    """
    Compute global magnitude threshold for pruning.
    For large models, sample up to `sample_budget` weights across layers.
    """
    total = sum(int(m.weight.numel()) for m in models)
    fraction = min(max(sparsity, 0.0), 0.999)
    if total <= 0 or fraction <= 0:
        return None

    if total <= sample_budget:
        all_w = torch.cat([m.weight.detach().float().reshape(-1) for m in models])
        k = int(round(all_w.numel() * fraction))
        if k >= all_w.numel():
            return float(all_w.abs().max().item()) + 1.0
        return float(torch.topk(all_w.abs(), k, largest=False).values[-1].item())

    # Bounded sample path (big models)
    pieces = []
    for m in models:
        n = int(m.weight.numel())
        wanted = int(sample_budget * n / total)
        if wanted <= 0:
            continue
        wabs = m.weight.detach().abs().reshape(-1)
        if n == wanted:
            pieces.append(wabs.to(torch.float32))
        else:
            # Fixed bug: use wabs.device instead of undefined w.device
            idx = torch.randperm(n, device=wabs.device)[:wanted]
            pieces.append(wabs[idx].to(torch.float32))
    sample = torch.cat(pieces)
    k = int(round(sample.numel() * fraction))
    if k >= sample.numel():
        return float(sample.max().item()) + 1.0
    return float(torch.topk(sample, k, largest=False).values[-1].item())


def _extract_linear_layers(model: nn.Module):
    """Collect (dotted_name, module) for nn.Linear or HF Conv1D, protecting lm_head."""
    layers = []
    for name, module in model.named_modules():
        # Protect language model head from pruning so vocabulary projections remain intact
        if "lm_head" in name or name == "lm_head":
            continue
        if isinstance(module, nn.Linear) and not isinstance(module, SparseLinear):
            layers.append((name, module))
        elif _HFConv1D is not None and isinstance(module, _HFConv1D):
            layers.append((name, module))
    return layers


def apply_magnitude_pruning(
    model: nn.Module,
    sparsity: float,
) -> Tuple[nn.Module, float]:
    """
    Globally zero out the smallest-magnitude weights, then convert each Linear
    layer into a SparseLinear storing only survivors. Returns (model, achieved).
    """
    to_prune = _extract_linear_layers(model)
    if not to_prune:
        return model, 0.0

    threshold = _global_magnitude_threshold([m for _, m in to_prune], sparsity)
    if threshold is None:
        return model, 0.0

    total_kept = 0
    total_linear = 0
    for name, module in to_prune:
        if _HFConv1D is not None and isinstance(module, _HFConv1D):
            w = module.weight.detach().float().t()  # [out, in]
            bias = module.bias
            in_f, out_f = module.nx, module.nf
        else:
            w = module.weight.detach().float()
            bias = module.bias
            in_f, out_f = module.in_features, module.out_features

        keep = w.abs() >= threshold
        pruned_w = torch.where(keep, w, torch.zeros_like(w))
        pruned_w = torch.where(pruned_w.abs() < 1e-12, torch.zeros_like(pruned_w), pruned_w)

        sparse = SparseLinear(in_f, out_f, bias=bias)
        sparse.set_pruned_weight(pruned_w)

        parent = _find_parent(model, name)
        setattr(parent, name.split(".")[-1], sparse)

        total_kept += int((pruned_w != 0).sum().item())
        total_linear += pruned_w.numel()

    achieved = 1.0 - (total_kept / total_linear) if total_linear else 0.0
    return model, achieved


def apply_2_4_pruning(model: nn.Module) -> Tuple[nn.Module, float]:
    """
    Semi-structured 2:4 sparsity (NVIDIA Ampere/Ada/Hopper compatible pattern):
    In every 4 contiguous elements along the inner dimension, zero out the 2 smallest.
    Guarantees exact 50% sparsity.
    """
    to_prune = _extract_linear_layers(model)
    if not to_prune:
        return model, 0.0

    total_kept = 0
    total_linear = 0
    for name, module in to_prune:
        if _HFConv1D is not None and isinstance(module, _HFConv1D):
            w = module.weight.detach().float().t()  # [out, in]
            bias = module.bias
            in_f, out_f = module.nx, module.nf
        else:
            w = module.weight.detach().float()
            bias = module.bias
            in_f, out_f = module.in_features, module.out_features

        # Pad columns to multiple of 4 if needed
        pad_cols = (4 - (in_f % 4)) % 4
        if pad_cols > 0:
            w_padded = torch.nn.functional.pad(w, (0, pad_cols, 0, 0))
        else:
            w_padded = w

        out_rows, cols_padded = w_padded.shape
        w_blocks = w_padded.reshape(out_rows, cols_padded // 4, 4)

        # In each block of 4, find indices of the 2 largest absolute values
        _, top2_idx = torch.topk(w_blocks.abs(), k=2, dim=-1, largest=True)
        mask = torch.zeros_like(w_blocks, dtype=torch.bool)
        mask.scatter_(dim=-1, index=top2_idx, value=True)

        pruned_blocks = torch.where(mask, w_blocks, torch.zeros_like(w_blocks))
        pruned_w = pruned_blocks.reshape(out_rows, cols_padded)[:, :in_f].contiguous()

        sparse = SparseLinear(in_f, out_f, bias=bias)
        sparse.set_pruned_weight(pruned_w)

        parent = _find_parent(model, name)
        setattr(parent, name.split(".")[-1], sparse)

        total_kept += int((pruned_w != 0).sum().item())
        total_linear += pruned_w.numel()

    achieved = 1.0 - (total_kept / total_linear) if total_linear else 0.0
    return model, achieved


def apply_channel_pruning(model: nn.Module, sparsity: float = 0.3) -> Tuple[nn.Module, float]:
    """
    Structured channel/neuron pruning:
    Prunes the `sparsity` fraction of output channels (rows) with smallest L1 norm.
    """
    to_prune = _extract_linear_layers(model)
    if not to_prune:
        return model, 0.0

    fraction = min(max(sparsity, 0.0), 0.95)
    if fraction <= 0:
        return model, 0.0

    total_kept = 0
    total_linear = 0
    for name, module in to_prune:
        if _HFConv1D is not None and isinstance(module, _HFConv1D):
            w = module.weight.detach().float().t()  # [out, in]
            bias = module.bias
            in_f, out_f = module.nx, module.nf
        else:
            w = module.weight.detach().float()
            bias = module.bias
            in_f, out_f = module.in_features, module.out_features

        # Row-wise L1 norm
        row_norms = w.abs().sum(dim=1)
        k_prune = int(round(out_f * fraction))
        if k_prune >= out_f:
            k_prune = out_f - 1

        if k_prune > 0:
            pruned_rows = torch.topk(row_norms, k=k_prune, largest=False).indices
            mask = torch.ones(out_f, dtype=torch.bool, device=w.device)
            mask[pruned_rows] = False
            pruned_w = w * mask.unsqueeze(1).float()
            if bias is not None:
                bias = bias * mask.float()
        else:
            pruned_w = w

        sparse = SparseLinear(in_f, out_f, bias=bias)
        sparse.set_pruned_weight(pruned_w)

        parent = _find_parent(model, name)
        setattr(parent, name.split(".")[-1], sparse)

        total_kept += int((pruned_w != 0).sum().item())
        total_linear += pruned_w.numel()

    achieved = 1.0 - (total_kept / total_linear) if total_linear else 0.0
    return model, achieved
