"""Tests for magnitude, 2:4 structured, and channel pruning + sparse storage."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.models import TinyGPT, TinyGPTConfig
from backend.core.pruning import (
    SparseLinear,
    apply_2_4_pruning,
    apply_channel_pruning,
    apply_magnitude_pruning,
    compute_sparsity,
)


def test_sparse_linear_roundtrip():
    w = torch.randn(8, 16)
    w[:2, :] = 0  # create some zeros
    sparse = SparseLinear(16, 8)
    sparse.set_pruned_weight(w)
    out = sparse.dense_weight()
    assert torch.allclose(out, w, atol=1e-2)  # fp16 storage precision
    assert sparse.sparsity > 0


def test_apply_pruning_removes_weights():
    model = TinyGPT(TinyGPTConfig(vocab_size=200, n_embd=32, n_layer=1, n_head=2, n_ctx=32))
    pruned, achieved = apply_magnitude_pruning(model, 0.5)
    total = 0
    kept = 0
    for m in pruned.modules():
        if isinstance(m, SparseLinear):
            total += m.out_features * m.in_features
            kept += m.values.numel()
    sp = 1.0 - kept / total
    assert sp > 0.4, f"achieved sparsity too low: {sp}"
    assert 0.0 <= achieved <= 1.0


def test_apply_2_4_pruning():
    model = TinyGPT(TinyGPTConfig(vocab_size=200, n_embd=32, n_layer=1, n_head=2, n_ctx=32))
    pruned, achieved = apply_2_4_pruning(model)
    assert 0.45 <= achieved <= 0.55
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        out = pruned(x)
    assert out.logits.shape == (1, 8, 200)


def test_apply_channel_pruning():
    model = TinyGPT(TinyGPTConfig(vocab_size=200, n_embd=32, n_layer=1, n_head=2, n_ctx=32))
    pruned, achieved = apply_channel_pruning(model, 0.25)
    assert achieved > 0.15
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        out = pruned(x)
    assert out.logits.shape == (1, 8, 200)


def test_pruned_model_forward():
    model = TinyGPT(TinyGPTConfig(vocab_size=200, n_embd=32, n_layer=1, n_head=2, n_ctx=32))
    pruned, _ = apply_magnitude_pruning(model, 0.5)
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        out = pruned(x)
    assert out.logits.shape == (1, 8, 200)


if __name__ == "__main__":
    test_sparse_linear_roundtrip()
    test_apply_pruning_removes_weights()
    test_apply_2_4_pruning()
    test_apply_channel_pruning()
    test_pruned_model_forward()
    print("ALL_PRUNE_TESTS_PASSED")
