"""Tests for the quantization engine."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.models import TinyGPT, TinyGPTConfig
from backend.core.quantization import (
    pack_int4,
    pack_uint4,
    quantize_model,
    unpack_int4,
    unpack_uint4,
)


def test_pack_unpack_roundtrip():
    t = torch.randint(-8, 8, (4, 8)).to(torch.int8)
    packed = pack_int4(t)
    out = unpack_int4(packed, t.shape)
    assert out.shape == t.shape
    assert torch.equal(out, t), f"roundtrip mismatch: {out} != {t}"


def test_pack_unpack_uint4_roundtrip():
    t = torch.randint(0, 16, (4, 8)).to(torch.uint8)
    packed = pack_uint4(t)
    out = unpack_uint4(packed, t.shape)
    assert out.shape == t.shape
    assert torch.equal(out, t), f"uint4 roundtrip mismatch: {out} != {t}"


def test_quantize_int8_runs():
    model = TinyGPT(TinyGPTConfig(vocab_size=200, n_embd=32, n_layer=1, n_head=2, n_ctx=32))
    q = quantize_model(model, "int8")
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        out = q(x)
    assert out.logits.shape == (1, 8, 200)


def test_quantize_fp8_runs():
    model = TinyGPT(TinyGPTConfig(vocab_size=200, n_embd=32, n_layer=1, n_head=2, n_ctx=32))
    q = quantize_model(model, "fp8")
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        out = q(x)
    assert out.logits.shape == (1, 8, 200)


def test_quantize_int4_runs():
    model = TinyGPT(TinyGPTConfig(vocab_size=200, n_embd=32, n_layer=1, n_head=2, n_ctx=32))
    q = quantize_model(model, "int4", group_size=16)
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        out = q(x)
    assert out.logits.shape == (1, 8, 200)


def test_quantize_nf4_runs():
    model = TinyGPT(TinyGPTConfig(vocab_size=200, n_embd=32, n_layer=1, n_head=2, n_ctx=32))
    q = quantize_model(model, "nf4", group_size=16)
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        out = q(x)
    assert out.logits.shape == (1, 8, 200)


def test_quantize_fp16_runs():
    model = TinyGPT(TinyGPTConfig(vocab_size=200, n_embd=32, n_layer=1, n_head=2, n_ctx=32))
    q = quantize_model(model, "fp16")
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        out = q(x)
    assert out.logits.shape == (1, 8, 200)


if __name__ == "__main__":
    test_pack_unpack_roundtrip()
    test_pack_unpack_uint4_roundtrip()
    test_quantize_int8_runs()
    test_quantize_fp8_runs()
    test_quantize_int4_runs()
    test_quantize_nf4_runs()
    test_quantize_fp16_runs()
    print("ALL_QUANT_TESTS_PASSED")
