"""Tests for real pretrained models: registry, tokenizers, and GPT-2 Conv1D support."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.models import (
    AVAILABLE_MODELS,
    DEFAULT_HF_MODEL,
    MODEL_REGISTRY,
    LOCAL_TINYGPT,
    get_model_info,
    get_tokenizer_name,
    get_vocab_size,
    load_model,
    load_tokenizer,
)
from backend.core.pruning import apply_magnitude_pruning, compute_sparsity
from backend.core.quantization import quantize_model


def test_registry_has_real_models():
    for name in ("distilgpt2", "gpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        assert name in AVAILABLE_MODELS
        assert name in MODEL_REGISTRY
    # real vocab sizes
    assert get_vocab_size("distilgpt2") == 50257
    assert get_vocab_size("gpt2") == 50257
    assert get_vocab_size("TinyLlama/TinyLlama-1.1B-Chat-v1.0") == 32000
    assert get_vocab_size(DEFAULT_HF_MODEL) == 1000
    assert get_vocab_size(LOCAL_TINYGPT) == 2000


def test_registry_tokenizers():
    assert get_tokenizer_name("distilgpt2") == "distilgpt2"
    assert get_tokenizer_name("gpt2") == "gpt2"
    assert (
        get_tokenizer_name("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    assert get_tokenizer_name(LOCAL_TINYGPT) is None
    assert get_model_info("some/custom-model")["tokenizer"] == "some/custom-model"


def test_load_distilgpt2_tokenizer():
    tok = load_tokenizer("distilgpt2")
    assert tok is not None
    assert tok.vocab_size == 50257
    ids = tok("Hello world", add_special_tokens=False)["input_ids"]
    assert len(ids) > 0
    assert tok.decode(ids) == "Hello world"


def test_load_real_tiny_gpt2_model():
    model, source = load_model(DEFAULT_HF_MODEL)
    assert source == DEFAULT_HF_MODEL
    assert model.config.vocab_size == 1000
    x = torch.randint(0, 1000, (1, 8))
    with torch.no_grad():
        out = model(x)
    assert out.logits.shape == (1, 8, 1000)


def test_quantize_real_gpt2_conv1d():
    """GPT-2 attention/MLP use Conv1D; quantize_model must convert them."""
    model, _ = load_model(DEFAULT_HF_MODEL)
    n_conv1d = 0
    from transformers.pytorch_utils import Conv1D

    for m in model.modules():
        if isinstance(m, Conv1D):
            n_conv1d += 1
    assert n_conv1d > 0, "expected GPT-2 to contain Conv1D layers"

    q = quantize_model(model, "int8")
    x = torch.randint(0, 1000, (1, 8))
    with torch.no_grad():
        out = q(x)
    assert out.logits.shape == (1, 8, 1000)


def test_prune_real_gpt2_conv1d():
    model, _ = load_model(DEFAULT_HF_MODEL)
    pruned, achieved = apply_magnitude_pruning(model, 0.5)
    assert achieved > 0.35, f"achieved sparsity {achieved}"
    x = torch.randint(0, 1000, (1, 8))
    with torch.no_grad():
        out = pruned(x)
    assert out.logits.shape == (1, 8, 1000)


if __name__ == "__main__":
    test_registry_has_real_models()
    test_registry_tokenizers()
    test_load_distilgpt2_tokenizer()
    test_load_real_tiny_gpt2_model()
    test_quantize_real_gpt2_conv1d()
    test_prune_real_gpt2_conv1d()
    print("ALL_MODEL_TESTS_PASSED")