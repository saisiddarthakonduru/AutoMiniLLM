"""End-to-end tests for the compression pipeline (real measurements)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.models import TinyGPT, TinyGPTConfig
from backend.core.pipeline import CompressionConfig, compress_model
from backend.history import add_experiment, clear_history, list_experiments


def _make_model():
    return TinyGPT(TinyGPTConfig(vocab_size=200, n_embd=32, n_layer=1, n_head=2, n_ctx=32))


def test_compress_int8_measures():
    res = compress_model(
        _make_model(),
        CompressionConfig(mode="int8", vocab_size=200, gen_token_len=4, seq_len=6),
        save_dir="experiments/test",
    )
    assert res["ram_compression_ratio"] > 1.0
    assert res["perplexity"] > 0
    assert res["tokens_per_sec"] is not None and res["tokens_per_sec"] > 0
    assert res["file_size_mb"] > 0


def test_compress_int4_measures():
    res = compress_model(
        _make_model(),
        CompressionConfig(mode="int4", group_size=16, vocab_size=200, gen_token_len=4, seq_len=6),
        save_dir="experiments/test",
    )
    assert res["ram_compression_ratio"] > 1.0


def test_compress_nf4_measures():
    res = compress_model(
        _make_model(),
        CompressionConfig(mode="nf4", group_size=16, vocab_size=200, gen_token_len=4, seq_len=6),
        save_dir="experiments/test",
    )
    assert res["ram_compression_ratio"] > 1.0
    assert res["perplexity"] > 0


def test_compress_fp8_measures():
    res = compress_model(
        _make_model(),
        CompressionConfig(mode="fp8", vocab_size=200, gen_token_len=4, seq_len=6),
        save_dir="experiments/test",
    )
    assert res["ram_compression_ratio"] > 1.0
    assert res["perplexity"] > 0


def test_compress_prune_measures():
    res = compress_model(
        _make_model(),
        CompressionConfig(mode="prune", sparsity=0.5, vocab_size=200, gen_token_len=4, seq_len=6),
        save_dir="experiments/test",
    )
    assert res["achieved_sparsity"] > 30.0


def test_compress_prune_2_4_measures():
    res = compress_model(
        _make_model(),
        CompressionConfig(mode="prune_2_4", vocab_size=200, gen_token_len=4, seq_len=6),
        save_dir="experiments/test",
    )
    assert 40.0 <= res["achieved_sparsity"] <= 60.0


def test_compress_combined_int4_measures():
    """Validates combined pruning + INT4 packing bug fix."""
    res = compress_model(
        _make_model(),
        CompressionConfig(
            mode="combined",
            prune_mode="magnitude",
            sparsity=0.5,
            quant_mode="int4",
            group_size=16,
            vocab_size=200,
            gen_token_len=4,
            seq_len=6,
        ),
        save_dir="experiments/test",
    )
    assert res["achieved_sparsity"] > 30.0
    assert res["tokens_per_sec"] is not None


def test_compress_combined_nf4_measures():
    """Validates combined pruning + NF4 packing."""
    res = compress_model(
        _make_model(),
        CompressionConfig(
            mode="combined",
            prune_mode="2_4",
            quant_mode="nf4",
            group_size=16,
            vocab_size=200,
            gen_token_len=4,
            seq_len=6,
        ),
        save_dir="experiments/test",
    )
    assert res["achieved_sparsity"] > 30.0
    assert res["tokens_per_sec"] is not None


def test_history_roundtrip():
    clear_history()
    res = compress_model(
        _make_model(),
        CompressionConfig(mode="fp16", vocab_size=200, gen_token_len=4, seq_len=6),
        save_dir="experiments/test",
    )
    add_experiment(res)
    exps = list_experiments()
    assert len(exps) == 1
    assert exps[0]["label"] == "fp16"
    clear_history()


if __name__ == "__main__":
    test_compress_int8_measures()
    test_compress_int4_measures()
    test_compress_nf4_measures()
    test_compress_fp8_measures()
    test_compress_prune_measures()
    test_compress_prune_2_4_measures()
    test_compress_combined_int4_measures()
    test_compress_combined_nf4_measures()
    test_history_roundtrip()
    print("ALL_PIPELINE_TESTS_PASSED")
