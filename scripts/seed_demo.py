"""
Seed the dashboard with a real measured comparison set:
  - fp32, fp16, fp8, int8, int4, nf4
  - magnitude pruning, 2:4 structured pruning
  - combined prune + quantize

All metrics are measured live on real models.
"""
import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.models import (
    LOCAL_TINYGPT,
    TinyGPT,
    TinyGPTConfig,
    load_model,
    load_tokenizer,
    resolve_model_vocab_size,
)
from backend.core.pipeline import CompressionConfig, compress_model
from backend.core.utils import detect_device
from backend.history import add_experiment, clear_history

MODES = [
    ("fp32", {}),
    ("fp16", {}),
    ("fp8", {}),
    ("int8", {}),
    ("int4", {"group_size": 64}),
    ("nf4", {"group_size": 64}),
    ("prune", {"sparsity": 0.5}),
    ("prune_2_4", {}),
    ("combined", {"sparsity": 0.5, "quant_mode": "nf4", "prune_mode": "2_4"}),
]


def main():
    parser = argparse.ArgumentParser(description="Seed the dashboard with measured experiments.")
    parser.add_argument("--model", default="distilgpt2", help="HF model id or LocalTinyGPT")
    args = parser.parse_args()

    device = detect_device()
    print(f"Seeding demo experiments on {device} (model={args.model}) …")
    clear_history()

    tok = None
    if args.model == LOCAL_TINYGPT:
        base = TinyGPT(TinyGPTConfig(vocab_size=2000, n_embd=96, n_layer=3, n_head=3, n_ctx=128))
        vocab = 2000
    else:
        base, _ = load_model(args.model, device)
        vocab = resolve_model_vocab_size(args.model, base) or 1000
        tok = load_tokenizer(args.model)

    for mode, extra in MODES:
        print(f"  {mode} …", flush=True)
        config = CompressionConfig(
            mode=mode,
            vocab_size=vocab,
            gen_token_len=16,
            seq_len=12,
            tokenizer=tok,
            **extra,
        )
        res = compress_model(copy.deepcopy(base), config, save_dir="experiments", device=device)
        res["model_name"] = args.model
        res["model_source"] = args.model
        add_experiment(res)
        print(
            f"    -> {res['label']:<18} RAM {res['ram_mb']:>7.3f} MB  "
            f"ratio {res['ram_compression_ratio']:>5.2f}x  "
            f"PPL {res['perplexity']:>8.2f}  tok/s {res['tokens_per_sec']:>7.2f}"
        )

    print("\nDone. Start the backend + frontend and open the dashboard.")


if __name__ == "__main__":
    main()
