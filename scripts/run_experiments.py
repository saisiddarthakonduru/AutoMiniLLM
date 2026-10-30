"""
CLI benchmark: runs all compression modes and prints a summary table of the
measured metrics to the console + saves them to history.

Usage:
  python scripts/run_experiments.py                # local TinyGPT (offline default)
  python scripts/run_experiments.py --model distilgpt2
  python scripts/run_experiments.py --model gpt2 --small
  python scripts/run_experiments.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --small
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.models import LOCAL_TINYGPT, TinyGPT, TinyGPTConfig, load_model, resolve_model_vocab_size
from backend.core.pipeline import CompressionConfig, compress_model
from backend.history import add_experiment
from backend.core.utils import detect_device

MODES = ["fp32", "fp16", "int8", "int4", "prune", "combined"]


def main():
    parser = argparse.ArgumentParser(description="Run all compression modes on a model.")
    parser.add_argument("--model", default=LOCAL_TINYGPT, help="HF model id or LocalTinyGPT")
    parser.add_argument("--small", action="store_true", help="very short eval (fast)")
    args = parser.parse_args()

    device = detect_device()
    print(f"Running benchmark on device: {device}\n")

    if args.model == LOCAL_TINYGPT:
        base = TinyGPT(TinyGPTConfig(vocab_size=2000, n_embd=96, n_layer=3, n_head=3, n_ctx=128))
        vocab = 2000
    else:
        base, _ = load_model(args.model, device)
        vocab = resolve_model_vocab_size(args.model, base) or 1000
    gen_len = 8 if args.small else 16
    seq_len = 8 if args.small else 12

    headers = ["Mode", "RAM MB", "RAM ratio", "File MB", "File ratio", "PPL", "KL", "tok/s", "ms/tok", "Spars%"]
    rows = []
    for mode in MODES:
        print(f"  Compressing: {mode} …", flush=True)
        config = CompressionConfig(
            mode=mode,
            sparsity=0.5,
            group_size=64,
            quant_mode="int8" if mode != "combined" else "int8",
            vocab_size=vocab,
            gen_token_len=gen_len,
            seq_len=seq_len,
        )
        try:
            res = compress_model(base, config, save_dir="experiments", device=device)
            add_experiment(res)
            rows.append([
                mode,
                res["label"],
                res["ram_mb"],
                res["ram_compression_ratio"],
                res["base_ram_mb"],
                res["file_compression_ratio"],
                res["perplexity"],
                res["kl_vs_baseline"],
                res["tokens_per_sec"],
                res["ms_per_token"],
                res["achieved_sparsity"],
            ])
        except Exception as e:
            print(f"    ! {mode} failed: {e}", flush=True)

    print("\n=== RESULTS (real measured) ===")
    print(f"{'MODE':<10} {'LABEL':<18} {'RAM MB':>8} {'RATIO':>7} {'BASE MB':>8} {'FILE x':>8} {'PPL':>9} {'KL':>9} {'tok/s':>8} {'ms/tok':>8} {'SPARS%':>7}")
    for r in rows:
        print(
            f"{r[0]:<10} {r[1]:<18} {r[2]:>8.3f} {r[3]:>7.2f} {r[4]:>8.3f} "
            f"{r[5]:>8.2f} {r[6]:>9.2f} {r[7]:>9.6f} {r[8]:>8.2f} {r[9]:>8.3f} {r[10]:>7.1f}"
        )
    print("\nDone. Open the dashboard or GET /api/history to inspect.")


if __name__ == "__main__":
    main()