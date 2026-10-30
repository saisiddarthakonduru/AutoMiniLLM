"""Validate all compression modes on a real pretrained model (CPU)."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.models import load_model, resolve_model_vocab_size
from backend.core.pipeline import CompressionConfig, compress_model
from backend.core.utils import detect_device, run_gc

MODES = ["fp32", "fp16", "int8", "int4", "prune", "combined"]


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "distilgpt2"
    small = "--small" in sys.argv
    device = detect_device()
    print(f"Model: {model_name} | device: {device} | small={small}")

    gen_len = 8 if small else 12
    seq_len = 8 if small else 10

    for mode in MODES:
        print(f"\n=== {mode} ===", flush=True)
        t0 = time.perf_counter()
        model, source = load_model(model_name, device)
        vocab = resolve_model_vocab_size(model_name, model) or 1000
        cfg = CompressionConfig(
            mode=mode,
            sparsity=0.5,
            group_size=64,
            quant_mode="int8" if mode != "combined" else "int8",
            vocab_size=vocab,
            gen_token_len=gen_len,
            seq_len=seq_len,
        )
        res = compress_model(model, cfg, save_dir="experiments", device=device)
        dt = time.perf_counter() - t0
        print(
            f"  ratio(ram)={res['ram_compression_ratio']}  file_mb={res['file_size_mb']} "
            f"ppl={res['perplexity']}  tok/s={res['tokens_per_sec']}  "
            f"spars={res['achieved_sparsity']}%  {dt:.1f}s  => {res['label']}"
        )
        # a forward pass on the saved state isn't possible without config; check state file
        assert os.path.exists(res["saved_path"]), res["saved_path"]
        run_gc(device)
    print("\nALL_MODES_OK")


if __name__ == "__main__":
    main()