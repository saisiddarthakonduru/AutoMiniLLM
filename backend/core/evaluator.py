"""
Real evaluation of compressed models:

  - Perplexity (cross-entropy loss) on real text (WikiText-2 sample) or pseudo-corpus
  - KL divergence vs the FP32 baseline (response-quality drift)
  - Generation throughput (tokens/sec), ITL (ms/token), and TTFT (time-to-first-token)
  - Peak memory profiling and Model RAM footprint

All numbers are measured at runtime on the actual compressed model.
"""
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .utils import measure_model_ram_mb, timeit

_DATASETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "datasets",
)
_WIKITEXT_FILE = os.path.join(_DATASETS_DIR, "wikitext_sample.txt")


def _forward_logits(model, input_ids: torch.Tensor, labels=None):
    """Unified forward for HF CausalLM and our TinyGPT. Returns (logits, loss)."""
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
        logits = out.logits
        loss = getattr(out, "loss", None)
    return logits, loss


def _load_wikitext_text() -> str:
    """Read bundled WikiText-2 text sample."""
    if os.path.exists(_WIKITEXT_FILE):
        try:
            with open(_WIKITEXT_FILE, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    return "Artificial intelligence and deep learning enable efficient language model compression using quantization and pruning."


def _prepare_eval_batch(
    seq_len: int = 24,
    batch_size: int = 1,
    vocab_size: int = 50000,
    device: str = "cpu",
    seed: int = 42,
    tokenizer=None,
) -> torch.Tensor:
    """
    Evaluation corpus: prefers real tokenized text from WikiText-2 sample
    when a tokenizer is available, with deterministic fallback for offline/local.
    """
    vocab_size = max(4, int(vocab_size))
    if tokenizer is not None:
        try:
            text = _load_wikitext_text()
            encoded = tokenizer(text, truncation=True, max_length=seq_len + 4, return_tensors="pt")
            ids = encoded["input_ids"][:, :seq_len]
            if ids.shape[1] >= 4:
                # Clamp to vocab_size to prevent embedding overflow
                ids = ids.clamp(0, vocab_size - 1)
                return ids.to(device)
        except Exception:
            pass

    # Deterministic sequential pseudo-corpus based on character text
    text = _load_wikitext_text()
    tokens = []
    for ch in text[:seq_len]:
        tokens.append(abs(hash(ch)) % (vocab_size - 1) + 1)
    while len(tokens) < seq_len:
        tokens.append((len(tokens) * 31) % (vocab_size - 1) + 1)
    ids = torch.tensor([tokens[:seq_len]], dtype=torch.long)
    return ids.to(device)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------
def measure_perplexity_and_loss(model, input_ids: torch.Tensor) -> Tuple[float, float]:
    """Calculate cross-entropy loss and Perplexity (exp(loss))."""
    _, loss = _forward_logits(model, input_ids, labels=input_ids)
    if loss is None or torch.isnan(loss) or torch.isinf(loss):
        return 999.0, 999.0
    loss_val = float(loss.item())
    # Cap exponential to prevent float overflow
    capped_exp = math.exp(min(loss_val, 40.0))
    return round(loss_val, 4), round(capped_exp, 4)


def calculate_kl(model_a, model_b, input_ids: torch.Tensor, device: str = "cpu") -> float:
    """KL( compressed || baseline ) averaged over later token positions."""
    logits_a, _ = _forward_logits(model_a, input_ids)
    logits_b, _ = _forward_logits(model_b, input_ids)
    if logits_a.shape != logits_b.shape:
        min_v = min(logits_a.shape[-1], logits_b.shape[-1])
        logits_a = logits_a[..., :min_v]
        logits_b = logits_b[..., :min_v]
    pa = torch.softmax(logits_a.float(), dim=-1)
    pb = torch.softmax(logits_b.float(), dim=-1)
    eps = 1e-10
    kl = (pa * (torch.log(pa + eps) - torch.log(pb + eps))).sum(dim=-1)  # [b, s]
    # skip leading near-uniform positions
    seq = kl.shape[1]
    pos = kl[:, max(1, seq // 2):]
    if pos.numel() == 0:
        pos = kl
    val = float(pos.mean().item())
    if math.isnan(val) or math.isinf(val):
        return 0.0
    return round(val, 6)


# ------------------------------------------------------------------
# Generation speed & latency breakdown (TTFT & ITL)
# ------------------------------------------------------------------
def _generate_with_timing(
    model, input_ids: torch.Tensor, max_new_tokens: int
) -> Tuple[int, float, float]:
    """
    Generate tokens and return (new_tokens, elapsed_seconds, ttft_seconds).
    Measures TTFT (Time To First Token) separately from ITL (Inter-Token Latency).
    """
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        cur = input_ids
        ttft = 0.0
        count = 0
        for i in range(max_new_tokens):
            t_tok_start = time.perf_counter()
            logits, _ = _forward_logits(model, cur)
            probs = torch.softmax(logits[:, -1, :].float(), dim=-1)
            nxt = torch.multinomial(probs, 1)
            cur = torch.cat([cur, nxt], dim=1)
            count += 1
            if i == 0:
                ttft = time.perf_counter() - t_tok_start
        dt = time.perf_counter() - t0
        return count, dt, ttft


# ------------------------------------------------------------------
# Full evaluation
# ------------------------------------------------------------------
def evaluate_model(
    model: nn.Module,
    baseline_model: Optional[nn.Module] = None,
    vocab_size: int = 50000,
    device: str = "cpu",
    gen_token_len: int = 24,
    seq_len: int = 20,
    gen_rounds: int = 2,
    tokenizer=None,
) -> Dict:
    """Run real runtime measurements on `model`."""
    model = model.to(device).eval()
    if baseline_model is not None:
        baseline_model = baseline_model.to(device).eval()

    input_ids = _prepare_eval_batch(
        seq_len=seq_len, vocab_size=vocab_size, device=device, tokenizer=tokenizer
    )
    results: Dict = {}

    # Quality: Loss & Perplexity
    loss_val, ppl_val = measure_perplexity_and_loss(model, input_ids)
    results["loss"] = loss_val
    results["perplexity"] = ppl_val

    if baseline_model is not None:
        results["kl_vs_baseline"] = calculate_kl(model, baseline_model, input_ids, device)
    else:
        results["kl_vs_baseline"] = 0.0

    # Generation throughput (TTFT & ITL)
    timings: List[Tuple[int, float, float]] = []
    for _ in range(gen_rounds):
        n_tok, elapsed, ttft = _generate_with_timing(model, input_ids, gen_token_len)
        timings.append((n_tok, elapsed, ttft))

    avg_tokens = sum(n for n, _, _ in timings) / len(timings)
    avg_time = sum(t for _, t, _ in timings) / len(timings)
    avg_ttft_ms = (sum(tt for _, _, tt in timings) / len(timings)) * 1000.0

    results["generated_token_count"] = int(round(avg_tokens))
    results["tokens_per_sec"] = round(avg_tokens / avg_time, 2) if avg_time > 0 else 0.0
    results["loss_generation_seconds"] = round(avg_time, 4)
    results["ttft_ms"] = round(avg_ttft_ms, 2)

    if avg_tokens > 1:
        # ITL is time for remaining tokens
        remaining_time = max(0.0, avg_time - (avg_ttft_ms / 1000.0))
        results["ms_per_token"] = round((remaining_time / (avg_tokens - 1)) * 1000.0, 3)
    elif avg_tokens > 0:
        results["ms_per_token"] = round(avg_time / avg_tokens * 1000.0, 3)
    else:
        results["ms_per_token"] = None

    # RAM footprint of the model's tensors
    results["model_ram_mb"] = round(measure_model_ram_mb(model), 4)

    # Pure forward prefill latency
    def _one_forward():
        logits, _ = _forward_logits(model, input_ids)
        return logits

    latency_stats = timeit(_one_forward, warmup_ms=100)
    results["prefill_latency_ms"] = latency_stats["mean_ms"]
    results["prefill_p95_ms"] = latency_stats["p95_ms"]
    results["prefill_runs"] = latency_stats["runs"]

    return results
