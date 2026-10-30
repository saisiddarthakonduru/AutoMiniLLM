"""
FastAPI application for AutoMiniLLM.

Endpoints:
  GET  /api/health             -> service health
  GET  /api/system             -> CPU/GPU detection, RAM, torch version
  GET  /api/models             -> available model sources
  POST /api/compress           -> run a new compression experiment (real metrics)
  GET  /api/history            -> experiment history
  GET  /api/history/{id}       -> one experiment
  DELETE /api/history/{id}     -> delete one experiment
  DELETE /api/history          -> clear all
  POST /api/compare            -> compare two historical experiments
  POST /api/generate           -> sample text generation from a compressed model state file
  GET  /api/export/{id}        -> download or export compressed model package
  GET  /                       -> React dashboard (served from frontend/dist)
"""
import copy
import json
import logging
import os
import threading
import time
import traceback
from typing import Optional

import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.core.models import (
    AVAILABLE_MODELS,
    DEFAULT_HF_MODEL,
    LOCAL_TINYGPT,
    get_model_info,
    load_model,
    load_tokenizer,
    resolve_model_vocab_size,
)
from backend.core.pipeline import (
    ALL_MODES,
    CompressionConfig,
    _apply_combined,
    _apply_pruning,
    compress_model,
)
from backend.core.pruning import (
    PRUNE_MODES,
    apply_2_4_pruning,
    apply_channel_pruning,
    apply_magnitude_pruning,
)
from backend.core.quantization import QUANT_MODES, quantize_model
from backend.core.utils import detect_device, get_system_info, run_gc
from backend.history import (
    add_experiment,
    clear_history,
    delete_experiment,
    get_experiment,
    list_experiments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="AutoMiniLLM API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_lock = threading.Lock()
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(_base_dir, "experiments")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class CompressRequest(BaseModel):
    mode: str = Field(
        "int8",
        description="fp32|fp16|fp8|int8|int4|nf4|prune|prune_2_4|prune_channel|combined",
    )
    prune_mode: str = Field("magnitude", description="magnitude|2_4|channel")
    sparsity: float = Field(0.5, ge=0.0, le=0.99)
    group_size: int = Field(64, ge=16, le=256)
    quant_mode: str = Field(
        "int8", description="quant stage for combined (int8|fp8|int4|nf4)"
    )
    model: str = Field(DEFAULT_HF_MODEL)
    vocab_size: int = Field(50257, ge=100, le=200000)
    gen_token_len: int = Field(24, ge=4, le=512)
    seq_len: int = Field(20, ge=4, le=64)
    device: Optional[str] = Field(None, description="cuda|cpu; auto-detected if omitted")


class GenerateRequest(BaseModel):
    saved_path: str = ""
    exp_id: Optional[str] = None
    prompt: str = "The future of AI is"
    max_new_tokens: int = Field(64, ge=1, le=512)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    model_name: Optional[str] = Field(None, description="HF model id or LocalTinyGPT")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AutoMiniLLM", "version": "2.0.0"}


@app.get("/api/system")
def system():
    return get_system_info()


@app.get("/api/models")
def models():
    details = [
        {
            "name": name,
            "label": get_model_info(name).get("label", name),
            "tokenizer": get_model_info(name).get("tokenizer"),
            "vocab_size": get_model_info(name).get("vocab_size"),
        }
        for name in AVAILABLE_MODELS
    ]
    return {
        "available": AVAILABLE_MODELS,
        "details": details,
        "default": DEFAULT_HF_MODEL,
        "local": LOCAL_TINYGPT,
        "supported_modes": list(ALL_MODES),
        "supported_quant_modes": list(QUANT_MODES),
        "supported_prune_modes": list(PRUNE_MODES),
    }


@app.post("/api/compress")
def compress(req: CompressRequest):
    mode = req.mode.lower()
    if mode not in ALL_MODES:
        raise HTTPException(
            400, f"mode must be one of {ALL_MODES}, got '{mode}'"
        )
    if mode == "combined" and req.quant_mode not in ("int8", "fp8", "int4", "nf4"):
        raise HTTPException(
            400, "quant_mode for combined must be one of: int8, fp8, int4, nf4"
        )

    device = req.device or detect_device()
    with _lock:
        try:
            base_model, source = load_base_model(req.model)
            resolved_vocab = resolve_model_vocab_size(req.model, base_model) or req.vocab_size
            tok = load_tokenizer(req.model)

            config = CompressionConfig(
                mode=mode,
                prune_mode=req.prune_mode,
                sparsity=req.sparsity,
                group_size=req.group_size,
                quant_mode=req.quant_mode,
                vocab_size=resolved_vocab,
                gen_token_len=req.gen_token_len,
                seq_len=req.seq_len,
                tokenizer=tok,
            )
            result = compress_model(
                base_model,
                config,
                save_dir=SAVE_DIR,
                device=device,
            )
            result["model_source"] = source
            result["model_name"] = req.model
            result["vocab_size_used"] = resolved_vocab
            result["tokenizer"] = get_model_info(req.model).get("tokenizer")
            add_experiment(result)
            run_gc(device)
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Compression failed: %s\n%s", e, traceback.format_exc())
            run_gc(device)
            raise HTTPException(500, f"Compression failed: {e}")


@app.get("/api/history")
def history():
    return {"experiments": list_experiments()}


@app.get("/api/history/{exp_id}")
def history_one(exp_id: str):
    exp = get_experiment(exp_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")
    return exp


@app.delete("/api/history/{exp_id}")
def history_delete(exp_id: str):
    ok = delete_experiment(exp_id)
    if not ok:
        raise HTTPException(404, "Experiment not found")
    return {"deleted": exp_id}


@app.delete("/api/history")
def history_clear():
    n = clear_history()
    return {"deleted": n}


@app.post("/api/compare")
def compare(req: dict | None = None):
    if req is None:
        raise HTTPException(400, "Provide experiment ids")
    id_a = req.get("exp_a")
    id_b = req.get("exp_b")
    if not id_a or not id_b:
        raise HTTPException(400, "exp_a and exp_b required")
    a, b = get_experiment(id_a), get_experiment(id_b)
    if a is None or b is None:
        raise HTTPException(404, "One or both experiments not found")
    keys = [
        "ram_mb",
        "base_ram_mb",
        "ram_compression_ratio",
        "file_size_mb",
        "base_file_size_mb",
        "file_compression_ratio",
        "perplexity",
        "loss",
        "kl_vs_baseline",
        "tokens_per_sec",
        "ms_per_token",
        "ttft_ms",
        "prefill_latency_ms",
        "achieved_sparsity",
        "compress_seconds",
    ]
    rows = []
    for k in keys:
        va, vb = a.get(k), b.get(k)
        diff = None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va:
            diff = round((vb - va) / abs(va) * 100.0, 2)
        rows.append({"metric": k, "exp_a": va, "exp_b": vb, "pct_change_b_vs_a": diff})
    return {
        "exp_a": {"id": id_a, "label": a.get("label"), "mode": a.get("mode")},
        "exp_b": {"id": id_b, "label": b.get("label"), "mode": b.get("mode")},
        "rows": rows,
    }


@app.post("/api/generate")
def generate(req: GenerateRequest):
    """
    Generate real text from a saved compressed model state.
    Reconstructs the matching quantized/sparse architecture before loading weights.
    """
    path = req.saved_path
    if not path and req.exp_id:
        exp = get_experiment(req.exp_id)
        if exp and exp.get("saved_path"):
            path = exp["saved_path"]
            if not req.model_name:
                req.model_name = exp.get("model_name")

    if not path or not os.path.exists(path):
        raise HTTPException(400, "saved_path or exp_id must point to a valid .pt file")

    meta_path = path.replace(".pt", "_meta.json")
    meta = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass

    state = torch.load(path, map_location="cpu", weights_only=False)
    model_name = req.model_name or (meta.get("model_name") if meta else None) or DEFAULT_HF_MODEL

    try:
        # Load base architecture
        base_model, _ = load_base_model(model_name)
        tokenizer = load_tokenizer(model_name) if model_name != LOCAL_TINYGPT else None

        # Reconstruct layer modifications from metadata if present
        if meta:
            mode = meta.get("mode", "fp32")
            if mode in ("fp16", "fp8", "int8", "int4", "nf4"):
                base_model = quantize_model(base_model, mode, group_size=meta.get("group_size", 64))
            elif mode == "prune":
                base_model, _ = apply_magnitude_pruning(base_model, meta.get("sparsity", 0.5))
            elif mode == "prune_2_4":
                base_model, _ = apply_2_4_pruning(base_model)
            elif mode == "prune_channel":
                base_model, _ = apply_channel_pruning(base_model, meta.get("sparsity", 0.3))
            elif mode == "combined":
                cfg = CompressionConfig(
                    mode="combined",
                    prune_mode=meta.get("prune_mode", "magnitude"),
                    sparsity=meta.get("sparsity", 0.5),
                    quant_mode=meta.get("quant_mode", "int8"),
                    group_size=meta.get("group_size", 64),
                )
                base_model = _apply_combined(base_model, cfg)

        # Load weights into the accurately shaped model
        try:
            base_model.load_state_dict(state, strict=False)
        except Exception as e:
            logger.warning("Non-strict load warning: %s", e)

        base_model.eval()

        t0 = time.perf_counter()
        if tokenizer is not None:
            tokenizer_ids = tokenizer(req.prompt, add_special_tokens=False)["input_ids"]
            if not tokenizer_ids:
                tokenizer_ids = [tokenizer.bos_token_id or 1]
            input_ids = torch.tensor([tokenizer_ids], dtype=torch.long)
            with torch.no_grad():
                out = base_model.generate(
                    input_ids,
                    max_new_tokens=req.max_new_tokens,
                    temperature=max(0.1, float(req.temperature)),
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id or 0,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen_ids = out[0].tolist()
            gen_only_ids = gen_ids[len(tokenizer_ids):]
            generated = tokenizer.decode(gen_only_ids, skip_special_tokens=True)
            if not generated or not generated.strip():
                # Fallback: decode with special tokens
                generated = tokenizer.decode(gen_only_ids, skip_special_tokens=False)
            if not generated or not generated.strip():
                # Fallback: printable characters
                generated = id_to_text(gen_only_ids)
            if generated:
                generated = generated.replace("\ufffd", "").strip()
            tok_count = max(0, len(gen_only_ids))
        else:
            cfg = getattr(base_model, "config", None)
            vocab = getattr(cfg, "vocab_size", 2000)
            tokenizer_ids = simple_prompt_to_ids(req.prompt, vocab_size=vocab)
            input_ids = torch.tensor([tokenizer_ids], dtype=torch.long)
            with torch.no_grad():
                out = base_model.generate(
                    input_ids,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                )
            gen_ids = out[0].tolist()
            gen_only_ids = gen_ids[len(tokenizer_ids):]
            tok_count = max(0, len(gen_only_ids))
            generated = id_to_text(gen_only_ids)

        elapsed = time.perf_counter() - t0
        tokens_per_sec = round(tok_count / elapsed, 2) if elapsed > 0 else 0.0

        return {
            "prompt": req.prompt,
            "generated_text": generated,
            "tokens_generated": tok_count,
            "elapsed_ms": round(elapsed * 1000.0, 2),
            "tokens_per_sec": tokens_per_sec,
            "model_name": model_name,
        }
    except Exception as e:
        logger.exception("Generate failed: %s", e)
        raise HTTPException(500, f"Generation failed: {e}")


@app.get("/api/export/{exp_id}")
def export_experiment(exp_id: str):
    """Download the compressed model state file or view export package info."""
    exp = get_experiment(exp_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")
    path = exp.get("saved_path")
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Saved model file not found on disk")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=os.path.basename(path),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_base_model(model_name: str):
    """Load base model. Uses transformers if available; falls back to TinyGPT."""
    try:
        model, source = load_model(model_name)
    except Exception as e:
        logger.warning("HF load failed (%s); using LocalTinyGPT", e)
        model, source = load_model(LOCAL_TINYGPT)
    return model, source


def simple_prompt_to_ids(prompt: str, vocab_size: int) -> list:
    """Map a prompt string to token ids (hash-based, demo use only)."""
    out = []
    for ch in prompt:
        out.append(abs(hash(ch)) % max(4, vocab_size - 1) + 1)
    if not out:
        out = [1]
    return out


def id_to_text(ids: list) -> str:
    """Map ids back to approximate characters (demo)."""
    chars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-_'\"()[]{}:#@$%&*+=/\\|<>;"
    return "".join(
        chars[i % len(chars)] if i < len(chars) else chars[(i * 7) % len(chars)]
        for i in ids
    )


# ---------------------------------------------------------------------------
# Serve the React dashboard (built into frontend/dist)
# ---------------------------------------------------------------------------
_FRONTEND_DIST = os.path.join(_base_dir, "frontend", "dist")
if os.path.isdir(_FRONTEND_DIST):
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(_FRONTEND_DIST, "assets")),
        name="assets",
    )

    @app.get("/", response_class=FileResponse)
    async def serve_index():
        return FileResponse(os.path.join(_FRONTEND_DIST, "index.html"))

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404, "Not found")
        return FileResponse(os.path.join(_FRONTEND_DIST, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
