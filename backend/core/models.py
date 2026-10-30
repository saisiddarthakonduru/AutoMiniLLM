"""
Model source for AutoMiniLLM.

Provides:
  - DEFAULT_MODEL: a tiny real HuggingFace GPT-2 (hf-internal-testing/tiny-random-gpt2)
    which is extremely lightweight (a few MB) and perfect for laptops.
  - A pure-PyTorch fallback TinyGPT causal-LM if the HF download fails or is offline.
"""
import copy
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_HF_MODEL = "distilgpt2"
FALLBACK_SMALL_HF_MODEL = "hf-internal-testing/tiny-random-bert"  # unused, kept for info
LOCAL_TINYGPT = "LocalTinyGPT"

# Registry of shipped models. Each entry stores the exact HF id, the matching
# tokenizer id, and the REAL vocab size so the pipeline never guesses.
TESTING_HF_MODEL = "hf-internal-testing/tiny-random-gpt2"

MODEL_REGISTRY = {
    "distilgpt2": {
        "label": "DistilGPT-2 (82M - Fluent English)",
        "tokenizer": "distilgpt2",
        "vocab_size": 50257,
    },
    "gpt2": {
        "label": "GPT-2 (124M)",
        "tokenizer": "gpt2",
        "vocab_size": 50257,
    },
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "label": "TinyLlama-1.1B-Chat-v1.0",
        "tokenizer": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "vocab_size": 32000,
    },
    TESTING_HF_MODEL: {
        "label": "GPT-2 (tiny random, test fixture only)",
        "tokenizer": TESTING_HF_MODEL,
        "vocab_size": 1000,
    },
    LOCAL_TINYGPT: {
        "label": "LocalTinyGPT (offline benchmark, no download)",
        "tokenizer": None,
        "vocab_size": 2000,
    },
}

AVAILABLE_MODELS = [
    "distilgpt2",
    "gpt2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    TESTING_HF_MODEL,
    LOCAL_TINYGPT,
]


def get_model_info(model_name: str) -> dict:
    """Registry metadata for a model id (safe for arbitrary HF ids too)."""
    model_name = model_name or LOCAL_TINYGPT
    if model_name in MODEL_REGISTRY:
        info = dict(MODEL_REGISTRY[model_name])
    else:
        info = {"label": model_name, "tokenizer": model_name, "vocab_size": None}
    info["model_name"] = model_name
    return info


def get_tokenizer_name(model_name: str) -> Optional[str]:
    """The HF tokenizer id that matches the given model (None for LocalTinyGPT)."""
    info = get_model_info(model_name)
    tok = info.get("tokenizer")
    if tok:
        return tok
    return None if model_name == LOCAL_TINYGPT else model_name


def get_vocab_size(model_name: str) -> Optional[int]:
    """True vocab size from the registry, or None (resolved from the model config)."""
    vs = get_model_info(model_name).get("vocab_size")
    return int(vs) if vs else None


# ---------------------------------------------------------------------------
#  Tiny local GPT fallback (causal LM) — GPU/CPU friendly
# ---------------------------------------------------------------------------
class TinyGPTConfig:
    def __init__(
        self,
        vocab_size: int = 2000,
        n_ctx: int = 128,
        n_embd: int = 96,
        n_layer: int = 3,
        n_head: int = 3,
    ):
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head


class TinyGPTBlock(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        hidden = cfg.n_embd * 4
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        # plain q/k/v/out Linear projections so the quantizer/pruner can
        # convert them cleanly (nn.MultiheadAttention is not quantizer-friendly).
        self.q = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.k = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.v = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.out = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.n_embd),
        )

    def forward(self, x):
        bsz, seq, dim = x.shape
        hd = dim // self.n_head

        h = self.ln1(x)
        q = self.q(h).view(bsz, seq, self.n_head, hd).transpose(1, 2)
        k = self.k(h).view(bsz, seq, self.n_head, hd).transpose(1, 2)
        v = self.v(h).view(bsz, seq, self.n_head, hd).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (hd ** 0.5)
        causal = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        attn = probs @ v  # [b, nh, s, hd]
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq, dim)
        x = x + self.out(attn)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: Optional[TinyGPTConfig] = None):
        super().__init__()
        cfg = cfg or TinyGPTConfig()
        self.config = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_ctx, cfg.n_embd)
        self.blocks = nn.ModuleList([TinyGPTBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        device = input_ids.device
        bsz, seq = input_ids.shape
        x = self.wte(input_ids) + self.wpe(
            torch.arange(seq, device=device).unsqueeze(0)
        )
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return SimpleOutput(loss=loss, logits=logits)

    def wpe_pos(self, positions: torch.Tensor) -> torch.Tensor:
        return self.wpe(positions)

    def generate(self, input_ids, max_new_tokens=20, temperature=1.0, **kwargs):
        self.eval()
        out = input_ids
        with torch.no_grad():
            for _ in range(max_new_tokens):
                ctx = out[:, -self.config.n_ctx :]
                logits = self(ctx).logits[:, -1, :]
                if temperature != 1.0:
                    logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                out = torch.cat([out, nxt], dim=1)
        return out


class SimpleOutput:
    def __init__(self, logits, loss=None):
        self.logits = logits
        self.loss = loss

    def __getitem__(self, k):
        return getattr(self, k)


# ---------------------------------------------------------------------------
#  Loader
# ---------------------------------------------------------------------------
def load_tokenizer(model_name: str):
    """Load the tokenizer matching `model_name` (None for LocalTinyGPT / offline)."""
    model_name = model_name or LOCAL_TINYGPT
    if model_name == LOCAL_TINYGPT:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    tok_name = get_tokenizer_name(model_name) or model_name
    try:
        return AutoTokenizer.from_pretrained(tok_name, local_files_only=True)
    except Exception:
        pass
    try:
        return AutoTokenizer.from_pretrained(tok_name)
    except Exception as e:
        logger.warning("Tokenizer load failed for '%s': %s", tok_name, e)
        return None


def resolve_model_vocab_size(model_name: str, model=None) -> Optional[int]:
    """True vocab size: registry -> model config -> first embedding table."""
    vs = get_vocab_size(model_name)
    if vs:
        return vs
    if model is not None:
        cfg = getattr(model, "config", None)
        if cfg is not None and getattr(cfg, "vocab_size", None):
            return int(cfg.vocab_size)
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                return m.num_embeddings
    return None


def estimate_params_from_config(cfg) -> int:
    """Rough parameter estimate from a HF config (used to pick a safe dtype)."""
    try:
        vocab = getattr(cfg, "vocab_size", None)
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
        layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
        if not vocab or not hidden or not layers:
            return 0
        intermediate = getattr(cfg, "intermediate_size", None) or getattr(cfg, "n_inner", None) or int(hidden * 4)
        per_block = 4 * int(hidden) * int(hidden) + 3 * int(hidden) * int(intermediate)
        return int(int(layers) * per_block + int(vocab) * int(hidden) * 2)
    except Exception:
        return 0


def load_model(model_name: str = DEFAULT_HF_MODEL, device: str = "cpu") -> tuple:
    """
    Returns (model, source). Falls back to our local TinyGPT if HF is missing.
    model_name may be:
      - DEFAULT_HF_MODEL  : load tiny random GPT-2 from HuggingFace Hub
      - LOCAL_TINYGPT     : local pyTorch tiny GPT
      - "distilgpt2", "gpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (real models)
      - any HF causal LM id
    The matching tokenizer and real vocab size are resolved via the registry
    (see load_tokenizer / resolve_model_vocab_size). Very large models are
    loaded directly in fp16 so CPU machines without huge RAM can run them.
    """
    try:
        from huggingface_hub import HuggingFaceHubError  # deferred
    except ImportError:
        class HuggingFaceHubError(Exception):
            pass

    if not model_name or not model_name.strip():
        model_name = LOCAL_TINYGPT

    if model_name == LOCAL_TINYGPT:
        model = TinyGPT()
        return model, model_name

    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        logger.warning("transformers not installed; using local TinyGPT.")
        model = TinyGPT()
        return model, LOCAL_TINYGPT

    try:
        cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
    except Exception:
        cfg = None
    if cfg is None:
        try:
            cfg = AutoConfig.from_pretrained(model_name)
        except Exception as e:
            logger.warning("Could not fetch HF config: %s; using local TinyGPT.", e)
            model = TinyGPT()
            return model, LOCAL_TINYGPT

    try:
        torch_dtype = None
        load_kwargs = {}
        est = estimate_params_from_config(cfg)
        if est > 350_000_000:
            # Big model (> ~350M params): load fp16 from the start to avoid a
            # full-size fp32 allocation (TinyLlama-1.1B fits this way).
            torch_dtype = torch.float16
            load_kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except Exception as e:
        logger.warning("Could not load '%s' from HF hub (%s); local TinyGPT instead.", model_name, e)
        model = TinyGPT()
        return model, LOCAL_TINYGPT

    return model, model_name