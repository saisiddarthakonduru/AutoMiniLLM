# 🧠 AutoMiniLLM — Efficient LLM Compression Using Quantization and Pruning

A complete, production-grade LLM-compression and inference benchmarking system built with **Python, PyTorch, Hugging Face Transformers, FastAPI, and React**.

AutoMiniLLM compresses real causal language models and reports **measured** (not hardcoded) metrics: model size, memory usage, compression ratio, sparsity, inference latency, tokens/sec, TTFT (time-to-first-token), perplexity on real text, and KL divergence versus the FP32 baseline.

---

## ✨ Features & Compression Algorithms

| Technique | Description | Details |
|---|---|---|
| **FP32 / FP16** | Native precision & half precision | Direct dtype conversion |
| **FP8 (Float8 E4M3)** | Native 8-bit float (`torch.float8_e4m3fn`) | High-fidelity floating-point representation with dynamic range |
| **INT8 Quantization** | Per-channel symmetric quantization | Signed int8 storage with per-row scaling |
| **INT4 Quantization** | Group-wise symmetric (group 16-128) | Real 4-bit nibble packed storage (2 weights per uint8 byte) |
| **NF4 (NormalFloat4)** | QLoRA quantile mapping | 16 optimal normal-distribution quantiles packed into 4-bit storage |
| **Magnitude Pruning** | Global magnitude thresholding | COO sparse buffer storage (`indices` + `values` fp16) |
| **2:4 Semi-Structured Pruning** | NVIDIA Tensor Core compatible | Guarantees exactly 2 zeroed weights in every 4-element block |
| **Channel Pruning** | Structured neuron-level pruning | Prunes least-salient output rows by L1 norm |
| **Combined Prune + Quantize** | Multi-stage compression | Sparse + quantized layers (`QuantizedSparseLinear`) |
| **Live Playground** | Interactive text generation | Test prompts directly on compressed checkpoints with live tok/s & latency |
| **Pareto Frontier** | 2D Trade-off analysis | Interactive SVG chart visualizing RAM footprint vs Perplexity & Throughput |
| **Real Evaluation** | WikiText-2 real English sample | Real tokenized text loss & PPL alongside TTFT and ITL profiling |
| **Exportable Artifacts** | Checkpoint export (`/api/export/{id}`) | Download compressed weights (`.pt`) and architecture metadata |

---

## 📀 Model Options

Real pretrained models from HuggingFace Hub (auto-detects tokenizer and vocab size):

| Model id | Size | Vocab | Notes |
|---|---|---|---|
| `hf-internal-testing/tiny-random-gpt2` | ~2 MB | 1,000 | Ultra-fast local testing & CI/CD |
| `distilgpt2` | 82M params | 50,257 | Distilled OpenAI GPT-2 |
| `gpt2` | 124M params | 50,257 | Standard OpenAI GPT-2 base |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B params | 32,000 | 1.1B LLaMA chat model |
| `LocalTinyGPT` | Pure PyTorch | 2,000 | Zero-download offline fallback |

---

## 📁 Project Layout

```
AutoMiniLLM/
├─ backend/
│  ├─ app.py                 # FastAPI application, REST endpoints, and model reconstruction
│  ├─ history.py             # JSON experiment persistence
│  └─ core/
│     ├─ quantization.py     # FP16 / FP8 / INT8 / INT4 / NF4 quantization engine
│     ├─ pruning.py          # Magnitude, 2:4 semi-structured, and channel pruning
│     ├─ pipeline.py         # Compression pipeline, QuantizedSparseLinear, metadata
│     ├─ evaluator.py        # WikiText-2 PPL, KL drift, TTFT, ITL, throughput
│     ├─ models.py           # HF loader + LocalTinyGPT causal LM fallback
│     └─ utils.py            # Device detection, RAM profiling, benchmark timing
├─ datasets/
│  └─ wikitext_sample.txt    # Real English evaluation corpus sample
├─ frontend/                 # React + Vite dashboard
│  ├─ src/
│  │  ├─ components/
│  │  │  ├─ BarChart.jsx     # SVG Metric bar charts
│  │  │  ├─ Playground.jsx   # Interactive prompt testing UI
│  │  │  └─ ParetoChart.jsx  # Interactive Pareto Frontier scatter plot
│  │  ├─ api.js              # REST client
│  │  ├─ App.jsx             # Main dashboard with tabs & dynamic forms
│  │  └─ styles.css          # Dark-mode styling
├─ experiments/              # Saved model checkpoints + JSON history
├─ scripts/
│  ├─ run_experiments.py     # CLI benchmark runner
│  └─ seed_demo.py           # Seeds all compression modes into the dashboard
├─ tests/
│  ├─ test_quantization.py   # Pack/unpack, INT8, FP8, INT4, NF4 tests
│  ├─ test_pruning.py        # SparseLinear, 2:4, channel pruning tests
│  ├─ test_pipeline.py       # End-to-end pipeline, combined modes, history tests
│  └─ test_models.py         # Registry, tokenizers, Conv1D conversion tests
├─ requirements.txt
└─ README.md
```

---

## ▶️ Running AutoMiniLLM

### 1. Install Dependencies
```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### 2. Start the Backend (FastAPI)
```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Start the Frontend (React + Vite)
```bash
cd frontend
npm run dev
```
Open **http://localhost:5173** in your browser.

### 4. Seed Demo Checkpoints (Optional)
```bash
python scripts/seed_demo.py
```

### 5. Run the Automated Tests
```bash
python tests/test_quantization.py
python tests/test_pruning.py
python tests/test_pipeline.py
python tests/test_models.py
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health status |
| `GET` | `/api/system` | Hardware detection (CPU/GPU, RAM, PyTorch version) |
| `GET` | `/api/models` | Available model registry & supported modes |
| `POST` | `/api/compress` | Run a new compression experiment |
| `GET` | `/api/history` | List all historical experiments |
| `GET` | `/api/history/{id}` | Inspect one experiment |
| `DELETE` | `/api/history/{id}` | Delete one experiment |
| `POST` | `/api/compare` | Compare two experiments (`{"exp_a": id, "exp_b": id}`) |
| `POST` | `/api/generate` | Generate text using a saved compressed checkpoint |
| `GET` | `/api/export/{id}` | Download compressed model state file |

---

## License

MIT — free to use, modify, and distribute.
