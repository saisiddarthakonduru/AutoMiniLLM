import React, { useEffect, useState, useCallback } from "react";
import {
  getHealth,
  getSystem,
  getModels,
  getHistory,
  compress,
  removeExperiment,
  clearHistory,
  compare,
} from "./api.js";
import BarChart from "./components/BarChart.jsx";
import Playground from "./components/Playground.jsx";
import ParetoChart from "./components/ParetoChart.jsx";

const COMPRESSION_MODES = [
  { value: "int8", label: "INT8 (Per-channel Symmetric)" },
  { value: "int4", label: "INT4 (Group-wise Packed 4-bit)" },
  { value: "nf4", label: "NF4 (NormalFloat4 Quantile — QLoRA)" },
  { value: "fp8", label: "FP8 (E4M3 8-bit Float — High Fidelity)" },
  { value: "fp16", label: "FP16 (Half Precision Float)" },
  { value: "fp32", label: "FP32 (Baseline Full Precision)" },
  { value: "prune", label: "Magnitude Pruning (Global Unstructured)" },
  { value: "prune_2_4", label: "2:4 Semi-Structured Pruning (NVIDIA Tensor Core)" },
  { value: "prune_channel", label: "Structured Channel Pruning (Neuron-level)" },
  { value: "combined", label: "Combined Prune + Quantize" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("experiments"); // "experiments" | "playground" | "pareto"
  const [system, setSystem] = useState(null);
  const [models, setModels] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [form, setForm] = useState({
    mode: "int8",
    prune_mode: "magnitude",
    sparsity: 0.5,
    group_size: 64,
    quant_mode: "int8",
    model: "",
    vocab_size: 50257,
    gen_token_len: 24,
    seq_len: 20,
    device: "",
  });
  const [selected, setSelected] = useState({ a: "", b: "" });
  const [comparison, setComparison] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const [sys, mods, hist] = await Promise.all([
        getSystem(),
        getModels(),
        getHistory(),
      ]);
      setSystem(sys);
      setModels(mods);
      setHistory(hist.experiments || []);
      setForm((f) => (f.model ? f : { ...f, model: mods.default || "" }));
    } catch (e) {
      setError(String(e.message || e));
    }
  }, []);

  useEffect(() => {
    getHealth().catch(() =>
      setError("Backend not reachable. Start it with: python -m uvicorn backend.app:app --port 8000")
    );
    refresh();
  }, [refresh]);

  const runCompress = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const payload = { ...form };
      if (!payload.device) delete payload.device;
      await getHistory();
      const res = await compress(payload);
      await refresh();
      setSelected((s) => ({ ...s, a: res.id }));
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  const doCompare = async () => {
    setError("");
    if (!selected.a || !selected.b) {
      setError("Select two experiments to compare (each may equal the other for self-check).");
      return;
    }
    try {
      setComparison(await compare(selected.a, selected.b));
    } catch (err) {
      setError(err.message || String(err));
    }
  };

  const fmt = (v, digits = 2) => {
    if (v === null || v === undefined) return "—";
    if (typeof v === "number") return v.toFixed(digits);
    return v;
  };

  const set = (k) => (e) => {
    const val = e.target.type === "number" ? parseFloat(e.target.value) : e.target.value;
    setForm((f) => ({ ...f, [k]: val }));
  };

  const onModelChange = (e) => {
    const value = e.target.value;
    const detail = models?.details?.find((m) => m.name === value) || null;
    if (detail && detail.vocab_size) {
      setForm((f) => ({ ...f, model: value, vocab_size: detail.vocab_size }));
    } else {
      setForm((f) => ({ ...f, model: value }));
    }
  };

  const sendToPlayground = (expId) => {
    setSelected((s) => ({ ...s, a: expId }));
    setActiveTab("playground");
  };

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-content">
          <h1>🧠 AutoMiniLLM</h1>
          <p className="subtitle">
            Next-Gen LLM Compression Platform: INT8, INT4, NF4, FP8, 2:4 Structured & Channel Pruning — with real measured throughput and quality.
          </p>
        </div>
        {system && (
          <div className="system-badge">
            <span className="dot" /> {system.device_label}
            <br />
            PyTorch {system.torch} · RAM {system.available_ram_gb} GB free
          </div>
        )}
      </header>

      {/* Navigation Bar */}
      <nav className="tab-nav">
        <button
          className={activeTab === "experiments" ? "tab-btn active" : "tab-btn"}
          onClick={() => setActiveTab("experiments")}
        >
          🔬 Experiments & Benchmarking
        </button>
        <button
          className={activeTab === "playground" ? "tab-btn active" : "tab-btn"}
          onClick={() => setActiveTab("playground")}
        >
          ⚡ Live Text Playground
        </button>
        <button
          className={activeTab === "pareto" ? "tab-btn active" : "tab-btn"}
          onClick={() => setActiveTab("pareto")}
        >
          📈 Pareto Frontier & Trade-offs
        </button>
      </nav>

      {error && <div className="error-banner">⚠ {error}</div>}

      {/* TAB 1: EXPERIMENTS & BENCHMARKING */}
      {activeTab === "experiments" && (
        <main className="grid">
          {/* Compress form */}
          <section className="card">
            <h2>⚙️ New Compression Experiment</h2>
            <form onSubmit={runCompress} className="form">
              <label>
                Compression Mode
                <select value={form.mode} onChange={set("mode")}>
                  {COMPRESSION_MODES.map((m) => (
                    <option key={m.value} value={m.value}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </label>

              {/* Dynamic Pruning Controls */}
              {(form.mode === "prune" || form.mode === "prune_channel") && (
                <label>
                  Sparsity Target: <b>{Math.round(form.sparsity * 100)}%</b>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={form.sparsity}
                    onChange={set("sparsity")}
                  />
                  <small>Zeroes out the lowest magnitude / least salient weights.</small>
                </label>
              )}

              {form.mode === "prune_2_4" && (
                <div className="info-callout">
                  ℹ️ <b>2:4 Semi-Structured:</b> Zeroes 2 out of every 4 weights (exact 50% sparsity) with NVIDIA Tensor Core acceleration.
                </div>
              )}

              {/* Dynamic Quantization Controls */}
              {(form.mode === "int4" || form.mode === "nf4") && (
                <label>
                  Group Size
                  <select value={form.group_size} onChange={set("group_size")}>
                    <option value={16}>16 (Higher precision, larger scales)</option>
                    <option value={32}>32</option>
                    <option value={64}>64 (Recommended default)</option>
                    <option value={128}>128 (Higher compression)</option>
                  </select>
                  <small>
                    {form.mode === "nf4"
                      ? "NF4 maps weights to 16 normal distribution quantiles (QLoRA)."
                      : "Symmetric 4-bit nibble packing (2 weights per byte)."}
                  </small>
                </label>
              )}

              {/* Combined Mode Controls */}
              {form.mode === "combined" && (
                <>
                  <label>
                    Pruning Strategy
                    <select value={form.prune_mode} onChange={set("prune_mode")}>
                      <option value="magnitude">Magnitude Pruning</option>
                      <option value="2_4">2:4 Semi-Structured (50%)</option>
                      <option value="channel">Structured Channel Pruning</option>
                    </select>
                  </label>

                  {form.prune_mode !== "2_4" && (
                    <label>
                      Pruning Sparsity: <b>{Math.round(form.sparsity * 100)}%</b>
                      <input
                        type="range"
                        min="0.1"
                        max="0.9"
                        step="0.05"
                        value={form.sparsity}
                        onChange={set("sparsity")}
                      />
                    </label>
                  )}

                  <label>
                    Quantization Format
                    <select value={form.quant_mode} onChange={set("quant_mode")}>
                      <option value="int8">INT8 (Per-channel)</option>
                      <option value="int4">INT4 (Packed 4-bit)</option>
                      <option value="nf4">NF4 (NormalFloat4 QLoRA)</option>
                      <option value="fp8">FP8 (E4M3 8-bit Float)</option>
                    </select>
                  </label>

                  {(form.quant_mode === "int4" || form.quant_mode === "nf4") && (
                    <label>
                      Quant Group Size
                      <input
                        type="number"
                        value={form.group_size}
                        onChange={set("group_size")}
                        min={16}
                        max={128}
                      />
                    </label>
                  )}
                </>
              )}

              <label>
                Model Architecture
                <select value={form.model} onChange={onModelChange}>
                  {models && models.details
                    ? models.details.map((m) => (
                        <option key={m.name} value={m.name}>
                          {m.label} — {m.name}
                        </option>
                      ))
                    : <option>Loading…</option>}
                </select>
                <small>Tokenized on real text corpus (WikiText-2 sample).</small>
              </label>

              <div className="form-row">
                <label className="flex-1">
                  Vocab Size
                  <input
                    type="number"
                    value={form.vocab_size}
                    onChange={set("vocab_size")}
                    min={100}
                    max={200000}
                  />
                </label>
                <label className="flex-1">
                  Generated Tokens
                  <input
                    type="number"
                    value={form.gen_token_len}
                    onChange={set("gen_token_len")}
                    min={4}
                    max={512}
                  />
                </label>
              </div>

              <label>
                Device (blank = auto)
                <input
                  placeholder="cuda / cpu / mps"
                  value={form.device}
                  onChange={set("device")}
                />
              </label>

              <button className="primary" disabled={loading}>
                {loading ? "Compressing & Measuring Live..." : "Run Compression Experiment"}
              </button>
            </form>
          </section>

          {/* System */}
          <section className="card">
            <h2>🖥 Hardware & Environment</h2>
            {system ? (
              <table className="kv">
                <tbody>
                  <tr><td>Device</td><td><b>{system.device_label}</b></td></tr>
                  <tr><td>RAM</td><td>{system.available_ram_gb} / {system.total_ram_gb} GB free</td></tr>
                  <tr><td>CPU cores</td><td>{system.cpu_count} logical ({system.cpu_physical} physical)</td></tr>
                  <tr><td>GPU</td><td>{system.gpu_name || "None (CPU Serving)"}</td></tr>
                  <tr><td>PyTorch</td><td>v{system.torch}</td></tr>
                  <tr><td>Python</td><td>{system.python}</td></tr>
                  <tr><td>Corpus</td><td>WikiText-2 sample (real English benchmark)</td></tr>
                </tbody>
              </table>
            ) : <p>Loading…</p>}
          </section>

          {/* History table */}
          <section className="card wide">
            <div className="card-head">
              <h2>📊 Experiment History & Measurements</h2>
              <div>
                <button
                  className="link"
                  onClick={async () => {
                    await clearHistory();
                    refresh();
                  }}
                >
                  Clear all
                </button>
              </div>
            </div>
            {history.length === 0 ? (
              <p className="empty">No experiments yet. Run one above to profile compression.</p>
            ) : (
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Label</th>
                      <th>Mode</th>
                      <th>RAM (MB)</th>
                      <th>RAM Ratio</th>
                      <th>File (MB)</th>
                      <th>File Ratio</th>
                      <th>PPL ↓</th>
                      <th>KL Drift</th>
                      <th>tok/s ↑</th>
                      <th>ITL (ms/tok)</th>
                      <th>TTFT (ms)</th>
                      <th>Sparsity</th>
                      <th>Action</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((h) => (
                      <tr
                        key={h.id}
                        className={selected.a === h.id || selected.b === h.id ? "selected-row" : ""}
                      >
                        <td className="label-cell"><b>{h.label}</b></td>
                        <td><span className="mode-badge">{h.mode}</span></td>
                        <td>{fmt(h.ram_mb)}</td>
                        <td><b style={{ color: "#38bdf8" }}>{fmt(h.ram_compression_ratio)}×</b></td>
                        <td>{fmt(h.file_size_mb)}</td>
                        <td>{fmt(h.file_compression_ratio)}×</td>
                        <td><b>{fmt(h.perplexity)}</b></td>
                        <td>{fmt(h.kl_vs_baseline, 4)}</td>
                        <td><b style={{ color: "#10b981" }}>{fmt(h.tokens_per_sec)}</b></td>
                        <td>{fmt(h.ms_per_token, 2)}</td>
                        <td>{fmt(h.ttft_ms || h.prefill_latency_ms, 1)}</td>
                        <td>{fmt(h.achieved_sparsity, 1)}%</td>
                        <td>
                          <div className="action-cell">
                            <button
                              type="button"
                              className="chip-btn"
                              title="Test generation in Playground"
                              onClick={() => sendToPlayground(h.id)}
                            >
                              ▶ Test
                            </button>
                            <a
                              href={`/api/export/${h.id}`}
                              className="chip-btn export-btn"
                              title="Download compressed .pt state"
                              download
                            >
                              ⬇ .pt
                            </a>
                          </div>
                        </td>
                        <td>
                          <button
                            className="btn danger"
                            onClick={async () => {
                              await removeExperiment(h.id);
                              refresh();
                            }}
                          >
                            ✕
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <div className="compare-bar">
              <input
                placeholder="Experiment A id"
                value={selected.a}
                onChange={(e) => setSelected((s) => ({ ...s, a: e.target.value }))}
              />
              <input
                placeholder="Experiment B id"
                value={selected.b}
                onChange={(e) => setSelected((s) => ({ ...s, b: e.target.value }))}
              />
              <button className="btn" onClick={doCompare}>
                Compare A vs B
              </button>
            </div>
          </section>

          {/* Bar Charts */}
          <section className="card">
            <h2>Compression Ratio (RAM Savings)</h2>
            <BarChart experiments={history} valueKey="ram_compression_ratio" color="#4f8cff" />
          </section>

          <section className="card">
            <h2>Throughput (tokens/sec) Across Checkpoints</h2>
            <BarChart experiments={history} valueKey="tokens_per_sec" color="#10b981" />
          </section>

          {/* Comparison table */}
          {comparison && (
            <section className="card wide">
              <h2>🔁 Comparison: {comparison.exp_a.label} vs {comparison.exp_b.label}</h2>
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>A ({comparison.exp_a.label})</th>
                    <th>B ({comparison.exp_b.label})</th>
                    <th>Δ% (B vs A)</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.rows.map((r) => (
                    <tr key={r.metric}>
                      <td><b>{r.metric}</b></td>
                      <td>{fmt(r.exp_a)}</td>
                      <td>{fmt(r.exp_b)}</td>
                      <td>
                        {r.pct_change_b_vs_a !== null ? (
                          <span
                            style={{
                              color:
                                r.pct_change_b_vs_a > 0
                                  ? r.metric.includes("perplexity") ? "#f43f5e" : "#10b981"
                                  : r.metric.includes("perplexity") ? "#10b981" : "#f43f5e",
                            }}
                          >
                            {(r.pct_change_b_vs_a >= 0 ? "+" : "") + r.pct_change_b_vs_a + "%"}
                          </span>
                        ) : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}
        </main>
      )}

      {/* TAB 2: INTERACTIVE PLAYGROUND */}
      {activeTab === "playground" && (
        <main className="grid single-col">
          <Playground history={history} models={models} />
        </main>
      )}

      {/* TAB 3: PARETO FRONTIER & TRADE-OFFS */}
      {activeTab === "pareto" && (
        <main className="grid single-col">
          <section className="card">
            <h2>📈 Pareto Optimal Frontier</h2>
            <p className="subtitle">
              Visualizes the trade-off curve between Model Size (RAM Footprint) vs Quality (Perplexity) and Speed (tokens/sec).
            </p>
            <ParetoChart experiments={history} />
          </section>
        </main>
      )}
    </div>
  );
}
