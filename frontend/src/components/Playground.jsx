import React, { useState } from "react";
import { generateText } from "../api.js";

const PROMPT_PRESETS = [
  "The future of AI is",
  "Once upon a time in a digital world,",
  "Model compression enables",
  "In deep learning, quantization",
];

export default function Playground({ history, models }) {
  const [selectedExpId, setSelectedExpId] = useState("");
  const [prompt, setPrompt] = useState(PROMPT_PRESETS[0]);
  const [maxTokens, setMaxTokens] = useState(64);
  const [temperature, setTemperature] = useState(0.8);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  React.useEffect(() => {
    if (!selectedExpId && history && history.length > 0) {
      const preferred = history.find((h) => h.model_name === "distilgpt2") || history[0];
      if (preferred) {
        setSelectedExpId(preferred.id);
      }
    }
  }, [history, selectedExpId]);

  const handleGenerate = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    const exp = history.find((h) => h.id === selectedExpId);

    const payload = {
      prompt,
      max_new_tokens: Number(maxTokens),
      temperature: Number(temperature),
      saved_path: exp?.saved_path || "",
      exp_id: selectedExpId || undefined,
      model_name: exp?.model_name || models?.default || "LocalTinyGPT",
    };

    try {
      const res = await generateText(payload);
      setResult(res);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  const selectedExp = history.find((h) => h.id === selectedExpId);

  return (
    <section className="card playground-card">
      <div className="card-head">
        <div>
          <h2>⚡ Interactive Model Playground</h2>
          <p className="subtitle">
            Generate text from your compressed model checkpoints live to verify linguistic quality and latency.
          </p>
        </div>
      </div>

      {error && <div className="error-banner">⚠ {error}</div>}

      <form onSubmit={handleGenerate} className="playground-form">
        <div className="form-row">
          <label className="flex-1">
            Target Compressed Checkpoint
            <select
              value={selectedExpId}
              onChange={(e) => setSelectedExpId(e.target.value)}
            >
              <option value="">Default Baseline (LocalTinyGPT / Uncompressed)</option>
              {history.map((h) => (
                <option key={h.id} value={h.id}>
                  {h.label} ({h.mode}) — PPL: {h.perplexity} — RAM: {h.ram_mb} MB ({h.model_name || "tiny"})
                </option>
              ))}
            </select>
          </label>

          <label style={{ width: "170px" }}>
            Max Tokens: <b>{maxTokens}</b>
            <input
              type="range"
              min="8"
              max="256"
              step="8"
              value={maxTokens}
              onChange={(e) => setMaxTokens(e.target.value)}
            />
          </label>

          <label style={{ width: "160px" }}>
            Temp: <b>{temperature}</b>
            <input
              type="range"
              min="0.2"
              max="1.5"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(e.target.value)}
            />
          </label>
        </div>

        <label>
          Input Prompt
          <div className="preset-buttons">
            {PROMPT_PRESETS.map((p) => (
              <button
                key={p}
                type="button"
                className="chip-btn"
                onClick={() => setPrompt(p)}
              >
                "{p}"
              </button>
            ))}
          </div>
          <textarea
            rows="3"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Type your generation prompt here..."
            required
          />
        </label>

        <button className="primary" disabled={loading}>
          {loading ? "Generating tokens..." : "Generate Text"}
        </button>
      </form>

      {result && (
        <div className="generation-output-box">
          <div className="output-header">
            <span className="output-model-badge">
              Model: {selectedExp ? selectedExp.label : "Baseline"} ({result.model_name || "loaded"})
            </span>
            <div className="output-stats">
              <span>⚡ <b>{result.tokens_per_sec}</b> tok/s</span>
              <span>⏱ <b>{result.elapsed_ms}</b> ms</span>
              <span>🔢 <b>{result.tokens_generated}</b> tokens</span>
            </div>
          </div>
          <div className="output-content">
            <span className="prompt-echo">{result.prompt}</span>{" "}
            <span className="generated-text">{result.generated_text || "(Tokens generated successfully)"}</span>
          </div>
        </div>
      )}
    </section>
  );
}
