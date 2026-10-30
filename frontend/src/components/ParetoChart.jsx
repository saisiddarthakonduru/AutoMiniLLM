import React, { useState } from "react";

export default function ParetoChart({ experiments }) {
  const [metricY, setMetricY] = useState("perplexity"); // "perplexity" or "tokens_per_sec"
  const [hovered, setHovered] = useState(null);

  if (!experiments || experiments.length === 0) {
    return (
      <div className="empty-chart">
        <p>No experiment data available to plot Pareto frontier yet.</p>
      </div>
    );
  }

  // Filter valid data points
  const points = experiments
    .filter((e) => typeof e.ram_mb === "number" && typeof e[metricY] === "number" && e[metricY] > 0)
    .map((e) => ({
      id: e.id,
      label: e.label,
      mode: e.mode,
      x: e.ram_mb,
      y: e[metricY],
      speed: e.tokens_per_sec,
      ppl: e.perplexity,
    }));

  if (points.length === 0) {
    return <p className="empty">Run experiments to visualize the Pareto trade-off curve.</p>;
  }

  const width = 600;
  const height = 280;
  const pad = { top: 30, right: 30, bottom: 45, left: 60 };

  const xVals = points.map((p) => p.x);
  const yVals = points.map((p) => p.y);

  const minX = Math.min(...xVals) * 0.85;
  const maxX = Math.max(...xVals) * 1.15 || 1;
  const minY = Math.min(...yVals) * 0.85;
  const maxY = Math.max(...yVals) * 1.15 || 1;

  const scaleX = (val) => pad.left + ((val - minX) / (maxX - minX || 1)) * (width - pad.left - pad.right);
  const scaleY = (val) =>
    height - pad.bottom - ((val - minY) / (maxY - minY || 1)) * (height - pad.top - pad.bottom);

  // Compute Pareto optimal frontier
  // For Perplexity (lower X=RAM and lower Y=PPL is better)
  // For Tokens/sec (lower X=RAM and higher Y=Speed is better)
  const sorted = [...points].sort((a, b) => a.x - b.x);
  const paretoPoints = [];
  if (metricY === "perplexity") {
    let bestY = Infinity;
    for (const p of sorted) {
      if (p.y < bestY) {
        paretoPoints.push(p);
        bestY = p.y;
      }
    }
  } else {
    // For throughput: higher is better
    let bestY = -Infinity;
    for (const p of sorted) {
      if (p.y > bestY) {
        paretoPoints.push(p);
        bestY = p.y;
      }
    }
  }

  const pathD = paretoPoints.length > 1
    ? paretoPoints.map((p, i) => `${i === 0 ? "M" : "L"} ${scaleX(p.x)} ${scaleY(p.y)}`).join(" ")
    : "";

  const getColor = (mode) => {
    if (mode?.startsWith("prune")) return "#f43f5e"; // rose/red
    if (mode === "int4" || mode === "nf4") return "#10b981"; // emerald green
    if (mode === "int8" || mode === "fp8") return "#3b82f6"; // blue
    if (mode === "combined") return "#a855f7"; // purple
    return "#e2e8f0"; // slate
  };

  return (
    <div className="pareto-chart-container">
      <div className="chart-header">
        <span className="chart-title">
          Pareto Frontier: <b>RAM Footprint</b> vs <b>{metricY === "perplexity" ? "Perplexity (Quality)" : "Tokens/Sec (Speed)"}</b>
        </span>
        <div className="metric-toggle">
          <button
            type="button"
            className={metricY === "perplexity" ? "active-toggle" : ""}
            onClick={() => setMetricY("perplexity")}
          >
            Quality (PPL ↓)
          </button>
          <button
            type="button"
            className={metricY === "tokens_per_sec" ? "active-toggle" : ""}
            onClick={() => setMetricY("tokens_per_sec")}
          >
            Throughput (tok/s ↑)
          </button>
        </div>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} className="pareto-svg">
        {/* Grid lines */}
        <line
          x1={pad.left}
          y1={height - pad.bottom}
          x2={width - pad.right}
          y2={height - pad.bottom}
          stroke="#334155"
          strokeWidth="1"
        />
        <line
          x1={pad.left}
          y1={pad.top}
          x2={pad.left}
          y2={height - pad.bottom}
          stroke="#334155"
          strokeWidth="1"
        />

        {/* Pareto Frontier Line */}
        {pathD && (
          <path
            d={pathD}
            fill="none"
            stroke="#38bdf8"
            strokeWidth="2"
            strokeDasharray="4 4"
            opacity="0.8"
          />
        )}

        {/* Data points */}
        {points.map((p) => {
          const cx = scaleX(p.x);
          const cy = scaleY(p.y);
          const isPareto = paretoPoints.some((pt) => pt.id === p.id);
          const isHovered = hovered?.id === p.id;
          return (
            <g
              key={p.id}
              onMouseEnter={() => setHovered(p)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: "pointer" }}
            >
              {isPareto && (
                <circle
                  cx={cx}
                  cy={cy}
                  r="9"
                  fill="none"
                  stroke="#38bdf8"
                  strokeWidth="1.5"
                  opacity="0.6"
                />
              )}
              <circle
                cx={cx}
                cy={cy}
                r={isHovered ? 7 : 5}
                fill={getColor(p.mode)}
                stroke="#0f172a"
                strokeWidth="2"
              />
            </g>
          );
        })}

        {/* Axis labels */}
        <text
          x={width / 2}
          y={height - 8}
          textAnchor="middle"
          fill="#94a3b8"
          fontSize="11"
        >
          Model Tensor Footprint (MB) →
        </text>
        <text
          x={14}
          y={height / 2}
          textAnchor="middle"
          fill="#94a3b8"
          fontSize="11"
          transform={`rotate(-90 14 ${height / 2})`}
        >
          {metricY === "perplexity" ? "Perplexity (Lower is better) →" : "Throughput (Tokens/s) →"}
        </text>

        {/* Scale tick labels */}
        <text x={pad.left} y={height - pad.bottom + 16} fill="#64748b" fontSize="10">
          {minX.toFixed(1)}M
        </text>
        <text x={width - pad.right} y={height - pad.bottom + 16} textAnchor="end" fill="#64748b" fontSize="10">
          {maxX.toFixed(1)}M
        </text>
        <text x={pad.left - 8} y={pad.top + 4} textAnchor="end" fill="#64748b" fontSize="10">
          {maxY.toFixed(0)}
        </text>
        <text x={pad.left - 8} y={height - pad.bottom} textAnchor="end" fill="#64748b" fontSize="10">
          {minY.toFixed(0)}
        </text>
      </svg>

      {hovered && (
        <div className="pareto-tooltip">
          <b>{hovered.label}</b> ({hovered.mode})
          <div>RAM: {hovered.x.toFixed(2)} MB</div>
          <div>Perplexity: {hovered.ppl?.toFixed(2)}</div>
          <div>Throughput: {hovered.speed?.toFixed(1)} tok/s</div>
        </div>
      )}

      <div className="pareto-legend">
        <span><span className="legend-dot" style={{ background: "#3b82f6" }} /> INT8 / FP8</span>
        <span><span className="legend-dot" style={{ background: "#10b981" }} /> INT4 / NF4</span>
        <span><span className="legend-dot" style={{ background: "#f43f5e" }} /> Pruned (Sparse)</span>
        <span><span className="legend-dot" style={{ background: "#a855f7" }} /> Combined</span>
        <span><span className="legend-line" /> Pareto Frontier</span>
      </div>
    </div>
  );
}

