import React from "react";

/**
 * Minimal dependency-free grouped bar chart rendered with SVG.
 * Used for compression-ratio comparisons across experiments.
 */
export default function BarChart({ experiments, valueKey, labelKey = "label", color = "#4f8cff", height = 220 }) {
  if (!experiments || experiments.length === 0) {
    return <div className="empty-chart">No data yet — run a compression experiment.</div>;
  }

  const values = experiments
    .map((e) => ({ label: e[labelKey] || "?", value: e[valueKey] }))
    .filter((d) => typeof d.value === "number" && d.value > 0);

  if (values.length === 0) {
    return <div className="empty-chart">No numeric values for “{valueKey}”.</div>;
  }

  const maxV = Math.max(...values.map((d) => d.value));
  const pad = 30;
  const chartH = height - 40;
  const bw = 56;
  const gap = 16;
  const totalW = Math.max(320, values.length * (bw + gap) + 20);

  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${totalW} ${height}`} style={{ width: "100%", height }}>
        {values.map((d, i) => {
          const h = maxV > 0 ? (d.value / maxV) * chartH : 0;
          const x = 10 + i * (bw + gap);
          const y = chartH - h;
          return (
            <g key={i}>
              <rect x={x} y={y} width={bw} height={h} fill={color} rx={4} opacity={0.9}>
                <title>{`${d.label}: ${d.value}`}</title>
              </rect>
              <text x={x + bw / 2} y={y - 5} textAnchor="middle" className="bar-value">
                {typeof d.value === "number" ? d.value.toFixed(2) : d.value}
              </text>
              <text x={x + bw / 2} y={chartH + 15} textAnchor="middle" className="bar-label">
                {String(d.label).slice(-14)}
              </text>
            </g>
          );
        })}
        <line x1={0} y1={chartH} x2={totalW} y2={chartH} className="axis" />
      </svg>
    </div>
  );
}
