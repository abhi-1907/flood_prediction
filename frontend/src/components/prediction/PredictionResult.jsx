import React from "react";
import {
  RadialBarChart, RadialBar, PolarAngleAxis,
  ResponsiveContainer, Tooltip,
} from "recharts";
import RiskBadge from "./RiskBadge";

const LEVEL_COLOR = {
  low:      "var(--risk-low)",
  medium:   "var(--risk-medium)",
  high:     "var(--risk-high)",
  critical: "var(--risk-critical)",
  moderate: "var(--risk-medium)",
  extreme:  "var(--risk-critical)",
};

export default function PredictionResult({ result }) {
  if (!result) return null;

  const level    = (result.risk_level || "low").toLowerCase();
  const prob     = Math.round((result.flood_probability ?? 0) * 100);
  const conf     = Math.round((result.confidence ?? 0) * 100);
  const color    = LEVEL_COLOR[level] || "var(--risk-low)";

  const radialData = [{ value: prob, fill: color }];

  return (
    <div className="card animate-scale" style={{ marginTop: "1.5rem" }}>
      <div className="card-header">
        <h3 className="card-title">📊 Prediction Result</h3>
        <RiskBadge level={result.risk_level} showBar />
      </div>

      <div className="grid-3" style={{ gap: "1.5rem", alignItems: "center" }}>
        {/* Radial gauge */}
        <div style={{ textAlign: "center" }}>
          <div style={{ height: 160, position: "relative" }}>
            <ResponsiveContainer width="100%" height="100%">
              <RadialBarChart
                cx="50%" cy="50%" innerRadius="70%" outerRadius="100%"
                data={radialData} startAngle={90} endAngle={-270}
              >
                <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
                <RadialBar
                  dataKey="value" cornerRadius={10}
                  background={{ fill: "var(--bg-elevated)" }}
                />
              </RadialBarChart>
            </ResponsiveContainer>
            <div style={{
              position: "absolute", inset: 0,
              display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center",
              pointerEvents: "none",
            }}>
              <span style={{ fontSize: "2rem", fontWeight: 800, color, lineHeight: 1 }}>
                {prob}%
              </span>
              <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
                Flood Probability
              </span>
            </div>
          </div>
        </div>

        {/* Key metrics */}
        <div className="flex flex-col gap-3">
          <MetricRow label="Risk Level"   value={result.risk_level || "—"}   color={color} />
          <MetricRow label="Confidence"   value={`${conf}%`} />
          <MetricRow label="Models Used"  value={result.models_used?.length ?? 0} />
          <MetricRow label="Session ID"
            value={result.session_id ? result.session_id.slice(0, 12) + "…" : "—"}
          />
        </div>

        {/* Explanation */}
        <div>
          <div className="section-title" style={{ marginBottom: "0.6rem" }}>
            🔍 AI Explanation
          </div>
          <p style={{ fontSize: "0.825rem", color: "var(--text-secondary)", lineHeight: 1.7 }}>
            {result.explanation || "No explanation available."}
          </p>
          {result.warnings?.length > 0 && (
            <div className="alert alert-warning" style={{ marginTop: "0.75rem" }}>
              <span className="alert-icon">⚠️</span>
              <div className="alert-content">
                <div className="alert-title">Warnings</div>
                <div className="alert-message">{result.warnings.join(", ")}</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {result.models_used?.length > 0 && (
        <>
          <div className="divider" />
          <div className="section-title">Models Used</div>
          <div className="flex gap-2 flex-wrap">
            {result.models_used.map((m) => (
              <span key={m} className="badge badge-neutral">{m}</span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function MetricRow({ label, value, color }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <span style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>{label}</span>
      <span
        style={{
          fontSize: "0.875rem", fontWeight: 700,
          color: color || "var(--text-primary)",
        }}
      >
        {value}
      </span>
    </div>
  );
}
