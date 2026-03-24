import React from "react";

const RISK_CONFIG = {
  low:      { label: "Low",      color: "var(--risk-low)",      bg: "var(--risk-low-dim)",      pct: 20 },
  medium:   { label: "Medium",   color: "var(--risk-medium)",   bg: "var(--risk-medium-dim)",   pct: 50 },
  high:     { label: "High",     color: "var(--risk-high)",     bg: "var(--risk-high-dim)",     pct: 75 },
  critical: { label: "Critical", color: "var(--risk-critical)", bg: "var(--risk-critical-dim)", pct: 100 },
  moderate: { label: "Medium",   color: "var(--risk-medium)",   bg: "var(--risk-medium-dim)",   pct: 50 },
  extreme:  { label: "Critical", color: "var(--risk-critical)", bg: "var(--risk-critical-dim)", pct: 100 },
};

export default function RiskBadge({ level, showBar = false, className = "" }) {
  const key = (level || "low").toLowerCase();
  const cfg = RISK_CONFIG[key] || RISK_CONFIG.low;

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <span
        className="badge badge-dot"
        style={{ background: cfg.bg, color: cfg.color, border: `1px solid ${cfg.color}40` }}
      >
        {cfg.label} Risk
      </span>
      {showBar && (
        <div className="risk-bar-track" style={{ width: 120 }}>
          <div
            className={`risk-bar-fill ${key}`}
            style={{ width: `${cfg.pct}%` }}
          />
        </div>
      )}
    </div>
  );
}
