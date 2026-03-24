import React, { useState } from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "var(--bg-elevated)", border: "1px solid var(--border)",
      borderRadius: "var(--radius-sm)", padding: "0.6rem 0.875rem", fontSize: "0.8rem",
    }}>
      <div style={{ color: "var(--text-muted)", marginBottom: "0.3rem" }}>{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color, fontWeight: 600 }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(2) : p.value}
        </div>
      ))}
    </div>
  );
};

export default function SimulationChart({ timeline = [], peakHour }) {
  const [activeKey, setActiveKey] = useState("water_level_m");

  const SERIES = [
    { key: "water_level_m", label: "Water Level (m)",   color: "#38bdf8" },
    { key: "rainfall_mm",   label: "Rainfall (mm/h)",   color: "#818cf8" },
    { key: "area_km2",      label: "Inundated Area km²", color: "#ef4444" },
  ];

  const active = SERIES.find((s) => s.key === activeKey);

  return (
    <div className="card animate-up">
      <div className="card-header">
        <h3 className="card-title">📈 Flood Timeline</h3>
        <div className="tabs" style={{ width: "auto" }}>
          {SERIES.map((s) => (
            <button
              key={s.key}
              className={`tab${activeKey === s.key ? " active" : ""}`}
              onClick={() => setActiveKey(s.key)}
            >
              {s.label.split(" ")[0]}
            </button>
          ))}
        </div>
      </div>

      {timeline.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">📉</div>
          <div className="empty-text">Run a simulation to see the timeline.</div>
        </div>
      ) : (
        <div className="chart-container" style={{ height: 220 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={timeline} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={active.color} stopOpacity={0.4} />
                  <stop offset="95%" stopColor={active.color} stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="hour"
                     tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                     tickFormatter={(h) => `h${h}`}
                     axisLine={{ stroke: "var(--border)" }} tickLine={false} />
              <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                     axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              {peakHour != null && (
                <ReferenceLine x={peakHour} stroke="var(--risk-high)" strokeDasharray="4 2"
                               label={{ value: "Peak", fill: "var(--risk-high)", fontSize: 11 }} />
              )}
              <Area
                type="monotone" dataKey={activeKey} name={active.label}
                stroke={active.color} strokeWidth={2}
                fill="url(#grad)" dot={false} activeDot={{ r: 4 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
