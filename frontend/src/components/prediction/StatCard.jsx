import React from "react";

export default function StatCard({
  icon,
  label,
  value,
  sub,
  iconBg = "var(--primary-dim)",
  iconColor = "var(--primary)",
  trend,      // number (positive = up, negative = down)
  animate = true,
  className = "",
}) {
  return (
    <div className={`stat-card ${animate ? "animate-up" : ""} ${className}`}>
      <div>
        <div className="card-title" style={{ marginBottom: "0.5rem" }}>{label}</div>
        <div className="card-value">{value ?? "—"}</div>
        {sub && <div className="card-sub">{sub}</div>}
        {trend !== undefined && (
          <div
            className="card-sub"
            style={{ color: trend >= 0 ? "var(--risk-low)" : "var(--risk-high)", marginTop: "0.35rem" }}
          >
            {trend >= 0 ? "▲" : "▼"} {Math.abs(trend)}% vs prev
          </div>
        )}
      </div>
      <div
        className="stat-icon"
        style={{ background: iconBg, color: iconColor, fontSize: "1.3rem" }}
      >
        {icon}
      </div>
    </div>
  );
}
