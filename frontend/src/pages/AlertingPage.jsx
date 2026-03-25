/**
 * AlertingPage.jsx
 * Shows alerting results from the AI pipeline (from PipelineContext).
 * Also includes the existing alert subscriber management tabs.
 */
import React, { useState } from "react";
import { Link } from "react-router-dom";
import { usePipeline } from "../context/PipelineContext";
import AlertList from "../components/alerts/AlertList";
import AlertSubscribe from "../components/alerts/AlertSubscribe";

// ── Pipeline alert result panel ───────────────────────────────────────────────
function PipelineAlertResult({ data, prediction }) {
  const color = data.sent_count > 0 ? "var(--risk-low)" : "var(--text-muted)";
  const riskColor = {
    HIGH: "var(--risk-high)", MEDIUM: "var(--risk-medium)",
    LOW: "var(--risk-low)",   CRITICAL: "var(--risk-critical)",
  }[prediction?.risk_level?.toUpperCase()] || "var(--primary)";

  return (
    <div className="card animate-fade" style={{ borderTop: `3px solid ${color}` }}>
      <div className="card-title mb-4">🔔 Pipeline Alert Result</div>

      {/* Context from prediction */}
      {prediction?.risk_level && (
        <div style={{ marginBottom: "1rem", display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>TRIGGERED BY</span>
          <span className="badge" style={{
            background: riskColor + "22", color: riskColor,
            border: `1px solid ${riskColor}44`, fontWeight: 700, fontSize: "0.7rem",
          }}>
            {prediction.risk_level.toUpperCase()} RISK PREDICTION
          </span>
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: "1rem", marginBottom: "1rem" }}>
        {[
          { label: "Alerts Sent",  value: data.sent_count ?? "—",                      icon: "📤" },
          { label: "Subscribers",  value: data.total_subscribers ?? "—",               icon: "👥" },
          { label: "Channels",     value: data.channels_used?.join(", ") || "—",       icon: "📡" },
          { label: "Severity",     value: data.severity || "—",                        icon: "⚠️" },
        ].map(({ label, value, icon }) => (
          <div key={label} style={{
            background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)",
            padding: "0.875rem", textAlign: "center",
          }}>
            <div style={{ fontSize: "1.25rem", marginBottom: "0.25rem" }}>{icon}</div>
            <div style={{ fontWeight: 700, fontSize: "1rem", color: color, marginBottom: "0.15rem" }}>{value}</div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>{label}</div>
          </div>
        ))}
      </div>

      {data.summary && (
        <p style={{
          fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.7,
          background: "var(--bg-elevated)", padding: "0.75rem",
          borderRadius: "var(--radius-sm)", marginBottom: 0,
        }}>
          {data.summary}
        </p>
      )}
    </div>
  );
}

// ── Main Alerting Page ────────────────────────────────────────────────────────
export default function AlertingPage() {
  const { alertStatus, prediction, running } = usePipeline();
  const [activeSection, setActiveSection] = useState("pipeline");

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">🔔 Alerting</h1>
        <p className="page-subtitle">
          Pipeline-dispatched flood alerts and subscriber management
        </p>
      </div>

      {/* Section tabs */}
      <div className="tabs mb-6" style={{ width: "fit-content" }}>
        <button
          className={`tab${activeSection === "pipeline" ? " active" : ""}`}
          onClick={() => setActiveSection("pipeline")}
        >
          🚀 Pipeline Alerts
        </button>
        <button
          className={`tab${activeSection === "manage" ? " active" : ""}`}
          onClick={() => setActiveSection("manage")}
        >
          🚨 Manage Alerts
        </button>
        <button
          className={`tab${activeSection === "subscribe" ? " active" : ""}`}
          onClick={() => setActiveSection("subscribe")}
        >
          📲 Subscribe
        </button>
      </div>

      {/* ── Pipeline Alerts Tab ── */}
      {activeSection === "pipeline" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          {running && (
            <div className="alert alert-info" style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <span className="spinner spinner-sm" />
              <div className="alert-content">
                <div className="alert-message">Pipeline is running — alerting results will appear here automatically.</div>
              </div>
            </div>
          )}

          {!alertStatus && !running && (
            <div className="card" style={{ textAlign: "center", padding: "4rem 2rem" }}>
              <div style={{ fontSize: "3.5rem", marginBottom: "1rem" }}>🔔</div>
              <div style={{ fontWeight: 700, fontSize: "1.1rem", marginBottom: "0.5rem" }}>
                No Alert Data Yet
              </div>
              <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", marginBottom: "1.5rem", maxWidth: 380, margin: "0 auto 1.5rem" }}>
                Run the prediction pipeline first — the alerting agent dispatches notifications to subscribers automatically.
              </p>
              <Link to="/predict" className="btn btn-primary">
                🚀 Go to Prediction →
              </Link>
            </div>
          )}

          {alertStatus && (
            <>
              <PipelineAlertResult data={alertStatus} prediction={prediction} />

              <div className="card" style={{ borderLeft: "3px solid var(--accent)", padding: "1rem 1.25rem" }}>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600, marginBottom: "0.5rem", letterSpacing: "0.05em" }}>
                  CHANNEL INFORMATION
                </div>
                <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                  {["📧 Email", "📱 SMS", "🔔 Push", "🔗 Webhook"].map((ch) => (
                    <span key={ch} className="badge badge-primary" style={{ fontSize: "0.72rem" }}>{ch}</span>
                  ))}
                </div>
                <p style={{ fontSize: "0.775rem", color: "var(--text-muted)", marginTop: "0.5rem", marginBottom: 0 }}>
                  Alerts are dispatched to all subscribed channels. Manage subscriptions in the Subscribe tab.
                </p>
              </div>

              <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
                <Link to="/predict" className="btn btn-secondary" style={{ fontSize: "0.8rem" }}>
                  🧠 Back to Prediction →
                </Link>
                <Link to="/recommendations" className="btn btn-secondary" style={{ fontSize: "0.8rem" }}>
                  💡 View Recommendations →
                </Link>
              </div>
            </>
          )}
        </div>
      )}

      {/* ── Manage Alerts Tab ── */}
      {activeSection === "manage" && <AlertList />}

      {/* ── Subscribe Tab ── */}
      {activeSection === "subscribe" && (
        <div className="grid-2" style={{ alignItems: "start", gap: "1.5rem" }}>
          <AlertSubscribe onSubscribed={() => setActiveSection("manage")} />

          <div>
            <div className="section-title mb-4">Available Channels</div>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
              {[
                { icon: "📧", name: "Email",   desc: "Detailed HTML alerts with maps and action checklists",          status: "Available" },
                { icon: "📱", name: "SMS",     desc: "Concise 160-character safety messages via Twilio",              status: "Available" },
                { icon: "🔔", name: "Push",    desc: "Instant push notifications via Firebase Cloud Messaging",       status: "Available" },
                { icon: "🔗", name: "Webhook", desc: "JSON payload to custom endpoints (Slack, Discord, dashboards)", status: "Configure" },
              ].map((ch) => (
                <div key={ch.name} className="card" style={{ padding: "1rem", display: "flex", gap: "1rem", alignItems: "flex-start" }}>
                  <div style={{
                    fontSize: "1.5rem", width: 40, height: 40,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    background: "var(--primary-dim)", borderRadius: "var(--radius-sm)",
                    flexShrink: 0,
                  }}>
                    {ch.icon}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.25rem" }}>
                      <span style={{ fontWeight: 600, fontSize: "0.875rem" }}>{ch.name}</span>
                      <span className="badge badge-primary">{ch.status}</span>
                    </div>
                    <p style={{ color: "var(--text-muted)", fontSize: "0.775rem", margin: 0 }}>{ch.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
