import React from "react";
import AlertList from "../components/alerts/AlertList";
import AlertSubscribe from "../components/alerts/AlertSubscribe";
import { useState } from "react";

export default function AlertsPage() {
  const [activeSection, setActiveSection] = useState("manage");

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">🔔 Alert System</h1>
        <p className="page-subtitle">
          Multi-channel flood alerts via email, SMS, push notifications, and webhooks
        </p>
      </div>

      {/* Section toggles */}
      <div className="tabs mb-6" style={{ width: "fit-content" }}>
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

      {/* Manage section */}
      {activeSection === "manage" && (
        <AlertList />
      )}

      {/* Subscribe section */}
      {activeSection === "subscribe" && (
        <div className="grid-2" style={{ alignItems: "start", gap: "1.5rem" }}>
          <AlertSubscribe onSubscribed={() => setActiveSection("manage")} />

          <div>
            {/* Channel info cards */}
            <div className="section-title mb-4">Available Channels</div>
            <div className="flex flex-col gap-3">
              {[
                { icon: "📧", name: "Email", desc: "Detailed HTML alerts with maps and action checklists", status: "Available" },
                { icon: "📱", name: "SMS",   desc: "Concise 160-character safety messages via Twilio",    status: "Available" },
                { icon: "🔔", name: "Push",  desc: "Instant push notifications via Firebase Cloud Messaging", status: "Available" },
                { icon: "🔗", name: "Webhook",desc: "JSON payload to custom endpoints (Slack, Discord, dashboards)", status: "Configure" },
              ].map((ch) => (
                <div
                  key={ch.name}
                  className="card"
                  style={{ padding: "1rem", display: "flex", gap: "1rem", alignItems: "flex-start" }}
                >
                  <div style={{
                    fontSize: "1.5rem", width: 40, height: 40,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    background: "var(--primary-dim)", borderRadius: "var(--radius-sm)",
                    flexShrink: 0,
                  }}>
                    {ch.icon}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-semibold" style={{ fontSize: "0.875rem" }}>{ch.name}</span>
                      <span className="badge badge-primary">{ch.status}</span>
                    </div>
                    <p className="text-muted text-xs">{ch.desc}</p>
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
