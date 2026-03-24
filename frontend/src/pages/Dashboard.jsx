import React, { useState, useEffect } from "react";
import { healthCheck, getActiveAlerts, getSubscribers, quickPredict } from "../api/floodApi";
import StatCard from "../components/prediction/StatCard";
import RiskBadge from "../components/prediction/RiskBadge";
import { Link } from "react-router-dom";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

const RISK_COLORS = {
  low: "var(--risk-low)", medium: "var(--risk-medium)",
  high: "var(--risk-high)", critical: "var(--risk-critical)",
};

const INDIA_WEATHER = [
  { region: "Assam",    risk: "high",     rainfall: 220 },
  { region: "Kerala",   risk: "high",     rainfall: 180 },
  { region: "Bihar",    risk: "critical", rainfall: 260 },
  { region: "Odisha",   risk: "medium",   rainfall: 140 },
  { region: "Gujarat",  risk: "medium",   rainfall: 95  },
  { region: "UP",       risk: "low",      rainfall: 60  },
];

export default function Dashboard() {
  const [health, setHealth]   = useState(null);
  const [alerts, setAlerts]   = useState([]);
  const [subs, setSubs]       = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const [h, a, s] = await Promise.all([
          healthCheck(),
          getActiveAlerts(),
          getSubscribers(),
        ]);
        setHealth(h.data);
        setAlerts(a.data.alerts || []);
        setSubs(s.data.subscribers || []);
      } catch { /* silently handle */ } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Dashboard</h1>
        <p className="page-subtitle">
          Real-time overview of the FloodSense AI monitoring network
        </p>
      </div>

      {/* Stat cards */}
      <div className="grid-4 mb-6">
        <StatCard
          icon="🔌" label="API Status"
          value={health ? "Online" : loading ? "…" : "Offline"}
          sub={health?.uptime_seconds ? `${Math.round(health.uptime_seconds / 60)}m uptime` : undefined}
          iconBg="var(--risk-low-dim)" iconColor="var(--risk-low)"
          className="delay-1"
        />
        <StatCard
          icon="🚨" label="Active Alerts"
          value={loading ? "…" : alerts.length}
          sub={alerts.length ? "Flood events in progress" : "All clear"}
          iconBg={alerts.length ? "var(--risk-high-dim)" : "var(--risk-low-dim)"}
          iconColor={alerts.length ? "var(--risk-high)" : "var(--risk-low)"}
          className="delay-2"
        />
        <StatCard
          icon="👥" label="Subscribers"
          value={loading ? "…" : subs.length}
          sub="Registered for alerts"
          iconBg="var(--primary-dim)" iconColor="var(--primary)"
          className="delay-3"
        />
        <StatCard
          icon="🤖" label="AI Engine"
          value={health?.gemini_configured ? "Ready" : "⚠ Key Missing"}
          sub="Gemini 1.5 Pro / Flash"
          iconBg="var(--accent-dim)" iconColor="var(--accent)"
          className="delay-4"
        />
      </div>

      {/* Charts row */}
      <div className="grid-2 mb-6">
        {/* Regional Risk Bar Chart */}
        <div className="card animate-up delay-2">
          <div className="card-header">
            <h3 className="card-title">🗺️ India Regional Risk Overview</h3>
            <span className="badge badge-neutral">Live</span>
          </div>
          <div style={{ height: 220 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={INDIA_WEATHER} margin={{ top: 0, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="region" tick={{ fill: "var(--text-muted)", fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip
                  formatter={(v) => [`${v} mm`, "Rainfall"]}
                  contentStyle={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
                />
                <Bar dataKey="rainfall" radius={[4, 4, 0, 0]}>
                  {INDIA_WEATHER.map((d) => (
                    <Cell key={d.region} fill={RISK_COLORS[d.risk] || "var(--primary)"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Active Alerts list */}
        <div className="card animate-up delay-3">
          <div className="card-header">
            <h3 className="card-title">🚨 Active Alerts</h3>
            <Link to="/alerts" className="btn btn-ghost btn-sm">View All</Link>
          </div>
          {alerts.length === 0 ? (
            <div className="empty-state" style={{ padding: "1.5rem" }}>
              <div className="empty-icon" style={{ fontSize: "2rem" }}>✅</div>
              <div className="empty-text">No active alerts</div>
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              {alerts.slice(0, 5).map((a) => (
                <div key={a.alert_id}
                     className="flex justify-between items-center p-4"
                     style={{ background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)" }}>
                  <div>
                    <div className="font-semibold" style={{ fontSize: "0.875rem" }}>{a.location}</div>
                    <div className="text-muted text-xs">{a.severity}</div>
                  </div>
                  <RiskBadge level={a.risk_level} />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Quick action cards */}
      <div className="section-title">Quick Actions</div>
      <div className="grid-3">
        {[
          { to: "/predict",         icon: "🧠", title: "Run Prediction",    desc: "Analyze flood risk for any location using our multi-model AI pipeline"       },
          { to: "/simulation",      icon: "🗺️",  title: "Run Simulation",    desc: "Simulate what-if flood scenarios with SCS-CN hydrological modeling"          },
          { to: "/recommendations", icon: "💡", title: "Get Recommendations",desc: "Get personalized, AI-powered safety recommendations for citizens & authorities"},
        ].map((a, i) => (
          <Link
            key={a.to} to={a.to}
            className="card animate-up"
            style={{ textDecoration: "none", animationDelay: `${i * 50}ms`, cursor: "pointer" }}
          >
            <div style={{ fontSize: "2rem", marginBottom: "0.75rem" }}>{a.icon}</div>
            <div className="font-semibold" style={{ marginBottom: "0.35rem" }}>{a.title}</div>
            <div className="text-muted text-sm">{a.desc}</div>
          </Link>
        ))}
      </div>
    </div>
  );
}
