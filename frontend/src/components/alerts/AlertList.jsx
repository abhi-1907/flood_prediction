import React, { useState, useEffect } from "react";
import { getActiveAlerts, getSubscribers, triggerAlert, unsubscribe } from "../../api/floodApi";
import RiskBadge from "../prediction/RiskBadge";

function formatDate(dt) {
  if (!dt) return "—";
  return new Date(dt).toLocaleString("en-IN", { dateStyle: "short", timeStyle: "short" });
}

export default function AlertList() {
  const [alerts, setAlerts]       = useState([]);
  const [subscribers, setSubs]    = useState([]);
  const [activeTab, setActiveTab] = useState("alerts");
  const [loading, setLoading]     = useState(false);
  const [trigForm, setTrigForm]   = useState({ location: "", risk_level: "high", message: "" });
  const [trigLoading, setTrigLoading] = useState(false);
  const [trigMsg, setTrigMsg]     = useState(null);

  const refresh = async () => {
    setLoading(true);
    try {
      const [a, s] = await Promise.all([getActiveAlerts(), getSubscribers()]);
      setAlerts(a.data.alerts || []);
      setSubs(s.data.subscribers || []);
    } catch { /* silently ignore */ } finally {
      setLoading(false);
    }
  };

  useEffect(() => { refresh(); }, []);

  const handleTrigger = async (e) => {
    e.preventDefault();
    setTrigLoading(true); setTrigMsg(null);
    try {
      const res = await triggerAlert({
        location:   trigForm.location,
        risk_level: trigForm.risk_level,
        message:    trigForm.message || undefined,
      });
      setTrigMsg(`✅ Alert sent to ${res.data.sent_count} / ${res.data.total_subscribers} subscribers`);
      refresh();
    } catch (err) {
      setTrigMsg(`❌ ${err.message}`);
    } finally {
      setTrigLoading(false);
    }
  };

  const handleUnsub = async (id) => {
    try {
      await unsubscribe(id);
      setSubs((s) => s.filter((sub) => sub.id !== id));
    } catch { /* ignore */ }
  };

  return (
    <div className="animate-fade">
      {/* Tabs */}
      <div className="tabs mb-4" style={{ width: "fit-content" }}>
        {["alerts", "subscribers", "trigger"].map((t) => (
          <button key={t} className={`tab${activeTab === t ? " active" : ""}`} onClick={() => setActiveTab(t)}>
            {t === "alerts" ? `🚨 Active (${alerts.length})` :
             t === "subscribers" ? `👥 Subscribers (${subscribers.length})` : "⚡ Trigger"}
          </button>
        ))}
        <button className="btn btn-ghost btn-sm" onClick={refresh} style={{ marginLeft: "0.5rem" }}>
          {loading ? <span className="spinner spinner-sm" /> : "↻"}
        </button>
      </div>

      {/* Active Alerts */}
      {activeTab === "alerts" && (
        alerts.length === 0 ? (
          <div className="empty-state"><div className="empty-icon">✅</div>
            <div className="empty-text">No active flood alerts</div></div>
        ) : (
          <div className="flex flex-col gap-3">
            {alerts.map((a) => (
              <div key={a.alert_id} className="card" style={{ borderLeft: `3px solid var(--risk-high)` }}>
                <div className="flex justify-between items-center mb-2">
                  <span className="font-semibold">{a.location}</span>
                  <RiskBadge level={a.risk_level} />
                </div>
                <div className="grid-3 text-sm text-secondary">
                  <div><span className="text-muted">Severity: </span>{a.severity}</div>
                  <div><span className="text-muted">Deliveries: </span>{a.deliveries}</div>
                  <div><span className="text-muted">Expires: </span>{formatDate(a.expires_at)}</div>
                </div>
              </div>
            ))}
          </div>
        )
      )}

      {/* Subscribers */}
      {activeTab === "subscribers" && (
        subscribers.length === 0 ? (
          <div className="empty-state"><div className="empty-icon">👥</div>
            <div className="empty-text">No subscribers yet</div></div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Name</th><th>Location</th><th>Channels</th><th>Min Risk</th><th>Radius</th><th></th>
                </tr>
              </thead>
              <tbody>
                {subscribers.map((s) => (
                  <tr key={s.id}>
                    <td>{s.name}</td>
                    <td>{s.location}</td>
                    <td>
                      <div className="flex gap-1 flex-wrap">
                        {(s.channels || []).map((ch) => (
                          <span key={ch} className="badge badge-primary">{ch}</span>
                        ))}
                      </div>
                    </td>
                    <td><RiskBadge level={s.min_risk_level} /></td>
                    <td>{s.radius_km} km</td>
                    <td>
                      <button className="btn btn-danger btn-sm" onClick={() => handleUnsub(s.id)}>
                        Remove
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      )}

      {/* Manual Trigger */}
      {activeTab === "trigger" && (
        <form className="card" onSubmit={handleTrigger}>
          <div className="card-title mb-4">⚡ Manual Alert Trigger</div>
          {trigMsg && (
            <div className={`alert ${trigMsg.startsWith("✅") ? "alert-success" : "alert-error"} mb-4`}>
              <div className="alert-content"><div className="alert-message">{trigMsg}</div></div>
            </div>
          )}
          <div className="form-row mb-4">
            <div className="form-group">
              <label className="form-label">Location</label>
              <input className="form-input" placeholder="Bhubaneswar, Odisha"
                value={trigForm.location}
                onChange={(e) => setTrigForm((f) => ({ ...f, location: e.target.value }))} required />
            </div>
            <div className="form-group">
              <label className="form-label">Risk Level</label>
              <select className="form-select" value={trigForm.risk_level}
                onChange={(e) => setTrigForm((f) => ({ ...f, risk_level: e.target.value }))}>
                {["low","medium","high","critical"].map((r) => (
                  <option key={r} value={r}>{r.charAt(0).toUpperCase() + r.slice(1)}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="form-group mb-4">
            <label className="form-label">Custom Message (optional)</label>
            <textarea className="form-textarea" style={{minHeight:80}} placeholder="Override the auto-generated alert content..."
              value={trigForm.message}
              onChange={(e) => setTrigForm((f) => ({ ...f, message: e.target.value }))} />
          </div>
          <button type="submit" className="btn btn-danger btn-lg w-full" disabled={trigLoading}
            id="alert-trigger-btn">
            {trigLoading ? <><span className="spinner spinner-sm" /> Sending…</> : "🚨 Send Alert Now"}
          </button>
        </form>
      )}
    </div>
  );
}
