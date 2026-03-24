import React, { useState } from "react";
import { subscribeToAlerts } from "../../api/floodApi";

const CHANNELS = [
  { key: "email", icon: "📧", label: "Email"   },
  { key: "sms",   icon: "📱", label: "SMS"     },
  { key: "push",  icon: "🔔", label: "Push"    },
];

const RISK_THRESHOLDS = ["low", "medium", "high", "critical"];

export default function AlertSubscribe({ onSubscribed }) {
  const [form, setForm] = useState({
    name:         "",
    location:     "",
    latitude:     "",
    longitude:    "",
    email:        "",
    phone:        "",
    channels:     ["email"],
    min_risk_level: "medium",
    radius_km:    "25",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [success, setSuccess] = useState(null);

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  const toggleChannel = (ch) =>
    setForm((f) => ({
      ...f,
      channels: f.channels.includes(ch)
        ? f.channels.filter((c) => c !== ch)
        : [...f.channels, ch],
    }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.name || !form.location) {
      setError("Name and location are required.");
      return;
    }
    if (!form.channels.length) {
      setError("Select at least one alert channel.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await subscribeToAlerts({
        name:       form.name,
        location:   form.location,
        latitude:   parseFloat(form.latitude)  || null,
        longitude:  parseFloat(form.longitude) || null,
        email:      form.email  || null,
        phone:      form.phone  || null,
        channels:   form.channels,
        min_risk_level: form.min_risk_level,
        radius_km:  parseFloat(form.radius_km) || 25,
      });
      setSuccess(`Subscribed! Your ID: ${res.data.subscriber_id}`);
      onSubscribed && onSubscribed(res.data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form className="card animate-fade" onSubmit={handleSubmit}>
      <div className="card-header">
        <h2 className="card-title">🔔 Subscribe to Flood Alerts</h2>
      </div>

      {error && (
        <div className="alert alert-error mb-4">
          <span className="alert-icon">⚠️</span>
          <div className="alert-content"><div className="alert-message">{error}</div></div>
        </div>
      )}
      {success && (
        <div className="alert alert-success mb-4">
          <span className="alert-icon">✅</span>
          <div className="alert-content"><div className="alert-message">{success}</div></div>
        </div>
      )}

      <div className="form-row mb-4">
        <div className="form-group">
          <label className="form-label">Your Name *</label>
          <input className="form-input" placeholder="Ravi Kumar"
            value={form.name} onChange={set("name")} required />
        </div>
        <div className="form-group">
          <label className="form-label">Location *</label>
          <input className="form-input" placeholder="Guwahati, Assam"
            value={form.location} onChange={set("location")} required />
        </div>
      </div>

      <div className="form-row mb-4">
        <div className="form-group">
          <label className="form-label">Email</label>
          <input className="form-input" type="email" placeholder="ravi@example.com"
            value={form.email} onChange={set("email")} />
        </div>
        <div className="form-group">
          <label className="form-label">Phone (E.164)</label>
          <input className="form-input" placeholder="+919876543210"
            value={form.phone} onChange={set("phone")} />
        </div>
      </div>

      {/* Channels */}
      <div className="form-group mb-4">
        <label className="form-label">Alert Channels</label>
        <div className="flex gap-3 mt-1">
          {CHANNELS.map(({ key, icon, label }) => (
            <button
              key={key} type="button"
              className={`btn ${form.channels.includes(key) ? "btn-primary" : "btn-secondary"}`}
              onClick={() => toggleChannel(key)}
            >
              {icon} {label}
            </button>
          ))}
        </div>
      </div>

      <div className="form-row mb-4">
        <div className="form-group">
          <label className="form-label">Minimum Risk Level</label>
          <select className="form-select" value={form.min_risk_level} onChange={set("min_risk_level")}>
            {RISK_THRESHOLDS.map((r) => (
              <option key={r} value={r}>{r.charAt(0).toUpperCase() + r.slice(1)}</option>
            ))}
          </select>
        </div>
        <div className="form-group">
          <label className="form-label">Alert Radius (km)</label>
          <input className="form-input" type="number" min="5" max="200"
            value={form.radius_km} onChange={set("radius_km")} />
        </div>
      </div>

      <button
        type="submit" id="alert-subscribe-btn"
        className="btn btn-primary btn-lg w-full"
        disabled={loading}
      >
        {loading ? <><span className="spinner spinner-sm" /> Subscribing…</> : "📲 Subscribe to Alerts"}
      </button>
    </form>
  );
}
