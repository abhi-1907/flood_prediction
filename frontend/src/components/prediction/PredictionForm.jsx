import React, { useState } from "react";
import { quickPredict } from "../../api/floodApi";
import RiskBadge from "./RiskBadge";

export default function PredictionForm({ onResult }) {
  const [form, setForm] = useState({
    location: "",
    latitude: "",
    longitude: "",
    rain_mm_weekly: "",
    dist_major_river_km: "",
    elevation_m: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.location) { setError("Location is required."); return; }
    setError(null);
    setLoading(true);
    try {
      const payload = {
        location: form.location,
        latitude:            parseFloat(form.latitude)            || undefined,
        longitude:           parseFloat(form.longitude)           || undefined,
        rain_mm_weekly:      parseFloat(form.rain_mm_weekly)      || undefined,
        dist_major_river_km: parseFloat(form.dist_major_river_km) || undefined,
        elevation_m:         parseFloat(form.elevation_m)         || undefined,
      };
      const res = await quickPredict(payload);
      onResult && onResult(res.data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Quick-fill Indian cities
  const PRESETS = [
    { label: "Kochi",   location: "Kochi",   lat: 9.93,  lon: 76.27, rain: 150, river: 2,   elev: 5 },
    { label: "Patna",   location: "Patna",   lat: 25.59, lon: 85.14, rain: 200, river: 0.5, elev: 53 },
    { label: "Mumbai",  location: "Mumbai",  lat: 19.08, lon: 72.88, rain: 120, river: 8,   elev: 14 },
    { label: "Guwahati",location: "Guwahati",lat: 26.14, lon: 91.74, rain: 250, river: 1,   elev: 55 },
  ];

  const applyPreset = (p) =>
    setForm((f) => ({
      ...f,
      location: p.location,
      latitude:  String(p.lat),
      longitude: String(p.lon),
      rain_mm_weekly: String(p.rain),
      dist_major_river_km: String(p.river),
      elevation_m: String(p.elev),
    }));

  return (
    <form className="card animate-fade" onSubmit={handleSubmit}>
      <div className="card-header">
        <h2 className="card-title">🧠 Flood Prediction</h2>
        <div className="flex gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.label}
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => applyPreset(p)}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="alert alert-error mb-4">
          <span className="alert-icon">⚠️</span>
          <div className="alert-content">
            <div className="alert-message">{error}</div>
          </div>
        </div>
      )}

      {/* Location */}
      <div className="form-group mb-4">
        <label className="form-label">Location *</label>
        <input
          className="form-input"
          placeholder="e.g. Kochi, Kerala"
          value={form.location}
          onChange={set("location")}
          required
        />
      </div>

      {/* Coordinates */}
      <div className="form-row mb-4">
        <div className="form-group">
          <label className="form-label">Latitude</label>
          <input className="form-input" type="number" step="0.0001"
            placeholder="9.9312" value={form.latitude} onChange={set("latitude")} />
        </div>
        <div className="form-group">
          <label className="form-label">Longitude</label>
          <input className="form-input" type="number" step="0.0001"
            placeholder="76.2673" value={form.longitude} onChange={set("longitude")} />
        </div>
      </div>

      {/* Key features */}
      <div
        style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "1rem", marginBottom: "1.5rem" }}
      >
        <div className="form-group">
          <label className="form-label">Rainfall (weekly mm)</label>
          <input className="form-input" type="number" min="0" step="0.1"
            placeholder="50" value={form.rain_mm_weekly} onChange={set("rain_mm_weekly")} />
        </div>
        <div className="form-group">
          <label className="form-label">Dist. to River (km)</label>
          <input className="form-input" type="number" min="0" step="0.1"
            placeholder="5.0" value={form.dist_major_river_km} onChange={set("dist_major_river_km")} />
        </div>
        <div className="form-group">
          <label className="form-label">Elevation (m)</label>
          <input className="form-input" type="number" step="0.1"
            placeholder="30" value={form.elevation_m} onChange={set("elevation_m")} />
        </div>
      </div>

      <button
        type="submit"
        className="btn btn-primary btn-lg w-full"
        disabled={loading}
        id="predict-submit-btn"
      >
        {loading ? (
          <>
            <span className="spinner spinner-sm" />
            Analyzing… (may take 10–30s)
          </>
        ) : (
          "⚡ Run Flood Prediction"
        )}
      </button>
    </form>
  );
}
