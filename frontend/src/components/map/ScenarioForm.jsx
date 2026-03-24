import React, { useState } from "react";
import { runScenario } from "../../api/floodApi";

const SCENARIOS = [
  { label: "What-if Rain",    query: "What if 200mm of rain falls in {location} over 2 days?" },
  { label: "100-yr Flood",    query: "Simulate a 100-year return period flood for {location}" },
  { label: "Dam Break",       query: "What happens if the dam upstream of {location} releases 800 m³/s?" },
  { label: "Extreme Storm",   query: "Simulate an extreme 500mm storm event in {location}" },
];

export default function ScenarioForm({ onResult }) {
  const [form, setForm] = useState({
    query: "",
    location: "",
    latitude: "",
    longitude: "",
    rainfall_mm: "",
    rainfall_days: "1",
    return_period_years: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  const applyTemplate = (tpl) => {
    const q = tpl.replace("{location}", form.location || "the selected area");
    setForm((f) => ({ ...f, query: q }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.location) { setError("Please enter a location."); return; }
    setError(null);
    setLoading(true);
    try {
      const payload = {
        query:    form.query || `Simulate flood for ${form.location}`,
        location: form.location,
        latitude:  parseFloat(form.latitude)  || undefined,
        longitude: parseFloat(form.longitude) || undefined,
        rainfall_mm:         parseFloat(form.rainfall_mm)         || undefined,
        rainfall_days:       parseInt(form.rainfall_days, 10)     || 1,
        return_period_years: parseInt(form.return_period_years,10)|| undefined,
      };
      const res = await runScenario(payload);
      onResult && onResult(res.data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form className="card animate-fade" onSubmit={handleSubmit}>
      <div className="card-header">
        <h2 className="card-title">🗺️ What-If Scenario</h2>
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
      <div className="form-row mb-4">
        <div className="form-group">
          <label className="form-label">Location *</label>
          <input className="form-input" placeholder="Patna, Bihar"
            value={form.location} onChange={set("location")} required />
        </div>
        <div className="form-group">
          <label className="form-label">Rainfall (mm)</label>
          <input className="form-input" type="number" min="0"
            placeholder="200" value={form.rainfall_mm} onChange={set("rainfall_mm")} />
        </div>
      </div>

      <div className="form-row mb-4">
        <div className="form-group">
          <label className="form-label">Duration (days)</label>
          <input className="form-input" type="number" min="1" max="7"
            placeholder="2" value={form.rainfall_days} onChange={set("rainfall_days")} />
        </div>
        <div className="form-group">
          <label className="form-label">Return Period (years)</label>
          <input className="form-input" type="number"
            placeholder="100" value={form.return_period_years} onChange={set("return_period_years")} />
        </div>
      </div>

      {/* NL query */}
      <div className="form-group mb-2">
        <label className="form-label">Natural Language Scenario</label>
        <textarea
          className="form-textarea"
          style={{ minHeight: 80 }}
          placeholder="Describe your scenario…"
          value={form.query}
          onChange={set("query")}
        />
      </div>

      {/* Quick templates */}
      <div className="flex gap-2 flex-wrap mb-4">
        {SCENARIOS.map((s) => (
          <button
            type="button" key={s.label}
            className="btn btn-ghost btn-sm"
            onClick={() => applyTemplate(s.query)}
          >
            {s.label}
          </button>
        ))}
      </div>

      <button
        type="submit" id="scenario-run-btn"
        className="btn btn-primary btn-lg w-full"
        disabled={loading}
      >
        {loading ? (
          <><span className="spinner spinner-sm" /> Simulating…</>
        ) : "🚀 Run Simulation"}
      </button>
    </form>
  );
}
