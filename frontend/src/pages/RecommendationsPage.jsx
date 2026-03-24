import React, { useState } from "react";
import RecommendationList from "../components/recommendation/RecommendationList";
import { getRecommendations, getSafetyMessage, getAuthorityBrief } from "../api/floodApi";

const USER_TYPES = [
  { value: "general_public",  label: "🏘️ General Public"  },
  { value: "authority",       label: "🏛️ Authority"        },
  { value: "first_responder", label: "🚒 First Responder"  },
  { value: "engineer",        label: "🔧 Engineer"         },
];

const RISK_LEVELS = ["low", "medium", "high", "critical"];

export default function RecommendationsPage() {
  const [form, setForm] = useState({
    location: "",
    risk_level: "high",
    user_type: "general_public",
    has_elderly: false,
    has_children: false,
    has_disability: false,
    vehicle_access: true,
  });
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const set = (k) => (e) =>
    setForm((f) => ({
      ...f,
      [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value,
    }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.location) { setError("Location is required."); return; }
    setError(null); setLoading(true);
    try {
      const res = await getRecommendations(form);
      const data = res.data;

      // Fetch SMS + authority brief if session_id available
      let sms = null, brief = null;
      if (data.session_id) {
        try {
          const sm = await getSafetyMessage(data.session_id);
          sms = sm.data.safety_message;
        } catch { /* ignore */ }
        if (form.user_type === "authority" || form.user_type === "first_responder") {
          try {
            const ab = await getAuthorityBrief(data.session_id);
            brief = ab.data.authority_brief;
          } catch { /* ignore */ }
        }
      }
      setResult({ ...data, sms, brief });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">💡 Recommendations</h1>
        <p className="page-subtitle">
          AI-powered, context-aware flood safety guidance tailored to your role
        </p>
      </div>

      <div className="grid-2" style={{ alignItems: "start", gap: "1.5rem" }}>
        {/* Form */}
        <form className="card animate-fade" onSubmit={handleSubmit}>
          <div className="card-header">
            <h2 className="card-title">⚙️ Configuration</h2>
          </div>

          {error && (
            <div className="alert alert-error mb-4">
              <span className="alert-icon">⚠️</span>
              <div className="alert-content"><div className="alert-message">{error}</div></div>
            </div>
          )}

          <div className="form-group mb-4">
            <label className="form-label">Location *</label>
            <input className="form-input" placeholder="e.g. Varanasi, UP"
              value={form.location} onChange={set("location")} required />
          </div>

          <div className="form-row mb-4">
            <div className="form-group">
              <label className="form-label">Risk Level</label>
              <select className="form-select" value={form.risk_level} onChange={set("risk_level")}>
                {RISK_LEVELS.map((r) => (
                  <option key={r} value={r}>{r.charAt(0).toUpperCase() + r.slice(1)}</option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">User Type</label>
              <select className="form-select" value={form.user_type} onChange={set("user_type")}>
                {USER_TYPES.map(({ value, label }) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Vulnerability flags */}
          <div className="section-title mb-2">Household Context</div>
          <div className="flex flex-col gap-2 mb-6">
            {[
              { key: "has_elderly",    label: "👴 Elderly members present"    },
              { key: "has_children",   label: "👶 Children present"           },
              { key: "has_disability", label: "♿ Disability or special needs" },
            ].map(({ key, label }) => (
              <label key={key}
                     style={{ display: "flex", alignItems: "center", gap: "0.6rem", cursor: "pointer", fontSize: "0.875rem" }}>
                <input type="checkbox" checked={form[key]} onChange={set(key)}
                       style={{ accentColor: "var(--primary)" }} />
                {label}
              </label>
            ))}
            <label style={{ display: "flex", alignItems: "center", gap: "0.6rem", cursor: "pointer", fontSize: "0.875rem" }}>
              <input type="checkbox" checked={form.vehicle_access} onChange={set("vehicle_access")}
                     style={{ accentColor: "var(--primary)" }} />
              🚗 Vehicle access available
            </label>
          </div>

          <button
            type="submit" id="rec-submit-btn"
            className="btn btn-primary btn-lg w-full" disabled={loading}
          >
            {loading
              ? <><span className="spinner spinner-sm" /> Generating…</>
              : "✨ Generate Recommendations"
            }
          </button>
        </form>

        {/* Results */}
        <div>
          {!result ? (
            <div className="card" style={{ textAlign: "center", padding: "3rem" }}>
              <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>💡</div>
              <div className="font-semibold" style={{ marginBottom: "0.35rem" }}>
                AI Recommendations
              </div>
              <p className="text-muted text-sm">
                Configure your context on the left, then click Generate.
              </p>
            </div>
          ) : (
            <div className="card animate-scale">
              <div className="card-header">
                <h3 className="card-title">
                  Recommendations for {result.location || form.location}
                </h3>
                <span className="badge badge-primary">{form.user_type.replace("_", " ")}</span>
              </div>
              <RecommendationList
                recommendations={result.recommendations || []}
                summary={result.summary}
                safetyMessage={result.sms}
                authorityBrief={result.brief}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
