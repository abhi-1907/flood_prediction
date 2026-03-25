/**
 * PredictionPage.jsx
 * Entry point for the AI pipeline — form, run button, step tracker, prediction result.
 * All other agent pages read results from PipelineContext.
 */
import React, { useState } from "react";
import { usePipeline } from "../context/PipelineContext";
import RiskBadge from "../components/prediction/RiskBadge";
import { Link } from "react-router-dom";

// ── Agent step definitions ────────────────────────────────────────────────────
const STEPS = [
  { key: "data_ingestion", icon: "📥", label: "Data Ingestion",  desc: "Fetching & validating input data" },
  { key: "preprocessing",  icon: "⚙️", label: "Preprocessing",   desc: "Cleaning & feature engineering" },
  { key: "prediction",     icon: "🧠", label: "Prediction",      desc: "Running ensemble ML models" },
  { key: "recommendation", icon: "💡", label: "Recommendations", desc: "Generating AI safety guidance" },
  { key: "simulation",     icon: "🗺️", label: "Simulation",      desc: "Scenario flood modelling" },
  { key: "alerting",       icon: "🔔", label: "Alerting",        desc: "Dispatching subscriber alerts" },
];

const STATUS_STYLE = {
  idle:    { color: "var(--text-muted)",   icon: "○" },
  active:  { color: "var(--primary)",      icon: "◉" },
  done:    { color: "var(--risk-low)",     icon: "✓" },
  skipped: { color: "var(--text-muted)",   icon: "–" },
  error:   { color: "var(--risk-high)",    icon: "✗" },
};

// ── Step Card ─────────────────────────────────────────────────────────────────
function StepCard({ step, status, detail }) {
  const style = STATUS_STYLE[status] || STATUS_STYLE.idle;
  return (
    <div className="card" style={{
      display: "flex", alignItems: "flex-start", gap: "0.875rem",
      padding: "0.75rem 1rem",
      borderLeft: `3px solid ${style.color}`,
      opacity: status === "idle" ? 0.4 : 1,
      transition: "all 0.3s ease",
    }}>
      <div style={{
        width: 32, height: 32, borderRadius: "50%",
        background: style.color + "22",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: "1rem", flexShrink: 0,
        boxShadow: status === "active" ? `0 0 10px ${style.color}60` : "none",
        animation: status === "active" ? "pulse 1.2s ease-in-out infinite" : "none",
      }}>
        {step.icon}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginBottom: "0.1rem" }}>
          <span style={{ fontWeight: 600, fontSize: "0.8rem" }}>{step.label}</span>
          <span style={{ color: style.color, fontSize: "0.65rem", fontWeight: 700, letterSpacing: "0.05em" }}>
            {style.icon} {status.toUpperCase()}
          </span>
        </div>
        <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
          {detail || step.desc}
        </div>
      </div>
    </div>
  );
}

// ── Prediction Result Panel ───────────────────────────────────────────────────
function PredictionResult({ data }) {
  if (!data) return null;
  const pct = Math.round((data.flood_probability || 0) * 100);
  const riskColor = {
    HIGH: "var(--risk-high)", MEDIUM: "var(--risk-medium)",
    LOW: "var(--risk-low)",   CRITICAL: "var(--risk-critical)",
  }[data.risk_level?.toUpperCase()] || "var(--primary)";

  return (
    <div className="card animate-fade" style={{ borderTop: `3px solid ${riskColor}` }}>
      <div className="card-title mb-4">🧠 Prediction Results</div>
      <div style={{ display: "flex", gap: "2rem", flexWrap: "wrap", alignItems: "center" }}>
        {/* Probability dial */}
        <div style={{ textAlign: "center" }}>
          <div style={{
            width: 110, height: 110, borderRadius: "50%",
            background: `conic-gradient(${riskColor} ${pct * 3.6}deg, var(--bg-elevated) 0deg)`,
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: `0 0 20px ${riskColor}44`,
          }}>
            <div style={{
              width: 82, height: 82, borderRadius: "50%",
              background: "var(--bg-surface)",
              display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
            }}>
              <span style={{ fontSize: "1.4rem", fontWeight: 800, color: riskColor }}>{pct}%</span>
              <span style={{ fontSize: "0.6rem", color: "var(--text-muted)", marginTop: 2 }}>PROBABILITY</span>
            </div>
          </div>
        </div>
        {/* Stats */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "0.75rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <span style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>Risk Level</span>
            <RiskBadge level={data.risk_level} />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <span style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>Confidence</span>
            <span style={{ fontWeight: 600, fontSize: "0.875rem" }}>
              {Math.round((data.confidence || 0) * 100)}%
            </span>
          </div>
          {data.models_used?.length > 0 && (
            <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap" }}>
              {data.models_used.map((m) => (
                <span key={m} className="badge badge-primary" style={{ fontSize: "0.65rem" }}>{m}</span>
              ))}
            </div>
          )}
        </div>
      </div>
      {data.explanation && (
        <p style={{
          marginTop: "1rem", fontSize: "0.8rem", color: "var(--text-secondary)",
          lineHeight: 1.7, background: "var(--bg-elevated)", padding: "0.75rem",
          borderRadius: "var(--radius-sm)",
        }}>
          {data.explanation}
        </p>
      )}

      {/* Links to other agent pages */}
      <div style={{
        marginTop: "1rem", display: "flex", gap: "0.5rem", flexWrap: "wrap",
      }}>
        {[
          { to: "/recommendations", label: "💡 Recommendations" },
          { to: "/simulation",      label: "🗺️ Simulation" },
          { to: "/alerting",        label: "🔔 Alerting" },
        ].map(({ to, label }) => (
          <Link key={to} to={to} className="btn btn-secondary" style={{ fontSize: "0.75rem", padding: "0.4rem 0.875rem" }}>
            {label} →
          </Link>
        ))}
      </div>
    </div>
  );
}

// ── Main Prediction Page ──────────────────────────────────────────────────────
export default function PredictionPage() {
  const {
    running, done, error, stepStatus, stepDetail,
    prediction, elapsed, sessionId,
    runPipeline, abortPipeline,
  } = usePipeline();

  const [query,       setQuery]       = useState("");
  const [userType,    setUserType]    = useState("general_public");
  const [location,    setLocation]    = useState("");
  const [rain,        setRain]        = useState("");
  const [rainMonthly, setRainMonthly] = useState("");
  const [temp,        setTemp]        = useState("");
  const [humidity,    setHumidity]    = useState("");
  const [wind,        setWind]        = useState("");
  const [elevation,   setElevation]   = useState("");
  const [slope,       setSlope]       = useState("");
  const [terrain,     setTerrain]     = useState("rural");
  const [river,       setRiver]       = useState("");
  const [waterbody,   setWaterbody]   = useState(false);
  const [damCount,    setDamCount]    = useState("");
  const [file,        setFile]        = useState(null);
  const [localErr,    setLocalErr]    = useState(null);
  const [inputMode,   setInputMode]   = useState("nlp");

  const handleRun = (e) => {
    e.preventDefault();

    // Validate based on inputMode
    if (inputMode === "nlp" && !query.trim()) {
      setLocalErr("Please describe your scenario in the NLP query tab.");
      return;
    }
    if (inputMode === "manual" && !location.trim()) {
      setLocalErr("Please provide at minimum a location in the Parameters tab.");
      return;
    }
    if (inputMode === "file" && !file) {
      setLocalErr("Please upload a CSV or JSON file in the File Upload tab.");
      return;
    }

    setLocalErr(null);
    runPipeline({ 
      inputMode, query, userType, location, rain, rainMonthly, temp, humidity, wind, 
      elevation, slope, terrain, river, waterbody, damCount, file 
    });
  };

  const displayError = localErr || error;

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <h1 className="page-title">🧠 Flood Prediction</h1>
        <p className="page-subtitle">
          Run the AI pipeline — Gemini orchestrates ingestion, prediction, recommendations, simulation &amp; alerts automatically.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: "1.5rem", alignItems: "start" }}>

        {/* ── LEFT: Form + Result ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>

          <form className="card" onSubmit={handleRun}>
            <div className="card-title mb-4">🔍 Describe Your Scenario</div>

            {/* ── Input Mode Tabs ── */}
            <div style={{
              display: "flex", gap: "0.25rem",
              background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)",
              padding: "0.25rem", marginBottom: "1.25rem",
            }}>
              {[
                { id: "nlp",    label: "🗣️ NLP Query" },
                { id: "manual", label: "📋 Parameters" },
                { id: "file",   label: "📁 File Upload" },
              ].map(tab => (
                <button
                  key={tab.id}
                  type="button"
                  style={{
                    flex: 1, padding: "0.5rem", border: "none", borderRadius: "var(--radius-sm)",
                    cursor: "pointer", fontWeight: 600, fontSize: "0.75rem", transition: "all 0.2s",
                    background: inputMode === tab.id ? "var(--primary)" : "transparent",
                    color:      inputMode === tab.id ? "#fff"           : "var(--text-muted)",
                  }}
                  onClick={() => { setInputMode(tab.id); setLocalErr(null); }}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* ── NLP Tab ── */}
            {inputMode === "nlp" && (
              <div className="form-group mb-4">
                <label className="form-label">Natural Language Query *</label>
                <textarea
                  className="form-textarea"
                  rows={3}
                  placeholder='E.g. "Predict flood risk for Kochi with 120mm weekly rain and 5km from river"'
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  style={{ minHeight: 80 }}
                />
              </div>
            )}

            {/* ── Parameters Tab ── */}
            {inputMode === "manual" && (
              <div style={{
                background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)",
                padding: "1.25rem", marginBottom: "1.25rem",
                display: "flex", flexDirection: "column", gap: "1.5rem"
              }}>
                {/* 1. Geography */}
                <div>
                  <div style={{ fontSize: "0.7rem", fontWeight: 700, color: "var(--primary)", textTransform: "uppercase", marginBottom: "0.75rem", letterSpacing: "0.05em" }}>📍 Location & Geography</div>
                  <div className="form-row mb-3">
                    <div className="form-group">
                      <label className="form-label">Location Name *</label>
                      <input className="form-input" placeholder="Kochi, Kerala" value={location} onChange={e => setLocation(e.target.value)} />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Terrain Type</label>
                      <select className="form-select" value={terrain} onChange={e => setTerrain(e.target.value)}>
                        <option value="rural">Rural / Agricultural</option>
                        <option value="urban">Urban / Residential</option>
                        <option value="forest">Forest / Vegetation</option>
                        <option value="mountain">Mountain / Hilly</option>
                      </select>
                    </div>
                  </div>
                  <div className="form-row">
                    <div className="form-group">
                      <label className="form-label">Elevation (m)</label>
                      <input className="form-input" type="number" placeholder="15" value={elevation} onChange={e => setElevation(e.target.value)} />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Slope (degrees)</label>
                      <input className="form-input" type="number" placeholder="2.5" value={slope} onChange={e => setSlope(e.target.value)} />
                    </div>
                  </div>
                </div>

                {/* 2. Weather */}
                <div>
                  <div style={{ fontSize: "0.7rem", fontWeight: 700, color: "var(--primary)", textTransform: "uppercase", marginBottom: "0.75rem", letterSpacing: "0.05em" }}>☁️ Weather Conditions</div>
                  <div className="form-row mb-3">
                    <div className="form-group">
                      <label className="form-label">Weekly Rain (mm)</label>
                      <input className="form-input" type="number" placeholder="120" value={rain} onChange={e => setRain(e.target.value)} />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Monthly Rain (mm)</label>
                      <input className="form-input" type="number" placeholder="450" value={rainMonthly} onChange={e => setRainMonthly(e.target.value)} />
                    </div>
                  </div>
                  <div className="form-row">
                    <div className="form-group">
                      <label className="form-label">Avg Temp (°C)</label>
                      <input className="form-input" type="number" placeholder="28" value={temp} onChange={e => setTemp(e.target.value)} />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Humidity (%)</label>
                      <input className="form-input" type="number" placeholder="85" value={humidity} onChange={e => setHumidity(e.target.value)} />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Wind Speed (m/s)</label>
                      <input className="form-input" type="number" placeholder="4.2" value={wind} onChange={e => setWind(e.target.value)} />
                    </div>
                  </div>
                </div>

                {/* 3. Hydrology */}
                <div>
                  <div style={{ fontSize: "0.7rem", fontWeight: 700, color: "var(--primary)", textTransform: "uppercase", marginBottom: "0.75rem", letterSpacing: "0.05em" }}>🌊 Hydrology Details</div>
                  <div className="form-row mb-3">
                    <div className="form-group">
                      <label className="form-label">Dist. to River (km)</label>
                      <input className="form-input" type="number" placeholder="3.5" value={river} onChange={e => setRiver(e.target.value)} />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Dams in 50km</label>
                      <input className="form-input" type="number" placeholder="1" value={damCount} onChange={e => setDamCount(e.target.value)} />
                    </div>
                  </div>
                  <div className="form-group">
                    <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", cursor: "pointer", fontSize: "0.8rem" }}>
                      <input type="checkbox" checked={waterbody} onChange={e => setWaterbody(e.target.checked)} />
                      Major Waterbody Nearby?
                    </label>
                  </div>
                </div>
              </div>
            )}

            {/* ── File Upload Tab ── */}
            {inputMode === "file" && (
              <div style={{
                background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)",
                padding: "1rem", marginBottom: "1rem",
              }}>
                <div className="form-group mb-3">
                  <label className="form-label">Upload Data (CSV/JSON) *</label>
                  <input
                    className="form-input" type="file" accept=".csv,.json"
                    style={{ cursor: "pointer" }}
                    onChange={e => setFile(e.target.files[0] || null)}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Optional Description</label>
                  <input
                    className="form-input"
                    placeholder="E.g. Sensor data for Kerala 2024"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                </div>
              </div>
            )}

            {/* User Type (always visible) */}
            <div className="form-group mb-4">
              <label className="form-label">User Type</label>
              <select className="form-select" value={userType} onChange={e => setUserType(e.target.value)}>
                <option value="general_public">General Public</option>
                <option value="authority">Authority</option>
                <option value="first_responder">First Responder</option>
              </select>
            </div>

            {displayError && (
              <div className="alert alert-error mb-3">
                <span className="alert-icon">⚠️</span>
                <div className="alert-content"><div className="alert-message">{displayError}</div></div>
              </div>
            )}

            <div style={{ display: "flex", gap: "0.75rem" }}>
              <button
                type="submit"
                className="btn btn-primary btn-lg"
                style={{ flex: 1 }}
                disabled={running}
                id="pipeline-run-btn"
              >
                {running
                  ? <><span className="spinner spinner-sm" /> Running Pipeline…</>
                  : "🚀 Run Prediction"
                }
              </button>
              {running && (
                <button type="button" className="btn btn-secondary" onClick={abortPipeline}>
                  Stop
                </button>
              )}
            </div>
          </form>

          {/* Prediction result */}
          {prediction && (
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              {done && elapsed && (
                <div style={{
                  display: "flex", alignItems: "center", gap: "0.75rem",
                  color: "var(--risk-low)", fontSize: "0.8rem", fontWeight: 600,
                }}>
                  <span>✅</span>
                  <span>Pipeline completed in {elapsed.toFixed(1)}s</span>
                  {sessionId && (
                    <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>
                      · Session: {sessionId.slice(0, 12)}…
                    </span>
                  )}
                </div>
              )}
              <PredictionResult data={prediction} />
            </div>
          )}

          {/* "Running" state before any result */}
          {running && !prediction && (
            <div className="card" style={{ textAlign: "center", padding: "2.5rem" }}>
              <span className="spinner" style={{ width: 36, height: 36, marginBottom: "1rem" }} />
              <div style={{ fontWeight: 600, marginBottom: "0.35rem" }}>Pipeline running…</div>
              <p style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>
                Gemini is orchestrating agents. Watch the tracker →
              </p>
            </div>
          )}

          {/* Empty state */}
          {!running && !prediction && (
            <div className="card" style={{ textAlign: "center", padding: "3rem" }}>
              <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🌊</div>
              <div style={{ fontWeight: 600, marginBottom: "0.35rem" }}>Ready to Predict</div>
              <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
                Fill in the form and click <strong>"Run Prediction"</strong> to start the AI pipeline.
              </p>
            </div>
          )}
        </div>

        {/* ── RIGHT: Pipeline Tracker ── */}
        <div style={{ position: "sticky", top: "80px" }}>
          <div className="card">
            <div className="card-title mb-4" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span>Pipeline Tracker</span>
              {running && <span className="spinner spinner-sm" />}
              {done && <span style={{ color: "var(--risk-low)", fontSize: "0.75rem" }}>Done ✓</span>}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
              {STEPS.map((step) => (
                <StepCard
                  key={step.key}
                  step={step}
                  status={stepStatus[step.key] || "idle"}
                  detail={stepDetail[step.key]}
                />
              ))}
            </div>

            <div style={{
              marginTop: "1rem", padding: "0.75rem",
              background: "var(--accent-dim)", borderRadius: "var(--radius-sm)",
              fontSize: "0.72rem", color: "var(--accent)", lineHeight: 1.6,
            }}>
              🤖 <strong>Gemini orchestrates</strong> this pipeline — it decides which agents to invoke based on your query and context.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
