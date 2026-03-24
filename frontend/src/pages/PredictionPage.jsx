import React, { useState } from "react";
import PredictionForm from "../components/prediction/PredictionForm";
import PredictionResult from "../components/prediction/PredictionResult";
import StatCard from "../components/prediction/StatCard";
import { getRecommendations } from "../api/floodApi";
import { Link } from "react-router-dom";

export default function PredictionPage() {
  const [result, setResult] = useState(null);

  const handleResult = (data) => setResult(data);

  const MODELS = [
    { name: "XGBoost",       strength: "Tabular patterns",        icon: "⚡" },
    { name: "Random Forest", strength: "Ensemble robustness",     icon: "🌲" },
    { name: "LSTM",          strength: "Temporal sequences",      icon: "⏱️" },
    { name: "Logistic",      strength: "Interpretable baseline",  icon: "📊" },
  ];

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">🧠 Flood Prediction</h1>
        <p className="page-subtitle">
          Multi-model ensemble prediction powered by Gemini AI model selection
        </p>
      </div>

      <div className="grid-2" style={{ alignItems: "start", gap: "1.5rem" }}>
        {/* Left column – form */}
        <div>
          <PredictionForm onResult={handleResult} />

          {/* Model info cards */}
          <div className="section mt-6">
            <div className="section-title">Models in Ensemble</div>
            <div className="grid-2">
              {MODELS.map((m) => (
                <div key={m.name} className="card" style={{ padding: "1rem" }}>
                  <div className="flex items-center gap-2 mb-1">
                    <span style={{ fontSize: "1.1rem" }}>{m.icon}</span>
                    <span className="font-semibold" style={{ fontSize: "0.875rem" }}>{m.name}</span>
                  </div>
                  <div className="text-muted text-xs">{m.strength}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right column – results */}
        <div>
          {!result ? (
            <div className="card" style={{ textAlign: "center", padding: "3rem" }}>
              <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🌊</div>
              <div className="font-semibold" style={{ marginBottom: "0.35rem" }}>
                Ready to Predict
              </div>
              <p className="text-muted text-sm">
                Fill in the form and click "Run Flood Prediction" to start the AI pipeline.
              </p>
            </div>
          ) : (
            <>
              <PredictionResult result={result} />
              {/* Quick link to recommendations */}
              {result.risk_level && (
                <div className="alert alert-info" style={{ marginTop: "1rem" }}>
                  <span className="alert-icon">💡</span>
                  <div className="alert-content">
                    <div className="alert-title">Next Step</div>
                    <div className="alert-message">
                      Get personalized recommendations for{" "}
                      <Link to="/recommendations" style={{ color: "var(--primary)" }}>
                        {result.location || "this location"} →
                      </Link>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
