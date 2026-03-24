import React, { useState } from "react";
import ScenarioForm from "../components/map/ScenarioForm";
import FloodMap from "../components/map/FloodMap";
import SimulationChart from "../components/map/SimulationChart";
import StatCard from "../components/prediction/StatCard";

export default function SimulationPage() {
  const [result, setResult] = useState(null);

  const handleResult = (data) => setResult(data);

  const INDIA_CENTER = [20.5937, 78.9629];
  const mapCenter = result?.geojson
    ? [
        result.location?.lat ?? INDIA_CENTER[0],
        result.location?.lon ?? INDIA_CENTER[1],
      ]
    : INDIA_CENTER;

  const timeline = result?.timeline_chart || [];
  const impacts  = result?.impact_summary  || [];

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">🗺️ Flood Simulation</h1>
        <p className="page-subtitle">
          Hydrological what-if scenarios using SCS-CN model with India-specific parameters
        </p>
      </div>

      <div className="grid-2" style={{ alignItems: "start", gap: "1.5rem" }}>
        {/* Left – form */}
        <div>
          <ScenarioForm onResult={handleResult} />

          {result && (
            <div className="grid-2 mt-6">
              <StatCard
                icon="🌊" label="Peak Depth"
                value={`${(result.peak_depth_m || 0).toFixed(1)} m`}
                sub={`at hour ${result.peak_hour || 0}`}
                iconBg="var(--risk-high-dim)" iconColor="var(--risk-high)"
              />
              <StatCard
                icon="📐" label="Inundated Area"
                value={`${(result.total_area_km2 || 0).toFixed(1)} km²`}
                sub={`${(result.inundated_pct || 0).toFixed(1)}% of region`}
                iconBg="var(--risk-medium-dim)" iconColor="var(--risk-medium)"
              />
              {impacts.slice(0, 2).map((imp) => (
                <StatCard
                  key={imp.metric}
                  icon="📊" label={imp.metric}
                  value={`${imp.value.toLocaleString("en-IN")} ${imp.unit}`}
                  sub={imp.description}
                  iconBg="var(--accent-dim)" iconColor="var(--accent)"
                />
              ))}
            </div>
          )}
        </div>

        {/* Right – map + chart */}
        <div>
          <div className="card" style={{ padding: 0, overflow: "hidden", marginBottom: "1.5rem" }}>
            <div style={{ padding: "0.875rem 1.25rem", borderBottom: "1px solid var(--border)" }}>
              <span className="card-title">🗺️ Inundation Map</span>
              {result?.scenario_name && (
                <span className="badge badge-primary" style={{ marginLeft: "0.5rem" }}>
                  {result.scenario_name}
                </span>
              )}
            </div>
            <FloodMap
              geojson={result?.geojson}
              center={mapCenter}
              zoom={result?.geojson ? 8 : 5}
              height={380}
            />
          </div>

          <SimulationChart timeline={timeline} peakHour={result?.peak_hour} />

          {result?.summary && (
            <div className="alert alert-info" style={{ marginTop: "1rem" }}>
              <span className="alert-icon">🤖</span>
              <div className="alert-content">
                <div className="alert-title">AI Summary</div>
                <div className="alert-message">{result.summary}</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
