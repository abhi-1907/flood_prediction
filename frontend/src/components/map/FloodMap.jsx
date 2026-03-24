import React, { useEffect, useRef } from "react";
import { MapContainer, TileLayer, GeoJSON, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet"; // CSS loaded globally in index.html

// Fix leaflet default marker icons with Vite
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl:       "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl:     "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

const SEVERITY_STYLE = {
  extreme: { color: "#a855f7", fillColor: "#a855f7", fillOpacity: 0.55, weight: 0 },
  severe:  { color: "#ef4444", fillColor: "#ef4444", fillOpacity: 0.45, weight: 0 },
  moderate:{ color: "#f59e0b", fillColor: "#f59e0b", fillOpacity: 0.35, weight: 0 },
  minor:   { color: "#22c55e", fillColor: "#22c55e", fillOpacity: 0.25, weight: 0 },
};

const LEGEND_ITEMS = [
  { label: "Extreme (>3m)",   color: "#a855f7" },
  { label: "Severe (1–3m)",   color: "#ef4444" },
  { label: "Moderate (0.5–1m)", color: "#f59e0b" },
  { label: "Minor (<0.5m)",   color: "#22c55e" },
];

function ChangeView({ center, zoom }) {
  const map = useMap();
  useEffect(() => { map.setView(center, zoom); }, [center, zoom, map]);
  return null;
}

export default function FloodMap({ geojson, center = [20.5937, 78.9629], zoom = 6, height = 420 }) {
  const styleFeature = (feature) => {
    const sev = feature?.properties?.severity?.toLowerCase() || "minor";
    return SEVERITY_STYLE[sev] || SEVERITY_STYLE.minor;
  };

  const onEachFeature = (feature, layer) => {
    const p = feature.properties || {};
    layer.bindTooltip(
      `<div style="font-size:12px;line-height:1.6">
         <b>Severity:</b> ${p.severity || "—"}<br/>
         <b>Depth:</b> ${p.depth_m != null ? p.depth_m.toFixed(2) + " m" : "—"}<br/>
         <b>Land use:</b> ${p.land_use || "—"}
       </div>`,
      { sticky: true, className: "leaflet-tooltip-flood" }
    );
  };

  return (
    <div className="map-container" style={{ height }}>
      <MapContainer
        center={center}
        zoom={zoom}
        style={{ height: "100%", width: "100%", background: "#0d1521" }}
        attributionControl={false}
      >
        <ChangeView center={center} zoom={zoom} />
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; CartoDB'
        />
        {geojson && (
          <GeoJSON
            key={JSON.stringify(center)}
            data={geojson}
            style={styleFeature}
            onEachFeature={onEachFeature}
          />
        )}
        {center && (
          <Marker position={center}>
            <Popup>Simulation Centre</Popup>
          </Marker>
        )}
      </MapContainer>

      {/* Legend overlay */}
      <div className="map-legend">
        <div style={{ fontSize: "0.7rem", fontWeight: 700, marginBottom: "0.5rem", color: "var(--text-secondary)" }}>
          Flood Severity
        </div>
        {LEGEND_ITEMS.map((item) => (
          <div className="legend-item" key={item.label}>
            <div className="legend-swatch" style={{ background: item.color, opacity: 0.7 }} />
            <span style={{ color: "var(--text-secondary)" }}>{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
