/**
 * FloodZoneOverlay – Renders GeoJSON flood zone polygons as a Leaflet overlay.
 * Applies colour-coded styling based on the risk severity level.
 */
import React from "react";

const RISK_COLORS = {
  LOW: "#4CAF50",
  MODERATE: "#FF9800",
  HIGH: "#F44336",
  EXTREME: "#9C27B0",
};

function FloodZoneOverlay({ geojsonData }) {
  // TODO: Integrate react-leaflet GeoJSON component with RISK_COLORS style function
  return null;
}

export default FloodZoneOverlay;
