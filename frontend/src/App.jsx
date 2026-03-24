import React from "react";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Navbar from "./components/layout/Navbar";
import Sidebar from "./components/layout/Sidebar";
import Dashboard from "./pages/Dashboard";
import PredictionPage from "./pages/PredictionPage";
import SimulationPage from "./pages/SimulationPage";
import RecommendationsPage from "./pages/RecommendationsPage";
import AlertsPage from "./pages/AlertsPage";

/** 404 Not Found page */
function NotFound() {
  return (
    <div style={{ textAlign: "center", padding: "4rem 2rem" }}>
      <div style={{ fontSize: "5rem", marginBottom: "1rem" }}>🌊</div>
      <h1 style={{ fontSize: "2rem", fontWeight: 800, marginBottom: "0.5rem" }}>
        404 – Page Not Found
      </h1>
      <p style={{ color: "var(--text-muted)", marginBottom: "2rem" }}>
        This page doesn't exist. The flood goes elsewhere.
      </p>
      <Link to="/" className="btn btn-primary">
        ← Back to Dashboard
      </Link>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Navbar />
        <div className="app-body">
          <Sidebar />
          <main className="app-main">
            <Routes>
              <Route path="/"                element={<Dashboard />}         />
              <Route path="/predict"         element={<PredictionPage />}    />
              <Route path="/simulation"      element={<SimulationPage />}    />
              <Route path="/recommendations" element={<RecommendationsPage />} />
              <Route path="/alerts"          element={<AlertsPage />}        />
              <Route path="*"               element={<NotFound />}          />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App;
