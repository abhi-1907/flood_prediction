import React from "react";
import { BrowserRouter, Routes, Route, Link, Navigate } from "react-router-dom";
import Navbar from "./components/layout/Navbar";
import Sidebar from "./components/layout/Sidebar";
import Dashboard from "./pages/Dashboard";
import PredictionPage from "./pages/PredictionPage";
import RecommendationsPage from "./pages/RecommendationsPage";
import SimulationPage from "./pages/SimulationPage";
import AlertingPage from "./pages/AlertingPage";
import AlertsPage from "./pages/AlertsPage";
import { PipelineProvider } from "./context/PipelineContext";

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
      <PipelineProvider>
        <div className="app-shell">
          <Navbar />
          <div className="app-body">
            <Sidebar />
            <main className="app-main">
              <Routes>
                <Route path="/"                element={<Dashboard />}         />
                <Route path="/predict"         element={<PredictionPage />}    />
                <Route path="/recommendations" element={<RecommendationsPage />} />
                <Route path="/simulation"      element={<SimulationPage />}    />
                <Route path="/alerting"        element={<AlertingPage />}      />
                <Route path="/alerts"          element={<AlertsPage />}        />
                {/* Legacy redirect */}
                <Route path="/pipeline"        element={<Navigate to="/predict" replace />} />
                <Route path="*"               element={<NotFound />}           />
              </Routes>
            </main>
          </div>
        </div>
      </PipelineProvider>
    </BrowserRouter>
  );
}

export default App;
