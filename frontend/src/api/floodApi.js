/**
 * floodApi.js – Axios API client, consistent with all backend FastAPI routes.
 */
import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000",
  timeout: 60000,
  headers: { "Content-Type": "application/json" },
});

// Response interceptor – unwrap data
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const detail = err.response?.data?.detail || err.message;
    return Promise.reject(new Error(detail));
  }
);

// ── Health ─────────────────────────────────────────────────────────────────
export const healthCheck = () => api.get("/health");
export const deepHealthCheck = () => api.get("/health/deep");

// ── Orchestration ──────────────────────────────────────────────────────────
export const orchestrate = (payload) =>
  api.post("/orchestrate", payload);

export const orchestrateUpload = (formData) =>
  api.post("/orchestrate/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

export const getSessions = () => api.get("/orchestrate/sessions");
export const getSession  = (id) => api.get(`/orchestrate/sessions/${id}`);

// ── Data Ingestion ─────────────────────────────────────────────────────────
export const ingestText = (payload) => api.post("/ingest/text", payload);

export const ingestUpload = (formData) =>
  api.post("/ingest/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

export const ingestUrl   = (payload) => api.post("/ingest/url", payload);
export const getSources  = () => api.get("/ingest/sources");
export const getIngestionStatus = (sessionId) =>
  api.get(`/ingest/status/${sessionId}`);

// ── Prediction ─────────────────────────────────────────────────────────────
export const runPrediction      = (payload) => api.post("/predict", payload);
export const quickPredict       = (payload) => api.post("/predict/quick", payload);
export const getPrediction      = (sessionId) => api.get(`/predict/${sessionId}`);
export const getExplanation     = (sessionId) => api.get(`/predict/${sessionId}/explanation`);

// ── Recommendations ────────────────────────────────────────────────────────
export const getRecommendations = (payload) =>
  api.post("/recommendations", payload);
export const getRecommendation  = (sessionId) =>
  api.get(`/recommendations/${sessionId}`);
export const getSafetyMessage   = (sessionId) =>
  api.get(`/recommendations/${sessionId}/safety-message`);
export const getAuthorityBrief  = (sessionId) =>
  api.get(`/recommendations/${sessionId}/authority-brief`);

// ── Simulation ─────────────────────────────────────────────────────────────
export const runSimulation      = (payload) => api.post("/simulation", payload);
export const runScenario        = (payload) => api.post("/simulation/scenario", payload);
export const getSimulation      = (sessionId) => api.get(`/simulation/${sessionId}`);
export const getGeoJSON         = (sessionId) => api.get(`/simulation/${sessionId}/geojson`);
export const getTimeline        = (sessionId) => api.get(`/simulation/${sessionId}/timeline`);
export const getImpact          = (sessionId) => api.get(`/simulation/${sessionId}/impact`);

// ── Alerts ─────────────────────────────────────────────────────────────────
export const subscribeToAlerts  = (payload) => api.post("/alerts/subscribe", payload);
export const unsubscribe        = (id) => api.delete(`/alerts/unsubscribe/${id}`);
export const triggerAlert       = (payload) => api.post("/alerts/trigger", payload);
export const getAlertForSession = (sessionId) => api.get(`/alerts/${sessionId}`);
export const getActiveAlerts    = () => api.get("/alerts/active");
export const getSubscribers     = () => api.get("/alerts/subscribers");
export const bulkImport         = (payload) => api.post("/alerts/import", payload);

export default api;
