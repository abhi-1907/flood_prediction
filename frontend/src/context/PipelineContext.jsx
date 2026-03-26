/**
 * PipelineContext.jsx
 * Global shared state for the AI flood pipeline.
 * The Prediction page calls runPipeline(); all other agent pages read results from here.
 */
import React, { createContext, useContext, useReducer, useRef, useCallback } from "react";

const API_BASE =
  import.meta.env.VITE_API_URL ||
  `${window.location.protocol}//${window.location.hostname}:8000`;

// ── Initial state ─────────────────────────────────────────────────────────────
const INIT = {
  running: false,
  done: false,
  error: null,
  stepStatus: {},   // agent key → "idle"|"active"|"done"|"skipped"|"error"
  stepDetail: {},   // agent key → string message
  prediction: null,
  recommendations: null,
  recSummary: null,
  recSafetyMsg: null,
  simulation: null,
  alertStatus: null,
  elapsed: null,
  sessionId: null,
};

// ── Reducer ───────────────────────────────────────────────────────────────────
function reducer(state, action) {
  switch (action.type) {
    case "RESET":
      return { ...INIT };
    case "SET_RUNNING":
      return { ...state, running: action.value, error: null };
    case "SET_DONE":
      return { ...state, done: action.value, running: false };
    case "SET_ERROR":
      return { ...state, error: action.value, running: false };
    case "STEP_STATUS":
      return { ...state, stepStatus: { ...state.stepStatus, [action.key]: action.status } };
    case "STEP_DETAIL":
      return { ...state, stepDetail: { ...state.stepDetail, [action.key]: action.detail } };
    case "SET_PREDICTION":
      return { ...state, prediction: action.data };
    case "SET_RECOMMENDATIONS":
      return {
        ...state,
        recommendations: action.data.recommendations ?? state.recommendations,
        recSummary: action.data.summary ?? state.recSummary,
        recSafetyMsg: action.data.safety_message ?? state.recSafetyMsg,
      };
    case "SET_SIMULATION":
      return { ...state, simulation: action.data };
    case "SET_ALERT_STATUS":
      return { ...state, alertStatus: action.data };
    case "SET_ELAPSED":
      return { ...state, elapsed: action.value };
    case "SET_SESSION":
      return { ...state, sessionId: action.value };
    default:
      return state;
  }
}

// ── Context ───────────────────────────────────────────────────────────────────
export const PipelineContext = createContext(null);

export function usePipeline() {
  const ctx = useContext(PipelineContext);
  if (!ctx) throw new Error("usePipeline must be used inside <PipelineProvider>");
  return ctx;
}

// ── Provider ──────────────────────────────────────────────────────────────────
export function PipelineProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, INIT);
  const abortRef = useRef(null);

  // Apply a single agent's result data
  const applyAgentData = useCallback((key, data) => {
    if (!data) return;
    if (key === "prediction") {
      const flat = {
        ...data,
        ...(data.ensemble || {}),
        explanation: typeof data.explanation === 'string' ? data.explanation : (data.explanation?.llm_narrative || data.explanation?.summary || "")
      };
      dispatch({ type: "SET_PREDICTION", data: flat });
    }
    if (key === "recommendation" || key === "recommendations") {
      dispatch({ type: "SET_RECOMMENDATIONS", data });
    }
    if (key === "simulation") {
      dispatch({ type: "SET_SIMULATION", data });
    }
    if (key === "alerting") {
      dispatch({ type: "SET_ALERT_STATUS", data });
    }
  }, []);

  // Apply the final orchestration result blob
  const applyOrchestrationResult = useCallback((data) => {
    if (data.prediction) {
      const p = data.prediction;
      const flat = {
        ...p,
        ...(p.ensemble || {}),
        explanation: typeof p.explanation === 'string' ? p.explanation : (p.explanation?.llm_narrative || p.explanation?.summary || "")
      };
      dispatch({ type: "SET_PREDICTION", data: flat });
    }
    if (data.simulation)      dispatch({ type: "SET_SIMULATION",   data: data.simulation });
    if (data.alert_status)    dispatch({ type: "SET_ALERT_STATUS", data: data.alert_status });
    if (data.elapsed_seconds) dispatch({ type: "SET_ELAPSED",      value: data.elapsed_seconds });
    if (data.session_id)      dispatch({ type: "SET_SESSION",      value: data.session_id });

    if (data.recommendations) {
      const r = data.recommendations;
      if (typeof r === "string") {
        dispatch({ type: "SET_RECOMMENDATIONS", data: { summary: r } });
      } else {
        dispatch({ type: "SET_RECOMMENDATIONS", data: r });
      }
    }

    // Mark all steps from steps_summary
    (data.steps_summary || []).forEach((step) => {
      const key = step.agent?.toLowerCase().replace(/\s+/g, "_") || "";
      if (key) {
        dispatch({ type: "STEP_STATUS", key, status: step.status === "completed" ? "done" : step.status || "done" });
        dispatch({ type: "STEP_DETAIL", key, detail: step.summary || "" });
      }
    });
  }, []);

  // Handle a single SSE event
  const handleSSEEvent = useCallback((evt) => {
    const { event, type, agent, status, data, message, elapsed_seconds, session_id } = evt;

    // The backend uses "event" field; normalise to "type" for compatibility
    const evtType = event || type;

    if (session_id)      dispatch({ type: "SET_SESSION", value: session_id });
    if (elapsed_seconds) dispatch({ type: "SET_ELAPSED", value: elapsed_seconds });

    const agentKey = agent?.toLowerCase().replace(/\s+/g, "_") || evtType;

    // ── Backend SSE event types (from orchestrator.stream) ─────────────────
    if (evtType === "step_started") {
      const key = (data?.agent || agent || "").toLowerCase().replace(/\s+/g, "_");
      if (key) {
        dispatch({ type: "STEP_STATUS", key, status: "active" });
        dispatch({ type: "STEP_DETAIL", key, detail: data?.action || "Running…" });
      }
    }
    if (evtType === "step_done") {
      const key = (data?.agent || agent || "").toLowerCase().replace(/\s+/g, "_");
      const st  = data?.status === "SUCCEEDED" ? "done" : data?.status === "FAILED" ? "error" : "done";
      if (key) {
        dispatch({ type: "STEP_STATUS", key, status: st });
        dispatch({ type: "STEP_DETAIL", key, detail: st === "error" ? "Failed" : "Completed" });
      }
    }
    if (evtType === "plan_ready" && data) {
      // Mark all planned steps as idle/queued
      (data.plan || []).forEach((step) => {
        const key = step.agent?.toLowerCase().replace(/\s+/g, "_");
        if (key) dispatch({ type: "STEP_STATUS", key, status: "idle" });
      });
    }

    // ── Legacy / alternative event types ──────────────────────────────────
    if (evtType === "agent_start") {
      dispatch({ type: "STEP_STATUS", key: agentKey, status: "active" });
      dispatch({ type: "STEP_DETAIL", key: agentKey, detail: message || "" });
    }
    if (evtType === "agent_complete") {
      dispatch({ type: "STEP_STATUS", key: agentKey, status: "done" });
      dispatch({ type: "STEP_DETAIL", key: agentKey, detail: message || "Completed" });
      applyAgentData(agentKey, data);
    }
    if (evtType === "agent_skip") {
      dispatch({ type: "STEP_STATUS", key: agentKey, status: "skipped" });
      dispatch({ type: "STEP_DETAIL", key: agentKey, detail: message || "Skipped" });
    }
    if (evtType === "agent_error") {
      dispatch({ type: "STEP_STATUS", key: agentKey, status: "error" });
      dispatch({ type: "STEP_DETAIL", key: agentKey, detail: message || "Error" });
    }
    if (evtType === "complete" && data) {
      applyOrchestrationResult(data);
    }
  }, [applyAgentData, applyOrchestrationResult]);

  // ── runPipeline ─────────────────────────────────────────────────────────────
  const runPipeline = useCallback(async ({ 
    inputMode, query, userType, location, rain, rainMonthly, temp, humidity, wind, 
    elevation, slope, terrain, river, waterbody, damCount, file 
  }) => {
    // Abort any previous run
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();

    dispatch({ type: "RESET" });
    dispatch({ type: "SET_RUNNING", value: true });

    // Build context payload based on mode
    let finalQuery = "";
    let finalContext = null;

    if (inputMode === "nlp") {
      finalQuery = query?.trim();
    } else if (inputMode === "manual") {
      finalQuery = `Analyse flood risk for ${location || 'unknown'}`;
      finalContext = {};
      if (location)    finalContext.location             = location;
      if (rain)        finalContext.rain_mm_weekly        = parseFloat(rain);
      if (rainMonthly) finalContext.rain_mm_monthly       = parseFloat(rainMonthly);
      if (temp)        finalContext.temp_c_mean           = parseFloat(temp);
      if (humidity)    finalContext.rh_percent_mean       = parseFloat(humidity);
      if (wind)        finalContext.wind_ms_mean           = parseFloat(wind);
      
      if (elevation)   finalContext.elevation_m           = parseFloat(elevation);
      if (slope)       finalContext.slope_degree          = parseFloat(slope);
      if (terrain)     finalContext.terrain_type          = terrain;
      
      if (river)       finalContext.dist_major_river_km   = parseFloat(river);
      if (waterbody)   finalContext.waterbody_nearby      = waterbody; // boolean
      if (damCount)    finalContext.dam_count_50km        = parseInt(damCount);

      // Force downstream agents for manual risk analysis
      finalContext.wants_recommendations = true;
      finalContext.wants_simulation      = true;
    } else if (inputMode === "file") {
      finalQuery = query?.trim() || `Analyse data from uploaded file`;
    }

    try {
      let response;

      if (inputMode === "file" && file) {
        // multipart file upload — non-streaming
        const form = new FormData();
        form.append("input_mode", "file");
        form.append("query",     finalQuery);
        form.append("user_type", userType);
        form.append("file",      file);
        response = await fetch(`${API_BASE}/orchestrate/upload`, {
          method: "POST",
          body: form,
          signal: abortRef.current.signal,
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || `Server error ${response.status}`);
        }

        const data = await response.json();
        applyOrchestrationResult(data);
        dispatch({ type: "SET_DONE", value: true });
        return;
      }

      // SSE streaming
      response = await fetch(`${API_BASE}/orchestrate/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input_mode: inputMode,
          query: finalQuery,
          user_type: userType,
          context: finalContext,
        }),
        signal: abortRef.current.signal,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${response.status}`);
      }

      const reader  = response.body.getReader();
      const decoder = new TextDecoder();
      let   buf     = "";

      while (true) {
        const { done: streamDone, value } = await reader.read();
        if (streamDone) break;
        buf += decoder.decode(value, { stream: true });

        const lines = buf.split("\n");
        buf = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (payload === "[DONE]") {
            dispatch({ type: "SET_DONE", value: true });
            break;
          }
          try {
            handleSSEEvent(JSON.parse(payload));
          } catch { /* skip malformed */ }
        }
      }

      dispatch({ type: "SET_DONE", value: true });
    } catch (err) {
      if (err.name !== "AbortError") {
        dispatch({ type: "SET_ERROR", value: err.message });
      }
    }
  }, [handleSSEEvent, applyOrchestrationResult]);

  const abortPipeline = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return (
    <PipelineContext.Provider value={{ ...state, runPipeline, abortPipeline }}>
      {children}
    </PipelineContext.Provider>
  );
}
