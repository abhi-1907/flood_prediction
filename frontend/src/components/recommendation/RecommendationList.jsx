import React from "react";

const CATEGORY_ICON = {
  evacuation:   "🚨",
  safety:       "🛡️",
  medical:      "🏥",
  resource:     "📦",
  communication:"📡",
  shelter:      "🏠",
  water:        "💧",
  default:      "📋",
};

const PRIORITY_STYLE = {
  1:         { color: "var(--risk-high)",     label: "Immediate" },
  2:         { color: "var(--risk-medium)",   label: "Soon"      },
  3:         { color: "var(--risk-low)",      label: "Monitor"   },
  immediate: { color: "var(--risk-high)",     label: "Immediate" },
  soon:      { color: "var(--risk-medium)",   label: "Soon"      },
  monitor:   { color: "var(--risk-low)",      label: "Monitor"   },
};

function RecItem({ item }) {
  const icon  = CATEGORY_ICON[item.category?.toLowerCase?.() || item.category] || CATEGORY_ICON.default;
  const prioKey = item.priority != null ? String(item.priority).toLowerCase() : "monitor";
  const prio  = PRIORITY_STYLE[prioKey] || PRIORITY_STYLE.monitor;

  return (
    <div className="rec-item animate-up">
      <div className="rec-icon">{icon}</div>
      <div style={{ flex: 1 }}>
        <div className="flex items-center gap-2 mb-1">
          <span className="rec-title">{item.title}</span>
          <span
            className="badge"
            style={{ background: prio.color + "22", color: prio.color, fontSize: "0.65rem" }}
          >
            {prio.label}
          </span>
        </div>
        <p className="rec-desc">{item.description}</p>
        {item.action_steps?.length > 0 && (
          <div className="rec-actions">
            {item.action_steps.map((step, i) => (
              <span key={i} className="rec-action">✓ {step}</span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function RecommendationList({ recommendations = [], summary, safetyMessage, authorityBrief }) {
  if (!recommendations.length) {
    return (
      <div className="empty-state">
        <div className="empty-icon">💡</div>
        <div className="empty-text">No recommendations yet. Run a prediction first.</div>
      </div>
    );
  }

  // Group by category
  const groups = recommendations.reduce((acc, r) => {
    const key = r.category || "other";
    if (!acc[key]) acc[key] = [];
    acc[key].push(r);
    return acc;
  }, {});

  return (
    <div>
      {summary && (
        <div className="alert alert-info mb-4">
          <span className="alert-icon">🧠</span>
          <div className="alert-content">
            <div className="alert-title">AI Summary</div>
            <div className="alert-message">{summary}</div>
          </div>
        </div>
      )}

      {Object.entries(groups).map(([cat, items]) => (
        <div key={cat} className="section">
          <div className="section-title">
            {CATEGORY_ICON[cat] || "📋"} {cat.charAt(0).toUpperCase() + cat.slice(1)}
          </div>
          <div className="flex flex-col gap-3">
            {items.map((item, i) => <RecItem key={i} item={item} />)}
          </div>
        </div>
      ))}

      {safetyMessage && (
        <div className="card" style={{ borderColor: "var(--risk-low)40", marginTop: "1.5rem" }}>
          <div className="card-title" style={{ marginBottom: "0.75rem" }}>📱 SMS Safety Message</div>
          <p style={{
            fontFamily: "'JetBrains Mono', monospace", fontSize: "0.825rem",
            color: "var(--text-secondary)", lineHeight: 1.7,
            background: "var(--bg-elevated)", padding: "0.875rem",
            borderRadius: "var(--radius-sm)",
          }}>
            {safetyMessage}
          </p>
        </div>
      )}

      {authorityBrief && (
        <div className="card" style={{ borderColor: "var(--accent)40", marginTop: "1rem" }}>
          <div className="card-title" style={{ marginBottom: "0.75rem" }}>🏛️ Authority Brief</div>
          <p style={{ fontSize: "0.825rem", color: "var(--text-secondary)", lineHeight: 1.75, whiteSpace: "pre-wrap" }}>
            {authorityBrief}
          </p>
        </div>
      )}
    </div>
  );
}
