import React from "react";
import { NavLink } from "react-router-dom";

const NAV = [
  {
    section: "Overview",
    links: [
      { to: "/",              icon: "🏠", label: "Dashboard"       },
    ],
  },
  {
    section: "AI Pipeline",
    links: [
      { to: "/predict",        icon: "🧠", label: "Prediction"      },
      { to: "/simulation",     icon: "🗺️",  label: "Simulation"      },
      { to: "/recommendations",icon: "💡", label: "Recommendations" },
    ],
  },
  {
    section: "Operations",
    links: [
      { to: "/alerts",         icon: "🔔", label: "Alerts"          },
    ],
  },
];

const Sidebar = () => {
  return (
    <aside className="sidebar">
      {NAV.map((group) => (
        <div className="sidebar-section" key={group.section}>
          <div className="sidebar-section-label">{group.section}</div>
          <nav className="sidebar-nav">
            {group.links.map(({ to, icon, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  "sidebar-link" + (isActive ? " active" : "")
                }
              >
                <span className="sidebar-icon">{icon}</span>
                <span>{label}</span>
              </NavLink>
            ))}
          </nav>
        </div>
      ))}

      {/* Bottom version tag */}
      <div className="sidebar-bottom">
        <div className="sidebar-version">
          FloodSense AI v1.0.0<br />
          <span style={{ color: "var(--text-muted)" }}>
            Powered by Gemini 1.5 Pro
          </span>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
