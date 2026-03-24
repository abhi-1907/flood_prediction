import React, { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { healthCheck } from "../../api/floodApi";

const Navbar = () => {
  const [status, setStatus] = useState("loading"); // loading | online | offline
  const [uptime, setUptime] = useState(null);
  const location = useLocation();

  useEffect(() => {
    const check = async () => {
      try {
        const res = await healthCheck();
        setStatus("online");
        setUptime(res.data.uptime_seconds);
      } catch {
        setStatus("offline");
      }
    };
    check();
    const id = setInterval(check, 30000);
    return () => clearInterval(id);
  }, []);

  const statusLabel =
    status === "online"  ? "Backend Online"  :
    status === "offline" ? "Backend Offline" : "Connecting…";

  const formatUptime = (s) => {
    if (!s) return "";
    if (s < 60)   return `${Math.round(s)}s uptime`;
    if (s < 3600) return `${Math.round(s / 60)}m uptime`;
    return `${Math.round(s / 3600)}h uptime`;
  };

  return (
    <nav className="navbar">
      {/* Brand */}
      <Link to="/" className="navbar-brand">
        <span className="brand-wave">🌊</span>
        <span className="navbar-brand-text">FloodSense AI</span>
      </Link>

      {/* Centre – route breadcrumb */}
      <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
        {location.pathname === "/" ? "Dashboard" :
          location.pathname.replace("/", "").replace(/^\w/, c => c.toUpperCase())}
      </div>

      {/* Right */}
      <div className="navbar-right">
        <div
          className="navbar-status"
          data-tooltip={uptime ? formatUptime(uptime) : undefined}
        >
          <span className={`status-dot ${status}`} />
          {statusLabel}
        </div>
        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noreferrer"
          className="btn btn-ghost btn-sm"
        >
          API Docs ↗
        </a>
      </div>
    </nav>
  );
};

export default Navbar;
