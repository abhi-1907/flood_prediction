/**
 * AlertSubscriptionForm – Form for users to subscribe to flood alerts.
 * Collects contact info, location, preferred channels, and risk threshold.
 */
import React, { useState } from "react";

function AlertSubscriptionForm({ onSubscribe }) {
  const [form, setForm] = useState({
    name: "",
    email: "",
    phone: "",
    location: "",
    channels: ["email"],
    min_risk_level: "MODERATE",
  });

  // TODO: Implement form field handlers, validation, and submission
  return (
    <form className="alert-subscription-form">
      <h2>Subscribe to Flood Alerts</h2>
      {/* TODO: Input fields for name, email, phone, location, channels, threshold */}
    </form>
  );
}

export default AlertSubscriptionForm;
