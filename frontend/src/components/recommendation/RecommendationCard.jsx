/**
 * RecommendationCard – Displays a single LLM-generated recommendation block
 * with appropriate styling based on the user type (public vs. authority).
 */
import React from "react";

function RecommendationCard({ recommendation, userType }) {
  // TODO: Render formatted recommendation with icons, safe-zone list, contacts
  return (
    <div className={`recommendation-card user-${userType}`}>
      <pre className="recommendation-text">{recommendation}</pre>
    </div>
  );
}

export default RecommendationCard;
