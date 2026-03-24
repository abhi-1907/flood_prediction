/**
 * DataUploader – Allows users to upload CSV files or enter a free-text query
 * to trigger the data ingestion and prediction pipeline.
 */
import React from "react";

function DataUploader({ onSubmit }) {
  // TODO: File drop zone, text query input, submit handling
  return (
    <div className="data-uploader">
      <p>Upload CSV or enter a location query to begin prediction.</p>
    </div>
  );
}

export default DataUploader;
