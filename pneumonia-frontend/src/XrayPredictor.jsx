import React, { useState } from "react";

function XrayPredictor() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!file) {
      alert("Please upload an X-ray image");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      alert("Flask API not running");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mt-5">
      <h2 className="text-center mb-4">
        Chest X-Ray Pneumonia Detection
      </h2>

      <input
        type="file"
        className="form-control mb-3"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <button className="btn btn-primary w-100" onClick={handlePredict}>
        {loading ? "Predicting..." : "Predict"}
      </button>

      {result && (
        <div className="alert alert-info mt-4 text-center">
          <h4>Prediction: {result.prediction}</h4>
          <p>Confidence: {result.confidence}%</p>
        </div>
      )}
    </div>
  );
}

export default XrayPredictor;
