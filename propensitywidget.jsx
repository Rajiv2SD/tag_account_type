import React, { useState } from 'react';

const PropensityWidget = () => {
  // 1. State Management
  const [formData, setFormData] = useState({
    revenue: 1000000,
    employees: 50,
    country: "United Kingdom",
    industry: "Manufacturing"
  });

  const [predictionData, setPredictionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // 2. Options (Sync these with your Python Backend lists)
  const countries = [
    "France", "United Kingdom", "Italy", "Spain", "United Arab Emirates", 
    "Saudi Arabia", "Nigeria", "Egypt", "South Africa", "United States"
  ];

  const industries = [
    "Manufacturing", "Retail & Wholesale", "Professional Services", 
    "Built Environment & Construction", "Agri Food", 
    "IT, Communication & Media Services", "Energy (Electricity, Oil & Gas)", 
    "Healthcare", "Logistics, Transport & Distribution", 
    "Hospitality & Leisure", "Test Account"
  ];

  // 3. API Call Handler
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPredictionData(null);

    try {
      // ⚠️ UPDATE THIS URL to your actual FastAPI endpoint
      const API_URL = "http://localhost:8000/predict"; 

      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch prediction");
      }

      const result = await response.json();
      setPredictionData(result);
    } catch (err) {
      setError("Error connecting to the AI Engine. Please check your connection.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // 4. UI Helper: Color Coding
  const getBadgeStyle = (stage) => {
    switch (stage) {
      case "Target":
      case "Client":
        return "bg-green-100 text-green-800 border-green-200";
      case "Deactivated":
        return "bg-red-100 text-red-800 border-red-200";
      case "Free Account":
        return "bg-blue-100 text-blue-800 border-blue-200";
      default: // Prospect
        return "bg-yellow-100 text-yellow-800 border-yellow-200";
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6 max-w-2xl mx-auto font-sans">
      
      {/* Header */}
      <div className="mb-6 border-b pb-4">
        <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <span>🔮</span> Hybrid Propensity Engine
        </h2>
        <p className="text-sm text-gray-500 mt-1">
          Combines <strong>XGBoost AI</strong> with <strong>Strategic Business Rules</strong>.
        </p>
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="space-y-5">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {/* Revenue */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-1">Annual Revenue ($)</label>
            <input
              type="number"
              min="0"
              value={formData.revenue}
              onChange={(e) => setFormData({ ...formData, revenue: Number(e.target.value) })}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition"
            />
          </div>

          {/* Employees */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-1">Employees</label>
            <input
              type="number"
              min="1"
              value={formData.employees}
              onChange={(e) => setFormData({ ...formData, employees: Number(e.target.value) })}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition"
            />
          </div>

          {/* Country */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-1">Country</label>
            <select
              value={formData.country}
              onChange={(e) => setFormData({ ...formData, country: e.target.value })}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition bg-white"
            >
              {countries.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>

          {/* Industry */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-1">Industry</label>
            <select
              value={formData.industry}
              onChange={(e) => setFormData({ ...formData, industry: e.target.value })}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition bg-white"
            >
              {industries.map((ind) => <option key={ind} value={ind}>{ind}</option>)}
            </select>
          </div>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading}
          className={`w-full py-3 px-4 rounded-lg text-white font-semibold shadow-md transition-all 
            ${loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 active:scale-[0.98]'}`}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Thinking...
            </span>
          ) : "Generate Prediction"}
        </button>
      </form>

      {/* Error Message */}
      {error && (
        <div className="mt-6 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg flex items-center gap-2">
          ⚠️ {error}
        </div>
      )}

      {/* Results Display */}
      {predictionData && (
        <div className="mt-8 border-t border-gray-200 pt-6 animate-fade-in">
          
          {/* Header & Source Badge */}
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
            <div>
               <h3 className="text-gray-500 text-sm font-medium uppercase tracking-wide">Likeliest Outcome</h3>
               <div className={`mt-1 inline-flex items-center px-4 py-1.5 rounded-full border text-lg font-bold ${getBadgeStyle(predictionData.prediction)}`}>
                  {predictionData.prediction}
               </div>
            </div>
            
            <div className="text-right">
              <div className="text-sm text-gray-500 mb-1">Confidence Score</div>
              <div className="text-2xl font-black text-gray-900">
                {(predictionData.confidence * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Logic Source Pill */}
          <div className="mb-6">
             <span className={`px-3 py-1 rounded text-xs font-semibold ${predictionData.source.includes("Rule") ? "bg-blue-50 text-blue-700 border border-blue-200" : "bg-purple-50 text-purple-700 border border-purple-200"}`}>
               {predictionData.source.includes("Rule") ? "🛡️ " : "🤖 "}{predictionData.source}
             </span>
          </div>

          {/* Probability Bar Chart */}
          <div>
            <h4 className="text-sm font-semibold text-gray-700 mb-3">Probability Distribution</h4>
            <div className="space-y-3">
              {Object.entries(predictionData.probabilities)
                .sort(([, a], [, b]) => b - a) // Sort desc
                .map(([stage, prob]) => (
                <div key={stage} className="flex items-center text-sm">
                  <span className="w-28 text-gray-600 font-medium truncate" title={stage}>{stage}</span>
                  <div className="flex-1 h-3 bg-gray-100 rounded-full mx-3 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ease-out ${
                        stage === predictionData.prediction ? 'bg-blue-600' : 'bg-gray-400 opacity-40'
                      }`}
                      style={{ width: `${prob * 100}%` }}
                    ></div>
                  </div>
                  <span className="w-12 text-right font-mono text-gray-500">
                    {(prob * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

        </div>
      )}
    </div>
  );
};

export default PropensityWidget;