# CRM Account Auto-Tagging API

**Authenticated Machine-Learning API for CRM Lifecycle Prediction**

This service provides a secure, FastAPI-based API for automatically predicting CRM lifecycle stages (e.g., Target, Prospect, Client) along with probability distribution and next-best-action recommendations.

It replaces the previous Streamlit UI with a fully automated REST backend ready for integration into CRM systems, data platforms, sales engines, and automation workflows.

---

## ⚙️ Features

- ✔️ **Secure** — Basic Authentication (username/password)
- ✔️ Validates input for correctness (blocking errors)
- ✔️ Detects suspicious data combinations (non-blocking warnings)
- ✔️ Performs full feature engineering + preprocessing
- ✔️ Predicts stage probabilities using an XGBoost model
- ✔️ Returns next-best-action insights
- ✔️ Health-check endpoint for monitoring

---

## 📁 Project Structure

```
.
├── account_autotagging_API.py     # FastAPI application (with Basic Auth)
├── propensity_engine.json         # XGBoost model
├── preprocessor.joblib            # Preprocessing pipeline
├── label_encoder.joblib           # Label encoder
├── feature_names.joblib           # Optional feature schema
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## 🔐 Authentication

This API is protected using **HTTP Basic Authentication**.

### Default credentials:
- **Username:** `admin`
- **Password:** `S!tr0ngP@ssw0rd#2025`

### ⚠️ IMPORTANT:
Change these immediately in production or set environment variables:

**Linux/macOS:**
```bash
export API_AUTH_USERNAME="myuser"
export API_AUTH_PASSWORD="MyStrongPassword123!"
```

**Windows PowerShell:**
```powershell
$env:API_AUTH_USERNAME="myuser"
$env:API_AUTH_PASSWORD="MyStrongPassword123!"
```

---

## 📦 Installation

### 1️⃣ Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the API

### Local development
```bash
python account_autotagging_API.py
```

### Recommended Uvicorn command
```bash
uvicorn account_autotagging_API:app --host 127.0.0.1 --port 8000
```

---

## 🌐 API Documentation

Once running:

- **Swagger UI** → http://127.0.0.1:8000/docs
- **ReDoc** → http://127.0.0.1:8000/redoc
- **Health** → http://127.0.0.1:8000/health

> **Note:** Swagger requires entering Basic Auth credentials using the **Authorize** button.

---

## 📘 API Endpoints

### ✅ GET `/health`

Simple status indicator for uptime checks.

**Response:**
```json
{ "status": "ok" }
```

---

### 🔐 POST `/predict`
**(Authentication Required)**

Predicts CRM lifecycle stage with probabilities and insights.

#### 📥 Request Body
```json
{
  "revenue": 1500000,
  "employees": 50,
  "country": "United Kingdom",
  "industry": "Manufacturing"
}
```

---

## 🔎 Validation Rules

### Blocking Errors (API returns 400)

- ❌ Invalid country (must match allowed list)
- ❌ Invalid industry
- ❌ `revenue` ≤ 0
- ❌ `employees` < 1

### ✔️ Allowed Countries
- France
- United Kingdom
- Italy
- Spain
- United Arab Emirates
- Saudi Arabia
- Nigeria
- Egypt
- South Africa

### ✔️ Allowed Industries
- Manufacturing
- Retail & Wholesale
- Professional Services
- Built Environment & Construction
- Others
- Agri Food
- IT, Communication & Media Services
- Energy (Electricity, Oil & Gas)
- Healthcare
- Logistics, Transport & Distribution
- Hospitality & Leisure

---

## ⚠️ Suspicious Data Warnings (Non-Blocking)

Warnings appear in the response but do **NOT** block predictions.

**Examples include:**
- Revenue > 1B with < 5 employees
- Revenue < 1,000 with > 1,000 employees
- Revenue per employee > 10M
- Revenue per employee < 100

---

## 📤 Successful Prediction Response

```json
{
  "top_stage": "Target",
  "top_probability": 0.72,
  "probabilities": [
    { "stage": "Target", "probability": 0.72 },
    { "stage": "Prospect", "probability": 0.18 },
    { "stage": "Client", "probability": 0.08 },
    { "stage": "Free Account", "probability": 0.02 },
    { "stage": "Deactivated", "probability": 0.01 }
  ],
  "next_best_action": "High Value Fit. Immediate sales outreach recommended.",
  "raw_features": {
    "log_revenue": 14.225,
    "log_num_employees": 3.912,
    "revenue_per_employee": 30000,
    "address1_country": "United Kingdom",
    "industrycode_display": "Manufacturing",
    "qg_annualrevenue_display": "A  0-20 Million",
    "qg_numberofemployees_display": "A 1-50"
  },
  "warnings": []
}
```

---

## ❌ Example: Blocking Validation Error Response

**Request:**
```json
{
  "revenue": 1010000,
  "employees": 0,
  "country": "djkf",
  "industry": "adf"
}
```

**Response (400 Bad Request):**
```json
{
  "detail": {
    "valid": false,
    "errors": [
      "Invalid country: 'djkf'. Allowed countries: [...]",
      "Invalid industry: 'adf'. Allowed industries: [...]",
      "Invalid employees: must be at least 1."
    ],
    "warnings": []
  }
}
```

---

## 🧪 Example Authenticated Calls

### cURL
```bash
curl -u admin:'S!tr0ngP@ssw0rd#2025' \
  -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{ \"revenue\":1500000, \"employees\":50, \"country\":\"United Kingdom\", \"industry\":\"Manufacturing\" }"
```

### Python
```python
import requests
from requests.auth import HTTPBasicAuth

payload = {
    "revenue": 1500000,
    "employees": 50,
    "country": "United Kingdom",
    "industry": "Manufacturing"
}

r = requests.post(
    "http://127.0.0.1:8000/predict",
    json=payload,
    auth=HTTPBasicAuth("admin", "S!tr0ngP@ssw0rd#2025")
)

print(r.json())
```

---