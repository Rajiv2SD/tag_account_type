"""
account_autotagging_API.py

FastAPI service for CRM account auto-tagging / propensity scoring.

- Adds HTTP Basic Authentication for protected endpoints (predict).
- Default credentials read from environment vars:
    API_AUTH_USERNAME (default: "admin")
    API_AUTH_PASSWORD (default: a strong default - change it!)

Security NOTE: Use HTTPS in production to protect credentials in transit.
"""

from typing import List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field, conint, confloat
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import secrets

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("propensity_api")

# ---------- Allowed lists (match your Streamlit selectboxes) ----------
ALLOWED_COUNTRIES = {
    "France",
    "United Kingdom",
    "Italy",
    "Spain",
    "United Arab Emirates",
    "Saudi Arabia",
    "Nigeria",
    "Egypt",
    "South Africa",
}

ALLOWED_INDUSTRIES = {
    "Manufacturing",
    "Retail & Wholesale",
    "Professional Services",
    "Built Environment & Construction",
    "Others",
    "Agri Food",
    "IT, Communication & Media Services",
    "Energy (Electricity, Oil & Gas)",
    "Healthcare",
    "Logistics, Transport & Distribution",
    "Hospitality & Leisure",
}

# ---------- Auth config ----------
# Default credentials (change them in production!)
DEFAULT_AUTH_USERNAME = "admin"
DEFAULT_AUTH_PASSWORD = "admin123"  # change this immediately in prod

API_AUTH_USERNAME = os.getenv("API_AUTH_USERNAME", DEFAULT_AUTH_USERNAME)
API_AUTH_PASSWORD = os.getenv("API_AUTH_PASSWORD", DEFAULT_AUTH_PASSWORD)

security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """
    Verifies HTTP Basic credentials; raises 401 if invalid.
    Returns the username when valid.
    """
    supplied_username = credentials.username
    supplied_password = credentials.password

    # Use constant-time comparison
    is_user = secrets.compare_digest(supplied_username, API_AUTH_USERNAME)
    is_pass = secrets.compare_digest(supplied_password, API_AUTH_PASSWORD)

    if not (is_user and is_pass):
        # Ask client for credentials
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return supplied_username

# ---------- FastAPI app ----------
app = FastAPI(
    title="CRM Propensity Engine API (Authenticated)",
    description="Predict lifecycle stage + conversion probability from account features (requires Basic Auth)",
    version="1.3.0",
)

# NOTE: adjust CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic models ----------
class PredictRequest(BaseModel):
    revenue: confloat(gt=0) = Field(..., description="Annual revenue (must be > 0)")
    employees: conint(ge=1) = Field(..., description="Number of employees (must be >= 1)")
    country: str = Field(..., description="Country name (must be in allowed list)")
    industry: str = Field(..., description="Industry (must be in allowed list)")

class StageProb(BaseModel):
    stage: str
    probability: float

class PredictResponse(BaseModel):
    top_stage: str
    top_probability: float
    probabilities: List[StageProb]
    next_best_action: Optional[str] = None
    raw_features: Optional[dict] = None
    warnings: Optional[List[str]] = None

class ValidationErrorResponse(BaseModel):
    valid: bool = False
    errors: List[str]
    warnings: Optional[List[str]] = None

# ---------- Artifact loading ----------
MODEL_PATH = "propensity_engine.json"
PREPROCESSOR_PATH = "preprocessor.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"
FEATURE_NAMES_PATH = "feature_names.joblib"

model: Optional[xgb.Booster] = None
preprocessor = None
label_encoder = None
feature_names = None

def load_artifacts():
    global model, preprocessor, label_encoder, feature_names
    if model is not None:
        return
    try:
        model = xgb.Booster()
        model.load_model(MODEL_PATH)
        logger.info("Loaded XGBoost model from %s", MODEL_PATH)

        preprocessor = joblib.load(PREPROCESSOR_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        logger.info("Loaded preprocessor, label encoder, and feature names.")
    except Exception as e:
        logger.exception("Failed to load artifacts: %s", e)
        raise

@app.on_event("startup")
def startup_event():
    load_artifacts()

# ---------- Feature derivation ----------
def derive_features(revenue: float, employees: int, country: str, industry: str) -> pd.DataFrame:
    log_revenue = np.log1p(revenue)
    log_num_employees = np.log1p(employees) if employees > 0 else 0.0
    revenue_per_employee = float(revenue / employees) if employees > 0 else 0.0

    # Revenue banding
    if revenue <= 20_000_000:
        rev_band = "A  0-20 Million"
    elif revenue <= 50_000_000:
        rev_band = "B  >20-50 Million"
    elif revenue <= 100_000_000:
        rev_band = "C  >50-100 Million"
    elif revenue <= 250_000_000:
        rev_band = "D  >100-250 Million"
    elif revenue <= 500_000_000:
        rev_band = "E  >250-500 Million"
    elif revenue < 1_000_000_000:
        rev_band = "F  >500-<1000 Million"
    else:
        rev_band = "G 1 Billlion or Greater"

    # Employee banding
    if employees <= 50:
        emp_band = "A 1-50"
    elif employees <= 100:
        emp_band = "B 51-100"
    elif employees <= 250:
        emp_band = "C 101-250"
    elif employees <= 500:
        emp_band = "D 251-500"
    elif employees <= 999:
        emp_band = "E 501-999"
    else:
        emp_band = "F 1000 or Greater"

    df = pd.DataFrame({
        'log_revenue': [log_revenue],
        'log_num_employees': [log_num_employees],
        'revenue_per_employee': [revenue_per_employee],
        'address1_country': [country],
        'industrycode_display': [industry],
        'qg_annualrevenue_display': [rev_band],
        'qg_numberofemployees_display': [emp_band]
    })

    return df

# ---------- Next best action ----------
def next_best_action_for_stage(stage: str) -> str:
    mapping = {
        "Target": "High Value Fit. Immediate sales outreach recommended.",
        "Client": "Matches 'Client' profile. If not currently a client, this is a high-priority miss.",
        "Free Account": "High risk of low-monetization. Recommend automated nurturing rather than direct sales.",
        "Deactivated": "Forensic match with Churned accounts. Do not prioritize.",
        "Prospect": "Good fit, but requires nurturing to move to 'Target' or 'Client' status."
    }
    return mapping.get(stage, "No specific action defined for this stage.")

# ---------- Payload validation helper ----------
def validate_payload(payload: PredictRequest) -> Tuple[List[str], List[str]]:
    """
    Returns (errors, warnings).
    - errors: blocking issues (invalid country/industry, numeric constraints)
    - warnings: non-blocking suspicious combinations
    """
    errors: List[str] = []
    warnings: List[str] = []

    if payload.country not in ALLOWED_COUNTRIES:
        errors.append(
            f"Invalid country: '{payload.country}'. Allowed countries: {sorted(ALLOWED_COUNTRIES)}"
        )
    if payload.industry not in ALLOWED_INDUSTRIES:
        errors.append(
            f"Invalid industry: '{payload.industry}'. Allowed industries: {sorted(ALLOWED_INDUSTRIES)}"
        )

    if payload.revenue <= 0:
        errors.append("Invalid revenue: must be greater than 0.")
    if payload.employees < 1:
        errors.append("Invalid employees: must be at least 1.")

    # Suspicious checks
    rev = float(payload.revenue)
    emp = int(payload.employees)
    rev_per_emp = rev / emp if emp > 0 else float("inf")

    if rev > 1_000_000_000 and emp < 5:
        warnings.append(
            "Suspicious: extremely high revenue (> 1B) with very few employees (<5). Please verify."
        )

    if rev < 1_000 and emp > 1000:
        warnings.append(
            "Suspicious: very low revenue (<1k) with many employees (>1000). Please verify."
        )

    if rev_per_emp > 10_000_000:
        warnings.append(
            f"Suspicious: revenue per employee is very high ({rev_per_emp:,.0f}). Please verify."
        )

    if rev_per_emp < 100:
        warnings.append(
            f"Suspicious: revenue per employee is very low ({rev_per_emp:.2f}). Please verify."
        )

    return errors, warnings

# ---------- Prediction endpoint (protected by Basic Auth) ----------
@app.post("/predict", responses={
    200: {"model": PredictResponse},
    400: {"model": ValidationErrorResponse},
    401: {"description": "Unauthorized"}
})
def predict(payload: PredictRequest, username: str = Depends(verify_credentials)):
    """
    Protected endpoint: requires valid Basic Auth credentials.
    'username' returned by verify_credentials can be used for audit/logging.
    """
    # Validate first:
    errors, warnings = validate_payload(payload)
    if errors:
        # Blocking errors
        raise HTTPException(
            status_code=400,
            detail={"valid": False, "errors": errors, "warnings": warnings}
        )

    # proceed with inference (same as before)
    try:
        raw_df = derive_features(payload.revenue, payload.employees, payload.country, payload.industry)
    except Exception as e:
        logger.exception("Error deriving features: %s", e)
        raise HTTPException(status_code=400, detail=f"Feature derivation failed: {e}")

    try:
        X_processed = preprocessor.transform(raw_df)
    except Exception as e:
        logger.exception("Preprocessing transform error: %s", e)
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")

    try:
        dtest = xgb.DMatrix(X_processed)
        preds = model.predict(dtest)
        if preds.ndim == 2 and preds.shape[0] == 1:
            probs = preds[0].tolist()
        elif preds.ndim == 1:
            probs = preds.tolist()
        else:
            probs = preds.reshape(-1).tolist()
    except Exception as e:
        logger.exception("Model prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    try:
        class_labels = label_encoder.classes_
        items = [{"stage": str(s), "probability": float(p)} for s, p in zip(class_labels, probs)]
        items_sorted = sorted(items, key=lambda x: x["probability"], reverse=True)
        top_stage = items_sorted[0]["stage"]
        top_probability = items_sorted[0]["probability"]
    except Exception as e:
        logger.exception("Error mapping labels: %s", e)
        raise HTTPException(status_code=500, detail=f"Label mapping failed: {e}")

    # audit log example
    logger.info("User '%s' invoked /predict for country=%s, industry=%s", username, payload.country, payload.industry)

    response = PredictResponse(
        top_stage=top_stage,
        top_probability=float(top_probability),
        probabilities=[StageProb(stage=i["stage"], probability=i["probability"]) for i in items_sorted],
        next_best_action=next_best_action_for_stage(top_stage),
        raw_features=raw_df.to_dict(orient="records")[0],
        warnings=warnings if warnings else None
    )

    return response

# ---------- Health endpoint (public) ----------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Run ----------
if __name__ == "__main__":
    uvicorn.run("account_autotagging_API:app", host="127.0.0.1", port=8000, reload=False)
