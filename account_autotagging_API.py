
from typing import List, Optional, Tuple, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, status, Body
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field, field_validator
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import secrets

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("account_autotagging_api")

# -------------------------
# Allowed lists (streamlit selectbox values)
# -------------------------
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
    "Test Account"
}

# -------------------------
# Auth configuration
# -------------------------
DEFAULT_AUTH_USERNAME = "admin"
DEFAULT_AUTH_PASSWORD = "S!tr0ngP@ssw0rd#2025"

API_AUTH_USERNAME = os.getenv("API_AUTH_USERNAME", DEFAULT_AUTH_USERNAME)
API_AUTH_PASSWORD = os.getenv("API_AUTH_PASSWORD", DEFAULT_AUTH_PASSWORD)

security = HTTPBasic()

def verify_credentials(creds: HTTPBasicCredentials = Depends(security)) -> str:
    if not (secrets.compare_digest(creds.username, API_AUTH_USERNAME) and
            secrets.compare_digest(creds.password, API_AUTH_PASSWORD)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return creds.username

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(
    title="CRM Account Auto-Tagging API (Hybrid Rules + XGBoost)",
    version="1.4.0",
    description="Hybrid deterministic-rule + XGBoost propensity engine. Supports single & batch scoring. Protected by Basic Auth."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Pydantic models
# -------------------------
class AccountPayload(BaseModel):
    revenue: float = Field(..., gt=0, description="Annual revenue (must be > 0)")
    employees: int = Field(..., ge=1, description="Number of employees (must be >= 1)")
    country: str = Field(..., description="Country (must be in allowed list)")
    industry: str = Field(..., description="Industry (must be in allowed list)")

    @field_validator("country")
    def country_must_be_allowed(cls, v):
        if v not in ALLOWED_COUNTRIES:
            raise ValueError(f"Invalid country '{v}'. Allowed: {sorted(ALLOWED_COUNTRIES)}")
        return v

    @field_validator("industry")
    def industry_must_be_allowed(cls, v):
        if v not in ALLOWED_INDUSTRIES:
            raise ValueError(f"Invalid industry '{v}'. Allowed: {sorted(ALLOWED_INDUSTRIES)}")
        return v

class StageProb(BaseModel):
    stage: str
    probability: float

class PredictResult(BaseModel):
    top_stage: str
    top_probability: float
    probabilities: List[StageProb]
    next_best_action: Optional[str] = None
    raw_features: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None
    logic_source: Optional[str] = None  # "Rule: ..." or "XGBoost Model"

class BatchItemResult(BaseModel):
    input: AccountPayload
    result: Optional[PredictResult] = None
    error: Optional[str] = None

# -------------------------
# Artifacts paths (same folder)
# -------------------------
MODEL_PATH = "propensity_engine.json"
PREPROCESSOR_PATH = "preprocessor.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"
FEATURE_NAMES_PATH = "feature_names.joblib"

# Module-level caches
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
        # feature_names may be optional, wrap in try
        try:
            feature_names = joblib.load(FEATURE_NAMES_PATH)
        except Exception:
            feature_names = None
        logger.info("Loaded preprocessing artifacts.")
    except Exception as e:
        logger.exception("Failed to load artifacts: %s", e)
        raise

@app.on_event("startup")
def startup_event():
    load_artifacts()

# -------------------------
# Feature derivation (same logic as streamlit)
# -------------------------
def derive_features_df(revenue: float, employees: int, country: str, industry: str) -> pd.DataFrame:
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

# -------------------------
# Next best action mapping
# -------------------------
def next_best_action_for_stage(stage: str) -> str:
    mapping = {
        "Target": "High Value Fit. Immediate sales outreach recommended.",
        "Client": "Matches 'Client' profile. If not currently a client, this is a high-priority miss.",
        "Free Account": "High risk of low-monetization. Recommend automated nurturing rather than direct sales.",
        "Deactivated": "Forensic match with Churned accounts. Do not prioritize.",
        "Prospect": "Good fit, but requires nurturing to move to 'Target' or 'Client' status."
    }
    return mapping.get(stage, "No specific action defined for this stage.")

# -------------------------
# Validation helper (non-Pydantic checks)
# returns (errors, warnings)
# -------------------------
def validate_business(payload: AccountPayload) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    # Numeric checks (Pydantic already does most) - double-check
    if payload.revenue <= 0:
        errors.append("Invalid revenue: must be > 0.")
    if payload.employees < 1:
        errors.append("Invalid employees: must be >= 1.")

    # Allowed lists checked by validators in AccountPayload; keep redundant safety
    if payload.country not in ALLOWED_COUNTRIES:
        errors.append(f"Invalid country: '{payload.country}'.")
    if payload.industry not in ALLOWED_INDUSTRIES:
        errors.append(f"Invalid industry: '{payload.industry}'.")

    # Suspicious checks (non-blocking)
    rev = float(payload.revenue)
    emp = int(payload.employees)
    rev_per_emp = rev / emp if emp > 0 else float("inf")

    if rev > 1_000_000_000 and emp < 5:
        warnings.append("Suspicious: extremely high revenue (>1B) with very few employees (<5).")
    if rev < 1_000 and emp > 1000:
        warnings.append("Suspicious: very low revenue (<1k) with many employees (>1000).")
    if rev_per_emp > 10_000_000:
        warnings.append(f"Suspicious: revenue per employee is very high ({rev_per_emp:,.0f}).")
    if rev_per_emp < 100:
        warnings.append(f"Suspicious: revenue per employee is very low ({rev_per_emp:.2f}).")

    return errors, warnings

# -------------------------
# Hybrid rules: returns (rule_triggered: bool, logic_source: str, override_class: Optional[str])
# Adapted from your streamlit rules
# -------------------------
def apply_hard_rules(revenue: float, employees: int, industry: str) -> Tuple[bool, str, Optional[str]]:
    # Default: no rule triggered
    # RULE 1: Test Purge (if industry contains "Test")
    if "Test" in industry:
        return True, "Rule: Test Artifact Purge", "Deactivated"

    # RULE 2: Zombie company (high employees, very low revenue)
    if employees > 50 and revenue < 10_000:
        return True, "Rule: Zombie Company (High Emp / Low Rev)", "Deactivated"

    # RULE 3: Enterprise Target (very high revenue)
    if revenue > 100_000_000 and employees > 1:
        return True, "Rule: Enterprise Whitelist (Rev > $100M)", "Target"

    # RULE 4: Micro revenue (too small)
    if revenue < 1_000:
        return True, "Rule: Min. Revenue Threshold (Rev < $1k)", "Deactivated"

    return False, "XGBoost Model", None

# -------------------------
# Core single-inference routine
# -------------------------
def single_predict_logic(payload: AccountPayload) -> PredictResult:
    # Validate business rules + warnings
    errors, warnings = validate_business(payload)
    if errors:
        raise HTTPException(status_code=400, detail={"valid": False, "errors": errors, "warnings": warnings})

    # Determine if any hard rule triggers
    rule_triggered, logic_source, override_class = apply_hard_rules(payload.revenue, payload.employees, payload.industry)

    # Prepare class labels
    class_labels = list(label_encoder.classes_)

    if rule_triggered:
        # Build deterministic probability vector
        probs = np.zeros(len(class_labels), dtype=float)
        if override_class not in class_labels:
            # If override class not found, it's an application error — return 500
            raise HTTPException(status_code=500, detail=f"Override class '{override_class}' not present in label encoder classes.")
        idx = class_labels.index(override_class)
        probs[idx] = 1.0
    else:
        # Run ML model
        raw_df = derive_features_df(payload.revenue, payload.employees, payload.country, payload.industry)
        try:
            X_processed = preprocessor.transform(raw_df)
        except Exception as e:
            logger.exception("Preprocessor transform failed: %s", e)
            raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")

        try:
            dtest = xgb.DMatrix(X_processed)
            preds = model.predict(dtest)
            # preds shape handling
            if preds.ndim == 2 and preds.shape[0] == 1:
                probs = preds[0]
            elif preds.ndim == 1:
                probs = preds
            else:
                probs = preds.reshape(-1)
        except Exception as e:
            logger.exception("Model prediction failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # Map probabilities to stages
    items = [{"stage": s, "probability": float(p)} for s, p in zip(class_labels, probs)]
    items_sorted = sorted(items, key=lambda x: x["probability"], reverse=True)
    top_stage = items_sorted[0]["stage"]
    top_probability = items_sorted[0]["probability"]

    # raw_features for debugging
    raw_features_df = derive_features_df(payload.revenue, payload.employees, payload.country, payload.industry)
    raw_features = raw_features_df.to_dict(orient="records")[0]

    result = PredictResult(
        top_stage=top_stage,
        top_probability=float(top_probability),
        probabilities=[StageProb(stage=i["stage"], probability=i["probability"]) for i in items_sorted],
        next_best_action=next_best_action_for_stage(top_stage),
        raw_features=raw_features,
        warnings=warnings if warnings else None,
        logic_source=logic_source
    )

    return result

# -------------------------
# Endpoints
# -------------------------
@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResult, tags=["predict"])
def predict(payload: AccountPayload, username: str = Depends(verify_credentials)):
    """
    Single account prediction (protected). Returns PredictResult.
    """
    logger.info("User '%s' called /predict for country=%s industry=%s", username, payload.country, payload.industry)
    return single_predict_logic(payload)

@app.post("/predict/batch", response_model=List[BatchItemResult], tags=["predict"])
def predict_batch(payloads: List[AccountPayload] = Body(..., example=[
    {"revenue": 1500000, "employees": 50, "country": "United Kingdom", "industry": "Manufacturing"},
    {"revenue": 1000, "employees": 2, "country": "France", "industry": "Test Account"}
]), username: str = Depends(verify_credentials)):
    """
    Batch scoring endpoint. Returns per-item result or error.
    """
    logger.info("User '%s' called /predict/batch with %d items", username, len(payloads))
    responses: List[BatchItemResult] = []
    for p in payloads:
        try:
            res = single_predict_logic(p)
            responses.append(BatchItemResult(input=p, result=res, error=None))
        except HTTPException as he:
            # Preserve HTTPException detail for the single item
            err_detail = he.detail if isinstance(he.detail, (str, dict, list)) else str(he.detail)
            responses.append(BatchItemResult(input=p, result=None, error=str(err_detail)))
        except Exception as e:
            logger.exception("Unhandled error for batch item: %s", e)
            responses.append(BatchItemResult(input=p, result=None, error=str(e)))
    return responses

# -------------------------
# Run (dev)
# -------------------------
if __name__ == "__main__":
    uvicorn.run("account_autotagging_API:app", host="127.0.0.1", port=8000, reload=False)
