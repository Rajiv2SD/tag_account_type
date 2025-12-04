import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import csv
import os
from datetime import datetime

# ==========================================
# 0. Page Config (MUST BE FIRST)
# ==========================================
st.set_page_config(page_title="CRM Propensity Engine", layout="centered")

# ==========================================
# 1. Load Artifacts (Cached for performance)
# ==========================================
@st.cache_resource
def load_artifacts():
    # Load Model (Using your existing JSON file)
    model = xgb.Booster()
    model.load_model("propensity_engine.json")
    
    # Load Preprocessors (Using your existing Joblib files)
    preprocess = joblib.load("preprocessor.joblib")
    le = joblib.load("label_encoder.joblib")
    feature_names = joblib.load("feature_names.joblib")
    
    return model, preprocess, le, feature_names

# Load them AFTER setting the page config
try:
    model, preprocessor, le, feature_names = load_artifacts()
except Exception as e:
    st.error(f"Error loading files: {e}. Please make sure 'propensity_engine.json' and .joblib files are in the folder.")
    st.stop()

# ==========================================
# 2. Helper Functions
# ==========================================
def derive_features(revenue, employees, country, industry):
    """
    Transforms raw user inputs into the EXACT dataframe structure 
    the model was trained on.
    """
    # A. Mathematical Transformations
    log_revenue = np.log1p(revenue)
    log_num_employees = np.log1p(employees)
    
    if employees > 0:
        revenue_per_employee = revenue / employees
    else:
        revenue_per_employee = 0

    # B. Banding Logic
    if revenue <= 20000000: rev_band = "A  0-20 Million"
    elif revenue <= 50000000: rev_band = "B  >20-50 Million"
    elif revenue <= 100000000: rev_band = "C  >50-100 Million"
    elif revenue <= 250000000: rev_band = "D  >100-250 Million"
    elif revenue <= 500000000: rev_band = "E  >250-500 Million"
    elif revenue < 1000000000: rev_band = "F  >500-<1000 Million"
    else: rev_band = "G 1 Billlion or Greater"

    if employees <= 50: emp_band = "A 1-50"
    elif employees <= 100: emp_band = "B 51-100"
    elif employees <= 250: emp_band = "C 101-250"
    elif employees <= 500: emp_band = "D 251-500"
    elif employees <= 999: emp_band = "E 501-999"
    else: emp_band = "F 1000 or Greater"

    data = pd.DataFrame({
        'log_revenue': [log_revenue],
        'log_num_employees': [log_num_employees],
        'revenue_per_employee': [revenue_per_employee],
        'address1_country': [country],
        'industrycode_display': [industry],
        'qg_annualrevenue_display': [rev_band],
        'qg_numberofemployees_display': [emp_band]
    })
    
    return data

def log_feedback(company_name, revenue, employees, country, industry, predicted_class, actual_class, feedback_type):
    """
    Saves user feedback to a CSV file for future retraining.
    """
    file_path = "feedback_log.csv"
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header if file is new
        if not file_exists:
            writer.writerow(["timestamp", "company", "revenue", "employees", "country", "industry", "predicted", "actual", "type"])
        
        # Write data
        writer.writerow([
            datetime.now(), company_name, revenue, employees, country, industry, predicted_class, actual_class, feedback_type
        ])
    return True

def get_detailed_recommendation(stage, probability, logic_source):
    """
    Returns nuanced business advice based on stage, confidence, and source.
    """
    is_rule = "Rule" in logic_source
    
    # SPECIAL CHECK: If the Name Rule was triggered
    if "Name Check" in logic_source:
        return "⚠️ **Please input correct information.** The company name indicates test data."

    if stage == "Target":
        if is_rule:
            return "🔥 **Priority: Critical.** This account matched a strategic 'Must-Win' criteria. Bypass standard qualification and assign to a Senior AE immediately."
        elif probability > 0.8:
            return "🚀 **Priority: High.** Strong algorithmic match with our Ideal Customer Profile. Recommended action: Direct outreach via LinkedIn + Personalized Email sequence."
        else:
            return "👀 **Priority: Medium.** Good fit, but metrics are borderline. Recommended action: Verify budget availability before full sales engagement."

    elif stage == "Client":
        return "🤝 **Status: Customer.** System indicates this profile matches existing clients. **Action:** Check CRM for active contracts. If not active, this is a high-probability win-back opportunity."

    elif stage == "Prospect":
        if probability > 0.6:
            return "🌱 **Status: Nurture.** Good firmographics but missing 'Urgency' signals. **Action:** Add to 'Mid-Funnel' marketing campaign and monitor for intent signals."
        else:
            return "📉 **Status: Long-Term.** Low probability of immediate conversion. **Action:** Automate weekly newsletter, do not invest sales time yet."

    elif stage == "Free Account":
        return "🔍 **Status: Unclassified.** The model requires more information to make a confident prediction. **Action:** Assign to SDR for data enrichment and manual qualification."

    elif stage == "Deactivated":
        if is_rule:
             return "⛔ **Do Not Contact.** This account triggered a hard exclusion rule (e.g., Test Account or Bad Data). Archiving recommended."
        else:
             return "🚫 **Risk: Churn.** Profile strongly resembles past churned accounts. **Action:** Do not prioritize for acquisition."
    
    return "No specific recommendation available."

# ==========================================
# 3. Streamlit UI Layout
# ==========================================

# --- Header with Logos ---
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.title("CRM Autotagging Propensity Engine")
    st.markdown("Probabilistic AI Modeling with Human-in-the-Loop Validation")

with header_col2:
    # Ensure you have 'company_logos.png' in the folder
    if os.path.exists("company_logos.png"):
        st.image("company_logos.png", width=180)
    else:
        st.caption("")

st.divider()

# --- Input Form ---
with st.container():
    st.subheader("Account Details")
    
    # 1. Company Name
    company_name = st.text_input("Company Name", placeholder="e.g. Acme Corp")

    col1, col2 = st.columns(2)
    
    with col1:
        revenue = st.number_input("Annual Revenue ($)", min_value=0, value=1_000_000, step=10000)
        employees = st.number_input("Number of Employees", min_value=1, value=50)
    
    with col2:
        # Flexible Input: Country
        country_list = ["United Kingdom", "France", "Italy", "Spain", "United Arab Emirates", "Saudi Arabia", "Nigeria", "Egypt", "South Africa", "United States", "Other (Enter Manually)"]
        country_select = st.selectbox("Country", country_list)
        
        if country_select == "Other (Enter Manually)":
            country_final = st.text_input("Enter Country Name")
        else:
            country_final = country_select

        # Flexible Input: Industry
        industry_list = ["Manufacturing", "Retail & Wholesale", "Professional Services", "Built Environment & Construction", "Agri Food", "IT, Communication & Media Services", "Energy (Electricity, Oil & Gas)", "Healthcare", "Logistics, Transport & Distribution", "Hospitality & Leisure", "Test Account", "Other (Enter Manually)"]
        industry_select = st.selectbox("Industry", industry_list)
        
        if industry_select == "Other (Enter Manually)":
            industry_final = st.text_input("Enter Industry Name")
        else:
            industry_final = industry_select

    # Center the button
    st.write("")
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        submit = st.button("Generate Prediction", type="primary", use_container_width=True)

# ==========================================
# 4. Hybrid Inference Logic
# ==========================================
if submit:
    if not company_name:
        st.warning("Please enter a Company Name.")
        # Ensure name exists for display, even if empty
        company_name_display = "this Company"
    else:
        company_name_display = company_name
        
    # Prepare Class Labels early
    class_labels = le.classes_
    
    # --- LAYER 1: HARD RULES (Deterministic) ---
    rule_triggered = False
    logic_source = "Propensity Model" # Default AI Source Name
    final_probs = None
    override_class = None

    # RULE 1: Invalid Company Name
    if company_name and "test" in company_name.lower():
        rule_triggered = True
        logic_source = "Rule: Invalid Input (Name Check)"
        override_class = "Deactivated"

    # RULE 2: The "Test" Purge (Industry)
    elif "Test" in industry_final:
        rule_triggered = True
        logic_source = "Rule: Test Artifact Purge"
        override_class = "Deactivated"

    # RULE 3: The "Zombie Company" Filter
    elif employees > 50 and revenue < 10000:
        rule_triggered = True
        logic_source = "Rule: Zombie Company (High Emp / Low Rev)"
        override_class = "Deactivated"

    # RULE 4: Enterprise Target
    elif revenue > 100_000_000 and employees > 1:
        rule_triggered = True
        logic_source = "Rule: Enterprise Whitelist (Rev > $100M)"
        override_class = "Target"
        
    # RULE 5: Micro-Revenue Filter
    elif revenue < 1000:
        rule_triggered = True
        logic_source = "Rule: Min. Revenue Threshold (Rev < $1k)"
        override_class = "Deactivated"

    # --- EXECUTION ---
    if rule_triggered:
        probs = np.zeros(len(class_labels))
        try:
            target_idx = np.where(class_labels == override_class)[0][0]
            probs[target_idx] = 1.0
            final_probs = probs
        except IndexError:
            st.error(f"Error: Rule output '{override_class}' not found in model classes.")
            st.stop()
            
    else:
        # --- LAYER 2: AI MODEL ---
        raw_df = derive_features(revenue, employees, country_final, industry_final)
        
        try:
            X_processed = preprocessor.transform(raw_df)
        except Exception as e:
            st.error(f"Preprocessing Error: {e}")
            st.stop()

        dtest = xgb.DMatrix(X_processed)
        final_probs = model.predict(dtest)[0]

    # ==========================================
    # 5. Display Results
    # ==========================================
    # Map Probabilities
    result_df = pd.DataFrame({"Stage": class_labels, "Probability": final_probs})
    result_df = result_df.sort_values(by="Probability", ascending=False)
    
    top_class = result_df.iloc[0]["Stage"]
    top_prob = result_df.iloc[0]["Probability"]

    st.divider()
    
    # --- UPDATED RESULT SECTION ---
    st.subheader(f"Prediction for {company_name_display}")
    
    # Use columns to separate the Prediction from the Confidence
    res_col1, res_col2 = st.columns([3, 1])
    
    with res_col1:
        # Dynamic Color Logic
        color = "green" if top_class in ["Target", "Client"] else "red" if top_class == "Deactivated" else "orange"
        
        # 1. The Prediction (Big)
        st.markdown(f"### :{color}[{top_class}]")
        
        # 2. The Source (Small Text Below)
        # Using a distinct visual style for the source
        if "Rule" in logic_source:
             st.caption(f"🛡️ **Source:** {logic_source}")
        else:
             st.caption(f"🤖 **Source:** {logic_source}")

    with res_col2:
        # 3. Confidence
        st.metric(label="Confidence", value=f"{top_prob:.1%}")

    # B. The Verdict (Nuanced Recommendation)
    st.markdown("### AI Recommendation")
    recommendation_text = get_detailed_recommendation(top_class, top_prob, logic_source)
    st.markdown(f"> {recommendation_text}")

    # C. Visualization
    st.write("")
    st.write("### Probability Distribution")
    st.bar_chart(result_df.set_index("Stage"))
    
    # ==========================================
    # 6. Feedback Loop (Active Learning)
    # ==========================================
    st.divider()
    st.markdown("#### Model Feedback")
    st.caption("Help improve the AI by validating this prediction.")

    fb_col1, fb_col2 = st.columns(2)
    
    # Option 1: Correct
    with fb_col1:
        if st.button("✅ Accurate Prediction"):
            log_feedback(company_name, revenue, employees, country_final, industry_final, top_class, top_class, "Positive")
            st.toast("Feedback Saved! The model will learn from this.", icon="💾")

    # Option 2: Incorrect
    with fb_col2:
        with st.expander("❌ Report Issue / Correct"):
            st.write("What is the correct stage?")
            actual_stage = st.selectbox("Select Actual Stage", class_labels, key="feedback_select")
            
            if st.button("Submit Correction"):
                log_feedback(company_name, revenue, employees, country_final, industry_final, top_class, actual_stage, "Negative")
                st.toast(f"Correction Saved! Model labeled as '{actual_stage}' for retraining.", icon="💾")