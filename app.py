# ===============================================================
# 🏥 AI-Ready Health Data Platform — Streamlit App (All-in-One)
# ===============================================================

import os, json, joblib, hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Optional YAML pretty print (fallback to JSON if missing)
try:
    import yaml
    HAS_YAML = True
except Exception:
    HAS_YAML = False

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

import shap  # TreeExplainer for RandomForest

# FIX 1: Import imblearn for SMOTE
# NOTE: User must ensure 'imbalanced-learn' is installed
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None 
    st.error("Missing dependency: 'imblearn'. Please run: pip install imbalanced-learn")
    st.stop()


# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="AI-Ready Health Data Platform", page_icon="🏥", layout="wide")
st.title("🏥 AI-Ready Health Data Platform")
st.caption("Interoperability • Secure Data Access • Predictive Analytics • Explainability • Consent")

DATA_DIR = "data"
MODELS_DIR = "models"
GOV_DIR = "governance"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GOV_DIR, exist_ok=True)

READM_MODEL_PATH = os.path.join(MODELS_DIR, "readmission_risk_pipeline.joblib")
COND_MODEL_PATH = os.path.join(MODELS_DIR, "patient_condition.pkl")
AUDIT_PATH = os.path.join(GOV_DIR, "audit_log.csv")
CONSENT_PATH = os.path.join(DATA_DIR, "consent.csv")

# -------------------------------
# RBAC demo roles (Defined early for use in load_all)
# -------------------------------
ROLES = {
    "Admin": {"can_read":["*"], "can_score": True},
    "Analyst": {"can_read":["Patient","Encounter","Outcome", "Consent"], "can_score": True}, 
    "Clinician": {"can_read":["Patient","Encounter","Outcome"], "can_score": True},
    "Researcher": {"can_read":["Encounter","Outcome"], "can_score": False}
}
def can_read(role, resource):
    rules = ROLES.get(role, {})
    allowed = rules.get("can_read",[])
    return "*" in allowed or resource in allowed

def can_score(role):
    return ROLES.get(role,{}).get("can_score",False)


# -------------------------------
# Utilities
# -------------------------------
def write_audit(user_id, action, resource, record_id, purpose, result="success"):
    row = {
        "log_id": f"L{int(datetime.utcnow().timestamp()*1000)}",
        "user_id": user_id,
        "action": action,
        "resource": resource,
        "record_id": record_id,
        "timestamp": datetime.utcnow().isoformat(sep=" "),
        "purpose_of_use": purpose,
        "result": result
    }
    df = pd.DataFrame([row])
    header = not os.path.exists(AUDIT_PATH)
    df.to_csv(AUDIT_PATH, mode="a", header=header, index=False)

def hash_str(s):
    import hashlib as _h
    return _h.sha256(s.encode("utf-8")).hexdigest()[:32]

@st.cache_data(show_spinner=False)
def generate_synthetic_core():
    """Create minimal synthetic datasets if not found, and persist to /data."""
    np.random.seed(42)

    # Patients
    n_pat = 2000
    ages = np.clip(np.random.normal(56, 18, n_pat).astype(int), 0, 95)
    pids = [f"P{str(i).zfill(6)}" for i in range(1, n_pat + 1)]
    patients = pd.DataFrame({
        "patient_id": pids,
        "nhs_number_hash": [hash_str(f"NHS-{pid}") for pid in pids],
        "birth_date": [(datetime(2025, 1, 1) - timedelta(days=int(a * 365.25))).date().isoformat() for a in ages],
        "sex": np.random.choice(["Male", "Female"], size=n_pat),
        "ethnicity": np.random.choice(["White", "Black", "Asian", "Mixed", "Other"], size=n_pat, p=[0.75, 0.07, 0.12, 0.04, 0.02]),
        "postcode_lsoa": [f"E010{np.random.randint(1000000, 9999999)}" for _ in range(n_pat)],
        "imd_quintile": np.random.choice([1,2,3,4,5], size=n_pat, p=[0.25,0.25,0.2,0.2,0.1]),
        "age_years": ages,
        "bmi": np.round(np.random.normal(28, 6, n_pat), 1),
        "smoker": np.random.choice(["Yes", "No"], size=n_pat, p=[0.28, 0.72]),
        "exercise_freq": np.random.choice(["Low", "Moderate", "High"], size=n_pat, p=[0.4, 0.4, 0.2])
    })

    # Add vitals/biomarkers used by condition model
    patients["systolic_bp"] = np.random.randint(100, 180, n_pat)
    patients["diastolic_bp"] = np.random.randint(60, 110, n_pat)
    patients["glucose_mmol"] = np.round(np.random.uniform(3.5, 12.0, n_pat), 1)
    patients["cholesterol_mmol"] = np.round(np.random.uniform(3.0, 8.5, n_pat), 1)

    def assign_condition(row):
        if row["glucose_mmol"] > 8.5:
            return "Diabetes"
        elif row["systolic_bp"] > 140 or row["diastolic_bp"] > 90:
            return "Hypertension"
        elif row["cholesterol_mmol"] > 6.8 and row["bmi"] > 30:
            return "Heart Disease"
        elif row["bmi"] > 32:
            return "Obesity"
        else:
            return "Healthy"

    patients["condition"] = patients.apply(assign_condition, axis=1)

    # Encounters
    n_enc = 10000
    eids = [f"E{str(i).zfill(7)}" for i in range(1, n_enc + 1)]
    types = np.random.choice(["inpatient", "outpatient", "emergency", "virtual"], size=n_enc, p=[0.28, 0.45, 0.22, 0.05])
    start = np.array([datetime(2020, 1, 1) + timedelta(days=int(np.random.uniform(0, 2100))) for _ in range(n_enc)])
    los = np.zeros(n_enc, dtype=int)
    los[types == "inpatient"] = np.maximum(1, np.random.lognormal(1.3, 0.6, (types == "inpatient").sum()).astype(int))
    los[types == "emergency"] = np.random.choice([0, 1, 2], size=(types == "emergency").sum(), p=[0.6, 0.3, 0.1])
    end = start + np.array([timedelta(days=int(d), hours=np.random.randint(0, 12)) for d in los])

    encounters = pd.DataFrame({
        "encounter_id": eids,
        "patient_id": np.random.choice(pids, size=n_enc),
        "start_dt": [d.isoformat(sep=" ") for d in start],
        "end_dt": [d.isoformat(sep=" ") for d in end],
        "type": types,
        "los_days": los
    })

    # Outcomes (readmission & mortality; los copied for convenience)
    enc_df = encounters.merge(patients[["patient_id", "age_years"]], on="patient_id", how="left")
    sev = (enc_df["los_days"] > 3).astype(int) + (enc_df["type"] == "inpatient").astype(int)
    logits = -1.7 + 0.15 * sev + 0.01 * (enc_df["age_years"] - 55) + 0.12 * (enc_df["los_days"] > 5)
    proba = 1 / (1 + np.exp(-logits))
    outcomes = pd.DataFrame({
        "encounter_id": enc_df["encounter_id"],
        "readmission_30d": (np.random.rand(len(proba)) < proba * 0.9).astype(int),
        "mortality_90d": (np.random.rand(len(proba)) < proba * 0.2).astype(int),
        "los_days": enc_df["los_days"].fillna(0).astype(int)
    })

    # Consent
    if not os.path.exists(CONSENT_PATH):
        consent = pd.DataFrame({
            "patient_id": pids,
            "allow_research": np.random.choice([0,1], size=n_pat, p=[0.06, 0.94]),
            "allow_risk_scoring": np.random.choice([0,1], size=n_pat, p=[0.2, 0.8]),
            "last_updated": [(datetime(2022, 1, 1) + timedelta(days=int(np.random.uniform(0, 900)))).date().isoformat() for _ in range(n_pat)]
        })
        consent.to_csv(CONSENT_PATH, index=False)
    else:
        consent = pd.read_csv(CONSENT_PATH)

    # Persist
    patients.to_csv(os.path.join(DATA_DIR, "patients.csv"), index=False)
    encounters.to_csv(os.path.join(DATA_DIR, "encounters.csv"), index=False)
    outcomes.to_csv(os.path.join(DATA_DIR, "outcomes.csv"), index=False)
    if not os.path.exists(AUDIT_PATH):
        pd.DataFrame(columns=["log_id","user_id","action","resource","record_id","timestamp","purpose_of_use","result"]).to_csv(AUDIT_PATH, index=False)
    if not os.path.exists(os.path.join(DATA_DIR, "audit_log.csv")): 
        pd.DataFrame(columns=["log_id","user_id","action","resource","record_id","timestamp","purpose_of_use","result"]).to_csv(os.path.join(DATA_DIR, "audit_log.csv"), index=False)

    return patients, encounters, outcomes, consent

@st.cache_resource(show_spinner=False)
def ensure_condition_model(patients_df: pd.DataFrame):
    """Load patient_condition.pkl or train a quick RandomForest pipeline and save."""
    if os.path.exists(COND_MODEL_PATH):
        return joblib.load(COND_MODEL_PATH)

    feats_num = ["age_years","bmi","imd_quintile","systolic_bp","diastolic_bp","glucose_mmol","cholesterol_mmol"]
    feats_cat = ["sex","ethnicity","smoker","exercise_freq"]
    X = patients_df[feats_num + feats_cat]
    y = patients_df["condition"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    pre = ColumnTransformer([
        ("num", StandardScaler(), feats_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), feats_cat)
    ])
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight="balanced")
    pipe = Pipeline([("preprocessor", pre), ("model", rf)])
    pipe.fit(Xtr, ytr)
    joblib.dump(pipe, COND_MODEL_PATH)
    return pipe

@st.cache_resource(show_spinner=False)
def ensure_readmission_model(patients, encounters, outcomes):
    """Train (or load) a simple readmission risk model safely."""
    if os.path.exists(READM_MODEL_PATH):
        return joblib.load(READM_MODEL_PATH)

    # --- merge data ---
    df = encounters.merge(patients[["patient_id", "age_years", "sex", "ethnicity"]], on="patient_id", how="left")
    df = df.merge(outcomes[["encounter_id", "readmission_30d", "los_days"]], on="encounter_id", how="left")

    # --- handle missing or missing columns ---
    if "los_days" not in df.columns:
        st.warning("⚠️ 'los_days' column missing — creating synthetic length of stay.")
        df["los_days"] = np.random.randint(1, 10, size=len(df))
    else:
        mask = df["los_days"].isna()
        if mask.any():
            df.loc[mask, "los_days"] = np.random.randint(1, 10, size=mask.sum())
    
    df["los_days"] = df["los_days"].fillna(df["los_days"].median())
    df["is_inpatient"] = (df["type"] == "inpatient").astype(int)
    df["is_emergency"] = (df["type"] == "emergency").astype(int)
    df["log_los"] = np.log1p(df["los_days"])

    # --- define features ---
    features = [
        "age_years", "sex", "ethnicity",
        "is_inpatient", "is_emergency", "log_los"
    ]
    target = "readmission_30d"

    df = df.dropna(subset=[target, "age_years"])
    X, y = df[features], df[target]

    # --- preprocess ---
    num_features = ["age_years", "log_los"]
    cat_features = ["sex", "ethnicity", "is_inpatient", "is_emergency"]

    num_transformer = Pipeline([("scaler", StandardScaler())])
    cat_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features)
        ],
        remainder='passthrough'
    )

    # --- model pipeline ---
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=200, class_weight="balanced", random_state=42))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    st.success(f"✅ Readmission model trained successfully (AUC = {auc:.3f})")

    # --- save model for reuse ---
    joblib.dump(model, READM_MODEL_PATH)

    return model

@st.cache_data(show_spinner=False)
def load_all():
    # Load users first as it's needed for RBAC roles immediately
    if os.path.exists(os.path.join(DATA_DIR, "users.csv")):
         users_df = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    else:
         # Create a basic users file if not found
         users_df = pd.DataFrame({"user_id": [f"U{str(i).zfill(4)}" for i in range(1, 101)], "role": np.random.choice(list(ROLES.keys()), size=100)})
         users_df.to_csv(os.path.join(DATA_DIR, "users.csv"), index=False)

    # Load/Generate core data
    patients, encounters, outcomes, consent = generate_synthetic_core()

    # Load audit log (should be in GOV_DIR or DATA_DIR)
    if os.path.exists(AUDIT_PATH):
        audit_log = pd.read_csv(AUDIT_PATH)
    else:
        audit_log = pd.DataFrame(columns=["log_id","user_id","action","resource","record_id","timestamp","purpose_of_use","result"])
    
    return patients, encounters, outcomes, consent, users_df, audit_log


# -------------------------------
# Call functions AFTER they are defined (re-ordered execution block)
# -------------------------------
patients, encounters, outcomes, consent, users, audit_log = load_all()
cond_model = ensure_condition_model(patients)
readm_model = ensure_readmission_model(patients, encounters, outcomes)


# -------------------------------
# Sidebar: Role & Purpose
# -------------------------------
st.sidebar.header("🔐 Access Identity")
user_ids_for_roles = {r: users[users['role'] == r]['user_id'].iloc[0] for r in ROLES.keys() if not users[users['role'] == r].empty}
default_role = 'Analyst'
role = st.sidebar.selectbox("Role", list(ROLES.keys()), index=list(ROLES.keys()).index(default_role))
user_id = st.sidebar.text_input("User ID", value=user_ids_for_roles.get(role, "U0001"))
purpose = st.sidebar.selectbox("Purpose of use", ["care","research","population_health","governance"])

# -------------------------------
# Navigation
# -------------------------------
page = st.sidebar.radio("📑 Pages", [
    "Historical Dashboard",
    "Interoperability (FHIR-like)",
    "Secure Access & Audit Trail",
    "Predictive Analytics",
    "Governance: Explainability & Consent"
])

# ===============================================================
# 1) Historical Dashboard (UPDATED SECTION)
# ===============================================================
if page == "Historical Dashboard":
    st.subheader("📈 Historical Dashboard")

    enc = encounters.copy()
    enc["start_dt"] = pd.to_datetime(enc["start_dt"])
    enc["month"] = enc["start_dt"].dt.to_period("M").astype(str)
    
    # Calculate new metrics
    avg_los = outcomes['los_days'].mean()
    avg_age = patients['age_years'].mean()
    
    # --- Metrics ---
    col1, c2, c3, c4, c5 = st.columns(5)
    col1.metric("Patients", patients["patient_id"].nunique())
    c2.metric("Encounters", enc["encounter_id"].nunique())
    c3.metric("Readmission Rate", f"{outcomes['readmission_30d'].mean()*100:.1f}%")
    c4.metric("Avg. LOS (Days)", f"{avg_los:.1f}")
    c5.metric("Avg. Patient Age", f"{avg_age:.1f}")

    # --- New Charts: Clinical Analysis ---
    st.markdown("### Clinical Analysis")
    cols_los = st.columns(2)
    
    # Chart 4: LOS Distribution by Encounter Type
    with cols_los[0]:
        los_data = enc.merge(outcomes[['encounter_id', 'los_days']], on='encounter_id', how='left')
        st.plotly_chart(px.box(los_data, x="type", y="los_days_y", color="type", 
                            title="Length of Stay Distribution by Encounter Type", labels={"los_days_y": "Length of Stay (Days)"}), use_container_width=True)

    # Chart 5: Condition Prevalence by Ethnicity
    with cols_los[1]:
        ethnicity_cond = patients.groupby(["ethnicity", "condition"]).size().reset_index(name='count')
        total_by_eth = patients.groupby("ethnicity").size().reset_index(name='total')
        ethnicity_cond = ethnicity_cond.merge(total_by_eth, on="ethnicity")
        ethnicity_cond['prevalence'] = ethnicity_cond['count'] / ethnicity_cond['total'] * 100
        
        st.plotly_chart(px.bar(ethnicity_cond.sort_values('prevalence', ascending=False), 
                            x="ethnicity", y="prevalence", color="condition",
                            title="Condition Prevalence by Ethnicity (%)"), use_container_width=True)


    # --- Original Charts: Encounter and Outcome Trends ---
    st.markdown("### Encounter and Outcome Trends")
    
    # Chart 1: Monthly Encounters by Type
    vol = enc.groupby(["month","type"])["encounter_id"].count().reset_index(name="count")
    st.plotly_chart(px.line(vol, x="month", y="count", color="type", title="Monthly Encounters by Type"), use_container_width=True)

    # Chart 2: Monthly Outcome Rates Trend
    out_join = encounters[["encounter_id","start_dt"]].merge(outcomes, on="encounter_id")
    out_join["start_dt"] = pd.to_datetime(out_join["start_dt"])
    out_join["month"] = out_join["start_dt"].dt.to_period("M").astype(str)
    trend = out_join.groupby("month")[["readmission_30d","mortality_90d"]].mean().mul(100).reset_index()
    trend = trend.melt(id_vars=["month"], var_name="metric", value_name="rate_pct")
    st.plotly_chart(px.bar(trend, x="month", y="rate_pct", color="metric", barmode="group",
                           title="Monthly Outcome Rates (%)"), use_container_width=True)

    # Chart 3: Conditions by Age
    pats = patients.copy()
    pats["age_band"] = pd.cut(pats["age_years"], bins=[0,30,45,60,75,100], labels=["≤30","31–45","46–60","61–75","76+"])
    st.plotly_chart(px.histogram(pats, x="age_band", color="condition", barmode="group",
                                 title="Conditions by Age Band"), use_container_width=True)

# ===============================================================
# 2) Interoperability (FHIR-like schema)
# ===============================================================
elif page == "Interoperability (FHIR-like)":
    st.subheader("🔗 Interoperability — FHIR-like schema")
    st.markdown("In a real Data Platform, raw EHR data would be transformed into canonical models like FHIR to ensure interoperability and a standardized view of patient data.")
    st.markdown("Here is a conceptual FHIR-like schema and data sampling from the **raw data** in the `data` folder.")
    
    schema = {
        "Patient":{"pk":"patient_id","columns":["patient_id","nhs_number_hash","birth_date","sex","ethnicity","postcode_lsoa","imd_quintile","age_years","bmi","smoker","exercise_freq","systolic_bp","diastolic_bp","glucose_mmol","cholesterol_mmol","condition"]},
        "Encounter":{"pk":"encounter_id","fk":[{"patient_id":"Patient.patient_id"}],
                     "columns":["encounter_id","patient_id","start_dt","end_dt","type","los_days"]},
        "Outcome":{"pk":"encounter_id","fk":[{"encounter_id":"Encounter.encounter_id"}],
                   "columns":["encounter_id","readmission_30d","mortality_90d","los_days"]},
        "Consent":{"pk":"patient_id","fk":[{"patient_id":"Patient.patient_id"}],
                   "columns":["patient_id","allow_research","allow_risk_scoring","last_updated"]}
    }
    
    st.markdown("#### Conceptual Schema (FHIR-like Mapping)")
    if HAS_YAML:
        st.code(yaml.safe_dump(schema, sort_keys=False), language="yaml")
    else:
        st.code(json.dumps(schema, indent=2), language="json")

    st.markdown("#### Referential Integrity Checks (Raw Data)")
    missing_pat = (~encounters["patient_id"].isin(patients["patient_id"])).sum()
    missing_enc = (~outcomes["encounter_id"].isin(encounters["encounter_id"])).sum()
    missing_con = (~consent["patient_id"].isin(patients["patient_id"])).sum()
    colA,colB,colC = st.columns(3)
    colA.metric("Encounter → Patient missing links", int(missing_pat))
    colB.metric("Outcome → Encounter missing links", int(missing_enc))
    colC.metric("Consent → Patient missing links", int(missing_con))

    st.markdown("#### Sample Raw Tables")
    st.dataframe(patients.head(10))
    st.dataframe(encounters.head(10))
    st.dataframe(outcomes.head(10))
    st.dataframe(consent.head(10))

# ===============================================================
# 3) Secure Access & Audit Trail
# ===============================================================
elif page == "Secure Access & Audit Trail":
    st.subheader("🔒 Secure Data Access & Audit Trail")
    st.markdown("This section simulates **Role-Based Access Control (RBAC)** and maintains an **Audit Log** for all data access attempts.")

    # Convert all loaded CSVs to a dictionary keyed by resource name (no .csv) for the demo
    data_sources = {
        "Patient": patients,
        "Encounter": encounters,
        "Outcome": outcomes,
        "Consent": consent,
        "User": users
    }
    
    # --- Data Access Buttons ---
    st.markdown("### Data Access Simulation")
    
    cols = st.columns(len(data_sources))
    
    for i, resource in enumerate(data_sources.keys()):
        with cols[i]:
            if st.button(f"Read {resource}", key=f"read_{resource}"):
                if can_read(role, resource):
                    st.success(f"Access granted to {resource} (Role: {role}).")
                    
                    # Apply simple filtering/anonymization for demo purposes if needed
                    df_result = data_sources[resource].copy()
                    
                    if resource == "Patient" and role == "Analyst":
                        if 'nhs_number_hash' in df_result.columns:
                            df_result = df_result.drop(columns=['nhs_number_hash'])
                        st.caption("Note: NHS Hash dropped for Analyst role (anonymization).")
                    
                    if resource == "Patient" and role == "Researcher":
                         # Filter to only consented patients for research
                        consented_pids = consent[consent['allow_research'] == 1]['patient_id'].tolist()
                        df_result = df_result[df_result['patient_id'].isin(consented_pids)]
                        if 'nhs_number_hash' in df_result.columns:
                            df_result = df_result.drop(columns=['nhs_number_hash'])
                        st.caption("Note: Filtered by 'allow_research' and NHS Hash dropped for Researcher role.")

                    st.dataframe(df_result.head(10), use_container_width=True)
                    write_audit(user_id, "READ", resource, "ALL", purpose, "success")
                else:
                    st.error(f"Access denied to {resource} (Role: {role}).")
                    write_audit(user_id, "READ", resource, "ALL", purpose, "denied")

    st.markdown("### 📜 Audit Log")
    
    # Reload audit log for fresh display
    if os.path.exists(AUDIT_PATH):
        audit = pd.read_csv(AUDIT_PATH)
        st.dataframe(audit.sort_values("timestamp", ascending=False).head(20), use_container_width=True)
        
        # Audit Log Visualizations
        if not audit.empty:
            by_purpose = audit["purpose_of_use"].value_counts().reset_index(name="count")
            by_result = audit["result"].value_counts().reset_index(name="count")
            cols2 = st.columns(2)
            with cols2[0]:
                st.plotly_chart(px.bar(by_purpose, x="purpose_of_use", y="count", title="Events by Purpose"), use_container_width=True)
            with cols2[1]:
                st.plotly_chart(px.pie(by_result, names="result", values="count", title="Access Outcomes"), use_container_width=True)
        else:
            st.info("No audit events yet. Use the buttons above to generate some.")
    else:
        st.info("Audit log file not found.")

# ===============================================================
# 4) Predictive Analytics (tabs)
# ===============================================================
elif page == "Predictive Analytics":
    st.subheader("🤖 Predictive Analytics")
    tab_pred, tab_dash, tab_hist = st.tabs(["🧪 Prediction", "📊 Prediction Dashboard", "🕰️ Historical Dashboard"])

    # ---------- Tab 1: Prediction ----------
    with tab_pred:
        st.markdown("Choose a model and input features to generate a prediction.")
        model_choice = st.selectbox("Model", ["Readmission Risk (30d)", "Condition Detection"])
        
        if model_choice == "Readmission Risk (30d)":
            st.markdown("##### Readmission Risk Prediction (Logistic Regression)")
            with st.form("readm_form"):
                col_r_p1, col_r_p2 = st.columns(2)
                with col_r_p1:
                    age = st.number_input("Age (years)", 18, 100, 60)
                    etype = st.selectbox("Encounter type", ["inpatient","outpatient","emergency","virtual"], index=0)
                with col_r_p2:
                    imd = st.selectbox("IMD Quintile", [1,2,3,4,5], index=2)
                    los = st.number_input("Length of Stay (days)", 0, 60, 3)
                    
                submitted = st.form_submit_button("Predict Readmission Risk")
            
            if submitted:
                if can_score(role):
                    # Prepare sample input matching model features
                    sample = pd.DataFrame([{
                        "age_years": age, 
                        "log_los": np.log1p(los),
                        "is_inpatient": 1 if etype == "inpatient" else 0, 
                        "is_emergency": 1 if etype == "emergency" else 0,
                        "sex": "Male", # Default required for OHE in pipeline
                        "ethnicity": "White", # Default required for OHE in pipeline
                    }])
                    
                    try:
                        proba = readm_model.predict_proba(sample)[:, 1][0]
                        st.metric("Predicted 30d Readmission Risk", f"{proba*100:.1f}%")
                        write_audit(user_id, "SCORE", "model", "readmission", purpose, "success")
                    except Exception as e:
                        st.error(f"Prediction failed due to model error. {e}")
                        write_audit(user_id, "SCORE", "model", "readmission", purpose, "error")
                else:
                    st.error("Your role cannot score.")
                    write_audit(user_id, "SCORE", "model", "readmission", purpose, "denied")

        else: # Condition Detection
            st.markdown("##### Condition Detection Prediction (Random Forest)")
            with st.form("cond_form"):
                col1, col2 = st.columns(2)
                with col1:
                    age = st.number_input("Age (years)", 18, 100, 55, key="cond_age")
                    sex = st.selectbox("Sex", ["Male","Female"], key="cond_sex")
                    ethnicity = st.selectbox("Ethnicity", ["White","Black","Asian","Mixed","Other"], key="cond_ethnicity")
                    bmi = st.number_input("BMI", 10.0, 60.0, 29.0, key="cond_bmi")
                    smoker = st.selectbox("Smoker", ["Yes","No"], key="cond_smoker")
                with col2:
                    ex = st.selectbox("Exercise", ["Low","Moderate","High"], key="cond_ex")
                    imd = st.selectbox("IMD Quintile", [1,2,3,4,5], index=2, key="cond_imd")
                    sys = st.number_input("Systolic BP", 80, 220, 138, key="cond_sys")
                    dia = st.number_input("Diastolic BP", 40, 140, 86, key="cond_dia")
                    glu = st.number_input("Glucose (mmol/L)", 3.0, 20.0, 6.2, key="cond_glu")
                    chol = st.number_input("Cholesterol (mmol/L)", 2.0, 12.0, 5.3, key="cond_chol")
                submitted = st.form_submit_button("Predict Condition")
            
            if submitted:
                if can_score(role):
                    sample = pd.DataFrame([{
                        "age_years": age,"sex":sex,"ethnicity":ethnicity,"bmi":bmi,"smoker":smoker,
                        "exercise_freq": ex,"imd_quintile": imd,"num_conditions": 0,
                        "systolic_bp": sys,"diastolic_bp": dia,"glucose_mmol": glu,"cholesterol_mmol": chol
                    }])
                    
                    try:
                        pred = cond_model.predict(sample)[0]
                        probs = cond_model.predict_proba(sample)[0]
                        classes = cond_model.named_steps["model"].classes_
                        st.success(f"Predicted Condition: **{pred}**")
                        figp = px.bar(x=probs, y=classes, orientation="h",
                                      title="Condition Probabilities", color=probs, color_continuous_scale="Viridis")
                        figp.update_layout(xaxis=dict(range=[0,1]))
                        st.plotly_chart(figp, use_container_width=True)
                        write_audit(user_id, "SCORE", "model", "condition", purpose, "success")
                    except Exception as e:
                        st.error(f"Prediction failed due to model error. {e}")
                        write_audit(user_id, "SCORE", "model", "condition", purpose, "error")

                else:
                    st.error("Your role cannot score.")
                    write_audit(user_id, "SCORE", "model", "condition", purpose, "denied")

    # ---------- Tab 2: Prediction Dashboard ----------
    with tab_dash:
        st.markdown("Batch score a cohort and explore risk distributions.")
        
        model_choice2 = st.selectbox("Cohort model", ["Readmission Risk (30d)", "Condition Detection"], key="cmodel")
        sample_n = st.slider("Cohort size", 100, 2000, 500, 100)
        
        if st.button("Generate Cohort Analysis", key="gen_cohort"):
            cohort = patients.sample(sample_n, random_state=42).copy()
            
            if model_choice2 == "Readmission Risk (30d)":
                enc_sub = encounters.groupby("patient_id").head(1)[["patient_id","type","los_days"]]
                df = (cohort[["patient_id","age_years","sex","ethnicity"]]
                        .merge(enc_sub, on="patient_id", how="left"))
                df["type"].fillna("outpatient", inplace=True)
                df["los_days"].fillna(0, inplace=True)
                
                df["is_inpatient"] = (df["type"] == "inpatient").astype(int)
                df["is_emergency"] = (df["type"] == "emergency").astype(int)
                df["log_los"] = np.log1p(df["los_days"])
                
                if can_score(role):
                    X_input = df[["age_years", "sex", "ethnicity", "is_inpatient", "is_emergency", "log_los"]]
                    
                    try:
                        df["risk"] = readm_model.predict_proba(X_input)[:, 1]
                        
                        st.dataframe(df.sort_values("risk", ascending=False).head(20))
                        st.plotly_chart(px.histogram(df, x="risk", nbins=20, title="Readmission Risk Score Distribution"), use_container_width=True)
                        write_audit(user_id, "SCORE", "model", "readmission_batch", purpose, "success")
                    except Exception as e:
                        st.error(f"Prediction failed due to model error. {e}")
                        write_audit(user_id, "SCORE", "model", "readmission_batch", purpose, "error")

                else:
                    st.error("Your role cannot score.")
                    write_audit(user_id, "SCORE", "model", "readmission_batch", purpose, "denied")
            
            else: # Condition Detection
                X_cols = ["age_years","sex","ethnicity","bmi","smoker","exercise_freq",
                        "imd_quintile","systolic_bp","diastolic_bp","glucose_mmol","cholesterol_mmol"]
                X = cohort[X_cols].copy()
                X["num_conditions"] = 0
                
                if can_score(role):
                    try:
                        probs = cond_model.predict_proba(X)
                        classes = cond_model.named_steps["model"].classes_
                        dfp = pd.DataFrame(probs, columns=classes)
                        
                        st.plotly_chart(px.box(dfp, title="Condition Probability Distribution (Cohort)"), use_container_width=True)
                        write_audit(user_id, "SCORE", "model", "condition_batch", purpose, "success")
                    except Exception as e:
                        st.error(f"Prediction failed due to model error. {e}")
                        write_audit(user_id, "SCORE", "model", "condition_batch", purpose, "error")
                else:
                    st.error("Your role cannot score.")
                    write_audit(user_id, "SCORE", "model", "condition_batch", purpose, "denied")

    # ---------- Tab 3: Historical Dashboard ----------
    with tab_hist:
        st.markdown("Model performance snapshots over time (synthetic data only).")
        months = pd.date_range("2022-01-01", "2025-09-01", freq="MS")
        aucs = np.clip(np.random.normal(0.78, 0.02, len(months)), 0.7, 0.85)
        briers = np.clip(np.random.normal(0.095, 0.01, len(months)), 0.07, 0.12)
        perf = pd.DataFrame({"month": months, "AUC": aucs, "Brier": briers})
        c1,c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.line(perf, x="month", y="AUC", title="Readmission Model AUC Trend"), use_container_width=True)
        with c2:
            st.plotly_chart(px.line(perf, x="month", y="Brier", title="Readmission Model Brier Score Trend"), use_container_width=True)

# ===============================================================
# 5) Governance: Explainability & Consent
# ===============================================================
elif page == "Governance: Explainability & Consent":
    st.subheader("🧭 Governance — Explainable AI & Consent Simulation")

    exp_tabs = st.tabs(["🔍 Explainability (SHAP)", "📝 Consent Simulation", "📑 Fairness Snapshot"])

    # ---- Explainability
    with exp_tabs[0]:
        st.markdown("Global and local explanations for the **Condition Detection (Random Forest)** model.")
        
        if can_score(role):
            pre = cond_model.named_steps["preprocessor"]
            rf = cond_model.named_steps["model"]

            # Sample data for SHAP computation
            X = patients.sample(min(200, len(patients)), random_state=7).copy()[["age_years","sex","ethnicity","bmi","smoker","exercise_freq",
                                                    "imd_quintile","systolic_bp","diastolic_bp","glucose_mmol","cholesterol_mmol"]]
            X["num_conditions"] = 0
            
            # Transform data
            X_tr = pre.transform(X)
            if not isinstance(X_tr, np.ndarray):
                X_tr = X_tr.toarray()

            # Get feature names
            num_feats = ["age_years","bmi","imd_quintile","systolic_bp","diastolic_bp","glucose_mmol","cholesterol_mmol"]
            
            # FIX: Access OneHotEncoder directly
            ohe = pre.named_transformers_["cat"] 
            
            cat_feats = ohe.get_feature_names_out(["sex","ethnicity","smoker","exercise_freq"])
            all_feats = list(num_feats) + list(cat_feats)
            
            X_tr_df = pd.DataFrame(X_tr, columns=all_feats)

            # SHAP computation
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_tr) # shap_values is a list of arrays (N_classes, N_samples, N_features)

            st.markdown("#### Global Feature Importance (Summary Plot)")
            # FIX: Pass the full shap_values object (list of arrays) for multi-class summary plot.
            plt.figure(figsize=(8,5))
            shap.summary_plot(shap_values, X_tr_df, feature_names=all_feats, show=False) 
            st.pyplot(plt.gcf()); plt.clf()

            st.markdown("#### Local Explanation (Single Patient's Drivers)")
            idx = st.slider("Select patient sample index", 0, min(50, len(X)-1), 5)
            
            # SHAP values for the selected instance (Class 0 - shape (N_features,))
            shap_values_for_instance = shap_values[0][idx]
            # Feature values for the selected instance (Pandas Series - shape (N_features,))
            features_for_instance = X_tr_df.iloc[idx]
            
            st.markdown("##### Force Plot (for prediction at selected index)")
            # FIX: Use the pre-calculated 1D SHAP values and 1D Series data
            html_content = shap.force_plot(
                explainer.expected_value[0], 
                shap_values_for_instance, 
                features_for_instance, 
                matplotlib=False
            ).html()
            st.components.v1.html(html_content, height=350)
            
            write_audit(user_id, "READ", "model", "SHAP_explainability", purpose, "success")
        else:
             st.error("Your role cannot access model scoring/explainability.")
             write_audit(user_id, "READ", "model", "SHAP_explainability", purpose, "denied")

    # ---- Consent Simulation
    with exp_tabs[1]:
        st.markdown("This simulates updating a patient's consent status. This change impacts what data Researchers can access (Read Patients/Encounters/Outcomes) and whether a patient can be scored for Risk.")
        
        pid_options = patients["patient_id"].head(50).tolist()
        pid = st.selectbox("Select Patient ID to Edit Consent", pid_options, key="consent_pid")
        
        current_consent = consent[consent["patient_id"] == pid]
        
        if len(current_consent) == 0:
            st.error("Patient not found in consent (Use one from the Patient table list).")
        else:
            current_consent = current_consent.iloc[0]

            st.markdown(f"**Current Consent Status for {pid}:**")
            col_c_curr, col_c_new = st.columns(2)
            with col_c_curr:
                st.info(f"Allow Research: {'✅ Yes' if current_consent['allow_research'] == 1 else '❌ No'}")
                st.info(f"Allow Risk Scoring: {'✅ Yes' if current_consent['allow_risk_scoring'] == 1 else '❌ No'}")
            
            st.markdown("---")
            st.markdown("#### Update Consent (Simulated)")

            new_allow_research = st.radio("Allow Research", [1, 0], index=(0 if current_consent['allow_research'] == 1 else 1), format_func=lambda x: "Yes" if x == 1 else "No", key="new_allow_research")
            new_allow_risk_scoring = st.radio("Allow Risk Scoring", [1, 0], index=(0 if current_consent['allow_risk_scoring'] == 1 else 1), format_func=lambda x: "Yes" if x == 1 else "No", key="new_allow_risk_scoring")
            
            if st.button("Apply change (Updates data/consent.csv)"):
                consent.loc[consent["patient_id"] == pid, "allow_research"] = new_allow_research
                consent.loc[consent["patient_id"] == pid, "allow_risk_scoring"] = new_allow_risk_scoring
                
                consent.to_csv(CONSENT_PATH, index=False)
                
                write_audit(user_id, "WRITE", "Consent", pid, purpose, "success")
                
                st.success(f"Consent for Patient {pid} updated successfully. Please re-run the app to ensure all cached data loads the new consent.")

    # ---- Fairness Snapshot
    with exp_tabs[2]:
        st.markdown("#### Fairness Snapshot (Condition Model Prediction Distribution)")
        st.markdown("This checks the average model prediction confidence across different patient **Ethnicity** groups.")
        
        Xfair_cols = ["age_years","sex","ethnicity","bmi","smoker","exercise_freq",
                      "imd_quintile","systolic_bp","diastolic_bp","glucose_mmol","cholesterol_mmol"]
        Xfair = patients[Xfair_cols].copy()
        Xfair["num_conditions"] = 0
        
        if can_score(role):
            try:
                probs = cond_model.predict_proba(Xfair)
                pred = cond_model.predict(Xfair)
                df_f = patients[["ethnicity"]].copy()
                df_f["pred"] = pred
                df_f["max_prob"] = probs.max(axis=1) # Max probability as confidence score

                summ = (df_f.groupby("ethnicity")
                            .agg(Count=("pred","count"),
                                 Avg_Confidence=("max_prob","mean"),
                                 Num_Diabetes_Preds=("pred", lambda x: (x == "Diabetes").sum()))
                            .reset_index())
                st.dataframe(summ)
                
                st.plotly_chart(px.bar(summ, x="ethnicity", y="Avg_Confidence", 
                                       title="Average Model Confidence by Ethnicity"), use_container_width=True)

                st.plotly_chart(px.bar(summ, x="ethnicity", y="Num_Diabetes_Preds", 
                                       title="Number of Diabetes Predictions by Ethnicity"), use_container_width=True)
                
                write_audit(user_id, "READ", "model", "Fairness_snapshot", purpose, "success")
            except Exception as e:
                st.error(f"Could not generate fairness metrics: {e}")
                write_audit(user_id, "READ", "model", "Fairness_snapshot", purpose, "error")
        else:
            st.error("Your role cannot access model scoring/fairness analysis.")
            write_audit(user_id, "READ", "model", "Fairness_snapshot", purpose, "denied")

# ===============================================================
# Footer
# ===============================================================
st.markdown("---")
st.caption("© AI-Ready Health Data Platform — synthetic, for demonstration only.")