# ===============================================================
# 🏥 AI-Ready Health Data Platform — Streamlit App (All-in-One)
# Pages:
# 1) Historical Dashboard
# 2) Interoperability (FHIR-like schema)
# 3) Secure Data Access & Audit Trail
# 4) Predictive Analytics (tabs: Prediction | Prediction Dashboard | Historical Dashboard)
# 5) Governance: Explainable AI & Consent Simulation
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
#@st.cache_resource
def ensure_readmission_model(patients, encounters, outcomes):
    """Train (or load) a simple readmission risk model safely."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # --- merge data ---
    df = encounters.merge(patients[["patient_id", "age_years", "sex", "ethnicity"]], on="patient_id", how="left")
    df = df.merge(outcomes[["encounter_id", "readmission_30d", "los_days"]], on="encounter_id", how="left")

    # --- handle missing or missing columns ---
    # --- handle missing or missing columns ---
    if "los_days" not in df.columns:
        st.warning("⚠️ 'los_days' column missing — creating synthetic length of stay.")
        df["los_days"] = np.random.randint(1, 10, size=len(df))
    else:
    # Replace missing values safely (vectorized)
        mask = df["los_days"].isna()
        df.loc[mask, "los_days"] = np.random.randint(1, 10, size=mask.sum())

    df["los_days"] = df["los_days"].fillna(np.random.randint(1, 10, size=len(df)))
    df["is_inpatient"] = (df["type"] == "inpatient").astype(int)
    df["is_emergency"] = (df["type"] == "emergency").astype(int)
    df["log_los"] = np.log1p(df["los_days"])

    # --- define features ---
    features = [
        "age_years", "sex", "ethnicity",
        "is_inpatient", "is_emergency", "log_los"
    ]
    target = "readmission_30d"

    df = df.dropna(subset=[target])
    X, y = df[features], df[target]

    # --- preprocess ---
    num_features = ["age_years", "log_los"]
    cat_features = ["sex", "ethnicity", "is_inpatient", "is_emergency"]

    num_transformer = Pipeline([("scaler", StandardScaler())])
    cat_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features)
        ]
    )

    # --- model pipeline ---
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=200, class_weight="balanced"))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    st.success(f"✅ Readmission model trained successfully (AUC = {auc:.3f})")

    # --- save model for reuse ---
    import joblib
    joblib.dump(model, "readmission_model.pkl")

    return model

@st.cache_data(show_spinner=False)
def load_all():
    return generate_synthetic_core()

patients, encounters, outcomes, consent = load_all()
cond_model = ensure_condition_model(patients)
readm_model = ensure_readmission_model(patients, encounters, outcomes)

# ---------------------------------
# RBAC demo roles
# ---------------------------------
ROLES = {
    "Admin": {"can_read":["*"], "can_score": True},
    "Analyst": {"can_read":["Patient","Encounter","Outcome"], "can_score": True},
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
# Sidebar: Role & Purpose
# -------------------------------
st.sidebar.header("🔐 Access")
user_id = st.sidebar.text_input("User ID", value="U0001")
role = st.sidebar.selectbox("Role", list(ROLES.keys()), index=1)
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
# 1) Historical Dashboard
# ===============================================================
if page == "Historical Dashboard":
    st.subheader("📈 Historical Dashboard")

    enc = encounters.copy()
    enc["start_dt"] = pd.to_datetime(enc["start_dt"])
    enc["month"] = enc["start_dt"].dt.to_period("M").astype(str)

    vol = enc.groupby(["month","type"])["encounter_id"].count().reset_index(name="count")
    c1,c2,c3 = st.columns(3)
    c1.metric("Patients", patients["patient_id"].nunique())
    c2.metric("Encounters", enc["encounter_id"].nunique())
    c3.metric("Readmission Rate", f"{outcomes['readmission_30d'].mean()*100:.1f}%")

    st.plotly_chart(px.line(vol, x="month", y="count", color="type", title="Monthly Encounters by Type"), use_container_width=True)

    out_join = encounters[["encounter_id","start_dt"]].merge(outcomes, on="encounter_id")
    out_join["start_dt"] = pd.to_datetime(out_join["start_dt"])
    out_join["month"] = out_join["start_dt"].dt.to_period("M").astype(str)
    trend = out_join.groupby("month")[["readmission_30d","mortality_90d"]].mean().mul(100).reset_index()
    trend = trend.melt(id_vars=["month"], var_name="metric", value_name="rate_pct")
    st.plotly_chart(px.bar(trend, x="month", y="rate_pct", color="metric", barmode="group",
                           title="Monthly Outcome Rates (%)"), use_container_width=True)

    pats = patients.copy()
    pats["age_band"] = pd.cut(pats["age_years"], bins=[0,30,45,60,75,100], labels=["≤30","31–45","46–60","61–75","76+"])
    st.plotly_chart(px.histogram(pats, x="age_band", color="condition", barmode="group",
                                 title="Conditions by Age Band"), use_container_width=True)

# ===============================================================
# 2) Interoperability (FHIR-like schema)
# ===============================================================
elif page == "Interoperability (FHIR-like)":
    st.subheader("🔗 Interoperability — FHIR-like schema")
    schema = {
        "Patient":{"pk":"patient_id","columns":["patient_id","nhs_number_hash","birth_date","sex","ethnicity","postcode_lsoa","imd_quintile","age_years","bmi","smoker","exercise_freq","systolic_bp","diastolic_bp","glucose_mmol","cholesterol_mmol","condition"]},
        "Encounter":{"pk":"encounter_id","fk":[{"patient_id":"Patient.patient_id"}],
                     "columns":["encounter_id","patient_id","start_dt","end_dt","type","los_days"]},
        "Outcome":{"pk":"encounter_id","fk":[{"encounter_id":"Encounter.encounter_id"}],
                   "columns":["encounter_id","readmission_30d","mortality_90d","los_days"]},
        "Consent":{"pk":"patient_id","fk":[{"patient_id":"Patient.patient_id"}],
                   "columns":["patient_id","allow_research","allow_risk_scoring","last_updated"]}
    }
    if HAS_YAML:
        st.code(yaml.safe_dump(schema, sort_keys=False), language="yaml")
    else:
        st.code(json.dumps(schema, indent=2), language="json")

    st.markdown("#### Referential Integrity Checks")
    missing_pat = (~encounters["patient_id"].isin(patients["patient_id"])).sum()
    missing_enc = (~outcomes["encounter_id"].isin(encounters["encounter_id"])).sum()
    missing_con = (~consent["patient_id"].isin(patients["patient_id"])).sum()
    colA,colB,colC = st.columns(3)
    colA.metric("Encounter → Patient missing links", int(missing_pat))
    colB.metric("Outcome → Encounter missing links", int(missing_enc))
    colC.metric("Consent → Patient missing links", int(missing_con))

    st.markdown("#### Sample Tables")
    st.dataframe(patients.head(10))
    st.dataframe(encounters.head(10))
    st.dataframe(outcomes.head(10))
    st.dataframe(consent.head(10))

# ===============================================================
# 3) Secure Access & Audit Trail
# ===============================================================
elif page == "Secure Access & Audit Trail":
    st.subheader("🔒 Secure Data Access & Audit Trail")

    cols = st.columns(4)
    with cols[0]:
        if st.button("Read Patients"):
            if can_read(role, "Patient"):
                st.success("Access granted: Patients")
                st.dataframe(patients.head(20))
                write_audit(user_id, "READ", "Patient", "ALL", purpose, "success")
            else:
                st.error("Access denied.")
                write_audit(user_id, "READ", "Patient", "ALL", purpose, "denied")
    with cols[1]:
        if st.button("Read Encounters"):
            if can_read(role, "Encounter"):
                st.success("Access granted: Encounters")
                st.dataframe(encounters.head(20))
                write_audit(user_id, "READ", "Encounter", "ALL", purpose, "success")
            else:
                st.error("Access denied.")
                write_audit(user_id, "READ", "Encounter", "ALL", purpose, "denied")
    with cols[2]:
        if st.button("Read Outcomes"):
            if can_read(role, "Outcome"):
                st.success("Access granted: Outcomes")
                st.dataframe(outcomes.head(20))
                write_audit(user_id, "READ", "Outcome", "ALL", purpose, "success")
            else:
                st.error("Access denied.")
                write_audit(user_id, "READ", "Outcome", "ALL", purpose, "denied")
    with cols[3]:
        if st.button("Read Consent"):
            if can_read(role, "Consent"):
                st.success("Access granted: Consent")
                st.dataframe(consent.sample(20))
                write_audit(user_id, "READ", "Consent", "ALL", purpose, "success")
            else:
                st.error("Access denied.")
                write_audit(user_id, "READ", "Consent", "ALL", purpose, "denied")

    st.markdown("### 📜 Audit Log")
    if os.path.exists(AUDIT_PATH):
        audit = pd.read_csv(AUDIT_PATH)
        st.dataframe(audit.sort_values("timestamp", ascending=False).head(200), use_container_width=True)
        by_purpose = audit["purpose_of_use"].value_counts().reset_index()
        by_result = audit["result"].value_counts().reset_index()
        cols2 = st.columns(2)
        with cols2[0]:
            st.plotly_chart(px.bar(by_purpose, x="index", y="purpose_of_use", title="Events by Purpose"), use_container_width=True)
        with cols2[1]:
            st.plotly_chart(px.pie(by_result, names="index", values="result", title="Access Outcomes"), use_container_width=True)
    else:
        st.info("No audit events yet. Use the buttons above to generate some.")

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
            with st.form("readm_form"):
                age = st.number_input("Age (years)", 18, 100, 60)
                imd = st.selectbox("IMD Quintile", [1,2,3,4,5], index=2)
                los = st.number_input("Length of Stay (days)", 0, 60, 3)
                etype = st.selectbox("Encounter type", ["inpatient","outpatient","emergency","virtual"], index=0)
                is_inp = 1 if etype == "inpatient" else 0
                is_em = 1 if etype == "emergency" else 0
                submitted = st.form_submit_button("Predict Readmission Risk")
            if submitted:
                sample = pd.DataFrame([{
                    "age_years": age, "imd_quintile": imd, "log_los": np.log1p(los),
                    "is_inpatient": is_inp, "is_emergency": is_em, "type": etype
                }])
                if can_score(role):
                    proba = readm_model.predict_proba(sample)[:, 1][0]
                    st.metric("Predicted 30d Readmission Risk", f"{proba*100:.1f}%")
                    write_audit(user_id, "SCORE", "model", "readmission", purpose, "success")
                else:
                    st.error("Your role cannot score.")
                    write_audit(user_id, "SCORE", "model", "readmission", purpose, "denied")
        else:
            with st.form("cond_form"):
                col1, col2 = st.columns(2)
                with col1:
                    age = st.number_input("Age (years)", 18, 100, 55)
                    sex = st.selectbox("Sex", ["Male","Female"])
                    ethnicity = st.selectbox("Ethnicity", ["White","Black","Asian","Mixed","Other"])
                    bmi = st.number_input("BMI", 10.0, 60.0, 29.0)
                    smoker = st.selectbox("Smoker", ["Yes","No"])
                with col2:
                    ex = st.selectbox("Exercise", ["Low","Moderate","High"])
                    imd = st.selectbox("IMD Quintile", [1,2,3,4,5], index=2)
                    sys = st.number_input("Systolic BP", 80, 220, 138)
                    dia = st.number_input("Diastolic BP", 40, 140, 86)
                    glu = st.number_input("Glucose (mmol/L)", 3.0, 20.0, 6.2)
                    chol = st.number_input("Cholesterol (mmol/L)", 2.0, 12.0, 5.3)
                submitted = st.form_submit_button("Predict Condition")
            if submitted:
                sample = pd.DataFrame([{
                    "age_years": age,"sex":sex,"ethnicity":ethnicity,"bmi":bmi,"smoker":smoker,
                    "exercise_freq": ex,"imd_quintile": imd,"num_conditions": 0,
                    "systolic_bp": sys,"diastolic_bp": dia,"glucose_mmol": glu,"cholesterol_mmol": chol
                }])
                if can_score(role):
                    pred = cond_model.predict(sample)[0]
                    probs = cond_model.predict_proba(sample)[0]
                    classes = cond_model.named_steps["model"].classes_
                    st.success(f"Predicted Condition: **{pred}**")
                    figp = px.bar(x=probs, y=classes, orientation="h",
                                  title="Condition Probabilities", color=probs, color_continuous_scale="Viridis")
                    figp.update_layout(xaxis=dict(range=[0,1]))
                    st.plotly_chart(figp, use_container_width=True)
                    write_audit(user_id, "SCORE", "model", "condition", purpose, "success")
                else:
                    st.error("Your role cannot score.")
                    write_audit(user_id, "SCORE", "model", "condition", purpose, "denied")

    # ---------- Tab 2: Prediction Dashboard ----------
    with tab_dash:
        st.markdown("Batch score a cohort and explore risk distributions.")
        model_choice2 = st.selectbox("Cohort model", ["Readmission Risk (30d)", "Condition Detection"], key="cmodel")
        sample_n = st.slider("Cohort size", 100, 2000, 500, 100)
        cohort = patients.sample(sample_n, random_state=42).copy()

        if model_choice2 == "Readmission Risk (30d)":
            enc_sub = encounters.groupby("patient_id").head(1)[["patient_id","type","los_days"]]
            df = (cohort[["patient_id","age_years","imd_quintile"]]
                    .merge(enc_sub, on="patient_id", how="left"))
            df["type"].fillna("outpatient", inplace=True)
            df["los_days"].fillna(0, inplace=True)
            df["is_inpatient"] = (df["type"] == "inpatient").astype(int)
            df["is_emergency"] = (df["type"] == "emergency").astype(int)
            df["log_los"] = np.log1p(df["los_days"])
            if can_score(role):
                df["risk"] = readm_model.predict_proba(df[["age_years","imd_quintile","log_los","is_inpatient","is_emergency","type"]])[:, 1]
                st.dataframe(df.sort_values("risk", ascending=False).head(20))
                st.plotly_chart(px.histogram(df, x="risk", nbins=20, title="Risk Score Distribution"), use_container_width=True)
                write_audit(user_id, "SCORE", "model", "readmission_batch", purpose, "success")
            else:
                st.error("Your role cannot score.")
                write_audit(user_id, "SCORE", "model", "readmission_batch", purpose, "denied")
        else:
            X = cohort[["age_years","sex","ethnicity","bmi","smoker","exercise_freq","imd_quintile",
                        "systolic_bp","diastolic_bp","glucose_mmol","cholesterol_mmol"]]
            X["num_conditions"] = 0
            if can_score(role):
                probs = cond_model.predict_proba(X)
                classes = cond_model.named_steps["model"].classes_
                dfp = pd.DataFrame(probs, columns=classes)
                st.plotly_chart(px.box(dfp, title="Condition Probability Distribution (cohort)"), use_container_width=True)
                write_audit(user_id, "SCORE", "model", "condition_batch", purpose, "success")
            else:
                st.error("Your role cannot score.")
                write_audit(user_id, "SCORE", "model", "condition_batch", purpose, "denied")

    # ---------- Tab 3: Historical Dashboard ----------
    with tab_hist:
        st.markdown("Model performance snapshots over time (synthetic).")
        months = pd.date_range("2022-01-01", "2025-09-01", freq="MS")
        aucs = np.clip(np.random.normal(0.78, 0.02, len(months)), 0.7, 0.85)
        briers = np.clip(np.random.normal(0.095, 0.01, len(months)), 0.07, 0.12)
        perf = pd.DataFrame({"month": months, "AUC": aucs, "Brier": briers})
        c1,c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.line(perf, x="month", y="AUC", title="Readmission Model AUC over time"), use_container_width=True)
        with c2:
            st.plotly_chart(px.line(perf, x="month", y="Brier", title="Readmission Model Brier Score over time"), use_container_width=True)

# ===============================================================
# 5) Governance: Explainability & Consent
# ===============================================================
elif page == "Governance: Explainability & Consent":
    st.subheader("🧭 Governance — Explainable AI & Consent Simulation")

    exp_tabs = st.tabs(["🔍 Explainability (SHAP)", "📝 Consent Simulation", "📑 Fairness Snapshot"])

    # ---- Explainability
    with exp_tabs[0]:
        st.markdown("Global and local explanations for the **Condition Detection** model.")
        pre = cond_model.named_steps["preprocessor"]
        rf = cond_model.named_steps["model"]

        X = patients.sample(200, random_state=7)[["age_years","sex","ethnicity","bmi","smoker","exercise_freq",
                                                  "imd_quintile","systolic_bp","diastolic_bp","glucose_mmol","cholesterol_mmol"]]
        X["num_conditions"] = 0
        X_tr = pre.transform(X)
        if not isinstance(X_tr, np.ndarray):
            X_tr = X_tr.toarray()

        num_feats = ["age_years","bmi","imd_quintile","num_conditions","systolic_bp","diastolic_bp","glucose_mmol","cholesterol_mmol"]
        ohe = pre.named_transformers_["cat"].named_steps["onehot"]
        cat_feats = ohe.get_feature_names_out(["sex","ethnicity","smoker","exercise_freq"])
        all_feats = list(num_feats) + list(cat_feats)

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_tr)

        shap_mean = np.mean(np.abs(np.array(shap_values)), axis=0)
        st.markdown("#### Global Feature Importance")
        plt.figure(figsize=(8,5))
        shap.summary_plot(shap_mean, X_tr, feature_names=all_feats, plot_type="bar", show=False)
        st.pyplot(plt.gcf()); plt.clf()

        st.markdown("#### Local Explanation (single patient)")
        idx = st.slider("Sample index", 0, min(50, len(X)-1), 5)
        indiv = X_tr[[idx]]
        indiv_sv = explainer.shap_values(indiv)
        classes = rf.classes_
        for i, cls in enumerate(classes):
            contrib = pd.Series(indiv_sv[i][0], index=all_feats).abs().sort_values(ascending=False).head(8)
            fig = px.bar(contrib[::-1], orientation="h", title=f"Top drivers for class: {cls}")
            st.plotly_chart(fig, use_container_width=True)

    # ---- Consent Simulation
    with exp_tabs[1]:
        st.markdown("Toggle a patient's consent for risk scoring (demo only).")
        pid = st.selectbox("Patient ID", patients["patient_id"].sample(50, random_state=3))
        current = consent.loc[consent["patient_id"]==pid, "allow_risk_scoring"]
        if len(current) == 0:
            st.error("Patient not found in consent.")
        else:
            flag = int(current.iloc[0])
            st.write(f"Current allow_risk_scoring: **{flag}**")
            new_flag = st.selectbox("Set allow_risk_scoring", [0,1], index=(1 if flag==1 else 0))
            if st.button("Apply change"):
                consent.loc[consent["patient_id"]==pid, "allow_risk_scoring"] = new_flag
                consent.to_csv(CONSENT_PATH, index=False)
                st.success("Consent updated.")
                write_audit(user_id, "WRITE", "Consent", pid, purpose, "success")

    # ---- Fairness Snapshot
    with exp_tabs[2]:
        st.markdown("Quick fairness snapshot by **ethnicity** (Condition model).")
        Xfair = patients[["age_years","sex","ethnicity","bmi","smoker","exercise_freq",
                          "imd_quintile","systolic_bp","diastolic_bp","glucose_mmol","cholesterol_mmol"]].copy()
        Xfair["num_conditions"] = 0
        probs = cond_model.predict_proba(Xfair)
        pred = cond_model.predict(Xfair)
        df_f = patients[["ethnicity"]].copy()
        df_f["pred"] = pred
        df_f["prob"] = probs.max(axis=1)

        summ = (df_f.groupby("ethnicity")
                      .agg(count=("pred","count"),
                           avg_prob=("prob","mean"))
                      .reset_index())
        st.dataframe(summ)
        st.plotly_chart(px.bar(summ, x="ethnicity", y="avg_prob", title="Average Confidence by Ethnicity"), use_container_width=True)

# ===============================================================
# Footer
# ===============================================================
st.markdown("---")
st.caption("© AI-Ready Health Data Platform — synthetic, for demonstration only.")
