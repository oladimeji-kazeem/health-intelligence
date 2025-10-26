import streamlit as st
import streamlit.components.v1 as components
import shap
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns


# Define relative paths for data loading
DATA_DIR = "data"
DATA_LAKE_DIR = "data_lake"
# Assuming the user runs the app from the root of the project structure.

# Helper function to find a file, checking both current directory and a subdirectory
def find_file(relative_path):
    if os.path.exists(relative_path):
        return relative_path
    # This assumes a flat data/ or data_lake/ folder structure in the same dir as app.py
    # For the purposes of this task, we assume the data has been correctly placed.
    return relative_path


# Load datasets (assuming they are in the specified relative paths)
# We use find_file to ensure correct relative path usage
try:
    patients = pd.read_csv(find_file(os.path.join(DATA_DIR, "patients.csv")))
    encounters = pd.read_csv(find_file(os.path.join(DATA_DIR, "encounters.csv")))
    conditions = pd.read_csv(find_file(os.path.join(DATA_DIR, "conditions.csv")))
    observations = pd.read_csv(find_file(os.path.join(DATA_DIR, "observations.csv")))
    meds = pd.read_csv(find_file(os.path.join(DATA_DIR, "medication_requests.csv")))
    outcomes = pd.read_csv(find_file(os.path.join(DATA_DIR, "outcomes.csv")))
    consent = pd.read_csv(find_file(os.path.join(DATA_DIR, "consent.csv")))
    users = pd.read_csv(find_file(os.path.join(DATA_DIR, "users.csv")))
    # For audit_log, check both the governance folder (from initial app.py) and data folder (from notebook)
    if os.path.exists(find_file(os.path.join(DATA_DIR, "audit_log.csv"))):
         audit_log = pd.read_csv(find_file(os.path.join(DATA_DIR, "audit_log.csv")))
    elif os.path.exists(find_file(os.path.join("governance", "audit_log.csv"))):
         audit_log = pd.read_csv(find_file(os.path.join("governance", "audit_log.csv")))
    else:
         # If not found, create an empty one for logging
         audit_log = pd.DataFrame(columns=["log_id", "user_id", "action", "resource", "timestamp", "purpose_of_use", "result"])


    # Load FHIR-like datasets from the data lake structure
    fhir_patients = pd.read_csv(find_file(os.path.join(DATA_LAKE_DIR, "patients", "fhir_patients.csv")))
    fhir_encounters = pd.read_csv(find_file(os.path.join(DATA_LAKE_DIR, "encounters", "fhir_encounters.csv")))
    fhir_conditions = pd.read_csv(find_file(os.path.join(DATA_LAKE_DIR, "conditions", "fhir_conditions.csv")))
    fhir_observations = pd.read_csv(find_file(os.path.join(DATA_LAKE_DIR, "observations", "fhir_observations.csv")))
    fhir_meds = pd.read_csv(find_file(os.path.join(DATA_LAKE_DIR, "medication_requests", "fhir_medication_requests.csv")))

except FileNotFoundError as e:
    st.error(f"Error loading data files: {e}. Please ensure the data generation and ingestion steps were completed and that the 'data' and 'data_lake' directories with the necessary CSVs are correctly placed.")
    st.stop() # Stop the app if data loading fails


# Define the datasets dictionary at the top level so it's accessible everywhere
datasets = {
    "patients.csv": patients,
    "encounters.csv": encounters,
    "conditions.csv": conditions,
    "observations.csv": observations,
    "medication_requests.csv": meds,
    "outcomes.csv": outcomes,
    "consent.csv": consent,
    "users.csv": users,
    "audit_log.csv": audit_log
}


# Assume common_conditions list is available
common_conditions = ["I10", "E11", "I50", "J45", "J44", "N18", "C34", "C50", "F32", "E78"]

# --- Data Preparation for Models (using global dataframes) ---

# Readmission Model Data Prep
@st.cache_data
def prepare_readmission_data(fhir_encounters, fhir_patients, fhir_conditions):
    readmission_data = pd.merge(fhir_encounters, fhir_patients[["id", "gender", "extension.ethnicity", "extension.age_years", "extension.imd_quintile"]],
                                left_on="subject.reference", right_on="id", how="left", suffixes=('', '_patient'))
    readmission_data = readmission_data.drop(columns=["id_patient"])

    condition_list_per_encounter = fhir_conditions.groupby("encounter.reference")["code.coding.code"].agg(list).reset_index()
    condition_list_per_encounter.rename(columns={"encounter.reference": "encounter_id", "code.coding.code": "condition_codes"}, inplace=True)
    condition_list_per_encounter["encounter_id"] = condition_list_per_encounter["encounter_id"].str.replace("Encounter/", "")

    readmission_data = pd.merge(readmission_data, condition_list_per_encounter, left_on="id", right_on="encounter_id", how="left")
    readmission_data = readmission_data.drop(columns=["encounter_id"])

    # Simplify condition features: presence of common conditions
    common_conditions = ["I10", "E11", "I50", "J45", "J44", "N18", "C34", "C50", "F32", "E78"]
    for cond in common_conditions:
        readmission_data[f"has_condition_{cond}"] = readmission_data["condition_codes"].apply(lambda x: 1 if isinstance(x, list) and cond in x else 0)
    readmission_data = readmission_data.drop(columns=["condition_codes"])

    potential_feature_cols_readmission = [col for col in readmission_data.columns if col not in ["id", "patient_id", "subject.reference", "encounter.reference", "period.start", "period.end", "extension.readmission_30d", "extension.mortality_90d"]]
    
    numerical_cols_readmission = readmission_data[potential_feature_cols_readmission].select_dtypes(include=np.number).columns.tolist()
    categorical_cols_readmission = readmission_data[potential_feature_cols_readmission].select_dtypes(include='object').columns.tolist()

    for col in categorical_cols_readmission:
        readmission_data[col] = readmission_data[col].fillna('Missing')

    readmission_data = pd.get_dummies(readmission_data, columns=categorical_cols_readmission, dummy_na=False, drop_first=True)

    final_feature_cols_readmission = [col for col in readmission_data.columns if col not in ["id", "patient_id", "subject.reference", "encounter.reference", "period.start", "period.end", "extension.readmission_30d", "extension.mortality_90d", "extension.age_years", "extension.imd_quintile"]]
    
    X_readmission = readmission_data[final_feature_cols_readmission].copy()
    y_readmission = readmission_data["extension.readmission_30d"]

    for col in X_readmission.select_dtypes(include=np.number).columns:
        if X_readmission[col].isnull().any():
            median_val = X_readmission[col].median()
            X_readmission[col] = X_readmission[col].fillna(median_val)
            
    return X_readmission, y_readmission

X_readmission, y_readmission = prepare_readmission_data(fhir_encounters, fhir_patients, fhir_conditions)

# Chronic Disease Model Data Prep
@st.cache_data
def prepare_chronic_disease_data(fhir_conditions, fhir_patients, observations, meds):
    patients_with_diabetes = fhir_conditions[fhir_conditions["code.coding.code"] == "E11"]["subject.reference"].unique()
    patient_ids_all = fhir_patients["id"].unique()
    diabetes_target = pd.DataFrame({
        "id": patient_ids_all,
        "has_diabetes_E11": [1 if "Patient/"+patient_id in patients_with_diabetes else 0 for patient_id in patient_ids_all]
    })
    chronic_disease_data = pd.merge(fhir_patients[["id", "gender", "extension.ethnicity", "extension.age_years", "extension.imd_quintile"]],
                                    diabetes_target, on="id", how="left")
    latest_observations = observations.sort_values(by="taken_dt").drop_duplicates(subset=["patient_id", "code"], keep="last")
    obs_pivot = latest_observations.pivot(index="patient_id", columns="code", values="value_num").reset_index()
    chronic_disease_data = pd.merge(chronic_disease_data, obs_pivot, left_on="id", right_on="patient_id", how="left")
    chronic_disease_data = chronic_disease_data.drop(columns=["patient_id"])
    meds_relevant = meds[meds["drug_code"].isin(["MET", "INS"])].copy()
    meds_presence = meds_relevant.groupby("patient_id")["drug_code"].agg(lambda x: list(set(x))).reset_index()
    meds_presence.rename(columns={"drug_code": "medication_codes"}, inplace=True)
    for drug in ["MET", "INS"]:
        meds_presence[f"has_medication_{drug}"] = meds_presence["medication_codes"].apply(lambda x: 1 if isinstance(x, list) and drug in x else 0)
    meds_presence = meds_presence.drop(columns=["medication_codes"])
    chronic_disease_data = pd.merge(chronic_disease_data, meds_presence, left_on="id", right_on="patient_id", how="left")
    chronic_disease_data = chronic_disease_data.drop(columns=["patient_id"])
    conditions_relevant_chronic = fhir_conditions[fhir_conditions["code.coding.code"].isin(["E78", "I10"])].copy()
    conditions_relevant_chronic["patient_id"] = conditions_relevant_chronic["subject.reference"].str.replace("Patient/", "")
    conditions_presence_chronic = conditions_relevant_chronic.groupby("patient_id")["code.coding.code"].agg(lambda x: list(set(x))).reset_index()
    conditions_presence_chronic.rename(columns={"code.coding.code": "related_condition_codes"}, inplace=True)
    for cond in ["E78", "I10"]:
        conditions_presence_chronic[f"has_related_condition_{cond}"] = conditions_presence_chronic["related_condition_codes"].apply(lambda x: 1 if isinstance(x, list) and cond in x else 0)
    conditions_presence_chronic = conditions_presence_chronic.drop(columns=["related_condition_codes"])
    chronic_disease_data = pd.merge(chronic_disease_data, conditions_presence_chronic, left_on="id", right_on="patient_id", how="left")
    chronic_disease_data = chronic_disease_data.drop(columns=["patient_id"])

    numerical_cols_chronic = chronic_disease_data.select_dtypes(include=np.number).columns.tolist()
    numerical_cols_to_impute_chronic = [col for col in numerical_cols_chronic if col != "has_diabetes_E11"]
    for col in numerical_cols_to_impute_chronic:
        if chronic_disease_data[col].isnull().any():
            median_val = chronic_disease_data[col].median()
            chronic_disease_data[col] = chronic_disease_data[col].fillna(median_val)

    categorical_cols_chronic = chronic_disease_data.select_dtypes(include='object').columns.tolist()
    categorical_cols_to_impute_chronic = [col for col in categorical_cols_chronic if col != "id"]
    for col in categorical_cols_to_impute_chronic:
         chronic_disease_data[col] = chronic_disease_data[col].fillna('Missing')

    chronic_disease_data = pd.get_dummies(chronic_disease_data, columns=categorical_cols_to_impute_chronic, dummy_na=False, drop_first=True)
    chronic_disease_features = [col for col in chronic_disease_data.columns if col not in ["id", "has_diabetes_E11"]]
    chronic_disease_target = "has_diabetes_E11"
    X_chronic = chronic_disease_data[chronic_disease_features]
    y_chronic = chronic_disease_data[chronic_disease_target]
    
    return X_chronic, y_chronic

X_chronic, y_chronic = prepare_chronic_disease_data(fhir_conditions, fhir_patients, observations, meds)


# --- Model Training (cached to avoid re-training on each interaction) ---

@st.cache_resource
def train_readmission_model(X, y):
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    return model

@st.cache_resource
def train_chronic_model(X, y):
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    return model

model_readmission = train_readmission_model(X_readmission, y_readmission)
model_chronic = train_chronic_model(X_chronic, y_chronic)

# --- Secure Access Functions (now referencing global/cached dataframes) ---

def check_permission(user_id, resource):
    """Simulates checking user permissions based on role and resource."""
    user_info = users[users["user_id"] == user_id]
    if user_info.empty:
        return False # User not found
    user_role = user_info["role"].iloc[0]

    # Define access rules
    if user_role == "Admin":
        return True
    elif user_role == "Analyst":
        return resource in ["patients", "encounters", "conditions", "observations", "medication_requests", "outcomes", "users"]
    elif user_role == "Clinician":
        return resource in ["patients", "encounters", "conditions", "observations", "medication_requests", "outcomes"]
    elif user_role == "Researcher":
        return resource in ["patients", "encounters", "conditions", "observations", "medication_requests", "outcomes"]
    else:
        return False

# Global audit_log DataFrame (loaded or initialized empty)
# Since audit_log is already defined globally by the initial load attempt, we just ensure the function uses it.

def log_access(user_id, action, resource, purpose_of_use, result):
    """Logs data access attempts."""
    global audit_log
    log_entry = {
        # Using simple string interpolation for log_id now that we are outside the multiline f-string generation
        "log_id": f"L{str(len(audit_log) + 1).zfill(7)}",
        "user_id": user_id,
        "action": action,
        "resource": resource,
        "timestamp": datetime.now().isoformat(sep=" "),
        "purpose_of_use": purpose_of_use,
        "result": result
    }
    # Using pd.concat to append a new row
    if audit_log.empty:
        audit_log = pd.DataFrame([log_entry])
    else:
        audit_log = pd.concat([audit_log, pd.DataFrame([log_entry])], ignore_index=True)


def retrieve_data(user_id, resource, patient_id=None, purpose_of_use="unknown", allow_research=None):
    """Simulates data retrieval based on user permissions and optional filtering, with logging and consent check."""
    action = "READ"
    user_info = users[users["user_id"] == user_id]
    if user_info.empty:
        log_access(user_id, action, resource, purpose_of_use, "denied - user not found")
        return f"Access Denied: User {user_id} not found."
    user_role = user_info["role"].iloc[0]

    # Enforce consent for Researchers accessing patient-specific data
    if user_role == "Researcher" and patient_id:
        if allow_research is None:
             log_access(user_id, action, resource, purpose_of_use, "denied - consent unknown")
             return f"Access Denied: Consent status unknown for patient {patient_id}. Access denied."
        elif allow_research == 0:
            log_access(user_id, action, resource, purpose_of_use, "denied - no research consent")
            return f"Access Denied: Patient {patient_id} has not consented to research."

    has_permission = check_permission(user_id, resource)
    if not has_permission:
        log_access(user_id, action, resource, purpose_of_use, "denied")
        return f"Access Denied: User {user_id} does not have permission to access {resource}."

    # Load the data for the resource
    try:
        if resource in datasets:
            df = datasets[resource].copy()
        else:
             df_name = f"fhir_{resource}.csv"
             file_path_lake = os.path.join(DATA_LAKE_DIR, resource, df_name)
             df = pd.read_csv(file_path_lake).copy()
        
        log_access(user_id, action, resource, purpose_of_use, "success")
    except FileNotFoundError:
        log_access(user_id, action, resource, purpose_of_use, "error")
        return f"Error: Resource '{resource}' not found."
    except Exception as e:
        log_access(user_id, action, resource, purpose_of_use, "error")
        return f"An error occurred: {e}"


    # Implement filtering/anonymization
    if user_role == "Clinician" and patient_id:
        if resource == "patients":
            return df[df["id"] == patient_id]
        elif resource in ["encounters", "conditions", "observations", "medication_requests", "outcomes"]:
             return df[df["subject.reference"] == f"Patient/{patient_id}"]
        else:
             return "Filtering by patient ID is not supported for this resource."

    elif user_role == "Researcher" and resource in ["patients", "encounters", "conditions", "observations", "medication_requests", "outcomes"]:
        # Filter by consent (which was checked above)
        allowed_patients = consent[consent["allow_research"] == 1]["patient_id"].unique()
        if resource == "patients":
             # Researcher gets anonymized patient data if accessing the full list
             anonymized_df = df[df["id"].isin(allowed_patients)].drop(columns=["nhs_number_hash"], errors="ignore")
             return anonymized_df
        else:
             return df[df["subject.reference"].isin([f"Patient/{pid}" for pid in allowed_patients])]

    elif user_role == "Analyst" and resource == "patients":
         # Simulate anonymization for Analyst (e.g., drop NHS number hash)
         return df.drop(columns=["nhs_number_hash"], errors="ignore")

    return df


# --- Streamlit App Layout and Content ---

st.set_page_config(layout="wide")

st.title("🏥 AI-Ready Health Data Platform Showcase")
st.write("This application demonstrates the key features of an AI-Ready Health Data Platform using **synthetic NHS data**.")


# --- Sidebar Navigation ---
page = st.sidebar.radio(
    "Select a Section",
    ["Data Overview", "Predictive Models", "Secure Access & Governance"]
)


# --- Data Overview ---
if page == "Data Overview":
    st.header("📊 Data Overview")
    st.write("Explore the synthetic NHS datasets, their FHIR-like transformations, and key demographic distributions.")

    tab1, tab2, tab3 = st.tabs(["Synthetic Datasets", "FHIR-like Datasets", "Demographics & Observations"])

    with tab1:
        st.subheader("Synthetic NHS Datasets (Source Data)")
        
        # Define datasets dictionary for display
        datasets_display = {
            "patients.csv": patients,
            "encounters.csv": encounters,
            "conditions.csv": conditions,
            "observations.csv": observations,
            "medication_requests.csv": meds,
            "outcomes.csv": outcomes,
            "consent.csv": consent,
            "users.csv": users,
            "audit_log.csv": audit_log
        }
        
        # Display synthetic datasets
        for name, df in datasets_display.items():
            st.markdown(f"**{name}** (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
            st.dataframe(df.head(5))

    with tab2:
        st.subheader("FHIR-like Transformed Datasets (Data Lake)")
        
        fhir_datasets = {
            "fhir_patients.csv": fhir_patients,
            "fhir_encounters.csv": fhir_encounters,
            "fhir_conditions.csv": fhir_conditions,
            "fhir_observations.csv": fhir_observations,
            "fhir_medication_requests.csv": fhir_meds
        }
        # Display FHIR-like datasets
        for name, df in fhir_datasets.items():
            st.markdown(f"**{name}** (FHIR-like Resource) (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
            st.dataframe(df.head(5))

    with tab3:
        st.subheader("Demographics & Observations Visualizations")
        
        # --- Age Distribution ---
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(patients["age_years"], bins=20, kde=True, ax=ax)
        ax.set_title("Distribution of Patient Age")
        ax.set_xlabel("Age (Years)")
        ax.set_ylabel("Number of Patients")
        st.pyplot(fig)
        plt.close(fig)

        col1, col2 = st.columns(2)
        
        # --- Sex Distribution ---
        with col1:
            fig, ax = plt.subplots(figsize=(5, 5))
            patients["sex"].value_counts().plot(kind="bar", ax=ax, rot=0)
            ax.set_title("Distribution of Patient Sex")
            ax.set_xlabel("Sex")
            ax.set_ylabel("Number of Patients")
            st.pyplot(fig)
            plt.close(fig)
        
        # --- Ethnicity Distribution ---
        with col2:
            fig, ax = plt.subplots(figsize=(5, 5))
            patients["ethnicity"].value_counts().plot(kind="bar", ax=ax, rot=45)
            ax.set_title("Distribution of Patient Ethnicity")
            ax.set_xlabel("Ethnicity")
            ax.set_ylabel("Number of Patients")
            st.pyplot(fig)
            plt.close(fig)

        # --- BP & BMI Distributions ---
        st.markdown("---")
        st.subheader("Clinical Observations")
        
        bp_bmi_observations = observations[observations["code"].isin(["BP_SYS", "BP_DIA", "BMI"])].copy()
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.histplot(bp_bmi_observations[bp_bmi_observations["code"] == "BP_SYS"]["value_num"], bins=30, kde=True, ax=ax)
            ax.set_title("Systolic BP")
            ax.set_xlabel("BP_SYS (mmHg)")
            st.pyplot(fig)
            plt.close(fig)

        with col4:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.histplot(bp_bmi_observations[bp_bmi_observations["code"] == "BP_DIA"]["value_num"], bins=30, kde=True, ax=ax)
            ax.set_title("Diastolic BP")
            ax.set_xlabel("BP_DIA (mmHg)")
            st.pyplot(fig)
            plt.close(fig)

        with col5:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.histplot(bp_bmi_observations[bp_bmi_observations["code"] == "BMI"]["value_num"], bins=30, kde=True, ax=ax)
            ax.set_title("BMI")
            ax.set_xlabel("BMI (kg/m2)")
            st.pyplot(fig)
            plt.close(fig)


# --- Predictive Models ---
elif page == "Predictive Models":
    st.header("🤖 Predictive Models & Explainable AI")
    st.write("Interact with trained machine learning models for clinical risk prediction. Note the consent check for the Chronic Disease model.")

    model_tab1, model_tab2 = st.tabs(["Readmission Risk Model", "Chronic Disease Model (Diabetes)"])

    # Readmission Risk Model
    with model_tab1:
        st.subheader("30-Day Readmission Risk Model")
        
        # --- Model Prediction Input ---
        with st.form("readmission_form"):
            los_days = st.number_input("Length of Stay (days)", min_value=0, value=int(X_readmission['extension.los_days'].median()), key="readmission_los")

            st.markdown("**Encounter & Demographic Features**")
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                encounter_types = encounters["type"].unique().tolist() # Note: The model uses one-hot encoding derived from class.code, but the source data uses "type"
                selected_encounter_type = st.radio("Encounter Type", encounter_types, key="readmission_etype")

                genders = patients["sex"].unique().tolist()
                selected_gender_readmission = st.radio("Gender (Patient Sex)", genders, key="readmission_gender")
            
            with col_r2:
                specialties = encounters["admitting_specialty"].unique().tolist()
                selected_specialty = st.selectbox("Admitting Specialty", specialties, key="readmission_specialty")

                ethnicities = patients["ethnicity"].unique().tolist()
                selected_ethnicity_readmission = st.selectbox("Ethnicity", ethnicities, key="readmission_ethnicity")


            st.markdown("**Condition Features**")
            selected_conditions = st.multiselect("Select any relevant conditions", common_conditions, key="readmission_conditions")
            
            predict_button = st.form_submit_button("Predict Readmission Risk")

        if predict_button:
            # Create input DataFrame
            input_data_readmission = pd.DataFrame({'extension.los_days': [los_days]})

            # Handle one-hot encoded categorical features (using class.code columns which map to encounter "type")
            for etype in ['inpatient', 'outpatient', 'emergency', 'virtual']:
                input_data_readmission[f'class.code_{etype}'] = 1 if selected_encounter_type == etype else 0

            # Handle specialty encoding
            for specialty in specialties:
                 # The model uses 'serviceProvider.display_' prefix from the FHIR-like data
                 input_data_readmission[f'serviceProvider.display_{specialty}'] = 1 if selected_specialty == specialty else 0

            # Handle gender encoding
            for gender in genders:
                 input_data_readmission[f'gender_{gender}'] = 1 if selected_gender_readmission == gender else 0

            # Handle ethnicity encoding
            for ethnicity in ethnicities:
                 # The model uses 'extension.ethnicity_' prefix from the FHIR-like data
                 input_data_readmission[f'extension.ethnicity_{ethnicity}'] = 1 if selected_ethnicity_readmission == ethnicity else 0

            # Handle condition presence features
            for cond in common_conditions:
                input_data_readmission[f"has_condition_{cond}"] = 1 if cond in selected_conditions else 0

            # Ensure all columns in X_readmission are present, fill missing with 0
            for col in X_readmission.columns:
                if col not in input_data_readmission.columns:
                    input_data_readmission[col] = 0

            input_data_readmission = input_data_readmission[X_readmission.columns]

            # Make prediction
            readmission_prediction = model_readmission.predict(input_data_readmission)[0]
            readmission_probability = model_readmission.predict_proba(input_data_readmission)[:, 1][0]

            # Display prediction
            st.subheader("Prediction Result:")
            if readmission_prediction == 1:
                st.error(f"⚠️ High Risk of Readmission (Probability: {readmission_probability:.2f})")
            else:
                st.success(f"✅ Low Risk of Readmission (Probability: {readmission_probability:.2f})")
            
            st.markdown("---")
            
            # --- Model Performance (Readmission) ---
            st.subheader("Model Performance Snapshot")
            
            # Note: We need to re-calculate the metrics here as they weren't saved
            y_pred_readmission = model_readmission.predict(X_test_readmission)
            y_pred_proba_readmission = model_readmission.predict_proba(X_test_readmission)[:, 1]
            report_readmission = classification_report(y_test_readmission, y_pred_readmission, output_dict=True)
            auc_score_readmission = roc_auc_score(y_test_readmission, y_pred_proba_readmission)

            metrics_readmission = {
                "Precision (Class 1)": report_readmission["1"]["precision"],
                "Recall (Class 1)": report_readmission["1"]["recall"],
                "F1-score (Class 1)": report_readmission["1"]["f1-score"],
                "AUC Score": auc_score_readmission
            }
            metrics_df_readmission = pd.DataFrame(list(metrics_readmission.items()), columns=["Metric", "Score"])

            fig_perf, ax_perf = plt.subplots(figsize=(8, 4))
            sns.barplot(x="Metric", y="Score", data=metrics_df_readmission, ax=ax_perf)
            ax_perf.set_title("Readmission Risk Model Performance")
            ax_perf.set_ylim(0, 1.0)
            st.pyplot(fig_perf)
            plt.close(fig_perf)

    # Chronic Disease Detection Model
    with model_tab2:
        st.subheader("Chronic Disease Detection Model (Type 2 Diabetes - E11)")
        st.warning("Note: This model prediction requires **Patient Consent for Risk Scoring** to be checked for the provided Patient ID.")
        
        # --- Model Prediction Input ---
        with st.form("chronic_form"):
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                age_years = st.number_input("Age (Years)", min_value=0, max_value=100, value=int(chronic_disease_data['extension.age_years'].median()), key="chronic_age")
                imd_quintile = st.select_slider("IMD Quintile", options=[1, 2, 3, 4, 5], value=int(chronic_disease_data['extension.imd_quintile'].median()), key="chronic_imd")
                bmi = st.number_input("BMI (kg/m2)", min_value=10.0, max_value=60.0, value=float(chronic_disease_data['BMI'].median()), key="chronic_bmi")
                bp_sys = st.number_input("Systolic BP (mmHg)", min_value=50.0, max_value=250.0, value=float(chronic_disease_data['BP_SYS'].median()), key="chronic_bpsys")
                bp_dia = st.number_input("Diastolic BP (mmHg)", min_value=30.0, max_value=150.0, value=float(chronic_disease_data['BP_DIA'].median()), key="chronic_bpdia")
            
            with col_c2:
                chol = st.number_input("Cholesterol (mmol/L)", min_value=1.0, max_value=10.0, value=float(chronic_disease_data['CHOL'].median()), key="chronic_chol")
                creat = st.number_input("Creatinine (umol/L)", min_value=20.0, max_value=500.0, value=float(chronic_disease_data['CREAT'].median()), key="chronic_creat")
                hba1c = st.number_input("HbA1c (mmol/mol)", min_value=10.0, max_value=200.0, value=float(chronic_disease_data['HBA1C'].median()), key="chronic_hba1c")

                st.markdown("**History**")
                has_metformin = st.checkbox("Prescribed Metformin (MET)", key="chronic_metformin")
                has_insulin = st.checkbox("Prescribed Insulin (INS)", key="chronic_insulin")
                has_hyperlipidaemia = st.checkbox("Diagnosed Hyperlipidaemia (E78)", key="chronic_hyperlipidaemia")
                has_hypertension = st.checkbox("Diagnosed Hypertension (I10)", key="chronic_hypertension")

            patient_id_for_prediction = st.text_input("Enter Patient ID (e.g., P0000001) for Consent Check and Prediction", key="patient_id_prediction")
            
            col_c3, col_c4 = st.columns(2)
            with col_c3:
                predict_chronic_button = st.form_submit_button("Predict Chronic Disease Risk")
            with col_c4:
                explain_chronic_button = st.form_submit_button("Explain Last Prediction")

        
        # --- Prediction Logic ---
        if predict_chronic_button or explain_chronic_button:
            
            if patient_id_for_prediction:
                patient_info = fhir_patients[fhir_patients["id"] == patient_id_for_prediction]
                patient_consent = consent[consent["patient_id"] == patient_id_for_prediction]
                
                # Default to deny if patient not found or consent not available/denied
                is_consented = False
                if not patient_consent.empty and patient_consent["allow_risk_scoring"].iloc[0] == 1:
                    is_consented = True
                
                if is_consented or not predict_chronic_button: # Allow explanation only if prediction was made and consent granted
                    
                    # 1. Create Input DataFrame (use current session state for feature values)
                    input_data_chronic_base = pd.DataFrame({
                        'extension.age_years': [age_years],
                        'extension.imd_quintile': [imd_quintile],
                        'BMI': [bmi],
                        'BP_SYS': [bp_sys],
                        'BP_DIA': [bp_dia],
                        'CHOL': [chol],
                        'CREAT': [creat],
                        'HBA1C': [hba1c],
                        'has_medication_MET': [1 if has_metformin else 0],
                        'has_medication_INS': [1 if has_insulin else 0],
                        'has_related_condition_E78': [1 if has_hyperlipidaemia else 0],
                        'has_related_condition_I10': [1 if has_hypertension else 0]
                    })
                    
                    # 2. Handle categorical features
                    gender_cols = [col for col in X_chronic.columns if col.startswith('gender_')]
                    ethnicity_cols = [col for col in X_chronic.columns if col.startswith('extension.ethnicity_')]
                    
                    if not patient_info.empty:
                        selected_gender_chronic = patient_info["gender"].iloc[0]
                        selected_ethnicity_chronic = patient_info["extension.ethnicity"].iloc[0]
                    else:
                         # Fallback using the selection from the Readmission model section (not ideal, but necessary for demo completeness)
                        selected_gender_chronic = st.session_state.readmission_gender
                        selected_ethnicity_chronic = st.session_state.readmission_ethnicity

                    for col in gender_cols:
                        gender_value = col.replace('gender_', '')
                        input_data_chronic_base[col] = 1 if selected_gender_chronic == gender_value else 0

                    for col in ethnicity_cols:
                        ethnicity_value = col.replace('extension.ethnicity_', '')
                        input_data_chronic_base[col] = 1 if selected_ethnicity_chronic == ethnicity_value else 0

                    # 3. Finalize Input DataFrame
                    input_data_chronic = input_data_chronic_base.copy()
                    for col in X_chronic.columns:
                        if col not in input_data_chronic.columns:
                            input_data_chronic[col] = 0
                    input_data_chronic = input_data_chronic[X_chronic.columns]
                    
                    # Store input for explanation
                    st.session_state['last_chronic_input'] = input_data_chronic.iloc[[0]]
                    
                    # 4. Perform Prediction
                    if predict_chronic_button:
                        chronic_prediction = model_chronic.predict(input_data_chronic)[0]
                        chronic_probability = model_chronic.predict_proba(input_data_chronic)[:, 1][0]

                        # Display prediction
                        st.subheader("Prediction Result:")
                        if chronic_prediction == 1:
                            st.error(f"⚠️ High Risk of Type 2 Diabetes (Probability: {chronic_probability:.2f})")
                        else:
                            st.success(f"✅ Low Risk of Type 2 Diabetes (Probability: {chronic_probability:.2f})")


                
                else:
                    if predict_chronic_button:
                        st.warning(f"Prediction withheld for patient ID '{patient_id_for_prediction}' due to lack of consent for risk scoring.")
            
            else:
                 st.warning("Please enter a Patient ID to check consent and make a prediction.")

        
        # --- Explanation Logic ---
        if explain_chronic_button:
            if 'last_chronic_input' in st.session_state:
                sample_input_chronic = st.session_state['last_chronic_input']
                patient_id_for_prediction_exp = patient_id_for_prediction # Use the ID from the form
                
                patient_consent = consent[consent["patient_id"] == patient_id_for_prediction_exp]
                is_consented = False
                if not patient_consent.empty and patient_consent["allow_risk_scoring"].iloc[0] == 1:
                    is_consented = True

                if is_consented:
                    st.subheader("Explanation for Chronic Disease Prediction (Explainable AI):")
                    st.markdown("This **SHAP Force Plot** shows how each feature contributed to the final prediction for the selected patient. Features pushing the prediction higher (towards positive/Diabetes) are red, and those pushing it lower (towards negative/No Diabetes) are blue.")
                    
                    # Create a SHAP explainer for the chronic disease model
                    explainer_chronic = shap.Explainer(model_chronic, X_chronic.sample(100, random_state=42))

                    # Calculate SHAP values for the sample input
                    shap_values_chronic = explainer_chronic(sample_input_chronic)

                    # Generate and display the SHAP force plot for the individual prediction
                    # Use st.components.v1.html to display the plot
                    html_content = shap.force_plot(explainer_chronic.expected_value, shap_values_chronic.values[0,:], sample_input_chronic.iloc[0,:]).html()
                    components.html(html_content, width=1000, height=350)
                
                else:
                    st.warning(f"Explanation withheld for patient ID '{patient_id_for_prediction_exp}' due to lack of consent for risk scoring.")
            
            else:
                st.warning("Please make a prediction first to generate an explanation.")


# --- Secure Access & Governance ---
elif page == "Secure Access & Governance":
    st.header("🔒 Secure Access & Governance")
    st.write("This section demonstrates **Role-Based Access Control (RBAC)** and **Audit Trails** on the FHIR-like data lake.")

    governance_tab1, governance_tab2, governance_tab3 = st.tabs(["Secure Data Access", "Consent Simulation", "Audit Trail"])

    # Secure Data Access
    with governance_tab1:
        st.subheader("Simulated Secure Data Access (RBAC)")

        roles = users["role"].unique().tolist()
        selected_role = st.selectbox("Select User Role", roles, key="access_role")

        resource_names = [name.replace(".csv", "") for name in datasets.keys()]
        selected_resource = st.selectbox("Select Resource to Access", resource_names, key="access_resource")

        user_id_for_role = users[users["role"] == selected_role]["user_id"].iloc[0]

        patient_id_input = None
        if selected_role in ["Clinician", "Researcher"]:
            patient_id_input = st.text_input(f"Enter Patient ID (e.g., P0000001) for {selected_role}", key="access_patient_id")
            if patient_id_input == "":
                patient_id_input = None

        if st.button("Attempt Data Access"):
            st.info(f"User: **{user_id_for_role}** (Role: **{selected_role}**) attempting to access **{selected_resource}**.")

            allow_research = None
            if patient_id_input:
                 patient_consent_status = consent[consent["patient_id"] == patient_id_input]
                 if not patient_consent_status.empty:
                     allow_research = patient_consent_status["allow_research"].iloc[0]

            access_result = retrieve_data(user_id_for_role, selected_resource, patient_id=patient_id_input, purpose_of_use="care", allow_research=allow_research)

            st.subheader("Access Result:")
            if isinstance(access_result, pd.DataFrame):
                st.success("Access Granted with Filtering/Anonymization.")
                st.dataframe(access_result.head(10))
                st.write(f"Shape: {access_result.shape}")
                if selected_role == "Analyst" and selected_resource == "patients":
                    st.caption("Note: As an Analyst, direct identifiers like 'nhs_number_hash' are removed (anonymization).")
                elif selected_role == "Researcher" and selected_resource == "patients":
                    st.caption("Note: As a Researcher, access is filtered by 'allow_research' consent and direct identifiers are removed.")
                elif selected_role == "Clinician" and patient_id_input:
                    st.caption(f"Note: As a Clinician, data is filtered to records related to Patient ID: {patient_id_input}.")
            else:
                st.error(access_result)


    # Consent Simulation
    with governance_tab2:
        st.subheader("Consent Simulation")
        st.write("Simulate updating a patient's consent status (data update is temporary in this demo).")
        
        # Display current list of patients for selection
        consent_patient_id = st.selectbox("Select Patient ID to Edit Consent", patients["patient_id"].head(20).tolist(), key="consent_pid")
        
        # Get current consent status for the selected patient
        current_consent = consent[consent["patient_id"] == consent_patient_id]
        if not current_consent.empty:
            current_consent = current_consent.iloc[0]

            st.markdown(f"**Current Consent Status for {consent_patient_id}:**")
            col_c_curr, col_c_new = st.columns(2)
            with col_c_curr:
                st.info(f"Allow Research: {'✅ Yes' if current_consent['allow_research'] == 1 else '❌ No'}")
                st.info(f"Allow Risk Scoring: {'✅ Yes' if current_consent['allow_risk_scoring'] == 1 else '❌ No'}")
            
            st.markdown("---")
            st.markdown("**Update Consent:**")

            new_allow_research = st.radio("Allow Research", [1, 0], index=(0 if current_consent['allow_research'] == 1 else 1), format_func=lambda x: "Yes" if x == 1 else "No", key="new_allow_research")
            new_allow_risk_scoring = st.radio("Allow Risk Scoring", [1, 0], index=(0 if current_consent['allow_risk_scoring'] == 1 else 1), format_func=lambda x: "Yes" if x == 1 else "No", key="new_allow_risk_scoring")
            
            if st.button("Update Consent (Simulated)"):
                # Simulate the update by modifying the global consent dataframe (will reset on app rerun)
                consent.loc[consent["patient_id"] == consent_patient_id, "allow_research"] = new_allow_research
                consent.loc[consent["patient_id"] == consent_patient_id, "allow_risk_scoring"] = new_allow_risk_scoring
                
                # Log the action (as an Admin write action for the demo)
                log_access("U0001", "WRITE", "consent", "governance", "success")

                st.success(f"Consent for Patient {consent_patient_id} updated successfully.")
                st.rerun() # Rerun to update the display
        else:
             st.error("Consent record not found for this patient.")


    # Audit Trail
    with governance_tab3:
        st.subheader("Audit Trail Log")
        st.markdown("All data access attempts (success or denied) are automatically logged for accountability.")

        # Display the audit log
        if not audit_log.empty:
            st.dataframe(audit_log.sort_values("timestamp", ascending=False).head(20))
            
            col_a1, col_a2 = st.columns(2)
            
            with col_a1:
                fig_purpose, ax_purpose = plt.subplots(figsize=(6, 4))
                audit_log["purpose_of_use"].value_counts().plot(kind="bar", ax=ax_purpose, rot=45)
                ax_purpose.set_title("Events by Purpose of Use")
                st.pyplot(fig_purpose)
                plt.close(fig_purpose)

            with col_a2:
                fig_result, ax_result = plt.subplots(figsize=(6, 4))
                audit_log["result"].value_counts().plot(kind="pie", ax=ax_result, autopct='%1.1f%%', startangle=90)
                ax_result.set_title("Access Outcomes")
                ax_result.set_ylabel('')
                st.pyplot(fig_result)
                plt.close(fig_result)
        else:
            st.info("No audit events logged yet.")