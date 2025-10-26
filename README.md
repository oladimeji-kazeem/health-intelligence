# health-intelligence
A solution portfolio for the UK NHS with synthetic data simulated to replicate the possible datasets that could be generated from the health sector.

### 🏥 AI-Ready Health Data Platform Showcase

This project demonstrates a production-ready data and analytics platform for healthcare, built using Streamlit and Python. The application showcases key capabilities required for ethical and functional health intelligence systems: data interoperability (FHIR-like schema), secure access (RBAC and consent checks), predictive modeling (Readmission and Condition Risk), and explainable AI (SHAP).

### ✨ Key Features

- Data Unification: Simulates raw EHR data transformation into a canonical, FHIR-like schema.

- Secure Access & Governance: Implements Role-Based Access Control (RBAC), enforces Patient Consent for specific uses (Research, Risk Scoring), and maintains a comprehensive Audit Trail.

- Predictive Analytics: Features a trained Logistic Regression model for 30-day Readmission Risk and a Random Forest model for Condition Detection.

- Explainable AI (XAI): Uses SHAP values to provide local and global explanations for model predictions, ensuring transparency and trust.

- Historical Dashboard: Provides visualizations of patient demographics, encounter volume, and outcome trends.

### 🚀 Local Setup and Deployment

This application requires Python (3.9+) and specific file organization for data handling and model persistence.

### 1\. Prerequisites

You must create the following directory structure in your project root:

> Note: The application automatically generates synthetic data files and model artifacts upon the first run, ensuring the platform is ready to use immediately.


### 2\. Installation

1.  git clone \[YOUR\_REPO\_URL\]cd health-intelligence
    
2.  python -m venv venvsource venv/bin/activate # On Windows: venv\\Scripts\\activate
    
3.  pip install -r requirements.txt
    

### 3\. Run the App

Execute the Streamlit application. This step will automatically generate the synthetic data files (if missing) and train/load the ML models into the /models directory.
