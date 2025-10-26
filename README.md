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

1. Prerequisites

You must create the following directory structure in your project root:

Note: The application automatically generates synthetic data files and model artifacts upon the first run, ensuring the platform is ready to use immediately.

.
├── app.py
├── requirements.txt
├── data/
│   ├── patients.csv  (Synthetic data is generated into this)
│   ├── outcomes.csv  (directory by app.py)
│   └── consent.csv   (etc.)
├── models/
│   └── *.pkl, *.joblib (Trained models will be saved here)
└── governance/
    └── audit_log.csv (Access logs are stored here)


