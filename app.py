# customer_churn_app.py
# Streamlit app for Customer Churn Prediction (single form only, no uploads)
# Expects churn_model.pkl (a Pipeline with preprocessing + model) in the same folder

import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="💼 Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
try:
    model = joblib.load("churn_model.pkl")
except Exception as e:
    st.error(f"❌ Could not load churn_model.pkl: {e}")
    st.stop()

# ----------------------------
# Title
# ----------------------------
st.title("💼 Customer Churn Prediction")
st.write("Fill the form below to predict whether the customer will churn.")

# ----------------------------
# Input Form
# ----------------------------
with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Has Partner", ["Yes", "No"])
    Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=1000, value=12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=10000.0, value=70.0, format="%.2f")
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=100000.0, value=800.0, format="%.2f")

    submitted = st.form_submit_button("Predict 🔮")

# ----------------------------
# Prediction
# ----------------------------
if submitted:
    row = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "InternetService": InternetService,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    df = pd.DataFrame([row])

    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)
            p_churn = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            pred_label = (p_churn >= 0.5).astype(int)
            st.metric("Churn probability", f"{p_churn[0]:.2%}")
            st.success(f"Prediction: {'Will Churn (1)' if pred_label[0]==1 else 'Will NOT Churn (0)'}")
        else:
            pred = model.predict(df)
            st.success(f"Prediction: {pred[0]}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
