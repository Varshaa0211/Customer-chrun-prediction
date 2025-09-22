# ==============================
# Customer Churn Prediction App - Streamlit
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="🚢 Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Load Model, Scaler & Encoders
# ----------------------------
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
le_dict = joblib.load('label_encoders.pkl')

# ----------------------------
# App Title
# ----------------------------
st.markdown("## 📈 Customer Churn Prediction App")
st.markdown("Predict whether a customer will churn or stay based on their details!")

# ----------------------------
# Input Form
# ----------------------------
st.markdown("### 📝 Enter Customer Details:")
with st.form(key='churn_form'):
    # Example input fields - replace/add columns according to your dataset
    tenure = st.number_input("Tenure (Months) 📅", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges 💰", min_value=0.0, max_value=1000.0, value=70.0)
    total_charges = st.number_input("Total Charges 💵", min_value=0.0, max_value=100000.0, value=1500.0)
    
    # Example categorical fields
    gender = st.selectbox("Gender 👩‍🦰/👨‍🦱", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen 👵👴", ["No", "Yes"])
    partner = st.selectbox("Partner 👩‍❤️‍👨", ["No", "Yes"])
    
    submit_button = st.form_submit_button(label='Predict 🚀')

if submit_button:
    # ----------------------------
    # Prepare Input for Prediction
    # ----------------------------
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    for col, le in le_dict.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])
    
    # Scale numerical features
    input_df_scaled = scaler.transform(input_df)
    
    # ----------------------------
    # Prediction
    # ----------------------------
    prediction = model.predict(input_df_scaled)
    
    if prediction[0] == 1:
        st.markdown("❌ **Customer is likely to Churn!**")
    else:
        st.markdown("✅ **Customer is likely to Stay!**")
