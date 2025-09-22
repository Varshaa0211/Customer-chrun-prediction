import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="📊 Customer Churn Prediction",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_data
def load_model():
    model = joblib.load('churn_model.pkl')
    return model

model = load_model()

# ----------------------------
# App Title
# ----------------------------
st.markdown("<h1 style='text-align: center; color: #4B0082;'>💼 Customer Churn Predictor 💼</h1>", unsafe_allow_html=True)
st.write("Predict if a customer is likely to churn or stay! 📈")

# ----------------------------
# User Input
# ----------------------------
st.sidebar.header("Customer Input Features 📝")

def user_input_features():
    tenure = st.sidebar.number_input("Tenure (Months) ⏳", min_value=0, max_value=100, value=12)
    monthly_charges = st.sidebar.number_input("Monthly Charges 💰", min_value=0.0, max_value=1000.0, value=70.0)
    total_charges = st.sidebar.number_input("Total Charges 💵", min_value=0.0, max_value=10000.0, value=1500.0)
    contract = st.sidebar.selectbox("Contract Type 📄", ("Month-to-month", "One year", "Two year"))
    internet_service = st.sidebar.selectbox("Internet Service 🌐", ("DSL", "Fiber optic", "No"))
    payment_method = st.sidebar.selectbox("Payment Method 💳", ("Electronic check", "Mailed check", "Bank transfer", "Credit card"))
    
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet_service,
        'PaymentMethod': payment_method
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# ----------------------------
# Prediction
# ----------------------------
st.subheader("Prediction 🔮")

try:
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.error(f"The customer is likely to churn 😢 (Probability: {prediction_proba[0][1]*100:.2f}%)")
    else:
        st.success(f"The customer is likely to stay 😀 (Probability: {prediction_proba[0][0]*100:.2f}%)")
except Exception as e:
    st.warning(f"⚠️ Prediction failed: {e}")

# ----------------------------
# Show Input Data
# ----------------------------
st.subheader("Customer Input Features 🧾")
st.write(input_df)
