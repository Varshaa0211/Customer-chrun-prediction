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
# Load Model, Encoders, Scaler
# ----------------------------
@st.cache_data
def load_artifacts():
    model = joblib.load("churn_model.pkl")              # Trained ML model
    label_encoders = joblib.load("label_encoders.pkl")  # Dict of label encoders
    scaler = joblib.load("scaler.pkl")                  # Scaler
    # 🔑 Also load feature names used in training
    feature_names = joblib.load("feature_names.pkl")
    return model, label_encoders, scaler, feature_names

model, label_encoders, scaler, feature_names = load_artifacts()

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
# Align with Training Features
# ----------------------------
# Add missing features with default values
for col in feature_names:
    if col not in input_df.columns:
        if col.lower() == "customerid":
            input_df[col] = "0000"   # Dummy ID
        elif col in ['SeniorCitizen']:
            input_df[col] = 0        # Default numeric
        else:
            input_df[col] = "No"     # Default categorical

# Reorder to match training dataset
input_df = input_df[feature_names]

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_input(df, label_encoders, scaler):
    df_processed = df.copy()

    # Encode categorical features
    for col, le in label_encoders.items():
        if col in df_processed.columns:
            try:
                df_processed[col] = le.transform(df_processed[col])
            except ValueError:
                df_processed[col] = df_processed[col].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

    # Scale
    df_processed = scaler.transform(df_processed)
    return df_processed

# ----------------------------
# Prediction
# ----------------------------
st.subheader("Prediction 🔮")

try:
    processed_input = preprocess_input(input_df, label_encoders, scaler)
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)

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
