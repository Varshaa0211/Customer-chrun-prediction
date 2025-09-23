import streamlit as st
import numpy as np
import joblib

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="💼 Customer Churn Prediction",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("churn_model.pkl")
        return model
    except Exception as e:
        st.error(f"❌ Could not load churn_model.pkl: {e}")
        return None

model = load_model()

# ----------------------------
# UI
# ----------------------------
st.title("💼 Customer Churn Prediction App")
st.write("Predict whether a customer will **churn** or **stay** 📊")

if model:
    # Inputs for prediction
    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("🎂 Senior Citizen", [0, 1])
    partner = st.selectbox("💍 Partner", ["Yes", "No"])
    dependents = st.selectbox("👶 Dependents", ["Yes", "No"])
    tenure = st.number_input("📅 Tenure (months)", min_value=0, max_value=100, value=1)
    phone_service = st.selectbox("📞 Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("📡 Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("🔒 Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("💾 Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("📱 Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("🛠️ Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("📺 Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("🎬 Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("📝 Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("📃 Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("💳 Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("💵 Monthly Charges", min_value=0.0, value=50.0)
    total_charges = st.number_input("💰 Total Charges", min_value=0.0, value=100.0)

    # Convert to dictionary for pipeline
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # Prediction
    if st.button("🔮 Predict Churn"):
        try:
            pred = model.predict([list(input_data.values())])[0]
            prob = model.predict_proba([list(input_data.values())])[0][1]

            if pred == 1:
                st.error(f"⚠️ Customer is **likely to churn** (probability: {prob:.2f})")
            else:
                st.success(f"✅ Customer is **not likely to churn** (probability: {prob:.2f})")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.warning("⚠️ Please make sure `churn_model.pkl` exists in this folder.")
