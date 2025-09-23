import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="💼 Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
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
# Title
# ----------------------------
st.title("💼 Customer Churn Prediction 🚀")
st.write("Fill out the form below to check if a customer is likely to churn or stay. 🧾")

if model:
    # ----------------------------
    # Input Form
    # ----------------------------
    st.subheader("Customer Information ✨")
    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("🎂 Senior Citizen", [0, 1])
    partner = st.selectbox("💍 Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("👶 Has Dependents?", ["Yes", "No"])
    tenure = st.number_input("📅 Tenure (months)", min_value=0, max_value=100, value=12)
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
    monthly_charges = st.number_input("💵 Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("💰 Total Charges", min_value=0.0, value=800.0)

    # ----------------------------
    # Prepare input
    # ----------------------------
    input_dict = {
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
    }

    input_df = pd.DataFrame(input_dict)

    # ----------------------------
    # Prediction
    # ----------------------------
    if st.button("🔮 Predict Churn"):
        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.error(f"⚠️ Customer is **likely to churn** (probability: {probability:.2f})")
            else:
                st.success(f"✅ Customer is **not likely to churn** (probability: {probability:.2f})")

        except Exception as e:
            st.error(f"Prediction failed ❌: {e}")

else:
    st.warning("⚠️ Please make sure `churn_model.pkl` exists in this folder.")
