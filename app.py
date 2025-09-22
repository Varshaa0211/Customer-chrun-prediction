import streamlit as st
import numpy as np
import pickle

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="💼 Customer Churn Prediction",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Load Model, Scaler, and Encoder
# ----------------------------
try:
    with open("churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model or preprocessing files not found! Make sure 'churn_model.pkl', 'scaler.pkl', and 'label_encoder.pkl' are in the same folder as this app.")
    st.stop()

# ----------------------------
# App Title
# ----------------------------
st.title("💼 Customer Churn Prediction App")
st.markdown("""
Predict whether a customer is likely to **churn** or **stay**.  
Enter customer details below and click **Predict** 🚀
""")

# ----------------------------
# User Input Section
# ----------------------------
st.header("🔹 Enter Customer Details")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (Years with Company)", min_value=0, max_value=10, value=3)
balance = st.number_input("Account Balance", min_value=0.0, value=50000.0, step=1000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=5, value=2)
has_cr_card = st.selectbox("Has Credit Card?", ("Yes", "No"))
is_active_member = st.selectbox("Is Active Member?", ("Yes", "No"))
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)
gender = st.selectbox("Gender", ("Male", "Female"))

# ----------------------------
# Preprocess Input
# ----------------------------
try:
    gender_encoded = le.transform([gender])[0]
except:
    st.error("❌ Label Encoder Error! Make sure the encoder matches the training data categories.")
    st.stop()

has_cr_card_encoded = 1 if has_cr_card == "Yes" else 0
is_active_member_encoded = 1 if is_active_member == "Yes" else 0

user_input = np.array([[credit_score, age, tenure, balance,
                        num_of_products, has_cr_card_encoded, 
                        is_active_member_encoded, estimated_salary, gender_encoded]])

user_input_scaled = scaler.transform(user_input)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Churn 🔮"):
    prediction = model.predict(user_input_scaled)[0]
    probability = model.predict_proba(user_input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The customer is likely to **churn**. Probability: {probability:.2f}")
    else:
        st.success(f"✅ The customer is likely to **stay**. Probability: {1-probability:.2f}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
---
Developed by Varsha ❤️ 
""")
