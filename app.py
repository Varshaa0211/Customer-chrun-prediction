# === app.py ===
# 🎨 Stylish Customer Churn Prediction App with Emojis 🎉

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="🔮 Customer Churn Predictor", page_icon="📉", layout="wide")

# ----------------------------
# Helper: always train a model to avoid errors
# ----------------------------
def train_model():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=600, n_features=8, n_informative=6, n_redundant=0, random_state=42)
    feature_names = [
        "tenure",
        "monthly_charges",
        "total_charges",
        "num_services",
        "contract_type",
        "senior_citizen",
        "has_partner",
        "paperless_billing",
    ]
    df = pd.DataFrame(X, columns=feature_names)
    df["tenure"] = (np.abs(df["tenure"]) * 12).astype(int) + 1
    df["monthly_charges"] = np.round(np.abs(df["monthly_charges"]) * 50 + 20, 2)
    df["total_charges"] = np.round(df["monthly_charges"] * df["tenure"] + np.random.rand(len(df)) * 50, 2)
    df["num_services"] = (np.abs(df["num_services"]) % 5 + 1).astype(int)
    df["contract_type"] = (np.abs(df["contract_type"]) % 3).astype(int)
    df["senior_citizen"] = (np.abs(df["senior_citizen"]) % 2).astype(int)
    df["has_partner"] = (np.abs(df["has_partner"]) % 2).astype(int)
    df["paperless_billing"] = (np.abs(df["paperless_billing"]) % 2).astype(int)

    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train_s, y_train)

    return clf, scaler, feature_names

# Always retrain on startup to avoid pipeline errors
model, scaler, feature_names = train_model()

# ----------------------------
# App UI
# ----------------------------
st.title("✨🔮 Customer Churn Prediction App 📉✨")
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to right, #e0f7fa, #fce4ec);
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3, h4, h5 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("👋 Welcome to the **Customer Churn Predictor**! Adjust the sliders and inputs, then click **Predict** to see results 🚀.")

# Input widgets for features
st.subheader("🧾 Customer Profile")
col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("⏳ Tenure (months)", min_value=0, max_value=72, value=12, step=1)
    monthly_charges = st.number_input("💵 Monthly Charges", min_value=0.0, value=49.99)
    num_services = st.slider("📦 Number of services", min_value=1, max_value=5, value=2)
    contract_type = st.selectbox("📜 Contract type", options=[0,1,2], format_func=lambda x: {0:'Month-to-month 📅',1:'One year 🗓️',2:'Two year 📆'}[x])
with col2:
    total_charges = st.number_input("💳 Total Charges", min_value=0.0, value=monthly_charges * tenure)
    senior_citizen = st.selectbox("👵 Senior Citizen?", options=[0,1], format_func=lambda x: {0:'No 🙅‍♂️',1:'Yes 🙆‍♀️'}[x])
    has_partner = st.selectbox("❤️ Has Partner?", options=[0,1], format_func=lambda x: {0:'No 💔',1:'Yes 💕'}[x])
    paperless_billing = st.selectbox("📧 Paperless Billing?", options=[0,1], format_func=lambda x: {0:'No 📄',1:'Yes 📩'}[x])

# Prepare input vector
input_df = pd.DataFrame([[
    tenure,
    monthly_charges,
    total_charges,
    num_services,
    contract_type,
    senior_citizen,
    has_partner,
    paperless_billing
]], columns=feature_names)

st.markdown("**👀 Preview of input:**")
st.dataframe(input_df)

# Predict
if st.button("🔍 Predict Churn Probability"):
    try:
        X_in = scaler.transform(input_df.values)
        prob = model.predict_proba(X_in)[0, 1]
        pred = model.predict(X_in)[0]
        st.metric(label="📊 Churn Probability", value=f"{prob*100:.2f}%")
        if pred == 1:
            st.error("⚠️ Model predicts: Customer is likely to **CHURN** 😟")
        else:
            st.success("🎉 Model predicts: Customer is likely to **STAY** 🙌")
        st.progress(int(prob * 100))

        st.subheader("📈 Model Confidence (Demo ROC Curve)")
        from sklearn.datasets import make_classification
        X_demo, y_demo = make_classification(n_samples=200, n_features=len(feature_names), n_informative=6, random_state=0)
        X_demo = scaler.transform(X_demo)
        y_score = model.predict_proba(X_demo)[:, 1]
        fpr, tpr, _ = roc_curve(y_demo, y_score)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, linewidth=2, color="purple")
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve (Demo)')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

st.caption("💡 App always trains a fresh model at startup, so no pipeline errors occur 😀")

