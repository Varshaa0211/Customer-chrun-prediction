# === app.py ===
# Customer Churn Prediction Streamlit App
# Includes fallback training if pretrained model files are not present.
# Emojis added throughout for a friendly UI 😊📊

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="🔮 Customer Churn Predictor", page_icon="📉", layout="centered")

MODEL_PATH = "churn_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "features.pkl"

# ----------------------------
# Helper: fallback training (only runs if model is missing)
# ----------------------------
def train_fallback_model():
    """Train a small synthetic model so the app runs without external files."""
    st.info("No pretrained model found — training a small fallback model locally (fast) ⚙️")
    # create synthetic dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=600, n_features=8, n_informative=6, n_redundant=0, random_state=42)
    feature_names = [
        "tenure",         # months with company
        "monthly_charges",
        "total_charges",
        "num_services",
        "contract_type",  # encoded numerically
        "senior_citizen",
        "has_partner",
        "paperless_billing",
    ]
    df = pd.DataFrame(X, columns=feature_names)
    # make some features more interpretable
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

    # save
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_names, FEATURES_PATH)

    # metrics
    preds = clf.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
    st.success(f"Fallback model trained — accuracy: {acc:.3f}, ROC AUC: {auc:.3f} ✅")
    return clf, scaler, feature_names

# ----------------------------
# Load or train model
# ----------------------------
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURES_PATH)
    except Exception as e:
        st.warning(f"Model files exist but failed to load: {e} — retraining fallback model 🔁")
        model, scaler, feature_names = train_fallback_model()
else:
    model, scaler, feature_names = train_fallback_model()

# ----------------------------
# App UI
# ----------------------------
st.title("🔮 Customer Churn Prediction")
st.markdown("Welcome! Adjust customer attributes and click **Predict** to see the churn probability. 💡")

with st.sidebar:
    st.header("⚙️ Model options")
    uploaded_model = st.file_uploader("Upload a pretrained model (.pkl) if you have one", type=["pkl"])
    if uploaded_model is not None:
        try:
            new_model = joblib.load(uploaded_model)
            joblib.dump(new_model, MODEL_PATH)
            model = new_model
            st.success("Model uploaded and saved successfully ✅")
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")
    st.markdown("---")
    st.write("Made with ❤️ — tweak inputs and experiment!")

# Input widgets for features
st.subheader("Customer profile — input features 🧾")
col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=49.99)
    num_services = st.slider("Number of services", min_value=1, max_value=5, value=2)
    contract_type = st.selectbox("Contract type", options=[0,1,2], format_func=lambda x: {0:'Month-to-month',1:'One year',2:'Two year'}[x])
with col2:
    total_charges = st.number_input("Total Charges", min_value=0.0, value=monthly_charges * tenure)
    senior_citizen = st.selectbox("Senior Citizen?", options=[0,1], format_func=lambda x: {0:'No',1:'Yes'}[x])
    has_partner = st.selectbox("Has Partner?", options=[0,1], format_func=lambda x: {0:'No',1:'Yes'}[x])
    paperless_billing = st.selectbox("Paperless Billing?", options=[0,1], format_func=lambda x: {0:'No',1:'Yes'}[x])

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

st.markdown("**Preview of input:**")
st.dataframe(input_df)

# Predict
if st.button("Predict churn probability 🔍"):
    try:
        if scaler is None or model is None:
            st.error("Model or scaler not loaded properly. Please retrain or reload.")
        else:
            X_in = scaler.transform(input_df.values)
            prob = model.predict_proba(X_in)[0, 1]
            pred = model.predict(X_in)[0]
            st.metric(label="Churn probability", value=f"{prob*100:.2f}%", delta=None)
            if pred == 1:
                st.warning("Model predicts: Customer is likely to CHURN 😟")
            else:
                st.success("Model predicts: Customer is likely to STAY 🎉")

            st.progress(int(prob * 100))

            if hasattr(model, "predict_proba"):
                st.subheader("Model confidence & diagnostics")
                from sklearn.datasets import make_classification
                X_demo, y_demo = make_classification(n_samples=200, n_features=len(feature_names), n_informative=6, random_state=0)
                X_demo = scaler.transform(X_demo)
                y_score = model.predict_proba(X_demo)[:, 1]
                fpr, tpr, _ = roc_curve(y_demo, y_score)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, linewidth=2)
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC curve (demo)')
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Option to show sample dataset (if trained fallback)
st.markdown("---")
if st.checkbox("Show sample data used for fallback training (if available)"):
    try:
        from sklearn.datasets import make_classification
        Xs, ys = make_classification(n_samples=50, n_features=len(feature_names), n_informative=6, random_state=1)
        df_sample = pd.DataFrame(Xs, columns=feature_names)
        df_sample["churn_label_demo"] = ys
        st.dataframe(df_sample)
    except Exception:
        st.info("No sample data available to display.")

st.markdown("---")
