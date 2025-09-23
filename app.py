# customer_churn_app.py
# Streamlit app for Customer Churn Prediction
# - Supports manual single-row input and CSV batch predictions
# - Looks for model files: churn_model.pkl (required). Optional: scaler.pkl, encoder.pkl
# - If you don't have pickle files, you can upload them from the sidebar.
# - To run: pip install -r requirements.txt  &&  streamlit run customer_churn_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from typing import Optional

st.set_page_config(page_title="💼 Customer Churn Prediction", page_icon="📊", layout="wide")

# ---------------------- Utility functions ----------------------

def load_pickle_safe(path_or_buffer) -> Optional[object]:
    try:
        return joblib.load(path_or_buffer)
    except Exception as e:
        st.warning(f"Could not load pickle: {e}")
        return None


def get_default_feature_order():
    # A reasonable default feature order. Update to match your training pipeline.
    return [
        'gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines',
        'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
        'StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
        'MonthlyCharges','TotalCharges'
    ]


def prepare_features(df: pd.DataFrame, encoder=None, scaler=None, columns_order=None):
    """
    Prepares features to feed to the model. If encoder/scaler are provided they are applied.
    If columns_order is provided, the df will be reindexed to that order (missing cols -> filled with 0/NaN).
    """
    if columns_order is not None:
        for c in columns_order:
            if c not in df.columns:
                df[c] = np.nan
        df = df[columns_order]

    # If encoder exists and is a fitted transformer, try to use it
    if encoder is not None:
        try:
            X = encoder.transform(df)
        except Exception as e:
            st.warning(f"Encoder transform failed: {e} — passing raw df to model")
            X = df.values
    else:
        X = df.values

    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            st.warning(f"Scaler transform failed: {e} — returning pre-scaled features")

    return X


# ---------------------- Sidebar: model files & guidance ----------------------
st.sidebar.header("Model / Files")
uploaded_model = st.sidebar.file_uploader("Upload churn_model.pkl (or .joblib)", type=["pkl","joblib"], help="If left blank, app will try to load from local path 'churn_model.pkl'.")
uploaded_scaler = st.sidebar.file_uploader("(Optional) Upload scaler.pkl", type=["pkl","joblib"]) 
uploaded_encoder = st.sidebar.file_uploader("(Optional) Upload encoder.pkl", type=["pkl","joblib"]) 

st.sidebar.markdown("---")
st.sidebar.info("Model expectations: `churn_model.pkl` should be a scikit-learn estimator with a .predict_proba() or .predict() method. If you used a pipeline that includes preprocessing, you may not need encoder/scaler.")

# Attempt to load from uploaded files first, else local files
model = None
scaler = None
encoder = None

if uploaded_model is not None:
    model = load_pickle_safe(uploaded_model)
else:
    try:
        model = joblib.load('churn_model.pkl')
    except Exception:
        model = None

if uploaded_scaler is not None:
    scaler = load_pickle_safe(uploaded_scaler)
else:
    try:
        scaler = joblib.load('scaler.pkl')
    except Exception:
        scaler = None

if uploaded_encoder is not None:
    encoder = load_pickle_safe(uploaded_encoder)
else:
    try:
        encoder = joblib.load('encoder.pkl')
    except Exception:
        encoder = None

# ---------------------- Main layout ----------------------
st.title("💼 Customer Churn Prediction")
st.write("Use the form on the left for a single customer prediction, or upload a CSV for batch predictions.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Single prediction 🧾")
    with st.form("single_form"):
        # Build inputs with reasonable defaults
        gender = st.selectbox("Gender", options=["Female","Male"], index=0)
        SeniorCitizen = st.selectbox("Senior Citizen?", options=[0,1], index=0)
        Partner = st.selectbox("Has Partner?", options=["Yes","No"], index=1)
        Dependents = st.selectbox("Has Dependents?", options=["Yes","No"], index=1)
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=1000, value=12)
        PhoneService = st.selectbox("Phone Service", options=["Yes","No"], index=0)
        InternetService = st.selectbox("Internet Service", options=["DSL","Fiber optic","No"], index=0)
        Contract = st.selectbox("Contract", options=["Month-to-month","One year","Two year"], index=0)
        PaperlessBilling = st.selectbox("Paperless Billing", options=["Yes","No"], index=0)
        PaymentMethod = st.selectbox("Payment Method", options=["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"], index=0)
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=10000.0, value=70.0, format="%.2f")
        TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=100000.0, value=800.0, format="%.2f")
        submitted = st.form_submit_button("Predict 🔮")

    if submitted:
        if model is None:
            st.error("No model loaded. Upload `churn_model.pkl` in the sidebar or place it in the app folder.")
        else:
            # assemble a dataframe from inputs
            row = {
                'gender': gender,
                'SeniorCitizen': SeniorCitizen,
                'Partner': Partner,
                'Dependents': Dependents,
                'tenure': tenure,
                'PhoneService': PhoneService,
                'InternetService': InternetService,
                'Contract': Contract,
                'PaperlessBilling': PaperlessBilling,
                'PaymentMethod': PaymentMethod,
                'MonthlyCharges': MonthlyCharges,
                'TotalCharges': TotalCharges
            }
            df_row = pd.DataFrame([row])
            columns_order = get_default_feature_order()
            X = prepare_features(df_row, encoder=encoder, scaler=scaler, columns_order=columns_order)

            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)
                    # take probability of positive class (churn=1). If shape mismatch, try index -1
                    p_churn = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                    pred_label = (p_churn >= 0.5).astype(int)
                    st.metric(label="Churn probability", value=f"{p_churn[0]:.2%}")
                    st.success(f"Prediction: {'Will Churn (1)' if pred_label[0]==1 else 'Will NOT Churn (0)'}")
                else:
                    pred = model.predict(X)
                    st.write("Predicted class:", pred[0])
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with col2:
    st.subheader("Batch prediction (CSV upload) 📁")
    st.write("Upload a CSV with the same features used for training. We will try to align columns automatically.")
    uploaded_csv = st.file_uploader("Upload CSV for batch predictions", type=["csv"]) 

    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            st.write(f"Uploaded {df.shape[0]} rows and {df.shape[1]} columns.")
            st.dataframe(df.head())

            columns_order = get_default_feature_order()
            X = prepare_features(df.copy(), encoder=encoder, scaler=scaler, columns_order=columns_order)

            if model is None:
                st.error("No model loaded. Upload `churn_model.pkl` in the sidebar or place it in the app folder.")
            else:
                try:
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(X)
                        p_churn = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                        preds = (p_churn >= 0.5).astype(int)
                        result_df = df.copy()
                        result_df['churn_probability'] = p_churn
                        result_df['churn_pred'] = preds
                    else:
                        preds = model.predict(X)
                        result_df = df.copy()
                        result_df['churn_pred'] = preds

                    st.success("Batch prediction complete ✅")
                    st.dataframe(result_df.head(50))

                    # allow user to download results
                    towrite = io.BytesIO()
                    result_df.to_csv(towrite, index=False)
                    towrite.seek(0)
                    st.download_button("Download predictions (CSV)", data=towrite, file_name="churn_predictions.csv")
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

        except Exception as e:
            st.error(f"Failed to read CSV: {e}")


# ---------------------- Helpful tips and expected files ----------------------
st.markdown("---")
st.subheader("Tips & troubleshooting 🛠️")
st.markdown(
    """
- Make sure `churn_model.pkl` is a trained scikit-learn model or a Pipeline that accepts raw DataFrame values.
- If your model expects numerical arrays, provide `scaler.pkl` and `encoder.pkl` (e.g. a fitted ColumnTransformer) in the sidebar.
- If you get shape or dtype errors, check the feature order and column names. Update `get_default_feature_order()` in this file to match your training set.
- To save a model in training script: `joblib.dump(model, 'churn_model.pkl')` and similarly for scaler/encoder.
"""
)

# ---------------------- End of app ----------------------


# ---------------------- requirements.txt (below) ----------------------
# Save the following lines into requirements.txt
# streamlit
# pandas
# numpy
# scikit-learn
# joblib

# Example pinned versions (optional):
# streamlit==1.25.0
# pandas==2.1.0
# numpy==1.26.0
# scikit-learn==1.3.2
# joblib==1.3.2
