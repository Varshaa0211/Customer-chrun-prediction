# customer_churn_app.py (fixed to use a single pipeline model)
# Streamlit app for Customer Churn Prediction
# - Supports manual single-row input and CSV batch predictions
# - Only requires churn_model.pkl (a Pipeline that includes preprocessing + model)
# - To run: pip install -r requirements.txt  &&  streamlit run customer_churn_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

st.set_page_config(page_title="💼 Customer Churn Prediction", page_icon="📊", layout="wide")

# ---------------------- Load Model ----------------------
st.sidebar.header("Model Upload")
uploaded_model = st.sidebar.file_uploader("Upload churn_model.pkl (Pipeline)", type=["pkl","joblib"])

st.sidebar.info("Model should be a scikit-learn Pipeline that already includes preprocessing (encoder + scaler).\nExample: OneHotEncoder + StandardScaler + LogisticRegression inside Pipeline.")

model = None
if uploaded_model is not None:
    try:
        model = joblib.load(uploaded_model)
        st.sidebar.success("Model loaded successfully ✅")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
else:
    try:
        model = joblib.load('churn_model.pkl')
        st.sidebar.success("Loaded churn_model.pkl from local folder ✅")
    except Exception:
        st.sidebar.warning("No model loaded. Upload `churn_model.pkl`.")

# ---------------------- Main layout ----------------------
st.title("💼 Customer Churn Prediction")
st.write("Use the form on the left for a single customer prediction, or upload a CSV for batch predictions.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Single prediction 🧾")
    with st.form("single_form"):
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
            st.error("No model loaded. Upload `churn_model.pkl` in the sidebar.")
        else:
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
            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(df_row)
                    p_churn = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                    pred_label = (p_churn >= 0.5).astype(int)
                    st.metric(label="Churn probability", value=f"{p_churn[0]:.2%}")
                    st.success(f"Prediction: {'Will Churn (1)' if pred_label[0]==1 else 'Will NOT Churn (0)'}")
                else:
                    pred = model.predict(df_row)
                    st.write("Predicted class:", pred[0])
            except Exception as e:
                st.error(f"Prediction failed: {e}")

            if model is None:
                st.error("No model loaded. Upload `churn_model.pkl`.")
            else:
                try:
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(df)
                        p_churn = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                        preds = (p_churn >= 0.5).astype(int)
                        result_df = df.copy()
                        result_df['churn_probability'] = p_churn
                        result_df['churn_pred'] = preds
                    else:
                        preds = model.predict(df)
                        result_df = df.copy()
                        result_df['churn_pred'] = preds

                    st.success("Batch prediction complete ✅")
                    st.dataframe(result_df.head(50))

                    towrite = io.BytesIO()
                    result_df.to_csv(towrite, index=False)
                    towrite.seek(0)
                    st.download_button("Download predictions (CSV)", data=towrite, file_name="churn_predictions.csv")
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")


