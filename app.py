import streamlit as st
import pandas as pd
import joblib

# --------------------------------
# Load Model & Columns
# --------------------------------
model = joblib.load("churn_model.joblib")
model_columns = joblib.load("model_columns.joblib")

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(
    page_title="📊 Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# --------------------------------
# Custom CSS for Styling
# --------------------------------
st.markdown(
    """
    <style>
    .main {background-color: #f9fafc;}
    .stButton>button {
        background: linear-gradient(90deg, #4b6cb7, #182848);
        color:white;
        border-radius: 12px;
        font-size:16px;
        padding:10px 24px;
    }
    .prediction-card {
        padding:20px;
        border-radius:15px;
        background-color:white;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------
# Title Section
# --------------------------------
st.title("📊 Customer Churn Prediction App")
st.markdown(
    """
    Welcome to the *Churn Prediction Dashboard*.  
    Enter customer details in the sidebar to check whether they are likely to *Churn* or *Stay*.  
    """
)

# --------------------------------
# Sidebar Input
# --------------------------------
st.sidebar.header("📝 Enter Customer Details")

def user_input():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
    Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    MonthlyCharges = st.sidebar.slider("Monthly Charges", 0, 150, 70)
    TotalCharges = st.sidebar.slider("Total Charges", 0, 9000, 2500)

    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }
    return pd.DataFrame([data])

input_df = user_input()

# --------------------------------
# Preprocessing
# --------------------------------
df = pd.get_dummies(input_df)
df = df.reindex(columns=model_columns, fill_value=0)

# --------------------------------
# Prediction
# --------------------------------
prediction = model.predict(df)[0]
prediction_proba = model.predict_proba(df)[0]

# --------------------------------
# Result Display
# --------------------------------
st.subheader("🔮 Prediction Result")

with st.container():
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    if prediction == 1:
        st.error("⚠ *High Risk:* Customer is likely to Churn")
    else:
        st.success("✅ *Safe:* Customer is likely to Stay")

    st.write(f"*Confidence Score:* {round(max(prediction_proba)*100, 2)} %")
    st.progress(int(max(prediction_proba)*100))
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.markdown("🚀 Built with *Machine Learning & Streamlit* | Customer Churn Prediction Dashboard")
