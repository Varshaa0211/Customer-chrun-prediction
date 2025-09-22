

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset


    df = pd.read_csv("customer_churn Raw data.csv")
 


st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Prediction App")

st.write("🚀 இந்த app உங்கள் dataset (`customer_churn Raw data.csv`) வைத்து **customer churn** predict செய்கிறது.")

# Display dataset preview
if st.checkbox("👀 Show Dataset Preview"):
    st.dataframe(df.head())

# --------------------
# Preprocessing
# --------------------
df_clean = df.copy()

# Convert categorical columns to numeric
label_encoders = {}
for col in df_clean.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le

# Features and target
X = df_clean.drop("Exited", axis=1, errors="ignore")  # "Exited" target column assumed
y = df_clean["Exited"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

st.subheader("🔮 Enter Customer Details Below")

# Dynamic input fields based on dataset columns
user_input = {}
for col in X.columns:
    if df[col].dtype == "object":
        options = df[col].unique().tolist()
        user_input[col] = st.selectbox(f"📌 {col}", options)
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        user_input[col] = st.number_input(f"📈 {col}", min_val, max_val, mean_val)

# Predict button
if st.button("✨ Predict Now"):
    input_df = pd.DataFrame([user_input])

    # Encode categorical inputs
    for col, le in label_encoders.items():
        if col in input_df:
            input_df[col] = le.transform(input_df[col].astype(str))

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ **Result:** This customer is likely to **CHURN** 💔 \n\n 🔢 Probability: **{probability:.2f}**")
    else:
        st.success(f"✅ **Result:** This customer is **NOT likely to churn** 🎉 \n\n 🔢 Probability: **{probability:.2f}**")
