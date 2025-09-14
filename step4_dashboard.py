import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# Load scaler and model
scaler = joblib.load('./artifacts/scaler.pkl')
model = joblib.load('./artifacts/XGB_model.pkl')

st.title("💳 Credit Card Fraud Detection Dashboard")

st.markdown("""
Enter the transaction details below to predict if it is **Fraudulent** or **Normal**.
""")

# User input
V_features = [f"V{i}" for i in range(1,29)]
user_input = {}
for feat in V_features:
    user_input[feat] = st.number_input(f"{feat}", value=0.0, format="%.5f")

user_input['Amount'] = st.number_input("Amount", value=0.0, format="%.2f")

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Scale numerical features
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0][1]
    
    if pred == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! Probability: {pred_proba:.4f}")
    else:
        st.success(f"✅ Normal Transaction. Probability of Fraud: {pred_proba:.4f}")
