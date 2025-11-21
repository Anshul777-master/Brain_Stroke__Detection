import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD MODEL & SCALER
# =========================
model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Brain Stroke Prediction Dashboard", layout="wide")

st.title("🧠 Brain Stroke Prediction Dashboard")
st.write("Enter the patient details to predict the risk of stroke.")

# =========================
# USER INPUT FORM
# =========================
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])

with col2:
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children"])
    Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=400.0, value=100.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes"])

# =========================
# PREPROCESSING
# =========================
input_dict = {
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": Residence_type,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_status
}

df_input = pd.DataFrame([input_dict])

# One-hot encoding (same as training)
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df_input = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)

# Ensure same columns as training model
model_cols = scaler.feature_names_in_
df_input = df_input.reindex(columns=model_cols, fill_value=0)

# Scale numerical features
df_input_scaled = scaler.transform(df_input)

# =========================
# PREDICTION
# =========================
if st.button("Predict Stroke Risk"):
    prediction = model.predict(df_input_scaled)[0]
    probability = model.predict_proba(df_input_scaled)[0][1]

    st.subheader("🔍 Prediction Result:")

    if prediction == 1:
        st.error(f"⚠️ **High Stroke Risk Detected! Probability: {probability:.2f}**")
    else:
        st.success(f"✔️ **Low Stroke Risk. Probability: {probability:.2f}**")

    st.progress(float(probability))
    st.write("Probability of Stroke:", probability)
