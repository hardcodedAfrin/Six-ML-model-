import streamlit as st
import joblib
import numpy as np
# (0 = Female, 1 = Male)
# -----------------------
# LOAD MODEL
# -----------------------
model = joblib.load("model.pkl")

st.title("🧠 Autism Prediction Web App")

# -----------------------
# TRY TO GET FEATURE INFO
# -----------------------
try:
    feature_names = model.feature_names_in_
except:
    feature_names = None

n_features = model.n_features_in_

st.info(f"This model expects {n_features} features.")

# -----------------------
# INPUT SECTION
# -----------------------
inputs = []

for i in range(n_features):

    if feature_names is not None:
        label = feature_names[i]
    else:
        label = f"Feature {i+1}"

    val = st.number_input(label, value=0.0)
    inputs.append(val)

# -----------------------
# PREDICTION
# -----------------------
if st.button("Predict"):

    input_array = np.array([inputs])
    prediction = model.predict(input_array)[0]

    if prediction == 1:
        st.error("⚠️ Autism Detected")
    else:
        st.success("✅ No Autism Detected")