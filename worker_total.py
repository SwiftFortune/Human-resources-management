# simple_worker_predict_app.py
import streamlit as st
import numpy as np
import joblib

st.title("👷 Total Worker Prediction")

# Load model and scaler
MODEL_PATH = "total_worker_model.pkl"
SCALER_PATH = "total_worker_scaler.pkl"
STATES_CLASSES_PATH = "india_states_classes.npy"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    states_classes = np.load(STATES_CLASSES_PATH, allow_pickle=True)
    st.success("Model, scaler, and states loaded successfully.")
except:
    st.error("Model, scaler, or state classes not found. Please train or place files in folder.")
    st.stop()

# Input form
st.subheader("Enter feature values")
state_code = st.number_input("State Code", value=1)
district_code = st.number_input("District Code", value=1)
selected_state = st.selectbox("India/States", options=states_classes)
selected_state_encoded = int(np.where(states_classes == selected_state)[0][0])
division = st.number_input("Division", value=1)
group = st.number_input("Group", value=1)
class_col = st.number_input("Class", value=1)
marginal_total = st.number_input("Marginal Workers - Total - Persons", value=0)

if st.button("🔮 Predict Total Workers"):
    features = np.array([[state_code, district_code, selected_state_encoded,
                          division, group, class_col, marginal_total]])
    try:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        st.success(f"✅ Predicted Main Workers - Total - Persons: {int(round(prediction))}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
