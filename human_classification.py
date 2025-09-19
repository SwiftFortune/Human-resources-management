# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Set page config FIRST
# -------------------------------
st.set_page_config(page_title="Human Resource Classification", layout="wide")

# -------------------------------
# Load saved model, scaler, label encoder
# -------------------------------
@st.cache_data
def load_model():
    model = joblib.load("naive_bayes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_model()

# -------------------------------
# Load and clean data
# -------------------------------
@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data.columns = [col.strip().replace("`", "") for col in data.columns]
    return data

data = load_data(r"C:\sachin\Python\Human Resource Management\output.csv")

# -------------------------------
# Prepare numeric columns
# -------------------------------
numeric_cols = ["Division", "Group", "Class",
                "Marginal Workers - Total - Persons",
                "Marginal Workers - Total - Males",
                "Marginal Workers - Total - Females"]
numeric_cols = [col for col in numeric_cols if col in data.columns]

for col in numeric_cols:
    data[col] = data[col].astype(str).str.replace("`", "").str.extract("(\d+)")[0].astype(int)

# -------------------------------
# Encode India/States for model
# -------------------------------
if data["India/States"].dtype == object:
    le_states = LabelEncoder()
    data["India/States_encoded"] = le_states.fit_transform(data["India/States"])
    state_options = data["India/States"].unique()
else:
    le_states = None
    data["India/States_encoded"] = data["India/States"]
    state_options = data["India/States"].unique()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("👨‍💼 Human Resource Classification App")
st.write("Enter feature values below to predict the **NIC Name (Industry/Activity Type)**")

# Function to safely get numeric input with limits
def number_input_feature(column_name):
    if column_name in data.columns:
        min_val = int(data[column_name].min())
        max_val = int(data[column_name].max())
        default_val = min_val
        return st.number_input(f"{column_name} (min {min_val} - max {max_val})",
                               min_value=min_val, max_value=max_val, value=default_val, step=1)
    else:
        return 0

# -------------------------------
# Input fields
# -------------------------------
with st.form(key="input_form"):
    division = number_input_feature("Division")
    group = number_input_feature("Group")
    class_col = number_input_feature("Class")
    state_name = st.selectbox("India/States", options=state_options)

    if le_states is not None:
        india_states_encoded = le_states.transform([state_name])[0]
    else:
        india_states_encoded = state_name

    marginal_total = number_input_feature("Marginal Workers - Total - Persons")
    marginal_males = number_input_feature("Marginal Workers - Total - Males")
    marginal_females = number_input_feature("Marginal Workers - Total - Females")

    # -------------------------------
    # Prediction button inside form
    # -------------------------------
    submit_button = st.form_submit_button(label="🔍 Predict Industry")

# -------------------------------
# Prediction code (runs only after form submission)
# -------------------------------
if submit_button:
    features = np.array([[division, group, class_col, india_states_encoded,
                          marginal_total, marginal_males, marginal_females]])
    
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]
    
    st.success(f"✅ Predicted NIC Name (Industry): **{predicted_label}**")
