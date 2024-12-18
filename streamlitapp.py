# Streamlit Web App with Model Integration

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import joblib

# Load the trained model
model = joblib.load("best_model.joblib")  # Replace with the actual model filename

def preprocess_input(data):
    # Data preprocessing
    data["job"] = data["job"].replace(
        {"blue-collar": 0, "services": 1, "admin.": 2, "entrepreneur": 3, "self-employed": 4, "technician": 5,
         "management": 6, "student": 7, "retired": 8, "housemaid": 9, "unemployed": 10})
    data["marital"] = data["marital"].replace({"married": 1, "single": 2, "divorced": 3, "unknown": 0})
    data["education"] = data["education"].replace(
        {"basic.9y": 1, "high.school": 2, "university.degree": 3, "professional.course": 4, "basic.6y": 5,
         "basic.4y": 6, "illiterate": 7, "unknown": 0})
    data["default"] = data["default"].replace({"no": 1, "yes": 2, "unknown": 0})
    data["housing"] = data["housing"].replace({"no": 1, "yes": 2, "unknown": 0})
    data["loan"] = data["loan"].replace({"no": 1, "yes": 2, "unknown": 0})
    data["contact"] = data["contact"].replace({"cellular": 1, "telephone": 2, "unknown": 0})
    data["month"] = data["month"].replace(
        {"may": 1, "jun": 2, "nov": 3, "sep": 4, "jul": 5, "aug": 6, "mar": 7, "oct": 8, "apr": 9, "dec": 10})
    data["day_of_week"] = data["day_of_week"].replace({"fri": 1, "wed": 2, "mon": 3, "thu": 4, "tue": 5})
    data["poutcome"] = data["poutcome"].replace({"nonexistent": 1, "failure": 2, "success": 3})

    # Replace "unknown" values with NaN
    data.replace("unknown", np.nan, inplace=True)

    # Convert columns to numeric and impute missing values
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
        default_value = 0
        data[column].fillna(default_value, inplace=True)
        data[column] = data[column].astype(int)

    # Scale numerical features
    scaler = MinMaxScaler()
    numerical_features = ['duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.conf.idx', 'euribor3m']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data

def make_prediction(model, input_data):
    # Make predictions
    prediction = model.predict(input_data)
    return prediction

# Streamlit app
st.title("Bank Marketing Prediction App")

# User inputs
job = st.selectbox("Job", ["blue-collar", "services", "admin.", "entrepreneur", "self-employed", "technician",
                           "management", "student", "retired", "housemaid", "unemployed"])
marital_status = st.selectbox("Marital Status", ["married", "single", "divorced", "unknown"])
education = st.selectbox("Education", ["basic.9y", "high.school", "university.degree", "professional.course",
                                       "basic.6y", "basic.4y", "illiterate", "unknown"])
default = st.selectbox("Default", ["no", "yes", "unknown"])
housing = st.selectbox("Housing", ["no", "yes", "unknown"])
loan = st.selectbox("Loan", ["no", "yes", "unknown"])
contact = st.selectbox("Contact", ["cellular", "telephone", "unknown"])
month = st.selectbox("Month", ["may", "jun", "nov", "sep", "jul", "aug", "mar", "oct", "apr", "dec"])
day_of_week = st.selectbox("Day of Week", ["fri", "wed", "mon", "thu", "tue"])
duration = st.slider("Duration", min_value=0.0, max_value=3643.0, value=300.0)
campaign = st.slider("Campaign", min_value=0.0, max_value=35.0, value=10.0)
pdays = st.slider("Pdays", min_value=0.0, max_value=999.0, value=15.0)
previous = st.slider("Previous", min_value=0.0, max_value=6.0, value=5.0)
poutcome = st.selectbox("Poutcome", ["nonexistent", "failure", "success"])
emp_var_rate = st.slider("Employment Variation Rate", min_value=-3.0, max_value=1.0, value=0.0)
cons_conf_idx = st.slider("Consumer Confidence Index", min_value=-50.0, max_value=0.0, value=-35.0)
euribor3m = st.slider("Euribor 3 Month Rate", min_value=0.0, max_value=5.0, value=3.0)

# Create a dataframe with the user input
input_data = pd.DataFrame({
    'job': [job], 'marital': [marital_status], 'education': [education], 'default': [default],
    'housing': [housing], 'loan': [loan], 'contact': [contact], 'month': [month],
    'day_of_week': [day_of_week], 'duration': [duration], 'campaign': [campaign],
    'pdays': [pdays], 'previous': [previous], 'poutcome': [poutcome],
    'emp.var.rate': [emp_var_rate], 'cons.conf.idx': [cons_conf_idx], 'euribor3m': [euribor3m]
})

# Preprocess the input
processed_data = preprocess_input(input_data)

# Predict on button click
if st.button("Predict"):
    prediction = make_prediction(model, processed_data)
    st.write(f"Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
