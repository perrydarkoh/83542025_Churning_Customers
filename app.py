import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers.legacy import Adam


# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# Load the model
model = load_model('best_model.h5')

# Title for the web app
st.title("Customer Churn Prediction Web App")

# Sidebar for user input features
st.sidebar.header("Input User data")


monthly_charges = st.sidebar.slider("Monthly Charges", min_value=0.0, max_value=1000.0, step=1.0, value=0.0)

# Contract encoding mapping
PaperlessBilling_encoded_mapping = {'Yes': 1, 'No': 0}
PaperlessBilling = st.sidebar.selectbox("PaperlessBilling", options=list(PaperlessBilling_encoded_mapping.keys()))
PaperlessBilling_encoded = PaperlessBilling_encoded_mapping[PaperlessBilling]

# Contract encoding mapping
SeniorCitizen_mapping = {'Yes': 1, 'No': 0}
SeniorCitizen = st.sidebar.selectbox("SeniorCitizen", options=list(SeniorCitizen_mapping.keys()))
SeniorCitizen_encoded  = SeniorCitizen_mapping[SeniorCitizen]


# Payment method encoding mapping
payment_method_encoded_mapping = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}
payment_method = st.sidebar.selectbox("Payment Method", options=list(payment_method_encoded_mapping.keys()))
payment_method_encoded = payment_method_encoded_mapping[payment_method]


MultipleLines_encoded_mapping =  {'No': 0, 'Yes': 1}
MultipleLines_ = st.sidebar.selectbox("Multiple Lines_", options=list(MultipleLines_encoded_mapping.keys()))
ultipleLines_encoded = MultipleLines_encoded_mapping[MultipleLines_]

##'MonthlyCharges', 'PaperlessBilling_encoded', 'SeniorCitizen', 'PaymentMethod_encoded', 'MultipleLines_encoded']

# Creating a dataframe from user input
if st.sidebar.button("Submit"):
    user_input = pd.DataFrame({
        'MonthlyCharges': [monthly_charges],
        'PaperlessBilling_encoded': [PaperlessBilling_encoded],
        'SeniorCitizen': [SeniorCitizen_encoded],
        'PaymentMethod_encoded': [payment_method_encoded],
        'MultipleLines_encoded': [ultipleLines_encoded]
    })

    # Scaling the data
    user_input_scaled = loaded_scaler.transform(user_input)
    user_input_scaled_df = pd.DataFrame(user_input_scaled, columns=user_input.columns)

    # Using the tuned model to make predictions
    prediction = model.predict(user_input_scaled_df)

    # Displaying the prediction
    st.subheader("Prediction")
    churn_status = "Churn" if prediction[0, 0] > 0.5 else "No Churn"

    st.write(churn_status)
