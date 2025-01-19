import streamlit as st
import numpy as npG
import joblib

# Load the saved model
model = joblib.load('credit_card_model.pkl')

# Title of the web app
st.title("Credit Card Fraud Detection")

# Get user input (make sure to get all the 30 features as input)
st.subheader("Enter the features of the transaction:")

# Define feature names (excluding 'Class' which is the target)
feature_names = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", 
    "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", 
    "V24", "V25", "V26", "V27", "V28", "Amount"
]

# Create a list to store user inputs
inputs = []

# Loop through each feature and collect user input
for feature in feature_names:
    value = st.number_input(f"Enter value for {feature}:", format="%.6f")
    inputs.append(value)

# Convert the inputs to a numpy array and reshape it
input_data = np.array(inputs).reshape(1, -1)

# Button to make the prediction
if st.button('Predict'):
    # Make the prediction using the trained model
    prediction = model.predict(input_data)

    # Display the prediction result
    if prediction[0] == 0:
        st.write("The transaction is **Normal**.")
    else:
        st.write("The transaction is **Fraudulent**.")
