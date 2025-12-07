import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Load the Model
model_filename = 'loan_payback_model.pkl'

try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: The file '{model_filename}' was not found. Please upload it to the same directory.")
    st.stop()

# 2. App Title
st.title("Loan Payback Prediction App")
st.write("Enter the applicant's details below to predict if the loan will be paid back.")

# --- USER INPUT SECTION ---
st.sidebar.header("Applicant Information")

# Function to get user input
def user_input_features():
    # --- Numerical Features ---
    # Adjust min_value and max_value based on your real data range
    annual_income = st.sidebar.number_input("Annual Income ($)", min_value=30000, value=400000)
    loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=1000, value=50000)
    interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=3.2, value=20.0, step=0.1)

    # --- Categorical Features ---
    # IMPORTANT: You must list ALL options that were in your training data.
    # The LabelEncoder usually sorts them alphabetically (A-Z).
    # Please update these lists with the exact values from your dataset.
    
    gender_options = ["Female", "Male"] 
    employment_options = ["Employed", "Self-Employed", "Unemployed", "Student","Retired"] 
    education_options = ["High School", "Bachelor's", "Master's","Other", "PhD"] 
    marital_options = ["Single", "Married", "Divorced","Widoved"]
    
    # Select boxes
    gender = st.sidebar.selectbox("Gender", gender_options)
    marital_status = st.sidebar.selectbox("Marital Status", marital_options)
    education_level = st.sidebar.selectbox("Education Level", education_options)
    employment_status = st.sidebar.selectbox("Employment Status", employment_options)

    # --- Encoding (Text to Numbers) ---
    # Your notebook used LabelEncoder, which assigns numbers alphabetically (0, 1, 2...)
    # We replicate that logic here automatically by sorting the options.
    
    def encode_option(selected_option, all_options):
        # Sort options alphabetically to match LabelEncoder behavior
        sorted_options = sorted(all_options)
        # Return the index (0, 1, 2...)
        return sorted_options.index(selected_option)

    # Create the data dictionary with encoded values
    # NOTE: The order of columns here MUST match the order in your X_train
    data = {
        'annual_income': annual_income,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'gender': encode_option(gender, gender_options),
        'marital_status': encode_option(marital_status, marital_options),
        'education_level': encode_option(education_level, education_options),
        'employment_status': encode_option(employment_status, employment_options)
    }
    
    # If your model expects features in a specific order, you might need to reorder the DataFrame columns later.
    features = pd.DataFrame(data, index=[0])
    return features

# Get the input dataframe
input_df = user_input_features()

# Display the input parameters for verification
st.subheader("User Input parameters")
st.write(input_df)

# --- PREDICTION SECTION ---
if st.button("Predict"):
    try:
        # Note: input_df columns might need reordering to match training data exactly.
        # If you get a 'feature mismatch' error, check the column order.
        prediction = model.predict(input_df)
        
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success("Result: Loan will be **PAID BACK**")
        else:
            st.warning("Result: Loan will **NOT** be paid back")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Tip: Ensure the input columns match exactly what the model expects (order and number).")