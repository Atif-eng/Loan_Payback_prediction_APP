import pickle

import pandas as pd
import streamlit as st

from src.inference import load_model, prepare_user_input


st.set_page_config(page_title="Loan Payback Prediction App", page_icon="💰", layout="wide")


@st.cache_resource
def get_model(model_path: str = "loan_payback_model.pkl"):
    with open(model_path, "rb") as file:
        return pickle.load(file)


model = get_model()

st.title("Loan Payback Prediction App")
st.write("Enter borrower and loan information to estimate whether the loan will be repaid.")

st.sidebar.header("Applicant Information")

annual_income = st.sidebar.number_input("Annual Income ($)", min_value=5000.0, value=60000.0, step=1000.0)
debt_to_income_ratio = st.sidebar.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=680, step=1)
loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=500.0, value=18000.0, step=100.0)
interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=3.2, max_value=25.0, value=12.5, step=0.1)

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
marital_status = st.sidebar.selectbox("Marital Status", ["Divorced", "Married", "Single", "Widowed"])
education_level = st.sidebar.selectbox("Education Level", ["Bachelor's", "High School", "Master's", "Other", "PhD"])
employment_status = st.sidebar.selectbox("Employment Status", ["Employed", "Retired", "Self-employed", "Student", "Unemployed"])
loan_purpose = st.sidebar.selectbox("Loan Purpose", [
    "Auto",
    "Business",
    "Debt consolidation",
    "Education",
    "Home improvement",
    "Medical",
    "Moving",
    "Other",
    "Vacation",
    "Wedding",
])
grade_subgrade = st.sidebar.selectbox("Grade/Subgrade", [
    "A1", "A2", "A3", "A4", "A5",
    "B1", "B2", "B3", "B4", "B5",
    "C1", "C2", "C3", "C4", "C5",
    "D1", "D2", "D3", "D4", "D5",
    "F1", "F2", "F3", "F4", "F5",
])

user_inputs = {
    "annual_income": annual_income,
    "debt_to_income_ratio": debt_to_income_ratio,
    "credit_score": credit_score,
    "loan_amount": loan_amount,
    "interest_rate": interest_rate,
    "gender": gender,
    "marital_status": marital_status,
    "education_level": education_level,
    "employment_status": employment_status,
    "loan_purpose": loan_purpose,
    "grade_subgrade": grade_subgrade,
}

input_df = prepare_user_input(user_inputs)
st.subheader("User Input Parameters")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        result = int(prediction[0])

        st.subheader("Prediction Result")
        if result == 1:
            st.success("Result: Loan will be PAID BACK")
        else:
            st.warning("Result: Loan will NOT be paid back")
    except Exception as exc:
        st.error(f"An error occurred while generating the prediction: {exc}")
        st.info("Please verify that the model and feature ordering match the training pipeline.")
