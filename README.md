# Loan Payback Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Completed" />
</p>

A machine learning-powered web application that predicts whether a borrower is likely to repay a loan based on applicant and loan characteristics. The project combines a trained predictive model with an interactive Streamlit interface for easy real-time predictions.

## Project Overview

This project is designed to support credit risk assessment by identifying whether a loan is likely to be repaid or defaulted. It is especially useful for:

- financial institutions evaluating credit risk
- loan officers screening applicants
- data science teams experimenting with structured tabular ML workflows

The app takes demographic and financial inputs, processes them consistently with the training pipeline, and returns a binary prediction.

## Features

- Interactive Streamlit web app for real-time prediction
- Pretrained model saved as a pickle file
- Support for key borrower attributes such as income, loan amount, interest rate, gender, education, employment, and marital status
- Clean and user-friendly input form
- Binary classification output:
  - Paid Back
  - Not Paid Back

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Pickle
- Jupyter Notebook

## Dataset

The project uses a structured loan dataset with both financial and demographic features. The target variable is:

- loan_paid_back
  - 1 = Loan was paid back
  - 0 = Loan defaulted

Main features include:

- annual_income
- loan_amount
- interest_rate
- gender
- marital_status
- education_level
- employment_status
- other financial or profile attributes used during model training

## Model Workflow

The project follows a typical supervised learning lifecycle:

1. Data exploration and analysis
2. Data preprocessing and encoding
3. Feature engineering and scaling
4. Model training and comparison
5. Evaluation using classification metrics
6. Saving the best-performing model
7. Deploying the model in a Streamlit app

## Project Structure

```text
Loan_Payback_prediction_APP/
├── app.py                     # Streamlit application
├── prediction-loan-payback.ipynb  # Model training notebook
├── loan_payback_model.pkl     # Trained ML model
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
└── .gitignore                # Git ignore rules
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Atif-eng/Loan_Payback_prediction_APP.git
cd Loan_Payback_prediction_APP
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (typically http://localhost:8501).

## How to Use

1. Enter the applicant's details in the sidebar.
2. Adjust values such as income, loan amount, and interest rate.
3. Select categorical fields such as gender, marital status, education, and employment.
4. Click the Predict button.
5. The model returns whether the loan is expected to be repaid.

## Example Prediction

The app predicts a binary outcome:

- Result: Loan will be PAID BACK
- Result: Loan will NOT be paid back

## Model Notes

The model pipeline includes:

- categorical encoding using label-based transformation
- numerical scaling for consistency with training data
- prediction using a trained classifier saved to disk

To preserve compatibility, the app reproduces the same encoding logic used during training before making predictions.

## Requirements

The project dependencies are listed in `requirements.txt` and typically include:

- streamlit
- pandas
- numpy
- scikit-learn
- joblib or pickle-based model usage

## Author

Muhammad Atif

## License

This project is currently distributed without a formal license file. If you plan to reuse or distribute it publicly, consider adding an appropriate open-source license.

## Future Improvements

- add model explainability with SHAP or feature importance plots
- improve input validation and preprocessing consistency
- deploy to Streamlit Community Cloud or AWS/Azure
- add model accuracy and evaluation reporting in the app
- expand the dataset and retrain the model for better generalization

## Contributing

Contributions are welcome. If you would like to improve the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

If you want, I can also make this README even more polished for GitHub by adding:
- a screenshot section
- a demo GIF or app preview
- a badges row for deployment status
- a more formal project architecture section
- a dedicated "Business Impact" section for loan-risk use cases
