# Loan Payback Prediction 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Library](https://img.shields.io/badge/Library-XGBoost-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Project Overview
This project aims to predict whether a borrower will pay back their loan or default. By analyzing financial and demographic data, the model helps financial institutions assess credit risk, minimize losses, and make informed lending decisions.

**Author:** Muhammad Atif

## Dataset
The dataset includes various details about the borrowers and their loan attributes.

- **Target Variable:** `loan_paid_back` (1 = Paid Back, 0 = Defaulted).
- **Key Features:**
  - **Financial:** `annual_income`, `debt_to_income_ratio`, `credit_score`, `loan_amount`, `interest_rate`.
  - **Demographic:** `gender`, `marital_status`, `education_level`, `employment_status`.
  - **Loan Details:** `loan_purpose`, `grade_subgrade`.

## Tech Stack
- **Language:** Python
- **Libraries:**
  - `pandas` & `numpy` for data manipulation.
  - `matplotlib` & `seaborn` for data visualization.
  - `scikit-learn` for preprocessing and model evaluation.
  - `xgboost` for gradient boosting.
  - `imblearn` for handling class imbalance (SMOTE).
  - `pickle` for saving the trained model.

## Project Workflow

### 1. Data Analysis & Exploration
- **Statistical Analysis:** Examined distributions of income, loan amounts, and credit scores.
- **Missing & Duplicate Check:** Verified data integrity (dataset was clean with no missing values).
- **Visualization:** Explored relationships between features like Credit Score vs. Loan Status.

### 2. Data Preprocessing
- **Encoding:** Converted categorical variables (e.g., `education_level`, `gender`) using `LabelEncoder`.
- **Scaling:** Applied `StandardScaler` to normalize numerical features.
- **Handling Imbalance:** Used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset, ensuring the model doesn't just predict the majority class.

### 3. Model Building
Several machine learning models were trained to identify the best performer:
- **Decision Tree Classifier:** A baseline model for interpretability.
- **Random Forest Classifier:** To reduce overfitting and improve accuracy.
- **XGBoost Classifier:** For high-performance predictions on structured data.

### 4. Evaluation
Models were assessed using:
- **Accuracy Score**
- **Confusion Matrix**
- **F1-Score** (Crucial for imbalanced datasets)
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/loan-payback-prediction.git](https://github.com/your-username/loan-payback-prediction.git)
