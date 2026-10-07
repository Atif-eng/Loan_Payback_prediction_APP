import pickle

import pandas as pd

from src.feature_schema import CATEGORICAL_OPTIONS, FEATURE_COLUMNS, build_prediction_frame, encode_option


def load_model(model_path: str = "loan_payback_model.pkl"):
    with open(model_path, "rb") as file:
        return pickle.load(file)


def prepare_user_input(user_inputs: dict) -> pd.DataFrame:
    values = {}

    for categorical_field, options in CATEGORICAL_OPTIONS.items():
        selected_value = user_inputs[categorical_field]
        values[categorical_field] = encode_option(selected_value, options)

    for numeric_field in [
        "annual_income",
        "debt_to_income_ratio",
        "credit_score",
        "loan_amount",
        "interest_rate",
    ]:
        values[numeric_field] = user_inputs[numeric_field]

    return build_prediction_frame(values)
