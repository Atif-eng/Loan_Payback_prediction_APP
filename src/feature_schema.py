FEATURE_COLUMNS = [
    "annual_income",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount",
    "interest_rate",
    "gender",
    "marital_status",
    "education_level",
    "employment_status",
    "loan_purpose",
    "grade_subgrade",
]

CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "marital_status": ["Divorced", "Married", "Single", "Widowed"],
    "education_level": ["Bachelor's", "High School", "Master's", "Other", "PhD"],
    "employment_status": ["Employed", "Retired", "Self-employed", "Student", "Unemployed"],
    "loan_purpose": [
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
    ],
    "grade_subgrade": [
        "A1", "A2", "A3", "A4", "A5",
        "B1", "B2", "B3", "B4", "B5",
        "C1", "C2", "C3", "C4", "C5",
        "D1", "D2", "D3", "D4", "D5",
        "F1", "F2", "F3", "F4", "F5",
    ],
}


def encode_option(value: str, options: list[str]) -> int:
    """Match the label-encoding logic used in the training notebook."""
    ordered = sorted(options)
    return ordered.index(value)


def build_prediction_frame(data: dict) -> "pd.DataFrame":
    import pandas as pd

    for feature in FEATURE_COLUMNS:
        if feature not in data:
            raise KeyError(f"Missing required feature: {feature}")

    ordered = {feature: data[feature] for feature in FEATURE_COLUMNS}
    return pd.DataFrame([ordered], columns=FEATURE_COLUMNS)
