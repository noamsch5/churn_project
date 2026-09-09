"""
Preprocessing, data splitting, and ColumnTransformer construction.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.utils.helpers import (
    RAW_DATA_FILE,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    ID_COLUMN,
    TARGET_MAP,
    RANDOM_STATE,
    TEST_SIZE,
    DEMOGRAPHIC_FEATURES,
    RELATIONSHIP_FEATURES,
    ENGAGEMENT_FEATURES,
    FINANCIAL_FEATURES,
    TRANSACTION_FEATURES,
    get_logger,
)
from src.features.build_features import FeatureEngineer, ENGINEERED_NUMERICAL_COLS

logger = get_logger(__name__)

# Base categorical and numerical columns
CATEGORICAL_FEATURES = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

BASE_NUMERICAL_FEATURES = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

ALL_NUMERICAL_FEATURES = BASE_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_COLS


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified split into train and holdout test sets.
    Retains CLIENTNUM in separate series for tracking.
    Encodes target variable: Existing Customer -> 0, Attrited Customer -> 1.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map(TARGET_MAP)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Train set: %d rows (churn: %.2f%%), Test set: %d rows (churn: %.2f%%)",
        len(X_train),
        y_train.mean() * 100,
        len(X_test),
        y_test.mean() * 100,
    )

    return X_train, X_test, y_train, y_test


def create_preprocessor(
    numerical_cols: list[str] = ALL_NUMERICAL_FEATURES,
    categorical_cols: list[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """
    Build Scikit-learn ColumnTransformer for scaling and one-hot encoding.
    Ensures zero leakage by maintaining fit-only status until applied to training data.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def create_full_pipeline(estimator) -> Pipeline:
    """
    Constructs an end-to-end Pipeline: FeatureEngineer -> ColumnTransformer -> Estimator.
    """
    preprocessor = create_preprocessor()
    pipeline = Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])
    return pipeline


def prepare_and_save_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load raw data, split, and save train/test CSVs."""
    from src.data.load_data import load_raw_data

    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    logger.info("Saved train.csv and test.csv in %s.", PROCESSED_DATA_DIR)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prepare_and_save_splits()
