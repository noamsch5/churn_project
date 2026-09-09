"""
Data loading, validation, and data dictionary generation.
"""

from pathlib import Path
import json
import urllib.request
import pandas as pd
import numpy as np

from src.utils.helpers import (
    RAW_DATA_FILE,
    DATA_URL,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    ID_COLUMN,
    ALL_FEATURE_COLUMNS,
    DEMOGRAPHIC_FEATURES,
    RELATIONSHIP_FEATURES,
    ENGAGEMENT_FEATURES,
    FINANCIAL_FEATURES,
    TRANSACTION_FEATURES,
    get_logger,
)

logger = get_logger(__name__)


def download_data(url: str = DATA_URL, destination: Path = RAW_DATA_FILE) -> Path:
    """Download raw dataset if not present."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        logger.info("Downloading dataset from %s to %s...", url, destination)
        urllib.request.urlretrieve(url, destination)
        logger.info("Download complete.")
    else:
        logger.info("Dataset already exists at %s.", destination)
    return destination


def load_raw_data(filepath: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load raw dataset and ensure expected 21 columns."""
    if not filepath.exists():
        download_data(destination=filepath)

    df = pd.read_csv(filepath)

    # In case there are extra classifier columns from Kaggle export, filter to canonical 21 columns
    canonical_columns = [ID_COLUMN, TARGET_COLUMN] + ALL_FEATURE_COLUMNS
    missing_cols = [c for c in canonical_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    df = df[canonical_columns].copy()
    logger.info("Loaded dataset with shape %s.", df.shape)
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Perform comprehensive dataset validation.
    Returns validation summary dictionary.
    """
    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df[ID_COLUMN].duplicated().sum())
    null_counts = df.isnull().sum().to_dict()
    total_nulls = int(df.isnull().sum().sum())

    target_dist = df[TARGET_COLUMN].value_counts().to_dict()
    churn_rate = float(
        (df[TARGET_COLUMN] == "Attrited Customer").mean()
    )

    # Categorical unknown values
    unknown_counts = {}
    for col in df.select_dtypes(include=["object"]).columns:
        cnt = int((df[col] == "Unknown").sum())
        if cnt > 0:
            unknown_counts[col] = {
                "count": cnt,
                "percentage": round(cnt / n_rows * 100, 2),
            }

    validation_summary = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "duplicate_rows": duplicate_rows,
        "duplicate_ids": duplicate_ids,
        "total_nulls": total_nulls,
        "null_counts": null_counts,
        "target_distribution": target_dist,
        "churn_rate": round(churn_rate, 4),
        "unknown_categories": unknown_counts,
        "is_valid": (
            n_rows == 10127
            and duplicate_rows == 0
            and duplicate_ids == 0
            and total_nulls == 0
        ),
    }

    logger.info("Validation completed. Valid: %s", validation_summary["is_valid"])
    logger.info(
        "Target distribution: %s (Churn rate: %.2f%%)",
        target_dist,
        churn_rate * 100,
    )
    return validation_summary


def generate_data_dictionary(df: pd.DataFrame) -> list[dict]:
    """
    Generate structured data dictionary with descriptions, feature groups,
    and modeling applicability.
    """
    descriptions = {
        "CLIENTNUM": "Unique customer account identifier",
        "Attrition_Flag": "Target variable: Existing Customer (0) or Attrited Customer (1)",
        "Customer_Age": "Customer age in years",
        "Gender": "Customer gender (M=Male, F=Female)",
        "Dependent_count": "Number of financial dependents",
        "Education_Level": "Educational qualification of the account holder",
        "Marital_Status": "Marital status (Married, Single, Divorced, Unknown)",
        "Income_Category": "Annual income bracket of the account holder",
        "Card_Category": "Type of credit card held (Blue, Silver, Gold, Platinum)",
        "Months_on_book": "Tenure of customer relationship with the bank in months",
        "Total_Relationship_Count": "Total number of banking products held by the customer",
        "Months_Inactive_12_mon": "Number of inactive months in the last 12 months",
        "Contacts_Count_12_mon": "Number of contacts between customer and bank in last 12 months",
        "Credit_Limit": "Credit limit on the credit card account",
        "Total_Revolving_Bal": "Total revolving balance on the credit card",
        "Avg_Open_To_Buy": "Average open to buy credit line (Credit_Limit - Total_Revolving_Bal)",
        "Total_Amt_Chng_Q4_Q1": "Ratio of total transaction amount in Q4 compared to Q1",
        "Total_Trans_Amt": "Total transaction amount in the last 12 months",
        "Total_Trans_Ct": "Total transaction count in the last 12 months",
        "Total_Ct_Chng_Q4_Q1": "Ratio of total transaction count in Q4 compared to Q1",
        "Avg_Utilization_Ratio": "Average credit card utilization ratio",
    }

    dictionary = []
    for col in df.columns:
        if col == ID_COLUMN:
            group = "Identifier"
            used_for_modeling = False
        elif col == TARGET_COLUMN:
            group = "Target"
            used_for_modeling = False
        elif col in DEMOGRAPHIC_FEATURES:
            group = "Demographics"
            used_for_modeling = True
        elif col in RELATIONSHIP_FEATURES:
            group = "Product / Relationship"
            used_for_modeling = True
        elif col in ENGAGEMENT_FEATURES:
            group = "Engagement"
            used_for_modeling = True
        elif col in FINANCIAL_FEATURES:
            group = "Financial / Credit"
            used_for_modeling = True
        elif col in TRANSACTION_FEATURES:
            group = "Transaction Behavior"
            used_for_modeling = True
        else:
            group = "Other"
            used_for_modeling = True

        has_unknown = bool((df[col] == "Unknown").any()) if df[col].dtype == "object" else False
        null_count = int(df[col].isnull().sum())

        dictionary.append({
            "column_name": col,
            "data_type": str(df[col].dtype),
            "description": descriptions.get(col, ""),
            "feature_group": group,
            "used_for_modeling": used_for_modeling,
            "null_count": null_count,
            "has_unknown": has_unknown,
            "unique_values_count": int(df[col].nunique()),
            "example_values": [str(x) for x in df[col].dropna().unique()[:3]],
        })

    return dictionary


def save_metadata(validation_summary: dict, data_dict: list[dict]) -> None:
    """Save validation summary and data dictionary to data/processed."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED_DATA_DIR / "validation_summary.json", "w") as f:
        json.dump(validation_summary, f, indent=2)

    with open(PROCESSED_DATA_DIR / "data_dictionary.json", "w") as f:
        json.dump(data_dict, f, indent=2)

    # Also save a human-readable markdown table
    md_lines = [
        "# Bank Churners Data Dictionary\n",
        "| Column Name | Data Type | Feature Group | Modeling | Nulls | Has 'Unknown' | Description |",
        "|---|---|---|:---:|:---:|:---:|---|",
    ]
    for item in data_dict:
        md_lines.append(
            f"| `{item['column_name']}` | {item['data_type']} | {item['feature_group']} | "
            f"{'Yes' if item['used_for_modeling'] else 'No'} | {item['null_count']} | "
            f"{'Yes' if item['has_unknown'] else 'No'} | {item['description']} |"
        )

    with open(PROCESSED_DATA_DIR / "DATA_DICTIONARY.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    logger.info("Saved data dictionary and validation summary to %s.", PROCESSED_DATA_DIR)


if __name__ == "__main__":
    df = load_raw_data()
    val = validate_data(df)
    d_dict = generate_data_dictionary(df)
    save_metadata(val, d_dict)
    print("Data loading & validation successfully executed.")
