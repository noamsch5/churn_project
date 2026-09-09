"""
Utility functions and configuration constants for the Bank Churn & Customer Segmentation project.
"""

from pathlib import Path
import json
import logging
from typing import Any
import numpy as np

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

RAW_DATA_FILE = RAW_DATA_DIR / "BankChurners.csv"
DATA_URL = "https://raw.githubusercontent.com/azar-s91/dataset/master/BankChurners.csv"

# Modeling constants
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "Attrition_Flag"
ID_COLUMN = "CLIENTNUM"

# Target mapping
TARGET_MAP = {
    "Existing Customer": 0,
    "Attrited Customer": 1
}

# Feature groupings
DEMOGRAPHIC_FEATURES = [
    "Customer_Age",
    "Gender",
    "Dependent_count",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
]

RELATIONSHIP_FEATURES = [
    "Card_Category",
    "Months_on_book",
    "Total_Relationship_Count",
]

ENGAGEMENT_FEATURES = [
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
]

FINANCIAL_FEATURES = [
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Avg_Utilization_Ratio",
]

TRANSACTION_FEATURES = [
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
]

ALL_FEATURE_COLUMNS = (
    DEMOGRAPHIC_FEATURES
    + RELATIONSHIP_FEATURES
    + ENGAGEMENT_FEATURES
    + FINANCIAL_FEATURES
    + TRANSACTION_FEATURES
)

CLUSTERING_FEATURES = [
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Ct_Chng_Q4_Q1",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Total_Relationship_Count",
    "Credit_Limit",
    "Avg_Utilization_Ratio",
]


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


def get_logger(name: str) -> logging.Logger:
    """Configures and returns a standard logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
