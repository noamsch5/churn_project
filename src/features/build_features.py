"""
Feature engineering module with Scikit-learn transformer compatibility.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.helpers import get_logger

logger = get_logger(__name__)

ENGINEERED_NUMERICAL_COLS = [
    "avg_trans_value",
    "trans_per_month",
    "trans_amt_per_month",
    "inactivity_ratio",
    "declining_trans_ct_flag",
    "declining_trans_amt_flag",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that constructs business-grounded
    behavioral and activity features.
    """

    def __init__(self, create_flags: bool = True):
        self.create_flags = create_flags

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Derive interpretable behavioral features with division-by-zero protection.
        """
        X = X.copy()

        # 1. Average Transaction Value
        trans_ct = X["Total_Trans_Ct"].replace(0, np.nan)
        X["avg_trans_value"] = (X["Total_Trans_Amt"] / trans_ct).fillna(0.0)

        # 2. Transactions per relationship month (tenure)
        tenure = X["Months_on_book"].replace(0, np.nan)
        X["trans_per_month"] = (X["Total_Trans_Ct"] / tenure).fillna(0.0)

        # 3. Transaction amount per relationship month
        X["trans_amt_per_month"] = (X["Total_Trans_Amt"] / tenure).fillna(0.0)

        # 4. Normalized Inactivity Ratio over 12 months
        X["inactivity_ratio"] = X["Months_Inactive_12_mon"] / 12.0

        # 5. Behavioral Change Indicators (flags for >40% drops between Q4 and Q1)
        if self.create_flags:
            X["declining_trans_ct_flag"] = (X["Total_Ct_Chng_Q4_Q1"] < 0.6).astype(int)
            X["declining_trans_amt_flag"] = (X["Total_Amt_Chng_Q4_Q1"] < 0.6).astype(int)

        return X


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience functional interface for feature engineering."""
    fe = FeatureEngineer()
    return fe.transform(df)
