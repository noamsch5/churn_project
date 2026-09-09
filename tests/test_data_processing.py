"""
Unit tests for data loading, validation, preprocessing, and feature engineering.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.load_data import load_raw_data, validate_data
from src.features.build_features import FeatureEngineer, add_engineered_features
from src.data.preprocess import split_data, create_preprocessor


def test_data_validation():
    """Verify raw dataset conforms to canonical shape and validation constraints."""
    df = load_raw_data()
    assert df.shape == (10127, 21), f"Expected shape (10127, 21), got {df.shape}"
    val = validate_data(df)
    assert val["is_valid"] is True
    assert val["total_nulls"] == 0
    assert val["duplicate_rows"] == 0
    assert val["duplicate_ids"] == 0
    assert abs(val["churn_rate"] - 0.1607) < 0.005


def test_feature_engineering():
    """Test feature derivations and zero-division handling."""
    sample_data = pd.DataFrame({
        "Total_Trans_Amt": [0, 1000, 5000],
        "Total_Trans_Ct": [0, 20, 100],
        "Months_on_book": [0, 12, 36],
        "Months_Inactive_12_mon": [0, 2, 6],
        "Total_Ct_Chng_Q4_Q1": [0.4, 0.8, 1.2],
        "Total_Amt_Chng_Q4_Q1": [0.3, 0.9, 1.1],
    })

    fe = FeatureEngineer()
    out = fe.transform(sample_data)

    # Check safe division (no inf or NaN)
    assert not out["avg_trans_value"].isnull().any()
    assert not np.isinf(out["avg_trans_value"]).any()
    assert out.loc[0, "avg_trans_value"] == 0.0
    assert out.loc[1, "avg_trans_value"] == 50.0

    assert not out["trans_per_month"].isnull().any()
    assert out.loc[0, "trans_per_month"] == 0.0

    # Check inactivity ratio
    assert out.loc[2, "inactivity_ratio"] == 0.5

    # Check declining flags (< 0.60)
    assert out.loc[0, "declining_trans_ct_flag"] == 1
    assert out.loc[1, "declining_trans_ct_flag"] == 0
    assert out.loc[0, "declining_trans_amt_flag"] == 1
    assert out.loc[1, "declining_trans_amt_flag"] == 0


def test_train_test_split_stratification():
    """Verify stratified split preserves churn ratio within tolerance."""
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.20, random_state=42)

    assert len(X_train) == 8101
    assert len(X_test) == 2026

    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()

    assert abs(train_churn_rate - 0.1607) < 0.005
    assert abs(test_churn_rate - 0.1607) < 0.005
