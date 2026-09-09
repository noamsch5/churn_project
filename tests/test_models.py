"""
Unit tests for supervised and unsupervised machine learning pipelines.
"""

import pytest
import joblib
import numpy as np
import pandas as pd

from src.utils.helpers import MODELS_DIR, CLUSTERING_FEATURES
from src.data.load_data import load_raw_data


@pytest.fixture(scope="module")
def churn_pipeline():
    """Load serialized churn model pipeline."""
    model_path = MODELS_DIR / "churn_model.joblib"
    assert model_path.exists(), "Model file not found; ensure train_churn_model has been run."
    return joblib.load(model_path)


@pytest.fixture(scope="module")
def clustering_pipeline():
    """Load serialized clustering pipeline."""
    pipe_path = MODELS_DIR / "clustering_pipeline.joblib"
    assert pipe_path.exists(), "Clustering pipeline file not found."
    return joblib.load(pipe_path)


@pytest.fixture(scope="module")
def sample_customer_df():
    """Single sample customer dataframe."""
    return pd.DataFrame([{
        "Customer_Age": 45,
        "Gender": "M",
        "Dependent_count": 2,
        "Education_Level": "Graduate",
        "Marital_Status": "Married",
        "Income_Category": "$60K - $80K",
        "Card_Category": "Blue",
        "Months_on_book": 36,
        "Total_Relationship_Count": 4,
        "Months_Inactive_12_mon": 2,
        "Contacts_Count_12_mon": 2,
        "Credit_Limit": 8500.0,
        "Total_Revolving_Bal": 1200.0,
        "Avg_Open_To_Buy": 7300.0,
        "Total_Amt_Chng_Q4_Q1": 0.75,
        "Total_Trans_Amt": 4200.0,
        "Total_Trans_Ct": 65,
        "Total_Ct_Chng_Q4_Q1": 0.70,
        "Avg_Utilization_Ratio": 0.18,
    }])


def test_churn_prediction_shape_and_range(churn_pipeline, sample_customer_df):
    """Test that churn pipeline outputs valid probabilities bounded in [0, 1]."""
    probs = churn_pipeline.predict_proba(sample_customer_df)
    assert probs.shape == (1, 2)
    p_churn = probs[0, 1]
    assert 0.0 <= p_churn <= 1.0
    preds = churn_pipeline.predict(sample_customer_df)
    assert preds[0] in [0, 1]


def test_clustering_prediction(clustering_pipeline, sample_customer_df):
    """Test that clustering pipeline assigns a valid cluster index."""
    X_clust = sample_customer_df[CLUSTERING_FEATURES]
    cluster_id = clustering_pipeline.predict(X_clust)
    assert len(cluster_id) == 1
    assert cluster_id[0] in [0, 1, 2, 3]
