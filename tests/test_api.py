"""
Integration tests for FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app, load_all_artifacts

# Explicitly initialize artifacts before running client tests
load_all_artifacts()
client = TestClient(app)

SAMPLE_CUSTOMER = {
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
}


def test_health_endpoint():
    """Verify /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True
    assert data["dataset_records_available"] > 0


def test_model_info_endpoint():
    """Verify /model-info returns valid metadata, metrics, and coefficients."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "metrics_default_threshold" in data
    assert "all_coefficients" in data
    assert len(data["all_coefficients"]) > 0


def test_clusters_endpoint():
    """Verify /clusters returns optimal_k, profiles, and PCA loadings."""
    response = client.get("/clusters")
    assert response.status_code == 200
    data = response.json()
    assert data["optimal_k"] == 4
    assert len(data["cluster_profiles"]) == 4


def test_analytics_summary_endpoint():
    """Verify /analytics-summary returns complete aggregated intelligence."""
    response = client.get("/analytics-summary")
    assert response.status_code == 200
    data = response.json()
    assert "eda" in data
    assert "statistical_tests" in data
    assert "model_performance" in data
    assert "segmentation" in data


def test_customers_explorer_endpoint():
    """Verify /customers returns paginated customer records with filtering."""
    response = client.get("/customers?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 10127
    assert len(data["customers"]) == 10
    assert "churn_probability" in data["customers"][0]


def test_predict_endpoint():
    """Verify /predict returns probability, prediction, and risk tier."""
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["predicted_churn"] in [0, 1]
    assert data["risk_level"] in ["Low", "Medium", "High", "Critical"]


def test_segment_endpoint():
    """Verify /segment returns valid cluster assignment and strategy."""
    response = client.post("/segment", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert data["cluster_id"] in [0, 1, 2, 3]
    assert len(data["segment_name"]) > 0
    assert len(data["recommended_strategy"]) > 0


def test_analyze_customer_endpoint():
    """Verify /analyze-customer returns unified intelligence."""
    response = client.post("/analyze-customer", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "segment_name" in data
    assert "key_drivers" in data
