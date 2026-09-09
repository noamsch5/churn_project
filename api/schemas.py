"""
Pydantic data schemas for FastAPI request validation and response models.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """Features for a single customer credit card account."""
    Customer_Age: int = Field(45, ge=18, le=100, description="Customer age in years")
    Gender: str = Field("M", description="Gender: 'M' or 'F'")
    Dependent_count: int = Field(2, ge=0, le=10, description="Number of dependents")
    Education_Level: str = Field("Graduate", description="Education: High School, Graduate, Uneducated, College, Post-Graduate, Doctorate, Unknown")
    Marital_Status: str = Field("Married", description="Marital status: Married, Single, Divorced, Unknown")
    Income_Category: str = Field("$60K - $80K", description="Income bracket: Less than $40K, $40K - $60K, $60K - $80K, $80K - $120K, $120K +, Unknown")
    Card_Category: str = Field("Blue", description="Card category: Blue, Silver, Gold, Platinum")
    Months_on_book: int = Field(36, ge=1, le=100, description="Account tenure with bank in months")
    Total_Relationship_Count: int = Field(4, ge=1, le=10, description="Total banking products held")
    Months_Inactive_12_mon: int = Field(2, ge=0, le=12, description="Number of inactive months in past 12 months")
    Contacts_Count_12_mon: int = Field(2, ge=0, le=20, description="Number of contacts between bank and customer in past 12 months")
    Credit_Limit: float = Field(8500.0, ge=0.0, description="Total credit limit")
    Total_Revolving_Bal: float = Field(1200.0, ge=0.0, description="Total revolving balance")
    Avg_Open_To_Buy: float = Field(7300.0, ge=0.0, description="Average open to buy credit line")
    Total_Amt_Chng_Q4_Q1: float = Field(0.75, ge=0.0, description="Ratio of transaction amount in Q4 vs Q1")
    Total_Trans_Amt: float = Field(4200.0, ge=0.0, description="Total transaction amount in last 12 months")
    Total_Trans_Ct: int = Field(65, ge=0, description="Total transaction count in last 12 months")
    Total_Ct_Chng_Q4_Q1: float = Field(0.70, ge=0.0, description="Ratio of transaction count in Q4 vs Q1")
    Avg_Utilization_Ratio: float = Field(0.18, ge=0.0, le=1.0, description="Average credit utilization ratio")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictResponse(BaseModel):
    """Churn prediction response."""
    churn_probability: float
    churn_probability_pct: float
    predicted_churn: int
    predicted_status: str
    risk_level: str
    threshold_applied: float


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    customers: list[CustomerInput]


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    predictions: list[PredictResponse]
    total_customers: int
    predicted_attritions: int


class SegmentResponse(BaseModel):
    """Customer segmentation response."""
    cluster_id: int
    segment_name: str
    description: str
    recommended_strategy: str
    segment_churn_rate_pct: float


class CombinedAnalysisResponse(BaseModel):
    """Combined churn prediction and customer segmentation output."""
    churn_probability: float
    churn_probability_pct: float
    predicted_churn: int
    predicted_status: str
    risk_level: str
    threshold_applied: float
    cluster_id: int
    segment_name: str
    segment_description: str
    recommended_strategy: str
    key_drivers: list[dict[str, Any]]


class CustomerRecord(BaseModel):
    """Record in searchable customer database."""
    client_num: int
    actual_status: str
    churn_probability: float
    predicted_churn: int
    risk_level: str
    cluster_id: int
    segment_name: str
    age: int
    gender: str
    total_trans_amt: float
    total_trans_ct: int
    months_inactive: int
    contacts_count: int
    credit_limit: float
    utilization_ratio: float


class CustomerListResponse(BaseModel):
    """Paginated list of customer records."""
    total_records: int
    page: int
    page_size: int
    total_pages: int
    customers: list[CustomerRecord]
