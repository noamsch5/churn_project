"""
FastAPI application serving churn predictions, customer segmentation,
and analytical intelligence dashboards.
"""

from pathlib import Path
from typing import Optional
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.utils.helpers import (
    ROOT_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    CLUSTERING_FEATURES,
    ID_COLUMN,
    TARGET_COLUMN,
    get_logger,
)
from src.features.build_features import add_engineered_features
from api.schemas import (
    CustomerInput,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    SegmentResponse,
    CombinedAnalysisResponse,
    CustomerRecord,
    CustomerListResponse,
)

logger = get_logger(__name__)

from contextlib import asynccontextmanager

# Global in-memory cache for artifacts
artifacts = {}


def load_all_artifacts():
    """Load serialized models, pipelines, and analytical metadata into memory."""
    logger.info("Loading models and metadata artifacts...")
    churn_model_path = MODELS_DIR / "churn_model.joblib"
    clustering_path = MODELS_DIR / "clustering_pipeline.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"
    clusters_path = MODELS_DIR / "cluster_profiles.json"
    stats_path = PROCESSED_DATA_DIR / "statistical_tests.json"
    eda_path = PROCESSED_DATA_DIR / "eda_summary.json"
    intelligence_path = PROCESSED_DATA_DIR / "customer_intelligence.csv"

    if churn_model_path.exists() and clustering_path.exists():
        artifacts["churn_model"] = joblib.load(churn_model_path)
        artifacts["clustering_pipeline"] = joblib.load(clustering_path)

    if meta_path.exists():
        with open(meta_path, "r") as f:
            artifacts["model_metadata"] = json.load(f)

    if clusters_path.exists():
        with open(clusters_path, "r") as f:
            artifacts["cluster_profiles"] = json.load(f)

    if stats_path.exists():
        with open(stats_path, "r") as f:
            artifacts["statistical_tests"] = json.load(f)

    if eda_path.exists():
        with open(eda_path, "r") as f:
            artifacts["eda_summary"] = json.load(f)

    if intelligence_path.exists():
        artifacts["customer_intelligence"] = pd.read_csv(intelligence_path)
    else:
        artifacts["customer_intelligence"] = pd.DataFrame()

    logger.info("Artifacts successfully loaded into memory.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_artifacts()
    yield


app = FastAPI(
    title="Credit Card Customer Churn & Segmentation API",
    description="Portfolio-grade API serving interpretable churn probabilities, K-Means segmentation, and analytics.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local React/Vite development and dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_risk_level(prob: float) -> str:
    """Classify the churn probability into descriptive, non-decision risk bands."""
    if prob < 0.25:
        return "Low"
    elif prob < 0.50:
        return "Medium"
    elif prob < 0.75:
        return "High"
    else:
        return "Critical"


@app.get("/health", tags=["System"])
def health():
    """Health check verifying model readiness."""
    models_ready = "churn_model" in artifacts and "clustering_pipeline" in artifacts
    return {
        "status": "healthy" if models_ready else "degraded",
        "models_loaded": models_ready,
        "dataset_records_available": len(artifacts.get("customer_intelligence", [])),
    }


@app.get("/model-info", tags=["Analytics"])
def model_info():
    """Returns supervised model architecture, metrics, threshold, and feature weights."""
    if "model_metadata" not in artifacts:
        raise HTTPException(status_code=503, detail="Model metadata not loaded.")
    return artifacts["model_metadata"]


@app.get("/clusters", tags=["Analytics"])
def clusters_info():
    """Returns K-Means cluster profiles, evaluation metrics, and PCA loadings."""
    if "cluster_profiles" not in artifacts:
        raise HTTPException(status_code=503, detail="Cluster profiles not loaded.")
    return artifacts["cluster_profiles"]


@app.get("/analytics-summary", tags=["Analytics"])
def analytics_summary():
    """Pre-computed EDA distributions, statistical hypothesis tests, ROC/PR curves, and PCA projection."""
    if "statistical_tests" not in artifacts or "eda_summary" not in artifacts:
        raise HTTPException(status_code=503, detail="Analytics data not loaded.")

    return {
        "eda": artifacts["eda_summary"],
        "statistical_tests": artifacts["statistical_tests"],
        "model_performance": {
            "metrics": artifacts["model_metadata"]["metrics_default_threshold"],
            "selected_threshold_metrics": artifacts["model_metadata"]["metrics_selected_threshold"],
            "baseline_metrics": artifacts["model_metadata"]["baseline_metrics"],
            "curves": artifacts["model_metadata"]["curves"],
            "threshold_sweep": artifacts["model_metadata"]["threshold_sweep"],
            "selected_threshold": artifacts["model_metadata"]["selected_threshold"],
            "threshold_selection": artifacts["model_metadata"]["threshold_selection"],
            "coefficients": artifacts["model_metadata"]["all_coefficients"],
            "odds_ratio_notes": artifacts["model_metadata"]["odds_ratio_notes"],
        },
        "segmentation": artifacts["cluster_profiles"],
    }


@app.get("/customers", response_model=CustomerListResponse, tags=["Customer Explorer"])
def list_customers(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(25, ge=5, le=100, description="Items per page"),
    risk_level: Optional[str] = Query(None, description="Filter by risk tier: Low, Medium, High, Critical"),
    segment: Optional[str] = Query(None, description="Filter by segment name"),
    search: Optional[str] = Query(None, description="Search by CLIENTNUM substring"),
    sort_by: str = Query("churn_probability", description="Column to sort by"),
    order: str = Query("desc", description="Sort order: 'asc' or 'desc'"),
):
    """Searchable, filterable, and paginated customer intelligence records."""
    df: pd.DataFrame = artifacts.get("customer_intelligence")
    if df is None or df.empty:
        raise HTTPException(status_code=503, detail="Customer database not loaded.")

    filtered_df = df.copy()

    # Search filter
    if search:
        search_str = str(search).strip()
        filtered_df = filtered_df[filtered_df[ID_COLUMN].astype(str).str.contains(search_str)]

    # Risk level filter
    if risk_level:
        filtered_df = filtered_df[filtered_df["risk_level"].str.lower() == risk_level.lower()]

    # Segment filter
    if segment:
        filtered_df = filtered_df[filtered_df["segment_name"].str.lower() == segment.lower()]

    # Sorting
    ascending = order.lower() == "asc"
    if sort_by in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

    total_records = len(filtered_df)
    total_pages = max(1, (total_records + page_size - 1) // page_size)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    sliced = filtered_df.iloc[start_idx:end_idx]

    records = [
        CustomerRecord(
            client_num=int(row[ID_COLUMN]),
            actual_status=str(row[TARGET_COLUMN]),
            churn_probability=float(row["churn_probability"]),
            predicted_churn=int(row["predicted_churn"]),
            risk_level=str(row["risk_level"]),
            cluster_id=int(row["cluster"]),
            segment_name=str(row["segment_name"]),
            age=int(row["Customer_Age"]),
            gender=str(row["Gender"]),
            total_trans_amt=float(row["Total_Trans_Amt"]),
            total_trans_ct=int(row["Total_Trans_Ct"]),
            months_inactive=int(row["Months_Inactive_12_mon"]),
            contacts_count=int(row["Contacts_Count_12_mon"]),
            credit_limit=float(row["Credit_Limit"]),
            utilization_ratio=float(row["Avg_Utilization_Ratio"]),
            prediction_scope=str(row["prediction_scope"]),
            threshold_applied=float(row["classification_threshold"]),
        )
        for _, row in sliced.iterrows()
    ]

    return CustomerListResponse(
        total_records=total_records,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        customers=records,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_churn(customer: CustomerInput):
    """Predicts churn probability and risk tier for a single credit card account."""
    if "churn_model" not in artifacts:
        raise HTTPException(status_code=503, detail="Churn model not loaded.")

    model = artifacts["churn_model"]
    threshold = artifacts["model_metadata"].get("selected_threshold", 0.50)

    # Convert Pydantic model to DataFrame
    input_df = pd.DataFrame([customer.model_dump()])

    prob = float(model.predict_proba(input_df)[0, 1])
    pred = int(prob >= threshold)
    risk = get_risk_level(prob)

    return PredictResponse(
        churn_probability=round(prob, 4),
        churn_probability_pct=round(prob * 100, 2),
        predicted_churn=pred,
        predicted_status="Attrited Customer" if pred == 1 else "Existing Customer",
        risk_level=risk,
        threshold_applied=threshold,
    )


@app.post("/predict-batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_churn_batch(batch: BatchPredictRequest):
    """Batch churn probability scoring for multiple customer records."""
    if "churn_model" not in artifacts:
        raise HTTPException(status_code=503, detail="Churn model not loaded.")

    model = artifacts["churn_model"]
    threshold = artifacts["model_metadata"].get("selected_threshold", 0.50)

    records = [c.model_dump() for c in batch.customers]
    input_df = pd.DataFrame(records)

    probs = model.predict_proba(input_df)[:, 1]
    responses = []
    attrited_count = 0

    for prob in probs:
        p = float(prob)
        pred = int(p >= threshold)
        if pred == 1:
            attrited_count += 1
        responses.append(
            PredictResponse(
                churn_probability=round(p, 4),
                churn_probability_pct=round(p * 100, 2),
                predicted_churn=pred,
                predicted_status="Attrited Customer" if pred == 1 else "Existing Customer",
                risk_level=get_risk_level(p),
                threshold_applied=threshold,
            )
        )

    return BatchPredictResponse(
        predictions=responses,
        total_customers=len(responses),
        predicted_attritions=attrited_count,
    )


@app.post("/segment", response_model=SegmentResponse, tags=["Inference"])
def segment_customer(customer: CustomerInput):
    """Assigns customer into an unsupervised behavioral cluster."""
    if "clustering_pipeline" not in artifacts or "cluster_profiles" not in artifacts:
        raise HTTPException(status_code=503, detail="Clustering pipeline not loaded.")

    cluster_pipe = artifacts["clustering_pipeline"]
    profiles = artifacts["cluster_profiles"]["cluster_profiles"]

    input_df = pd.DataFrame([customer.model_dump()])
    X_cluster = input_df[CLUSTERING_FEATURES]

    cluster_id = int(cluster_pipe.predict(X_cluster)[0])
    profile = next((p for p in profiles if p["cluster_id"] == cluster_id), profiles[0])

    return SegmentResponse(
        cluster_id=cluster_id,
        segment_name=profile["segment_name"],
        description=profile["description"],
        recommended_strategy=profile["strategy"],
        segment_churn_rate_pct=profile["churn_rate_pct"],
    )


@app.post("/analyze-customer", response_model=CombinedAnalysisResponse, tags=["Inference"])
def analyze_customer(customer: CustomerInput):
    """
    Combined customer intelligence endpoint returning churn risk, behavioral cluster,
    and observed behavioral risk indicators.
    """
    churn_res = predict_churn(customer)
    segment_res = segment_customer(customer)

    # These are transparent analysis-derived heuristics, not local model
    # attribution values (for example, SHAP values).
    input_dict = customer.model_dump()
    behavioral_risk_indicators = []

    if input_dict["Total_Trans_Ct"] < 50:
        behavioral_risk_indicators.append({
            "feature": "Total_Trans_Ct",
            "value": input_dict["Total_Trans_Ct"],
            "direction": "High Risk",
            "message": "Low annual transaction count is associated with higher observed attrition.",
        })
    if input_dict["Total_Ct_Chng_Q4_Q1"] < 0.60:
        behavioral_risk_indicators.append({
            "feature": "Total_Ct_Chng_Q4_Q1",
            "value": input_dict["Total_Ct_Chng_Q4_Q1"],
            "direction": "High Risk",
            "message": "Transaction count declined by over 40% between Q1 and Q4.",
        })
    if input_dict["Months_Inactive_12_mon"] >= 3:
        behavioral_risk_indicators.append({
            "feature": "Months_Inactive_12_mon",
            "value": input_dict["Months_Inactive_12_mon"],
            "direction": "High Risk",
            "message": "Prolonged card dormancy (3+ months inactive) strongly correlates with attrition.",
        })
    if input_dict["Contacts_Count_12_mon"] >= 4:
        behavioral_risk_indicators.append({
            "feature": "Contacts_Count_12_mon",
            "value": input_dict["Contacts_Count_12_mon"],
            "direction": "High Risk",
            "message": "Elevated support contact frequency signals unresolved customer friction.",
        })
    if input_dict["Total_Relationship_Count"] <= 2:
        behavioral_risk_indicators.append({
            "feature": "Total_Relationship_Count",
            "value": input_dict["Total_Relationship_Count"],
            "direction": "High Risk",
            "message": "A shallow product relationship is associated with higher observed attrition.",
        })
    if input_dict["Total_Trans_Ct"] >= 75 and input_dict["Total_Revolving_Bal"] > 1000:
        behavioral_risk_indicators.append({
            "feature": "Total_Trans_Ct & Revolving_Bal",
            "value": f"{input_dict['Total_Trans_Ct']} trans / ${input_dict['Total_Revolving_Bal']} bal",
            "direction": "Protective",
            "message": "High swipe volume and a steady revolving balance are associated with lower observed attrition.",
        })

    return CombinedAnalysisResponse(
        churn_probability=churn_res.churn_probability,
        churn_probability_pct=churn_res.churn_probability_pct,
        predicted_churn=churn_res.predicted_churn,
        predicted_status=churn_res.predicted_status,
        risk_level=churn_res.risk_level,
        threshold_applied=churn_res.threshold_applied,
        cluster_id=segment_res.cluster_id,
        segment_name=segment_res.segment_name,
        segment_description=segment_res.description,
        recommended_strategy=segment_res.recommended_strategy,
        behavioral_risk_indicators=behavioral_risk_indicators,
    )


# Mount built React frontend if distribution directory exists
frontend_dist = ROOT_DIR / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", tags=["Frontend SPA"])
    async def serve_spa(full_path: str):
        # Exclude API endpoints from SPA fallback
        if full_path.startswith("api/") or full_path in ["health", "model-info", "clusters", "analytics-summary", "customers", "predict", "segment", "analyze-customer", "docs", "openapi.json"]:
            raise HTTPException(status_code=404, detail="API endpoint not found.")
        file_path = frontend_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
