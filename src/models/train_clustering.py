"""
Unsupervised learning module for Customer Segmentation.
Fits K-Means clustering and PCA for dimensionality reduction and visualization.
Evaluates Elbow & Silhouette scores, profiles clusters post-hoc with churn rates,
and outputs combined customer intelligence dataset.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

from src.utils.helpers import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    ID_COLUMN,
    TARGET_MAP,
    CLUSTERING_FEATURES,
    NpEncoder,
    get_logger,
)

logger = get_logger(__name__)


def evaluate_k_values(
    X_scaled: np.ndarray,
    k_range: list[int] = [2, 3, 4, 5, 6],
) -> list[dict]:
    """
    Evaluates KMeans over k_range using inertia (Elbow) and Silhouette scores.
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = float(km.inertia_)
        # Calculate silhouette score (using sample for high speed if dataset is large, or full)
        sil = float(silhouette_score(X_scaled, labels, sample_size=3000, random_state=RANDOM_STATE))
        results.append({
            "k": k,
            "inertia": round(inertia, 2),
            "silhouette_score": round(sil, 4),
        })
        logger.info("k=%d: Inertia=%.2f, Silhouette=%.4f", k, inertia, sil)
    return results


def train_clustering_and_pca(selected_k: int = 4) -> dict:
    """
    Main clustering training, profiling, and PCA visualization generator.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    from src.data.load_data import load_raw_data

    df = load_raw_data()
    n_total = len(df)

    # 1. Feature matrix without Attrition_Flag or CLIENTNUM
    X_cluster = df[CLUSTERING_FEATURES].copy()

    # Preprocessing: StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # 2. Evaluate K values
    logger.info("Evaluating K values for K-Means...")
    k_eval = evaluate_k_values(X_scaled, k_range=[2, 3, 4, 5, 6])

    # 3. Fit the selected K-Means solution. This is a business-informed choice,
    # not a claim that k=4 maximizes every quantitative validity metric.
    logger.info("Fitting final K-Means model with selected k=%d...", selected_k)
    kmeans = KMeans(n_clusters=selected_k, random_state=RANDOM_STATE, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Build scikit-learn pipeline
    clustering_pipeline = Pipeline([
        ("scaler", scaler),
        ("kmeans", kmeans),
    ])
    joblib.dump(clustering_pipeline, MODELS_DIR / "clustering_pipeline.joblib")

    # 4. PCA for dimensionality reduction and 2D visualization
    logger.info("Fitting PCA for 2D visualization...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    joblib.dump(pca, MODELS_DIR / "pca_model.joblib")

    explained_variance = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    logger.info("PCA Explained Variance Ratio: PC1=%.4f, PC2=%.4f (Total=%.4f)",
                explained_variance[0], explained_variance[1], sum(explained_variance))

    # PCA Component Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2"],
        index=CLUSTERING_FEATURES,
    )
    loadings_dict = [
        {
            "feature": feat,
            "PC1_loading": round(float(loadings.loc[feat, "PC1"]), 4),
            "PC2_loading": round(float(loadings.loc[feat, "PC2"]), 4),
        }
        for feat in CLUSTERING_FEATURES
    ]

    # 5. Cluster Profiling (Post-Hoc Analysis with Churn Rate)
    df_profile = df.copy()
    df_profile["cluster"] = cluster_labels
    df_profile["target_churn"] = (df_profile[TARGET_COLUMN] == "Attrited Customer").astype(int)

    cluster_profiles = []
    # Identify segment characteristics to assign grounded business names
    for c_id in range(selected_k):
        c_sub = df_profile[df_profile["cluster"] == c_id]
        c_size = len(c_sub)
        churn_cnt = int(c_sub["target_churn"].sum())
        churn_rate = round(float(churn_cnt / c_size) * 100, 2)

        profile = {
            "cluster_id": c_id,
            "customer_count": c_size,
            "percentage_of_total": round(float(c_size / n_total) * 100, 2),
            "churn_count": churn_cnt,
            "churn_rate_pct": churn_rate,
            "avg_age": round(float(c_sub["Customer_Age"].mean()), 1),
            "avg_months_on_book": round(float(c_sub["Months_on_book"].mean()), 1),
            "avg_transaction_amt": round(float(c_sub["Total_Trans_Amt"].mean()), 2),
            "avg_transaction_count": round(float(c_sub["Total_Trans_Ct"].mean()), 1),
            "avg_amt_change": round(float(c_sub["Total_Amt_Chng_Q4_Q1"].mean()), 3),
            "avg_ct_change": round(float(c_sub["Total_Ct_Chng_Q4_Q1"].mean()), 3),
            "avg_inactive_months": round(float(c_sub["Months_Inactive_12_mon"].mean()), 2),
            "avg_contacts_count": round(float(c_sub["Contacts_Count_12_mon"].mean()), 2),
            "avg_relationship_count": round(float(c_sub["Total_Relationship_Count"].mean()), 2),
            "avg_credit_limit": round(float(c_sub["Credit_Limit"].mean()), 2),
            "avg_utilization_ratio": round(float(c_sub["Avg_Utilization_Ratio"].mean()), 3),
        }
        cluster_profiles.append(profile)

    # Assign segment identities from relative profile rankings, then build every
    # numerical description from the current cluster metadata. This prevents stale
    # hard-coded claims after retraining.
    remaining_ids = {p["cluster_id"] for p in cluster_profiles}
    high_spend_id = max(cluster_profiles, key=lambda p: p["avg_transaction_amt"])["cluster_id"]
    remaining_ids.remove(high_spend_id)
    high_churn_id = max(
        (p for p in cluster_profiles if p["cluster_id"] in remaining_ids),
        key=lambda p: p["churn_rate_pct"],
    )["cluster_id"]
    remaining_ids.remove(high_churn_id)
    high_util_id = max(
        (p for p in cluster_profiles if p["cluster_id"] in remaining_ids),
        key=lambda p: p["avg_utilization_ratio"],
    )["cluster_id"]
    remaining_ids.remove(high_util_id)
    relationship_id = next(iter(remaining_ids))

    for p in cluster_profiles:
        if p["cluster_id"] == high_spend_id:
            p["segment_name"] = "High-Volume Power Spenders"
            p["description"] = (
                f"Highest-spend segment, averaging ${p['avg_transaction_amt']:,.0f} and "
                f"{p['avg_transaction_count']:.1f} transactions annually; observed churn is "
                f"{p['churn_rate_pct']:.1f}%."
            )
            p["strategy"] = "Exclusive concierge perks, tier status retention, and premium partner rewards."
        elif p["cluster_id"] == high_churn_id:
            p["segment_name"] = "Disengaged & Underutilized (At-Risk)"
            p["description"] = (
                f"Highest observed-churn segment ({p['churn_rate_pct']:.1f}%), with "
                f"{p['avg_utilization_ratio'] * 100:.1f}% average utilization, a "
                f"{p['avg_ct_change']:.3f} Q4/Q1 transaction-count ratio, and "
                f"{p['avg_contacts_count']:.1f} annual contacts."
            )
            p["strategy"] = "Targeted win-back campaigns, card-usage incentives, fee reviews, and contact resolution."
        elif p["cluster_id"] == high_util_id:
            p["segment_name"] = "High-Utilization Credit Revolvers"
            p["description"] = (
                f"Highest-utilization remaining segment ({p['avg_utilization_ratio'] * 100:.1f}%), "
                f"averaging {p['avg_transaction_count']:.1f} annual transactions and "
                f"{p['churn_rate_pct']:.1f}% observed churn."
            )
            p["strategy"] = "Consider credit-line reviews, low-APR offers, and payment automation tools."
        elif p["cluster_id"] == relationship_id:
            p["segment_name"] = "Relationship-Rich Growth Customers"
            p["description"] = (
                f"Relationship-rich segment averaging {p['avg_relationship_count']:.1f} products, "
                f"a {p['avg_amt_change']:.3f} Q4/Q1 transaction-amount ratio, and "
                f"{p['churn_rate_pct']:.1f}% observed churn."
            )
            p["strategy"] = "Use relevant cross-sell offers and evaluate eligibility for premium card tiers."

    # Map names back to dataframe
    name_map = {p["cluster_id"]: p["segment_name"] for p in cluster_profiles}
    df_profile["segment_name"] = df_profile["cluster"].map(name_map)

    # 6. Combined Customer Intelligence Layer
    # Predict churn probability using trained supervised model
    churn_model = joblib.load(MODELS_DIR / "churn_model.joblib")
    churn_probs = churn_model.predict_proba(df)[:, 1]
    with open(MODELS_DIR / "model_metadata.json", "r") as f:
        churn_metadata = json.load(f)
    selected_threshold = float(churn_metadata["selected_threshold"])
    holdout_ids = set(pd.read_csv(PROCESSED_DATA_DIR / "test.csv")[ID_COLUMN].astype(int))

    df_profile["churn_probability"] = np.round(churn_probs, 4)
    df_profile["predicted_churn"] = (churn_probs >= selected_threshold).astype(int)
    df_profile["classification_threshold"] = selected_threshold
    df_profile["prediction_scope"] = np.where(
        df_profile[ID_COLUMN].astype(int).isin(holdout_ids),
        "holdout_test",
        "training_in_sample_demo",
    )

    def assign_risk_level(prob: float) -> str:
        if prob < 0.25:
            return "Low"
        elif prob < 0.50:
            return "Medium"
        elif prob < 0.75:
            return "High"
        else:
            return "Critical"

    df_profile["risk_level"] = [assign_risk_level(p) for p in churn_probs]
    df_profile["PC1"] = np.round(X_pca[:, 0], 3)
    df_profile["PC2"] = np.round(X_pca[:, 1], 3)

    # Save complete customer intelligence dataset
    customer_intelligence_cols = [
        ID_COLUMN,
        TARGET_COLUMN,
        "churn_probability",
        "predicted_churn",
        "classification_threshold",
        "prediction_scope",
        "risk_level",
        "cluster",
        "segment_name",
        "PC1",
        "PC2",
        "Customer_Age",
        "Gender",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Utilization_Ratio",
    ]
    df_intelligence = df_profile[customer_intelligence_cols]
    df_intelligence.to_csv(PROCESSED_DATA_DIR / "customer_intelligence.csv", index=False)

    # Sample 1000 points for lightweight 2D PCA scatter visualization in frontend
    sample_df = df_intelligence.sample(n=min(1000, len(df_intelligence)), random_state=RANDOM_STATE)
    pca_scatter_points = [
        {
            "client_num": int(row[ID_COLUMN]),
            "pc1": float(row["PC1"]),
            "pc2": float(row["PC2"]),
            "cluster_id": int(row["cluster"]),
            "segment_name": str(row["segment_name"]),
            "churn_prob": float(row["churn_probability"]),
            "actual_churn": int(row[TARGET_COLUMN] == "Attrited Customer"),
            "risk_level": str(row["risk_level"]),
        }
        for _, row in sample_df.iterrows()
    ]

    output_metadata = {
        "k_evaluation": k_eval,
        "selected_k": selected_k,
        "selection_statement": f"Selected k = {selected_k} for business interpretability after evaluating k=2–6.",
        "selection_rationale": {
            "silhouette_note": "k=2 has the highest Silhouette score; k=4 was not selected as an unambiguous quantitative optimum.",
            "criteria": ["elbow pattern", "interpretability", "cluster sizes", "business usefulness"],
        },
        "clustering_features": CLUSTERING_FEATURES,
        "cluster_profiles": cluster_profiles,
        "pca": {
            "explained_variance_ratio": explained_variance,
            "total_explained_variance": round(sum(explained_variance), 4),
            "loadings": loadings_dict,
            "scatter_sample": pca_scatter_points,
        }
    }

    with open(MODELS_DIR / "cluster_profiles.json", "w") as f:
        json.dump(output_metadata, f, indent=2, cls=NpEncoder)

    logger.info("Saved clustering_pipeline.joblib, pca_model.joblib, cluster_profiles.json, and customer_intelligence.csv.")
    return output_metadata


if __name__ == "__main__":
    meta = train_clustering_and_pca(selected_k=4)
    print("Clustering and PCA pipeline successfully completed.")
    for p in meta["cluster_profiles"]:
        print(f"Cluster {p['cluster_id']} ({p['segment_name']}): {p['customer_count']} customers ({p['percentage_of_total']}%), Churn Rate: {p['churn_rate_pct']}%")
