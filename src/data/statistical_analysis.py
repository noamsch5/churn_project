"""
Statistical analysis, hypothesis testing, and EDA summary generation.
Conducts Chi-Square tests, Mann-Whitney U tests, Welch's t-tests,
effect size estimations, and confidence intervals without making causal claims.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.utils.helpers import (
    RAW_DATA_FILE,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    ID_COLUMN,
    TARGET_MAP,
    DEMOGRAPHIC_FEATURES,
    RELATIONSHIP_FEATURES,
    ENGAGEMENT_FEATURES,
    FINANCIAL_FEATURES,
    TRANSACTION_FEATURES,
    NpEncoder,
    get_logger,
)
from src.features.build_features import add_engineered_features

logger = get_logger(__name__)


def compute_cramers_v(confusion_matrix: np.ndarray) -> float:
    """Calculate Cramér's V statistic for categorical association."""
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    r, k = confusion_matrix.shape
    phi2 = chi2 / n
    min_dim = min(r - 1, k - 1)
    if min_dim == 0:
        return 0.0
    return float(np.sqrt(phi2 / min_dim))


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cohen's d effect size for two independent samples."""
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    # Pooled standard deviation
    s_pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if s_pooled == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / s_pooled)


def run_categorical_tests(df: pd.DataFrame) -> list[dict]:
    """
    Run Chi-Square Tests of Independence for categorical features against churn.
    """
    cat_features = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]

    results = []
    for col in cat_features:
        contingency = pd.crosstab(df[col], df[TARGET_COLUMN])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
        v = compute_cramers_v(contingency.values)

        # Detailed breakdown of churn rate by category
        breakdown = []
        for cat, row in contingency.iterrows():
            total = int(row.sum())
            churned = int(row.get("Attrited Customer", 0))
            retained = int(row.get("Existing Customer", 0))
            rate = round((churned / total) * 100, 2) if total > 0 else 0.0
            breakdown.append({
                "category": str(cat),
                "total_customers": total,
                "churned_customers": churned,
                "retained_customers": retained,
                "churn_rate_pct": rate,
            })

        # Sort breakdown by churn rate descending
        breakdown = sorted(breakdown, key=lambda x: x["churn_rate_pct"], reverse=True)

        is_significant = bool(p_val < 0.05)
        effect_label = "Negligible"
        if v >= 0.25:
            effect_label = "Strong"
        elif v >= 0.15:
            effect_label = "Moderate"
        elif v >= 0.06:
            effect_label = "Small"

        interpretation = (
            f"Statistically significant association detected (p = {p_val:.2e} < 0.05, Cramér's V = {v:.3f}, {effect_label} effect). "
            f"Churn rates vary across categories from {breakdown[-1]['churn_rate_pct']}% to {breakdown[0]['churn_rate_pct']}%. "
            f"This indicates a measurable pattern of dependence, though not a causal relationship."
            if is_significant else
            f"No statistically significant association observed (p = {p_val:.4f} >= 0.05, Cramér's V = {v:.3f}). "
            f"Observed category variations are consistent with random fluctuation."
        )

        results.append({
            "feature": col,
            "test_type": "Chi-Square Test of Independence",
            "chi2_statistic": round(float(chi2), 4),
            "degrees_of_freedom": int(dof),
            "p_value": float(p_val),
            "cramers_v": round(float(v), 4),
            "effect_size_label": effect_label,
            "is_significant": is_significant,
            "interpretation": interpretation,
            "categories_breakdown": breakdown,
        })

    return results


def run_numerical_tests(df: pd.DataFrame) -> list[dict]:
    """
    Run Welch's t-test and Mann-Whitney U test for numerical features comparing
    churned vs retained customers. Computes 95% CIs and effect sizes.
    """
    num_features = [
        "Customer_Age",
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
        "avg_trans_value",
        "trans_per_month",
        "trans_amt_per_month",
        "inactivity_ratio",
    ]

    retained = df[df[TARGET_COLUMN] == "Existing Customer"]
    churned = df[df[TARGET_COLUMN] == "Attrited Customer"]

    results = []
    for col in num_features:
        ret_vals = retained[col].dropna().values
        chu_vals = churned[col].dropna().values

        # Welch's t-test (unequal variances)
        t_stat, t_pval = stats.ttest_ind(chu_vals, ret_vals, equal_var=False)

        # Mann-Whitney U test (non-parametric)
        u_stat, u_pval = stats.mannwhitneyu(chu_vals, ret_vals, alternative="two-sided")

        # Rank-biserial correlation: r = 1 - (2*U / (n1*n2))
        n_chu, n_ret = len(chu_vals), len(ret_vals)
        rank_biserial = float(1.0 - (2.0 * u_stat) / (n_chu * n_ret))

        # Cohen's d: (mean_churned - mean_retained) / s_pooled
        cohen_d = compute_cohens_d(chu_vals, ret_vals)

        mean_chu = float(np.mean(chu_vals))
        mean_ret = float(np.mean(ret_vals))
        median_chu = float(np.median(chu_vals))
        median_ret = float(np.median(ret_vals))

        diff_mean = mean_chu - mean_ret

        # 95% Confidence interval for difference in means (Welch-Satterthwaite)
        se_diff = np.sqrt(np.var(chu_vals, ddof=1) / n_chu + np.var(ret_vals, ddof=1) / n_ret)
        ci_lower = float(diff_mean - 1.96 * se_diff)
        ci_upper = float(diff_mean + 1.96 * se_diff)

        is_significant = bool(u_pval < 0.05)
        
        # Plain English direction & interpretation
        direction = "higher" if diff_mean > 0 else "lower"
        abs_d = abs(cohen_d)
        effect_label = "Negligible"
        if abs_d >= 0.8:
            effect_label = "Large"
        elif abs_d >= 0.5:
            effect_label = "Medium"
        elif abs_d >= 0.2:
            effect_label = "Small"

        interpretation = (
            f"Statistically significant difference (Mann-Whitney p = {u_pval:.2e} < 0.05, Cohen's d = {cohen_d:.3f}, {effect_label} effect). "
            f"Attrited customers exhibit on average {direction} {col} (mean: {mean_chu:.2f} vs {mean_ret:.2f}; "
            f"95% CI for difference: [{ci_lower:.2f}, {ci_upper:.2f}]). "
            f"This confirms a strong statistical disparity between retained and departing customers."
            if is_significant else
            f"No statistically significant difference detected (Mann-Whitney p = {u_pval:.4f} >= 0.05, Cohen's d = {cohen_d:.3f}). "
            f"Means are comparable ({mean_chu:.2f} vs {mean_ret:.2f})."
        )

        results.append({
            "feature": col,
            "mean_churned": round(mean_chu, 3),
            "mean_retained": round(mean_ret, 3),
            "median_churned": round(median_chu, 3),
            "median_retained": round(median_ret, 3),
            "diff_mean": round(diff_mean, 3),
            "ci_95_diff": [round(ci_lower, 3), round(ci_upper, 3)],
            "t_statistic": round(float(t_stat), 4),
            "t_p_value": float(t_pval),
            "mann_whitney_u": round(float(u_stat), 2),
            "mann_whitney_p_value": float(u_pval),
            "cohens_d": round(cohen_d, 4),
            "rank_biserial_correlation": round(rank_biserial, 4),
            "effect_size_label": effect_label,
            "is_significant": is_significant,
            "interpretation": interpretation,
        })

    # Sort numerical tests by absolute Cohen's d descending
    results = sorted(results, key=lambda x: abs(x["cohens_d"]), reverse=True)
    return results


def generate_eda_summary(df: pd.DataFrame) -> dict:
    """
    Generate comprehensive structured EDA summary for dashboard and notebooks.
    """
    n_total = len(df)
    churn_counts = df[TARGET_COLUMN].value_counts().to_dict()
    churn_rate = float((df[TARGET_COLUMN] == "Attrited Customer").mean())

    # Numerical summaries
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(ID_COLUMN, errors="ignore")
    numerical_summaries = {}
    for col in num_cols:
        series = df[col]
        numerical_summaries[col] = {
            "mean": round(float(series.mean()), 3),
            "std": round(float(series.std()), 3),
            "median": round(float(series.median()), 3),
            "q25": round(float(series.quantile(0.25)), 3),
            "q75": round(float(series.quantile(0.75)), 3),
            "min": round(float(series.min()), 3),
            "max": round(float(series.max()), 3),
        }

    # Correlation matrix of numerical features with numeric target
    numeric_df = df[num_cols].copy()
    numeric_df["target_churn"] = (df[TARGET_COLUMN] == "Attrited Customer").astype(int)
    corr_matrix = numeric_df.corr().round(4).to_dict()

    target_correlations = [
        {"feature": col, "correlation": corr_matrix["target_churn"][col]}
        for col in num_cols
    ]
    target_correlations = sorted(target_correlations, key=lambda x: abs(x["correlation"]), reverse=True)

    # Demographic overview
    demographics = {}
    for col in ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]:
        counts = df[col].value_counts().to_dict()
        demographics[col] = counts

    return {
        "dataset_size": n_total,
        "retained_count": int(churn_counts.get("Existing Customer", 0)),
        "attrited_count": int(churn_counts.get("Attrited Customer", 0)),
        "churn_rate": round(churn_rate, 4),
        "target_correlations": target_correlations,
        "numerical_summaries": numerical_summaries,
        "demographics_overview": demographics,
    }


def run_and_save_analysis():
    """Load dataset, add engineered features, run tests, and save outputs."""
    from src.data.load_data import load_raw_data

    df = load_raw_data()
    df_feat = add_engineered_features(df)

    cat_tests = run_categorical_tests(df_feat)
    num_tests = run_numerical_tests(df_feat)

    # Control the expected false discovery rate across the full family of primary
    # tests (categorical chi-square + numerical Mann-Whitney) with Benjamini-Hochberg.
    primary_p_values = [r["p_value"] for r in cat_tests] + [
        r["mann_whitney_p_value"] for r in num_tests
    ]
    rejected, adjusted_p_values, _, _ = multipletests(
        primary_p_values, alpha=0.05, method="fdr_bh"
    )
    all_results = cat_tests + num_tests
    for result, is_rejected, adjusted_p in zip(all_results, rejected, adjusted_p_values):
        result["fdr_adjusted_p_value"] = float(adjusted_p)
        result["is_significant_fdr_0_05"] = bool(is_rejected)
    eda = generate_eda_summary(df_feat)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    statistical_report = {
        "multiple_testing": {
            "method": "Benjamini-Hochberg false discovery rate correction",
            "alpha": 0.05,
            "family_size": len(primary_p_values),
            "primary_tests": "Categorical chi-square and numerical Mann-Whitney tests",
        },
        "categorical_tests": cat_tests,
        "numerical_tests": num_tests,
    }

    with open(PROCESSED_DATA_DIR / "statistical_tests.json", "w") as f:
        json.dump(statistical_report, f, indent=2, cls=NpEncoder)

    with open(PROCESSED_DATA_DIR / "eda_summary.json", "w") as f:
        json.dump(eda, f, indent=2, cls=NpEncoder)

    logger.info("Saved statistical_tests.json and eda_summary.json in %s.", PROCESSED_DATA_DIR)


if __name__ == "__main__":
    run_and_save_analysis()
