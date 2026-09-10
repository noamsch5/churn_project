"""
Supervised learning module for Credit Card Churn Prediction.
Trains Dummy Classifier baseline, standard Logistic Regression, and balanced Logistic Regression.
Evaluates cross-validation, validation-based threshold selection, one final holdout
test evaluation, and odds-ratio interpretations.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from src.utils.helpers import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    ID_COLUMN,
    NpEncoder,
    get_logger,
)
from src.data.preprocess import (
    split_data,
    create_preprocessor,
    prepare_and_save_splits,
    CATEGORICAL_FEATURES,
    ALL_NUMERICAL_FEATURES,
)
from src.features.build_features import FeatureEngineer
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_curve_points,
    select_threshold,
)

logger = get_logger(__name__)


def extract_feature_names(preprocessor, feature_engineered_df: pd.DataFrame) -> list[str]:
    """
    Extract readable feature names after ColumnTransformer one-hot encoding.
    """
    feature_names = []
    # 1. Numerical features pass through StandardScaler directly
    feature_names.extend(ALL_NUMERICAL_FEATURES)

    # 2. Categorical features from OneHotEncoder
    cat_pipeline = preprocessor.named_transformers_["cat"]
    ohe = cat_pipeline.named_steps["ohe"]
    ohe_cols = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(list(ohe_cols))

    return feature_names


def train_and_evaluate_churn_model() -> dict:
    """
    Full pipeline training and rigorous evaluation workflow.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = prepare_and_save_splits()

    logger.info("Instantiating preprocessing pipeline...")
    fe = FeatureEngineer()
    preprocessor = create_preprocessor()

    # Pre-transform training features for cross-validation and inspection
    X_train_fe = fe.fit_transform(X_train)

    # -------------------------------------------------------------
    # 1. Dummy Baseline Classifier (Most Frequent)
    # -------------------------------------------------------------
    logger.info("Evaluating Dummy Baseline Classifier...")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_test_preds = dummy.predict(X_test)
    dummy_test_probs = dummy.predict_proba(X_test)[:, 1]

    dummy_metrics = compute_classification_metrics(
        y_test.values, dummy_test_probs, threshold=0.5
    )

    # -------------------------------------------------------------
    # 2. Stratified 5-Fold Cross Validation for Models
    # -------------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    # Model A: Standard Logistic Regression
    pipe_standard = Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor", create_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    cv_res_standard = cross_validate(pipe_standard, X_train, y_train, cv=cv, scoring=scoring)

    # Model B: Balanced Class Weight Logistic Regression
    pipe_balanced = Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor", create_preprocessor()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])
    cv_res_balanced = cross_validate(pipe_balanced, X_train, y_train, cv=cv, scoring=scoring)

    logger.info(
        "Standard LogReg CV - ROC-AUC: %.4f, Recall: %.4f, F1: %.4f",
        cv_res_standard["test_roc_auc"].mean(),
        cv_res_standard["test_recall"].mean(),
        cv_res_standard["test_f1"].mean(),
    )
    logger.info(
        "Balanced LogReg CV - ROC-AUC: %.4f, Recall: %.4f, F1: %.4f",
        cv_res_balanced["test_roc_auc"].mean(),
        cv_res_balanced["test_recall"].mean(),
        cv_res_balanced["test_f1"].mean(),
    )

    # We select the balanced model or standard model based on operational retention goals
    # Balanced weights provide higher sensitivity (recall) at the default threshold.
    # We train balanced Logistic Regression as our primary model for superior default recall.
    selected_pipeline = pipe_balanced

    # Select the operating threshold exclusively from out-of-fold predictions on
    # the training partition. Every row is scored by a model that did not train on
    # that row; the untouched test set plays no role in threshold selection.
    logger.info("Selecting decision threshold from 5-fold out-of-fold training predictions...")
    oof_train_probs = cross_val_predict(
        selected_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    selected_threshold, threshold_sweep = select_threshold(
        y_train.values, oof_train_probs, metric="f1"
    )

    # Fit once on all training data only after model and threshold decisions are fixed.
    selected_pipeline.fit(X_train, y_train)

    # -------------------------------------------------------------
    # 3. One Final Holdout Test Set Evaluation
    # -------------------------------------------------------------
    y_test_probs = selected_pipeline.predict_proba(X_test)[:, 1]

    # Evaluate at default threshold 0.50
    default_metrics = compute_classification_metrics(y_test.values, y_test_probs, threshold=0.50)

    selected_threshold_metrics = compute_classification_metrics(
        y_test.values, y_test_probs, threshold=selected_threshold
    )

    curve_points = compute_curve_points(y_test.values, y_test_probs)

    # -------------------------------------------------------------
    # 4. Feature Interpretability & Odds Ratios
    # -------------------------------------------------------------
    clf = selected_pipeline.named_steps["classifier"]
    fitted_prep = selected_pipeline.named_steps["preprocessor"]
    feature_names = extract_feature_names(fitted_prep, X_train_fe)

    coefs = clf.coef_[0]
    odds_ratios = np.exp(coefs)

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "odds_ratio": odds_ratios,
        "abs_coef": np.abs(coefs),
    }).sort_values(by="coefficient", ascending=False)

    positive_drivers = coef_df[coef_df["coefficient"] > 0].to_dict(orient="records")
    negative_drivers = coef_df[coef_df["coefficient"] < 0].sort_values("coefficient", ascending=True).to_dict(orient="records")

    ohe = fitted_prep.named_transformers_["cat"].named_steps["ohe"]
    categorical_reference_categories = {
        feature: str(categories[0])
        for feature, categories in zip(CATEGORICAL_FEATURES, ohe.categories_)
    }

    # Format clean list for JSON, including the correct odds-ratio comparison unit.
    ranked_coefficients = [
        {
            "feature": row["feature"],
            "coefficient": round(float(row["coefficient"]), 4),
            "odds_ratio": round(float(row["odds_ratio"]), 4),
            "impact": "Positive association with churn odds" if row["coefficient"] > 0 else "Negative association with churn odds",
            "comparison_basis": (
                "one standard deviation increase"
                if row["feature"] in ALL_NUMERICAL_FEATURES
                else "category indicator versus the feature's reference category"
            ),
        }
        for _, row in coef_df.iterrows()
    ]

    # -------------------------------------------------------------
    # 5. Save Artifacts
    # -------------------------------------------------------------
    model_path = MODELS_DIR / "churn_model.joblib"
    joblib.dump(selected_pipeline, model_path)

    # Also save preprocessor standalone for inference speed if needed
    joblib.dump(fitted_prep, MODELS_DIR / "preprocessing_pipeline.joblib")

    model_metadata = {
        "model_name": "Logistic Regression (Class-Weighted)",
        "model_type": "LogisticRegression",
        "solver": "lbfgs",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "selected_threshold": selected_threshold,
        "default_threshold": 0.50,
        "metrics_default_threshold": default_metrics,
        "metrics_selected_threshold": selected_threshold_metrics,
        "evaluation_scope": "untouched_holdout_test",
        "test_set_size": int(len(y_test)),
        "threshold_selection": {
            "source": "5-fold out-of-fold predictions on the training partition",
            "metric": "f1",
            "test_set_used_for_selection": False,
        },
        "baseline_metrics": dummy_metrics,
        "cross_validation": {
            "standard_logreg": {
                "cv_roc_auc_mean": round(float(cv_res_standard["test_roc_auc"].mean()), 4),
                "cv_roc_auc_std": round(float(cv_res_standard["test_roc_auc"].std()), 4),
                "cv_recall_mean": round(float(cv_res_standard["test_recall"].mean()), 4),
                "cv_f1_mean": round(float(cv_res_standard["test_f1"].mean()), 4),
                "cv_accuracy_mean": round(float(cv_res_standard["test_accuracy"].mean()), 4),
            },
            "balanced_logreg": {
                "cv_roc_auc_mean": round(float(cv_res_balanced["test_roc_auc"].mean()), 4),
                "cv_roc_auc_std": round(float(cv_res_balanced["test_roc_auc"].std()), 4),
                "cv_recall_mean": round(float(cv_res_balanced["test_recall"].mean()), 4),
                "cv_f1_mean": round(float(cv_res_balanced["test_f1"].mean()), 4),
                "cv_accuracy_mean": round(float(cv_res_balanced["test_accuracy"].mean()), 4),
            },
        },
        "curves": curve_points,
        "threshold_sweep": threshold_sweep,
        "threshold_sweep_scope": "out_of_fold_training_predictions",
        "odds_ratio_notes": {
            "numerical_features": "Odds ratios represent a one-standard-deviation increase because numerical inputs are StandardScaled.",
            "categorical_features": "Odds ratios compare each one-hot category with the omitted reference category.",
            "categorical_reference_categories": categorical_reference_categories,
        },
        "top_churn_drivers": positive_drivers[:10],
        "top_retention_drivers": negative_drivers[:10],
        "all_coefficients": ranked_coefficients,
    }

    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(model_metadata, f, indent=2, cls=NpEncoder)

    logger.info("Saved churn_model.joblib and model_metadata.json to %s.", MODELS_DIR)
    return model_metadata


if __name__ == "__main__":
    meta = train_and_evaluate_churn_model()
    print("Supervised churn model successfully trained and evaluated.")
    print("ROC-AUC:", meta["metrics_default_threshold"]["roc_auc"])
    print("Recall (Default 0.50):", meta["metrics_default_threshold"]["recall"])
    print("Recall (Selected Threshold):", meta["metrics_selected_threshold"]["recall"])
    print("F1 (Selected Threshold):", meta["metrics_selected_threshold"]["f1"])
    print("Validation-Selected Threshold:", meta["selected_threshold"])
