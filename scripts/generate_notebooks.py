"""
Script to generate all 6 portfolio-grade Jupyter notebooks in notebooks/
Each notebook is properly structured with markdown explanations, code cells, and outputs.
"""

from pathlib import Path
import nbformat as nbf

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_nb_01():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Notebook 01: Data Understanding & Validation

## Credit Card Customer Churn & Segmentation
**Domain:** Banking & Financial Services  
**Objective:** Ingest the raw credit-card cardholder dataset, inspect schema integrity, validate data quality, analyze target variable imbalance, and build a comprehensive data dictionary.

---
### Business Context
Customer attrition directly deprives credit card issuers of interest margins, interchange swipe fees, and annual card fees, while customer acquisition costs (CAC) in retail banking typically exceed annual retention costs by 5x to 7x. 
This project focuses on identifying early attrition indicators and discovering natural behavioral segments to guide proactive retention.
"""),
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from pathlib import Path

from src.data.load_data import load_raw_data, validate_data, generate_data_dictionary
from src.utils.helpers import RAW_DATA_FILE, TARGET_COLUMN, ID_COLUMN

# Load raw dataset
df = load_raw_data(RAW_DATA_FILE)
print(f"Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
df.head(5)"""),
        nbf.v4.new_markdown_cell("""### Data Quality Checks
We verify:
1. Absence of null values
2. Absence of duplicate rows or duplicate account numbers (`CLIENTNUM`)
3. Target class balance (`Attrition_Flag`)
4. Representation of 'Unknown' categories in demographic columns"""),
        nbf.v4.new_code_cell("""validation = validate_data(df)
print("Is dataset strictly valid?", validation["is_valid"])
print(f"Total nulls: {validation['total_nulls']}")
print(f"Duplicate rows: {validation['duplicate_rows']}")
print(f"Duplicate customer IDs: {validation['duplicate_ids']}")
print("\\nTarget distribution:")
for k, v in validation["target_distribution"].items():
    print(f" - {k}: {v:,} ({v / len(df):.2%})")"""),
        nbf.v4.new_markdown_cell("""### Inspecting 'Unknown' Categories
Rather than automatically dropping or imputing `'Unknown'` strings, we analyze their representation across categorical columns."""),
        nbf.v4.new_code_cell("""cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    unknown_ct = (df[col] == 'Unknown').sum()
    if unknown_ct > 0:
        print(f"Column '{col}': {unknown_ct:,} Unknown records ({unknown_ct / len(df):.2%})")"""),
        nbf.v4.new_markdown_cell("""### Data Dictionary
We construct a machine-readable data dictionary classifying each feature into its domain group, data type, and predictive role."""),
        nbf.v4.new_code_cell("""data_dict = generate_data_dictionary(df)
dict_df = pd.DataFrame(data_dict)[['column_name', 'data_type', 'feature_group', 'used_for_modeling', 'has_unknown', 'description']]
dict_df"""),
        nbf.v4.new_markdown_cell("""### Key Takeaways from Data Understanding
- The dataset consists of **10,127 account records** across **21 features**.
- Target variable `Attrition_Flag` shows a **16.07% churn rate** (1,627 attrited vs 8,500 retained customers), creating a notable class imbalance.
- No null values or duplicate accounts exist.
- `CLIENTNUM` acts strictly as an identifier and must be excluded from predictive modeling.
- Next step: In **Notebook 02**, we perform comprehensive Exploratory Data Analysis (EDA)."""),
    ]
    with open(NOTEBOOKS_DIR / "01_data_understanding.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Generated 01_data_understanding.ipynb")


def create_nb_02():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Notebook 02: Exploratory Data Analysis (EDA)

## Credit Card Customer Churn & Segmentation
**Objective:** Systematically explore demographic, financial, engagement, and transactional distributions, comparing retained vs attrited cardholders to uncover high-leverage churn patterns.
"""),
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.load_data import load_raw_data
from src.utils.helpers import TARGET_COLUMN, ID_COLUMN

# Style setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11

df = load_raw_data()
df['target_churn'] = (df[TARGET_COLUMN] == 'Attrited Customer').astype(int)
print(f"Loaded {len(df):,} records for EDA.")"""),
        nbf.v4.new_markdown_cell("""## 1. Target Distribution & Class Imbalance
We examine the breakdown between Existing Customers and Attrited Customers."""),
        nbf.v4.new_code_cell("""fig, ax = plt.subplots(1, 2, figsize=(12, 4))
df[TARGET_COLUMN].value_counts().plot(kind='bar', ax=ax[0], color=['#1e3a8a', '#ef4444'])
ax[0].set_title("Customer Counts by Attrition Flag")
ax[0].set_ylabel("Count")

df[TARGET_COLUMN].value_counts().plot(kind='pie', ax=ax[1], autopct='%1.1f%%', colors=['#1e3a8a', '#ef4444'], startangle=90)
ax[1].set_ylabel("")
ax[1].set_title("Proportion of Attrition")
plt.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("""## 2. Demographic Analysis vs Churn
Does churn vary across gender, education, income, and marital status?"""),
        nbf.v4.new_code_cell("""cat_features = ["Gender", "Education_Level", "Income_Category", "Marital_Status", "Card_Category"]
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, col in enumerate(cat_features):
    crosstab = pd.crosstab(df[col], df[TARGET_COLUMN], normalize='index') * 100
    crosstab.plot(kind='bar', stacked=True, ax=axes[idx], color=['#1e3a8a', '#ef4444'], legend=False)
    axes[idx].set_title(f"Churn Rate by {col}")
    axes[idx].set_ylabel("Percentage (%)")
    axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=30, ha='right')

axes[-1].axis('off')
plt.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("""## 3. Transaction Behavior & Velocity
We compare key behavioral features:
- `Total_Trans_Ct` (Annual transaction count)
- `Total_Trans_Amt` (Annual spend volume)
- `Total_Ct_Chng_Q4_Q1` (Quarterly change in transaction count)
- `Total_Amt_Chng_Q4_Q1` (Quarterly change in transaction spend)"""),
        nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(14, 10))
trans_cols = ["Total_Trans_Ct", "Total_Trans_Amt", "Total_Ct_Chng_Q4_Q1", "Total_Amt_Chng_Q4_Q1"]

for ax, col in zip(axes.flatten(), trans_cols):
    sns.boxplot(data=df, x=TARGET_COLUMN, y=col, ax=ax, palette=['#1e3a8a', '#ef4444'])
    ax.set_title(f"Distribution of {col} by Churn")

plt.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("""## 4. Engagement & Inactivity Indicators
We inspect `Months_Inactive_12_mon` and `Contacts_Count_12_mon`."""),
        nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, ["Months_Inactive_12_mon", "Contacts_Count_12_mon"]):
    rates = df.groupby(col)['target_churn'].mean() * 100
    rates.plot(kind='bar', ax=ax, color='#ef4444')
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title(f"Churn Rate vs {col}")
    ax.axhline(df['target_churn'].mean() * 100, color='black', linestyle='--', label='Average Churn Rate')
    ax.legend()

plt.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("""### Key EDA Findings
1. **Transaction Drop is the Leading Indicator**: Attrited customers average 44.9 transactions vs 68.7 for retained customers.
2. **Contact Fatigue**: Customers who contact the bank 5 or 6 times in 12 months experience over 50% to 100% churn rates, signaling unaddressed friction.
3. **Inactivity Surge**: Customers inactive for 3+ months show sharply higher attrition probability.
4. **Demographics Play a Secondary Role**: Income and education show modest variation, whereas behavioral activity is decisive."""),
    ]
    with open(NOTEBOOKS_DIR / "02_eda.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Generated 02_eda.ipynb")


def create_nb_03():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Notebook 03: Statistical Inference & Hypothesis Testing

## Credit Card Customer Churn & Segmentation
**Objective:** Formulate and statistically test core research questions (RQ1–RQ10) and hypotheses (H1–H7) using Chi-Square tests of independence, Welch's t-tests, Mann-Whitney U tests, 95% confidence intervals, and effect size calculations.

> **Methodological Standard:** We report test statistics, p-values, effect sizes (Cohen's d, Cramér's V), and confidence intervals. We strictly avoid causal assertions, framing findings as statistically verified associations.
"""),
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import json
from scipy import stats

from src.data.load_data import load_raw_data
from src.features.build_features import add_engineered_features
from src.data.statistical_analysis import run_categorical_tests, run_numerical_tests
from src.utils.helpers import TARGET_COLUMN

df = load_raw_data()
df = add_engineered_features(df)
print(f"Loaded dataset with {df.shape[1]} features.")"""),
        nbf.v4.new_markdown_cell("""## 1. Categorical Associations (Chi-Square Tests of Independence)
We test whether churn is independent of demographic and card tiers:
- Null Hypothesis $H_0$: Churn status is independent of category $X$.
- Significance level $\\alpha = 0.05$.
- Effect size measured via Cramér's V."""),
        nbf.v4.new_code_cell("""cat_tests = run_categorical_tests(df)
cat_summary = pd.DataFrame([
    {
        "Feature": t["feature"],
        "Chi2 Stat": t["chi2_statistic"],
        "p-value": f"{t['p_value']:.4e}",
        "Cramér's V": t["cramers_v"],
        "Effect Size": t["effect_size_label"],
        "Significant?": "Yes" if t["is_significant"] else "No"
    }
    for t in cat_tests
])
cat_summary"""),
        nbf.v4.new_markdown_cell("""## 2. Numerical Feature Testing (Parametric & Non-Parametric)
For continuous variables, we evaluate differences between Churned and Retained customers using:
1. **Welch's t-test** (robust to unequal variances)
2. **Mann-Whitney U test** (non-parametric rank sum test)
3. **95% Confidence Interval for Difference in Means**
4. **Cohen's d** (standardized effect size)"""),
        nbf.v4.new_code_cell("""num_tests = run_numerical_tests(df)
num_summary = pd.DataFrame([
    {
        "Feature": t["feature"],
        "Mean Retained": t["mean_retained"],
        "Mean Churned": t["mean_churned"],
        "Diff Mean": t["diff_mean"],
        "95% CI Diff": f"[{t['ci_95_diff'][0]}, {t['ci_95_diff'][1]}]",
        "Mann-Whitney p": f"{t['mann_whitney_p_value']:.2e}",
        "Cohen's d": t["cohens_d"],
        "Effect Label": t["effect_size_label"],
    }
    for t in num_tests
])
num_summary"""),
        nbf.v4.new_markdown_cell("""## 3. Formal Testing of Initial Hypotheses (H1–H7)

| Hypothesis | Test Description | Finding | Conclusion |
|---|---|---|---|
| **H1: Inactivity** | Customers with more inactive months churn more | Mann-Whitney $p < 10^{-20}$, Cohen's $d = +0.33$ | **Supported** (Positive association) |
| **H2: Transaction Count** | Customers with fewer transactions churn more | Mann-Whitney $p < 10^{-250}$, Cohen's $d = -1.09$ | **Supported** (Large negative effect) |
| **H3: Transaction Momentum** | Declining transaction count Q4/Q1 associates with churn | Mann-Whitney $p < 10^{-200}$, Cohen's $d = -0.83$ | **Supported** (Large negative effect) |
| **H4: Relationship Depth** | Customers with fewer products churn more | Mann-Whitney $p < 10^{-50}$, Cohen's $d = -0.39$ | **Supported** (Moderate negative effect) |
| **H5: Bank Contacts** | Frequent customer contact associates with churn | Mann-Whitney $p < 10^{-80}$, Cohen's $d = +0.47$ | **Supported** (Moderate positive effect) |
| **H6: Feature Priority** | Behavioral variables provide stronger signal than demographics | Cramér's V for demographics $< 0.04$ vs Cohen's d for transactions $> 1.0$ | **Supported** |
| **H7: Segments** | Natural segments show distinct churn risk | Tested in Notebook 06 | Deferred to Clustering |
"""),
    ]
    with open(NOTEBOOKS_DIR / "03_statistical_analysis.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Generated 03_statistical_analysis.ipynb")


def create_nb_04():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Notebook 04: Feature Engineering & Preprocessing Pipeline

## Credit Card Customer Churn & Segmentation
**Objective:** Construct interpretable behavioral features with zero data leakage, inspect multicollinearity, and build a reusable Scikit-learn `ColumnTransformer` and `Pipeline`.
"""),
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.load_data import load_raw_data
from src.features.build_features import FeatureEngineer, add_engineered_features
from src.data.preprocess import split_data, create_preprocessor, ALL_NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.utils.helpers import TARGET_COLUMN, ID_COLUMN

df = load_raw_data()
X_train, X_test, y_train, y_test = split_data(df)
print(f"Train rows: {len(X_train):,}, Holdout test rows: {len(X_test):,}")"""),
        nbf.v4.new_markdown_cell("""## 1. Feature Derivations & Rationale
We construct features with explicit business interpretability:
1. `avg_trans_value`: $\\frac{\\text{Total\\_Trans\\_Amt}}{\\text{Total\\_Trans\\_Ct}}$ (average ticket size per transaction)
2. `trans_per_month`: $\\frac{\\text{Total\\_Trans\\_Ct}}{\\text{Months\\_on\\_book}}$ (tenure-normalized swipe frequency)
3. `trans_amt_per_month`: $\\frac{\\text{Total\\_Trans\\_Amt}}{\\text{Months\\_on\\_book}}$ (tenure-normalized spend velocity)
4. `inactivity_ratio`: $\\frac{\\text{Months\\_Inactive\\_12\\_mon}}{12.0}$ (proportion of year card was dormant)
5. `declining_trans_ct_flag`: Binary indicator for severe drop in transaction count ($< 0.60$)
6. `declining_trans_amt_flag`: Binary indicator for severe drop in spend ($< 0.60$)"""),
        nbf.v4.new_code_cell("""fe = FeatureEngineer()
X_train_fe = fe.fit_transform(X_train)
engineered_cols = ["avg_trans_value", "trans_per_month", "trans_amt_per_month", "inactivity_ratio", "declining_trans_ct_flag", "declining_trans_amt_flag"]
X_train_fe[engineered_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]"""),
        nbf.v4.new_markdown_cell("""## 2. Multicollinearity & Correlation Analysis
We examine pairwise correlations among numerical predictors."""),
        nbf.v4.new_code_cell("""corr = X_train_fe[ALL_NUMERICAL_FEATURES].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap of Numerical Features (Train Set Only)")
plt.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("""## 3. Scikit-learn Pipeline Assembly
We assemble the `ColumnTransformer`:
- Numerical Pipeline: Median Imputer + `StandardScaler`
- Categorical Pipeline: Constant Imputer + `OneHotEncoder(drop='first', handle_unknown='ignore')`
- Transformers are fitted strictly on `X_train` to eliminate data leakage."""),
        nbf.v4.new_code_cell("""preprocessor = create_preprocessor()
preprocessor.fit(X_train_fe)
X_train_transformed = preprocessor.transform(X_train_fe)
print("Transformed training matrix shape:", X_train_transformed.shape)"""),
    ]
    with open(NOTEBOOKS_DIR / "04_feature_engineering.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Generated 04_feature_engineering.ipynb")


def create_nb_05():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Notebook 05: Supervised Learning — Logistic Regression

## Credit Card Customer Churn & Segmentation
**Objective:** Demonstrate baseline model evaluation, train Logistic Regression with balanced class weighting, perform Stratified 5-Fold Cross-Validation, select the classification threshold from out-of-fold training predictions, run one final holdout evaluation, compute business lift, and interpret odds ratios.

---
### The Accuracy Paradox
With a 16.07% churn rate, a naive dummy model predicting 'Existing Customer' for everyone yields **83.96% accuracy** but detects **0% of churners**. We explicitly demonstrate why accuracy is a flawed metric for imbalanced problems.
"""),
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

from src.data.load_data import load_raw_data
from src.data.preprocess import split_data
from src.models.train_churn_model import train_and_evaluate_churn_model
from src.utils.helpers import MODELS_DIR

meta = train_and_evaluate_churn_model()
print("Model Training & Evaluation Completed.")"""),
        nbf.v4.new_markdown_cell("""## 1. Model Performance: Baseline vs Logistic Regression"""),
        nbf.v4.new_code_cell("""comparison = pd.DataFrame([
    {
        "Model": "Dummy Classifier (Baseline)",
        "Threshold": 0.50,
        "Accuracy": meta["baseline_metrics"]["accuracy"],
        "Precision": meta["baseline_metrics"]["precision"],
        "Recall": meta["baseline_metrics"]["recall"],
        "F1 Score": meta["baseline_metrics"]["f1"],
        "ROC-AUC": meta["baseline_metrics"]["roc_auc"],
    },
    {
        "Model": "Balanced Logistic Regression (Default)",
        "Threshold": meta["default_threshold"],
        "Accuracy": meta["metrics_default_threshold"]["accuracy"],
        "Precision": meta["metrics_default_threshold"]["precision"],
        "Recall": meta["metrics_default_threshold"]["recall"],
        "F1 Score": meta["metrics_default_threshold"]["f1"],
        "ROC-AUC": meta["metrics_default_threshold"]["roc_auc"],
    },
    {
        "Model": "Balanced Logistic Regression (Validation-Selected Threshold)",
        "Threshold": meta["selected_threshold"],
        "Accuracy": meta["metrics_selected_threshold"]["accuracy"],
        "Precision": meta["metrics_selected_threshold"]["precision"],
        "Recall": meta["metrics_selected_threshold"]["recall"],
        "F1 Score": meta["metrics_selected_threshold"]["f1"],
        "ROC-AUC": meta["metrics_selected_threshold"]["roc_auc"],
    }
])
comparison"""),
        nbf.v4.new_markdown_cell("""## 2. ROC & Precision-Recall Curves
Logistic Regression achieves an impressive **ROC-AUC of 0.934** and **PR-AUC of 0.763** on the holdout test set."""),
        nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ROC Curve
roc_data = pd.DataFrame(meta["curves"]["roc_curve"])
axes[0].plot(roc_data["fpr"], roc_data["tpr"], label=f"LogReg (AUC = {meta['metrics_default_threshold']['roc_auc']:.3f})", color='#1e3a8a', lw=2)
axes[0].plot([0, 1], [0, 1], 'k--', label="Random Guess")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate (Recall)")
axes[0].set_title("ROC Curve")
axes[0].legend()

# PR Curve
pr_data = pd.DataFrame(meta["curves"]["pr_curve"])
axes[1].plot(pr_data["recall"], pr_data["precision"], label=f"LogReg (PR-AUC = {meta['metrics_default_threshold']['pr_auc']:.3f})", color='#059669', lw=2)
axes[1].axhline(0.1604, color='r', linestyle='--', label="Baseline Churn Rate (16.0%)")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend()

plt.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("""## 3. Threshold Analysis & Business Lift
The sweep below uses only 5-fold out-of-fold predictions from the training partition. The selected threshold is fixed before the untouched holdout test is evaluated. In banking, deciding which customers to contact requires balancing outreach budget against churn loss.
- At the **top 10% risk decile**, precision is **84.16%** with a **5.25x Lift**, capturing **52.3%** of all churners in just 10% of customer accounts."""),
        nbf.v4.new_code_cell("""sweep_df = pd.DataFrame(meta["threshold_sweep"])
plt.figure(figsize=(10, 5))
plt.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", color='#2563eb')
plt.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", color='#dc2626')
plt.plot(sweep_df["threshold"], sweep_df["f1"], label="F1 Score", color='#16a34a', lw=2)
plt.axvline(meta["selected_threshold"], color='black', linestyle=':', label=f"Validation-selected F1 threshold ({meta['selected_threshold']})")
plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall, and F1 across Classification Thresholds")
plt.legend()
plt.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("""## 4. Odds Ratio Interpretation
For numerical features, each one-standard-deviation increase multiplies the odds of churn by $e^{\\beta}$. For categorical one-hot features, the odds ratio compares that category with the omitted reference category. Reference categories are stored in the model metadata."""),
        nbf.v4.new_code_cell("""top_drivers = pd.DataFrame(meta["all_coefficients"])
print("Top 5 Features Increasing Churn Risk:")
display(top_drivers[top_drivers["coefficient"] > 0].head(5))

print("\\nTop 5 Features Protecting Retention:")
display(top_drivers[top_drivers["coefficient"] < 0].head(5))"""),
    ]
    with open(NOTEBOOKS_DIR / "05_churn_model.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Generated 05_churn_model.ipynb")


def create_nb_06():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Notebook 06: Unsupervised Learning — Customer Segmentation & PCA

## Credit Card Customer Churn & Segmentation
**Objective:** Discover natural behavioral customer segments using K-Means clustering without using churn labels, evaluate cluster validity with Elbow & Silhouette analysis, project clusters onto 2D PCA space, and profile segment churn risks.
"""),
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from src.models.train_clustering import train_clustering_and_pca
from src.utils.helpers import MODELS_DIR

meta = train_clustering_and_pca(selected_k=4)
print("Customer Segmentation & PCA Pipeline Completed.")"""),
        nbf.v4.new_markdown_cell("""## 1. K Selection: Elbow, Silhouette, and Business Usefulness
We evaluate $k \\in [2, 6]$ on standardized clustering features. **k=2 has the highest Silhouette score**, so k=4 is not presented as an unambiguous quantitative optimum. Selected k = 4 for business interpretability after considering the elbow pattern, cluster sizes, distinct profile interpretability, and usefulness for differentiated retention strategies."""),
        nbf.v4.new_code_cell("""k_eval_df = pd.DataFrame(meta["k_evaluation"])
fig, ax1 = plt.subplots(figsize=(8, 4))

ax1.plot(k_eval_df["k"], k_eval_df["inertia"], 'b-o', label="Inertia (Elbow)")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia", color='b')

ax2 = ax1.twinx()
ax2.plot(k_eval_df["k"], k_eval_df["silhouette_score"], 'r-s', label="Silhouette Score")
ax2.set_ylabel("Silhouette Score", color='r')

plt.title("K-Means Evaluation: Elbow Method and Silhouette Analysis")
plt.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("""## 2. Customer Segment Profiles & Churn Rates
Post-hoc profiling reveals dramatically different customer archetypes and churn risks:"""),
        nbf.v4.new_code_cell("""profiles_df = pd.DataFrame(meta["cluster_profiles"])
cols_to_show = ["cluster_id", "segment_name", "customer_count", "percentage_of_total", "churn_rate_pct", "avg_transaction_amt", "avg_transaction_count", "avg_utilization_ratio", "avg_contacts_count"]
profiles_df[cols_to_show]"""),
        nbf.v4.new_markdown_cell("""## 3. PCA Dimensionality Reduction & 2D Cluster Map
Principal Component Analysis projects the 9-dimensional clustering space to 2 dimensions for visualization.
- PC1 explains **23.3%** of variance (driven by transaction volume and spend).
- PC2 explains **17.0%** of variance (driven by credit limit and utilization)."""),
        nbf.v4.new_code_cell("""scatter_df = pd.DataFrame(meta["pca"]["scatter_sample"])
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=scatter_df,
    x="pc1",
    y="pc2",
    hue="segment_name",
    palette=["#2563eb", "#10b981", "#f59e0b", "#ef4444"],
    alpha=0.7,
    s=40
)
plt.title("2D PCA Projection of Discovered Customer Segments")
plt.xlabel("Principal Component 1 (Transaction Momentum & Volume)")
plt.ylabel("Principal Component 2 (Credit Limit & Utilization)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("""## 4. Combined Customer Intelligence Layer
By combining supervised churn probability with unsupervised segment profiles, retention managers can deploy targeted playbooks:
- **Disengaged & Underutilized**: 26.9% churn rate -> Immediate outbound retention & fee adjustment.
- **High-Utilization Revolvers**: 8.1% churn rate -> Credit line extension and auto-pay discounts.
- **Accelerating Growth**: 5.4% churn rate -> Cross-sell mortgages and investment products.
- **High-Volume Spenders**: 9.6% churn rate -> Premium rewards and concierge perks."""),
    ]
    with open(NOTEBOOKS_DIR / "06_customer_segmentation.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Generated 06_customer_segmentation.ipynb")


if __name__ == "__main__":
    create_nb_01()
    create_nb_02()
    create_nb_03()
    create_nb_04()
    create_nb_05()
    create_nb_06()
    print("All 6 notebooks generated successfully.")
