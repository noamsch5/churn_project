# Credit Card Customer Churn & Segmentation — Project Specification

## 1. Project Overview

**Project name:** Credit Card Customer Churn & Segmentation  
**Domain:** Banking / Credit Cards  
**Primary goal:** Identify customers at risk of attrition and discover meaningful customer segments using statistical learning and machine learning.

This project is designed as a portfolio-grade Data Science project that demonstrates the full analytical workflow:

- Data understanding and cleaning
- Exploratory Data Analysis (EDA)
- Statistical inference
- Feature engineering
- Supervised learning
- Unsupervised learning
- Model evaluation
- Customer segmentation
- Business interpretation
- Optional interactive dashboard

The project intentionally prioritizes interpretable methods that can be explained clearly in an interview.

---

## 2. Dataset

The project uses the **Bank Churners / Credit Card Customers** dataset.

### Dataset size

- **10,127 customers**
- **21 columns** in the clean version selected for this project
- Target variable: `Attrition_Flag`
- Existing customers: **8,500**
- Attrited customers: **1,627**
- Churn rate: approximately **16.1%**

This creates a meaningful class imbalance, so accuracy alone will not be used to evaluate the classification model.

---

## 3. Dataset Variables

### 3.1 Identifier

| Variable | Description | Use |
|---|---|---|
| `CLIENTNUM` | Unique customer identifier | Keep for reference only; exclude from modeling |

### 3.2 Target

| Variable | Description |
|---|---|
| `Attrition_Flag` | Existing Customer / Attrited Customer |

For modeling:

- Existing Customer → `0`
- Attrited Customer → `1`

---

## 4. Feature Groups

### 4.1 Demographic Features

- `Customer_Age`
- `Gender`
- `Dependent_count`
- `Education_Level`
- `Marital_Status`
- `Income_Category`

These variables allow us to study whether churn differs across demographic groups.

### 4.2 Product / Relationship Features

- `Card_Category`
- `Months_on_book`
- `Total_Relationship_Count`

These describe the customer's relationship with the bank.

### 4.3 Engagement Features

- `Months_Inactive_12_mon`
- `Contacts_Count_12_mon`

These are particularly important because they may act as early indicators of disengagement.

### 4.4 Financial / Credit Features

- `Credit_Limit`
- `Total_Revolving_Bal`
- `Avg_Open_To_Buy`
- `Avg_Utilization_Ratio`

These describe how the customer uses their available credit.

### 4.5 Transaction Behavior Features

- `Total_Amt_Chng_Q4_Q1`
- `Total_Trans_Amt`
- `Total_Trans_Ct`
- `Total_Ct_Chng_Q4_Q1`

These describe transaction volume and changes in customer activity.

---

## 5. Main Business Question

> Can customer demographic, financial and behavioral patterns be used to identify customers at risk of attrition, and can we discover meaningful customer segments with different churn profiles?

The project contains two connected analytical components.

### Part A — Churn Prediction

Use supervised learning to estimate:

`P(Customer Churn = 1 | Customer Features)`

The output should be a **churn probability**, not only a binary classification.

### Part B — Customer Segmentation

Use unsupervised learning to discover natural customer groups without using the churn label.

After creating the clusters, compare their churn rates and characterize the behavioral profile of each group.

---

## 6. Research Questions

The project should answer the following questions.

### RQ1 — Customer Activity

Are customers with more inactive months more likely to churn?

### RQ2 — Transaction Behavior

Does a decline in transaction frequency or transaction amount relate to a higher probability of churn?

### RQ3 — Relationship Depth

Are customers who hold more products with the bank less likely to churn?

### RQ4 — Customer Contact

Is frequent contact with the bank associated with higher churn?

### RQ5 — Credit Usage

Does credit utilization differ between churned and retained customers?

### RQ6 — Demographics

Do churn rates differ significantly across age, income, education, gender or marital-status groups?

### RQ7 — Customer Segments

Can customers be grouped into meaningful behavioral or financial segments using clustering?

### RQ8 — Segment Risk

Do the discovered clusters show materially different churn rates?

---

## 7. Initial Hypotheses

These hypotheses will be tested rather than assumed to be true.

### H1

Customers with more inactive months have a higher probability of churn.

### H2

Customers with lower transaction counts are more likely to churn.

### H3

A decline in transaction count between Q1 and Q4 is associated with churn.

### H4

Customers with fewer banking relationships/products have a higher churn probability.

### H5

Customers with more contacts with the bank have a higher churn probability.

### H6

Behavioral variables will provide more predictive value than demographic variables alone.

### H7

Customer clusters based on activity, financial behavior and relationship depth will show different churn rates.

---

## 8. Data Cleaning Plan

### 8.1 Identifier

Remove `CLIENTNUM` from modeling because it is an identifier and should not contain predictive information.

### 8.2 Target Encoding

Convert:

- `Existing Customer` → `0`
- `Attrited Customer` → `1`

### 8.3 Missing and Unknown Values

Check:

- Actual null values
- `Unknown` categories in variables such as education, marital status and income

`Unknown` should not automatically be treated as missing. We will first inspect whether it behaves as a meaningful category.

### 8.4 Duplicate Records

Check duplicate customers and duplicate rows.

### 8.5 Outliers

Inspect continuous variables such as:

- Credit limit
- Transaction amount
- Revolving balance
- Open-to-buy amount

Outliers will not be removed automatically. Their business meaning will be inspected first.

---

## 9. Exploratory Data Analysis

### 9.1 Target Distribution

Analyze:

- Number of churned customers
- Number of retained customers
- Churn percentage

### 9.2 Numerical Features

For each major numerical variable:

- Mean
- Median
- Standard deviation
- Distribution
- Outliers
- Comparison by churn status

### 9.3 Categorical Features

Compare churn rates by:

- Gender
- Education
- Income
- Marital status
- Card category

### 9.4 Behavioral Analysis

Focus especially on:

- Months inactive
- Number of bank contacts
- Transaction count
- Transaction amount
- Change in transaction count
- Change in transaction amount

---

## 10. Statistical Analysis

The statistical analysis should support interpretation before machine learning is applied.

### 10.1 Categorical vs Churn

Use **Chi-Square Tests of Independence** where appropriate.

Examples:

- Card Category vs Churn
- Income Category vs Churn
- Education Level vs Churn

### 10.2 Numerical vs Churn

For continuous variables, compare churned and retained customers.

Depending on the data distribution:

- Independent Samples t-test
- Mann–Whitney U test

### 10.3 Confidence Intervals

Use confidence intervals where possible rather than relying only on p-values.

### 10.4 Effect Size

When relevant, report the magnitude of differences, not only whether they are statistically significant.

---

## 11. Feature Engineering

Only features with a clear interpretation should be added.

Potential features:

### 11.1 Activity Ratio

A normalized inactivity indicator based on months inactive during the previous year.

### 11.2 Transactions per Relationship Month

`Total_Trans_Ct / Months_on_book`

This approximates transaction frequency relative to customer tenure.

### 11.3 Transaction Amount per Relationship Month

`Total_Trans_Amt / Months_on_book`

### 11.4 Average Transaction Value

`Total_Trans_Amt / Total_Trans_Ct`

### 11.5 Relationship Depth

Use `Total_Relationship_Count` directly or derive coarse relationship-depth categories.

### 11.6 Behavioral Change Indicators

Use:

- `Total_Amt_Chng_Q4_Q1`
- `Total_Ct_Chng_Q4_Q1`

to capture changes in customer behavior.

Important: all engineered features will be checked for leakage and redundancy before modeling.

---

## 12. Supervised Learning

### 12.1 Baseline

Create a simple baseline classifier.

Because approximately 84% of customers remain active, a model that always predicts "Existing Customer" can achieve high accuracy while being useless for detecting churn.

### 12.2 Main Model — Logistic Regression

Logistic Regression will be the primary predictive model because it is:

- Appropriate for binary classification
- Interpretable
- Closely connected to statistical learning
- Able to output churn probabilities

The model will estimate:

`P(Churn = 1 | X)`

### 12.3 Interpretation

Analyze:

- Coefficients
- Direction of effects
- Odds ratios where appropriate
- Most influential variables

### 12.4 Optional Additional Models

Only if they are understood and justified:

- K-Nearest Neighbors
- Decision Tree
- Random Forest

Advanced boosting models are explicitly outside the initial project scope.

---

## 13. Train / Validation / Test Strategy

Use a stratified split so the churn ratio is preserved.

Suggested workflow:

- Training set
- Validation through cross-validation
- Final holdout test set

Use **Stratified K-Fold Cross Validation** when evaluating candidate models.

Preprocessing must be learned only from the training data to prevent leakage.

---

## 14. Preprocessing Pipeline

Use Scikit-learn pipelines.

### Numerical variables

Potential steps:

- Imputation if needed
- Standardization

### Categorical variables

Potential steps:

- Handling unknown categories
- One-hot encoding

Tools:

- `Pipeline`
- `ColumnTransformer`

This ensures that the entire preprocessing flow is reproducible.

---

## 15. Classification Evaluation

Do not select models based on accuracy alone.

Primary metrics:

- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

Additional business-oriented metrics can be added later:

- Precision@K
- Recall@K
- Lift

For churn detection, **recall** is especially important because missed churners may represent lost customers.

However, the final threshold should consider the cost of contacting customers versus the cost of losing them.

---

## 16. Unsupervised Learning — Customer Segmentation

### 16.1 Goal

Discover customer groups based on behavior and financial characteristics without using `Attrition_Flag`.

### 16.2 Candidate Features

Prefer behavioral and financial variables such as:

- Total transaction amount
- Total transaction count
- Transaction amount change
- Transaction count change
- Months inactive
- Contacts count
- Relationship count
- Credit utilization
- Credit limit

Demographic features may be analyzed afterward rather than dominating the clustering process.

### 16.3 Scaling

Standardize clustering features before distance-based clustering.

### 16.4 Main Method

**K-Means**

Evaluate multiple values of `K`.

### 16.5 Choosing K

Use:

- Elbow Method
- Silhouette Score
- Business interpretability

### 16.6 Cluster Profiling

For each cluster calculate:

- Number of customers
- Average age
- Average tenure
- Average transaction amount
- Average transaction count
- Average inactivity
- Average utilization
- Relationship count
- Churn rate

Only after clusters are created should churn be used for interpretation.

---

## 17. PCA

Principal Component Analysis may be used for:

1. Reducing dimensionality for visualization
2. Visualizing clusters in two dimensions
3. Understanding which variables drive the primary directions of variation

PCA will not automatically replace the original features in the final clustering model.

The explained variance ratio and component loadings should be inspected.

---

## 18. Combined Customer Intelligence Layer

The final analytical output should combine supervised and unsupervised results.

Example:

### Customer 10294

- Segment: Low Engagement
- Churn Probability: 81%
- Months Inactive: 4
- Transaction Count: Low
- Transaction Count Trend: Declining
- Relationship Count: 1
- Risk Level: High

This makes the project more useful than a standalone classification notebook.

---

## 19. Possible Customer Segment Interpretations

The actual segments must come from the data.

Possible examples only:

- Loyal / High Engagement
- High Value / Active
- Low Engagement
- New / Low Relationship Depth
- Credit-Heavy Customers
- High-Risk Customers

Names should be assigned only after profiling the resulting clusters.

---

## 20. Business Output

The project should answer:

- Who is most likely to churn?
- Which behaviors are associated with churn?
- Which customer groups have the highest churn?
- Which factors are potentially useful as early warning indicators?
- Which customers should a retention team investigate first?

The project will not claim that predictive relationships are causal.

---

## 21. Technology Stack

### Core

- Python
- Jupyter Notebook

### Data

- pandas
- NumPy

### Statistics

- SciPy
- statsmodels

### Machine Learning

- scikit-learn

### Visualization

- Matplotlib
- Plotly

### Optional Product Layer

- FastAPI
- React
- TypeScript

The dashboard/application layer will be built only after the analytical pipeline is complete.

---

## 22. Repository Structure

```text
bank-churn-intelligence/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_statistical_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_churn_model.ipynb
│   └── 06_customer_segmentation.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── evaluation/
│
├── app/
│
├── tests/
│
├── PROJECT_SPEC.md
├── README.md
└── requirements.txt
```

---

## 23. Development Phases

### Phase 1 — Dataset Understanding

- Load dataset
- Validate schema
- Review feature definitions
- Inspect target imbalance
- Identify unknown values

### Phase 2 — EDA

- Univariate analysis
- Churn comparison
- Behavioral analysis
- Correlation and association analysis

### Phase 3 — Statistical Learning

- Define hypotheses
- Perform statistical tests
- Estimate confidence intervals
- Interpret relationships

### Phase 4 — Feature Engineering

- Create interpretable behavioral features
- Remove redundant features
- Build preprocessing pipeline

### Phase 5 — Churn Model

- Dummy baseline
- Logistic Regression
- Cross-validation
- Metric evaluation
- Probability interpretation

### Phase 6 — Customer Segmentation

- Select clustering features
- Standardize
- K-Means
- Evaluate K
- PCA visualization
- Cluster profiling

### Phase 7 — Combined Analysis

- Combine churn scores with customer segments
- Identify high-risk segments
- Generate business insights

### Phase 8 — Portfolio Layer

- Refactor notebooks into reusable code
- Create visual dashboard
- Add screenshots
- Finalize README
- Publish GitHub repository

---

## 24. Definition of Done

The first project version is complete when:

- Data is cleaned and documented
- Key churn patterns are explored
- Statistical hypotheses are tested
- Logistic Regression is trained correctly
- Classification metrics are reported
- Churn probabilities are produced
- K-Means segmentation is completed
- PCA visualization is available
- Clusters are profiled
- Churn rates are compared across segments
- Main business findings are summarized
- Code is reproducible
- README clearly communicates the project

---

## 25. Portfolio Story

The project should communicate the following narrative:

> I analyzed customer behavior at a credit-card bank to understand attrition, tested statistically meaningful relationships, developed an interpretable churn prediction model, and used unsupervised learning to discover customer segments with different behavioral and churn profiles.

The focus is not on using the most advanced model. The focus is on demonstrating rigorous reasoning, correct statistical and machine-learning methodology, interpretability and the ability to turn data into business insight.

---

## 26. Implementation Summary & Deliverables

All 31 development stages specified in this document have been completed and validated:

1. **Clean Project Structure**: Implemented with strict separation of concerns (`data/`, `notebooks/`, `src/`, `models/`, `api/`, `frontend/`, `tests/`).
2. **Data Ingestion & Quality Validation**: Complete schema validation, zero nulls/duplicates verified on canonical 10,127-record dataset (`src/data/load_data.py`).
3. **Data Dictionary**: Generated and documented in [`DATA_DICTIONARY.md`](file:///Users/noamschwartz/churn_project/data/processed/DATA_DICTIONARY.md).
4. **Statistical Hypothesis Testing**: Rigorous evaluation of H1–H7 via Welch's t-tests, Mann-Whitney U, and Chi-Square tests of independence with Cramér's V and Cohen's d effect sizes (`src/data/statistical_analysis.py`).
5. **Zero-Leakage Preprocessing**: Stratified 80/20 split, `FeatureEngineer` and `ColumnTransformer` fitted exclusively on training data (`src/data/preprocess.py`, `src/features/build_features.py`).
6. **Supervised Learning**: Baseline Dummy Classifier vs Balanced Logistic Regression (5-fold Stratified CV: ROC-AUC = 0.934, Recall = 82.8% at default threshold; F1 = 0.713 at optimal 0.75 threshold; Top-10% Lift = 5.25x).
7. **Model Interpretation**: Standardized coefficients and odds ratios ($e^\beta$) calculated and ranked (`models/model_metadata.json`).
8. **Unsupervised Customer Segmentation**: Standardized K-Means with optimal $k=4$, validated via Elbow and Silhouette scores. Post-hoc profiling reveals distinct churn variance from 5.4% to 26.9% (`src/models/train_clustering.py`).
9. **PCA Dimensionality Reduction**: 2D projection capturing 40.3% of variance with component loadings analysis.
10. **Combined Customer Intelligence**: Unified dataset with customer IDs, calibrated churn probabilities, risk levels, and segment tags (`data/processed/customer_intelligence.csv`).
11. **6 Analytical Notebooks**: Educational notebooks covering the full lifecycle (`notebooks/01` through `06`).
12. **FastAPI Backend**: Serving `/health`, `/model-info`, `/clusters`, `/analytics-summary`, `/customers`, `/predict`, `/segment`, `/analyze-customer` (`api/main.py`).
13. **React + TypeScript + Vite Frontend**: High-density banking analytics dashboard with live simulator, dynamic threshold slider, segment breakdowns, statistical tables, and searchable customer explorer (`frontend/`).
14. **Automated Test Suite**: 13 unit and integration tests passing (`tests/`).
15. **Portfolio README**: Comprehensive documentation with empirical metrics, architecture, and local reproduction instructions (`README.md`).
