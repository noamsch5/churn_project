/**
 * TypeScript definitions for Churn & Segmentation API data structures.
 */

export interface CustomerInput {
  Customer_Age: number;
  Gender: string;
  Dependent_count: number;
  Education_Level: string;
  Marital_Status: string;
  Income_Category: string;
  Card_Category: string;
  Months_on_book: number;
  Total_Relationship_Count: number;
  Months_Inactive_12_mon: number;
  Contacts_Count_12_mon: number;
  Credit_Limit: number;
  Total_Revolving_Bal: number;
  Avg_Open_To_Buy: number;
  Total_Amt_Chng_Q4_Q1: number;
  Total_Trans_Amt: number;
  Total_Trans_Ct: number;
  Total_Ct_Chng_Q4_Q1: number;
  Avg_Utilization_Ratio: number;
}

export interface PredictResponse {
  churn_probability: number;
  churn_probability_pct: number;
  predicted_churn: number;
  predicted_status: string;
  risk_level: 'Low' | 'Medium' | 'High' | 'Critical';
  threshold_applied: number;
}

export interface SegmentResponse {
  cluster_id: number;
  segment_name: string;
  description: string;
  recommended_strategy: string;
  segment_churn_rate_pct: number;
}

export interface KeyDriver {
  feature: string;
  value: any;
  direction: string;
  message: string;
}

export interface CombinedAnalysisResponse extends PredictResponse {
  cluster_id: number;
  segment_name: string;
  segment_description: string;
  recommended_strategy: string;
  key_drivers: KeyDriver[];
}

export interface CustomerRecord {
  client_num: number;
  actual_status: string;
  churn_probability: number;
  predicted_churn: number;
  risk_level: string;
  cluster_id: number;
  segment_name: string;
  age: number;
  gender: string;
  total_trans_amt: number;
  total_trans_ct: number;
  months_inactive: number;
  contacts_count: number;
  credit_limit: number;
  utilization_ratio: number;
}

export interface CustomerListResponse {
  total_records: number;
  page: number;
  page_size: number;
  total_pages: number;
  customers: CustomerRecord[];
}

export interface ClusterProfile {
  cluster_id: number;
  customer_count: number;
  percentage_of_total: number;
  churn_count: number;
  churn_rate_pct: number;
  avg_age: number;
  avg_months_on_book: number;
  avg_transaction_amt: number;
  avg_transaction_count: number;
  avg_amt_change: number;
  avg_ct_change: number;
  avg_inactive_months: number;
  avg_contacts_count: number;
  avg_relationship_count: number;
  avg_credit_limit: number;
  avg_utilization_ratio: number;
  segment_name: string;
  description: string;
  strategy: string;
}

export interface PCAPoint {
  client_num: number;
  pc1: number;
  pc2: number;
  cluster_id: number;
  segment_name: string;
  churn_prob: number;
  actual_churn: number;
  risk_level: string;
}

export interface AnalyticsSummary {
  eda: {
    dataset_size: number;
    retained_count: number;
    attrited_count: number;
    churn_rate: number;
    target_correlations: { feature: string; correlation: number }[];
    numerical_summaries: Record<string, any>;
    demographics_overview: Record<string, Record<string, number>>;
  };
  statistical_tests: {
    categorical_tests: {
      feature: string;
      test_type: string;
      chi2_statistic: number;
      degrees_of_freedom: number;
      p_value: number;
      cramers_v: number;
      effect_size_label: string;
      is_significant: boolean;
      interpretation: string;
      categories_breakdown: {
        category: string;
        total_customers: number;
        churned_customers: number;
        retained_customers: number;
        churn_rate_pct: number;
      }[];
    }[];
    numerical_tests: {
      feature: string;
      mean_churned: number;
      mean_retained: number;
      median_churned: number;
      median_retained: number;
      diff_mean: number;
      ci_95_diff: [number, number];
      t_statistic: number;
      t_p_value: number;
      mann_whitney_u: number;
      mann_whitney_p_value: number;
      cohens_d: number;
      rank_biserial_correlation: number;
      effect_size_label: string;
      is_significant: boolean;
      interpretation: string;
    }[];
  };
  model_performance: {
    metrics: {
      threshold: number;
      accuracy: number;
      precision: number;
      recall: number;
      f1: number;
      roc_auc: number;
      pr_auc: number;
      confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
      business_metrics: {
        precision_at_10_pct: number;
        recall_at_10_pct: number;
        lift_at_10_pct: number;
      };
    };
    optimized_metrics: {
      threshold: number;
      accuracy: number;
      precision: number;
      recall: number;
      f1: number;
      roc_auc: number;
      pr_auc: number;
      confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
      business_metrics: {
        precision_at_10_pct: number;
        recall_at_10_pct: number;
        lift_at_10_pct: number;
      };
    };
    baseline_metrics: {
      accuracy: number;
      precision: number;
      recall: number;
      f1: number;
      roc_auc: number;
      pr_auc: number;
      confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
    };
    curves: {
      roc_curve: { fpr: number; tpr: number }[];
      pr_curve: { recall: number; precision: number }[];
    };
    threshold_sweep: {
      threshold: number;
      precision: number;
      recall: number;
      f1: number;
      accuracy: number;
    }[];
    selected_threshold: number;
    coefficients: {
      feature: string;
      coefficient: number;
      odds_ratio: number;
      impact: string;
    }[];
  };
  segmentation: {
    k_evaluation: { k: number; inertia: number; silhouette_score: number }[];
    optimal_k: number;
    clustering_features: string[];
    cluster_profiles: ClusterProfile[];
    pca: {
      explained_variance_ratio: number[];
      total_explained_variance: number;
      loadings: { feature: string; PC1_loading: number; PC2_loading: number }[];
      scatter_sample: PCAPoint[];
    };
  };
}
