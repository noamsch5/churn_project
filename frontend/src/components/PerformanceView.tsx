import React, { useState } from 'react';
import {
  Sliders,
  AlertCircle,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
} from 'recharts';
import type { AnalyticsSummary } from '../types/api';

interface PerformanceViewProps {
  summary: AnalyticsSummary | null;
}

export const PerformanceView: React.FC<PerformanceViewProps> = ({ summary }) => {
  const [activeThreshold, setActiveThreshold] = useState<number>(
    summary?.model_performance.selected_threshold ?? 0.5
  );

  if (!summary) {
    return <div className="card-box">Loading model evaluation intelligence...</div>;
  }

  const { model_performance } = summary;
  const { metrics, selected_threshold_metrics, curves, threshold_sweep, selected_threshold, coefficients, threshold_selection, odds_ratio_notes } =
    model_performance;

  // Find sweep row matching active slider
  const currentThresholdData =
    threshold_sweep.find((s) => Math.abs(s.threshold - activeThreshold) < 0.015) || {
      threshold: activeThreshold,
      precision: selected_threshold_metrics.precision,
      recall: selected_threshold_metrics.recall,
      f1: selected_threshold_metrics.f1,
      accuracy: selected_threshold_metrics.accuracy,
    };

  // Top 10 positive & negative coefficients for chart
  const topDrivers = coefficients.slice(0, 8);
  const topProtectors = coefficients.slice(-8).reverse();

  return (
    <div>
      <div className="page-intro">
        <h2 className="page-title">Model Evaluation & Interpretability</h2>
        <p className="page-description">
          Rigorous validation of regularized Logistic Regression against Dummy Baselines, featuring
          ROC/PR curves, dynamic decision threshold simulation, top-decile lift, and odds-ratio analysis.
        </p>
      </div>

      {/* Accuracy Paradox Callout */}
      <div
        style={{
          backgroundColor: '#eff6ff',
          border: '1px solid #bfdbfe',
          borderRadius: 8,
          padding: '1rem 1.25rem',
          marginBottom: '1.5rem',
          display: 'flex',
          gap: '1rem',
          alignItems: 'flex-start',
        }}
      >
        <AlertCircle size={24} color="#2563eb" style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#1e40af', marginBottom: '0.25rem' }}>
            The Accuracy Paradox in Imbalanced Financial Attrition
          </h4>
          <p style={{ fontSize: '0.85rem', color: '#1e3a8a', lineHeight: 1.5 }}>
            Because 83.96% of accounts remain active, a naive Dummy Model predicting "Existing Customer" for every cardholder
            achieves <strong>83.96% accuracy</strong> while catching <strong>zero churners (0% Recall, F1 = 0)</strong>.
            Our class-weighted Logistic Regression achieves an <strong>ROC-AUC of {metrics.roc_auc.toFixed(3)}</strong>,
            identifying <strong>{(selected_threshold_metrics.recall * 100).toFixed(1)}% of holdout churners</strong> at the validation-selected threshold, and delivering a <strong>{selected_threshold_metrics.business_metrics.lift_at_10_pct}x Lift</strong> in the top decile.
          </p>
        </div>
      </div>

      {/* Model Benchmark Comparison Table */}
      <div className="card-box">
        <h3 className="card-title" style={{ marginBottom: '1rem' }}>
          Model Benchmark: Baseline vs Balanced Logistic Regression
        </h3>
        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Threshold</th>
                <th>ROC-AUC</th>
                <th>Recall (Sensitivity)</th>
                <th>Precision</th>
                <th>F1 Score</th>
                <th>Accuracy</th>
                <th>Top-10% Lift</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Dummy Baseline (Most Frequent)</strong></td>
                <td className="num-cell">0.50</td>
                <td className="num-cell">0.500</td>
                <td className="num-cell" style={{ color: '#ef4444' }}>0.0%</td>
                <td className="num-cell" style={{ color: '#ef4444' }}>0.0%</td>
                <td className="num-cell">0.000</td>
                <td className="num-cell">84.0%</td>
                <td className="num-cell">1.00x</td>
              </tr>
              <tr>
                <td><strong>Logistic Regression (Default Threshold)</strong></td>
                <td className="num-cell">0.50</td>
                <td className="num-cell" style={{ color: '#10b981', fontWeight: 700 }}>
                  {metrics.roc_auc.toFixed(3)}
                </td>
                <td className="num-cell" style={{ color: '#10b981', fontWeight: 700 }}>
                  {(metrics.recall * 100).toFixed(1)}%
                </td>
                <td className="num-cell">{(metrics.precision * 100).toFixed(1)}%</td>
                <td className="num-cell">{metrics.f1.toFixed(3)}</td>
                <td className="num-cell">{(metrics.accuracy * 100).toFixed(1)}%</td>
                <td className="num-cell" style={{ fontWeight: 700, color: '#2563eb' }}>
                  {metrics.business_metrics.lift_at_10_pct}x
                </td>
              </tr>
              <tr style={{ backgroundColor: '#f0fdf4' }}>
                <td><strong>Logistic Regression (Validation-Selected F1 Threshold)</strong></td>
                <td className="num-cell" style={{ fontWeight: 700 }}>{selected_threshold.toFixed(2)}</td>
                <td className="num-cell">{selected_threshold_metrics.roc_auc.toFixed(3)}</td>
                <td className="num-cell">{(selected_threshold_metrics.recall * 100).toFixed(1)}%</td>
                <td className="num-cell" style={{ color: '#10b981', fontWeight: 700 }}>
                  {(selected_threshold_metrics.precision * 100).toFixed(1)}%
                </td>
                <td className="num-cell" style={{ color: '#10b981', fontWeight: 700 }}>
                  {selected_threshold_metrics.f1.toFixed(3)}
                </td>
                <td className="num-cell" style={{ fontWeight: 700 }}>
                  {(selected_threshold_metrics.accuracy * 100).toFixed(1)}%
                </td>
                <td className="num-cell" style={{ fontWeight: 700, color: '#2563eb' }}>
                  {selected_threshold_metrics.business_metrics.lift_at_10_pct}x
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Interactive Threshold Slider Simulator */}
      <div className="card-box">
        <div className="card-header">
          <div>
            <h3 className="card-title">Interactive Classification Threshold Simulator</h3>
            <p className="card-subtitle">
              Training cross-validation sweep: adjust the threshold to inspect the precision/recall trade-off without using holdout labels.
            </p>
          </div>
          <span className="badge badge-medium">
            Active Threshold: {(activeThreshold * 100).toFixed(0)}%
          </span>
        </div>
        <p style={{ fontSize: '0.78rem', color: '#64748b', marginTop: '0.75rem' }}>
          Selected from {threshold_selection.source}. The final reported model metrics are evaluated once on the untouched holdout test set.
        </p>

        <div className="threshold-slider-container">
          <Sliders size={20} color="#2563eb" />
          <input
            type="range"
            min="0.05"
            max="0.95"
            step="0.01"
            className="threshold-slider"
            value={activeThreshold}
            onChange={(e) => setActiveThreshold(parseFloat(e.target.value))}
          />
          <span style={{ fontFamily: 'monospace', fontWeight: 700, minWidth: '45px' }}>
            {activeThreshold.toFixed(2)}
          </span>
        </div>

        {/* Dynamic Metric Cards */}
        <div className="kpi-grid" style={{ marginBottom: 0 }}>
          <div className="kpi-card accent-emerald">
            <div className="kpi-label">Precision</div>
            <div className="kpi-value">{(currentThresholdData.precision * 100).toFixed(1)}%</div>
            <div className="kpi-subtext">Accuracy of flagged accounts</div>
          </div>
          <div className="kpi-card accent-red">
            <div className="kpi-label">Recall (Sensitivity)</div>
            <div className="kpi-value">{(currentThresholdData.recall * 100).toFixed(1)}%</div>
            <div className="kpi-subtext">% of churners captured</div>
          </div>
          <div className="kpi-card accent-emerald">
            <div className="kpi-label">F1 Score</div>
            <div className="kpi-value">{currentThresholdData.f1.toFixed(3)}</div>
            <div className="kpi-subtext">Harmonic mean balance</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Overall Accuracy</div>
            <div className="kpi-value">{(currentThresholdData.accuracy * 100).toFixed(1)}%</div>
            <div className="kpi-subtext">Overall portfolio concordance</div>
          </div>
        </div>
      </div>

      {/* ROC and PR Curves Grid */}
      <div className="grid-2">
        {/* ROC Curve */}
        <div className="card-box">
          <h3 className="card-title">ROC Curve (Receiver Operating Characteristic)</h3>
          <p className="card-subtitle" style={{ marginBottom: '1rem' }}>
            ROC-AUC = {metrics.roc_auc.toFixed(3)} indicates exceptional discriminatory ability.
          </p>
          <div style={{ height: 280, width: '100%' }}>
            <ResponsiveContainer>
              <LineChart data={curves.roc_curve} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <XAxis dataKey="fpr" tick={{ fontSize: 11 }} label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -10, fontSize: 11 }} />
                <YAxis dataKey="tpr" tick={{ fontSize: 11 }} label={{ value: 'True Positive Rate (Recall)', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                <Tooltip formatter={(val: any) => [Number(val).toFixed(3), 'Rate']} />
                <Line type="monotone" dataKey="tpr" stroke="#2563eb" strokeWidth={2} dot={false} name="Logistic Regression" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* PR Curve */}
        <div className="card-box">
          <h3 className="card-title">Precision-Recall Curve</h3>
          <p className="card-subtitle" style={{ marginBottom: '1rem' }}>
            PR-AUC = {metrics.pr_auc.toFixed(3)} (Benchmark random guessing = 0.160).
          </p>
          <div style={{ height: 280, width: '100%' }}>
            <ResponsiveContainer>
              <LineChart data={curves.pr_curve} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <XAxis dataKey="recall" tick={{ fontSize: 11 }} label={{ value: 'Recall', position: 'insideBottom', offset: -10, fontSize: 11 }} />
                <YAxis dataKey="precision" tick={{ fontSize: 11 }} label={{ value: 'Precision', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                <Tooltip formatter={(val: any) => [Number(val).toFixed(3), 'Score']} />
                <Line type="monotone" dataKey="precision" stroke="#10b981" strokeWidth={2} dot={false} name="Precision vs Recall" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Feature Coefficients / Odds Ratios */}
      <div className="card-box">
        <h3 className="card-title">Model Coefficients & Odds Ratios ($e^\beta$)</h3>
        <p className="card-subtitle" style={{ marginBottom: '1.25rem' }}>
          {odds_ratio_notes.numerical_features} {odds_ratio_notes.categorical_features}
        </p>

        <div className="grid-2">
          <div>
            <h4 style={{ fontSize: '0.85rem', color: '#ef4444', fontWeight: 700, marginBottom: '0.5rem' }}>
              ▲ Largest Positive Associations with Churn Odds
            </h4>
            <div className="data-table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Coeff ($\beta$)</th>
                    <th>Odds Ratio ($e^\beta$)</th>
                  </tr>
                </thead>
                <tbody>
                  {topDrivers.map((d, i) => (
                    <tr key={i}>
                      <td><strong>{d.feature}</strong></td>
                      <td className="num-cell" style={{ color: '#ef4444' }}>+{d.coefficient.toFixed(3)}</td>
                      <td className="num-cell"><strong>{d.odds_ratio.toFixed(2)}x</strong></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div>
            <h4 style={{ fontSize: '0.85rem', color: '#10b981', fontWeight: 700, marginBottom: '0.5rem' }}>
              ▼ Largest Negative Associations with Churn Odds
            </h4>
            <div className="data-table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Coeff ($\beta$)</th>
                    <th>Odds Ratio ($e^\beta$)</th>
                  </tr>
                </thead>
                <tbody>
                  {topProtectors.map((d, i) => (
                    <tr key={i}>
                      <td><strong>{d.feature}</strong></td>
                      <td className="num-cell" style={{ color: '#10b981' }}>{d.coefficient.toFixed(3)}</td>
                      <td className="num-cell"><strong>{d.odds_ratio.toFixed(2)}x</strong></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
      <div className="card-box">
        <h3 className="card-title">Categorical Reference Categories</h3>
        <p className="card-subtitle">
          {Object.entries(odds_ratio_notes.categorical_reference_categories)
            .map(([feature, category]) => `${feature}: ${category}`)
            .join(' · ')}
        </p>
      </div>
    </div>
  );
};
