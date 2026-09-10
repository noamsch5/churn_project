import React, { useState } from 'react';
import {
  CheckCircle2,
  XCircle,
  Info,
} from 'lucide-react';
import type { AnalyticsSummary } from '../types/api';

interface InsightsViewProps {
  summary: AnalyticsSummary | null;
}

export const InsightsView: React.FC<InsightsViewProps> = ({ summary }) => {
  const [activeTab, setActiveTab] = useState<'numerical' | 'categorical'>('numerical');

  if (!summary) {
    return <div className="card-box">Loading statistical hypothesis tests...</div>;
  }

  const { statistical_tests } = summary;
  return (
    <div>
      <div className="page-intro">
        <h2 className="page-title">Statistical Inference & Hypothesis Testing</h2>
        <p className="page-description">
          Formal hypothesis tests answering Research Questions (RQ1–RQ10) and Hypotheses (H1–H7) using
          Welch's t-tests, Mann-Whitney U rank-sum tests, Chi-Square tests of independence, 95% confidence intervals, and effect size metrics.
        </p>
      </div>

      {/* Caution Alert: Non-Causality */}
      <div
        style={{
          backgroundColor: '#fffbeb',
          border: '1px solid #fde68a',
          borderRadius: 8,
          padding: '1rem 1.25rem',
          marginBottom: '1.5rem',
          display: 'flex',
          gap: '1rem',
          alignItems: 'flex-start',
        }}
      >
        <Info size={24} color="#d97706" style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#b45309', marginBottom: '0.25rem' }}>
            Methodological Standard: Statistical Association vs. Causal Attribution
          </h4>
          <p style={{ fontSize: '0.85rem', color: '#92400e', lineHeight: 1.5 }}>
            All reported p-values and effect sizes quantify empirical associations observed in historical observational data.
            A statistically significant difference does <em>not</em> prove that variable $X$ causes customer attrition.
            Interventions should be treated as retention hypotheses subject to randomized A/B experimentation.
          </p>
          <p style={{ fontSize: '0.85rem', color: '#92400e', lineHeight: 1.5, marginTop: '0.5rem' }}>
            Multiple testing is controlled across {statistical_tests.multiple_testing.family_size} primary tests using the
            {' '}{statistical_tests.multiple_testing.method} at α = {statistical_tests.multiple_testing.alpha}.
          </p>
        </div>
      </div>

      {/* Test Tabs */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.25rem' }}>
        <button
          className={`btn-secondary ${activeTab === 'numerical' ? 'active' : ''}`}
          style={{
            backgroundColor: activeTab === 'numerical' ? '#2563eb' : '#ffffff',
            color: activeTab === 'numerical' ? '#ffffff' : '#0f172a',
            borderColor: activeTab === 'numerical' ? '#2563eb' : '#cbd5e1',
            fontWeight: 600,
          }}
          onClick={() => setActiveTab('numerical')}
        >
          Continuous Variables (Mann-Whitney U & Welch's t-test)
        </button>
        <button
          className={`btn-secondary ${activeTab === 'categorical' ? 'active' : ''}`}
          style={{
            backgroundColor: activeTab === 'categorical' ? '#2563eb' : '#ffffff',
            color: activeTab === 'categorical' ? '#ffffff' : '#0f172a',
            borderColor: activeTab === 'categorical' ? '#2563eb' : '#cbd5e1',
            fontWeight: 600,
          }}
          onClick={() => setActiveTab('categorical')}
        >
          Categorical Variables (Chi-Square Independence)
        </button>
      </div>

      {/* Numerical Hypothesis Tests Table */}
      {activeTab === 'numerical' && (
        <div className="card-box">
          <div className="card-header">
            <div>
              <h3 className="card-title">Two-Sample Difference Tests: Retained vs Attrited Customers</h3>
              <p className="card-subtitle">
                Sorted by Cohen's d effect size magnitude. Evaluates difference in location and distribution.
              </p>
            </div>
          </div>
          <div className="data-table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Retained Mean</th>
                  <th>Churned Mean</th>
                  <th>Mean Diff</th>
                  <th>95% CI of Difference</th>
                  <th>Mann-Whitney p-val</th>
                  <th>Cohen's d</th>
                  <th>Effect Magnitude</th>
                  <th>Scientific Interpretation</th>
                </tr>
              </thead>
              <tbody>
                {statistical_tests.numerical_tests.map((t, i) => (
                  <tr key={i}>
                    <td><strong>{t.feature}</strong></td>
                    <td className="num-cell">{t.mean_retained.toLocaleString()}</td>
                    <td className="num-cell">{t.mean_churned.toLocaleString()}</td>
                    <td className="num-cell" style={{ color: t.diff_mean < 0 ? '#ef4444' : '#10b981' }}>
                      {t.diff_mean > 0 ? `+${t.diff_mean.toLocaleString()}` : t.diff_mean.toLocaleString()}
                    </td>
                    <td className="num-cell" style={{ fontSize: '0.775rem' }}>
                      [{t.ci_95_diff[0]}, {t.ci_95_diff[1]}]
                    </td>
                    <td className="num-cell" style={{ fontWeight: 600 }}>
                      {t.mann_whitney_p_value < 1e-4 ? '< 0.0001' : t.mann_whitney_p_value.toFixed(4)}
                    </td>
                    <td className="num-cell" style={{ fontWeight: 700 }}>
                      {t.cohens_d.toFixed(2)}
                    </td>
                    <td>
                      <span
                        className={`badge ${
                          t.effect_size_label === 'Large'
                            ? 'badge-critical'
                            : t.effect_size_label === 'Medium'
                            ? 'badge-high'
                            : t.effect_size_label === 'Small'
                            ? 'badge-medium'
                            : 'badge-low'
                        }`}
                      >
                        {t.effect_size_label}
                      </span>
                    </td>
                    <td style={{ fontSize: '0.8rem', color: '#475569', minWidth: '240px' }}>
                      {t.interpretation}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Categorical Chi-Square Tests Table */}
      {activeTab === 'categorical' && (
        <div className="card-box">
          <div className="card-header">
            <div>
              <h3 className="card-title">Chi-Square Tests of Independence</h3>
              <p className="card-subtitle">
                Testing null hypothesis of independence between demographic/card attributes and customer attrition.
              </p>
            </div>
          </div>
          <div className="data-table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Categorical Feature</th>
                  <th>$\chi^2$ Statistic</th>
                  <th>Degrees of Freedom</th>
                  <th>p-value</th>
                  <th>Cramér's V</th>
                  <th>Effect Size</th>
                  <th>Statistically Significant?</th>
                  <th>Scientific Interpretation</th>
                </tr>
              </thead>
              <tbody>
                {statistical_tests.categorical_tests.map((t, i) => (
                  <tr key={i}>
                    <td><strong>{t.feature}</strong></td>
                    <td className="num-cell">{t.chi2_statistic.toFixed(2)}</td>
                    <td className="num-cell">{t.degrees_of_freedom}</td>
                    <td className="num-cell" style={{ fontWeight: 600 }}>
                      {t.p_value < 1e-4 ? '< 0.0001' : t.p_value.toFixed(4)}
                    </td>
                    <td className="num-cell" style={{ fontWeight: 700 }}>{t.cramers_v.toFixed(3)}</td>
                    <td>
                      <span className="badge badge-low">{t.effect_size_label}</span>
                    </td>
                    <td>
                      {t.is_significant ? (
                        <span style={{ color: '#10b981', display: 'flex', alignItems: 'center', gap: '0.25rem', fontWeight: 600 }}>
                          <CheckCircle2 size={16} /> Significant
                        </span>
                      ) : (
                        <span style={{ color: '#64748b', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                          <XCircle size={16} /> Not Significant
                        </span>
                      )}
                    </td>
                    <td style={{ fontSize: '0.8rem', color: '#475569', minWidth: '260px' }}>
                      {t.interpretation}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
