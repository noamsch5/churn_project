import React from 'react';
import {
  Users,
  TrendingDown,
  Activity,
  Award,
  Layers,
  Sparkles,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ScatterChart,
  Scatter,
  Cell,
  PieChart,
  Pie,
} from 'recharts';
import type { AnalyticsSummary } from '../types/api';

interface OverviewViewProps {
  summary: AnalyticsSummary | null;
  onNavigate: (tab: string) => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({ summary, onNavigate }) => {
  if (!summary) {
    return <div className="card-box">Loading executive intelligence summary...</div>;
  }

  const { eda, model_performance, segmentation } = summary;

  // Segment churn data
  const segmentData = segmentation.cluster_profiles.map((p) => ({
    name: p.segment_name,
    churn_rate: p.churn_rate_pct,
    customers: p.customer_count,
    id: p.cluster_id,
  }));
  const highestRiskSegment = segmentation.cluster_profiles.reduce((highest, profile) =>
    profile.churn_rate_pct > highest.churn_rate_pct ? profile : highest
  );
  const highestRiskChurnShare = highestRiskSegment.churn_count / eda.attrited_count;

  // Target split data
  const targetSplit = [
    { name: 'Retained Customers', value: eda.retained_count, color: '#1e3a8a' },
    { name: 'Attrited Customers', value: eda.attrited_count, color: '#ef4444' },
  ];

  // PCA colors
  const clusterColors = ['#2563eb', '#10b981', '#ef4444', '#8b5cf6'];

  return (
    <div>
      <div className="page-intro">
        <h2 className="page-title">Executive Portfolio Overview</h2>
        <p className="page-description">
          End-to-end analytical intelligence synthesizing statistical hypothesis testing,
          regularized logistic regression, and unsupervised K-Means segmentation.
        </p>
      </div>

      {/* KPI Cards */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">
            <span>Total Accounts</span>
            <Users size={16} color="#2563eb" />
          </div>
          <div className="kpi-value">{eda.dataset_size.toLocaleString()}</div>
          <div className="kpi-subtext">Active cardholder portfolio</div>
        </div>

        <div className="kpi-card accent-red">
          <div className="kpi-label">
            <span>Portfolio Churn Rate</span>
            <TrendingDown size={16} color="#ef4444" />
          </div>
          <div className="kpi-value">{(eda.churn_rate * 100).toFixed(1)}%</div>
          <div className="kpi-subtext">{eda.attrited_count.toLocaleString()} lost cardholders</div>
        </div>

        <div className="kpi-card accent-emerald">
          <div className="kpi-label">
            <span>Model ROC-AUC</span>
            <Award size={16} color="#10b981" />
          </div>
          <div className="kpi-value">{model_performance.metrics.roc_auc.toFixed(3)}</div>
          <div className="kpi-subtext">Balanced Logistic Regression</div>
        </div>

        <div className="kpi-card accent-emerald">
          <div className="kpi-label">
            <span>Holdout Recall</span>
            <Activity size={16} color="#10b981" />
          </div>
          <div className="kpi-value">{(model_performance.selected_threshold_metrics.recall * 100).toFixed(1)}%</div>
          <div className="kpi-subtext">
            At validation-selected threshold {model_performance.selected_threshold.toFixed(2)}
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">
            <span>Top-Decile Lift</span>
            <Sparkles size={16} color="#2563eb" />
          </div>
          <div className="kpi-value">{model_performance.metrics.business_metrics.lift_at_10_pct}x</div>
          <div className="kpi-subtext">
            {model_performance.metrics.business_metrics.lift_at_10_pct}x random targeting efficiency
          </div>
        </div>

        <div className="kpi-card accent-amber">
          <div className="kpi-label">
            <span>Discovered Segments</span>
            <Layers size={16} color="#f59e0b" />
          </div>
          <div className="kpi-value">{segmentation.selected_k}</div>
          <div className="kpi-subtext">K-Means + 2D PCA profiling</div>
        </div>
      </div>

      {/* Primary Analytics Grid */}
      <div className="grid-2">
        {/* Churn Rate by Segment */}
        <div className="card-box">
          <div className="card-header">
            <div>
              <h3 className="card-title">Attrition Rate Across Discovered Segments</h3>
              <p className="card-subtitle">
                Clustering was performed without churn labels; post-hoc profiling demonstrates sharp risk divergence.
              </p>
            </div>
            <button className="btn-secondary" onClick={() => onNavigate('segments')}>
              View Segments
            </button>
          </div>
          <div style={{ height: 260, width: '100%' }}>
            <ResponsiveContainer>
              <BarChart data={segmentData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 11 }}
                  interval={0}
                  angle={-15}
                  textAnchor="end"
                />
                <YAxis unit="%" domain={[0, 30]} />
                <Tooltip
                  formatter={(val: any) => [`${Number(val).toFixed(1)}%`, 'Churn Rate']}
                  contentStyle={{ backgroundColor: '#ffffff', borderRadius: 8, border: '1px solid #cbd5e1' }}
                />
                <Bar dataKey="churn_rate" radius={[4, 4, 0, 0]}>
                  {segmentData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.churn_rate > 20 ? '#ef4444' : entry.churn_rate < 8 ? '#10b981' : '#2563eb'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p style={{ fontSize: '0.8rem', color: '#64748b', marginTop: '0.5rem' }}>
            <strong>Observed pattern:</strong> {highestRiskSegment.segment_name} (Cluster {highestRiskSegment.cluster_id}) has a{' '}
            <strong>{highestRiskSegment.churn_rate_pct.toFixed(1)}% churn rate</strong> and contains{' '}
            {(highestRiskChurnShare * 100).toFixed(1)}% of observed portfolio churners.
          </p>
        </div>

        {/* Portfolio Target Class Breakdown */}
        <div className="card-box">
          <div className="card-header">
            <div>
              <h3 className="card-title">Portfolio Attrition Imbalance</h3>
              <p className="card-subtitle">
                16.07% churn rate introduces class imbalance, making standard accuracy deceptive.
              </p>
            </div>
            <button className="btn-secondary" onClick={() => onNavigate('performance')}>
              Model Proof
            </button>
          </div>
          <div className="portfolio-split">
            <div className="portfolio-chart">
              <ResponsiveContainer>
                <PieChart>
                  <Pie
                    data={targetSplit}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={85}
                    paddingAngle={3}
                  >
                    {targetSplit.map((entry, idx) => (
                      <Cell key={idx} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(val: any) => [Number(val).toLocaleString(), 'Customers']} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="portfolio-legend">
              <div style={{ marginBottom: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                  <div style={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: '#1e3a8a' }} />
                  <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Retained</span>
                </div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700 }}>
                  {eda.retained_count.toLocaleString()}{' '}
                  <span style={{ fontSize: '0.8rem', color: '#64748b', fontWeight: 400 }}>
                    ({((eda.retained_count / eda.dataset_size) * 100).toFixed(1)}%)
                  </span>
                </div>
              </div>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                  <div style={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: '#ef4444' }} />
                  <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Attrited</span>
                </div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700 }}>
                  {eda.attrited_count.toLocaleString()}{' '}
                  <span style={{ fontSize: '0.8rem', color: '#ef4444', fontWeight: 400 }}>
                    ({((eda.attrited_count / eda.dataset_size) * 100).toFixed(1)}%)
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 2D PCA Cluster Map */}
      <div className="card-box">
        <div className="card-header">
          <div>
            <h3 className="card-title">2D PCA Customer Topology & Cluster Scatter</h3>
            <p className="card-subtitle">
              Projection of 9-dimensional customer behavior onto PC1 (Transaction Volume) and PC2 (Credit Limit & Utilization).
            </p>
          </div>
          <button className="btn-secondary" onClick={() => onNavigate('explorer')}>
            Explore Customers
          </button>
        </div>
        <div style={{ height: 360, width: '100%' }}>
          <ResponsiveContainer>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
              <XAxis
                type="number"
                dataKey="pc1"
                name="PC1 (Transaction Momentum)"
                tick={{ fontSize: 11 }}
                label={{ value: 'PC1 (Transaction Count & Spend Volume)', position: 'insideBottom', offset: -10, fontSize: 12 }}
              />
              <YAxis
                type="number"
                dataKey="pc2"
                name="PC2 (Credit Limit & Utilization)"
                tick={{ fontSize: 11 }}
                label={{ value: 'PC2 (Credit Limit & Revolving Bal)', angle: -90, position: 'insideLeft', fontSize: 12 }}
              />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div style={{ backgroundColor: '#ffffff', padding: '0.75rem', borderRadius: 8, border: '1px solid #cbd5e1', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
                        <p style={{ fontWeight: 700, fontSize: '0.85rem' }}>Account #{data.client_num}</p>
                        <p style={{ fontSize: '0.8rem', color: '#475569' }}>Segment: <strong>{data.segment_name}</strong></p>
                        <p style={{ fontSize: '0.8rem', color: '#475569' }}>Churn Prob: <strong>{(data.churn_prob * 100).toFixed(1)}%</strong></p>
                        <p style={{ fontSize: '0.8rem', color: data.actual_churn ? '#ef4444' : '#10b981' }}>
                          Status: <strong>{data.actual_churn ? 'Attrited' : 'Retained'}</strong>
                        </p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Scatter name="Customers" data={segmentation.pca.scatter_sample} fill="#8884d8">
                {segmentation.pca.scatter_sample.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={clusterColors[entry.cluster_id % clusterColors.length]}
                    opacity={0.65}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div style={{ display: 'flex', gap: '1.5rem', justifyContent: 'center', marginTop: '0.75rem', flexWrap: 'wrap' }}>
          {segmentation.cluster_profiles.map((p, idx) => (
            <div key={p.cluster_id} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8rem' }}>
              <div style={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: clusterColors[idx] }} />
              <span>{p.segment_name}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
