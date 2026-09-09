import React from 'react';
import {
  Users,
  ShieldAlert,
  CreditCard,
  Zap,
  TrendingUp,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from 'recharts';
import type { AnalyticsSummary } from '../types/api';

interface SegmentsViewProps {
  summary: AnalyticsSummary | null;
}

export const SegmentsView: React.FC<SegmentsViewProps> = ({ summary }) => {
  if (!summary) {
    return <div className="card-box">Loading segmentation intelligence...</div>;
  }

  const { segmentation } = summary;
  const profiles = segmentation.cluster_profiles;

  // Comparison chart data
  const comparisonData = profiles.map((p) => ({
    name: p.segment_name.split(' ')[0] + '...',
    fullName: p.segment_name,
    churn_rate: p.churn_rate_pct,
    avg_spend: p.avg_transaction_amt,
    avg_trans_ct: p.avg_transaction_count,
    utilization: +(p.avg_utilization_ratio * 100).toFixed(1),
  }));

  const getClusterIcon = (id: number) => {
    switch (id) {
      case 0:
        return <TrendingUp size={20} color="#10b981" />;
      case 1:
        return <CreditCard size={20} color="#2563eb" />;
      case 2:
        return <ShieldAlert size={20} color="#ef4444" />;
      case 3:
        return <Zap size={20} color="#8b5cf6" />;
      default:
        return <Users size={20} color="#64748b" />;
    }
  };

  return (
    <div>
      <div className="page-intro">
        <h2 className="page-title">Unsupervised Customer Segmentation</h2>
        <p className="page-description">
          Natural customer behavioral clusters discovered via K-Means (StandardScaler + $k=4$, evaluated via
          Silhouette & Elbow criteria). Churn labels were strictly excluded during training.
        </p>
      </div>

      {/* Cluster Cards */}
      <div className="grid-2" style={{ marginBottom: '2rem' }}>
        {profiles.map((p) => {
          const isHighRisk = p.churn_rate_pct > 20;
          return (
            <div
              key={p.cluster_id}
              className="card-box"
              style={{
                borderTop: `4px solid ${isHighRisk ? '#ef4444' : p.churn_rate_pct < 8 ? '#10b981' : '#2563eb'}`,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <div
                    style={{
                      padding: '0.5rem',
                      borderRadius: 8,
                      backgroundColor: isHighRisk ? '#fef2f2' : '#f0fdf4',
                    }}
                  >
                    {getClusterIcon(p.cluster_id)}
                  </div>
                  <div>
                    <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>{p.segment_name}</h3>
                    <p style={{ fontSize: '0.775rem', color: '#64748b' }}>
                      Cluster #{p.cluster_id} · {p.customer_count.toLocaleString()} cardholders ({p.percentage_of_total}%)
                    </p>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '1.25rem', fontWeight: 800, color: isHighRisk ? '#ef4444' : '#10b981' }}>
                    {p.churn_rate_pct.toFixed(1)}%
                  </div>
                  <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Segment Churn</span>
                </div>
              </div>

              <p style={{ fontSize: '0.85rem', color: '#475569', marginBottom: '1rem', lineHeight: 1.45 }}>
                {p.description}
              </p>

              {/* Metric Highlights Pill Grid */}
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3, 1fr)',
                  gap: '0.5rem',
                  backgroundColor: '#f8fafc',
                  padding: '0.75rem',
                  borderRadius: 8,
                  marginBottom: '1rem',
                }}
              >
                <div>
                  <span style={{ fontSize: '0.7rem', color: '#64748b', display: 'block' }}>Avg Spend</span>
                  <strong style={{ fontSize: '0.9rem' }}>${p.avg_transaction_amt.toLocaleString()}</strong>
                </div>
                <div>
                  <span style={{ fontSize: '0.7rem', color: '#64748b', display: 'block' }}>Avg Tx Count</span>
                  <strong style={{ fontSize: '0.9rem' }}>{p.avg_transaction_count.toFixed(0)} / yr</strong>
                </div>
                <div>
                  <span style={{ fontSize: '0.7rem', color: '#64748b', display: 'block' }}>Credit Limit</span>
                  <strong style={{ fontSize: '0.9rem' }}>${p.avg_credit_limit.toLocaleString()}</strong>
                </div>
                <div>
                  <span style={{ fontSize: '0.7rem', color: '#64748b', display: 'block' }}>Utilization</span>
                  <strong style={{ fontSize: '0.9rem' }}>{(p.avg_utilization_ratio * 100).toFixed(1)}%</strong>
                </div>
                <div>
                  <span style={{ fontSize: '0.7rem', color: '#64748b', display: 'block' }}>Inactive Mos</span>
                  <strong style={{ fontSize: '0.9rem' }}>{p.avg_inactive_months.toFixed(1)} mos</strong>
                </div>
                <div>
                  <span style={{ fontSize: '0.7rem', color: '#64748b', display: 'block' }}>Contacts</span>
                  <strong style={{ fontSize: '0.9rem' }}>{p.avg_contacts_count.toFixed(1)} calls</strong>
                </div>
              </div>

              {/* Action Plan */}
              <div
                style={{
                  backgroundColor: isHighRisk ? '#fff7ed' : '#eff6ff',
                  border: `1px solid ${isHighRisk ? '#fed7aa' : '#bfdbfe'}`,
                  borderRadius: 6,
                  padding: '0.65rem 0.75rem',
                }}
              >
                <span
                  style={{
                    fontSize: '0.775rem',
                    fontWeight: 700,
                    color: isHighRisk ? '#c2410c' : '#1e40af',
                    display: 'block',
                    marginBottom: '0.2rem',
                  }}
                >
                  Strategic Retention Playbook:
                </span>
                <span style={{ fontSize: '0.8rem', color: '#334155' }}>{p.strategy}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Comparative Chart */}
      <div className="card-box">
        <div className="card-header">
          <div>
            <h3 className="card-title">Comparative Behavioral Benchmarks across Segments</h3>
            <p className="card-subtitle">
              Comparing average annual transaction spend against segment attrition rates.
            </p>
          </div>
        </div>

        <div style={{ height: 320, width: '100%' }}>
          <ResponsiveContainer>
            <BarChart data={comparisonData} margin={{ top: 20, right: 20, left: 10, bottom: 20 }}>
              <XAxis dataKey="fullName" tick={{ fontSize: 11 }} />
              <YAxis yAxisId="left" orientation="left" unit="$" label={{ value: 'Avg Spend ($)', angle: -90, position: 'insideLeft', fontSize: 12 }} />
              <YAxis yAxisId="right" orientation="right" unit="%" label={{ value: 'Churn Rate (%)', angle: 90, position: 'insideRight', fontSize: 12 }} />
              <Tooltip
                formatter={(val: any, name: any) => [
                  name === 'churn_rate' ? `${Number(val).toFixed(1)}%` : `$${Number(val).toLocaleString()}`,
                  name === 'churn_rate' ? 'Churn Rate' : 'Avg Spend',
                ]}
              />
              <Legend />
              <Bar yAxisId="left" dataKey="avg_spend" name="Avg Annual Spend ($)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              <Bar yAxisId="right" dataKey="churn_rate" name="Segment Churn Rate (%)" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
