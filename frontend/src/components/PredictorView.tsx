import React, { useState } from 'react';
import {
  Gauge,
  Layers,
} from 'lucide-react';
import type { CustomerInput, CombinedAnalysisResponse } from '../types/api';
import { analyzeCustomer } from '../services/api';

const DEFAULT_CUSTOMER: CustomerInput = {
  Customer_Age: 45,
  Gender: 'M',
  Dependent_count: 2,
  Education_Level: 'Graduate',
  Marital_Status: 'Married',
  Income_Category: '$60K - $80K',
  Card_Category: 'Blue',
  Months_on_book: 36,
  Total_Relationship_Count: 4,
  Months_Inactive_12_mon: 2,
  Contacts_Count_12_mon: 2,
  Credit_Limit: 8500.0,
  Total_Revolving_Bal: 1200.0,
  Avg_Open_To_Buy: 7300.0,
  Total_Amt_Chng_Q4_Q1: 0.75,
  Total_Trans_Amt: 4200.0,
  Total_Trans_Ct: 65,
  Total_Ct_Chng_Q4_Q1: 0.70,
  Avg_Utilization_Ratio: 0.18,
};

const PERSONAS: { name: string; description: string; data: CustomerInput }[] = [
  {
    name: 'High Risk Attriter',
    description: 'Dormant card, declining transactions, multiple support complaints',
    data: {
      Customer_Age: 48,
      Gender: 'F',
      Dependent_count: 3,
      Education_Level: 'High School',
      Marital_Status: 'Single',
      Income_Category: 'Less than $40K',
      Card_Category: 'Blue',
      Months_on_book: 39,
      Total_Relationship_Count: 2,
      Months_Inactive_12_mon: 4,
      Contacts_Count_12_mon: 5,
      Credit_Limit: 3200.0,
      Total_Revolving_Bal: 0.0,
      Avg_Open_To_Buy: 3200.0,
      Total_Amt_Chng_Q4_Q1: 0.38,
      Total_Trans_Amt: 1450.0,
      Total_Trans_Ct: 28,
      Total_Ct_Chng_Q4_Q1: 0.42,
      Avg_Utilization_Ratio: 0.0,
    },
  },
  {
    name: 'Loyal High-Volume Spender',
    description: 'High velocity transactor, premium card, deep relationship',
    data: {
      Customer_Age: 42,
      Gender: 'M',
      Dependent_count: 2,
      Education_Level: 'Doctorate',
      Marital_Status: 'Married',
      Income_Category: '$120K +',
      Card_Category: 'Gold',
      Months_on_book: 34,
      Total_Relationship_Count: 5,
      Months_Inactive_12_mon: 1,
      Contacts_Count_12_mon: 1,
      Credit_Limit: 24000.0,
      Total_Revolving_Bal: 1850.0,
      Avg_Open_To_Buy: 22150.0,
      Total_Amt_Chng_Q4_Q1: 0.88,
      Total_Trans_Amt: 11200.0,
      Total_Trans_Ct: 104,
      Total_Ct_Chng_Q4_Q1: 0.85,
      Avg_Utilization_Ratio: 0.08,
    },
  },
  {
    name: 'High-Utilization Revolver',
    description: 'Active credit card borrower, consistent interest payer, sticky',
    data: {
      Customer_Age: 46,
      Gender: 'F',
      Dependent_count: 1,
      Education_Level: 'College',
      Marital_Status: 'Married',
      Income_Category: '$40K - $60K',
      Card_Category: 'Blue',
      Months_on_book: 38,
      Total_Relationship_Count: 4,
      Months_Inactive_12_mon: 2,
      Contacts_Count_12_mon: 2,
      Credit_Limit: 4000.0,
      Total_Revolving_Bal: 2400.0,
      Avg_Open_To_Buy: 1600.0,
      Total_Amt_Chng_Q4_Q1: 0.72,
      Total_Trans_Amt: 4100.0,
      Total_Trans_Ct: 72,
      Total_Ct_Chng_Q4_Q1: 0.74,
      Avg_Utilization_Ratio: 0.60,
    },
  },
  {
    name: 'Accelerating Growth Account',
    description: 'Recent strong uptick in transactions and cross-sell candidate',
    data: {
      Customer_Age: 38,
      Gender: 'M',
      Dependent_count: 2,
      Education_Level: 'Graduate',
      Marital_Status: 'Married',
      Income_Category: '$80K - $120K',
      Card_Category: 'Silver',
      Months_on_book: 24,
      Total_Relationship_Count: 6,
      Months_Inactive_12_mon: 1,
      Contacts_Count_12_mon: 2,
      Credit_Limit: 14500.0,
      Total_Revolving_Bal: 1100.0,
      Avg_Open_To_Buy: 13400.0,
      Total_Amt_Chng_Q4_Q1: 1.25,
      Total_Trans_Amt: 5400.0,
      Total_Trans_Ct: 88,
      Total_Ct_Chng_Q4_Q1: 1.30,
      Avg_Utilization_Ratio: 0.12,
    },
  },
];

export const PredictorView: React.FC = () => {
  const [formData, setFormData] = useState<CustomerInput>(DEFAULT_CUSTOMER);
  const [result, setResult] = useState<CombinedAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (field: keyof CustomerInput, value: any) => {
    setFormData((prev) => {
      const updated = { ...prev, [field]: value };
      // Keep open to buy consistent if limit or revolving balance changes
      if (field === 'Credit_Limit' || field === 'Total_Revolving_Bal') {
        const lim = field === 'Credit_Limit' ? Number(value) : prev.Credit_Limit;
        const bal = field === 'Total_Revolving_Bal' ? Number(value) : prev.Total_Revolving_Bal;
        updated.Avg_Open_To_Buy = Math.max(0, lim - bal);
        updated.Avg_Utilization_Ratio = lim > 0 ? Math.min(1.0, Number((bal / lim).toFixed(3))) : 0;
      }
      return updated;
    });
  };

  const handlePersonaSelect = (persona: (typeof PERSONAS)[0]) => {
    setFormData(persona.data);
    runAnalysis(persona.data);
  };

  const runAnalysis = async (inputToUse?: CustomerInput) => {
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeCustomer(inputToUse || formData);
      setResult(data);
    } catch (err: any) {
      setError('Unable to score customer via API. Verify the FastAPI backend is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  // Determine risk meter color
  const getRiskColor = (prob: number) => {
    if (prob < 0.25) return '#10b981';
    if (prob < 0.50) return '#f59e0b';
    if (prob < 0.75) return '#f97316';
    return '#ef4444';
  };

  return (
    <div>
      <div className="page-intro">
        <h2 className="page-title">Interactive Churn Risk Simulator</h2>
        <p className="page-description">
          Adjust customer behavioral and financial parameters or load archetypes to evaluate real-time
          logistic regression probabilities and K-Means segmentation playbooks.
        </p>
      </div>

      {/* Preset Personas Row */}
      <div className="personas-row">
        <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#475569' }}>
          Load Test Persona:
        </span>
        {PERSONAS.map((p) => (
          <button
            key={p.name}
            className="persona-btn"
            onClick={() => handlePersonaSelect(p)}
            title={p.description}
          >
            {p.name}
          </button>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '1.5rem', alignItems: 'start' }}>
        {/* Input Parameters Form */}
        <div className="card-box">
          <div className="card-header">
            <div>
              <h3 className="card-title">Customer Account Profile</h3>
              <p className="card-subtitle">21 input features configured for zero-leakage inference.</p>
            </div>
            <button className="btn-primary" onClick={() => runAnalysis()} disabled={loading}>
              <Gauge size={16} />
              <span>{loading ? 'Evaluating...' : 'Analyze Customer'}</span>
            </button>
          </div>

          {/* Section 1: Demographics */}
          <div style={{ marginBottom: '1.25rem' }}>
            <h4 style={{ fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#64748b', marginBottom: '0.75rem' }}>
              1. Demographics
            </h4>
            <div className="form-grid">
              <div className="form-group">
                <label className="form-label">Age</label>
                <input
                  type="number"
                  className="form-input"
                  value={formData.Customer_Age}
                  onChange={(e) => handleInputChange('Customer_Age', parseInt(e.target.value) || 18)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Gender</label>
                <select
                  className="form-select"
                  value={formData.Gender}
                  onChange={(e) => handleInputChange('Gender', e.target.value)}
                >
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Dependents</label>
                <input
                  type="number"
                  className="form-input"
                  value={formData.Dependent_count}
                  onChange={(e) => handleInputChange('Dependent_count', parseInt(e.target.value) || 0)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Education Level</label>
                <select
                  className="form-select"
                  value={formData.Education_Level}
                  onChange={(e) => handleInputChange('Education_Level', e.target.value)}
                >
                  <option value="High School">High School</option>
                  <option value="Graduate">Graduate</option>
                  <option value="Uneducated">Uneducated</option>
                  <option value="College">College</option>
                  <option value="Post-Graduate">Post-Graduate</option>
                  <option value="Doctorate">Doctorate</option>
                  <option value="Unknown">Unknown</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Income Category</label>
                <select
                  className="form-select"
                  value={formData.Income_Category}
                  onChange={(e) => handleInputChange('Income_Category', e.target.value)}
                >
                  <option value="Less than $40K">Less than $40K</option>
                  <option value="$40K - $60K">$40K - $60K</option>
                  <option value="$60K - $80K">$60K - $80K</option>
                  <option value="$80K - $120K">$80K - $120K</option>
                  <option value="$120K +">$120K +</option>
                  <option value="Unknown">Unknown</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Marital Status</label>
                <select
                  className="form-select"
                  value={formData.Marital_Status}
                  onChange={(e) => handleInputChange('Marital_Status', e.target.value)}
                >
                  <option value="Married">Married</option>
                  <option value="Single">Single</option>
                  <option value="Divorced">Divorced</option>
                  <option value="Unknown">Unknown</option>
                </select>
              </div>
            </div>
          </div>

          {/* Section 2: Engagement & Relationship */}
          <div style={{ marginBottom: '1.25rem' }}>
            <h4 style={{ fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#64748b', marginBottom: '0.75rem' }}>
              2. Product & Engagement
            </h4>
            <div className="form-grid">
              <div className="form-group">
                <label className="form-label">Card Tier</label>
                <select
                  className="form-select"
                  value={formData.Card_Category}
                  onChange={(e) => handleInputChange('Card_Category', e.target.value)}
                >
                  <option value="Blue">Blue</option>
                  <option value="Silver">Silver</option>
                  <option value="Gold">Gold</option>
                  <option value="Platinum">Platinum</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Tenure (Months on Book)</label>
                <input
                  type="number"
                  className="form-input"
                  value={formData.Months_on_book}
                  onChange={(e) => handleInputChange('Months_on_book', parseInt(e.target.value) || 1)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Total Products Held</label>
                <input
                  type="number"
                  className="form-input"
                  value={formData.Total_Relationship_Count}
                  onChange={(e) => handleInputChange('Total_Relationship_Count', parseInt(e.target.value) || 1)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Inactive Months (12m)</label>
                <input
                  type="number"
                  className="form-input"
                  value={formData.Months_Inactive_12_mon}
                  onChange={(e) => handleInputChange('Months_Inactive_12_mon', parseInt(e.target.value) || 0)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Bank Contacts (12m)</label>
                <input
                  type="number"
                  className="form-input"
                  value={formData.Contacts_Count_12_mon}
                  onChange={(e) => handleInputChange('Contacts_Count_12_mon', parseInt(e.target.value) || 0)}
                />
              </div>
            </div>
          </div>

          {/* Section 3: Financial & Transactions */}
          <div>
            <h4 style={{ fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#64748b', marginBottom: '0.75rem' }}>
              3. Financial & Transaction Behavior
            </h4>
            <div className="form-grid">
              <div className="form-group">
                <label className="form-label">Credit Limit ($)</label>
                <input
                  type="number"
                  className="form-input"
                  value={formData.Credit_Limit}
                  onChange={(e) => handleInputChange('Credit_Limit', parseFloat(e.target.value) || 0)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Revolving Balance ($)</label>
                <input
                  type="number"
                  className="form-input"
                  value={formData.Total_Revolving_Bal}
                  onChange={(e) => handleInputChange('Total_Revolving_Bal', parseFloat(e.target.value) || 0)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Annual Transaction Count</label>
                <input
                  type="number"
                  className="form-input"
                  value={formData.Total_Trans_Ct}
                  onChange={(e) => handleInputChange('Total_Trans_Ct', parseInt(e.target.value) || 0)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Annual Spend Amount ($)</label>
                <input
                  type="number"
                  className="form-input"
                  value={formData.Total_Trans_Amt}
                  onChange={(e) => handleInputChange('Total_Trans_Amt', parseFloat(e.target.value) || 0)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Quarterly Tx Count Ratio (Q4/Q1)</label>
                <input
                  type="number"
                  step="0.05"
                  className="form-input"
                  value={formData.Total_Ct_Chng_Q4_Q1}
                  onChange={(e) => handleInputChange('Total_Ct_Chng_Q4_Q1', parseFloat(e.target.value) || 0)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Quarterly Tx Amount Ratio (Q4/Q1)</label>
                <input
                  type="number"
                  step="0.05"
                  className="form-input"
                  value={formData.Total_Amt_Chng_Q4_Q1}
                  onChange={(e) => handleInputChange('Total_Amt_Chng_Q4_Q1', parseFloat(e.target.value) || 0)}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Real-time Risk Gauge & Intelligence Output */}
        <div>
          {error && (
            <div style={{ backgroundColor: '#fef2f2', border: '1px solid #fecaca', color: '#991b1b', padding: '1rem', borderRadius: 8, marginBottom: '1rem' }}>
              {error}
            </div>
          )}

          {result ? (
            <div>
              {/* Risk Gauge Card */}
              <div className="card-box" style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '0.85rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#64748b' }}>
                  Predicted Churn Risk
                </div>
                <div
                  className="risk-percentage"
                  style={{ color: getRiskColor(result.churn_probability), margin: '0.75rem 0' }}
                >
                  {result.churn_probability_pct.toFixed(1)}%
                </div>

                <div className="risk-bar-track">
                  <div
                    className="risk-bar-fill"
                    style={{
                      width: `${Math.min(100, Math.max(2, result.churn_probability_pct))}%`,
                      backgroundColor: getRiskColor(result.churn_probability),
                    }}
                  />
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.5rem' }}>
                  <span className={`badge badge-${result.risk_level.toLowerCase()}`}>
                    {result.risk_level.toUpperCase()} RISK
                  </span>
                  <span style={{ fontSize: '0.825rem', color: '#64748b' }}>
                    Model Decision Threshold: <strong>{(result.threshold_applied * 100).toFixed(0)}%</strong>
                  </span>
                </div>

                <div style={{ marginTop: '1rem', padding: '0.75rem', backgroundColor: '#f8fafc', borderRadius: 8, fontSize: '0.85rem' }}>
                  Classification Status:{' '}
                  <strong style={{ color: result.predicted_churn ? '#ef4444' : '#10b981' }}>
                    {result.predicted_status}
                  </strong>
                </div>
              </div>

              {/* Segment Card */}
              <div className="card-box">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  <Layers size={18} color="#2563eb" />
                  <h4 style={{ fontSize: '0.95rem', fontWeight: 600 }}>
                    Assigned Customer Segment: {result.segment_name}
                  </h4>
                </div>
                <p style={{ fontSize: '0.85rem', color: '#475569', marginBottom: '0.75rem' }}>
                  {result.segment_description}
                </p>
                <div style={{ backgroundColor: '#eff6ff', border: '1px solid #bfdbfe', padding: '0.75rem', borderRadius: 8 }}>
                  <strong style={{ fontSize: '0.8rem', color: '#1e40af', display: 'block', marginBottom: '0.25rem' }}>
                    Recommended Retention Action Plan:
                  </strong>
                  <span style={{ fontSize: '0.825rem', color: '#1e3a8a' }}>{result.recommended_strategy}</span>
                </div>
              </div>

              {/* Key Explanatory Drivers */}
              <div className="card-box">
                <h4 style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '0.75rem' }}>
                  Top Model Drivers for this Account
                </h4>
                {result.key_drivers.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {result.key_drivers.map((d, i) => (
                      <div
                        key={i}
                        style={{
                          padding: '0.6rem 0.75rem',
                          borderRadius: 6,
                          fontSize: '0.825rem',
                          backgroundColor: d.direction === 'High Risk' ? '#fff7ed' : '#ecfdf5',
                          border: `1px solid ${d.direction === 'High Risk' ? '#fed7aa' : '#a7f3d0'}`,
                        }}
                      >
                        <strong style={{ color: d.direction === 'High Risk' ? '#c2410c' : '#047857' }}>
                          [{d.direction}] {d.feature}:{' '}
                        </strong>
                        <span>{d.message}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p style={{ fontSize: '0.825rem', color: '#64748b' }}>
                    Features fall within typical portfolio bounds.
                  </p>
                )}
              </div>
            </div>
          ) : (
            <div className="card-box" style={{ textAlign: 'center', padding: '3rem 1.5rem', color: '#64748b' }}>
              <Gauge size={48} style={{ opacity: 0.3, margin: '0 auto 1rem auto' }} />
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#334155' }}>
                Simulator Ready
              </h3>
              <p style={{ fontSize: '0.85rem', marginTop: '0.25rem' }}>
                Click "Analyze Customer" or select a preset persona above to generate instant risk metrics and cluster intelligence.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
