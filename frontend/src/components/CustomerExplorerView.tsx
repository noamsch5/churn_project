import React, { useState, useEffect } from 'react';
import {
  Search,
  ChevronLeft,
  ChevronRight,
  ArrowUpDown,
} from 'lucide-react';
import type { CustomerRecord, CustomerListResponse } from '../types/api';
import { fetchCustomers } from '../services/api';

export const CustomerExplorerView: React.FC = () => {
  const [data, setData] = useState<CustomerListResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [page, setPage] = useState<number>(1);
  const [pageSize] = useState<number>(20);
  const [search, setSearch] = useState<string>('');
  const [riskLevel, setRiskLevel] = useState<string>('');
  const [segment, setSegment] = useState<string>('');
  const [sortBy, setSortBy] = useState<string>('churn_probability');
  const [order, setOrder] = useState<string>('desc');
  const [selectedCustomer, setSelectedCustomer] = useState<CustomerRecord | null>(null);

  const loadData = async () => {
    setLoading(true);
    try {
      const res = await fetchCustomers({
        page,
        page_size: pageSize,
        search: search.trim() || undefined,
        risk_level: riskLevel || undefined,
        segment: segment || undefined,
        sort_by: sortBy,
        order,
      });
      setData(res);
    } catch (err) {
      console.error('Failed to load customer list', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [page, riskLevel, segment, sortBy, order]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    loadData();
  };

  const toggleSort = (column: string) => {
    if (sortBy === column) {
      setOrder(order === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      setOrder('desc');
    }
  };

  return (
    <div>
      <div className="page-intro">
        <h2 className="page-title">Customer Intelligence Explorer</h2>
        <p className="page-description">
          Real-time queryable database of all 10,127 portfolio accounts scored with calibrated churn probabilities,
          risk tiers, and K-Means segment assignments.
        </p>
      </div>

      {/* Filter and Search Bar */}
      <div className="card-box" style={{ padding: '1rem 1.25rem', marginBottom: '1.25rem' }}>
        <form onSubmit={handleSearchSubmit} style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
          {/* Search */}
          <div style={{ position: 'relative', flex: '1 1 220px' }}>
            <Search size={16} color="#94a3b8" style={{ position: 'absolute', left: 10, top: 12 }} />
            <input
              type="text"
              className="form-input"
              style={{ paddingLeft: '2rem', width: '100%' }}
              placeholder="Search by Account #..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>

          {/* Risk Level Filter */}
          <div style={{ minWidth: '150px' }}>
            <select
              className="form-select"
              value={riskLevel}
              onChange={(e) => {
                setRiskLevel(e.target.value);
                setPage(1);
              }}
            >
              <option value="">All Risk Tiers</option>
              <option value="Critical">Critical Risk ({'>'}75%)</option>
              <option value="High">High Risk (50-75%)</option>
              <option value="Medium">Medium Risk (25-50%)</option>
              <option value="Low">Low Risk ({'<'}25%)</option>
            </select>
          </div>

          {/* Segment Filter */}
          <div style={{ minWidth: '180px' }}>
            <select
              className="form-select"
              value={segment}
              onChange={(e) => {
                setSegment(e.target.value);
                setPage(1);
              }}
            >
              <option value="">All Segments</option>
              <option value="Accelerating Growth Customers">Accelerating Growth Customers</option>
              <option value="High-Utilization Credit Revolvers">High-Utilization Credit Revolvers</option>
              <option value="Disengaged & Underutilized (At-Risk)">Disengaged & Underutilized (At-Risk)</option>
              <option value="High-Volume Power Spenders">High-Volume Power Spenders</option>
            </select>
          </div>

          <button type="submit" className="btn-primary" style={{ padding: '0.6rem 1.2rem' }}>
            Search
          </button>
        </form>
      </div>

      {/* Table */}
      <div className="card-box">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <span style={{ fontSize: '0.85rem', color: '#64748b' }}>
            Showing {data ? `${data.customers.length} of ${data.total_records.toLocaleString()}` : 0} matching accounts
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <button
              className="btn-secondary"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              style={{ padding: '0.4rem 0.75rem' }}
            >
              <ChevronLeft size={16} /> Prev
            </button>
            <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>
              Page {data ? data.page : 1} of {data ? data.total_pages : 1}
            </span>
            <button
              className="btn-secondary"
              disabled={!data || page >= data.total_pages}
              onClick={() => setPage((p) => p + 1)}
              style={{ padding: '0.4rem 0.75rem' }}
            >
              Next <ChevronRight size={16} />
            </button>
          </div>
        </div>

        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th onClick={() => toggleSort('CLIENTNUM')} style={{ cursor: 'pointer' }}>
                  Account # <ArrowUpDown size={12} style={{ display: 'inline' }} />
                </th>
                <th onClick={() => toggleSort('churn_probability')} style={{ cursor: 'pointer' }}>
                  Churn Risk <ArrowUpDown size={12} style={{ display: 'inline' }} />
                </th>
                <th>Risk Tier</th>
                <th>Assigned Segment</th>
                <th onClick={() => toggleSort('Total_Trans_Amt')} style={{ cursor: 'pointer' }}>
                  Annual Spend <ArrowUpDown size={12} style={{ display: 'inline' }} />
                </th>
                <th onClick={() => toggleSort('Total_Trans_Ct')} style={{ cursor: 'pointer' }}>
                  Transactions <ArrowUpDown size={12} style={{ display: 'inline' }} />
                </th>
                <th onClick={() => toggleSort('Months_Inactive_12_mon')} style={{ cursor: 'pointer' }}>
                  Inactivity <ArrowUpDown size={12} style={{ display: 'inline' }} />
                </th>
                <th onClick={() => toggleSort('Avg_Utilization_Ratio')} style={{ cursor: 'pointer' }}>
                  Utilization <ArrowUpDown size={12} style={{ display: 'inline' }} />
                </th>
                <th>Actual Status</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={10} style={{ textAlign: 'center', padding: '2rem', color: '#64748b' }}>
                    Querying customer portfolio records...
                  </td>
                </tr>
              ) : data && data.customers.length > 0 ? (
                data.customers.map((c) => (
                  <tr key={c.client_num}>
                    <td className="num-cell" style={{ fontWeight: 600 }}>
                      #{c.client_num}
                    </td>
                    <td className="num-cell" style={{ fontWeight: 700, color: c.churn_probability > 0.5 ? '#ef4444' : '#10b981' }}>
                      {(c.churn_probability * 100).toFixed(1)}%
                    </td>
                    <td>
                      <span className={`badge badge-${c.risk_level.toLowerCase()}`}>
                        {c.risk_level}
                      </span>
                    </td>
                    <td style={{ fontSize: '0.8rem' }}>{c.segment_name}</td>
                    <td className="num-cell">${c.total_trans_amt.toLocaleString()}</td>
                    <td className="num-cell">{c.total_trans_ct} / yr</td>
                    <td className="num-cell">{c.months_inactive} mos</td>
                    <td className="num-cell">{(c.utilization_ratio * 100).toFixed(1)}%</td>
                    <td>
                      <span
                        style={{
                          fontSize: '0.75rem',
                          fontWeight: 600,
                          color: c.actual_status === 'Attrited Customer' ? '#dc2626' : '#16a34a',
                        }}
                      >
                        {c.actual_status}
                      </span>
                    </td>
                    <td>
                      <button
                        className="btn-secondary"
                        style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                        onClick={() => setSelectedCustomer(c)}
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={10} style={{ textAlign: 'center', padding: '2rem', color: '#64748b' }}>
                    No matching customer accounts found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Customer Detail Modal Drawer */}
      {selectedCustomer && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 100,
            padding: '1rem',
          }}
          onClick={() => setSelectedCustomer(null)}
        >
          <div
            className="card-box"
            style={{ maxWidth: '540px', width: '100%', maxHeight: '90vh', overflowY: 'auto' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="card-header">
              <div>
                <h3 className="card-title">Account Deep-Dive: #{selectedCustomer.client_num}</h3>
                <p className="card-subtitle">Comprehensive financial and risk profile</p>
              </div>
              <button
                className="btn-secondary"
                style={{ padding: '0.3rem 0.6rem' }}
                onClick={() => setSelectedCustomer(null)}
              >
                ✕ Close
              </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div style={{ backgroundColor: '#f8fafc', padding: '0.75rem', borderRadius: 8 }}>
                <span style={{ fontSize: '0.75rem', color: '#64748b', display: 'block' }}>Predicted Churn Risk</span>
                <span style={{ fontSize: '1.5rem', fontWeight: 800, color: selectedCustomer.churn_probability > 0.5 ? '#ef4444' : '#10b981' }}>
                  {(selectedCustomer.churn_probability * 100).toFixed(1)}%
                </span>
                <span className={`badge badge-${selectedCustomer.risk_level.toLowerCase()}`} style={{ marginTop: '0.25rem' }}>
                  {selectedCustomer.risk_level} Risk
                </span>
              </div>
              <div style={{ backgroundColor: '#f8fafc', padding: '0.75rem', borderRadius: 8 }}>
                <span style={{ fontSize: '0.75rem', color: '#64748b', display: 'block' }}>Assigned Segment</span>
                <strong style={{ fontSize: '0.95rem', display: 'block', margin: '0.25rem 0' }}>
                  {selectedCustomer.segment_name}
                </strong>
                <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Cluster #{selectedCustomer.cluster_id}</span>
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.85rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid #f1f5f9' }}>
                <span style={{ color: '#64748b' }}>Demographics:</span>
                <strong>Age {selectedCustomer.age} · Gender {selectedCustomer.gender}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid #f1f5f9' }}>
                <span style={{ color: '#64748b' }}>Annual Transactions:</span>
                <strong>{selectedCustomer.total_trans_ct} swipes (${selectedCustomer.total_trans_amt.toLocaleString()})</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid #f1f5f9' }}>
                <span style={{ color: '#64748b' }}>Credit Line & Usage:</span>
                <strong>${selectedCustomer.credit_limit.toLocaleString()} ({(selectedCustomer.utilization_ratio * 100).toFixed(1)}% util)</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid #f1f5f9' }}>
                <span style={{ color: '#64748b' }}>Inactivity (12m):</span>
                <strong>{selectedCustomer.months_inactive} months dormant</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid #f1f5f9' }}>
                <span style={{ color: '#64748b' }}>Bank Contacts (12m):</span>
                <strong>{selectedCustomer.contacts_count} service contacts</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0' }}>
                <span style={{ color: '#64748b' }}>Actual Account Status:</span>
                <strong style={{ color: selectedCustomer.actual_status === 'Attrited Customer' ? '#dc2626' : '#16a34a' }}>
                  {selectedCustomer.actual_status}
                </strong>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
