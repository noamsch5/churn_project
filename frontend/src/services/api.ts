/**
 * API service client for interacting with the FastAPI backend.
 */

import type {
  CustomerInput,
  PredictResponse,
  CombinedAnalysisResponse,
  CustomerListResponse,
  AnalyticsSummary,
} from '../types/api';

// Production uses the same origin as FastAPI; Vite proxies these paths locally.
const API_BASE = import.meta.env.VITE_API_URL || '';

export async function checkHealth(): Promise<{ status: string; models_loaded: boolean }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export async function fetchAnalyticsSummary(): Promise<AnalyticsSummary> {
  const res = await fetch(`${API_BASE}/analytics-summary`);
  if (!res.ok) throw new Error('Failed to load analytics summary');
  return res.json();
}

export async function fetchCustomers(params: {
  page?: number;
  page_size?: number;
  risk_level?: string;
  segment?: string;
  search?: string;
  sort_by?: string;
  order?: string;
}): Promise<CustomerListResponse> {
  const query = new URLSearchParams();
  if (params.page) query.append('page', params.page.toString());
  if (params.page_size) query.append('page_size', params.page_size.toString());
  if (params.risk_level) query.append('risk_level', params.risk_level);
  if (params.segment) query.append('segment', params.segment);
  if (params.search) query.append('search', params.search);
  if (params.sort_by) query.append('sort_by', params.sort_by);
  if (params.order) query.append('order', params.order);

  const res = await fetch(`${API_BASE}/customers?${query.toString()}`);
  if (!res.ok) throw new Error('Failed to fetch customers list');
  return res.json();
}

export async function predictChurn(input: CustomerInput): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error('Prediction request failed');
  return res.json();
}

export async function analyzeCustomer(input: CustomerInput): Promise<CombinedAnalysisResponse> {
  const res = await fetch(`${API_BASE}/analyze-customer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error('Customer analysis request failed');
  return res.json();
}
