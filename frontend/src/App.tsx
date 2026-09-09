import { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { OverviewView } from './components/OverviewView';
import { PredictorView } from './components/PredictorView';
import { SegmentsView } from './components/SegmentsView';
import { PerformanceView } from './components/PerformanceView';
import { InsightsView } from './components/InsightsView';
import { CustomerExplorerView } from './components/CustomerExplorerView';
import type { AnalyticsSummary } from './types/api';
import { checkHealth, fetchAnalyticsSummary } from './services/api';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    async function init() {
      try {
        const health = await checkHealth();
        setIsBackendConnected(health.models_loaded);
        const data = await fetchAnalyticsSummary();
        setSummary(data);
      } catch (err) {
        console.warn('API connection failed. Retrying or working in offline mode.', err);
        setIsBackendConnected(false);
      } finally {
        setLoading(false);
      }
    }
    init();
  }, []);

  return (
    <div className="app-container">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isBackendConnected={isBackendConnected}
      />

      <main className="main-content">
        {loading && !summary ? (
          <div className="card-box" style={{ textAlign: 'center', padding: '3rem' }}>
            <p style={{ color: '#64748b' }}>Connecting to model server and loading analytical intelligence...</p>
          </div>
        ) : (
          <>
            {activeTab === 'overview' && (
              <OverviewView summary={summary} onNavigate={(tab) => setActiveTab(tab)} />
            )}
            {activeTab === 'predictor' && <PredictorView />}
            {activeTab === 'segments' && <SegmentsView summary={summary} />}
            {activeTab === 'performance' && <PerformanceView summary={summary} />}
            {activeTab === 'insights' && <InsightsView summary={summary} />}
            {activeTab === 'explorer' && <CustomerExplorerView />}
          </>
        )}
      </main>

      <footer className="app-footer">
        Credit Card Customer Churn & Segmentation Intelligence System · Built with Python, Scikit-Learn, FastAPI, React & TypeScript
      </footer>
    </div>
  );
}

export default App;
