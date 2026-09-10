import React, { useState } from 'react';
import {
  BarChart3,
  Users,
  ShieldAlert,
  Gauge,
  FileSpreadsheet,
  Database,
  Building2,
  Menu,
  X,
} from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isBackendConnected: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  isBackendConnected,
}) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const tabs = [
    { id: 'overview', label: 'Executive Overview', icon: BarChart3 },
    { id: 'predictor', label: 'Churn Risk Simulator', icon: Gauge },
    { id: 'segments', label: 'Customer Segments', icon: Users },
    { id: 'performance', label: 'Model Performance', icon: ShieldAlert },
    { id: 'insights', label: 'Data & Hypothesis Tests', icon: FileSpreadsheet },
    { id: 'explorer', label: 'Customer Database', icon: Database },
  ];

  return (
    <header className="app-header">
      <div className="header-top">
        <div className="brand-wrapper">
          <div className="brand-icon-box">
            <Building2 size={22} color="#ffffff" />
          </div>
          <div>
            <h1 className="brand-title">Credit Card Churn & Customer Intelligence</h1>
            <p className="brand-subtitle">
              Portfolio Data Science System · Statistical Inference · Logistic Regression · K-Means · PCA
            </p>
          </div>
        </div>
        <div className="header-status-badge">
          <div className={`status-dot ${isBackendConnected ? '' : 'offline'}`} />
          <span>{isBackendConnected ? 'FastAPI Model Server Online' : 'Connecting to API...'}</span>
        </div>
        <button
          type="button"
          className="mobile-menu-button"
          aria-label={isMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
          aria-expanded={isMenuOpen}
          aria-controls="primary-navigation"
          onClick={() => setIsMenuOpen((open) => !open)}
        >
          {isMenuOpen ? <X size={21} /> : <Menu size={21} />}
        </button>
      </div>

      <nav
        id="primary-navigation"
        className={`nav-tabs-bar ${isMenuOpen ? 'mobile-open' : ''}`}
      >
        <div className="nav-tabs-container">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                className={`nav-tab-button ${isActive ? 'active' : ''}`}
                onClick={() => {
                  setActiveTab(tab.id);
                  setIsMenuOpen(false);
                }}
              >
                <Icon size={16} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </nav>
    </header>
  );
};
