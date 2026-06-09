import React, { useState, useEffect } from 'react';
import { BrainCircuit, GitFork, Activity, Network, Play } from 'lucide-react';
import SearchMonitor from './tabs/SearchMonitor';
import ArchitectureViewer from './tabs/ArchitectureViewer';
import InferenceDemo from './tabs/InferenceDemo';
import { api } from './api';
import './index.css';

const TABS = [
  { id: 'monitor',      label: 'Search Monitor',      icon: Activity },
  { id: 'architecture', label: 'Architecture Viewer',  icon: Network  },
  { id: 'inference',    label: 'Inference Demo',       icon: Play     },
];

const STATUS_COLORS = {
  idle:        { bg: '#95A5A6', label: 'Idle' },
  pretraining: { bg: '#E67E22', label: 'Pretraining' },
  searching:   { bg: '#2E75B6', label: 'Searching' },
  complete:    { bg: '#27AE60', label: 'Complete' },
};

export default function App() {
  const [activeTab, setActiveTab] = useState('monitor');
  const [status, setStatus]       = useState('idle');

  useEffect(() => {
    const poll = () => api.getStatus().then(r => setStatus(r.data.status)).catch(() => {});
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, []);

  const sc = STATUS_COLORS[status] || STATUS_COLORS.idle;

  return (
    <div style={{ fontFamily: "'Inter', sans-serif", minHeight: '100vh', background: '#F8FAFC' }}>
      {/* NAV BAR */}
      <nav style={{
        background: '#1E3A5F', padding: '0 32px', display: 'flex',
        alignItems: 'center', justifyContent: 'space-between', height: 60,
        boxShadow: '0 2px 12px rgba(0,0,0,0.18)', position: 'sticky', top: 0, zIndex: 100,
      }}>
        {/* Logo + Title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <BrainCircuit size={28} color="#E8F4FD" />
          <span style={{ color: '#fff', fontSize: 20, fontWeight: 700, letterSpacing: '-0.3px' }}>
            NeuroSearch
          </span>
          <span style={{ color: '#93C5FD', fontSize: 13, marginLeft: 4, opacity: 0.8 }}>
            NAS with RL
          </span>
        </div>

        {/* Tab Buttons */}
        <div style={{ display: 'flex', gap: 4 }}>
          {TABS.map(tab => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
                background: active ? 'rgba(255,255,255,0.12)' : 'transparent',
                border: 'none', color: active ? '#fff' : '#93C5FD',
                padding: '8px 16px', borderRadius: 8, cursor: 'pointer',
                fontFamily: 'inherit', fontSize: 14, fontWeight: active ? 600 : 400,
                display: 'flex', alignItems: 'center', gap: 7,
                borderBottom: active ? '2px solid #E8F4FD' : '2px solid transparent',
                transition: 'all 0.2s',
              }}>
                <Icon size={15} />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* GitHub + Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <span style={{
            background: sc.bg, color: '#fff', fontSize: 12, fontWeight: 600,
            padding: '4px 12px', borderRadius: 20, letterSpacing: '0.3px',
          }}>{sc.label}</span>
          <a href="https://github.com/KhushiKhanna142/NeuroSearch" target="_blank" rel="noreferrer"
            style={{ color: '#93C5FD', display: 'flex', alignItems: 'center' }}>
            <GitFork size={20} />
          </a>
        </div>
      </nav>

      {/* CONTENT */}
      <main style={{ maxWidth: 1200, margin: '0 auto', padding: '28px 24px' }}>
        {activeTab === 'monitor'      && <SearchMonitor />}
        {activeTab === 'architecture' && <ArchitectureViewer />}
        {activeTab === 'inference'    && <InferenceDemo />}
      </main>
    </div>
  );
}
