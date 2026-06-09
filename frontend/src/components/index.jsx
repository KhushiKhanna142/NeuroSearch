import React from 'react';

/**
 * Summary stat card used in the Search Monitor stats bar.
 * Props: label, value, sub (optional small text below)
 */
export function StatCard({ label, value, sub, color = '#2E75B6' }) {
  return (
    <div style={{
      background: '#fff', border: '1px solid #E2E8F0', borderRadius: 12,
      padding: '18px 22px', flex: 1, minWidth: 160,
      boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
      borderTop: `3px solid ${color}`,
    }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: '#1E3A5F', lineHeight: 1.1 }}>
        {value}
      </div>
      <div style={{ fontSize: 13, color: '#64748B', marginTop: 4, fontWeight: 500 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94A3B8', marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

/**
 * Section header with optional subtitle.
 */
export function SectionHeader({ title, subtitle }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <h2 style={{ fontSize: 18, fontWeight: 700, color: '#1E3A5F' }}>{title}</h2>
      {subtitle && <p style={{ fontSize: 13, color: '#64748B', marginTop: 3 }}>{subtitle}</p>}
    </div>
  );
}

/**
 * Confidence bar for the inference results.
 * Props: label, value (0–1), isTop (boolean)
 */
export function ConfidenceBar({ label, value, isTop }) {
  const pct = (value * 100).toFixed(1);
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 13, fontWeight: isTop ? 600 : 400, color: isTop ? '#1E3A5F' : '#475569' }}>
          {label}
        </span>
        <span style={{ fontSize: 13, fontWeight: 600, color: isTop ? '#27AE60' : '#2E75B6' }}>
          {pct}%
        </span>
      </div>
      <div style={{
        background: '#F1F5F9', borderRadius: 6, height: 10, overflow: 'hidden'
      }}>
        <div style={{
          width: `${pct}%`, height: '100%',
          background: isTop ? '#27AE60' : '#2E75B6',
          borderRadius: 6,
          transition: 'width 0.5s ease',
        }} />
      </div>
    </div>
  );
}

/**
 * Loading spinner.
 */
export function Spinner() {
  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: 48 }}>
      <div style={{
        width: 36, height: 36,
        border: '3px solid #E2E8F0',
        borderTop: '3px solid #2E75B6',
        borderRadius: '50%',
        animation: 'spin 0.8s linear infinite',
      }} />
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

/**
 * Error banner.
 */
export function ErrorBanner({ message }) {
  return (
    <div style={{
      background: '#FEF2F2', border: '1px solid #FECACA', borderRadius: 8,
      padding: '12px 16px', color: '#DC2626', fontSize: 13,
    }}>
      ⚠️ {message}
    </div>
  );
}
