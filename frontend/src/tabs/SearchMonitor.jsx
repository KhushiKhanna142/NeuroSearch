import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ReferenceLine, ResponsiveContainer, ScatterChart, Scatter,
  BarChart, Bar, Cell,
} from 'recharts';
import { StatCard, SectionHeader, Spinner, ErrorBanner } from '../components';
import { api } from '../api';

const MAX_ENT = 2.079; // log(8) — maximum possible entropy over 8 ops

export default function SearchMonitor() {
  const [data,    setData]    = useState(null);
  const [finetuneData, setFinetuneData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  const fetchData = () => {
    api.getSearchLog()
      .then(r => { setData(r.data.episodes); setLoading(false); setError(null); })
      .catch(e => { setError('Cannot reach backend — is it running on port 8000?'); setLoading(false); });

    api.getFinetuneResults()
      .then(r => setFinetuneData(r.data))
      .catch(() => {});
  };

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 5000);
    return () => clearInterval(id);
  }, []);

  if (loading) return <Spinner />;
  if (error)   return <ErrorBanner message={error} />;
  if (!data || data.length === 0) return (
    <div style={{ textAlign: 'center', padding: 60, color: '#64748B' }}>
      <div style={{ fontSize: 48, marginBottom: 12 }}>🔍</div>
      <div style={{ fontSize: 18, fontWeight: 600 }}>No search data yet</div>
      <div style={{ fontSize: 14, marginTop: 6 }}>Run the pipeline: <code>python3 -m nas_rl.train</code></div>
    </div>
  );

  // Derived stats
  const bestAcc  = Math.max(...data.map(d => d.accuracy));
  const bestFlops= data.find(d => d.accuracy === bestAcc)?.flops || 0;
  const unique   = new Set(data.map(d => JSON.stringify({ a: d.accuracy, f: d.flops }))).size;

  // FLOPs histogram
  const buckets = [
    { label: '0-20M', min: 0,   max: 20e6  },
    { label: '20-40M',min: 20e6,max: 40e6  },
    { label: '40-60M',min: 40e6,max: 60e6  },
    { label: '60-80M',min: 60e6,max: 80e6  },
    { label: '80M+',  min: 80e6,max: Infinity },
  ];
  const histData = buckets.map(b => {
    const inBucket = data.filter(d => d.flops >= b.min && d.flops < b.max);
    const avgAcc = inBucket.length
      ? inBucket.reduce((s, d) => s + d.accuracy, 0) / inBucket.length
      : 0;
    return { label: b.label, count: inBucket.length, avgAcc };
  });

  // Scatter points with color gradient by episode index
  const scatterData = data.map((d, i) => ({
    x: +(d.flops / 1e6).toFixed(2),
    y: +(d.accuracy * 100).toFixed(2),
    episode: d.episode,
    progress: i / Math.max(data.length - 1, 1),
  }));

  const top5Set = new Set(
    [...data].sort((a, b) => b.reward - a.reward).slice(0, 5).map(d => d.episode)
  );

  const cardColor = (i) => ['#2E75B6', '#27AE60', '#E67E22', '#8B5CF6'][i % 4];

  return (
    <div>
      <SectionHeader title="Search Monitor"
        subtitle="Live view of the NAS training process — updates every 5 seconds during an active run" />

      {/* Fine-Tuning Banner */}
      {finetuneData && finetuneData.test_top1 && (
        <div style={{
          background: 'linear-gradient(135deg, #1E3A5F, #2E75B6)',
          borderRadius: 12, padding: '16px 20px', color: '#fff',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          marginBottom: 20, boxShadow: '0 4px 12px rgba(46,117,182,0.15)',
        }}>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6 }}>
              <span>🏆</span> Post-Search Fine-Tuned Model Results
            </div>
            <div style={{ fontSize: 12, opacity: 0.85, marginTop: 2 }}>
              Best candidate fine-tuned for {finetuneData.epochs} epochs on full CIFAR-10 training set
            </div>
          </div>
          <div style={{ display: 'flex', gap: 24, textAlign: 'right', flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontSize: 20, fontWeight: 800, color: '#4ADE80' }}>{(finetuneData.test_top1 * 100).toFixed(1)}%</div>
              <div style={{ fontSize: 10, opacity: 0.8 }}>Top-1 Test Acc</div>
            </div>
            <div>
              <div style={{ fontSize: 20, fontWeight: 800, color: '#38BDF8' }}>{(finetuneData.test_top5 * 100).toFixed(1)}%</div>
              <div style={{ fontSize: 10, opacity: 0.8 }}>Top-5 Test Acc</div>
            </div>
            <div>
              <div style={{ fontSize: 20, fontWeight: 800 }}>{(finetuneData.params / 1e3).toFixed(0)}K</div>
              <div style={{ fontSize: 10, opacity: 0.8 }}>Parameters</div>
            </div>
          </div>
        </div>
      )}

      {/* Stats Bar */}
      <div style={{ display: 'flex', gap: 14, marginBottom: 28, flexWrap: 'wrap' }}>
        <StatCard label="Episodes Run"      value={data.length}                        color={cardColor(0)} />
        <StatCard label="Best Val Accuracy" value={`${(bestAcc * 100).toFixed(1)}%`}   color={cardColor(1)} />
        <StatCard label="Best FLOPs"        value={`${(bestFlops / 1e6).toFixed(1)}M`} color={cardColor(2)} />
        <StatCard label="Unique Archs Found"value={unique}                              color={cardColor(3)} />
      </div>

      {/* Charts grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>

        {/* Chart 1: Reward Curve */}
        <ChartCard title="Reward Curve" subtitle="Raw vs Normalised reward over episodes">
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={data} margin={{ top: 10, right: 20, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
              <XAxis dataKey="episode" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ fontSize: 12 }} />
              <Legend iconSize={10} wrapperStyle={{ fontSize: 12 }} />
              <ReferenceLine y={0} strokeDasharray="4 4" stroke="#94A3B8" label="" />
              <Line type="monotone" dataKey="raw_reward"  stroke="#94A3B8" dot={false} name="Raw"        strokeWidth={1.5} />
              <Line type="monotone" dataKey="reward"      stroke="#2E75B6" dot={false} name="Normalised" strokeWidth={2.5} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Chart 2: Entropy Curve */}
        <ChartCard title="Controller Entropy" subtitle="Decreases as the controller converges">
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={data} margin={{ top: 10, right: 20, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
              <XAxis dataKey="episode" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 2.5]} tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ fontSize: 12 }} />
              <ReferenceLine y={MAX_ENT} strokeDasharray="4 4" stroke="#94A3B8"
                label={{ value: 'max', position: 'right', fontSize: 10, fill: '#94A3B8' }} />
              <ReferenceLine y={1.0} strokeDasharray="4 4" stroke="#EF4444"
                label={{ value: 'collapse ⚠', position: 'right', fontSize: 10, fill: '#EF4444' }} />
              <Line type="monotone" dataKey="entropy" stroke="#E67E22" dot={false} name="Entropy" strokeWidth={2.5} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Chart 3: Accuracy vs FLOPs Scatter */}
        <ChartCard title="Accuracy vs FLOPs" subtitle="Each point is one evaluated architecture — top-5 shown larger">
          <ResponsiveContainer width="100%" height={240}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
              <XAxis dataKey="x" name="FLOPs (M)" unit="M" tick={{ fontSize: 11 }}
                label={{ value: 'FLOPs (M)', position: 'insideBottom', offset: -12, fontSize: 11 }} />
              <YAxis dataKey="y" name="Accuracy" unit="%" tick={{ fontSize: 11 }}
                label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft', fontSize: 11 }} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ fontSize: 12 }}
                formatter={(v, n) => [n === 'x' ? `${v}M` : `${v}%`, n === 'x' ? 'FLOPs' : 'Accuracy']} />
              <Scatter data={scatterData} shape={(props) => {
                const { cx, cy, payload } = props;
                const isTop = top5Set.has(payload.episode);
                const blue = Math.round(46 + payload.progress * (182 - 46));
                const color = `rgb(46, ${blue}, 182)`;
                return isTop
                  ? <polygon points={`${cx},${cy - 8} ${cx + 7},${cy + 5} ${cx - 7},${cy + 5}`}
                      fill="#F59E0B" stroke="#D97706" strokeWidth={1} />
                  : <circle cx={cx} cy={cy} r={4} fill={color} opacity={0.75} />;
              }} />
            </ScatterChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 11, color: '#94A3B8', marginTop: 4 }}>
            ▲ Top-5 architectures &nbsp;● All architectures (grey→blue = early→late episodes)
          </div>
        </ChartCard>

        {/* Chart 4: FLOPs Distribution */}
        <ChartCard title="FLOPs Distribution" subtitle="Bars shaded by average accuracy — darker = more accurate">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={histData} margin={{ top: 10, right: 20, bottom: 20, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ fontSize: 12 }}
                formatter={(v, n, p) => [v, n === 'count' ? 'Architectures' : 'Avg Acc %']} />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {histData.map((entry, i) => {
                  const intensity = Math.round(46 + entry.avgAcc * 120);
                  return <Cell key={i} fill={`rgb(30, 58, ${intensity})`} />;
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

      </div>
    </div>
  );
}

function ChartCard({ title, subtitle, children }) {
  return (
    <div style={{
      background: '#fff', border: '1px solid #E2E8F0', borderRadius: 12,
      padding: '18px 20px', boxShadow: '0 1px 4px rgba(0,0,0,0.05)',
    }}>
      <div style={{ marginBottom: 14 }}>
        <div style={{ fontSize: 15, fontWeight: 600, color: '#1E3A5F' }}>{title}</div>
        {subtitle && <div style={{ fontSize: 11, color: '#94A3B8', marginTop: 2 }}>{subtitle}</div>}
      </div>
      {children}
    </div>
  );
}
