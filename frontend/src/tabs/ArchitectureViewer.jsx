import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { SectionHeader, Spinner, ErrorBanner } from '../components';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { api } from '../api';

const OP_COLORS = {
  conv3x3:      '#2E75B6',
  conv1x1:      '#60A5FA',
  sep3x3:       '#0D9488',
  sep5x5:       '#34D399',
  maxpool:      '#E67E22',
  avgpool:      '#F59E0B',
  skip:         '#27AE60',
  zero:         '#95A5A6',
  FactorizedReduce: '#8B5CF6',
  Identity:     '#27AE60',
  ConvBNReLU:   '#2E75B6',
  SepConv:      '#0D9488',
  MaxPool2d:    '#E67E22',
  AvgPool2d:    '#F59E0B',
  Zero:         '#95A5A6',
};
const getOpColor = (op) => OP_COLORS[op] || '#94A3B8';

const EDGE_KEYS = [
  '0_2','1_2','0_3','1_3','2_3',
  '0_4','1_4','2_4','3_4',
  '0_5','1_5','2_5','3_5','4_5',
];

const NODE_LABELS = { 0: 'Input 0', 1: 'Input 1', 2: 'Node A', 3: 'Node B', 4: 'Node C', 5: 'Node D' };

export default function ArchitectureViewer() {
  const [results, setResults] = useState(null);
  const [selected, setSelected] = useState(1);
  const [archData, setArchData] = useState(null);
  const [finetuneData, setFinetuneData] = useState(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);
  const svgRef = useRef(null);

  // Load top-5 results
  useEffect(() => {
    api.getSearchResults()
      .then(r => { setResults(r.data.top5); setLoading(false); })
      .catch(() => { setError('Cannot reach backend — is it running on port 8000?'); setLoading(false); });

    api.getFinetuneResults()
      .then(r => setFinetuneData(r.data))
      .catch(() => {});
  }, []);

  // Load selected architecture
  useEffect(() => {
    if (!results) return;
    api.getArchitecture(selected)
      .then(r => setArchData(r.data))
      .catch(() => setArchData(null));
  }, [selected, results]);

  // Draw DAG when archData changes
  useEffect(() => {
    if (!archData?.arch_spec) return;
    const cellSpec = archData.arch_spec[0]; // Show normal cell (first cell)
    if (cellSpec) drawDAG(svgRef.current, cellSpec);
  }, [archData]);

  if (loading) return <Spinner />;
  if (error)   return <ErrorBanner message={error} />;
  if (!results) return <ErrorBanner message="No search results found." />;

  // Op distribution from arch_spec
  const opCounts = {};
  if (archData?.arch_spec) {
    archData.arch_spec.forEach(cell => {
      Object.values(cell).forEach(op => {
        opCounts[op] = (opCounts[op] || 0) + 1;
      });
    });
  }
  const opDistData = Object.entries(opCounts).map(([op, count]) => ({ op, count }))
    .sort((a, b) => b.count - a.count);

  const totalOps = opDistData.reduce((s, d) => s + d.count, 0);

  return (
    <div>
      <SectionHeader title="Architecture Viewer"
        subtitle="Leaderboard of best architectures and the cell DAG for each" />

      <div style={{ display: 'flex', gap: 20 }}>
        {/* Left: Leaderboard */}
        <div style={{ width: '35%', flexShrink: 0 }}>
          <div style={{
            background: '#fff', border: '1px solid #E2E8F0', borderRadius: 12,
            overflow: 'hidden', boxShadow: '0 1px 4px rgba(0,0,0,0.05)',
          }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#1E3A5F', color: '#fff' }}>
                  {['Rank', 'Accuracy', 'FLOPs', 'Reward', 'Episode'].map(h => (
                    <th key={h} style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, fontSize: 12 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => {
                  const isSelected = selected === r.rank;
                  return (
                    <tr key={r.rank} onClick={() => setSelected(r.rank)}
                      style={{
                        background: isSelected ? '#E8F4FD' : i % 2 === 0 ? '#F8FAFC' : '#fff',
                        cursor: 'pointer', borderBottom: '1px solid #F1F5F9',
                        transition: 'background 0.15s',
                      }}>
                      <td style={{ padding: '10px 12px', fontWeight: 700, color: '#1E3A5F' }}>
                        {r.rank === 1 ? '🥇' : r.rank === 2 ? '🥈' : r.rank === 3 ? '🥉' : `#${r.rank}`}
                      </td>
                      <td style={{ padding: '10px 12px', fontWeight: 600, color: '#27AE60' }}>
                        {((r.rank === 1 && finetuneData?.val_acc ? finetuneData.val_acc : r.accuracy) * 100).toFixed(1)}%
                        {r.rank === 1 && finetuneData?.val_acc && (
                          <span style={{ fontSize: 9, display: 'block', color: '#1E3A5F', fontWeight: 500 }}>
                            (fine-tuned)
                          </span>
                        )}
                      </td>
                      <td style={{ padding: '10px 12px' }}>{(r.flops / 1e6).toFixed(1)}M</td>
                      <td style={{ padding: '10px 12px', color: '#2E75B6', fontWeight: 500 }}>
                        {r.reward.toFixed(3)}
                      </td>
                      <td style={{ padding: '10px 12px', color: '#64748B' }}>Ep {r.episode}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right: DAG + Op Distribution */}
        <div style={{ flex: 1 }}>
          {/* DAG */}
          <div style={{
            background: '#fff', border: '1px solid #E2E8F0', borderRadius: 12,
            padding: 20, marginBottom: 16, boxShadow: '0 1px 4px rgba(0,0,0,0.05)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
              <div>
                <div style={{ fontSize: 15, fontWeight: 600, color: '#1E3A5F' }}>Cell DAG</div>
                <div style={{ fontSize: 11, color: '#94A3B8', marginTop: 2 }}>Normal cell — op choices baked in by the controller</div>
              </div>
              {archData && (
                <div style={{ fontSize: 12, color: '#64748B', textAlign: 'right' }}>
                  <span style={{ fontWeight: 600, color: '#27AE60' }}>
                    {((selected === 1 && finetuneData?.val_acc ? finetuneData.val_acc : archData.accuracy) * 100).toFixed(1)}%
                  </span> acc {selected === 1 && finetuneData?.val_acc && "(fine-tuned)"} &nbsp;|&nbsp;
                  <span style={{ fontWeight: 600, color: '#2E75B6' }}>{(archData.flops / 1e6).toFixed(1)}M</span> FLOPs
                </div>
              )}
            </div>

            {archData?.arch_spec ? (
              <>
                <svg ref={svgRef} width="100%" height="280" />
                {/* Op Legend */}
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 12 }}>
                  {Object.entries(OP_COLORS).slice(0, 8).map(([op, color]) => (
                    <div key={op} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11 }}>
                      <div style={{ width: 10, height: 10, borderRadius: 2, background: color }} />
                      <span style={{ color: '#475569' }}>{op}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div style={{ textAlign: 'center', padding: 40, color: '#94A3B8', fontSize: 13 }}>
                Detailed arch spec not available — run the full pipeline to enable DAG view
              </div>
            )}
          </div>

          {/* Op Distribution */}
          {opDistData.length > 0 && (
            <div style={{
              background: '#fff', border: '1px solid #E2E8F0', borderRadius: 12,
              padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.05)',
            }}>
              <div style={{ fontSize: 15, fontWeight: 600, color: '#1E3A5F', marginBottom: 4 }}>Op Distribution</div>
              <div style={{ fontSize: 11, color: '#94A3B8', marginBottom: 14 }}>
                Breakdown of op types across all edges in this architecture
              </div>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={opDistData} layout="vertical" margin={{ left: 60 }}>
                  <XAxis type="number" tick={{ fontSize: 11 }}
                    tickFormatter={v => `${Math.round(v / totalOps * 100)}%`} />
                  <YAxis type="category" dataKey="op" tick={{ fontSize: 11 }} width={70} />
                  <Tooltip formatter={(v) => [`${Math.round(v / totalOps * 100)}%`, 'Share']}
                    contentStyle={{ fontSize: 12 }} />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                    {opDistData.map((d, i) => (
                      <Cell key={i} fill={getOpColor(d.op)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ---- D3 DAG Drawing ----
function drawDAG(svgEl, cellSpec) {
  if (!svgEl) return;
  const svg = d3.select(svgEl);
  svg.selectAll('*').remove();

  const W = svgEl.clientWidth || 500;
  const H = 280;
  const nodeR = 22;

  // Node positions: inputs on left, intermediates in layers
  const positions = {
    0: { x: W * 0.08, y: H * 0.35, label: 'In 0', type: 'input' },
    1: { x: W * 0.08, y: H * 0.65, label: 'In 1', type: 'input' },
    2: { x: W * 0.30, y: H * 0.25, label: 'A', type: 'mid' },
    3: { x: W * 0.50, y: H * 0.50, label: 'B', type: 'mid' },
    4: { x: W * 0.68, y: H * 0.25, label: 'C', type: 'mid' },
    5: { x: W * 0.88, y: H * 0.50, label: 'Out', type: 'output' },
  };

  const defs = svg.append('defs');
  const marker = defs.append('marker')
    .attr('id', 'arrow').attr('markerWidth', 8).attr('markerHeight', 6)
    .attr('refX', 6).attr('refY', 3).attr('orient', 'auto');
  marker.append('polygon').attr('points', '0 0, 8 3, 0 6').attr('fill', '#94A3B8');

  // Edges
  Object.entries(cellSpec).forEach(([key, op]) => {
    const [from, to] = key.split('_').map(Number);
    const src = positions[from];
    const tgt = positions[to];
    if (!src || !tgt) return;

    const color = getOpColor(op);
    const dx = tgt.x - src.x;
    const dy = tgt.y - src.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const nx = dx / dist;
    const ny = dy / dist;
    const x1 = src.x + nx * nodeR;
    const y1 = src.y + ny * nodeR;
    const x2 = tgt.x - nx * (nodeR + 8);
    const y2 = tgt.y - ny * (nodeR + 8);

    // Curved path
    const mx = (x1 + x2) / 2 - ny * 20;
    const my = (y1 + y2) / 2 + nx * 20;

    svg.append('path')
      .attr('d', `M${x1},${y1} Q${mx},${my} ${x2},${y2}`)
      .attr('stroke', color).attr('stroke-width', 2).attr('fill', 'none')
      .attr('marker-end', 'url(#arrow)').attr('opacity', 0.8);

    // Edge label
    svg.append('text')
      .attr('x', (x1 + x2) / 2 - ny * 22).attr('y', (y1 + y2) / 2 + nx * 22)
      .attr('text-anchor', 'middle').attr('font-size', 9).attr('fill', color).attr('font-weight', 600)
      .text(op);
  });

  // Nodes
  Object.entries(positions).forEach(([id, pos]) => {
    const g = svg.append('g').attr('transform', `translate(${pos.x},${pos.y})`);
    const fill = pos.type === 'input' ? '#95A5A6' : pos.type === 'output' ? '#1E3A5F' : '#2E75B6';
    g.append('circle').attr('r', nodeR).attr('fill', fill).attr('stroke', '#fff').attr('stroke-width', 2);
    g.append('text').attr('text-anchor', 'middle').attr('dy', 4)
      .attr('font-size', 10).attr('fill', '#fff').attr('font-weight', 600)
      .text(pos.label);
  });
}
