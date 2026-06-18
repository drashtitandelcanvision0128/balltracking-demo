'use client';

import React from 'react';
import type { Analytics } from '../lib/api';

interface AnalyticsPanelProps {
  analytics: Analytics | null;
  avgSpeed?: number;
  maxSpeed?: number;
  avgConfidence?: number;
}

export default function AnalyticsPanel({ analytics, avgSpeed, maxSpeed, avgConfidence }: AnalyticsPanelProps) {
  if (!analytics || analytics.total_balls === 0) {
    return (
      <div style={panelStyle}>
        <div style={titleStyle}>BOWLING ANALYTICS</div>
        <p style={{ color: '#8AA898', fontSize: '0.85rem' }}>Process a video to see analytics</p>
      </div>
    );
  }

  const metrics = [
    { label: 'Dot Ball %', value: `${analytics.dot_ball_pct}%`, color: '#50DC50' },
    { label: 'Boundary %', value: `${analytics.boundary_pct}%`, color: '#FF8C32' },
    { label: 'Wicket %', value: `${analytics.wicket_pct}%`, color: '#F03870' },
    { label: 'Yorker %', value: `${analytics.yorker_pct}%`, color: '#FFD200' },
    { label: 'Good Length %', value: `${analytics.good_length_pct}%`, color: '#38F0B0' },
    { label: 'Short Ball %', value: `${analytics.short_ball_pct}%`, color: '#8AA898' },
    { label: 'Full Toss %', value: `${analytics.full_toss_pct}%`, color: '#B43CC8' },
    { label: 'Avg Speed', value: `${avgSpeed ?? analytics.avg_speed_kmh} km/h`, color: '#5080FF' },
    { label: 'Max Speed', value: `${maxSpeed ?? analytics.max_speed_kmh ?? '—'} km/h`, color: '#FF8C32' },
    { label: 'Pace', value: analytics.pace_label || '—', color: '#38F0B0' },
    { label: 'Consistency', value: `${analytics.bowling_consistency_score}`, color: '#38F0B0' },
    { label: 'Accuracy', value: `${analytics.accuracy_score}%`, color: '#38F0B0' },
  ];

  if (avgConfidence != null) {
    metrics.push({ label: 'Bounce Confidence', value: `${avgConfidence.toFixed(1)}%`, color: '#A3C2B2' });
  }

  return (
    <div style={panelStyle}>
      <div style={titleStyle}>BOWLING ANALYTICS</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
        {metrics.map((m) => (
          <div key={m.label} style={itemStyle}>
            <span style={labelStyle}>{m.label}</span>
            <span style={{ ...valueStyle, color: m.color }}>{m.value}</span>
          </div>
        ))}
      </div>

      <div style={{ marginTop: '12px' }}>
        <div style={{ ...titleStyle, fontSize: '0.7rem', marginBottom: '6px' }}>LENGTH DISTRIBUTION</div>
        {Object.entries(analytics.length_distribution).map(([zone, count]) => (
          count > 0 && (
            <div key={zone} style={barRow}>
              <span style={{ fontSize: '0.7rem', color: '#A3C2B2', width: 120 }}>{zone}</span>
              <div style={barBg}>
                <div style={{ ...barFill, width: `${(count / analytics.total_balls) * 100}%` }} />
              </div>
              <span style={{ fontSize: '0.7rem', color: '#EEF5F0', width: 24 }}>{count}</span>
            </div>
          )
        ))}
      </div>

      <div style={{ marginTop: '12px' }}>
        <div style={{ ...titleStyle, fontSize: '0.7rem', marginBottom: '6px' }}>LINE DISTRIBUTION</div>
        {Object.entries(analytics.line_distribution).map(([zone, count]) => (
          count > 0 && (
            <div key={zone} style={barRow}>
              <span style={{ fontSize: '0.7rem', color: '#A3C2B2', width: 120 }}>{zone}</span>
              <div style={barBg}>
                <div style={{ ...barFill, width: `${(count / analytics.total_balls) * 100}%`, background: '#5080FF' }} />
              </div>
              <span style={{ fontSize: '0.7rem', color: '#EEF5F0', width: 24 }}>{count}</span>
            </div>
          )
        ))}
      </div>
    </div>
  );
}

const panelStyle: React.CSSProperties = {
  background: '#142A24',
  borderRadius: '12px',
  padding: '16px',
  border: '1px solid rgba(56,240,176,0.2)',
};

const titleStyle: React.CSSProperties = {
  fontSize: '0.75rem',
  fontWeight: 700,
  color: '#38F0B0',
  letterSpacing: '0.08em',
  marginBottom: '12px',
};

const itemStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '2px',
};

const labelStyle: React.CSSProperties = {
  fontSize: '0.7rem',
  color: '#8AA898',
};

const valueStyle: React.CSSProperties = {
  fontSize: '1rem',
  fontWeight: 700,
};

const barRow: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  marginBottom: '4px',
};

const barBg: React.CSSProperties = {
  flex: 1,
  height: 6,
  background: 'rgba(255,255,255,0.1)',
  borderRadius: 3,
};

const barFill: React.CSSProperties = {
  height: '100%',
  background: '#38F0B0',
  borderRadius: 3,
};
