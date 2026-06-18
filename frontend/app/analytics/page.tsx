'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import PitchMapCanvas from '../../components/PitchMapCanvas';
import AnalyticsPanel from '../../components/AnalyticsPanel';
import { getAnalytics, getPitchmapData, API_BASE, type Analytics } from '../../lib/api';

const ZONE_FILTERS = [
  { id: 'all', label: 'Overall' },
  { id: 'yorker', label: 'Yorker' },
  { id: 'good_length', label: 'Good Length' },
  { id: 'short_ball', label: 'Short Ball' },
  { id: 'full_toss', label: 'Full Toss' },
];

export default function AnalyticsPage() {
  const [bounces, setBounces] = useState<Record<string, unknown>[]>([]);
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [zoneFilter, setZoneFilter] = useState('all');
  const [heatmapUrl, setHeatmapUrl] = useState('');
  const [sessionId, setSessionId] = useState('');

  useEffect(() => {
    loadData();
  }, [sessionId, zoneFilter]);

  async function loadData() {
    try {
      const [pitchData, analyticsData] = await Promise.all([
        getPitchmapData(sessionId || undefined),
        getAnalytics(sessionId || undefined),
      ]);
      setBounces(pitchData.bounces || []);
      setAnalytics(analyticsData);
      setHeatmapUrl(
        `${API_BASE}/api/v1/heatmaps?zone_filter=${zoneFilter}&format=image${sessionId ? `&session_id=${sessionId}` : ''}`
      );
    } catch {
      setBounces([]);
    }
  }

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <div style={styles.header}>
          <div>
            <h1 style={styles.title}>Cricket Analytics Dashboard</h1>
            <p style={styles.subtitle}>Real bounce data only — no trajectory prediction</p>
          </div>
          <Link href="/" style={styles.backLink}>← Back to Analysis</Link>
        </div>

        <div style={styles.grid}>
          <div style={styles.panel}>
            <div style={styles.panelTitle}>PITCH MAP</div>
            <PitchMapCanvas bounces={bounces} width={300} />
            <p style={styles.note}>{bounces.length} bounce markers from tracked deliveries</p>
          </div>

          <div style={styles.panel}>
            <div style={styles.panelTitle}>HEATMAP</div>
            <div style={styles.filterRow}>
              {ZONE_FILTERS.map((z) => (
                <button
                  key={z.id}
                  onClick={() => setZoneFilter(z.id)}
                  style={{
                    ...styles.filterBtn,
                    ...(zoneFilter === z.id ? styles.filterActive : {}),
                  }}
                >
                  {z.label}
                </button>
              ))}
            </div>
            {heatmapUrl && (
              <img
                src={heatmapUrl}
                alt="Bowling heatmap"
                style={{ width: '100%', maxWidth: 300, borderRadius: 8, marginTop: 8 }}
              />
            )}
          </div>

          <div style={{ gridColumn: '1 / -1' }}>
            <AnalyticsPanel analytics={analytics} />
          </div>
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    background: '#0A1216',
    minHeight: '100vh',
    padding: '2rem',
    fontFamily: "'Inter', system-ui, sans-serif",
  },
  card: {
    maxWidth: 1100,
    margin: '0 auto',
    background: '#0F1A1A',
    borderRadius: '1.5rem',
    padding: '2rem',
    border: '1px solid rgba(56,240,176,0.2)',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: '2rem',
    borderBottom: '1px solid rgba(56,240,176,0.2)',
    paddingBottom: '1rem',
  },
  title: {
    fontSize: '1.75rem',
    fontWeight: 800,
    color: '#EEF5F0',
    margin: 0,
  },
  subtitle: { color: '#8AA898', marginTop: 4 },
  backLink: {
    color: '#38F0B0',
    textDecoration: 'none',
    fontWeight: 600,
    fontSize: '0.9rem',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '1.5rem',
  },
  panel: {
    background: '#142A24',
    borderRadius: 12,
    padding: 16,
    border: '1px solid rgba(56,240,176,0.15)',
  },
  panelTitle: {
    fontSize: '0.75rem',
    fontWeight: 700,
    color: '#38F0B0',
    letterSpacing: '0.08em',
    marginBottom: 12,
  },
  note: { fontSize: '0.75rem', color: '#8AA898', marginTop: 8 },
  filterRow: { display: 'flex', flexWrap: 'wrap', gap: 6 },
  filterBtn: {
    background: 'transparent',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'rgba(56,240,176,0.3)',
    color: '#A3C2B2',
    padding: '4px 10px',
    borderRadius: 20,
    fontSize: '0.7rem',
    cursor: 'pointer',
  },
  filterActive: {
    background: 'rgba(56,240,176,0.2)',
    color: '#38F0B0',
    borderColor: '#38F0B0',
  },
};
