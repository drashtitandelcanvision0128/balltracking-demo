const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

export interface BounceEvent {
  coords?: number[];
  type?: string;
  length?: string;
  length_type?: string;
  line_type?: string;
  bounce_x?: number;
  bounce_y?: number;
  speed_kmh?: number;
  frame?: number;
  bounce_confidence?: number;
  clip_index?: number;
  clip_start?: number;
  clip_end?: number;
}

export interface DeliveryClip {
  index: number;
  start: number;
  end: number;
  release_frame?: number;
  bounce_frame?: number | null;
  outcome?: string;
  length?: string;
  speed_kmh?: number;
  start_time?: number;
  end_time?: number;
  track_frames?: number;
}

export interface JobSummary {
  frames_processed?: number;
  hit_detected?: boolean;
  event_status?: string;
  bounce_events?: BounceEvent[];
  ball_stats?: {
    total?: number;
    dots?: number;
    runs?: number;
    boundaries?: number;
    wickets?: number;
  };
  analytics?: Analytics;
  speed_stats?: {
    avg_speed_kmh?: number;
    max_speed_kmh?: number;
    pace_label?: string;
  };
  clips?: DeliveryClip[];
  clip_count?: number;
  processing_mode?: string;
  fps?: number;
  report_pdf_url?: string;
}

export interface Analytics {
  total_balls: number;
  dot_ball_pct: number;
  boundary_pct: number;
  wicket_pct: number;
  yorker_pct: number;
  good_length_pct: number;
  short_ball_pct: number;
  full_toss_pct: number;
  avg_bounce_x: number;
  avg_bounce_y: number;
  avg_speed_kmh: number;
  max_speed_kmh?: number;
  min_speed_kmh?: number;
  pace_tier?: string;
  pace_label?: string;
  pace_avg_range?: string;
  pace_max_cap?: number;
  length_distribution: Record<string, number>;
  line_distribution: Record<string, number>;
  bowling_consistency_score: number;
  accuracy_score: number;
}

export async function uploadVideo(
  file: File,
  matchId?: string,
  sessionId?: string,
  bowlerId?: string,
  pitchCalibration?: number[][],
) {
  const form = new FormData();
  form.append('video', file);
  if (pitchCalibration && pitchCalibration.length === 4) {
    form.append('pitch_calibration', JSON.stringify(pitchCalibration));
  }
  const params = new URLSearchParams();
  if (matchId) params.set('match_id', matchId);
  if (sessionId) params.set('session_id', sessionId);
  if (bowlerId) params.set('bowler_id', bowlerId);
  const qs = params.toString() ? `?${params}` : '';
  const res = await fetch(`${API_BASE}/api/v1/videos/upload${qs}`, { method: 'POST', body: form });
  return res.json();
}

export async function getJobStatus(jobId: string) {
  const res = await fetch(`${API_BASE}/api/v1/jobs/${jobId}`);
  return res.json();
}

export async function getAnalytics(sessionId?: string) {
  const qs = sessionId ? `?session_id=${sessionId}` : '';
  const res = await fetch(`${API_BASE}/api/v1/analytics${qs}`);
  return res.json() as Promise<Analytics>;
}

export async function getPitchmapData(sessionId?: string) {
  const qs = sessionId ? `?session_id=${sessionId}` : '';
  const res = await fetch(`${API_BASE}/api/v1/pitchmap/data${qs}`);
  return res.json();
}

export async function getHeatmaps(sessionId?: string, zoneFilter = 'all') {
  const params = new URLSearchParams({ zone_filter: zoneFilter });
  if (sessionId) params.set('session_id', sessionId);
  const res = await fetch(`${API_BASE}/api/v1/heatmaps?${params}`);
  return res.json();
}

export function buildVideoUrl(path: string) {
  if (path.startsWith('http')) return path;
  return `${API_BASE}${path}`;
}

export function buildReportPdfUrl(path: string) {
  if (path.startsWith('http')) return path;
  return `${API_BASE}${path}`;
}

/** @deprecated use buildVideoUrl */
export const videoUrl = buildVideoUrl;

export { API_BASE };
