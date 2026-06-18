'use client';

import React, { useRef, useEffect } from 'react';

interface Bounce {
  coords?: number[];
  bounce_x?: number;
  bounce_y?: number;
  type?: string;
  length_type?: string;
  line_type?: string;
}

const PITCH_L = 185, PITCH_R = 535, PITCH_TOP = 70, PITCH_BOT = 780;
const MAP_W = 720, MAP_H = 960;

const OUTCOME_COLORS: Record<string, string> = {
  DOTS: '#3C3C3C',
  RUNS: '#FFD200',
  BOUNDARIES: '#DC7820',
  WICKETS: '#FFFFFF',
};

const ZONES = [
  { y1: 70, y2: 130, color: '#B43CC8', label: 'FULL TOSS' },
  { y1: 130, y2: 195, color: '#FFC800', label: 'YORKER' },
  { y1: 195, y2: 295, color: '#78C8FF', label: 'HALF VOLLEY' },
  { y1: 295, y2: 395, color: '#50C850', label: 'FULL' },
  { y1: 395, y2: 495, color: '#FF8C32', label: 'LENGTH' },
  { y1: 495, y2: 600, color: '#DC3232', label: 'BACK OF LENGTH' },
  { y1: 600, y2: 780, color: '#8C8C8C', label: 'SHORT' },
];

interface PitchMapCanvasProps {
  bounces: Bounce[];
  width?: number;
  onBounceClick?: (index: number) => void;
}

export default function PitchMapCanvas({ bounces, width = 280, onBounceClick }: PitchMapCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const height = Math.round(width * (MAP_H / MAP_W));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const scaleX = width / MAP_W;
    const scaleY = height / MAP_H;

    ctx.fillStyle = '#286428';
    ctx.fillRect(0, 0, width, height);

    for (const zone of ZONES) {
      ctx.fillStyle = zone.color + '99';
      ctx.fillRect(
        PITCH_L * scaleX, zone.y1 * scaleY,
        (PITCH_R - PITCH_L) * scaleX, (zone.y2 - zone.y1) * scaleY
      );
    }

    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;
    ctx.strokeRect(PITCH_L * scaleX, PITCH_TOP * scaleY, (PITCH_R - PITCH_L) * scaleX, (PITCH_BOT - PITCH_TOP) * scaleY);

    const cx = ((PITCH_L + PITCH_R) / 2) * scaleX;
    ctx.beginPath();
    ctx.moveTo(cx, PITCH_TOP * scaleY);
    ctx.lineTo(cx, PITCH_BOT * scaleY);
    ctx.strokeStyle = '#C8783C';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    bounces.forEach((b, i) => {
      let px: number, py: number;
      if (b.coords && b.coords.length >= 2) {
        [px, py] = b.coords;
      } else if (b.bounce_x != null && b.bounce_y != null) {
        px = PITCH_L + (b.bounce_x / 3.05 + 0.5) * (PITCH_R - PITCH_L);
        py = PITCH_TOP + (b.bounce_y / 20.12) * (PITCH_BOT - PITCH_TOP);
      } else return;

      const x = px * scaleX;
      const y = py * scaleY;
      const color = OUTCOME_COLORS[b.type || 'DOTS'] || OUTCOME_COLORS.DOTS;

      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  }, [bounces, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{ borderRadius: 8, border: '1px solid rgba(56,240,176,0.3)' }}
      onClick={(e) => {
        if (!onBounceClick) return;
        const rect = canvasRef.current?.getBoundingClientRect();
        if (!rect) return;
        const x = (e.clientX - rect.left) / rect.width * MAP_W;
        const y = (e.clientY - rect.top) / rect.height * MAP_H;
        let best = -1, bestD = 30;
        bounces.forEach((b, i) => {
          const [bx, by] = b.coords || [0, 0];
          const d = Math.hypot(bx - x, by - y);
          if (d < bestD) { bestD = d; best = i; }
        });
        if (best >= 0) onBounceClick(best);
      }}
    />
  );
}
