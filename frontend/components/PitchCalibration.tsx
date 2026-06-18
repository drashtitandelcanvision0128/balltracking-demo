"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";

export type PitchPoint = [number, number];

const CORNER_LABELS = ["Top-Left (bowler end)", "Top-Right", "Bottom-Right", "Bottom-Left (batsman end)"];

interface PitchCalibrationProps {
  videoUrl: string;
  onChange: (quad: PitchPoint[] | null) => void;
}

export default function PitchCalibration({ videoUrl, onChange }: PitchCalibrationProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [points, setPoints] = useState<PitchPoint[]>([]);
  const [active, setActive] = useState(false);
  const [videoSize, setVideoSize] = useState({ w: 0, h: 0 });

  const syncCanvas = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = rect.width / (video.videoWidth || rect.width);
    const scaleY = rect.height / (video.videoHeight || rect.height);

    const drawPts = points.map(([x, y]) => [x * scaleX, y * scaleY] as PitchPoint);

    if (drawPts.length >= 2) {
      ctx.strokeStyle = "#00ff88";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(drawPts[0][0], drawPts[0][1]);
      for (let i = 1; i < drawPts.length; i++) {
        ctx.lineTo(drawPts[i][0], drawPts[i][1]);
      }
      if (drawPts.length === 4) {
        ctx.closePath();
        ctx.fillStyle = "rgba(0, 255, 136, 0.12)";
        ctx.fill();
      }
      ctx.stroke();
    }

    drawPts.forEach(([x, y], i) => {
      ctx.fillStyle = i < 2 ? "#ffcc00" : "#00ccff";
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "12px sans-serif";
      ctx.fillText(String(i + 1), x + 10, y - 6);
    });
  }, [points]);

  useEffect(() => {
    syncCanvas();
  }, [syncCanvas, points, videoUrl]);

  useEffect(() => {
    if (points.length === 4) {
      onChange(points);
    } else {
      onChange(null);
    }
  }, [points, onChange]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!active) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || points.length >= 4) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (video.videoWidth / rect.width);
    const y = (e.clientY - rect.top) * (video.videoHeight / rect.height);
    setPoints((prev) => [...prev, [Math.round(x), Math.round(y)]]);
  };

  const reset = () => {
    setPoints([]);
    onChange(null);
  };

  return (
    <div style={{ marginTop: 12, padding: 12, background: "rgba(0,0,0,0.35)", borderRadius: 8, border: "1px solid #333" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <strong style={{ color: "#e0e0e0", fontSize: 14 }}>Pitch Calibration (optional — improves accuracy)</strong>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            type="button"
            onClick={() => setActive((v) => !v)}
            style={{
              padding: "6px 12px",
              background: active ? "#1a6b3c" : "#2a4a6b",
              color: "#fff",
              border: "none",
              borderRadius: 4,
              cursor: "pointer",
              fontSize: 12,
            }}
          >
            {active ? "Clicking corners…" : "Calibrate Pitch"}
          </button>
          <button
            type="button"
            onClick={reset}
            style={{
              padding: "6px 12px",
              background: "#444",
              color: "#fff",
              border: "none",
              borderRadius: 4,
              cursor: "pointer",
              fontSize: 12,
            }}
          >
            Reset
          </button>
        </div>
      </div>
      <p style={{ color: "#aaa", fontSize: 12, margin: "0 0 8px" }}>
        Click 4 pitch corners in order: {CORNER_LABELS.join(" → ")}.
        {points.length < 4 && active ? ` (${points.length}/4)` : points.length === 4 ? " ✓ Done" : ""}
      </p>
      <div style={{ position: "relative", display: "inline-block", maxWidth: "100%" }}>
        <video
          ref={videoRef}
          src={videoUrl}
          muted
          playsInline
          style={{ maxWidth: "100%", maxHeight: 280, display: "block", borderRadius: 6 }}
          onLoadedMetadata={(e) => {
            const v = e.currentTarget;
            setVideoSize({ w: v.videoWidth, h: v.videoHeight });
            syncCanvas();
          }}
        />
        <canvas
          ref={canvasRef}
          onClick={handleClick}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            cursor: active && points.length < 4 ? "crosshair" : "default",
          }}
        />
      </div>
      {videoSize.w > 0 && (
        <p style={{ color: "#666", fontSize: 11, marginTop: 6 }}>
          Video: {videoSize.w}×{videoSize.h}px — manual calibration overrides auto-detection
        </p>
      )}
    </div>
  );
}
