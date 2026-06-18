"""
Generate PDF session report with clip-by-clip summary, analytics, and pitch map.
"""

from __future__ import annotations

import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from pitch_map_renderer import create_hawkeye_template, render_pitch_map


def _outcome_label(t: str) -> str:
    return {
        "DOTS": "DOT",
        "RUNS": "RUN",
        "BOUNDARIES": "4/6",
        "WICKETS": "OUT",
    }.get(t or "DOTS", t or "DOT")


def _frame_to_time(frame: int, fps: float) -> str:
    if fps <= 0:
        return "—"
    secs = frame / fps
    m = int(secs // 60)
    s = int(secs % 60)
    return f"{m}:{s:02d}"


def generate_session_pdf(
    output_path: str,
    summary: dict,
    *,
    video_name: str = "",
    bowler_name: str = "Bowler",
) -> str:
    """Build a multi-page PDF report. Returns the saved file path."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    bounce_events = summary.get("bounce_events") or []
    ball_stats = summary.get("ball_stats") or {}
    analytics = summary.get("analytics") or {}
    speed_stats = summary.get("speed_stats") or {}
    clips = summary.get("clips") or []
    fps = float(summary.get("fps") or 25.0)
    mode = summary.get("processing_mode", "clip")

    pitch_bounces = [
        {
            "coords": tuple(b["coords"]) if b.get("coords") else (360, 400),
            "type": b.get("type", "DOTS"),
            "length": b.get("length"),
            "speed_kmh": b.get("speed_kmh", 0),
        }
        for b in bounce_events
    ]
    base = create_hawkeye_template(bowler_name)
    map_bgr = render_pitch_map(pitch_bounces, bowler_name=bowler_name, base_img=base)
    map_rgb = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2RGB)

    generated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    with PdfPages(output_path) as pdf:
        # --- Page 1: Cover / summary ---
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        lines = [
            "CRICKET BALL TRACKING REPORT",
            "Clip-by-Clip Analysis Summary",
            "",
            f"Generated: {generated}",
            f"Video: {video_name or '—'}",
            f"Processing: {mode}",
            "",
            "SESSION TOTALS",
            f"  Deliveries tracked: {ball_stats.get('total', len(bounce_events))}",
            f"  DOT balls: {ball_stats.get('dots', 0)}",
            f"  RUN balls: {ball_stats.get('runs', 0)}",
            f"  Boundaries: {ball_stats.get('boundaries', 0)}",
            f"  Wickets: {ball_stats.get('wickets', 0)}",
            "",
            "BOWLING SPEED",
            f"  Average: {speed_stats.get('avg_speed_kmh') or analytics.get('avg_speed_kmh', '—')} km/h",
            f"  Maximum: {speed_stats.get('max_speed_kmh') or analytics.get('max_speed_kmh', '—')} km/h",
            f"  Pace tier: {speed_stats.get('pace_label') or analytics.get('pace_label', '—')}",
            "",
            "PITCH ANALYTICS",
            f"  Accuracy score: {analytics.get('accuracy_score', '—')}%",
            f"  Consistency: {analytics.get('bowling_consistency_score', '—')}%",
            f"  Avg bounce (Y): {analytics.get('avg_bounce_y', '—')} m",
        ]
        ax.text(0.08, 0.92, "\n".join(lines), va="top", fontsize=11, family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- Page 2: Clip-by-clip table ---
        if clips:
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            ax.set_title("Clip-by-Clip Tracking", fontsize=14, fontweight="bold", pad=16)
            headers = ["#", "Start", "End", "Bounce", "Length", "Speed", "Outcome"]
            rows = []
            for c in clips:
                idx = c.get("index", "")
                start = _frame_to_time(c.get("start", 0), fps)
                end = _frame_to_time(c.get("end", 0), fps)
                bounce_f = c.get("bounce_frame")
                bounce_t = _frame_to_time(bounce_f, fps) if bounce_f else "—"
                rows.append([
                    str(idx),
                    start,
                    end,
                    bounce_t,
                    c.get("length") or "—",
                    f"{int(c.get('speed_kmh') or 0)} km/h" if c.get("speed_kmh") else "—",
                    _outcome_label(c.get("outcome", "")),
                ])
            table = ax.table(
                cellText=rows,
                colLabels=headers,
                loc="upper center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.4)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # --- Page 3: Delivery detail ---
        if bounce_events:
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            ax.set_title("Per-Delivery Details", fontsize=14, fontweight="bold", pad=16)
            headers = ["#", "Clip", "Frame", "Length", "Line", "Speed", "Conf", "Result"]
            rows = []
            for i, b in enumerate(bounce_events, 1):
                rows.append([
                    str(i),
                    str(b.get("clip_index", "—")),
                    str(b.get("frame", "—")),
                    b.get("length") or "—",
                    b.get("line_type") or "—",
                    f"{int(b.get('speed_kmh') or 0)} km/h" if b.get("speed_kmh") else "—",
                    f"{(b.get('bounce_confidence') or 0) * 100:.0f}%",
                    _outcome_label(b.get("type", "")),
                ])
            table = ax.table(cellText=rows, colLabels=headers, loc="upper center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.35)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # --- Page 4: Pitch map ---
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.imshow(map_rgb)
        ax.axis("off")
        ax.set_title("Pitch Map — All Deliveries", fontsize=14, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return output_path
