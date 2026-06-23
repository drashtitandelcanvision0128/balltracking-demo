"""
Cricket bowling speed calibration — realistic km/h from tracked ball positions.

Uses homography (pitch metres) when available; falls back to scaled pixels.
Output clamped to ICC-realistic bowling ranges per pace tier.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.config import CONFIG
from core.pitch_coords import video_to_world

_SPEED = CONFIG.get("bowling_speed", {})

GLOBAL_MIN_KMH = float(_SPEED.get("global_min_kmh", 70))
GLOBAL_MAX_KMH = float(_SPEED.get("global_max_kmh", 180))

PACE_TIERS: list[dict[str, Any]] = _SPEED.get(
    "tiers",
    [
        {
            "id": "medium",
            "label": "Medium Pace",
            "avg_min": 80,
            "avg_max": 120,
            "max_speed": 140,
        },
        {
            "id": "fast_medium",
            "label": "Fast Medium",
            "avg_min": 120,
            "avg_max": 150,
            "max_speed": 160,
        },
        {
            "id": "fast",
            "label": "Fast Bowling",
            "avg_min": 130,
            "avg_max": 160,
            "max_speed": 180,
        },
    ],
)


def clamp_speed(kmh: float) -> float:
    if kmh <= 0:
        return 0.0
    return round(max(GLOBAL_MIN_KMH, min(GLOBAL_MAX_KMH, kmh)), 1)


def _segment_speeds_world(raw_pts: list, fps: float, h_matrix: np.ndarray) -> list[float]:
    """Metres per second → km/h for each frame pair using pitch homography."""
    speeds: list[float] = []
    dt = 1.0 / max(fps, 1.0)
    prev = None
    for x, y in raw_pts:
        try:
            wx, wy = video_to_world(float(x), float(y), h_matrix)
        except Exception:
            continue
        if prev is not None:
            dist_m = math.hypot(wx - prev[0], wy - prev[1])
            if dist_m > 0.02:
                speeds.append(dist_m / dt * 3.6)
        prev = (wx, wy)
    return speeds


def _segment_speeds_pixels(raw_pts: list, fps: float, height: int) -> list[float]:
    speeds: list[float] = []
    dt = 1.0 / max(fps, 1.0)
    pitch_span_px = max(height * 0.42, 80.0)
    meters_per_px = 20.12 / pitch_span_px
    for i in range(1, len(raw_pts)):
        d_px = math.hypot(
            raw_pts[i][0] - raw_pts[i - 1][0],
            raw_pts[i][1] - raw_pts[i - 1][1],
        )
        dist_m = d_px * meters_per_px
        if dist_m > 0.02:
            speeds.append(dist_m / dt * 3.6)
    return speeds


def _robust_peak(speeds: list[float]) -> float:
    """85th percentile — ignores single-frame jitter spikes."""
    if not speeds:
        return 0.0
    speeds = sorted(speeds)
    idx = min(len(speeds) - 1, int(len(speeds) * 0.85))
    return speeds[idx]


def _in_flight_segment(raw_pts: list, h_matrix: np.ndarray | None) -> list:
    """
    Prefer pre-bounce flight: points while ball travels toward batsman (increasing Y in video).
    """
    if len(raw_pts) < 4:
        return raw_pts
    ys = [p[1] for p in raw_pts]
    peak_i = int(np.argmax(ys))
    if peak_i >= 3:
        return raw_pts[: peak_i + 1]
    return raw_pts


def compute_bowling_speed_kmh(
    raw_pts: list,
    fps: float,
    *,
    h_matrix: np.ndarray | None = None,
    height: int = 720,
    stump_scale: float = 1.0,
) -> float:
    """
    Calibrated bowling speed (km/h) for one delivery track.
    Calculates speed by taking the total distance traveled during the in-flight
    segment divided by total time elapsed, which is extremely robust against
    frame-to-frame pixel tracking jitter.
    """
    if len(raw_pts) < 2 or fps <= 0:
        return 0.0

    segment = _in_flight_segment(raw_pts, h_matrix)
    if len(segment) < 2:
        segment = raw_pts

    # To reduce single-frame boundary noise, we average the first and last two points if possible
    if len(segment) >= 4:
        p0_x = (float(segment[0][0]) + float(segment[1][0])) / 2.0
        p0_y = (float(segment[0][1]) + float(segment[1][1])) / 2.0
        pn_x = (float(segment[-1][0]) + float(segment[-2][0])) / 2.0
        pn_y = (float(segment[-1][1]) + float(segment[-2][1])) / 2.0
        p0 = (p0_x, p0_y)
        pn = (pn_x, pn_y)
        dt = (len(segment) - 2) / max(fps, 1.0)
    else:
        p0 = segment[0]
        pn = segment[-1]
        dt = (len(segment) - 1) / max(fps, 1.0)

    if dt <= 0:
        return 0.0

    # Calculate real-world distance in meters
    use_pixels = True
    if h_matrix is not None:
        try:
            w0 = video_to_world(float(p0[0]), float(p0[1]), h_matrix)
            wn = video_to_world(float(pn[0]), float(pn[1]), h_matrix)
            dist_m = math.hypot(wn[0] - w0[0], wn[1] - w0[1])
            # If distance is too small or homography fails, fallback to pixel-based calculation
            if dist_m > 1.0:
                use_pixels = False
        except Exception:
            pass

    if use_pixels:
        pitch_span_px = max(height * 0.42, 80.0)
        meters_per_px = 20.12 / pitch_span_px
        d_px = math.hypot(float(pn[0]) - float(p0[0]), float(pn[1]) - float(p0[1]))
        dist_m = d_px * meters_per_px

    # Calculate average speed in km/h
    avg_speed_kmh = (dist_m / dt) * 3.6

    # Release speed is higher than average speed due to air resistance (drag).
    # Typically, a cricket ball loses ~10% speed before bounce, so we scale by 1.10.
    raw_kmh = avg_speed_kmh * 1.10

    # Stump-width reference correction from pitch calibration
    if stump_scale and abs(stump_scale - 1.0) > 0.02:
        raw_kmh = raw_kmh / stump_scale

    # If the computed speed is still unrealistically high (e.g. tracking anomaly),
    # softly compress it rather than hard-capping to 180 or collapsing to 180.
    if raw_kmh > 150.0:
        raw_kmh = 150.0 + (raw_kmh - 150.0) * 0.15

    return clamp_speed(raw_kmh)


def classify_pace_tier(avg_speed_kmh: float) -> dict[str, Any]:
    """Classify bowler pace from session average speed."""
    if avg_speed_kmh <= 0:
        return {"id": "unknown", "label": "Unknown", "avg_min": 0, "avg_max": 0, "max_speed": 0}

    if avg_speed_kmh >= 130:
        tier = PACE_TIERS[2]  # Fast Bowling 130–160 avg, max 170–180
    elif avg_speed_kmh >= 115:
        tier = PACE_TIERS[1]  # Fast Medium 120–150 avg, max 160
    else:
        tier = PACE_TIERS[0]  # Medium 80–120 avg, max 140
    return tier


def session_speed_stats(bounces: list[dict]) -> dict[str, Any]:
    """Average, maximum, and pace tier for a session."""
    speeds = [
        float(b["speed_kmh"])
        for b in bounces
        if b.get("speed_kmh") and float(b["speed_kmh"]) > 0
    ]
    if not speeds:
        return {
            "avg_speed_kmh": 0.0,
            "max_speed_kmh": 0.0,
            "min_speed_kmh": 0.0,
            "pace_tier": "unknown",
            "pace_label": "Unknown",
            "pace_avg_range": "",
            "pace_max_cap": 0,
        }

    avg = round(sum(speeds) / len(speeds), 1)
    mx = round(max(speeds), 1)
    tier = classify_pace_tier(avg)
    return {
        "avg_speed_kmh": avg,
        "max_speed_kmh": mx,
        "min_speed_kmh": round(min(speeds), 1),
        "pace_tier": tier["id"],
        "pace_label": tier["label"],
        "pace_avg_range": f"{tier['avg_min']}–{tier['avg_max']} km/h",
        "pace_max_cap": tier["max_speed"],
    }


def format_speed_display(kmh: float) -> str:
    if kmh <= 0:
        return "—"
    return f"{kmh:.0f} km/h"
