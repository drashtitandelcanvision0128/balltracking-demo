"""
Physics-informed trajectory helpers: parabolic fit, world-coords bounce, gap fill.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from core.pitch_coords import video_to_world


def fit_parabolic_y(xs: list[float], ys: list[float]) -> tuple[float, float, float] | None:
    """Fit y = a*x^2 + b*x + c. Returns (a, b, c) or None."""
    if len(xs) < 4:
        return None
    try:
        coeffs = np.polyfit(xs, ys, 2)
        return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    except (np.linalg.LinAlgError, ValueError):
        return None


def interpolate_track_gaps(
    raw_pts: list[tuple[int, int]],
    fps: float,
    max_gap_frames: int = 4,
) -> list[tuple[int, int]]:
    """
    Fill short detection gaps via parabolic interpolation on recent segment.
    Only fills gaps <= max_gap_frames between known points.
    """
    if len(raw_pts) < 3 or fps <= 0:
        return raw_pts

    out = [raw_pts[0]]
    for i in range(1, len(raw_pts)):
        prev = out[-1]
        curr = raw_pts[i]
        gap = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
        step_est = max(8.0, gap)
        est_frames = max(1, int(gap / step_est))
        if est_frames > 1 and est_frames <= max_gap_frames:
            seg_xs = [p[0] for p in out[-6:]] + [curr[0]]
            seg_ys = [p[1] for p in out[-6:]] + [curr[1]]
            t = np.linspace(0, 1, est_frames + 1)[1:-1]
            for ti in t:
                ix = int(prev[0] + (curr[0] - prev[0]) * ti)
                iy = int(prev[1] + (curr[1] - prev[1]) * ti)
                if len(seg_xs) >= 4:
                    fit = fit_parabolic_y(seg_xs, seg_ys)
                    if fit is not None:
                        a, b, c = fit
                        ix_f = prev[0] + (curr[0] - prev[0]) * ti
                        iy = int(a * ix_f * ix_f + b * ix_f + c)
                out.append((ix, iy))
        out.append(curr)
    return out


def world_trajectory(
    raw_pts: Sequence[tuple[int, int]],
    h_matrix: np.ndarray,
) -> list[tuple[float, float]]:
    pts = []
    for x, y in raw_pts:
        try:
            wx, wy = video_to_world(float(x), float(y), h_matrix)
            pts.append((wx, wy))
        except Exception:
            continue
    return pts


def refine_bounce_world(
    raw_pts: list[tuple[int, int]],
    h_matrix: np.ndarray,
    height: int,
    lookback: int = 18,
) -> tuple[int, int] | None:
    """
    Bounce = local minimum in world Y (metres from stumps) after downward flight.
    More stable than pixel-Y apex across camera angles.
    """
    if len(raw_pts) < 5:
        return None

    start = max(0, len(raw_pts) - lookback)
    segment = raw_pts[start:]
    world = world_trajectory(segment, h_matrix)
    if len(world) < 5:
        return None

    best_i, best_score = None, -1e9
    for i in range(2, len(world) - 2):
        wy = [world[j][1] for j in range(i - 2, i + 3)]
        # Bounce: world Y has local minimum (ball closest to batsman / lowest pitch point)
        if not (wy[1] > wy[2] and wy[3] > wy[2] and wy[0] > wy[2]):
            continue
        depth = (wy[1] - wy[2]) + (wy[3] - wy[2])
        flatness = abs(wy[1] - wy[3])
        score = depth - flatness * 0.4
        if score > best_score:
            best_score = score
            best_i = i

    if best_i is None:
        return None
    px, py = segment[best_i]
    if py < height * 0.18:
        return None
    return int(px), int(py)


def predict_pre_bounce_landing(
    raw_pts: list[tuple[int, int]],
    h_matrix: np.ndarray,
    fps: float,
) -> tuple[float, float] | None:
    """Extrapolate parabolic world trajectory to pitch plane (y_m ≈ 0)."""
    world = world_trajectory(raw_pts, h_matrix)
    if len(world) < 5 or fps <= 0:
        return None
    xs = [p[0] for p in world]
    ys = [p[1] for p in world]
    fit = fit_parabolic_y(xs, ys)
    if fit is None:
        return None
    a, b, c = fit
    if abs(a) < 1e-6:
        return None
    # Solve a*x^2 + b*x + c = 0 for positive y side (approach to stumps)
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    roots = [(-b + math.sqrt(disc)) / (2 * a), (-b - math.sqrt(disc)) / (2 * a)]
    valid = [r for r in roots if 0 <= r <= 22]
    if not valid:
        return None
    x_land = min(valid, key=abs)
    return round(x_land, 3), 0.0
