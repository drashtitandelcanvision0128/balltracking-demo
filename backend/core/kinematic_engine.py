"""
Module 2 — deterministic bounce & hit from YOLO+Kalman trajectory (PDF architecture).
No LLM, no overfitting: pure math on (x, y) track points.
"""

from __future__ import annotations

import math

import numpy as np

from core.ball_detection_filters import effective_bounce_ground_y_min, is_landscape_frame

# PDF: sharp angle change → hit/deflection
HIT_DOT_PRODUCT_THRESHOLD = 0.85


def detect_bounce_point(
    trajectory: list[tuple[float, float]],
    *,
    height: int = 1080,
    width: int = 0,
    min_drop_px: float | None = None,
) -> tuple[float, float] | None:
    """
    Bounce = local Y maximum on screen (ball falls then rises).
    trajectory: [(x, y), ...] in pixel coords, time-ordered.
    """
    if len(trajectory) < 4:
        return None

    ys = [p[1] for p in trajectory]
    need_drop = min_drop_px if min_drop_px is not None else max(8.0, height * 0.008)
    landscape = width > 0 and is_landscape_frame(width, height)
    ground_y = int(height * effective_bounce_ground_y_min(width, height)) if width > 0 else int(height * 0.58)
    if landscape:
        ground_y = int(height * max(0.35, effective_bounce_ground_y_min(width, height) - 0.08))

    for i in range(1, len(ys) - 1):
        y_prev, y_curr, y_next = ys[i - 1], ys[i], ys[i + 1]
        if not (y_curr >= y_prev and y_curr >= y_next):
            continue
        if y_curr - ys[0] < need_drop:
            continue
        if y_next >= y_curr - max(6.0, height * 0.006):
            continue
        x, y = trajectory[i]
        if width > 0 and is_landscape_frame(width, height) and y < ground_y:
            continue
        return float(x), float(y)
    return None


def detect_hit_deflection(
    trajectory: list[tuple[float, float]],
    *,
    dot_threshold: float = HIT_DOT_PRODUCT_THRESHOLD,
) -> tuple[float, float] | None:
    """
    Hit / deflection = sharp trajectory angle change (PDF dot product < 0.85).
    """
    if len(trajectory) < 5:
        return None

    for i in range(2, len(trajectory) - 2):
        v1 = np.array([
            trajectory[i][0] - trajectory[i - 2][0],
            trajectory[i][1] - trajectory[i - 2][1],
        ], dtype=np.float64)
        v2 = np.array([
            trajectory[i + 2][0] - trajectory[i][0],
            trajectory[i + 2][1] - trajectory[i][1],
        ], dtype=np.float64)
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 < 3.0 or n2 < 3.0:
            continue
        dot = float(np.dot(v1 / n1, v2 / n2))
        if dot < dot_threshold:
            return float(trajectory[i][0]), float(trajectory[i][1])
    return None


def detect_bounce_and_hit(
    trajectory: list[tuple[float, float]],
    *,
    height: int = 1080,
    width: int = 0,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """PDF entry point — returns (bounce_xy, hit_xy)."""
    bounce = detect_bounce_point(trajectory, height=height, width=width)
    hit = detect_hit_deflection(trajectory)
    return bounce, hit


def bounce_from_frame_track(
    track_pts: list,
    release_frame: int,
    *,
    height: int = 1080,
    width: int = 0,
) -> tuple[int, float, float] | None:
    """
    Map kinematic bounce onto frame-indexed YOLO track.
    Returns (frame, x, y) or None.
    """
    seg = [(int(p[0]), float(p[1]), float(p[2])) for p in track_pts if int(p[0]) >= release_frame]
    if len(seg) < 4:
        return None
    skip = max(6, int(len(seg) * 0.22))
    if len(seg) - skip < 4:
        return None
    seg = seg[skip:]
    traj = [(x, y) for _, x, y in seg]
    pt = detect_bounce_point(traj, height=height, width=width)
    if pt is None:
        return None
    bx, by = pt
    best_i = min(range(len(seg)), key=lambda i: math.hypot(seg[i][1] - bx, seg[i][2] - by))
    f, x, y = seg[best_i]
    return int(f), float(x), float(y)
