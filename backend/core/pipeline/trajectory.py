"""Trajectory smoothing and gap filling."""

from __future__ import annotations

from core.trajectory_physics import fit_parabolic_y, interpolate_track_gaps
from core.pipeline.types import TrackPoint


def smooth_trajectory(
    points: list[TrackPoint],
    fps: float,
    *,
    max_gap_frames: int = 4,
) -> list[tuple[int, int]]:
    """
    Return gap-filled pixel path suitable for drawing and bounce analysis.

    Parabolic interpolation fills short missed-detection gaps without inventing
    long invisible flight segments.
    """
    raw = [(int(p.x), int(p.y)) for p in points]
    if len(raw) < 2:
        return raw
    return interpolate_track_gaps(raw, fps, max_gap_frames=max_gap_frames)


def moving_average_path(
    points: list[tuple[int, float, float]],
    *,
    window: int = 3,
) -> list[tuple[int, float, float]]:
    """Light moving-average smoothing on tracked centres."""
    if len(points) < 2:
        return list(points)
    import numpy as np

    w = max(1, window)
    xs = np.array([p[1] for p in points], dtype=np.float64)
    ys = np.array([p[2] for p in points], dtype=np.float64)
    k = np.ones(w) / w
    sx = np.convolve(xs, k, mode="same")
    sy = np.convolve(ys, k, mode="same")
    return [(points[i][0], float(sx[i]), float(sy[i])) for i in range(len(points))]
