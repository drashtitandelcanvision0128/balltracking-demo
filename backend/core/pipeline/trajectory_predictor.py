"""Kalman-based future trajectory extrapolation."""

from __future__ import annotations

import math

from core.ball_kalman import BallKalmanFilter


def predict_future_path(
    kf: BallKalmanFilter,
    *,
    n_frames: int = 10,
    start_frame: int = 0,
) -> list[tuple[int, float, float]]:
    """
    Extrapolate ball position forward using constant-velocity Kalman state.

    Why: shows where the ball is heading before bounce is confirmed — useful for
    operators and for pre-bounce landing estimates.
    """
    if not kf.initialized or n_frames <= 0:
        return []

    x, y = kf.get_position()
    vx, vy = kf.get_velocity()
    dt = float(kf.dt)
    out: list[tuple[int, float, float]] = []
    for i in range(1, n_frames + 1):
        t = i * dt
        px = x + vx * t
        py = y + vy * t
        out.append((start_frame + i, float(px), float(py)))
    return out


def predict_to_ground_line(
    kf: BallKalmanFilter,
    ground_y: float,
    *,
    max_frames: int = 60,
    start_frame: int = 0,
) -> tuple[float, float] | None:
    """Intersect Kalman ray with pitch ground line (image Y)."""
    if not kf.initialized:
        return None
    x, y = kf.get_position()
    vx, vy = kf.get_velocity()
    if abs(vy) < 0.5:
        return None
    for i in range(1, max_frames + 1):
        t = i * float(kf.dt)
        py = y + vy * t
        if py >= ground_y:
            px = x + vx * t
            return float(px), float(ground_y)
        if vy > 0 and py > ground_y + 200:
            break
    return None
