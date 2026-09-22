"""Visualization — trajectory and predicted bounce marker."""

from __future__ import annotations

import cv2
import numpy as np

from core.bounce_predictor import BallPoint, smooth_positions, track_history
from core.pipeline.types import BounceResult, TrackPoint


def draw_trajectory(
    frame,
    points: list[TrackPoint] | list[tuple[int, float, float]],
    *,
    bounce_index: int | None = None,
    before_color: tuple[int, int, int] = (0, 220, 255),
    after_color: tuple[int, int, int] = (255, 180, 0),
    thickness: int = 2,
) -> None:
    """Draw smoothed path, optionally split at bounce."""
    if not points:
        return
    if isinstance(points[0], TrackPoint):
        raw = [(p.frame_id, p.x, p.y) for p in points]
    else:
        raw = list(points)
    hist = track_history(raw, maxlen=999)
    smooth = smooth_positions(hist, window=3)
    if len(smooth) < 2:
        return

    def _line(segment: list[BallPoint], color):
        arr = np.array([(int(p.x), int(p.y)) for p in segment], dtype=np.int32)
        if len(arr) >= 2:
            cv2.polylines(frame, [arr], False, color, thickness, cv2.LINE_AA)

    if bounce_index is not None and 0 <= bounce_index < len(smooth):
        _line(smooth[: bounce_index + 1], before_color)
        _line(smooth[bounce_index:], after_color)
    else:
        _line(smooth, before_color)


def draw_bounce_marker(
    frame,
    bounce: BounceResult,
    *,
    label: str = "Predicted Bounce",
    dot_color: tuple[int, int, int] = (0, 0, 255),
) -> None:
    """Red circle + label at the predicted pitch point."""
    bx, by = int(bounce.image_x), int(bounce.image_y)
    h, w = frame.shape[:2]
    if not (0 <= bx < w and 0 <= by < h):
        return
    cv2.circle(frame, (bx, by), 14, dot_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (bx, by), 14, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        frame, label, (bx + 16, by - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, dot_color, 2, cv2.LINE_AA,
    )
    meta = f"{bounce.confidence:.0%} | {bounce.pitch_y_m:.1f}m"
    cv2.putText(
        frame, meta, (bx + 16, by + 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )


def draw_delivery(
    frame,
    trajectory: list[TrackPoint],
    bounce: BounceResult | None,
) -> None:
    """Full delivery overlay: trajectory + bounce marker."""
    idx = bounce.history_index if bounce else None
    draw_trajectory(frame, trajectory, bounce_index=idx)
    if bounce is not None:
        draw_bounce_marker(frame, bounce)
