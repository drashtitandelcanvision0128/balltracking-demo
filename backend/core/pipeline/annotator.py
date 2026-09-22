"""Full-frame video annotation — trajectory, bounce, future path, HUD."""

from __future__ import annotations

import cv2

from core.pipeline.types import BounceResult, DeliveryAnalysis, TrackPoint
from core.pipeline.visualizer import draw_bounce_marker, draw_trajectory


def draw_current_ball(
    frame,
    x: float,
    y: float,
    *,
    track_id: int = 0,
    predicted: bool = False,
) -> None:
    cx, cy = int(x), int(y)
    h, w = frame.shape[:2]
    if not (0 <= cx < w and 0 <= cy < h):
        return
    color = (0, 200, 255) if not predicted else (0, 140, 255)
    cv2.circle(frame, (cx, cy), 10, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2, cv2.LINE_AA)
    if track_id > 0:
        cv2.putText(
            frame, f"ID:{track_id}", (cx + 12, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
        )


def draw_future_path(
    frame,
    future_pts: list[tuple[int, float, float]],
    *,
    color: tuple[int, int, int] = (180, 255, 80),
) -> None:
    """Dashed-style future Kalman extrapolation."""
    if len(future_pts) < 2:
        return
    arr = [(int(p[1]), int(p[2])) for p in future_pts]
    for i in range(1, len(arr)):
        cv2.line(frame, arr[i - 1], arr[i], color, 1, cv2.LINE_AA)
    cv2.circle(frame, arr[-1], 5, color, -1, cv2.LINE_AA)


def draw_session_hud(frame, delivery_count: int, active_track_id: int = 0) -> None:
    cv2.rectangle(frame, (8, 8), (220, 52), (0, 0, 0), -1)
    cv2.putText(
        frame, f"Deliveries: {delivery_count}", (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
    )
    if active_track_id > 0:
        cv2.putText(
            frame, f"Track ID: {active_track_id}", (16, 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1, cv2.LINE_AA,
        )


def annotate_frame(
    frame,
    *,
    active: DeliveryAnalysis | None = None,
    future_path: list[tuple[int, float, float]] | None = None,
    locked_bounces: list[BounceResult] | None = None,
    delivery_count: int = 0,
    show_trajectory: bool = True,
    show_future: bool = True,
    show_hud: bool = True,
) -> None:
    """Draw complete delivery overlay on one video frame."""
    track_id = 0
    if active and active.trajectory:
        track_id = active.trajectory[-1].track_id
        last = active.trajectory[-1]
        if show_trajectory:
            draw_trajectory(
                frame, active.trajectory,
                bounce_index=active.bounce.history_index if active.bounce else None,
            )
        draw_current_ball(
            frame, last.x, last.y,
            track_id=track_id, predicted=last.predicted,
        )
        if show_future and future_path:
            draw_future_path(frame, future_path)
        if active.bounce is not None:
            draw_bounce_marker(frame, active.bounce)

    for bounce in locked_bounces or []:
        draw_bounce_marker(frame, bounce)

    if show_hud:
        draw_session_hud(frame, delivery_count, track_id)
