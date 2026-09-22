"""
Ball tracking with Kalman smoothing and unique track IDs.

Handles missed detections by coasting the Kalman prediction for a limited
number of frames before dropping the track.
"""

from __future__ import annotations

import math

from core.ball_kalman import BallKalmanFilter, create_ball_kalman
from core.pipeline.types import TrackPoint


class BallTracker:
    """Single-ball tracker — one active delivery at a time."""

    def __init__(self, fps: float, *, coast_limit: int | None = None, reacquire_frames: int | None = None) -> None:
        self.fps = max(fps, 1.0)
        self.coast_limit = coast_limit or max(12, int(self.fps * 0.4))
        self.reacquire_frames = reacquire_frames or max(18, int(self.fps * 0.6))
        self._track_id = 0
        self._kf: BallKalmanFilter | None = None
        self._frames_since_det = 999
        self._history: list[TrackPoint] = []
        self._lost_frames = 0
        self._last_x = 0.0
        self._last_y = 0.0
        self._last_vx = 0.0
        self._last_vy = 0.0

    @property
    def active(self) -> bool:
        return self._kf is not None and self._kf.initialized

    @property
    def track_id(self) -> int:
        return self._track_id

    @property
    def history(self) -> list[TrackPoint]:
        return list(self._history)

    @property
    def kalman(self) -> BallKalmanFilter | None:
        return self._kf

    def current_position(self) -> tuple[float, float] | None:
        if not self.active:
            return None
        return self._kf.get_position()

    def reset(self) -> None:
        self._track_id += 1
        self._kf = None
        self._frames_since_det = 999
        self._lost_frames = 0
        self._history.clear()

    def _save_motion_state(self) -> None:
        if self._kf and self._kf.initialized:
            self._last_x, self._last_y = self._kf.get_position()
            self._last_vx, self._last_vy = self._kf.get_velocity()

    def _try_reacquire(self, frame_id: int, x: float, y: float) -> bool:
        """Re-attach after brief occlusion — same track ID."""
        if self._lost_frames > self.reacquire_frames:
            return False
        pred_x = self._last_x + self._last_vx * self._lost_frames
        pred_y = self._last_y + self._last_vy * self._lost_frames
        gate = max(80.0, self.coast_limit * 4.0)
        if math.hypot(x - pred_x, y - pred_y) > gate:
            return False
        self._kf = create_ball_kalman(self.fps)
        self._kf.init(x, y, self._last_vx, self._last_vy)
        self._frames_since_det = 0
        self._lost_frames = 0
        pt = TrackPoint(frame_id, x, y, self._track_id, predicted=False)
        self._history.append(pt)
        return True

    def start(self, frame_id: int, x: float, y: float) -> TrackPoint:
        self._kf = create_ball_kalman(self.fps)
        self._kf.init(x, y)
        self._frames_since_det = 0
        pt = TrackPoint(frame_id, x, y, self._track_id, predicted=False)
        self._history.append(pt)
        return pt

    def update(self, frame_id: int, detection: tuple[float, float] | None) -> TrackPoint | None:
        """Fuse YOLO detection or coast Kalman when detection is missing."""
        if not self.active:
            if detection is None:
                self._lost_frames += 1
                return None
            dx, dy = detection
            if self._lost_frames > 0 and self._try_reacquire(frame_id, dx, dy):
                return self._history[-1]
            if self._lost_frames > 0:
                self.reset()
            return self.start(frame_id, dx, dy)

        self._frames_since_det += 1
        self._kf.predict()

        if detection is not None:
            dx, dy = detection
            if self._kf.correct(dx, dy):
                self._frames_since_det = 0
                self._lost_frames = 0
                x, y = self._kf.get_position()
                pt = TrackPoint(frame_id, x, y, self._track_id, predicted=False)
                self._history.append(pt)
                return pt

        if self._frames_since_det <= self.coast_limit:
            x, y = self._kf.get_position()
            pt = TrackPoint(frame_id, x, y, self._track_id, predicted=True)
            self._history.append(pt)
            return pt

        self._save_motion_state()
        self._kf = None
        self._lost_frames = 1
        return None

    def frame_points(self) -> list[tuple[int, float, float]]:
        return [(p.frame_id, p.x, p.y) for p in self._history]
