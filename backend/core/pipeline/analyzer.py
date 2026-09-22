"""
Delivery analyzer — orchestrates detection → tracking → bounce → output.

Usage
-----
    analyzer = DeliveryAnalyzer(fps=30.0, h_matrix=H, cam_quad=quad, height=1080)
    for frame_id, det in yolo_detections:
        analysis = analyzer.step(frame_id, det)  # det = (x,y) or None
    result = analyzer.finalize()
"""

from __future__ import annotations

import numpy as np

from core.pipeline.ball_tracker import BallTracker
from core.pipeline.bounce_engine import predict_bounce
from core.pipeline.trajectory import smooth_trajectory
from core.pipeline.types import DeliveryAnalysis, BounceResult
from core.pipeline.visualizer import draw_delivery


class DeliveryAnalyzer:
    """Stateful per-delivery analysis with Kalman tracking and bounce fusion."""

    def __init__(
        self,
        fps: float,
        *,
        h_matrix: np.ndarray | None = None,
        cam_quad: np.ndarray | None = None,
        height: int = 1080,
        min_bounce_confidence: float = 0.55,
        coast_limit: int | None = None,
    ) -> None:
        self.fps = fps
        self.h_matrix = h_matrix
        self.cam_quad = cam_quad
        self.height = height
        self.min_bounce_confidence = min_bounce_confidence
        self.tracker = BallTracker(fps, coast_limit=coast_limit)
        self._bounce: BounceResult | None = None
        self._locked_bounce = False

    def step(
        self,
        frame_id: int,
        detection: tuple[float, float] | None,
    ) -> DeliveryAnalysis:
        self.tracker.update(frame_id, detection)
        bounce = self._update_bounce()
        path = smooth_trajectory(self.tracker.history, self.fps)
        return DeliveryAnalysis(
            track_id=self.tracker.track_id,
            trajectory=self.tracker.history,
            bounce=bounce,
            smoothed_path=path,
        )

    def _update_bounce(self) -> BounceResult | None:
        if self._locked_bounce and self._bounce is not None:
            return self._bounce
        pts = self.tracker.frame_points()
        if len(pts) < 10:
            return self._bounce
        result = predict_bounce(
            pts,
            h_matrix=self.h_matrix,
            cam_quad=self.cam_quad,
            height=self.height,
            fps=self.fps,
            min_confidence=self.min_bounce_confidence,
        )
        if result is not None:
            self._bounce = result
        return self._bounce

    def lock_bounce(self) -> BounceResult | None:
        """Freeze bounce once enough frames have passed after contact."""
        if self._bounce is not None:
            self._locked_bounce = True
        return self._bounce

    def reset(self) -> None:
        self.tracker.reset()
        self._bounce = None
        self._locked_bounce = False

    def finalize(self) -> DeliveryAnalysis:
        bounce = self._bounce or self._update_bounce()
        return DeliveryAnalysis(
            track_id=self.tracker.track_id,
            trajectory=self.tracker.history,
            bounce=bounce,
            smoothed_path=smooth_trajectory(self.tracker.history, self.fps),
        )

    def draw(self, frame, analysis: DeliveryAnalysis | None = None) -> None:
        data = analysis or self.finalize()
        draw_delivery(frame, data.trajectory, data.bounce)

    def to_dict(self, analysis: DeliveryAnalysis | None = None) -> dict:
        data = analysis or self.finalize()
        out = {
            "track_id": data.track_id,
            "trajectory": [
                {"frame": p.frame_id, "x": p.x, "y": p.y, "predicted": p.predicted}
                for p in data.trajectory
            ],
            "smoothed_path": data.smoothed_path,
            "bounce": None,
        }
        if data.bounce:
            b = data.bounce
            out["bounce"] = {
                "bounce_frame": b.bounce_frame,
                "image_x": b.image_x,
                "image_y": b.image_y,
                "pitch_x_m": b.pitch_x_m,
                "pitch_y_m": b.pitch_y_m,
                "pitch_map_x": b.pitch_map_x,
                "pitch_map_y": b.pitch_map_y,
                "confidence": b.confidence,
                "method": b.method,
            }
        return out
