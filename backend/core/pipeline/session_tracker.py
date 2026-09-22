"""Multi-delivery session tracking for full-video analysis."""

from __future__ import annotations

import numpy as np

from core.pipeline.analyzer import DeliveryAnalyzer
from core.pipeline.types import DeliveryAnalysis, BounceResult


class SessionTracker:
    """
    Track all deliveries in one video.

    - One active delivery at a time (Kalman + bounce fusion)
    - Auto-splits on long detection gaps
    - Counts completed deliveries with bounce metadata
    """

    def __init__(
        self,
        fps: float,
        *,
        h_matrix: np.ndarray | None = None,
        cam_quad: np.ndarray | None = None,
        height: int = 1080,
        width: int = 1920,
        min_bounce_confidence: float = 0.55,
        gap_sec: float = 0.85,
        min_delivery_frames: int = 8,
        bounce_confirm_frames: int = 5,
    ) -> None:
        self.fps = fps
        self.height = height
        self.width = width
        self.gap_frames = max(12, int(fps * gap_sec))
        self.min_delivery_frames = min_delivery_frames
        self.bounce_confirm_frames = bounce_confirm_frames
        self._analyzer = DeliveryAnalyzer(
            fps,
            h_matrix=h_matrix,
            cam_quad=cam_quad,
            height=height,
            min_bounce_confidence=min_bounce_confidence,
        )
        self._completed: list[DeliveryAnalysis] = []
        self._last_active_frame = -9999
        self._last_detection_frame = -9999
        self._locked_bounces: list[BounceResult] = []

    @property
    def delivery_count(self) -> int:
        return len(self._completed)

    @property
    def completed_deliveries(self) -> list[DeliveryAnalysis]:
        return list(self._completed)

    @property
    def active(self) -> DeliveryAnalysis | None:
        if not self._analyzer.tracker.active:
            return None
        return self._analyzer.finalize()

    @property
    def locked_bounces(self) -> list[BounceResult]:
        return list(self._locked_bounces)

    def step(
        self,
        frame_id: int,
        detection: tuple[float, float] | None,
    ) -> DeliveryAnalysis | None:
        """Feed one frame; auto-close delivery on long gap."""
        if self._analyzer.tracker.active:
            gap = frame_id - self._last_detection_frame
            if detection is None and gap > self.gap_frames:
                self._close_delivery(frame_id)
            elif frame_id - self._last_active_frame > int(self.fps * 2.5):
                self._close_delivery(frame_id)

        if detection is not None:
            self._last_detection_frame = frame_id

        analysis = self._analyzer.step(frame_id, detection)
        if self._analyzer.tracker.active:
            self._last_active_frame = frame_id
            self._try_lock_bounce(frame_id, analysis)
            return analysis
        return None

    def _try_lock_bounce(self, frame_id: int, analysis: DeliveryAnalysis) -> None:
        bounce = analysis.bounce
        if bounce is None:
            return
        if frame_id - bounce.bounce_frame < self.bounce_confirm_frames:
            return
        if any(b.bounce_frame == bounce.bounce_frame for b in self._locked_bounces):
            return
        self._locked_bounces.append(bounce)
        self._analyzer.lock_bounce()

    def _close_delivery(self, frame_id: int) -> None:
        if not self._analyzer.tracker.active:
            return
        result = self._analyzer.finalize()
        if len(result.trajectory) >= self.min_delivery_frames:
            self._completed.append(result)
        self._analyzer.reset()

    def finalize(self) -> list[DeliveryAnalysis]:
        """Close any open delivery at end of video."""
        if self._analyzer.tracker.active:
            self._close_delivery(10**9)
        return list(self._completed)

    def to_dict(self) -> dict:
        deliveries = []
        for i, d in enumerate(self._completed, 1):
            entry = self._analyzer.to_dict(d)
            entry["delivery_number"] = i
            deliveries.append(entry)
        return {
            "delivery_count": len(self._completed),
            "deliveries": deliveries,
            "bounces": [
                {
                    "bounce_frame": b.bounce_frame,
                    "image_x": b.image_x,
                    "image_y": b.image_y,
                    "pitch_x_m": b.pitch_x_m,
                    "pitch_y_m": b.pitch_y_m,
                    "confidence": b.confidence,
                    "method": b.method,
                }
                for b in self._locked_bounces
            ],
        }
