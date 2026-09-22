"""
Periodic homography recalibration when camera angle / shake changes mid-video.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.config import CONFIG
from core.pitch_calibrator import CalibrationResult, calibrate_pitch_robust

_AUTO = CONFIG.get("processing", {}).get("fully_automatic", {})
_RECALIB_SEC = float(_AUTO.get("recalibrate_interval_sec", 18.0))
_SHAKE_PX = float(_AUTO.get("recalibrate_shake_px", 3.5))
_CORNER_DRIFT = float(_AUTO.get("recalibrate_corner_drift_px", 28.0))


class HomographyRecalibrator:
    """Re-fit pitch quad when stabilization reports significant camera motion."""

    def __init__(self, width: int, height: int, fps: float, *, enabled: bool = True):
        self.width = width
        self.height = height
        self.fps = fps
        self.enabled = enabled and bool(_AUTO.get("auto_recalibrate", True))
        self.interval_frames = max(30, int(fps * _RECALIB_SEC))
        self._last_frame = 0
        self._last_quad: np.ndarray | None = None
        self._recalib_count = 0

    def set_baseline(self, quad: np.ndarray) -> None:
        self._last_quad = np.asarray(quad, dtype=np.float32).reshape(4, 2).copy()

    def maybe_recalibrate(
        self,
        frame_index: int,
        frame: np.ndarray,
        *,
        shake_px: float = 0.0,
    ) -> CalibrationResult | None:
        if not self.enabled or self._last_quad is None:
            return None
        if frame_index - self._last_frame < self.interval_frames:
            return None
        if shake_px < _SHAKE_PX:
            return None

        self._last_frame = frame_index
        cap = _SingleFrameCap(frame)
        result = calibrate_pitch_robust(
            cap, self.width, self.height, max_samples=1,
        )
        if result.confidence < 0.35:
            return None

        new_q = np.asarray(result.quad, dtype=np.float32).reshape(4, 2)
        drift = float(np.mean(np.linalg.norm(new_q - self._last_quad, axis=1)))
        if drift < _CORNER_DRIFT:
            return None

        self._last_quad = new_q.copy()
        self._recalib_count += 1
        print(
            f"[Recalib] frame={frame_index} source={result.source} "
            f"conf={result.confidence:.2f} drift={drift:.1f}px (#{self._recalib_count})",
            flush=True,
        )
        return result

    @property
    def recalibration_count(self) -> int:
        return self._recalib_count


class _SingleFrameCap:
    """Minimal cv2.VideoCapture shim for one-frame calibration."""

    def __init__(self, frame: np.ndarray):
        self._frame = frame

    def read(self):
        return True, self._frame

    def set(self, prop, value):
        return True

    def get(self, prop):
        return 0

    def release(self):
        pass
