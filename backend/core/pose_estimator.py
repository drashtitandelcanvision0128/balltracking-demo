"""
Batsman pose estimation using MediaPipe Pose.
Samples key frames during batting zone for coaching analytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# MediaPipe is optional — graceful fallback if not installed
try:
    import mediapipe as mp

    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

POSE_LANDMARKS = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


@dataclass
class PoseFrame:
    frame_index: int
    landmarks: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    head_position: tuple[float, float] | None = None
    bat_zone: tuple[float, float] | None = None
    front_foot: tuple[float, float] | None = None
    back_foot: tuple[float, float] | None = None
    balance_score: float = 0.0


class BatsmanPoseEstimator:
    """Estimate batsman pose on sampled frames during delivery."""

    def __init__(self, min_detection_confidence: float = 0.5):
        self.enabled = _MP_AVAILABLE
        self._pose = None
        if _MP_AVAILABLE:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5,
            )

    def process_frame(
        self, frame: np.ndarray, frame_index: int, batting_zone_only: bool = True
    ) -> PoseFrame | None:
        if not self.enabled or self._pose is None:
            return None

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks.landmark
        landmarks = {}
        for name, idx in POSE_LANDMARKS.items():
            pt = lm[idx]
            landmarks[name] = (pt.x * w, pt.y * h, pt.visibility)

        nose = landmarks.get("nose")
        if batting_zone_only and nose and nose[1] < h * 0.25:
            return None

        head_pos = (nose[0], nose[1]) if nose else None
        lw = landmarks.get("left_wrist")
        rw = landmarks.get("right_wrist")
        bat_zone = None
        if lw and rw:
            bat_zone = ((lw[0] + rw[0]) / 2, (lw[1] + rw[1]) / 2)

        la = landmarks.get("left_ankle")
        ra = landmarks.get("right_ankle")
        front_foot, back_foot = None, None
        if la and ra:
            if la[1] > ra[1]:
                front_foot = (la[0], la[1])
                back_foot = (ra[0], ra[1])
            else:
                front_foot = (ra[0], ra[1])
                back_foot = (la[0], la[1])

        balance = _compute_balance(landmarks)

        return PoseFrame(
            frame_index=frame_index,
            landmarks={k: (v[0], v[1]) for k, v in landmarks.items()},
            head_position=head_pos,
            bat_zone=bat_zone,
            front_foot=front_foot,
            back_foot=back_foot,
            balance_score=balance,
        )

    def sample_delivery_poses(
        self,
        frames: list[tuple[int, np.ndarray]],
        max_samples: int = 5,
    ) -> list[dict[str, Any]]:
        """Process a list of (frame_index, frame) tuples from a delivery."""
        if not self.enabled:
            return []
        step = max(1, len(frames) // max_samples)
        results = []
        for i in range(0, len(frames), step):
            fidx, frame = frames[i]
            pose = self.process_frame(frame, fidx)
            if pose:
                results.append(pose_to_dict(pose))
        return results

    def close(self):
        if self._pose is not None:
            self._pose.close()


def _compute_balance(landmarks: dict[str, tuple[float, float, float]]) -> float:
    """Simple balance score from shoulder/hip alignment (0–100)."""
    ls = landmarks.get("left_shoulder")
    rs = landmarks.get("right_shoulder")
    lh = landmarks.get("left_hip")
    rh = landmarks.get("right_hip")
    if not all([ls, rs, lh, rh]):
        return 50.0
    shoulder_tilt = abs(ls[1] - rs[1])
    hip_tilt = abs(lh[1] - rh[1])
    tilt = (shoulder_tilt + hip_tilt) / 2
    return round(max(0.0, min(100.0, 100.0 - tilt * 2)), 1)


def pose_to_dict(pose: PoseFrame) -> dict[str, Any]:
    return {
        "frame_index": pose.frame_index,
        "head_position": pose.head_position,
        "bat_zone": pose.bat_zone,
        "front_foot": pose.front_foot,
        "back_foot": pose.back_foot,
        "balance_score": pose.balance_score,
        "landmarks": pose.landmarks,
    }
