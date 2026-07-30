"""
Constant-velocity Kalman filter for cricket ball tracking (image pixels).
Tuned for 60fps fast deliveries — config via config.yaml → tracking.kalman.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from core.config import CONFIG

_TRACK = CONFIG.get("tracking", {})
_KF = _TRACK.get("kalman", {})


def kalman_config() -> dict:
    return {
        "process_noise_pos": float(_KF.get("process_noise_pos", 0.12)),
        "process_noise_vel": float(_KF.get("process_noise_vel", 0.90)),
        "measurement_noise": float(_KF.get("measurement_noise", 0.035)),
        "max_gate_px": float(_KF.get("max_gate_px", 220)),
        "init_velocity_decay": float(_KF.get("init_velocity_decay", 0.85)),
    }


class BallKalmanFilter:
    """4-state CV model: [x, y, vx, vy]. dt = 1/fps per frame."""

    def __init__(self, dt: float, cfg: dict | None = None):
        self.dt = float(dt)
        cfg = cfg or kalman_config()
        self.max_gate_px = float(cfg["max_gate_px"])

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

        q = np.eye(4, dtype=np.float32)
        q[0, 0] = q[1, 1] = float(cfg["process_noise_pos"])
        q[2, 2] = q[3, 3] = float(cfg["process_noise_vel"])
        self.kf.processNoiseCov = q
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(cfg["measurement_noise"])
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def init(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        self.kf.statePost = np.array([[x], [y], [vx], [vy]], dtype=np.float32)
        self.initialized = True

    def predict(self):
        return self.kf.predict()

    def correct(self, x: float, y: float) -> bool:
        """Fuse YOLO measurement; reject outliers far from prediction."""
        if self.initialized:
            px, py = self.get_position()
            if math.hypot(x - px, y - py) > self.max_gate_px:
                return False
        self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))
        return True

    def get_position(self) -> tuple[int, int]:
        return int(self.kf.statePost[0, 0]), int(self.kf.statePost[1, 0])

    def get_velocity(self) -> tuple[float, float]:
        return float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0])

    def speed_px_per_frame(self) -> float:
        vx, vy = self.get_velocity()
        return math.hypot(vx, vy)


def create_ball_kalman(fps: float) -> BallKalmanFilter:
    fps = max(fps, 1.0)
    return BallKalmanFilter(dt=1.0 / fps)
