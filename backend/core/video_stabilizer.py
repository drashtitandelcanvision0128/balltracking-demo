"""
Online video stabilization for handheld / shaky cricket clips.

Estimates camera motion between consecutive frames (ORB + affine) and warps
each frame so high-frequency handshake is reduced. Background features drive
the estimate; the fast-moving ball is treated as an RANSAC outlier.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.config import CONFIG

_PROC = CONFIG.get("processing", {})
_STAB = _PROC.get("stabilization", {}) if isinstance(_PROC.get("stabilization"), dict) else {}


def stabilization_config() -> dict:
    mode = str(_STAB.get("mode", _PROC.get("stabilize_mode", "auto"))).lower()
    if mode in ("1", "true", "yes", "on"):
        mode = "always"
    if mode in ("0", "false", "no", "off"):
        mode = "off"
    if mode not in ("auto", "always", "off"):
        mode = "auto"
    return {
        "mode": mode,
        "smooth": float(_STAB.get("smooth", 0.90)),
        "max_shift_ratio": float(_STAB.get("max_shift_ratio", 0.07)),
        "max_rotate_deg": float(_STAB.get("max_rotate_deg", 3.5)),
        "shake_enable_px": float(_STAB.get("shake_enable_px", 2.8)),
        "probe_frames": int(_STAB.get("probe_frames", 24)),
        "downsample": int(_STAB.get("downsample", 480)),
        "min_matches": int(_STAB.get("min_matches", 18)),
    }


class VideoStabilizer:
    """Per-frame camera-shake reduction for streaming pipelines."""

    def __init__(self, width: int, height: int, cfg: dict | None = None):
        self.cfg = cfg or stabilization_config()
        self.width = int(width)
        self.height = int(height)
        self.mode = self.cfg["mode"]
        self.enabled = self.mode == "always"
        self._probing = self.mode == "auto"
        self._probe_mags: list[float] = []
        self._prev_gray = None
        self._traj_x = 0.0
        self._traj_y = 0.0
        self._traj_a = 0.0
        self._smooth_x = 0.0
        self._smooth_y = 0.0
        self._smooth_a = 0.0
        self._frames = 0
        self._applied = 0
        self._last_mag = 0.0
        self._orb = cv2.ORB_create(nfeatures=600, fastThreshold=12)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        scale = self.cfg["downsample"] / max(width, height, 1)
        self._scale = min(1.0, scale)
        self._sw = max(32, int(width * self._scale))
        self._sh = max(32, int(height * self._scale))
        self._max_shift = max(8.0, self.cfg["max_shift_ratio"] * min(width, height))
        self._max_rot = np.deg2rad(self.cfg["max_rotate_deg"])
        self._alpha = float(np.clip(1.0 - self.cfg["smooth"], 0.03, 0.5))

    @property
    def active(self) -> bool:
        return self.enabled

    def stats(self) -> dict:
        return {
            "mode": self.mode,
            "enabled": self.enabled,
            "frames": self._frames,
            "applied": self._applied,
            "last_shake_px": round(self._last_mag, 2),
        }

    def reset(self):
        self._prev_gray = None
        self._traj_x = self._traj_y = self._traj_a = 0.0
        self._smooth_x = self._smooth_y = self._smooth_a = 0.0

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Stabilize one BGR frame; returns original on failure / while off."""
        if frame is None or self.mode == "off":
            return frame
        self._frames += 1

        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._scale < 1.0:
            gray = cv2.resize(gray_full, (self._sw, self._sh), interpolation=cv2.INTER_AREA)
        else:
            gray = gray_full

        if self._prev_gray is None:
            self._prev_gray = gray
            return frame

        dx, dy, da, mag = self._estimate_motion(self._prev_gray, gray)
        self._prev_gray = gray
        inv_s = 1.0 / self._scale if self._scale > 1e-6 else 1.0
        dx *= inv_s
        dy *= inv_s
        self._last_mag = mag * inv_s

        if self._probing:
            self._probe_mags.append(self._last_mag)
            if len(self._probe_mags) >= self.cfg["probe_frames"]:
                med = float(np.median(self._probe_mags))
                self.enabled = med >= self.cfg["shake_enable_px"]
                self._probing = False
                print(
                    f"[Stabilize] auto probe shake={med:.1f}px "
                    f"{'ON' if self.enabled else 'OFF'} "
                    f"(threshold={self.cfg['shake_enable_px']})"
                )
            if not self.enabled:
                return frame

        if not self.enabled:
            return frame

        self._traj_x += dx
        self._traj_y += dy
        self._traj_a += da

        # EMA of trajectory ≈ intentional pan; correction cancels residual shake
        self._smooth_x += self._alpha * (self._traj_x - self._smooth_x)
        self._smooth_y += self._alpha * (self._traj_y - self._smooth_y)
        self._smooth_a += self._alpha * (self._traj_a - self._smooth_a)

        corr_x = float(np.clip(self._smooth_x - self._traj_x, -self._max_shift, self._max_shift))
        corr_y = float(np.clip(self._smooth_y - self._traj_y, -self._max_shift, self._max_shift))
        corr_a = float(np.clip(self._smooth_a - self._traj_a, -self._max_rot, self._max_rot))

        M = cv2.getRotationMatrix2D(
            (self.width / 2.0, self.height / 2.0), np.rad2deg(corr_a), 1.0,
        )
        M[0, 2] += corr_x
        M[1, 2] += corr_y
        out = cv2.warpAffine(
            frame, M, (self.width, self.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        self._applied += 1
        return out

    def _estimate_motion(self, prev: np.ndarray, curr: np.ndarray):
        """Return (dx, dy, da_rad, magnitude_px) of camera motion prev→curr."""
        kp1, des1 = self._orb.detectAndCompute(prev, None)
        kp2, des2 = self._orb.detectAndCompute(curr, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return 0.0, 0.0, 0.0, 0.0

        matches = self._bf.knnMatch(des1, des2, k=2)
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < self.cfg["min_matches"]:
            return 0.0, 0.0, 0.0, 0.0

        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000,
        )
        if M is None:
            return 0.0, 0.0, 0.0, 0.0

        a, b = float(M[0, 0]), float(M[1, 0])
        dx, dy = float(M[0, 2]), float(M[1, 2])
        da = float(np.arctan2(b, a))
        mag = float(np.hypot(dx, dy))
        if inliers is not None and int(inliers.sum()) < max(6, self.cfg["min_matches"] // 2):
            return 0.0, 0.0, 0.0, mag
        return dx, dy, da, mag
