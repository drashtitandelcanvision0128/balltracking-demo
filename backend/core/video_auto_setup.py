"""
Automatic video profiling + pitch calibration on upload.

Reads duration, resolution, quality (blur/brightness/shake), then picks
calibration samples and processing settings per video.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from core.config import CONFIG
from core.homography import is_landscape_frame
from core.pitch_calibrator import CalibrationResult, calibrate_pitch_robust
from core.video_calibration import derive_zones_from_quad

_PROC = CONFIG.get("processing", {})
_GPU = CONFIG.get("gpu", {})


@dataclass
class VideoProfile:
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float
    orientation: str
    resolution_tier: str
    quality_label: str
    blur_score: float
    brightness: float
    shake_px: float
    shaky: bool
    calib_samples: int
    detection_conf: float
    waiting_conf: float
    stabilize: str
    infer_waiting_imgsz: int
    infer_active_imgsz: int
    infer_max_dim: int
    calibration: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolution_tier(width: int, height: int) -> str:
    longest = max(width, height)
    if longest >= 3000:
        return "uhd"
    if longest >= 1900:
        return "fhd"
    if longest >= 1100:
        return "hd"
    return "sd"


def _quality_label(blur: float, brightness: float, shake: float) -> str:
    score = 0
    if blur >= 180:
        score += 2
    elif blur >= 90:
        score += 1
    if 55 <= brightness <= 200:
        score += 1
    if shake < 3.5:
        score += 1
    if score >= 3:
        return "good"
    if score >= 2:
        return "fair"
    return "low"


def _calib_samples_for_duration(duration_sec: float) -> int:
    if duration_sec < 20:
        return 15
    if duration_sec < 60:
        return 22
    if duration_sec < 180:
        return 30
    return min(45, int(20 + duration_sec / 12))


def _infer_dims(width: int, height: int, tier: str, quality: str) -> tuple[int, int, int]:
    longest = max(width, height, 1)
    if tier == "uhd":
        imgsz = int(_GPU.get("uhd_imgsz", 1920))
        max_dim = int(_GPU.get("uhd_max_dim", 1920))
    elif tier == "fhd":
        imgsz = int(_GPU.get("hd_imgsz", 1600))
        max_dim = int(_GPU.get("hd_max_dim", 1600))
    else:
        imgsz = int(_GPU.get("waiting_imgsz", 1280))
        max_dim = int(_GPU.get("waiting_max_dim", 1280))
    max_dim = min(max_dim, longest)
    imgsz = min(max(imgsz, 640), longest)
    if quality == "low":
        imgsz = min(longest, max(imgsz, 1280))
        max_dim = min(longest, max(max_dim, 1280))
    return imgsz, imgsz, max_dim


def _estimate_shake(frames: list[np.ndarray]) -> float:
    if len(frames) < 3:
        return 0.0
    mags: list[float] = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev = cv2.resize(prev, (320, int(320 * frames[0].shape[0] / max(frames[0].shape[1], 1))))
    for frame in frames[1:8]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (prev.shape[1], prev.shape[0]))
        flow = cv2.calcOpticalFlowFarneback(
            prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        mag = float(np.median(np.hypot(flow[..., 0], flow[..., 1])))
        mags.append(mag)
        prev = gray
    return float(np.median(mags)) if mags else 0.0


def _frame_quality_metrics(frames: list[np.ndarray]) -> tuple[float, float]:
    blurs: list[float] = []
    brights: list[float] = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurs.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        brights.append(float(np.mean(gray)))
    return (
        float(np.median(blurs)) if blurs else 0.0,
        float(np.median(brights)) if brights else 128.0,
    )


def _sample_probe_frames(cap, total_frames: int, n: int = 12) -> list[np.ndarray]:
    if total_frames <= 0:
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return [frame] if ret and frame is not None else []
    indices = np.linspace(int(total_frames * 0.08), int(total_frames * 0.75), n, dtype=int)
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frames


def analyze_and_calibrate_video(
    video_path: str,
    *,
    manual_quad: list | np.ndarray | None = None,
) -> VideoProfile:
    """
    Run once on upload: measure video + auto pitch calibration.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if fps > 0 and total_frames > 0 else 0.0

    probe_frames = _sample_probe_frames(cap, total_frames, n=12)
    blur, brightness = _frame_quality_metrics(probe_frames)
    shake = _estimate_shake(probe_frames)

    tier = _resolution_tier(width, height)
    quality = _quality_label(blur, brightness, shake)
    calib_samples = _calib_samples_for_duration(duration_sec)
    waiting_imgsz, active_imgsz, max_dim = _infer_dims(width, height, tier, quality)

    base_conf = float(CONFIG.get("model", {}).get("confidence", 0.06))
    waiting_conf = float(_PROC.get("waiting_conf", 0.04))
    if quality == "low":
        base_conf = max(0.04, base_conf * 0.85)
        waiting_conf = max(0.03, waiting_conf * 0.85)
    elif quality == "good" and tier in ("fhd", "uhd"):
        base_conf = min(0.10, base_conf * 1.05)

    stabilize = str(_PROC.get("stabilization", {}).get("mode", "auto"))
    if shake >= float(_PROC.get("stabilization", {}).get("shake_enable_px", 2.8)):
        stabilize = "always"

    calib = calibrate_pitch_robust(
        cap, width, height,
        manual_quad=manual_quad,
        max_samples=calib_samples,
    )
    cap.release()

    orientation = "landscape" if is_landscape_frame(width, height) else "portrait"
    zones = derive_zones_from_quad(calib.quad, width, height)
    profile = VideoProfile(
        width=width,
        height=height,
        fps=round(fps, 3),
        total_frames=total_frames,
        duration_sec=round(duration_sec, 2),
        orientation=orientation,
        resolution_tier=tier,
        quality_label=quality,
        blur_score=round(blur, 1),
        brightness=round(brightness, 1),
        shake_px=round(shake, 2),
        shaky=shake >= float(_PROC.get("stabilization", {}).get("shake_enable_px", 2.8)),
        calib_samples=calib_samples,
        detection_conf=round(base_conf, 4),
        waiting_conf=round(waiting_conf, 4),
        stabilize=stabilize,
        infer_waiting_imgsz=waiting_imgsz,
        infer_active_imgsz=active_imgsz,
        infer_max_dim=max_dim,
        calibration={
            "quad": calib.quad.tolist(),
            "source": calib.source,
            "confidence": round(float(calib.confidence), 3),
            "stump_scale": round(float(calib.stump_scale), 4),
            "zones": zones,
        },
    )
    return profile


def save_video_profile(video_path: str, profile: VideoProfile) -> str:
    base, _ = os.path.splitext(video_path)
    out = f"{base}_profile.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)
    return out


def load_video_profile(video_path: str) -> VideoProfile | None:
    base, _ = os.path.splitext(video_path)
    path = f"{base}_profile.json"
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return VideoProfile(**data)


def calibration_from_profile(profile: VideoProfile) -> CalibrationResult:
    cal = profile.calibration
    return CalibrationResult(
        quad=np.array(cal["quad"], dtype=np.float32),
        source=str(cal.get("source", "fallback")),
        confidence=float(cal.get("confidence", 0.25)),
        stump_scale=float(cal.get("stump_scale", 1.0)),
    )


def log_video_profile(profile: VideoProfile) -> None:
    cal = profile.calibration
    print(
        f"[AutoSetup] {profile.width}x{profile.height} {profile.duration_sec:.1f}s "
        f"{profile.orientation} tier={profile.resolution_tier} quality={profile.quality_label} "
        f"blur={profile.blur_score:.0f} shake={profile.shake_px:.1f}px "
        f"calib={cal.get('source')} conf={cal.get('confidence')} samples={profile.calib_samples} "
        f"infer={profile.infer_max_dim}px det_conf={profile.detection_conf} stabilize={profile.stabilize}",
        flush=True,
    )
