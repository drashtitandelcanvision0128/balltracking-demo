"""Load and expose system configuration from config.yaml."""

import os
from pathlib import Path
from typing import Any

import yaml

_BASE_DIR = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _BASE_DIR / "config.yaml"

_DEFAULTS: dict[str, Any] = {
    "model": {
        "confidence": 0.35,
        "image_size": 1280,
        "path": "runs/detect/train5/weights/best.pt",
        "tracker": "bytetrack",
    },
    "physics": {
        "hit_angle_threshold": 35,
        "speed_range": [80, 180],
        "spin_multiplier": 0.4,
        "swing_multiplier": 0.2,
    },
    "bowling_speed": {
        "global_min_kmh": 70,
        "global_max_kmh": 180,
        "tiers": [
            {"id": "medium", "label": "Medium Pace", "avg_min": 80, "avg_max": 120, "max_speed": 140},
            {"id": "fast_medium", "label": "Fast Medium", "avg_min": 120, "avg_max": 150, "max_speed": 160},
            {"id": "fast", "label": "Fast Bowling", "avg_min": 130, "avg_max": 160, "max_speed": 180},
        ],
    },
    "processing": {
        "ffmpeg_path": "ffmpeg",
        "max_file_size_mb": 500,
        "supported_formats": [".mp4", ".avi", ".mov"],
    },
    "tracking": {
        "lost_threshold": 5,
        "max_queue_size": 120,
        "max_skip_frames": 10,
        "stride": 1,
        "coast_seconds": 0.45,
    },
    "pitch": {
        "length_m": 20.12,
        "width_m": 3.05,
        "stump_width_m": 0.2286,
        "calibration_samples": 30,
        "prefer_manual_calibration": True,
    },
    "visualization": {
        "show_heatmap": True,
        "show_predictions": False,
        "trail_length": 0,
        "show_trajectory_lines": False,
        "show_corner_pitch_map": False,
        "show_summary_pitch_map": False,
        "show_pitch_zone_overlay": False,
    },
    "database": {
        "url": os.environ.get(
            "DATABASE_URL",
            f"sqlite:///{_BASE_DIR / 'cricket_analytics.db'}",
        ),
    },
    "redis": {
        "url": os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    },
    "gpu": {
        "require_cuda": False,
        "device_id": 0,
        "half_precision": True,
        "cudnn_benchmark": True,
        "allow_tf32": True,
        "fuse_layers": True,
        "memory_fraction": 0.85,
        "warmup_imgsz": 640,
        "waiting_imgsz": 640,
        "waiting_max_dim": 640,
        "waiting_stride": 3,
        "active_imgsz": 1280,
        "active_max_dim": 1280,
        "nvenc": True,
        "enable_pose": True,
        "calibration_samples": 30,
        "summary_seconds": 1,
    },
    "delivery_filter": {
        "min_track_frames": 6,
        "min_track_frames_strict": 8,
        "min_y_travel_ratio": 0.08,
        "min_delivery_gap_sec": 0.8,
        "min_new_detection_conf": 0.25,
        "absurd_speed_kmh": 500,
        "min_frames_between_markers": 14,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or _CONFIG_PATH
    cfg = dict(_DEFAULTS)
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, loaded)
    model_path = cfg["model"]["path"]
    if not os.path.isabs(model_path):
        cfg["model"]["path"] = str(_BASE_DIR / model_path)
    return cfg


CONFIG = load_config()
