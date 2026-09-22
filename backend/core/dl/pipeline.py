"""
Unified deep-learning pipeline for cricket ball analytics.

Components (all PyTorch / neural nets):
  1. YOLO ball detection      — existing Ultralytics CNN
  2. YOLO person detection    — optional yolo11n
  3. MediaPipe pose           — batsman / bat zone
  4. BounceTemporalNet        — LSTM bounce from track
  5. HitClassifierNet         — bat-ball contact
  6. LengthClassifierNet      — Yorker / Good Length / etc.
  7. Gemini vision            — optional umpire verify
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.config import CONFIG
from core.dl.models import BounceTemporalNet, HitClassifierNet, LENGTH_CLASSES, LengthClassifierNet
from core.dl.train_synthetic import FEAT_DIM, SEQ_LEN, _track_to_features, ensure_dl_weights

_DL = CONFIG.get("deep_learning", {})
_PROC = CONFIG.get("processing", {})


def dl_config() -> dict:
    return dict(_DL)


def is_dl_enabled() -> bool:
    return bool(_DL.get("enabled", False))


def _device() -> torch.device:
    if torch.cuda.is_available() and CONFIG.get("gpu", {}).get("require_cuda", False) is not False:
        return torch.device(f"cuda:{CONFIG.get('gpu', {}).get('device_id', 0)}")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


class DeepLearningPipeline:
    """Lazy-loaded DL models for bounce, hit, and length."""

    _instance: DeepLearningPipeline | None = None

    def __init__(self) -> None:
        self.enabled = is_dl_enabled()
        self.weights_dir = Path(_DL.get("weights_dir", "runs/dl"))
        self.min_bounce_conf = float(_DL.get("min_bounce_confidence", 0.55))
        self.prefer_dl = bool(_DL.get("prefer_dl_over_rules", True))
        self._device = _device()
        self._bounce: BounceTemporalNet | None = None
        self._hit: HitClassifierNet | None = None
        self._length: LengthClassifierNet | None = None
        self._ready = False

    @classmethod
    def get(cls) -> DeepLearningPipeline:
        if cls._instance is None:
            cls._instance = DeepLearningPipeline()
        return cls._instance

    def initialize(self) -> None:
        if not self.enabled or self._ready:
            return
        if bool(_DL.get("auto_train_bootstrap", True)):
            ensure_dl_weights(self.weights_dir)

        bounce_p = self.weights_dir / "bounce_temporal.pt"
        hit_p = self.weights_dir / "hit_classifier.pt"
        length_p = self.weights_dir / "length_classifier.pt"

        if bounce_p.exists():
            ckpt = torch.load(bounce_p, map_location=self._device, weights_only=False)
            self._bounce = BounceTemporalNet(
                ckpt.get("seq_len", SEQ_LEN),
                ckpt.get("feat_dim", FEAT_DIM),
            ).to(self._device)
            self._bounce.load_state_dict(ckpt["state_dict"])
            self._bounce.eval()

        if hit_p.exists():
            self._hit = HitClassifierNet(24).to(self._device)
            ckpt = torch.load(hit_p, map_location=self._device, weights_only=False)
            self._hit.load_state_dict(ckpt["state_dict"])
            self._hit.eval()

        if length_p.exists():
            self._length = LengthClassifierNet().to(self._device)
            ckpt = torch.load(length_p, map_location=self._device, weights_only=False)
            self._length.load_state_dict(ckpt["state_dict"])
            self._length.eval()

        self._ready = True
        parts = []
        if self._bounce:
            parts.append("bounce")
        if self._hit:
            parts.append("hit")
        if self._length:
            parts.append("length")
        print(f"[DL] Pipeline ready on {self._device} — {', '.join(parts) or 'no weights'}", flush=True)

    def predict_bounce(
        self,
        track_pts: list,
        release_frame: int,
        *,
        height: int,
        width: int,
        fps: float = 30.0,
        delivery_lane=None,
    ) -> tuple[int, float, float, float] | None:
        """
        Returns (frame, x, y, confidence) or None.
        """
        if not self.enabled or self._bounce is None:
            return None

        seg = []
        for p in track_pts:
            if int(p[0]) < release_frame:
                continue
            if len(p) >= 3:
                seg.append((float(p[1]), float(p[2])))
        if len(seg) < 8:
            return None

        pts = seg[-SEQ_LEN:]
        feats = _track_to_features(pts, height, width)
        x_t = torch.tensor(feats[np.newaxis], device=self._device, dtype=torch.float32)

        with torch.no_grad():
            probs = self._bounce(x_t).cpu().numpy()[0]

        n = len(pts)
        probs = probs[:n]
        bi = int(np.argmax(probs))
        conf = float(probs[bi])
        if conf < self.min_bounce_conf:
            return None

        min_after = max(6, int(fps * 0.10))
        if bi < min_after:
            return None

        x, y = pts[bi]
        if delivery_lane is not None and not delivery_lane.between_ends(x, y):
            return None

        frame_offset = len(seg) - n
        frame_idx = release_frame + frame_offset + bi
        return int(frame_idx), float(x), float(y), conf

    def predict_hit(
        self,
        raw_pts: list,
        hist_pts: list,
        height: int,
        fps: float,
        *,
        pose_frames: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, float]:
        if not self.enabled or self._hit is None or len(hist_pts) < 5:
            return False, 0.0

        feat = np.zeros(24, dtype=np.float32)
        pts = hist_pts[-8:]
        speeds = []
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            speeds.append(math.hypot(dx, dy))
        if speeds:
            feat[0] = max(speeds) / max(height * 0.05, 1)
            feat[1] = np.std(speeds) / max(height * 0.02, 1)
        if len(pts) >= 3:
            v1 = (pts[-2][0] - pts[-3][0], pts[-2][1] - pts[-3][1])
            v2 = (pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1])
            n1 = math.hypot(*v1) or 1.0
            n2 = math.hypot(*v2) or 1.0
            feat[2] = (v1[0] / n1 * v2[0] / n2 + v1[1] / n1 * v2[1] / n2)

        if pose_frames:
            bat_dists = []
            cx, cy = pts[-1][0], pts[-1][1]
            for pf in pose_frames:
                bat = pf.get("bat_zone")
                if bat:
                    bat_dists.append(math.hypot(cx - bat[0], cy - bat[1]) / height)
            if bat_dists:
                feat[8] = max(0.0, 1.0 - min(bat_dists) * 5.0)

        x_t = torch.tensor(feat[np.newaxis], device=self._device, dtype=torch.float32)
        with torch.no_grad():
            prob = float(self._hit(x_t).cpu().item())
        return prob >= 0.5, prob

    def classify_length(self, x_m: float, y_m: float) -> tuple[str, float]:
        if not self.enabled or self._length is None:
            return "", 0.0
        x_t = torch.tensor([[x_m / 1.5, y_m / 20.12]], device=self._device, dtype=torch.float32)
        with torch.no_grad():
            logits = self._length(x_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return LENGTH_CLASSES[idx], float(probs[idx])


def apply_dl_config_overrides() -> dict[str, bool]:
    """When DL mode is on, enable pose / players / gemini bounce."""
    if not is_dl_enabled():
        return {}
    overrides = {}
    gpu = CONFIG.setdefault("gpu", {})
    auto = _PROC.setdefault("fully_automatic", {})
    gem = CONFIG.setdefault("gemini", {})

    if bool(_DL.get("enable_pose", True)):
        gpu["enable_pose"] = True
        overrides["pose"] = True
    if bool(_DL.get("enable_player_detection", True)):
        auto["player_detection"] = True
        overrides["players"] = True
    if bool(_DL.get("enable_gemini_bounce", False)):
        gem["bounce_enabled"] = True
        overrides["gemini_bounce"] = True
    return overrides
