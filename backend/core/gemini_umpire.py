"""
Module 3 — Gemini vision umpire (PDF architecture).
ONLY bat-ball contact verification on 3 cropped frames. Temperature 0.
NOT used for ball tracking or bounce detection.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from core.config import CONFIG

_GEMINI = CONFIG.get("gemini", {})


@dataclass(frozen=True)
class GeminiVerdict:
    hit: bool
    miss: bool
    confidence: float
    reason: str
    source: str = "gemini"


def _load_env_file() -> None:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env_file()


class GeminiUmpire:
    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = cfg or _GEMINI
        self.enabled = bool(cfg.get("enabled", False))
        self.model_name = str(cfg.get("model", "gemini-2.0-flash"))
        env_name = str(cfg.get("api_key_env", "GEMINI_API_KEY"))
        self.api_key = os.environ.get(env_name, "").strip()
        self.mode = str(cfg.get("mode", "umpire")).lower()
        self.min_confidence = float(cfg.get("min_confidence", 0.65))
        self.review_low = float(cfg.get("review_band_low", 0.18))
        self.review_high = float(cfg.get("review_band_high", 0.55))
        self.verify_frames = int(cfg.get("verify_frames", 3))
        self.max_width = int(cfg.get("max_frame_width", 720))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.bounce_enabled = bool(cfg.get("bounce_enabled", False))
        self._model = None
        self._gen_config = None
        self._configured = bool(self.api_key)
        self._quota_blocked_until = 0.0
        self._quota_warned = False
        if self.enabled and not self._configured:
            print("[Gemini] enabled but GEMINI_API_KEY missing — copy backend/.env.example → backend/.env")
        elif self.available:
            print(
                f"[Gemini] Module 3 verify-only | model={self.model_name} "
                f"frames={self.verify_frames} temp={self.temperature}"
            )

    @property
    def available(self) -> bool:
        if time.time() < self._quota_blocked_until:
            return False
        return self.enabled and self.mode != "off" and bool(self.api_key)

    def _note_quota_error(self, exc: Exception) -> None:
        msg = str(exc)
        if "429" not in msg and "quota" not in msg.lower():
            return
        self._quota_blocked_until = time.time() + 120.0
        if not self._quota_warned:
            self._quota_warned = True
            print("[Gemini] API quota exceeded — hit verify paused for 2 min (heuristics only)")

    @property
    def configured(self) -> bool:
        return self._configured

    def should_review(self, heuristic_conf: float, heuristic_decided: bool) -> bool:
        if not self.available:
            return False
        if self.mode == "always":
            return True
        if self.mode != "umpire":
            return False
        if not heuristic_decided:
            return True
        return self.review_low <= heuristic_conf <= self.review_high

    def _model_client(self):
        if self._model is not None:
            return self._model
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(self.model_name)
        return self._model

    def _generation_config(self):
        if self._gen_config is not None:
            return self._gen_config
        import google.generativeai as genai

        self._gen_config = genai.types.GenerationConfig(temperature=self.temperature)
        return self._gen_config

    @staticmethod
    def _encode_frame(frame: np.ndarray, max_width: int) -> bytes:
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / w
            frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise ValueError("JPEG encode failed")
        return buf.tobytes()

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    @staticmethod
    def crop_bat_ball_sequence(
        frames: list[np.ndarray],
        *,
        ball_xy: tuple[int, int] | None = None,
        height: int = 1080,
        width: int = 1920,
    ) -> list[np.ndarray]:
        """PDF: 3-frame zoomed bat-ball crop sequence."""
        crops = []
        pad_y = int(height * 0.14)
        pad_x = int(width * 0.18)
        for frame in frames:
            h, w = frame.shape[:2]
            if ball_xy is not None:
                cx, cy = ball_xy
                x1 = max(0, cx - pad_x * 2)
                x2 = min(w, cx + pad_x * 2)
                y1 = max(0, cy - pad_y)
                y2 = min(h, cy + pad_y)
            else:
                x1, x2 = int(w * 0.15), int(w * 0.88)
                y1, y2 = int(h * 0.12), int(h * 0.92)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
        return crops

    def verify_bat_ball_contact(
        self,
        frames: list[np.ndarray],
        *,
        ball_xy: tuple[int, int] | None = None,
        height: int = 1080,
        width: int = 1920,
        heuristic_hit: bool = False,
        heuristic_miss: bool = False,
        heuristic_conf: float = 0.0,
    ) -> GeminiVerdict | None:
        if not self.available or not frames:
            return None

        heuristic_decided = heuristic_hit or heuristic_miss
        if not self.should_review(heuristic_conf, heuristic_decided):
            return None

        cropped = self.crop_bat_ball_sequence(
            frames[-self.verify_frames :], ball_xy=ball_xy, height=height, width=width,
        )
        if not cropped:
            return None

        try:
            parts: list[Any] = [
                (
                    "Cricket bat-ball contact verification (3 consecutive frames, batsman end).\n"
                    "Did the bat touch the ball? Or did the batsman miss/leave the ball?\n"
                    "Reply JSON only:\n"
                    '{"hit": boolean, "miss": boolean, "confidence": 0.0-1.0, "reason": "short"}\n'
                    "- hit=true: clear bat-ball contact or edge/deflection\n"
                    "- miss=true: ball passes without bat contact\n"
                    f"Math engine hint: hit={heuristic_hit} miss={heuristic_miss} conf={heuristic_conf:.2f}"
                )
            ]
            for img in cropped:
                parts.append({"mime_type": "image/jpeg", "data": self._encode_frame(img, self.max_width)})

            response = self._model_client().generate_content(
                parts, generation_config=self._generation_config(),
            )
            data = self._parse_json((response.text or "").strip())
            return GeminiVerdict(
                hit=bool(data.get("hit", False)),
                miss=bool(data.get("miss", False)),
                confidence=float(data.get("confidence", 0.0)),
                reason=str(data.get("reason", ""))[:120],
            )
        except Exception as exc:
            self._note_quota_error(exc)
            if "429" not in str(exc) and "quota" not in str(exc).lower():
                print(f"[Gemini] contact verify failed: {exc}")
            return None

    def classify_hit_miss(
        self,
        frames: list[np.ndarray],
        *,
        heuristic_hit: bool = False,
        heuristic_miss: bool = False,
        heuristic_conf: float = 0.0,
        ball_xy: tuple[int, int] | None = None,
        height: int = 1080,
        width: int = 1920,
    ) -> GeminiVerdict | None:
        return self.verify_bat_ball_contact(
            frames,
            ball_xy=ball_xy,
            height=height,
            width=width,
            heuristic_hit=heuristic_hit,
            heuristic_miss=heuristic_miss,
            heuristic_conf=heuristic_conf,
        )


_umpire: GeminiUmpire | None = None


def get_gemini_umpire() -> GeminiUmpire:
    global _umpire
    if _umpire is None:
        _umpire = GeminiUmpire()
    return _umpire
