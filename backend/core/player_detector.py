"""
AI player detection — bowler, striker, non-striker from YOLO person boxes + pitch zones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from core.ball_detection_filters import is_landscape_frame
from core.config import CONFIG

_AUTO = CONFIG.get("processing", {}).get("fully_automatic", {})
_PERSON_MODEL = str(_AUTO.get("person_model", "yolo11n.pt"))
_PERSON_CONF = float(_AUTO.get("person_conf", 0.35))
_SAMPLE_FRAMES = int(_AUTO.get("player_sample_frames", 8))


@dataclass
class DetectedPlayer:
    role: str
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    confidence: float


@dataclass
class PlayerScene:
    landscape: bool
    players: list[DetectedPlayer] = field(default_factory=list)
    striker: DetectedPlayer | None = None
    bowler: DetectedPlayer | None = None
    non_striker: DetectedPlayer | None = None

    def to_dict(self) -> dict[str, Any]:
        def _p(p: DetectedPlayer | None) -> dict | None:
            if p is None:
                return None
            return {
                "role": p.role,
                "bbox": list(p.bbox),
                "center": [round(p.center[0], 1), round(p.center[1], 1)],
                "confidence": round(p.confidence, 3),
            }

        return {
            "landscape": self.landscape,
            "striker": _p(self.striker),
            "bowler": _p(self.bowler),
            "non_striker": _p(self.non_striker),
            "count": len(self.players),
        }


class CricketPlayerDetector:
    """Detect persons and assign cricket roles using calibrated pitch zones."""

    def __init__(self, device: str | int = "cpu", half: bool = False):
        self.device = device
        self.half = half
        self._model = None

    def _model_client(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(_PERSON_MODEL)
            self._model.to(self.device)
        return self._model

    def detect_scene(
        self,
        frames: list[np.ndarray],
        width: int,
        height: int,
        *,
        zones: dict | None = None,
    ) -> PlayerScene:
        if not frames or not bool(_AUTO.get("player_detection", True)):
            return PlayerScene(landscape=is_landscape_frame(width, height))

        landscape = is_landscape_frame(width, height)
        boxes: list[tuple[int, int, int, int, float, float, float]] = []
        model = self._model_client()
        for frame in frames[:_SAMPLE_FRAMES]:
            small = frame
            scale = 1.0
            if max(width, height) > 960:
                scale = 960 / max(width, height)
                small = cv2.resize(
                    frame,
                    (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            results = model.predict(
                small,
                conf=_PERSON_CONF,
                classes=[0],
                verbose=False,
                device=self.device,
                half=self.half,
                imgsz=640,
            )
            inv = 1.0 / scale
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                cx = (x1 + x2) / 2 * inv
                cy = (y1 + y2) / 2 * inv
                boxes.append((
                    int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv),
                    cx, cy, conf,
                ))

        if not boxes:
            return PlayerScene(landscape=landscape)

        # Cluster duplicate detections across frames (median center per person)
        clusters = _cluster_boxes(boxes, merge_px=max(width, height) * 0.06)
        players = [
            DetectedPlayer(
                role="unknown",
                bbox=(int(np.median([b[0] for b in c])), int(np.median([b[1] for b in c])),
                      int(np.median([b[2] for b in c])), int(np.median([b[3] for b in c]))),
                center=(float(np.median([b[4] for b in c])), float(np.median([b[5] for b in c]))),
                confidence=float(np.mean([b[6] for b in c])),
            )
            for c in clusters
        ]

        mach = (zones or {}).get("machine_release_zone", {})
        bat = (zones or {}).get("batsman_end", {})
        striker, bowler, non_striker = _assign_roles(players, width, height, landscape, mach, bat)
        for p in players:
            if striker and p.center == striker.center:
                p.role = "striker"
            elif bowler and p.center == bowler.center:
                p.role = "bowler"
            elif non_striker and p.center == non_striker.center:
                p.role = "non_striker"

        return PlayerScene(
            landscape=landscape,
            players=players,
            striker=striker,
            bowler=bowler,
            non_striker=non_striker,
        )


def _cluster_boxes(
    boxes: list[tuple[int, int, int, int, float, float, float]],
    merge_px: float,
) -> list[list[tuple]]:
    clusters: list[list[tuple]] = []
    for b in boxes:
        placed = False
        for cluster in clusters:
            cx, cy = cluster[0][4], cluster[0][5]
            if abs(b[4] - cx) < merge_px and abs(b[5] - cy) < merge_px:
                cluster.append(b)
                placed = True
                break
        if not placed:
            clusters.append([b])
    return [c for c in clusters if len(c) >= 1]


def _in_zone_ratio(cx: float, cy: float, width: int, height: int, zone: dict) -> bool:
    if not zone:
        return False
    xr, yr = cx / max(width, 1), cy / max(height, 1)
    return (
        float(zone.get("x_min_ratio", 0)) <= xr <= float(zone.get("x_max_ratio", 1))
        and float(zone.get("y_min_ratio", 0)) <= yr <= float(zone.get("y_max_ratio", 1))
    )


def _assign_roles(
    players: list[DetectedPlayer],
    width: int,
    height: int,
    landscape: bool,
    mach: dict,
    bat: dict,
) -> tuple[DetectedPlayer | None, DetectedPlayer | None, DetectedPlayer | None]:
    if not players:
        return None, None, None

    striker = bowler = non_striker = None
    if landscape:
        bat_cands = [p for p in players if _in_zone_ratio(p.center[0], p.center[1], width, height, bat)]
        mach_cands = [p for p in players if _in_zone_ratio(p.center[0], p.center[1], width, height, mach)]
        if not bat_cands or not mach_cands:
            # Closest / largest person = batsman; farthest along pitch = bowler.
            by_size = sorted(players, key=lambda p: -_bbox_area(p))
            striker = by_size[0]
            rest = [p for p in players if p is not striker]
            if rest:
                bowler = max(rest, key=lambda p: _dist(p, striker))
            if bat_cands:
                striker = max(bat_cands, key=lambda p: p.confidence)
            if mach_cands:
                bowler = max(mach_cands, key=lambda p: p.confidence)
        else:
            striker = max(bat_cands, key=lambda p: p.confidence)
            bowler = max(mach_cands, key=lambda p: p.confidence)
        others = [p for p in players if p not in (striker, bowler)]
        if striker and others:
            non_striker = min(others, key=lambda p: _dist(p, striker))
    else:
        # Behind-batsman portrait: batsman is near/bottom, bowler is far/top.
        # Batsman stands BESIDE the pitch — pick the near person closest to center, not a sideline body.
        mid_x = width * 0.5
        lower = [p for p in players if p.center[1] >= height * 0.42]
        upper = [p for p in players if p.center[1] <= height * 0.62]
        if not lower:
            lower = sorted(players, key=lambda p: -p.bbox[3])[:2]
        if not upper:
            upper = sorted(players, key=lambda p: p.center[1])[:2]
        striker = min(lower, key=lambda p: abs(p.center[0] - mid_x) * 0.65 + (height - p.bbox[3]) * 0.002)
        bowler_cands = [p for p in upper if p is not striker]
        bowler = min(bowler_cands, key=lambda p: p.center[1]) if bowler_cands else None
        others = [p for p in players if p not in (striker, bowler)]
        non_striker = min(others, key=lambda p: _dist(p, striker)) if striker and others else None

    return striker, bowler, non_striker


def _bbox_area(p: DetectedPlayer) -> float:
    x1, y1, x2, y2 = p.bbox
    return max(1.0, float(x2 - x1) * float(y2 - y1))


def _dist(a: DetectedPlayer, b: DetectedPlayer) -> float:
    return float(np.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]))


def draw_player_labels(frame: np.ndarray, scene: PlayerScene) -> None:
    """Overlay role labels on annotated output video."""
    colors = {
        "striker": (0, 200, 255),
        "bowler": (0, 255, 120),
        "non_striker": (255, 180, 0),
    }
    for role_key in ("striker", "bowler", "non_striker"):
        p = getattr(scene, role_key, None)
        if p is None:
            continue
        x1, y1, x2, y2 = p.bbox
        col = colors.get(role_key, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
        label = role_key.replace("_", " ").title()
        cv2.putText(
            frame, label, (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA,
        )
