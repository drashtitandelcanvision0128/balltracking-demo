"""
Machine → pitch → batsman flight corridor (trapezoid in image space).

Rejects YOLO ball detections outside the delivery path so tracking locks
early at the bowling machine instead of only near the batsman.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.ball_detection_filters import (
    MACHINE_RELEASE_X_MAX,
    MACHINE_RELEASE_X_MIN,
    MACHINE_RELEASE_Y_MAX,
    MACHINE_RELEASE_Y_MIN,
    is_landscape_frame,
)
from core.config import CONFIG

_PROC = CONFIG.get("processing", {})
_CORR = _PROC.get("flight_corridor", {})


@dataclass
class FlightCorridor:
    """Convex quadrilateral covering machine release through batsman approach."""

    width: int
    height: int
    landscape: bool
    polygon: np.ndarray  # (4, 2) float32, clockwise
    source: str = "config"

    def contains(self, cx: int, cy: int, *, margin_px: float = 0.0) -> bool:
        if self.polygon is None or len(self.polygon) < 3:
            return True
        dist = cv2.pointPolygonTest(self.polygon, (float(cx), float(cy)), True)
        return dist >= -float(margin_px)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "landscape": self.landscape,
            "source": self.source,
            "polygon": self.polygon.astype(float).tolist(),
        }


def _pitch_left_edge_from_frame(frame: np.ndarray, height: int, width: int) -> int | None:
    """Estimate batsman-side pitch edge from tan/green strip (landscape)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (25, 18, 40), (95, 255, 255))
    pitch = cv2.inRange(hsv, (8, 12, 70), (32, 210, 255))
    mask = cv2.bitwise_or(grass, pitch)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    xs: list[int] = []
    for y_ratio in (0.45, 0.58, 0.72):
        y0 = int(height * (y_ratio - 0.04))
        y1 = int(height * (y_ratio + 0.04))
        band = mask[max(0, y0):min(height, y1), :]
        if band.size == 0:
            continue
        col_sum = band.sum(axis=0).astype(np.float32)
        if col_sum.max() < 1:
            continue
        thresh = col_sum.max() * 0.30
        cols = np.where(col_sum >= thresh)[0]
        if len(cols) < 8:
            continue
        xs.append(int(cols[0]))
    if not xs:
        return None
    return int(np.median(xs))


def _sample_pitch_left(cap, width: int, height: int, n_frames: int) -> int | None:
    if cap is None or n_frames <= 0:
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > n_frames * 3:
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
    else:
        indices = list(range(min(n_frames, max(total, 1))))

    edges: list[int] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        edge = _pitch_left_edge_from_frame(frame, height, width)
        if edge is not None:
            edges.append(edge)
    if not edges:
        return None
    return int(np.median(edges))


def _landscape_corridor(
    width: int,
    height: int,
    quad: np.ndarray | None,
    *,
    pitch_left_x: int | None = None,
    zones: dict | None = None,
) -> np.ndarray:
    expand_x = int(width * float(_CORR.get("expand_x_ratio", 0.06)))
    expand_y = int(height * float(_CORR.get("expand_y_ratio", 0.08)))
    flight_margin = int(_CORR.get("flight_margin_px", _PROC.get("pitch_flight_margin_px", 110)))

    mach_l = int(width * MACHINE_RELEASE_X_MIN) - expand_x
    mach_r = int(width * min(0.98, MACHINE_RELEASE_X_MAX + 0.04)) + expand_x
    mach_t = max(0, int(height * (MACHINE_RELEASE_Y_MIN - 0.06)) - expand_y)
    mach_b = min(height, int(height * (MACHINE_RELEASE_Y_MAX + 0.05)) + expand_y)

    bat_cfg = (zones or {}).get("batsman_end") or _CORR.get("batsman_end", {})
    bat_l = int(width * float(bat_cfg.get("x_min_ratio", 0.02)))
    bat_t = int(height * float(bat_cfg.get("y_min_ratio", 0.10)))
    bat_b = int(height * float(bat_cfg.get("y_max_ratio", 0.94)))

    if quad is not None and len(quad) >= 4:
        q = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
        left_x = float(min(q[0][0], q[2][0]))
        ys = [float(q[i][1]) for i in range(4)]
        bat_l = max(bat_l, int(left_x) - expand_x - flight_margin // 2)
        bat_t = min(bat_t, max(0, int(min(ys)) - flight_margin))
        bat_b = max(bat_b, min(height, int(max(ys)) + expand_y))

    if pitch_left_x is not None:
        bat_l = max(bat_l, pitch_left_x - expand_x - flight_margin // 3)

    mach_l = max(int(width * 0.38), mach_l)
    bat_l = min(bat_l, mach_l - int(width * 0.08))
    bat_l = max(0, bat_l)

    # Trapezoid: batsman (left) ↔ machine (right), full vertical flight band
    poly = np.array(
        [
            [bat_l, bat_t],
            [mach_r, mach_t],
            [mach_r, mach_b],
            [bat_l, bat_b],
        ],
        dtype=np.float32,
    )
    return poly


def _portrait_corridor(width: int, height: int, quad: np.ndarray | None) -> np.ndarray:
    expand_x = int(width * float(_CORR.get("expand_x_ratio", 0.05)))
    flight_margin = int(_CORR.get("flight_margin_px", _PROC.get("pitch_flight_margin_px", 110)))
    bat_cfg = _CORR.get("batsman_end", {})

    x_lo = int(width * float(bat_cfg.get("x_min_ratio", 0.14))) - expand_x
    x_hi = int(width * float(bat_cfg.get("x_max_ratio", 0.86))) + expand_x
    y_top = int(height * float(bat_cfg.get("y_min_ratio", 0.08))) - flight_margin // 2
    y_bot = int(height * float(bat_cfg.get("y_max_ratio", 0.94)))

    if quad is not None and len(quad) >= 4:
        q = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
        xs = [float(q[i][0]) for i in range(4)]
        ys = [float(q[i][1]) for i in range(4)]
        x_lo = min(x_lo, int(min(xs)) - expand_x)
        x_hi = max(x_hi, int(max(xs)) + expand_x)
        y_top = min(y_top, max(0, int(min(ys)) - flight_margin))
        y_bot = max(y_bot, min(height, int(max(ys)) + expand_x))

    poly = np.array(
        [
            [max(0, x_lo), max(0, y_top)],
            [min(width, x_hi), max(0, y_top)],
            [min(width, x_hi), min(height, y_bot)],
            [max(0, x_lo), min(height, y_bot)],
        ],
        dtype=np.float32,
    )
    return poly


def calibrate_flight_corridor(
    width: int,
    height: int,
    *,
    quad: np.ndarray | list | None = None,
    cap=None,
    zones: dict | None = None,
) -> FlightCorridor:
    """Build delivery corridor from pitch quad + machine zone (+ optional frame refine)."""
    landscape = is_landscape_frame(width, height)
    quad_np = np.asarray(quad, dtype=np.float32) if quad is not None else None
    source = "quad+config" if quad_np is not None else "config"

    pitch_left: int | None = None
    if cap is not None and bool(_CORR.get("auto_refine", True)) and landscape:
        n = int(_CORR.get("sample_frames", 20))
        pitch_left = _sample_pitch_left(cap, width, height, n)
        if pitch_left is not None:
            source = "quad+auto"

    if landscape:
        polygon = _landscape_corridor(width, height, quad_np, pitch_left_x=pitch_left, zones=zones)
    else:
        polygon = _portrait_corridor(width, height, quad_np)

    return FlightCorridor(
        width=width,
        height=height,
        landscape=landscape,
        polygon=polygon,
        source=source,
    )


def in_flight_corridor(
    cx: int,
    cy: int,
    corridor: FlightCorridor | None,
    *,
    margin_px: float | None = None,
) -> bool:
    if corridor is None or not bool(_CORR.get("enabled", True)):
        return True
    margin = float(margin_px if margin_px is not None else _CORR.get("margin_px", 40))
    return corridor.contains(cx, cy, margin_px=margin)
