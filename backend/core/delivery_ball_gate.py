"""
Delivery-ball-only gate: auto-calibrated bowler/machine → batsman corridor.

Only accepts YOLO detections that look like a ball in flight along the delivery
path (not outfield, crowd, pads, or static objects).
"""

from __future__ import annotations

from core.ball_detection_filters import (
    ball_area_limits,
    ball_bbox_size_ok,
    moving_toward_batsman,
    reject_bird_false_positive,
)
from core.config import CONFIG
from core.delivery_lane import DeliveryLane

_PROC = CONFIG.get("processing", {})
_SB = _PROC.get("small_ball_detect", {})
DELIVERY_BALL_ONLY = bool(_PROC.get("delivery_ball_only", True))
SMALL_BALL_ENABLED = bool(_SB.get("enabled", True))
SMALL_BALL_AREA_MULT = float(_SB.get("min_area_scale", 0.35))
LANE_MARGIN_PX = float(_SB.get("lane_margin_px", 72))
HIGH_CONF_BYPASS = float(_SB.get("high_conf_bypass", 0.32))


def small_ball_area_ok(area: float, height: int, width: int = 0) -> bool:
    """Allow smaller bboxes for distant / fast balls when still within sane limits."""
    if area <= 0:
        return True
    if ball_bbox_size_ok(area, height, width):
        return True
    if not SMALL_BALL_ENABLED:
        return False
    lo, hi = ball_area_limits(height, width)
    return lo * SMALL_BALL_AREA_MULT <= area <= hi


def is_delivery_ball_detection(
    cx: int,
    cy: int,
    area: float,
    det_conf: float,
    height: int,
    width: int,
    *,
    delivery_lane: DeliveryLane | None = None,
    track_active: bool = False,
    recent_points: list | None = None,
    frame=None,
    roi=None,
) -> bool:
    """
    True only for balls in the auto-calibrated delivery corridor moving toward batsman.
    """
    if not DELIVERY_BALL_ONLY:
        return True

    if track_active:
        return True

    if cy < height * 0.12 and reject_bird_false_positive(
        cx, cy, roi, height, frame=frame, width=width,
    ):
        return False

    if not small_ball_area_ok(area, height, width):
        return False

    if delivery_lane is not None and width > 0:
        if not delivery_lane.contains_detection(cx, cy, margin_px=LANE_MARGIN_PX):
            return False
        if not track_active:
            rel = delivery_lane.along_progress(float(cx), float(cy))
            if rel < -0.08 or rel > 1.06:
                return False

    pts = list(recent_points or [])
    if not track_active and len(pts) >= 2 and det_conf < HIGH_CONF_BYPASS:
        if not moving_toward_batsman(pts, height, width=width):
            return False

    return True
