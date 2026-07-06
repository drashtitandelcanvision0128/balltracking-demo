"""
Shared YOLO ball-candidate filters — cricket ball colours + reject sky/bird false positives.
"""

from __future__ import annotations

import cv2

from core.config import CONFIG

_PROC = CONFIG.get("processing", {})
# Ball must reach this vertical band (toward batsman) before we lock a delivery — ignores flight in the air
BALL_APPROACH_Y_MIN = float(_PROC.get("ball_approach_y_min", 0.42))


def ball_roi_passes(roi) -> bool:
    """Accept red/pink or white cricket ball pixels; reject flat grass and dark shadows."""
    if roi is None or roi.size == 0:
        return False
    if len(roi.shape) == 2:
        return float(roi.mean()) >= 70.0
    brightness = float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean())
    b_ch, g_ch, r_ch = [float(x) for x in roi.reshape(-1, 3).mean(axis=0)]
    color_range = max(b_ch, g_ch, r_ch) - min(b_ch, g_ch, r_ch)
    # White ball (ODI / many net sessions)
    if brightness >= 125 and color_range < 65 and min(b_ch, g_ch, r_ch) >= 95:
        return True
    # Red / pink leather ball
    if brightness >= 75 and r_ch >= 100 and color_range >= 22:
        if r_ch >= g_ch - 25 and r_ch >= b_ch - 15:
            return True
    return False


def _strong_red_ball(roi) -> bool:
    if roi is None or roi.size == 0 or len(roi.shape) != 3:
        return False
    b_ch, g_ch, r_ch = [float(x) for x in roi.reshape(-1, 3).mean(axis=0)]
    return r_ch >= 118 and r_ch >= g_ch + 12 and r_ch >= b_ch + 18


def _looks_like_white_blob(roi) -> bool:
    if roi is None or roi.size == 0:
        return False
    if len(roi.shape) == 2:
        return float(roi.mean()) >= 120.0
    brightness = float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean())
    b_ch, g_ch, r_ch = [float(x) for x in roi.reshape(-1, 3).mean(axis=0)]
    color_range = max(b_ch, g_ch, r_ch) - min(b_ch, g_ch, r_ch)
    return brightness >= 118 and color_range < 72 and min(b_ch, g_ch, r_ch) >= 88


def _sky_background_near(cx: int, cy: int, frame, width: int, height: int) -> bool:
    """True when pixels above/near the box look like open sky (birds, clouds)."""
    pad = 14
    y2 = max(0, cy - 6)
    y1 = max(0, y2 - 36)
    x1 = max(0, cx - pad * 2)
    x2 = min(width, cx + pad * 2)
    if y2 <= y1 or x2 <= x1:
        return False
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0 or len(patch.shape) != 3:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sat = float(hsv[:, :, 1].mean())
    val = float(hsv[:, :, 2].mean())
    # Pale sky / cloud: low saturation, fairly bright
    return sat < 55 and val > 145


def reject_bird_false_positive(
    cx: int,
    cy: int,
    roi,
    height: int,
    frame=None,
    width: int = 0,
) -> bool:
    """Return True when detection should be discarded (likely bird / sky clutter)."""
    sky_top = height * 0.22
    upper_mid = height * 0.42

    # Top sky band — only a clear red cricket ball is allowed
    if cy < sky_top:
        if _strong_red_ball(roi):
            return False
        return True

    # White blobs in upper frame (seagulls, kit, clouds) are not deliveries
    if cy < upper_mid and _looks_like_white_blob(roi):
        return True

    if frame is not None and width > 0:
        if cy < upper_mid and _looks_like_white_blob(roi) and _sky_background_near(cx, cy, frame, width, height):
            return True
        # Any light round blob against open sky
        if cy < height * 0.30 and _sky_background_near(cx, cy, frame, width, height):
            if not _strong_red_ball(roi):
                return True

    return False


def reject_ball_in_flight(cy: int, height: int, post_contact: bool = False) -> bool:
    """Return True when the ball is still in the air (should not be detected)."""
    if post_contact:
        y_min = float(_PROC.get("ball_approach_y_min_after_contact", 0.28))
    else:
        y_min = BALL_APPROACH_Y_MIN
    return cy < int(height * y_min)


def in_batsman_approach_zone(cy: int, height: int, y_min_ratio: float | None = None) -> bool:
    """True when the ball is in the lower approach band near the batsman (not high in flight)."""
    threshold = BALL_APPROACH_Y_MIN if y_min_ratio is None else y_min_ratio
    return cy >= int(height * threshold)


def allow_ball_detection(cy: int, height: int, *, post_contact: bool = False) -> bool:
    """YOLO may only return a ball that is not in the pre-bounce flight path."""
    return not reject_ball_in_flight(cy, height, post_contact=post_contact)


def allow_ball_lock(cy: int, height: int, track_active: bool) -> bool:
    """Deprecated alias — always reject flight detections."""
    return allow_ball_detection(cy, height)


def ball_candidate_ok(
    cx: int,
    cy: int,
    roi,
    height: int,
    width: int,
    frame=None,
    *,
    post_contact: bool = False,
) -> bool:
    if reject_ball_in_flight(cy, height, post_contact=post_contact):
        return False
    if not ball_roi_passes(roi):
        return False
    if reject_bird_false_positive(cx, cy, roi, height, frame=frame, width=width):
        return False
    return True
