"""
Shared YOLO ball-candidate filters — cricket ball colours + reject sky/bird false positives.
"""

from __future__ import annotations

import math

import cv2

from core.config import CONFIG

_PROC = CONFIG.get("processing", {})
# Track ball from release through flight toward batsman (not only near stumps)
TRACK_DELIVERY_IN_FLIGHT = bool(_PROC.get("track_delivery_in_flight", True))
REJECT_GROUND_RESTING_BALL = bool(_PROC.get("reject_ground_resting_ball", True))
# Ball must reach this vertical band (toward batsman) before we lock a delivery — legacy mode only
BALL_APPROACH_Y_MIN = float(_PROC.get("ball_approach_y_min", 0.42))
BOUNCE_GROUND_Y_MIN = float(_PROC.get("bounce_ground_y_min", 0.65))
GROUND_RESTING_Y_OFFSET = float(_PROC.get("ground_resting_y_offset", 0.06))
_FILT = CONFIG.get("delivery_filter", {})
MIN_LOCK_DY_PX = float(_FILT.get("min_lock_dy_px", 8))
MIN_LOCK_TRAVEL_PX = float(_FILT.get("min_lock_travel_px", 10))
# Size limits tuned on ~478x850 portrait; scale with resolution for 1080p/4K clips
BALL_SIZE_REF_HEIGHT = float(_PROC.get("ball_size_ref_height", 850))
BALL_MIN_AREA_REF = float(_PROC.get("ball_min_area", 25))
BALL_MAX_AREA_REF = float(_PROC.get("ball_max_area", 550))
BALL_CLASS_IDS = {0, 13}  # ball, tennis-ball
REJECT_METALLIC_HARDWARE = bool(_PROC.get("reject_metallic_hardware", True))
REQUIRE_BALL_COLOR_FOR_LOCK = bool(_PROC.get("require_ball_color_for_lock", True))
DETECT_PITCH_AREA_ONLY = bool(_PROC.get("detect_pitch_area_only", True))
PITCH_FLIGHT_MARGIN_PX = int(_PROC.get("pitch_flight_margin_px", 110))
BALL_COLOR_SKIP_CONF = float(_PROC.get("ball_color_skip_conf", 0.22))
_MACH = _PROC.get("machine_release_zone", {})
MACHINE_RELEASE_ENABLED = bool(_MACH.get("enabled", True))
MACHINE_RELEASE_X_MIN = float(_MACH.get("x_min_ratio", 0.50))
MACHINE_RELEASE_X_MAX = float(_MACH.get("x_max_ratio", 0.90))
MACHINE_RELEASE_Y_MIN = float(_MACH.get("y_min_ratio", 0.18))
MACHINE_RELEASE_Y_MAX = float(_MACH.get("y_max_ratio", 0.76))


def is_ball_class(cls_id: int) -> bool:
    return int(cls_id) in BALL_CLASS_IDS


def is_landscape_frame(width: int, height: int) -> bool:
    """Wide-angle / side-on clips — batsman is not at the bottom edge."""
    return width > height * 1.12


def effective_approach_y_min(width: int, height: int) -> float:
    if is_landscape_frame(width, height):
        return float(_PROC.get("ball_approach_y_min_landscape", 0.42))
    return BALL_APPROACH_Y_MIN


def effective_bounce_ground_y_min(width: int, height: int) -> float:
    if is_landscape_frame(width, height):
        return float(_PROC.get("bounce_ground_y_min_landscape", 0.52))
    return BOUNCE_GROUND_Y_MIN


def ball_area_limits(height: int, width: int = 0) -> tuple[float, float]:
    """Min/max bbox area (px²) scaled for video resolution."""
    scale = max(0.45, height / BALL_SIZE_REF_HEIGHT) ** 2
    min_a = BALL_MIN_AREA_REF * scale * 0.55
    max_a = BALL_MAX_AREA_REF * scale * 2.2
    if width > 0:
        max_a = min(max_a, width * height * 0.0018)
    return min_a, max_a


def ball_bbox_size_ok(area: float, height: int, width: int = 0) -> bool:
    min_a, max_a = ball_area_limits(height, width)
    return min_a <= area <= max_a


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


def _looks_like_dark_hardware(roi) -> bool:
    """Shadowed nuts/bolts — very dark, near-neutral (not red/white ball)."""
    if roi is None or roi.size == 0 or len(roi.shape) != 3:
        return False
    if _strong_red_ball(roi) or _looks_like_white_blob(roi):
        return False
    brightness = float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean())
    b_ch, g_ch, r_ch = [float(x) for x in roi.reshape(-1, 3).mean(axis=0)]
    color_range = max(b_ch, g_ch, r_ch) - min(b_ch, g_ch, r_ch)
    return brightness < 58 and color_range < 22


def looks_like_metallic_hardware(roi) -> bool:
    """Machine nuts/bolts — gray/silver, low chroma (not red/white cricket ball)."""
    if roi is None or roi.size == 0 or len(roi.shape) != 3:
        return False
    if _strong_red_ball(roi) or _looks_like_white_blob(roi):
        return False
    if _looks_like_dark_hardware(roi):
        return True
    brightness = float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean())
    b_ch, g_ch, r_ch = [float(x) for x in roi.reshape(-1, 3).mean(axis=0)]
    color_range = max(b_ch, g_ch, r_ch) - min(b_ch, g_ch, r_ch)
    sat = float(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
    if brightness < 45 or brightness > 215:
        return False
    if color_range > 42 or sat > 62:
        return False
    if abs(r_ch - g_ch) < 24 and abs(g_ch - b_ch) < 24 and abs(r_ch - b_ch) < 24:
        return True
    return False


def reject_machine_zone(cx: int, cy: int, width: int, height: int) -> bool:
    """Static machine hardware on turf — not the ball release window."""
    if width <= 0 or not is_landscape_frame(width, height):
        return False
    if in_machine_release_zone(cx, cy, width, height):
        return False
    xr, yr = cx / max(width, 1), cy / max(height, 1)
    return xr > 0.68 and yr > 0.78


def in_machine_release_zone(cx: int, cy: int, width: int, height: int) -> bool:
    """Bowling-machine ball exit — right side, mid-frame (landscape side-on)."""
    if not MACHINE_RELEASE_ENABLED or width <= 0 or not is_landscape_frame(width, height):
        return False
    xr, yr = cx / max(width, 1), cy / max(height, 1)
    return (
        MACHINE_RELEASE_X_MIN <= xr <= MACHINE_RELEASE_X_MAX
        and MACHINE_RELEASE_Y_MIN <= yr <= MACHINE_RELEASE_Y_MAX
    )


def _roi_looks_like_cricket_ball(roi) -> bool:
    return ball_roi_passes(roi) or _strong_red_ball(roi) or _looks_like_white_blob(roi)


def in_pitch_detection_zone(
    cx: int,
    cy: int,
    width: int,
    height: int,
    h_matrix=None,
    *,
    allow_flight_above: bool = True,
) -> bool:
    """True when detection lies on the pitch strip (or in flight corridor above it)."""
    if in_machine_release_zone(cx, cy, width, height):
        return True

    if h_matrix is not None:
        if cy < int(height * 0.16):
            return False
        from core.homography import video_to_pitchmap
        from pitch_map_renderer import PITCH_L, PITCH_R, PITCH_TOP, PITCH_BOT
        px, py = video_to_pitchmap(cx, cy, h_matrix)
        if not (PITCH_L - 18 <= px <= PITCH_R + 18):
            return False
        top = PITCH_TOP - (PITCH_FLIGHT_MARGIN_PX if allow_flight_above else 0)
        return top <= py <= PITCH_BOT + 20

    if is_landscape_frame(width, height):
        if reject_machine_zone(cx, cy, width, height):
            return False
        xr, yr = cx / max(width, 1), cy / max(height, 1)
        if allow_flight_above:
            return 0.18 <= xr <= 0.82 and 0.20 <= yr <= 0.90
        return 0.18 <= xr <= 0.82 and 0.46 <= yr <= 0.90

    xr, yr = cx / max(width, 1), cy / max(height, 1)
    if allow_flight_above:
        return 0.18 <= xr <= 0.82 and 0.30 <= yr <= 0.92
    return 0.22 <= xr <= 0.78 and 0.52 <= yr <= 0.92


def reject_hardware_false_positive(
    cx: int,
    cy: int,
    roi,
    height: int,
    width: int = 0,
) -> bool:
    """Reject nuts/bolts and other gray machine hardware mislabeled as ball."""
    if width > 0 and in_machine_release_zone(cx, cy, width, height):
        if _roi_looks_like_cricket_ball(roi):
            return False
    if not REJECT_METALLIC_HARDWARE or not looks_like_metallic_hardware(roi):
        return False
    if width > 0 and in_ground_resting_band(cy, height, width):
        return True
    if width > 0 and is_landscape_frame(width, height):
        if cx > int(width * 0.68) and cy > int(height * 0.70):
            return True
    return not ball_roi_passes(roi)


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


def moving_toward_batsman(
    points: list,
    height: int,
    min_dy: float | None = None,
    width: int = 0,
) -> bool:
    """True when track moves like a delivery (down on rear cam, across on side-on)."""
    if len(points) < 2:
        return False
    dx = abs(points[-1][0] - points[0][0])
    dy = points[-1][1] - points[0][1]
    need = min_dy if min_dy is not None else max(MIN_LOCK_DY_PX, height * 0.004)
    travel = math.hypot(dx, dy)
    if width > 0 and is_landscape_frame(width, height):
        return travel >= max(MIN_LOCK_TRAVEL_PX, height * 0.004) and (
            dx >= need * 0.5 or dy >= need
        )
    return dy >= need


def in_ground_resting_band(cy: int, height: int, width: int = 0) -> bool:
    """Bottom strip only — where dead balls sit after stopping (not the whole pitch path)."""
    if is_landscape_frame(width, height):
        y_ratio = float(_PROC.get("ground_resting_y_min_landscape", 0.82))
    else:
        y_ratio = float(_PROC.get("ground_resting_y_min_portrait", 0.78))
    return cy >= int(height * y_ratio)


def reject_resting_ball_on_ground(
    cy: int,
    height: int,
    width: int = 0,
    *,
    track_active: bool = False,
    recent_points: list | None = None,
) -> bool:
    """Reject stationary balls on the turf — never block first detections in the pitch band."""
    if not TRACK_DELIVERY_IN_FLIGHT or not REJECT_GROUND_RESTING_BALL:
        return False
    if not in_ground_resting_band(cy, height, width):
        return False
    pts = list(recent_points or [])
    # Need motion history before calling something static (avoid blocking lock startup)
    if len(pts) < 2:
        return False
    if track_active and moving_toward_batsman(pts, height, width=width):
        return False
    if moving_toward_batsman(pts, height, width=width):
        return False
    return True


def reject_ball_in_flight(cy: int, height: int, post_contact: bool = False, width: int = 0) -> bool:
    """Legacy: reject upper-frame ball. Disabled when tracking full delivery flight."""
    if TRACK_DELIVERY_IN_FLIGHT and not post_contact:
        return False
    if post_contact:
        y_min = float(_PROC.get("ball_approach_y_min_after_contact", 0.28))
    elif width > 0:
        y_min = effective_approach_y_min(width, height)
    else:
        y_min = BALL_APPROACH_Y_MIN
    return cy < int(height * y_min)


def in_batsman_approach_zone(cy: int, height: int, y_min_ratio: float | None = None, width: int = 0) -> bool:
    """True when the ball is in the lower approach band near the batsman (not high in flight)."""
    if y_min_ratio is not None:
        threshold = y_min_ratio
    elif width > 0:
        threshold = effective_approach_y_min(width, height)
    else:
        threshold = BALL_APPROACH_Y_MIN
    return cy >= int(height * threshold)


def in_bounce_ground_zone(cy: int, height: int, width: int = 0) -> bool:
    """True when the ball is low enough in frame to be on/near the pitch (not mid-air approach)."""
    y_min = effective_bounce_ground_y_min(width, height) if width > 0 else BOUNCE_GROUND_Y_MIN
    return cy >= int(height * y_min)


def allow_ball_detection(
    cy: int,
    height: int,
    *,
    post_contact: bool = False,
    width: int = 0,
    track_active: bool = False,
) -> bool:
    """Kalman coast — allow full flight path during active delivery tracking."""
    if TRACK_DELIVERY_IN_FLIGHT and track_active and not post_contact:
        return True
    return not reject_ball_in_flight(cy, height, post_contact=post_contact, width=width)


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
    det_conf: float = 0.0,
    track_active: bool = False,
    recent_points: list | None = None,
    h_matrix=None,
) -> bool:
    on_pitch = width > 0 and in_pitch_detection_zone(cx, cy, width, height, h_matrix)

    if DETECT_PITCH_AREA_ONLY and width > 0 and not on_pitch:
        return False

    if reject_resting_ball_on_ground(
        cy, height, width, track_active=track_active, recent_points=recent_points,
    ):
        return False
    if reject_ball_in_flight(cy, height, post_contact=post_contact, width=width):
        return False
    if reject_hardware_false_positive(cx, cy, roi, height, width):
        return False
    if width > 0 and reject_machine_zone(cx, cy, width, height):
        return False
    need_color = REQUIRE_BALL_COLOR_FOR_LOCK and not track_active and not on_pitch
    if need_color:
        if not _roi_looks_like_cricket_ball(roi):
            return False
    else:
        ground_band = width > 0 and in_ground_resting_band(cy, height, width)
        skip_color = det_conf >= BALL_COLOR_SKIP_CONF and not ground_band
        if skip_color and roi is not None and len(roi.shape) == 3:
            if _looks_like_dark_hardware(roi) or looks_like_metallic_hardware(roi):
                skip_color = False
            elif not _roi_looks_like_cricket_ball(roi):
                skip_color = False
        if skip_color:
            if cy < height * 0.20 and reject_bird_false_positive(cx, cy, roi, height, frame=frame, width=width):
                return False
            return True
        if on_pitch:
            if _looks_like_dark_hardware(roi) or looks_like_metallic_hardware(roi):
                return False
        elif not _roi_looks_like_cricket_ball(roi):
            return False

    if reject_bird_false_positive(cx, cy, roi, height, frame=frame, width=width):
        return False
    return True
