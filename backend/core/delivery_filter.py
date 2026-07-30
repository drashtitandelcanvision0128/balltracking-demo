"""
Filter false ball detections — light check for real bounces/hits,
stricter check only for fallback 'close delivery' markers.
"""

from __future__ import annotations

import math

from core.config import CONFIG

_FILT = CONFIG.get("delivery_filter", {})

MIN_TRACK_FRAMES = int(_FILT.get("min_track_frames", 10))
MIN_TRACK_FRAMES_STRICT = int(_FILT.get("min_track_frames_strict", 12))
MIN_Y_TRAVEL_RATIO = float(_FILT.get("min_y_travel_ratio", 0.15))
MIN_DELIVERY_GAP_SEC = float(_FILT.get("min_delivery_gap_sec", 0.8))
MIN_NEW_DET_CONF = float(_FILT.get("min_new_detection_conf", 0.30))
ABSURD_SPEED_KMH = float(_FILT.get("absurd_speed_kmh", 350))
MIN_FRAMES_BETWEEN_MARKERS = int(_FILT.get("min_frames_between_markers", 18))
MIN_LOCK_FRAMES = int(_FILT.get("min_lock_frames", 3))
MIN_LOCK_TRAVEL_PX = float(_FILT.get("min_lock_travel_px", 12))
MIN_LOCK_DY_PX = float(_FILT.get("min_lock_dy_px", 8))
STATIC_REJECT_PX = float(_FILT.get("static_reject_px", 6))
STATIC_TRACK_FRAMES = int(_FILT.get("static_track_frames", 4))


def min_gap_frames(fps: float) -> int:
    return max(MIN_FRAMES_BETWEEN_MARKERS, int(fps * MIN_DELIVERY_GAP_SEC))


def compute_track_speed_kmh(raw_pts: list, fps: float, height: int) -> float:
    if len(raw_pts) < 2 or fps <= 0:
        return 0.0
    speeds = []
    for i in range(1, len(raw_pts)):
        d = math.hypot(raw_pts[i][0] - raw_pts[i - 1][0], raw_pts[i][1] - raw_pts[i - 1][1])
        speeds.append(d * fps)
    peak_px_s = max(speeds)
    pitch_span_px = max(height * 0.42, 80.0)
    meters_per_px = 18.0 / pitch_span_px
    return peak_px_s * meters_per_px * 3.6


def has_min_motion(raw_pts: list, height: int, min_frames: int | None = None, width: int = 0) -> bool:
    """Basic motion check — enough frames and visible travel."""
    from core.ball_detection_filters import is_landscape_frame

    need = min_frames or MIN_TRACK_FRAMES
    if len(raw_pts) < need:
        return False
    ys = [p[1] for p in raw_pts]
    xs = [p[0] for p in raw_pts]
    y_travel = max(ys) - min(ys)
    x_travel = max(xs) - min(xs)
    if width > 0 and is_landscape_frame(width, height):
        if x_travel >= height * MIN_Y_TRAVEL_RATIO * 0.55:
            return True
    if y_travel < height * MIN_Y_TRAVEL_RATIO:
        return False
    return True


def is_absurd_speed(raw_pts: list, fps: float, height: int) -> bool:
    """Pixel-speed glitches (false tracks) often show 500+ km/h."""
    speed = compute_track_speed_kmh(raw_pts, fps, height)
    return speed > ABSURD_SPEED_KMH


def is_valid_delivery_track(
    raw_pts: list,
    height: int,
    fps: float,
    *,
    strict: bool = False,
    width: int = 0,
) -> bool:
    """
    strict=False: real bounce/hit markers (lenient).
    strict=True:  fallback close-delivery only (reject obvious noise).
    """
    need = MIN_TRACK_FRAMES_STRICT if strict else MIN_TRACK_FRAMES
    if not has_min_motion(raw_pts, height, need, width=width):
        return False
    # Speed glitch filter only for fallback close-delivery (pixel speed is unreliable)
    if strict and is_absurd_speed(raw_pts, fps, height):
        return False
    if strict:
        ys = [p[1] for p in raw_pts]
        xs = [p[0] for p in raw_pts]
        x_spread = max(xs) - min(xs)
        y_travel = max(ys) - min(ys)
        if x_spread > height * 0.85 and y_travel < height * 0.12:
            return False
    return True


def pending_delivery_confirmed(points: list, height: int, width: int = 0) -> bool:
    """Require visible motion before locking — rejects nuts/bolts and static false positives."""
    from core.ball_detection_filters import is_landscape_frame, in_ground_resting_band, in_machine_release_zone

    if len(points) < MIN_LOCK_FRAMES:
        return False
    recent = points[-MIN_LOCK_FRAMES:]
    xs = [p[0] for p in recent]
    ys = [p[1] for p in recent]
    path_len = sum(
        math.hypot(recent[i][0] - recent[i - 1][0], recent[i][1] - recent[i - 1][1])
        for i in range(1, len(recent))
    )
    net_span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    min_travel = max(MIN_LOCK_TRAVEL_PX, height * 0.006)
    if width > 0 and all(in_ground_resting_band(p[1], height, width) for p in recent):
        min_travel = max(min_travel * 2.5, height * 0.018)
    elif width > 0 and in_machine_release_zone(recent[0][0], recent[0][1], width, height):
        min_travel = max(MIN_LOCK_TRAVEL_PX * 0.55, height * 0.003)
    if path_len < min_travel or net_span < min_travel * 0.65:
        return False
    if net_span < STATIC_REJECT_PX:
        return False
    dx = abs(recent[-1][0] - recent[0][0])
    dy = recent[-1][1] - recent[0][1]
    min_dy = max(MIN_LOCK_DY_PX, height * 0.004)
    # Side-on landscape: ball travels horizontally across frame toward batsman
    if width > 0 and is_landscape_frame(width, height):
        from_machine = in_machine_release_zone(recent[0][0], recent[0][1], width, height)
        if from_machine:
            if path_len >= min_travel or abs(dy) >= min_dy * 0.6 or dx >= min_travel * 0.35:
                return True
        if dx >= min_travel * 0.45 or dy >= min_dy:
            return True
        return net_span >= min_travel
    if dy < min_dy:
        return False
    return True


def track_is_static(points: list, min_points: int | None = None) -> bool:
    """True when recent track points barely move (stationary object)."""
    need = min_points or STATIC_TRACK_FRAMES
    if len(points) < need:
        return False
    recent = points[-need:]
    xs = [p[0] for p in recent]
    ys = [p[1] for p in recent]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys)) < STATIC_REJECT_PX


def can_start_new_delivery(
    frame_index: int,
    last_marker_frame: int,
    gap_frames: int,
    fps: float,
    detection_conf: float,
    from_waiting: bool,
) -> bool:
    if frame_index - last_marker_frame < 5:
        return False
    if from_waiting and detection_conf < MIN_NEW_DET_CONF:
        return False
    return True


def should_register_marker(
    raw_pts: list,
    height: int,
    fps: float,
    frame_index: int,
    last_marker_frame: int,
    *,
    strict: bool = False,
) -> bool:
    """Register bounce/hit marker if cooldown passed and track has real motion."""
    if frame_index - last_marker_frame < 5:
        return False
    return is_valid_delivery_track(raw_pts, height, fps, strict=strict)
