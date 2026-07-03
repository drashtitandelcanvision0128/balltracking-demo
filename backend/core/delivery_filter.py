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


def min_gap_frames(fps: float) -> int:
    return max(MIN_FRAMES_BETWEEN_MARKERS, int(fps * MIN_DELIVERY_GAP_SEC))


MIN_BALL_SPEED_KMH = 32.0
_FEET_ZONE_Y_RATIO = 0.58          # bottom ~42% — batsman feet / pads only
_CREASE_Y_RATIO = 0.52             # wider guard for starting junk tracks
_CREASE_MIN_PITCH_Y_M = 3.0
_FEET_MIN_PITCH_Y_M = 2.0          # stumps line only
_FAST_BALL_PEAK_KMH = 50.0        # real delivery always exceeds this somewhere


def _track_in_feet_zone(vx, vy, height, h_matrix=None) -> bool:
    """Strict feet/stumps band — used to block markers, not whole deliveries."""
    if vy > height * _FEET_ZONE_Y_RATIO:
        return True
    if h_matrix is not None:
        try:
            from core.pitch_coords import video_to_world
            _, y_m = video_to_world(float(vx), float(vy), h_matrix)
            if y_m < _FEET_MIN_PITCH_Y_M:
                return True
        except Exception:
            pass
    return False


def _track_in_crease(vx, vy, height, h_matrix=None) -> bool:
    if vy > height * _CREASE_Y_RATIO:
        return True
    if h_matrix is not None:
        try:
            from core.pitch_coords import video_to_world
            _, y_m = video_to_world(float(vx), float(vy), h_matrix)
            if y_m < _CREASE_MIN_PITCH_Y_M:
                return True
        except Exception:
            pass
    return False


def track_peak_speed_kmh(raw_pts: list, fps: float, height: int) -> float:
    return compute_track_speed_kmh(raw_pts, fps, height)


def is_feet_false_track(
    raw_pts: list,
    height: int,
    fps: float,
    h_matrix=None,
) -> bool:
    """
    Feet / pad shuffle: slow, horizontal, stuck in feet zone.
    Real balls are NOT rejected — they always have a fast segment.
    """
    if len(raw_pts) < 6:
        return False
    peak = track_peak_speed_kmh(raw_pts, fps, height)
    if peak >= _FAST_BALL_PEAK_KMH:
        return False

    recent = raw_pts[-min(10, len(raw_pts)):]
    speeds = []
    for i in range(1, len(recent)):
        d = math.hypot(recent[i][0] - recent[i - 1][0], recent[i][1] - recent[i - 1][1])
        speeds.append(d * fps * 3.6 * (18.0 / max(height * 0.42, 80.0)))
    avg_kmh = sum(speeds) / len(speeds) if speeds else 0.0

    feet_n = sum(1 for x, y in recent if _track_in_feet_zone(x, y, height, h_matrix))
    xs = [p[0] for p in recent]
    ys = [p[1] for p in recent]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    avg_y = sum(ys) / len(ys)

    if feet_n >= len(recent) * 0.6 and avg_kmh < 38:
        return True
    if avg_y > height * _FEET_ZONE_Y_RATIO and peak < 42 and avg_kmh < 42:
        return True
    if x_span > height * 0.05 and y_span < height * 0.03 and peak < 45:
        return True
    return False


def is_batsman_shuffle_track(
    raw_pts: list,
    height: int,
    fps: float,
    h_matrix=None,
) -> bool:
    return is_feet_false_track(raw_pts, height, fps, h_matrix)


def is_plausible_ball_track(
    raw_pts: list,
    height: int,
    fps: float,
    h_matrix=None,
) -> bool:
    """Accept real deliveries; reject only obvious feet/shuffle tracks."""
    if len(raw_pts) < 5:
        return False
    if is_feet_false_track(raw_pts, height, fps, h_matrix):
        return False
    peak = track_peak_speed_kmh(raw_pts, fps, height)
    if peak < MIN_BALL_SPEED_KMH:
        return False
    ys = [p[1] for p in raw_pts]
    if max(ys) - min(ys) < height * 0.05:
        return False
    return True


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


def has_min_motion(raw_pts: list, height: int, min_frames: int | None = None) -> bool:
    """Basic motion check — enough frames and visible travel."""
    need = min_frames or MIN_TRACK_FRAMES
    if len(raw_pts) < need:
        return False
    ys = [p[1] for p in raw_pts]
    if max(ys) - min(ys) < height * MIN_Y_TRAVEL_RATIO:
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
) -> bool:
    """
    strict=False: real bounce/hit markers (lenient).
    strict=True:  fallback close-delivery only (reject obvious noise).
    """
    need = MIN_TRACK_FRAMES_STRICT if strict else MIN_TRACK_FRAMES
    if not has_min_motion(raw_pts, height, need):
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
    h_matrix=None,
) -> bool:
    """Register bounce/hit marker if cooldown passed and track looks like a real ball."""
    if frame_index - last_marker_frame < 5:
        return False
    if is_feet_false_track(raw_pts, height, fps, h_matrix):
        return False
    if not is_valid_delivery_track(raw_pts, height, fps, strict=strict):
        return False
    return is_plausible_ball_track(raw_pts, height, fps, h_matrix)
