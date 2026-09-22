"""
First pitch mark = where the ball actually bounces on the cricket pitch.

Behind-batsman cameras: image Y keeps growing as the ball comes toward the
lens, so max-Y is the batsman/stumps — not the bounce. Bounce is the first
dip of the ball onto the stump-to-stump corridor (Y residual peak).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.ball_detection_filters import bounce_on_pitch, is_landscape_frame
from core.homography import point_in_calib_quad


@dataclass(frozen=True)
class PitchPointMarker:
    frame: int
    x: float
    y: float
    delivery_id: int


def _smooth(values: list[float], window: int = 3) -> np.ndarray:
    if len(values) < 2:
        return np.array(values, dtype=np.float64)
    w = max(1, window)
    k = np.ones(w) / w
    return np.convolve(np.array(values, dtype=np.float64), k, mode="same")


def _parse_det_track(points: list) -> list[tuple[int, float, float, float]]:
    seg: list[tuple[int, float, float, float]] = []
    for p in points:
        if len(p) >= 4 and float(p[3]) > 0:
            seg.append((int(p[0]), float(p[1]), float(p[2]), float(p[3])))
    return seg


def _on_pitch_corridor(
    x: float,
    y: float,
    *,
    width: int,
    height: int,
    cam_quad=None,
    h_matrix=None,
) -> bool:
    """True only on the stump-to-stump pitch, not batsman/bowler bodies."""
    if width > 0 and height > 0 and not is_landscape_frame(width, height):
        # Portrait nets: bounce lives on the centre mat, mid-pitch.
        if not (width * 0.38 <= x <= width * 0.62):
            return False
        if not (height * 0.36 <= y <= height * 0.74):
            return False
    if cam_quad is not None:
        return point_in_calib_quad(x, y, cam_quad, margin_px=36.0)
    if width > 0:
        return bounce_on_pitch(int(x), int(y), width, height, h_matrix)
    return True


def _residual_bounce(
    seg: list[tuple[int, float, float, float]],
    height: int,
    fps: float,
) -> tuple[int, float, float, int] | None:
    n = len(seg)
    if n < 6:
        return None
    ts = np.array([p[0] for p in seg], dtype=np.float64)
    ys = np.array([p[2] for p in seg], dtype=np.float64)
    if float(ys.max() - ys.min()) < max(12.0, height * 0.012):
        return None
    t0 = ts - ts[0]
    deg = 2 if n >= 10 else 1
    try:
        resid = _smooth((ys - np.polyval(np.polyfit(t0, ys, deg=deg), t0)).tolist(), window=5)
    except np.linalg.LinAlgError:
        return None
    skip = min(max(2, int(fps * 0.08), int(n * 0.10)), n - 4)
    min_prom = max(3.5, height * 0.004)
    for i in range(skip, n - 2):
        if not (resid[i] >= resid[i - 1] and resid[i] >= resid[i + 1]):
            continue
        before = resid[max(0, i - 6): i]
        after = resid[i + 1: min(i + 7, n)]
        if len(before) == 0 or len(after) == 0:
            continue
        if (resid[i] - float(np.min(before))) < min_prom:
            continue
        if (resid[i] - float(np.min(after))) < min_prom * 0.55:
            continue
        f, x, y, _a = seg[i]
        # Keep the ball's own x — do not snap onto a player.
        return int(f), float(x), float(y), int(seg[0][0])
    # First local raw-Y peak on this corridor track (side-on / true reversal).
    raw = ys.tolist()
    sm = _smooth(raw)
    for i in range(skip, n - 2):
        if sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1] and raw[i] >= raw[i + 1] - 1.0:
            if sm[i] - sm[0] < max(8.0, height * 0.008):
                continue
            f, x, y, _a = seg[i]
            return int(f), float(x), float(y), int(seg[0][0])
    return None


def find_lowest_y_on_pitch(
    points: list,
    release_frame: int,
    *,
    height: int = 1080,
    width: int = 0,
    fps: float = 30.0,
    cam_quad=None,
    h_matrix=None,
    min_frames_after_release: int = 6,
) -> tuple[int, float, float, int] | None:
    """
    Mark the first bounce on the cricket pitch.

    Uses bbox-bottom Y. Only detections on the stump corridor count — gloves,
    pads, and off-mat points cannot become the mark.
    """
    seg = _parse_det_track([p for p in points if p[0] >= release_frame])
    min_after = max(min_frames_after_release, int(fps * 0.10))
    if len(seg) < min_after + 3:
        return None

    on_pitch = [
        p for p in seg
        if _on_pitch_corridor(
            p[1], p[2], width=width, height=height,
            cam_quad=cam_quad, h_matrix=h_matrix,
        )
    ]
    if len(on_pitch) < 5:
        return None

    # Static lock on a player/stump is not a delivery bounce.
    xs = [p[1] for p in on_pitch]
    ys = [p[2] for p in on_pitch]
    if (max(xs) - min(xs)) + (max(ys) - min(ys)) < max(28.0, height * 0.03):
        return None

    return _residual_bounce(on_pitch, height, fps)


def find_bounce_reversal_point(
    points: list,
    release_frame: int,
    *,
    height: int = 1080,
    width: int = 0,
    fps: float = 30.0,
    min_frames_after_release: int = 6,
    cam_quad=None,
    h_matrix=None,
) -> tuple[int, float, float, int] | None:
    return find_lowest_y_on_pitch(
        points, release_frame,
        height=height, width=width, fps=fps,
        cam_quad=cam_quad, h_matrix=h_matrix,
        min_frames_after_release=min_frames_after_release,
    )


def find_pitch_ground_touch(
    points: list,
    release_frame: int,
    *,
    cam_quad=None,
    width: int = 0,
    height: int = 1080,
    ground_y_ratio: float = 0.72,
    min_frames_after_release: int = 6,
    fps: float = 30.0,
    h_matrix=None,
) -> tuple[int, float, float] | None:
    result = find_lowest_y_on_pitch(
        points, release_frame,
        height=height, width=width, fps=fps,
        cam_quad=cam_quad, h_matrix=h_matrix,
        min_frames_after_release=min_frames_after_release,
    )
    if result is None:
        return None
    f, x, y, _ = result
    return f, x, y


def draw_pitch_point_markers(frame, markers: list[PitchPointMarker], *, radius: int = 9) -> None:
    import cv2

    h, w = frame.shape[:2]
    for m in markers:
        cx, cy = int(m.x), int(m.y)
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        cv2.circle(frame, (cx, cy), radius, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 1, cv2.LINE_AA)
