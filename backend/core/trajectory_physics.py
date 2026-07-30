"""
Physics-informed trajectory helpers: parabolic fit, world-coords bounce, gap fill.
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from core.pitch_coords import pitchmap_to_world, video_to_pitchmap, video_to_world


def fit_parabolic_y(xs: list[float], ys: list[float]) -> tuple[float, float, float] | None:
    """Fit y = a*x^2 + b*x + c. Returns (a, b, c) or None."""
    if len(xs) < 4:
        return None
    try:
        coeffs = np.polyfit(xs, ys, 2)
        return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    except (np.linalg.LinAlgError, ValueError):
        return None


def interpolate_track_gaps(
    raw_pts: list[tuple[int, int]],
    fps: float,
    max_gap_frames: int = 4,
) -> list[tuple[int, int]]:
    """
    Fill short detection gaps via parabolic interpolation on recent segment.
    Only fills gaps <= max_gap_frames between known points.
    """
    if len(raw_pts) < 3 or fps <= 0:
        return raw_pts

    out = [raw_pts[0]]
    for i in range(1, len(raw_pts)):
        prev = out[-1]
        curr = raw_pts[i]
        gap = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
        step_est = max(8.0, gap)
        est_frames = max(1, int(gap / step_est))
        if est_frames > 1 and est_frames <= max_gap_frames:
            seg_xs = [p[0] for p in out[-6:]] + [curr[0]]
            seg_ys = [p[1] for p in out[-6:]] + [curr[1]]
            t = np.linspace(0, 1, est_frames + 1)[1:-1]
            for ti in t:
                ix = int(prev[0] + (curr[0] - prev[0]) * ti)
                iy = int(prev[1] + (curr[1] - prev[1]) * ti)
                if len(seg_xs) >= 4:
                    fit = fit_parabolic_y(seg_xs, seg_ys)
                    if fit is not None:
                        a, b, c = fit
                        ix_f = prev[0] + (curr[0] - prev[0]) * ti
                        iy = int(a * ix_f * ix_f + b * ix_f + c)
                out.append((ix, iy))
        out.append(curr)
    return out


def world_trajectory(
    raw_pts: Sequence[tuple[int, int]],
    h_matrix: np.ndarray,
) -> list[tuple[float, float]]:
    pts = []
    for x, y in raw_pts:
        try:
            wx, wy = video_to_world(float(x), float(y), h_matrix)
            pts.append((wx, wy))
        except Exception:
            continue
    return pts


def refine_bounce_world(
    raw_pts: list[tuple[int, int]],
    h_matrix: np.ndarray,
    height: int,
    lookback: int = 18,
) -> tuple[int, int] | None:
    """
    Bounce = local minimum in world Y (metres from stumps) after downward flight.
    More stable than pixel-Y apex across camera angles.
    """
    if len(raw_pts) < 5:
        return None

    start = max(0, len(raw_pts) - lookback)
    segment = raw_pts[start:]
    world = world_trajectory(segment, h_matrix)
    if len(world) < 5:
        return None

    best_i, best_score = None, -1e9
    for i in range(2, len(world) - 2):
        wy = [world[j][1] for j in range(i - 2, i + 3)]
        # Bounce: world Y has local minimum (ball closest to batsman / lowest pitch point)
        if not (wy[1] > wy[2] and wy[3] > wy[2] and wy[0] > wy[2]):
            continue
        depth = (wy[1] - wy[2]) + (wy[3] - wy[2])
        flatness = abs(wy[1] - wy[3])
        score = depth - flatness * 0.4
        if score > best_score:
            best_score = score
            best_i = i

    if best_i is None:
        return None
    px, py = segment[best_i]
    if py < height * 0.25:
        return None

    # Verify that the ball has actually risen (Y-coordinate decreased in pixels)
    # in the subsequent frames to avoid false bounces while in the air.
    has_risen = False
    for j in range(best_i + 1, len(segment)):
        if segment[j][1] < py - 2:
            has_risen = True
            break
    if not has_risen:
        return None

    return int(px), int(py)


def predict_pre_bounce_landing(
    raw_pts: list[tuple[int, int]],
    h_matrix: np.ndarray,
    fps: float,
) -> tuple[float, float] | None:
    """Extrapolate parabolic world trajectory to pitch plane (y_m ≈ 0)."""
    world = world_trajectory(raw_pts, h_matrix)
    if len(world) < 5 or fps <= 0:
        return None
    xs = [p[0] for p in world]
    ys = [p[1] for p in world]
    fit = fit_parabolic_y(xs, ys)
    if fit is None:
        return None
    a, b, c = fit
    if abs(a) < 1e-6:
        return None
    # Solve a*x^2 + b*x + c = 0 for positive y side (approach to stumps)
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    roots = [(-b + math.sqrt(disc)) / (2 * a), (-b - math.sqrt(disc)) / (2 * a)]
    valid = [r for r in roots if -1.6 <= r <= 1.6]
    if not valid:
        return None
    x_land = min(valid, key=abs)
    return round(x_land, 3), 0.0


def predict_bounce_pixel_parabola(
    raw_pts: list[tuple[int, int]],
    height: int,
    ground_y_ratio: float = 0.65,
) -> tuple[int, int] | None:
    """Extrapolate image-plane parabola to the pitch ground line (pixels)."""
    if len(raw_pts) < 6:
        return None
    seg = raw_pts[-min(14, len(raw_pts)) :]
    ground_y = height * ground_y_ratio
    if seg[-1][1] >= ground_y - 8:
        return None
    if seg[-1][1] <= seg[0][1] + height * 0.015:
        return None

    xs = [float(p[0]) for p in seg]
    ys = [float(p[1]) for p in seg]
    fit = fit_parabolic_y(xs, ys)
    if fit is None:
        return None
    a, b, c = fit
    if abs(a) < 1e-8:
        if abs(b) < 1e-6:
            return None
        x_land = (ground_y - c) / b
    else:
        disc = b * b - 4 * a * (c - ground_y)
        if disc < 0:
            return None
        roots = [(-b + math.sqrt(disc)) / (2 * a), (-b - math.sqrt(disc)) / (2 * a)]
        x_land = roots[0]
        if len(roots) > 1:
            mid_x = xs[-1]
            x_land = min(roots, key=lambda r: abs(r - mid_x))

    x_min, x_max = min(xs), max(xs)
    span = max(40.0, (x_max - x_min) * 2.5)
    if not (x_min - span <= x_land <= x_max + span):
        return None
    return int(round(x_land)), int(round(ground_y))


def predict_bounce_landing(
    raw_pts: list[tuple[int, int]],
    h_matrix: np.ndarray | None,
    h_inv: np.ndarray | None,
    fps: float,
    height: int,
    width: int = 0,
    ground_y_ratio: float = 0.65,
) -> dict | None:
    """
    Predict where the ball will bounce before contact.
    Returns video + pitch-map + world coords when trajectory is stable enough.
    """
    if len(raw_pts) < 6:
        return None
    seg = raw_pts[-min(16, len(raw_pts)) :]
    if seg[-1][1] <= seg[0][1] + max(6, height * 0.02):
        return None

    world_xy = None
    pitch_xy = None
    video_xy = None
    method = None

    if h_matrix is not None:
        world_xy = predict_pre_bounce_landing(seg, h_matrix, fps)
        if world_xy is not None and h_inv is not None:
            from core.pitch_coords import world_to_pitchmap

            x_m, y_m = world_xy
            if not (0.0 <= y_m <= 20.5 and abs(x_m) <= 2.2):
                world_xy = None
            else:
                px, py = world_to_pitchmap(x_m, y_m)
                pt = cv2.perspectiveTransform(
                    np.array([[[float(px), float(py)]]], dtype=np.float32), h_inv
                )
                video_xy = (int(pt[0, 0, 0]), int(pt[0, 0, 1]))
                pitch_xy = (px, py)
                method = "world_parabola"

    if video_xy is None:
        pixel_pt = predict_bounce_pixel_parabola(seg, height, ground_y_ratio=ground_y_ratio)
        if pixel_pt is None:
            return None
        video_xy = pixel_pt
        method = "pixel_parabola"
        if h_matrix is not None:
            try:
                px, py = video_to_pitchmap(float(video_xy[0]), float(video_xy[1]), h_matrix)
                pitch_xy = (px, py)
                world_xy = pitchmap_to_world(px, py)
            except Exception:
                pitch_xy = None
                world_xy = None

    if video_xy is None:
        return None
    vx, vy = video_xy
    if not (-width * 0.15 <= vx <= width * 1.15 and -height * 0.1 <= vy <= height * 1.05):
        return None

    out = {
        "video": video_xy,
        "method": method,
    }
    if pitch_xy is not None:
        out["pitchmap"] = pitch_xy
    if world_xy is not None:
        out["bounce_x"] = world_xy[0]
        out["bounce_y"] = world_xy[1]
    return out
