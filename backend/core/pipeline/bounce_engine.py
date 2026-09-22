"""
Multi-feature bounce (pitch point) prediction.

Fuses:
  1. Vertical direction change (dy flip)
  2. Parabolic trajectory apex
  3. World-coordinate homography bounce
  4. Ground-plane snap
  5. Velocity / acceleration at inflection

Returns None when features disagree — never guesses.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from core.bounce_predictor import BallPoint, detect_bounce, smooth_positions, track_history
from core.homography import snap_to_pitch_ground
from core.pitch_coords import pitchmap_to_world, video_to_pitchmap
from core.pipeline.types import BounceResult
from core.trajectory_physics import fit_parabolic_y, refine_bounce_world


def _velocity_profile(history: Sequence[BallPoint]) -> tuple[list[float], list[float]]:
    vx, vy, ax, ay = [], [], [], []
    for i in range(1, len(history)):
        dt = max(1, history[i].frame_id - history[i - 1].frame_id)
        dx = (history[i].x - history[i - 1].x) / dt
        dy = (history[i].y - history[i - 1].y) / dt
        vx.append(dx)
        vy.append(dy)
    for i in range(1, len(vy)):
        dt = max(1, history[i + 1].frame_id - history[i].frame_id)
        ax.append((vy[i] - vy[i - 1]) / dt)
        ay.append((vx[i] - vx[i - 1]) / dt)
    return vy, ax


def _parabolic_bounce_index(history: Sequence[BallPoint]) -> int | None:
    """Local Y apex with post-bounce rise — avoids picking noise spikes."""
    if len(history) < 7:
        return None
    xs = [p.x for p in history]
    ys = [p.y for p in history]
    fit = fit_parabolic_y(xs, ys)
    if fit is None:
        return None
    best_i, best_y = None, -1.0
    for i in range(2, len(history) - 2):
        y = history[i].y
        if y <= history[i - 1].y or y <= history[i + 1].y:
            continue
        rise = y - min(history[j].y for j in range(i + 1, min(i + 4, len(history))))
        if rise < 4.0:
            continue
        if y > best_y:
            best_y = y
            best_i = i
    return best_i


def _accel_inflection_index(vy: list[float], ax: list[float]) -> int | None:
    """Index where vertical velocity crosses zero with positive jerk upward."""
    if len(vy) < 4 or len(ax) < 2:
        return None
    for i in range(2, len(vy) - 1):
        if vy[i - 1] > 1.0 and vy[i] < -0.5 and ax[i - 1] > 0:
            return i
    return None


def confidence_estimation(
    history: Sequence[BallPoint],
    bounce_idx: int,
    *,
    vy: list[float] | None = None,
    votes: int = 1,
    max_votes: int = 4,
) -> float:
    """Combine motion quality with multi-feature agreement."""
    if bounce_idx < 1 or bounce_idx >= len(history):
        return 0.0

    from core.bounce_predictor import confidence_estimation as base_conf

    base = base_conf(history, bounce_idx)
    vote_score = votes / max(max_votes, 1)

    vel_score = 0.5
    if vy and 0 < bounce_idx < len(vy):
        pre = vy[bounce_idx - 1] if bounce_idx - 1 < len(vy) else 0.0
        post = vy[bounce_idx] if bounce_idx < len(vy) else 0.0
        if pre > 0.8 and post < -0.4:
            vel_score = min(1.0, (pre + abs(post)) / 6.0)

    return round(min(1.0, 0.50 * base + 0.30 * vote_score + 0.20 * vel_score), 3)


def predict_bounce(
    points: Sequence[tuple[int, float, float]],
    *,
    h_matrix: np.ndarray | None = None,
    cam_quad: np.ndarray | None = None,
    height: int = 1080,
    fps: float = 30.0,
    min_confidence: float = 0.55,
) -> BounceResult | None:
    """
    Fuse multiple bounce cues into one pitch-point estimate.

    Returns structured output with image + world coordinates, or None.
    """
    if len(points) < 10:
        return None

    history = track_history(points, maxlen=60)
    smooth = smooth_positions(history, window=3)
    pixel_path = [(int(p.x), int(p.y)) for p in smooth]

    candidates: dict[int, dict] = {}

    dir_pred = detect_bounce(smooth, min_confidence=0.40)
    if dir_pred is not None:
        candidates[dir_pred.history_index] = {
            "frame": dir_pred.bounce_frame,
            "x": dir_pred.bounce_x,
            "y": dir_pred.bounce_y,
            "method": "direction",
            "votes": 1,
        }

    para_i = _parabolic_bounce_index(smooth)
    if para_i is not None:
        pt = smooth[para_i]
        entry = candidates.get(para_i, {"votes": 0})
        entry.update({
            "frame": pt.frame_id, "x": pt.x, "y": pt.y,
            "method": entry.get("method", "parabola"),
        })
        entry["votes"] = entry.get("votes", 0) + 1
        candidates[para_i] = entry

    vy, ax = _velocity_profile(smooth)
    acc_i = _accel_inflection_index(vy, ax)
    if acc_i is not None and acc_i < len(smooth):
        pt = smooth[acc_i]
        entry = candidates.get(acc_i, {"votes": 0})
        entry.update({
            "frame": pt.frame_id, "x": pt.x, "y": pt.y,
            "method": entry.get("method", "acceleration"),
        })
        entry["votes"] = entry.get("votes", 0) + 1
        candidates[acc_i] = entry

    if h_matrix is not None:
        world_pt = refine_bounce_world(pixel_path, h_matrix, height)
        if world_pt is not None:
            wx, wy = world_pt
            best_i = min(
                range(len(smooth)),
                key=lambda i: math.hypot(smooth[i].x - wx, smooth[i].y - wy),
            )
            pt = smooth[best_i]
            entry = candidates.get(best_i, {"votes": 0})
            entry.update({
                "frame": pt.frame_id, "x": pt.x, "y": pt.y,
                "method": entry.get("method", "homography"),
            })
            entry["votes"] = entry.get("votes", 0) + 1
            candidates[best_i] = entry

    if not candidates:
        return None

    best_idx = max(candidates, key=lambda i: candidates[i]["votes"])
    cand = candidates[best_idx]
    bx, by = float(cand["x"]), float(cand["y"])

    if cam_quad is not None:
        snapped = snap_to_pitch_ground(int(bx), int(by), cam_quad, height)
        if snapped is not None:
            bx, by = snapped

    conf = confidence_estimation(
        smooth, best_idx, vy=vy,
        votes=int(cand["votes"]), max_votes=4,
    )
    if conf < min_confidence:
        return None

    pitch_x_m, pitch_y_m = 0.0, 0.0
    pitch_map_x, pitch_map_y = 0, 0
    if h_matrix is not None:
        try:
            pitch_map_x, pitch_map_y = video_to_pitchmap(bx, by, h_matrix)
            pitch_x_m, pitch_y_m = pitchmap_to_world(pitch_map_x, pitch_map_y)
        except Exception:
            pass

    methods = cand.get("method", "fusion")
    if cand["votes"] > 1:
        methods = f"fusion({cand['votes']})"

    return BounceResult(
        bounce_frame=int(cand["frame"]),
        image_x=bx,
        image_y=by,
        pitch_x_m=pitch_x_m,
        pitch_y_m=pitch_y_m,
        pitch_map_x=pitch_map_x,
        pitch_map_y=pitch_map_y,
        confidence=conf,
        method=methods,
        history_index=best_idx,
    )
