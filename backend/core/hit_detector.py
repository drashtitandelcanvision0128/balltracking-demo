"""
Enhanced hit detection — heuristic scoring + pose/bat proximity.
"""

from __future__ import annotations

import math
from typing import Any

from accuracy_engine import score_hit as _heuristic_score_hit
from accuracy_engine import score_hit_post_bounce as _heuristic_post_bounce


def batsman_zone_from_pose(
    pose_frames: list[dict[str, Any]] | None,
    height: int,
    default: tuple[float, float] = (0.22, 0.92),
) -> tuple[float, float]:
    """Dynamic batting zone Y bounds from pose samples."""
    if not pose_frames:
        return default
    ys = []
    for pf in pose_frames:
        head = pf.get("head_position")
        feet = pf.get("front_foot") or pf.get("back_foot")
        if head:
            ys.append(head[1] / height)
        if feet:
            ys.append(feet[1] / height)
    if len(ys) < 2:
        return default
    y_min = max(0.12, min(ys) - 0.08)
    y_max = min(0.96, max(ys) + 0.06)
    return y_min, y_max


def _bat_proximity_bonus(
    contact: tuple[int, int] | None,
    pose_frames: list[dict[str, Any]] | None,
    height: int,
) -> float:
    """Bonus confidence when ball contact is near estimated bat zone."""
    if contact is None or not pose_frames:
        return 0.0
    cx, cy = contact
    best = 0.0
    for pf in pose_frames:
        bat = pf.get("bat_zone")
        if not bat:
            continue
        dist = math.hypot(cx - bat[0], cy - bat[1])
        norm = height * 0.12
        if dist < norm:
            best = max(best, 0.25 * (1.0 - dist / norm))
    return best


def score_hit_enhanced(
    raw_pts: list,
    hist_pts: list,
    height: int,
    fps: float,
    bounced: bool,
    frames_since_bounce: int,
    *,
    bounce_hist_idx: int | None = None,
    pose_frames: list[dict[str, Any]] | None = None,
) -> tuple[bool, float, tuple[int, int] | None]:
    """
    Heuristic hit score + pose bat-zone proximity + dynamic batsman zone filter.
    """
    y_min, y_max = batsman_zone_from_pose(pose_frames, height)

    if bounced and bounce_hist_idx is not None and frames_since_bounce >= 2:
        is_hit, conf, contact = _heuristic_post_bounce(hist_pts, height, fps, bounce_hist_idx)
    else:
        is_hit, conf, contact = _heuristic_score_hit(
            raw_pts, hist_pts, height, fps, bounced, frames_since_bounce,
            bounce_hist_idx=bounce_hist_idx,
        )

    if contact is not None:
        cy_norm = contact[1] / height
        if cy_norm < y_min or cy_norm > y_max:
            return False, 0.0, None

    pose_bonus = _bat_proximity_bonus(contact, pose_frames, height)
    conf = min(1.0, conf + pose_bonus)

    # Pose proximity can confirm marginal hits
    threshold = 0.34 if pose_bonus > 0.1 else 0.36
    is_hit = conf >= threshold
    return is_hit, conf, contact if is_hit else None
