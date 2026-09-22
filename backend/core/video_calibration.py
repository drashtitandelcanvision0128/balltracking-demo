"""
Per-video zones derived from pitch homography quad (manual or auto calib).

Sets machine release + batsman approach from the actual pitch corners in frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.ball_detection_filters import is_landscape_frame


def derive_zones_from_quad(
    quad: np.ndarray | list,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Build machine / batsman / pitch ratios from calibrated 4-corner pitch quad."""
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    xs = q[:, 0]
    ys = q[:, 1]
    left_x = float(min(xs))
    right_x = float(max(xs))
    top_y = float(min(ys))
    bot_y = float(max(ys))
    pitch_w = max(right_x - left_x, width * 0.08)
    pitch_h = max(bot_y - top_y, height * 0.08)

    landscape = is_landscape_frame(width, height)
    if landscape:
        # Side-on: machine right, batsman left (pitch quad TR/BR = bowler end)
        machine = {
            "enabled": True,
            "x_min_ratio": round(max(0.40, (right_x - pitch_w * 0.05) / width), 4),
            "x_max_ratio": round(min(0.98, (right_x + pitch_w * 0.55) / width), 4),
            "y_min_ratio": round(max(0.08, (top_y - pitch_h * 0.12) / height), 4),
            "y_max_ratio": round(min(0.96, (bot_y + pitch_h * 0.10) / height), 4),
        }
        batsman = {
            "x_min_ratio": round(max(0.0, (left_x - pitch_w * 0.35) / width), 4),
            "x_max_ratio": round(min(0.98, (left_x + pitch_w * 0.25) / width), 4),
            "y_min_ratio": round(max(0.06, (top_y - pitch_h * 0.15) / height), 4),
            "y_max_ratio": round(min(0.96, (bot_y + pitch_h * 0.12) / height), 4),
        }
    else:
        # Portrait/rear: bowler top, batsman bottom
        machine = {
            "enabled": True,
            "x_min_ratio": round(max(0.10, (left_x - pitch_w * 0.08) / width), 4),
            "x_max_ratio": round(min(0.92, (right_x + pitch_w * 0.08) / width), 4),
            "y_min_ratio": round(max(0.04, (top_y - pitch_h * 0.20) / height), 4),
            "y_max_ratio": round(min(0.55, (top_y + pitch_h * 0.35) / height), 4),
        }
        batsman = {
            "x_min_ratio": round(max(0.08, (left_x - pitch_w * 0.06) / width), 4),
            "x_max_ratio": round(min(0.92, (right_x + pitch_w * 0.06) / width), 4),
            "y_min_ratio": round(max(0.35, (bot_y - pitch_h * 0.35) / height), 4),
            "y_max_ratio": round(min(0.98, (bot_y + pitch_h * 0.08) / height), 4),
        }

    return {
        "landscape": landscape,
        "machine_release_zone": machine,
        "batsman_end": batsman,
        "pitch_quad": q.tolist(),
    }


def apply_video_zones(zones: dict[str, Any] | None) -> dict[str, Any] | None:
    """Patch runtime filter ratios from per-video calibration."""
    if not zones:
        return None
    import core.ball_detection_filters as bdf

    mach = zones.get("machine_release_zone") or {}
    if mach.get("enabled", True):
        bdf.MACHINE_RELEASE_ENABLED = True
        bdf.MACHINE_RELEASE_X_MIN = float(mach.get("x_min_ratio", bdf.MACHINE_RELEASE_X_MIN))
        bdf.MACHINE_RELEASE_X_MAX = float(mach.get("x_max_ratio", bdf.MACHINE_RELEASE_X_MAX))
        bdf.MACHINE_RELEASE_Y_MIN = float(mach.get("y_min_ratio", bdf.MACHINE_RELEASE_Y_MIN))
        bdf.MACHINE_RELEASE_Y_MAX = float(mach.get("y_max_ratio", bdf.MACHINE_RELEASE_Y_MAX))

    bat = zones.get("batsman_end")
    if bat:
        pass  # used via calibrate_flight_corridor(zones=...)

    return zones


def player_ground_point(bbox: tuple | list) -> tuple[float, float]:
    """Feet / crease contact: midpoint of the person box bottom edge."""
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return ((x1 + x2) * 0.5, y2)


def quad_from_player_ends(
    width: int,
    height: int,
    *,
    batsman_bbox: tuple | list | None = None,
    bowler_bbox: tuple | list | None = None,
    batsman_xy: tuple[float, float] | None = None,
    bowler_xy: tuple[float, float] | None = None,
) -> np.ndarray | None:
    """
    Pitch quad from batsman stance (near crease) to bowler / machine end.
    Order matches auto_pitch_quad: TL, TR, BL, BR. Never drawn on video.
    """
    if batsman_bbox is not None:
        batsman_xy = player_ground_point(batsman_bbox)
    if bowler_bbox is not None:
        bowler_xy = player_ground_point(bowler_bbox)

    landscape = is_landscape_frame(width, height)
    if batsman_xy is None and bowler_xy is None:
        return None

    if batsman_xy is None and bowler_xy is not None:
        bx, by = bowler_xy
        if landscape:
            batsman_xy = (max(width * 0.08, bx - width * 0.42), min(height * 0.94, by + height * 0.36))
        else:
            batsman_xy = (bx, min(height * 0.96, by + height * 0.58))
    if bowler_xy is None and batsman_xy is not None:
        ax, ay = batsman_xy
        if landscape:
            bowler_xy = (min(width * 0.92, ax + width * 0.42), max(height * 0.16, ay - height * 0.36))
        else:
            bowler_xy = (ax, max(height * 0.14, ay - height * 0.55))

    ax, ay = float(batsman_xy[0]), float(batsman_xy[1])
    bx, by = float(bowler_xy[0]), float(bowler_xy[1])
    # Batsman stands beside the pitch. Keep the corridor on the stump-to-stump
    # centre line (bowler x → frame centre), not pulled to the batsman's side.
    if not landscape:
        near_x = width * 0.50
        far_x = float(bx)
        ax, bx = near_x, far_x
    dx, dy = bx - ax, by - ay
    length = float(np.hypot(dx, dy))
    if length < max(80.0, min(width, height) * 0.12):
        return None

    ux, uy = dx / length, dy / length
    # Extend slightly past crease (batsman) and bowling end so landings stay inside.
    ax -= ux * length * 0.06
    ay -= uy * length * 0.06
    bx += ux * length * 0.10
    by += uy * length * 0.10
    dx, dy = bx - ax, by - ay
    length = float(np.hypot(dx, dy))
    ux, uy = dx / length, dy / length
    px, py = -uy, ux

    near_w = width * 0.14
    if batsman_bbox is not None and landscape:
        near_w = max(near_w, abs(float(batsman_bbox[2]) - float(batsman_bbox[0])) * 2.3)
    near_w = float(np.clip(near_w, width * 0.08, width * 0.28))
    far_w = float(np.clip(near_w * 0.48, width * 0.04, near_w * 0.75))
    if not landscape:
        near_w = float(np.clip(width * 0.18, width * 0.10, width * 0.22))
        far_w = float(np.clip(width * 0.08, width * 0.05, width * 0.12))

    bat_l = (ax - px * near_w, ay - py * near_w)
    bat_r = (ax + px * near_w, ay + py * near_w)
    bowl_l = (bx - px * far_w, by - py * far_w)
    bowl_r = (bx + px * far_w, by + py * far_w)
    corners = np.array([bowl_l, bowl_r, bat_l, bat_r], dtype=np.float32)
    corners[:, 0] = np.clip(corners[:, 0], 2.0, width - 3.0)
    corners[:, 1] = np.clip(corners[:, 1], 2.0, height - 3.0)

    y_sorted = corners[np.argsort(corners[:, 1])]
    top, bottom = y_sorted[:2], y_sorted[2:]
    tl = top[np.argmin(top[:, 0])]
    tr = top[np.argmax(top[:, 0])]
    bl = bottom[np.argmin(bottom[:, 0])]
    br = bottom[np.argmax(bottom[:, 0])]
    return np.array([tl, tr, bl, br], dtype=np.float32)


def apply_player_pitch_calibration(
    width: int,
    height: int,
    players: Any,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """Rebuild pitch quad + zones from detected striker/bowler. Returns (quad, zones) or (None, None)."""
    if players is None:
        return None, None
    if not isinstance(players, dict):
        players = players.to_dict() if hasattr(players, "to_dict") else None
    if not players:
        return None, None
    striker = players.get("striker")
    bowler = players.get("bowler")
    bat_bb = (striker or {}).get("bbox") if striker else None
    bowl_bb = (bowler or {}).get("bbox") if bowler else None
    quad = quad_from_player_ends(
        width, height,
        batsman_bbox=bat_bb,
        bowler_bbox=bowl_bb,
    )
    if quad is None:
        return None, None
    zones = derive_zones_from_quad(quad, width, height)
    return quad, zones


def zones_for_log(zones: dict[str, Any]) -> str:
    m = zones.get("machine_release_zone", {})
    return (
        f"machine x={m.get('x_min_ratio')}-{m.get('x_max_ratio')} "
        f"y={m.get('y_min_ratio')}-{m.get('y_max_ratio')}"
    )
