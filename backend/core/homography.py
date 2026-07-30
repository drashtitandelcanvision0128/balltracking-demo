"""
Pitch homography — video pixels ↔ pitch-map template.
Validates transforms and provides landscape-aware fallback quads.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.config import CONFIG
from pitch_map_renderer import PITCH_L, PITCH_R, PITCH_TOP, PITCH_BOT, TEMPLATE_CORNERS

_PITCH_CFG = CONFIG.get("pitch", {})
_LAND = _PITCH_CFG.get("landscape_quad", {})


def is_landscape_frame(width: int, height: int) -> bool:
    return width > height * 1.12


def auto_pitch_quad(width: int, height: int) -> np.ndarray:
    """
    Fallback pitch trapezoid in video pixels.
    Landscape side-on: pitch strip left-of-center (excludes machine/right clutter).
    """
    portrait = height > width * 1.15
    if portrait:
        cx = width * 0.5
        top_y, bot_y = height * 0.53, height * 0.95
        top_hw, bot_hw = width * 0.09, width * 0.29
        return np.array([
            [cx - top_hw, top_y],
            [cx + top_hw, top_y],
            [cx - bot_hw, bot_y],
            [cx + bot_hw, bot_y],
        ], dtype=np.float32)

    top_y = height * float(_LAND.get("top_y_ratio", 0.40))
    bot_y = height * float(_LAND.get("bot_y_ratio", 0.88))
    left_x = width * float(_LAND.get("left_x_ratio", 0.16))
    right_x = width * float(_LAND.get("right_x_ratio", 0.56))
    bot_spread = width * float(_LAND.get("bot_spread_ratio", 0.05))
    return np.array([
        [left_x, top_y],
        [right_x, top_y],
        [left_x - bot_spread * 0.4, bot_y],
        [right_x + bot_spread, bot_y],
    ], dtype=np.float32)


def build_homography(cam_quad: np.ndarray, template: np.ndarray | None = None):
    """Return (H_video_to_map, H_map_to_video) with sanity checks."""
    template = template if template is not None else TEMPLATE_CORNERS
    quad = np.asarray(cam_quad, dtype=np.float32).reshape(4, 2)
    if not _quad_valid(quad):
        quad = auto_pitch_quad(int(quad[:, 0].max()), int(quad[:, 1].max()))

    h_matrix = cv2.getPerspectiveTransform(quad, template.astype(np.float32))
    h_inv = cv2.getPerspectiveTransform(template.astype(np.float32), quad)

    if not homography_is_sane(h_matrix, h_inv):
        quad = auto_pitch_quad(int(quad[:, 0].max() * 1.2), int(quad[:, 1].max()))
        h_matrix = cv2.getPerspectiveTransform(quad, template.astype(np.float32))
        h_inv = cv2.getPerspectiveTransform(template.astype(np.float32), quad)

    return h_matrix, h_inv, quad


def _quad_valid(quad: np.ndarray) -> bool:
    if quad.shape != (4, 2):
        return False
    xs, ys = quad[:, 0], quad[:, 1]
    if np.any(xs < 0) or np.any(ys < 0):
        return False
    if (xs.max() - xs.min()) < 20 or (ys.max() - ys.min()) < 30:
        return False
    area = cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.float32))
    return area > 500


def homography_is_sane(h_matrix: np.ndarray, h_inv: np.ndarray) -> bool:
    """Reject degenerate homographies."""
    try:
        test_pts = np.array([[[PITCH_L, PITCH_TOP], [PITCH_R, PITCH_BOT]]], dtype=np.float32)
        back = cv2.perspectiveTransform(
            cv2.perspectiveTransform(test_pts, h_inv), h_matrix,
        )
        err = float(np.max(np.abs(back - test_pts)))
        if err > 4.0:
            return False
        det = float(np.linalg.det(h_matrix[:2, :2]))
        return 1e-6 < abs(det) < 1e4
    except Exception:
        return False


def video_to_pitchmap(vx: float, vy: float, h_matrix: np.ndarray) -> tuple[int, int]:
    pt = np.array([[[float(vx), float(vy)]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, h_matrix)
    return int(mapped[0, 0, 0]), int(mapped[0, 0, 1])


def pitch_ground_y_at_x(vx: float, cam_quad: np.ndarray) -> float:
    """Pitch turf Y at video X — bottom edge of homography quad (side-on landscape)."""
    quad = np.asarray(cam_quad, dtype=np.float32).reshape(4, 2)
    bl, br = quad[2], quad[3]
    x0, y0 = float(bl[0]), float(bl[1])
    x1, y1 = float(br[0]), float(br[1])
    if abs(x1 - x0) < 1.0:
        return y0
    t = (float(vx) - x0) / (x1 - x0)
    t = max(0.0, min(1.0, t))
    return y0 + t * (y1 - y0)


def snap_to_pitch_ground(
    vx: float,
    vy: float,
    cam_quad: np.ndarray,
    height: int,
    *,
    air_tol_ratio: float = 0.028,
) -> tuple[int, int] | None:
    """
    Snap bounce marker onto pitch turf. Returns None if point is still clearly in the air.
    """
    ground_y = pitch_ground_y_at_x(vx, cam_quad)
    air_tol = max(12.0, height * air_tol_ratio)
    if float(vy) < ground_y - air_tol:
        return None
    snapped_y = int(round(max(float(vy), ground_y - 2.0)))
    return int(round(vx)), snapped_y


def is_on_pitch_map(vx: float, vy: float, h_matrix: np.ndarray, margin: int = 10) -> bool:
    px, py = video_to_pitchmap(vx, vy, h_matrix)
    return (
        PITCH_L - margin <= px <= PITCH_R + margin
        and PITCH_TOP - margin <= py <= PITCH_BOT + margin
    )
