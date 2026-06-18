"""
Real-world pitch coordinate transforms.

Maps pitch-map pixel coordinates to metres from the striker's stumps.
Origin: centre of batting crease (stumps line), X = off-side positive.
"""

import cv2
import numpy as np

from pitch_map_renderer import PITCH_L, PITCH_R, PITCH_TOP, PITCH_BOT

# Standard cricket pitch dimensions (ICC)
PITCH_LENGTH_M = 20.12
PITCH_WIDTH_M = 3.05
STUMP_LINE_WIDTH_M = 0.2286  # 9 inches

# Pitch map pixel span
_MAP_LENGTH_PX = PITCH_BOT - PITCH_TOP
_MAP_WIDTH_PX = PITCH_R - PITCH_L


def pitchmap_to_world(px: float, py: float) -> tuple[float, float]:
    """
    Convert pitch-map pixels to real-world metres.
    Returns (x_m, y_m) where y is distance from stumps toward bowler.
    """
    x_norm = (px - PITCH_L) / _MAP_WIDTH_PX
    y_norm = (py - PITCH_TOP) / _MAP_LENGTH_PX
    x_m = (x_norm - 0.5) * PITCH_WIDTH_M
    y_m = y_norm * PITCH_LENGTH_M
    return round(x_m, 3), round(y_m, 3)


def world_to_pitchmap(x_m: float, y_m: float) -> tuple[int, int]:
    """Convert real-world metres to pitch-map pixel coordinates."""
    x_norm = x_m / PITCH_WIDTH_M + 0.5
    y_norm = y_m / PITCH_LENGTH_M
    px = int(PITCH_L + x_norm * _MAP_WIDTH_PX)
    py = int(PITCH_TOP + y_norm * _MAP_LENGTH_PX)
    return px, py


def video_to_world(
    cam_x: float, cam_y: float, h_matrix: np.ndarray
) -> tuple[float, float]:
    """Transform video pixel coordinates to real-world pitch metres."""
    pt = np.array([[[cam_x, cam_y]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, h_matrix)
    px, py = float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
    return pitchmap_to_world(px, py)


def video_to_pitchmap(
    cam_x: float, cam_y: float, h_matrix: np.ndarray
) -> tuple[int, int]:
    pt = np.array([[[cam_x, cam_y]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, h_matrix)
    return int(mapped[0, 0, 0]), int(mapped[0, 0, 1])
