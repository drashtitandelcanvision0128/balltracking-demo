"""
Enhanced pitch homography calibration.

Combines colour-based pitch mask, stump/crease line detection,
multi-frame outlier rejection, and optional manual corner overrides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from accuracy_engine import sample_video_frames, _pitch_mask
from pitch_map_renderer import auto_pitch_quad

STUMP_WIDTH_M = 0.2286


@dataclass
class CalibrationResult:
    quad: np.ndarray
    source: str  # manual | stump | colour | fallback
    confidence: float
    stump_scale: float  # metres-per-pixel scale correction vs ICC stump width


def _quad_from_bands(binary: np.ndarray, height: int, width: int) -> np.ndarray | None:
    row_bands = [
        (int(height * 0.52), int(height * 0.58)),
        (int(height * 0.68), int(height * 0.74)),
        (int(height * 0.82), int(height * 0.90)),
    ]
    top_pts, bot_pts = [], []
    for y0, y1 in row_bands:
        band = binary[y0:y1, :]
        if band.size == 0:
            continue
        col_sum = band.sum(axis=0).astype(np.float32)
        if col_sum.max() < 1:
            continue
        thresh = col_sum.max() * 0.35
        cols = np.where(col_sum >= thresh)[0]
        if len(cols) < 10:
            continue
        left, right = int(cols[0]), int(cols[-1])
        mid_y = (y0 + y1) // 2
        if mid_y < height * 0.65:
            top_pts.append((left, mid_y, right, mid_y))
        else:
            bot_pts.append((left, mid_y, right, mid_y))
    if not top_pts or not bot_pts:
        return None
    tl_x = int(np.median([p[0] for p in top_pts]))
    tr_x = int(np.median([p[2] for p in top_pts]))
    ty = int(np.median([p[1] for p in top_pts]))
    bl_x = int(np.median([p[0] for p in bot_pts]))
    br_x = int(np.median([p[2] for p in bot_pts]))
    by = int(np.median([p[1] for p in bot_pts]))
    if (br_x - bl_x) < (tr_x - tl_x) * 0.9:
        return None
    if by <= ty + height * 0.12:
        return None
    return np.array([[tl_x, ty], [tr_x, ty], [bl_x, by], [br_x, by]], dtype=np.float32)


def _reject_outlier_quads(quads: list[np.ndarray], height: int, width: int) -> np.ndarray | None:
    """Median-of-quads with MAD outlier rejection."""
    if not quads:
        return None
    if len(quads) == 1:
        return quads[0]
    stack = np.stack(quads, axis=0)
    median = np.median(stack, axis=0)
    dists = np.mean(np.abs(stack - median), axis=(1, 2))
    mad = np.median(dists) + 1e-6
    keep = stack[dists <= np.median(dists) + 2.5 * mad]
    if len(keep) == 0:
        return median.astype(np.float32)
    return np.median(keep, axis=0).astype(np.float32)


def _detect_stump_quad(frame: np.ndarray, height: int, width: int) -> np.ndarray | None:
    """
    Detect batting crease / stump line via white-line Hough transform.
    Returns trapezoid when a horizontal crease is found in lower frame.
    """
    roi = frame[int(height * 0.45) : int(height * 0.92), :]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=int(width * 0.12), maxLineGap=12)
    if lines is None:
        return None

    horiz = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle < 12 or angle > 168:
            length = math.hypot(x2 - x1, y2 - y1)
            horiz.append((x1, y1 + int(height * 0.45), x2, y2 + int(height * 0.45), length))

    if not horiz:
        return None
    horiz.sort(key=lambda h: h[4], reverse=True)
    best = horiz[0]
    cx = width * 0.5
    crease_y = int((best[1] + best[3]) / 2)
    stump_half_px = max(8, int(width * 0.018))
    top_y = max(int(height * 0.40), crease_y - int(height * 0.22))
    bot_y = min(int(height * 0.95), crease_y + int(height * 0.08))
    top_hw = stump_half_px * 3.2
    bot_hw = stump_half_px * 8.5
    return np.array([
        [cx - top_hw, top_y],
        [cx + top_hw, top_y],
        [cx - bot_hw, bot_y],
        [cx + bot_hw, bot_y],
    ], dtype=np.float32)


def _stump_scale_from_quad(quad: np.ndarray, width: int) -> float:
    """Scale factor: detected stump span vs ICC 22.86 cm at batting end."""
    top_w = abs(quad[1, 0] - quad[0, 0])
    if top_w < 5:
        return 1.0
    expected_px = width * 0.022
    return float(np.clip(expected_px / top_w, 0.75, 1.35))


def calibrate_pitch_robust(
    cap,
    width: int,
    height: int,
    *,
    manual_quad: np.ndarray | list | None = None,
    max_samples: int = 30,
    blend_fallback: float = 0.55,
) -> CalibrationResult:
    """
    Best-available pitch quad for homography.
    Priority: manual > stump+colour consensus > colour > auto fallback.
    """
    fallback = auto_pitch_quad(width, height)

    if manual_quad is not None:
        quad = np.array(manual_quad, dtype=np.float32).reshape(4, 2)
        return CalibrationResult(
            quad=quad,
            source="manual",
            confidence=1.0,
            stump_scale=_stump_scale_from_quad(quad, width),
        )

    frames = sample_video_frames(cap, width, height, max_samples=max_samples)
    colour_quads: list[np.ndarray] = []
    stump_quads: list[np.ndarray] = []

    for frame in frames:
        m = _pitch_mask(frame, height)
        accum = m.astype(np.float32) / 255.0
        binary = (accum > 0.22).astype(np.uint8) * 255
        if binary.sum() >= width * height * 0.02:
            q = _quad_from_bands(binary, height, width)
            if q is not None:
                colour_quads.append(q)
        sq = _detect_stump_quad(frame, height, width)
        if sq is not None:
            stump_quads.append(sq)

    colour_median = _reject_outlier_quads(colour_quads, height, width)
    stump_median = _reject_outlier_quads(stump_quads, height, width)

    if colour_median is not None and stump_median is not None:
        # Consensus: colour width + stump vertical placement
        blend = 0.6
        quad = colour_median * blend + stump_median * (1 - blend)
        quad = quad * (1 - blend_fallback) + fallback * blend_fallback
        conf = min(1.0, 0.5 + 0.1 * len(colour_quads) + 0.1 * len(stump_quads))
        return CalibrationResult(
            quad=quad.astype(np.float32),
            source="stump+colour",
            confidence=conf,
            stump_scale=_stump_scale_from_quad(quad, width),
        )

    if colour_median is not None:
        quad = colour_median * (1 - blend_fallback) + fallback * blend_fallback
        return CalibrationResult(
            quad=quad.astype(np.float32),
            source="colour",
            confidence=min(1.0, 0.4 + 0.05 * len(colour_quads)),
            stump_scale=_stump_scale_from_quad(quad, width),
        )

    if stump_median is not None:
        quad = stump_median * (1 - blend_fallback) + fallback * blend_fallback
        return CalibrationResult(
            quad=quad.astype(np.float32),
            source="stump",
            confidence=min(1.0, 0.35 + 0.08 * len(stump_quads)),
            stump_scale=_stump_scale_from_quad(quad, width),
        )

    return CalibrationResult(
        quad=fallback.copy(),
        source="fallback",
        confidence=0.25,
        stump_scale=1.0,
    )
