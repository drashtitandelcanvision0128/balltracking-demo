"""
Delivery lane: bowler / bowling-machine end → batsman end.

All ball detection and the first pitch bounce must lie in this lane,
regardless of camera angle (portrait or landscape).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core.ball_detection_filters import (
    effective_bounce_ground_y_min,
    is_landscape_frame,
)
from core.flight_corridor import FlightCorridor, calibrate_flight_corridor
from core.video_calibration import derive_zones_from_quad


def _zone_center(zone: dict, width: int, height: int) -> tuple[float, float]:
    cx = (float(zone.get("x_min_ratio", 0)) + float(zone.get("x_max_ratio", 1))) * 0.5 * width
    cy = (float(zone.get("y_min_ratio", 0)) + float(zone.get("y_max_ratio", 1))) * 0.5 * height
    return cx, cy


@dataclass
class DeliveryLane:
    """Fixed release end (bowler/machine) and batsman end — ball travels between them."""

    width: int
    height: int
    landscape: bool
    release_center: tuple[float, float]
    batsman_center: tuple[float, float]
    corridor: FlightCorridor | None = None
    zones: dict | None = None

    @classmethod
    def calibrate(
        cls,
        width: int,
        height: int,
        *,
        quad: np.ndarray | list | None = None,
        zones: dict | None = None,
        cap=None,
        batsman_xy: tuple[float, float] | None = None,
        release_xy: tuple[float, float] | None = None,
    ) -> DeliveryLane:
        z = zones or (derive_zones_from_quad(quad, width, height) if quad is not None else {})
        mach = z.get("machine_release_zone") or {}
        bat = z.get("batsman_end") or {}
        landscape = bool(z.get("landscape", is_landscape_frame(width, height)))
        release = release_xy or _zone_center(mach, width, height)
        batsman = batsman_xy or _zone_center(bat, width, height)
        corridor = calibrate_flight_corridor(width, height, quad=quad, cap=cap, zones=z)
        return cls(
            width=width,
            height=height,
            landscape=landscape,
            release_center=release,
            batsman_center=batsman,
            corridor=corridor,
            zones=z,
        )

    def along_progress(self, x: float, y: float) -> float:
        """0 ≈ release end, 1 ≈ batsman end (can be <0 or >1 outside lane)."""
        ox, oy = self.release_center
        bx, by = self.batsman_center
        dx, dy = bx - ox, by - oy
        denom = dx * dx + dy * dy
        if denom < 64.0:
            if self.landscape:
                axis = self.batsman_center[0] - self.release_center[0]
                if abs(axis) < 8.0:
                    return 0.5
                return (x - ox) / axis
            axis = self.batsman_center[1] - self.release_center[1]
            if abs(axis) < 8.0:
                return 0.5
            return (y - oy) / axis
        return float(((x - ox) * dx + (y - oy) * dy) / denom)

    def between_ends(
        self,
        x: float,
        y: float,
        *,
        min_ratio: float = 0.10,
        max_ratio: float = 0.92,
    ) -> bool:
        t = self.along_progress(x, y)
        return min_ratio <= t <= max_ratio

    def contains_detection(self, cx: int, cy: int, *, margin_px: float = 40.0) -> bool:
        if self.corridor is not None:
            return self.corridor.contains(cx, cy, margin_px=margin_px)
        return self.between_ends(float(cx), float(cy), min_ratio=-0.05, max_ratio=1.05)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "landscape": self.landscape,
            "release_center": list(self.release_center),
            "batsman_center": list(self.batsman_center),
            "corridor": self.corridor.to_dict() if self.corridor else None,
        }


def _parse_det_track(points: list) -> list[tuple[int, float, float, float]]:
    seg: list[tuple[int, float, float, float]] = []
    for p in points:
        if len(p) >= 4 and float(p[3]) > 0:
            seg.append((int(p[0]), float(p[1]), float(p[2]), float(p[3])))
        elif len(p) >= 3:
            seg.append((int(p[0]), float(p[1]), float(p[2]), -1.0))
    return seg


def find_first_bounce_between_ends(
    points: list,
    release_frame: int,
    lane: DeliveryLane,
    *,
    height: int,
    width: int = 0,
    fps: float = 30.0,
    h_matrix=None,
    min_along: float = 0.12,
    max_along: float = 0.90,
    min_frames_after_release: int = 6,
) -> tuple[int, float, float] | None:
    """
    First bounce on the pitch between bowler/machine and batsman.

    Uses Y-reversal (ball down then up) on the YOLO detection track, constrained to
    the delivery lane between the two fixed ends.
    """
    seg = _parse_det_track([p for p in points if int(p[0]) >= release_frame])
    if len(seg) < min_frames_after_release + 4:
        return None

    min_after = max(min_frames_after_release, int(fps * 0.10))
    min_drop = max(12.0, height * 0.012)
    min_rise = max(8.0, height * 0.008)
    ground_y_min = int(height * effective_bounce_ground_y_min(width, height))

    ys_raw = [p[2] for p in seg]
    # Light smooth for peak finding; mark using raw detection coords
    if len(ys_raw) >= 3:
        k = np.ones(3) / 3.0
        ys = np.convolve(np.array(ys_raw, dtype=np.float64), k, mode="same")
    else:
        ys = np.array(ys_raw, dtype=np.float64)

    for i in range(min_after, len(seg) - 2):
        y_prev, y_curr, y_next = ys[i - 1], ys[i], ys[i + 1]
        ry_prev, ry_curr, ry_next = ys_raw[i - 1], ys_raw[i], ys_raw[i + 1]

        if not (y_curr >= y_prev and y_curr >= y_next):
            continue
        if not (ry_curr >= ry_prev and ry_curr >= ry_next):
            continue
        if y_curr - ys[0] < min_drop:
            continue

        rise = ry_curr - ry_next
        if rise < min_rise:
            continue

        f, x, y = seg[i][0], seg[i][1], seg[i][2]

        if not lane.between_ends(x, y, min_ratio=min_along, max_ratio=max_along):
            continue
        if y < ground_y_min:
            continue

        dy_before = ry_curr - ys_raw[max(0, i - 2)]
        dy_after = ry_next - ry_curr
        if dy_before < 4.0 or dy_after > -2.5:
            continue

        return int(f), float(x), float(y)

    return None
