"""
Heatmap generation from real bounce positions only.
Produces density grids and zone-filtered heatmaps for analytics dashboards.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import cv2
import numpy as np

from pitch_map_renderer import PITCH_L, PITCH_R, PITCH_TOP, PITCH_BOT, create_hawkeye_template

HEATMAP_W = PITCH_R - PITCH_L
HEATMAP_H = PITCH_BOT - PITCH_TOP

LENGTH_FILTERS = {
    "all": None,
    "yorker": ("YORKER",),
    "full_length": ("FULL LENGTH", "FULL"),
    "good_length": ("GOOD LENGTH", "LENGTH"),
    "short_ball": ("SHORT BALL", "SHORT", "BOUNCER", "BACK OF LENGTH"),
    "full_toss": ("FULL TOSS",),
}


def _filter_bounces(bounces: list[dict], zone_filter: str | None) -> list[dict]:
    if not zone_filter or zone_filter == "all":
        return bounces
    allowed = LENGTH_FILTERS.get(zone_filter)
    if not allowed:
        return bounces
    return [
        b
        for b in bounces
        if b.get("length_type") in allowed or b.get("length") in allowed
    ]


def generate_heatmap_grid(
    bounces: list[dict],
    zone_filter: str = "all",
    grid_size: int = 32,
) -> dict[str, Any]:
    """
    Build a 2D density grid from bounce positions (pitch-map or world coords).
    Returns grid data suitable for Plotly or API JSON responses.
    """
    filtered = _filter_bounces(bounces, zone_filter)
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    for b in filtered:
        if "coords" in b:
            px, py = b["coords"]
        elif b.get("bounce_x") is not None:
            from core.pitch_coords import world_to_pitchmap
            px, py = world_to_pitchmap(b["bounce_x"], b["bounce_y"])
        else:
            continue

        gx = int((px - PITCH_L) / HEATMAP_W * (grid_size - 1))
        gy = int((py - PITCH_TOP) / HEATMAP_H * (grid_size - 1))
        gx = max(0, min(grid_size - 1, gx))
        gy = max(0, min(grid_size - 1, gy))
        grid[gy, gx] += 1.0

    # Gaussian blur for visual density
    if grid.max() > 0:
        grid = cv2.GaussianBlur(grid, (5, 5), 1.5)

    return {
        "grid": grid.tolist(),
        "grid_size": grid_size,
        "zone_filter": zone_filter,
        "ball_count": len(filtered),
        "max_density": float(grid.max()),
    }


def render_heatmap_image(
    bounces: list[dict],
    zone_filter: str = "all",
    title: str = "BOWLING HEATMAP",
) -> np.ndarray:
    """Render a colour heatmap image overlaid on pitch template."""
    filtered = _filter_bounces(bounces, zone_filter)
    base = create_hawkeye_template(title)
    overlay = np.zeros((base.shape[0], base.shape[1], 3), dtype=np.uint8)
    density = np.zeros((HEATMAP_H, HEATMAP_W), dtype=np.float32)

    for b in filtered:
        if "coords" in b:
            px, py = int(b["coords"][0]), int(b["coords"][1])
        elif b.get("bounce_x") is not None:
            from core.pitch_coords import world_to_pitchmap
            px, py = world_to_pitchmap(b["bounce_x"], b["bounce_y"])
        else:
            continue
        lx = px - PITCH_L
        ly = py - PITCH_TOP
        if 0 <= lx < HEATMAP_W and 0 <= ly < HEATMAP_H:
            density[ly, lx] += 1.0

    if density.max() > 0:
        density = cv2.GaussianBlur(density, (21, 21), 6)
        norm = density / density.max()
        heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        roi = overlay[PITCH_TOP:PITCH_BOT, PITCH_L:PITCH_R]
        mask = norm > 0.05
        roi[mask] = heat[mask]
        overlay[PITCH_TOP:PITCH_BOT, PITCH_L:PITCH_R] = roi
        blended = cv2.addWeighted(base, 0.55, overlay, 0.45, 0)
    else:
        blended = base

    label = f"{title} ({len(filtered)} balls)"
    cv2.putText(
        blended, label, (PITCH_L, PITCH_TOP - 12),
        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return blended


def heatmap_to_base64(img: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode heatmap image as base64 data URI."""
    _, buf = cv2.imencode(fmt, img)
    b64 = base64.b64encode(buf).decode("ascii")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"


def generate_all_heatmaps(bounces: list[dict]) -> dict[str, Any]:
    """Generate heatmap data for all zone filters."""
    return {
        zone: generate_heatmap_grid(bounces, zone_filter=zone)
        for zone in LENGTH_FILTERS
    }
