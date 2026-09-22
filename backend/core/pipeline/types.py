"""Shared types for the cricket ball delivery pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrackPoint:
    frame_id: int
    x: float
    y: float
    track_id: int
    predicted: bool = False
    confidence: float = 1.0


@dataclass
class BounceResult:
    bounce_frame: int
    image_x: float
    image_y: float
    pitch_x_m: float
    pitch_y_m: float
    pitch_map_x: int
    pitch_map_y: int
    confidence: float
    method: str
    history_index: int = -1


@dataclass
class DeliveryAnalysis:
    track_id: int
    trajectory: list[TrackPoint] = field(default_factory=list)
    bounce: BounceResult | None = None
    smoothed_path: list[tuple[int, int]] = field(default_factory=list)
