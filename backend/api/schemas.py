"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MatchCreate(BaseModel):
    name: str
    venue: str | None = None


class MatchResponse(BaseModel):
    id: str
    name: str
    venue: str | None
    status: str
    created_at: datetime | None

    class Config:
        from_attributes = True


class SessionCreate(BaseModel):
    match_id: str
    name: str | None = None
    bowler_id: str | None = None


class SessionResponse(BaseModel):
    id: str
    match_id: str
    name: str | None
    status: str
    bowler_id: str | None
    video_path: str | None
    processed_video_path: str | None

    class Config:
        from_attributes = True


class PlayerCreate(BaseModel):
    name: str
    role: str | None = None
    style: str | None = None


class PlayerResponse(BaseModel):
    id: str
    name: str
    role: str | None
    style: str | None

    class Config:
        from_attributes = True


class BounceData(BaseModel):
    bounce_x: float
    bounce_y: float
    confidence: float
    length_type: str | None = None
    line_type: str | None = None


class BallResponse(BaseModel):
    id: str | None = None
    ball_number: int | None = None
    frame: int | None = None
    bounce_x: float | None = None
    bounce_y: float | None = None
    coords: list[int] | None = None
    length_type: str | None = None
    line_type: str | None = None
    type: str | None = None
    speed_kmh: float | None = None
    detection_confidence: float | None = None
    tracking_confidence: float | None = None


class AnalyticsResponse(BaseModel):
    total_balls: int
    dot_ball_pct: float
    boundary_pct: float
    wicket_pct: float
    yorker_pct: float
    good_length_pct: float
    short_ball_pct: float
    full_toss_pct: float
    avg_bounce_x: float
    avg_bounce_y: float
    avg_speed_kmh: float
    length_distribution: dict[str, int]
    line_distribution: dict[str, int]
    bowling_consistency_score: float
    accuracy_score: float


class HeatmapRequest(BaseModel):
    session_id: str | None = None
    match_id: str | None = None
    bowler_id: str | None = None
    zone_filter: str = Field(default="all", description="all|yorker|full_length|good_length|short_ball|full_toss")


class ProcessMatchRequest(BaseModel):
    match_id: str
    session_id: str | None = None
    bowler_id: str | None = None


class JobStatusResponse(BaseModel):
    status: str
    progress: float | None = None
    frame: int | None = None
    total_frames: int | None = None
    pass_info: str | None = None
    video_url: str | None = None
    report_pdf_url: str | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None
    queue_position: int | None = None
