"""SQLAlchemy ORM models for cricket analytics."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Match(Base):
    __tablename__ = "matches"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(200), nullable=False)
    venue = Column(String(200))
    match_date = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default="pending")

    sessions = relationship("Session", back_populates="match", cascade="all, delete-orphan")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=_uuid)
    match_id = Column(String(36), ForeignKey("matches.id"), nullable=False)
    name = Column(String(200))
    bowler_id = Column(String(36))
    batter_id = Column(String(36))
    video_path = Column(String(500))
    processed_video_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default="pending")

    match = relationship("Match", back_populates="sessions")
    balls = relationship("Ball", back_populates="session", cascade="all, delete-orphan")


class Player(Base):
    __tablename__ = "players"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(200), nullable=False)
    role = Column(String(50))  # bowler, batter, all-rounder
    style = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)


class Ball(Base):
    __tablename__ = "balls"

    id = Column(String(36), primary_key=True, default=_uuid)
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False)
    ball_number = Column(Integer)
    frame_index = Column(Integer)
    timestamp_sec = Column(Float)

    # Real-world bounce coordinates (metres)
    bounce_x = Column(Float)
    bounce_y = Column(Float)
    pitch_map_x = Column(Integer)
    pitch_map_y = Column(Integer)

    length_type = Column(String(50))
    line_type = Column(String(50))
    outcome = Column(String(50))  # DOTS, RUNS, BOUNDARIES, WICKETS
    runs = Column(Integer, default=0)
    is_wicket = Column(Boolean, default=False)

    speed_kmh = Column(Float)
    detection_confidence = Column(Float)
    tracking_confidence = Column(Float)
    bounce_confidence = Column(Float)

    bowler_id = Column(String(36))
    batter_id = Column(String(36))
    pose_data = Column(Text)  # JSON string of pose landmarks

    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="balls")
