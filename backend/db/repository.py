"""Data access layer for cricket analytics."""

import json
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from db.models import Ball, Match, Player, Session as MatchSession


class AnalyticsRepository:
    def __init__(self, db: Session):
        self.db = db

    # ---- Matches ----
    def create_match(self, name: str, venue: str | None = None) -> Match:
        match = Match(name=name, venue=venue, status="active")
        self.db.add(match)
        self.db.flush()
        return match

    def get_match(self, match_id: str) -> Match | None:
        return self.db.query(Match).filter(Match.id == match_id).first()

    def list_matches(self, limit: int = 50) -> list[Match]:
        return (
            self.db.query(Match)
            .order_by(Match.created_at.desc())
            .limit(limit)
            .all()
        )

    # ---- Sessions ----
    def create_session(
        self,
        match_id: str,
        name: str | None = None,
        bowler_id: str | None = None,
        video_path: str | None = None,
    ) -> MatchSession:
        session = MatchSession(
            match_id=match_id,
            name=name or f"Session {datetime.utcnow().strftime('%H:%M')}",
            bowler_id=bowler_id,
            video_path=video_path,
            status="pending",
        )
        self.db.add(session)
        self.db.flush()
        return session

    def get_session(self, session_id: str) -> MatchSession | None:
        return self.db.query(MatchSession).filter(MatchSession.id == session_id).first()

    def update_session_status(
        self, session_id: str, status: str, processed_video_path: str | None = None
    ):
        session = self.get_session(session_id)
        if session:
            session.status = status
            if processed_video_path:
                session.processed_video_path = processed_video_path

    # ---- Balls ----
    def save_ball(self, session_id: str, ball_data: dict[str, Any]) -> Ball:
        pose = ball_data.get("pose_data")
        ball = Ball(
            session_id=session_id,
            ball_number=ball_data.get("ball_number"),
            frame_index=ball_data.get("frame"),
            timestamp_sec=ball_data.get("timestamp_sec"),
            bounce_x=ball_data.get("bounce_x"),
            bounce_y=ball_data.get("bounce_y"),
            pitch_map_x=ball_data.get("pitch_map_x"),
            pitch_map_y=ball_data.get("pitch_map_y"),
            length_type=ball_data.get("length_type"),
            line_type=ball_data.get("line_type"),
            outcome=ball_data.get("type") or ball_data.get("outcome"),
            runs=ball_data.get("runs", 0),
            is_wicket=ball_data.get("type") == "WICKETS",
            speed_kmh=ball_data.get("speed_kmh"),
            detection_confidence=ball_data.get("detection_confidence"),
            tracking_confidence=ball_data.get("tracking_confidence"),
            bounce_confidence=ball_data.get("bounce_confidence"),
            bowler_id=ball_data.get("bowler_id"),
            batter_id=ball_data.get("batter_id"),
            pose_data=json.dumps(pose) if pose else None,
        )
        self.db.add(ball)
        self.db.flush()
        return ball

    def get_balls(
        self,
        session_id: str | None = None,
        match_id: str | None = None,
        bowler_id: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[Ball]:
        q = self.db.query(Ball)
        if session_id:
            q = q.filter(Ball.session_id == session_id)
        if match_id:
            q = q.join(MatchSession).filter(MatchSession.match_id == match_id)
        if bowler_id:
            q = q.filter(Ball.bowler_id == bowler_id)
        if date_from:
            q = q.filter(Ball.created_at >= date_from)
        if date_to:
            q = q.filter(Ball.created_at <= date_to)
        return q.order_by(Ball.ball_number).all()

    def balls_to_dicts(self, balls: list[Ball]) -> list[dict[str, Any]]:
        result = []
        for b in balls:
            d = {
                "id": b.id,
                "session_id": b.session_id,
                "ball_number": b.ball_number,
                "frame": b.frame_index,
                "bounce_x": b.bounce_x,
                "bounce_y": b.bounce_y,
                "coords": [b.pitch_map_x, b.pitch_map_y],
                "length_type": b.length_type,
                "line_type": b.line_type,
                "type": b.outcome,
                "speed_kmh": b.speed_kmh,
                "runs": b.runs,
                "detection_confidence": b.detection_confidence,
                "tracking_confidence": b.tracking_confidence,
                "bounce_confidence": b.bounce_confidence,
                "bowler_id": b.bowler_id,
            }
            if b.pose_data:
                try:
                    d["pose_data"] = json.loads(b.pose_data)
                except json.JSONDecodeError:
                    pass
            result.append(d)
        return result

    # ---- Players ----
    def create_player(self, name: str, role: str | None = None, style: str | None = None) -> Player:
        player = Player(name=name, role=role, style=style)
        self.db.add(player)
        self.db.flush()
        return player

    def list_players(self) -> list[Player]:
        return self.db.query(Player).order_by(Player.name).all()

    def get_player(self, player_id: str) -> Player | None:
        return self.db.query(Player).filter(Player.id == player_id).first()
