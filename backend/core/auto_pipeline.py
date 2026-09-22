"""
Fully automatic upload pipeline — pitch, homography, players, zones. Zero manual input.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from core.config import CONFIG
from core.player_detector import CricketPlayerDetector, PlayerScene
from core.video_auto_setup import VideoProfile, analyze_and_calibrate_video
from core.video_calibration import (
    apply_video_zones,
    derive_zones_from_quad,
    zones_for_log,
    apply_player_pitch_calibration,
)

_AUTO = CONFIG.get("processing", {}).get("fully_automatic", {})


def _sample_frames(cap, total: int, n: int = 10) -> list[np.ndarray]:
    if total <= 0:
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return [frame] if ret and frame is not None else []
    indices = np.linspace(int(total * 0.05), int(total * 0.70), n, dtype=int)
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frames


def run_automatic_setup(
    video_path: str,
    *,
    manual_quad: list | np.ndarray | None = None,
    device: str | int = "cpu",
    half: bool = False,
) -> tuple[VideoProfile, PlayerScene, dict[str, Any]]:
    """
    One-shot automatic setup after upload:
    pitch calibration → homography quad → zones → AI player roles.
    """
    profile = analyze_and_calibrate_video(video_path, manual_quad=manual_quad)
    quad = np.array(profile.calibration["quad"], dtype=np.float32)
    zones = profile.calibration.get("zones") or derive_zones_from_quad(quad, profile.width, profile.height)
    apply_video_zones(zones)

    cap = cv2.VideoCapture(video_path)
    frames = _sample_frames(cap, profile.total_frames, n=int(_AUTO.get("player_sample_frames", 8)))
    cap.release()

    players = PlayerScene(landscape=profile.orientation == "landscape")
    if bool(_AUTO.get("player_detection", True)) and frames:
        try:
            det = CricketPlayerDetector(device=device, half=half)
            players = det.detect_scene(frames, profile.width, profile.height, zones=zones)
        except Exception as exc:
            print(f"[AutoPipeline] player detection skipped: {exc}", flush=True)

    if manual_quad is None and (players.striker or players.bowler):
        player_quad, player_zones = apply_player_pitch_calibration(
            profile.width, profile.height, players,
        )
        if player_quad is not None and player_zones is not None:
            quad = player_quad
            zones = player_zones
            apply_video_zones(zones)
            profile.calibration["quad"] = player_quad.tolist()
            profile.calibration["source"] = "players"
            profile.calibration["zones"] = zones
            print(
                "[AutoPipeline] pitch from batsman stance → bowler end "
                f"(no overlay) | {zones_for_log(zones)}",
                flush=True,
            )

    meta = {
        "mode": "fully_automatic",
        "manual_quad_used": manual_quad is not None,
        "zones": zones,
        "zones_log": zones_for_log(zones),
        "players": players.to_dict(),
        "pitch_source": profile.calibration.get("source"),
        "pitch_confidence": profile.calibration.get("confidence"),
    }
    profile.calibration["players"] = players.to_dict()
    profile.calibration["auto_meta"] = {
        "mode": "fully_automatic",
        "player_detection": bool(_AUTO.get("player_detection", True)),
        "auto_recalibrate": bool(_AUTO.get("auto_recalibrate", True)),
    }

    print(
        f"[AutoPipeline] pitch={meta['pitch_source']} conf={meta['pitch_confidence']} | "
        f"{meta['zones_log']} | players={players.to_dict().get('count', 0)}",
        flush=True,
    )
    return profile, players, meta
