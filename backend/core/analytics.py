"""
Analytics engine — computes bowling statistics from real tracked bounce data.
No trajectory prediction; all metrics derived from actual detections.
"""

from __future__ import annotations

import math
from typing import Any

from core.classifier import length_distribution, line_distribution
from core.speed_calibrator import session_speed_stats


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(100.0 * numerator / denominator, 1)


def compute_session_analytics(bounces: list[dict]) -> dict[str, Any]:
    """Aggregate analytics from a list of bounce/delivery records."""
    total = len(bounces)
    if total == 0:
        return _empty_analytics()

    dots = sum(1 for b in bounces if _outcome(b) in ("DOTS", "DOT"))
    runs = sum(1 for b in bounces if _outcome(b) == "RUNS")
    boundaries = sum(1 for b in bounces if _outcome(b) in ("BOUNDARIES", "4", "6"))
    wickets = sum(1 for b in bounces if _outcome(b) in ("WICKETS", "OUT"))

    length_dist = length_distribution(bounces)
    line_dist = line_distribution(bounces)

    yorkers = length_dist.get("YORKER", 0)
    good_length = length_dist.get("GOOD LENGTH", 0)
    short_balls = length_dist.get("SHORT BALL", 0) + length_dist.get("BOUNCER", 0)
    full_toss = length_dist.get("FULL TOSS", 0)

    xs = [b.get("bounce_x", 0) for b in bounces if b.get("bounce_x") is not None]
    ys = [b.get("bounce_y", 0) for b in bounces if b.get("bounce_y") is not None]
    avg_x = round(sum(xs) / len(xs), 2) if xs else 0.0
    avg_y = round(sum(ys) / len(ys), 2) if ys else 0.0

    speeds = [float(b.get("speed_kmh", 0)) for b in bounces if b.get("speed_kmh")]
    speed_stats = session_speed_stats(bounces)
    avg_speed = speed_stats["avg_speed_kmh"]

    return {
        "total_balls": total,
        "dot_ball_pct": _pct(dots, total),
        "boundary_pct": _pct(boundaries, total),
        "wicket_pct": _pct(wickets, total),
        "run_pct": _pct(runs + boundaries, total),
        "yorker_pct": _pct(yorkers, total),
        "good_length_pct": _pct(good_length, total),
        "short_ball_pct": _pct(short_balls, total),
        "full_toss_pct": _pct(full_toss, total),
        "avg_bounce_x": avg_x,
        "avg_bounce_y": avg_y,
        "avg_speed_kmh": avg_speed,
        "max_speed_kmh": speed_stats["max_speed_kmh"],
        "min_speed_kmh": speed_stats["min_speed_kmh"],
        "pace_tier": speed_stats["pace_tier"],
        "pace_label": speed_stats["pace_label"],
        "pace_avg_range": speed_stats["pace_avg_range"],
        "pace_max_cap": speed_stats["pace_max_cap"],
        "length_distribution": length_dist,
        "line_distribution": line_dist,
        "dots": dots,
        "runs": runs,
        "boundaries": boundaries,
        "wickets": wickets,
        "bowling_consistency_score": _consistency_score(bounces),
        "accuracy_score": _accuracy_score(bounces),
    }


def _outcome(b: dict) -> str:
    return str(b.get("type") or b.get("outcome") or "DOTS").upper()


def _empty_analytics() -> dict[str, Any]:
    return {
        "total_balls": 0,
        "dot_ball_pct": 0.0,
        "boundary_pct": 0.0,
        "wicket_pct": 0.0,
        "run_pct": 0.0,
        "yorker_pct": 0.0,
        "good_length_pct": 0.0,
        "short_ball_pct": 0.0,
        "full_toss_pct": 0.0,
        "avg_bounce_x": 0.0,
        "avg_bounce_y": 0.0,
        "avg_speed_kmh": 0.0,
        "max_speed_kmh": 0.0,
        "min_speed_kmh": 0.0,
        "pace_tier": "unknown",
        "pace_label": "Unknown",
        "pace_avg_range": "",
        "pace_max_cap": 0,
        "length_distribution": {},
        "line_distribution": {},
        "dots": 0,
        "runs": 0,
        "boundaries": 0,
        "wickets": 0,
        "bowling_consistency_score": 0.0,
        "accuracy_score": 0.0,
    }


def _consistency_score(bounces: list[dict]) -> float:
    """
    Higher score = more consistent length (lower variance in bounce Y).
    Scale 0–100.
    """
    ys = [b.get("bounce_y") for b in bounces if b.get("bounce_y") is not None]
    if len(ys) < 2:
        return 0.0
    mean_y = sum(ys) / len(ys)
    variance = sum((y - mean_y) ** 2 for y in ys) / len(ys)
    std = math.sqrt(variance)
    # std of 0m → 100, std of 4m+ → ~0
    score = max(0.0, 100.0 - std * 25.0)
    return round(score, 1)


def _accuracy_score(bounces: list[dict]) -> float:
    """
    Fraction of balls landing in good-length or yorker zones, scaled 0–100.
    """
    if not bounces:
        return 0.0
    on_target = sum(
        1
        for b in bounces
        if b.get("length_type") in ("YORKER", "GOOD LENGTH", "FULL LENGTH")
        or b.get("length") in ("YORKER", "LENGTH", "FULL")
    )
    return round(100.0 * on_target / len(bounces), 1)


def player_statistics(
    bounces: list[dict], bowler_id: str | None = None
) -> dict[str, Any]:
    """Per-bowler stats filtered by bowler_id if provided."""
    if bowler_id:
        filtered = [b for b in bounces if b.get("bowler_id") == bowler_id]
    else:
        filtered = bounces
    stats = compute_session_analytics(filtered)
    stats["bowler_id"] = bowler_id
    return stats
