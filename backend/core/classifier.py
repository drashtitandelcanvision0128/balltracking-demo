"""
Line and length classification from real-world pitch coordinates.
Uses only actual bounce positions — no trajectory prediction.
"""

from dataclasses import dataclass

from core.pitch_coords import PITCH_LENGTH_M, PITCH_WIDTH_M, STUMP_LINE_WIDTH_M

# Length zone boundaries (metres from stumps toward bowler)
LENGTH_BOUNDARIES_M = {
    "FULL TOSS": (-0.5, 0.5),
    "YORKER": (0.5, 2.0),
    "FULL LENGTH": (2.0, 4.0),
    "GOOD LENGTH": (4.0, 6.5),
    "BACK OF LENGTH": (6.5, 8.5),
    "SHORT BALL": (8.5, 12.0),
    "BOUNCER": (12.0, PITCH_LENGTH_M + 1),
}

# Line zone boundaries (metres from centre stump, off-side positive)
HALF_STUMP = STUMP_LINE_WIDTH_M / 2
LINE_BOUNDARIES_M = [
    ("WIDE OUTSIDE OFF", PITCH_WIDTH_M / 2 + 0.15, float("inf")),
    ("OUTSIDE OFF", HALF_STUMP + 0.08, PITCH_WIDTH_M / 2 + 0.15),
    ("OFF STUMP", 0.04, HALF_STUMP + 0.08),
    ("MIDDLE STUMP", -0.04, 0.04),
    ("LEG STUMP", -(HALF_STUMP + 0.08), -0.04),
    ("WIDE LEG SIDE", float("-inf"), -(HALF_STUMP + 0.08)),
]


@dataclass
class BounceClassification:
    bounce_x: float
    bounce_y: float
    length_type: str
    line_type: str
    confidence: float
    length_legacy: str  # maps to pitch_map_renderer zone names


# Legacy name mapping for pitch map renderer compatibility
_LENGTH_LEGACY = {
    "FULL TOSS": "FULL TOSS",
    "YORKER": "YORKER",
    "FULL LENGTH": "FULL",
    "GOOD LENGTH": "LENGTH",
    "BACK OF LENGTH": "BACK OF A LENGTH",
    "SHORT BALL": "SHORT",
    "BOUNCER": "SHORT",
}


def classify_length(y_m: float, full_toss: bool = False) -> str:
    """Classify delivery length from bounce Y coordinate (metres from stumps)."""
    if full_toss or y_m < 0.5:
        return "FULL TOSS"
    for label, (lo, hi) in LENGTH_BOUNDARIES_M.items():
        if label == "FULL TOSS":
            continue
        if lo <= y_m < hi:
            return label
    return "BOUNCER"


def classify_line(x_m: float) -> str:
    """Classify delivery line from bounce X coordinate (metres from centre stump)."""
    for label, lo, hi in LINE_BOUNDARIES_M:
        if lo <= x_m < hi:
            return label
    return "MIDDLE STUMP"


def classify_bounce(
    x_m: float,
    y_m: float,
    detection_confidence: float = 1.0,
    tracking_confidence: float = 1.0,
    full_toss: bool = False,
) -> BounceClassification:
    """
    Full bounce classification with confidence score.
    Confidence is derived from detection + tracking quality only.
    """
    length_type = classify_length(y_m, full_toss=full_toss)
    line_type = classify_line(x_m)
    conf = round(min(1.0, detection_confidence * 0.5 + tracking_confidence * 0.5), 2)
    return BounceClassification(
        bounce_x=x_m,
        bounce_y=y_m,
        length_type=length_type,
        line_type=line_type,
        confidence=conf,
        length_legacy=_LENGTH_LEGACY.get(length_type, "LENGTH"),
    )


def length_distribution(bounces: list[dict]) -> dict[str, int]:
    """Count balls per length zone."""
    counts = {k: 0 for k in LENGTH_BOUNDARIES_M}
    for b in bounces:
        lt = b.get("length_type") or b.get("length", "GOOD LENGTH")
        key = lt if lt in counts else _LENGTH_LEGACY.get(lt, "GOOD LENGTH")
        if key in counts:
            counts[key] += 1
        else:
            counts["GOOD LENGTH"] = counts.get("GOOD LENGTH", 0) + 1
    return counts


def line_distribution(bounces: list[dict]) -> dict[str, int]:
    """Count balls per line zone."""
    counts = {label: 0 for label, _, _ in LINE_BOUNDARIES_M}
    for b in bounces:
        lt = b.get("line_type", "MIDDLE STUMP")
        if lt in counts:
            counts[lt] += 1
    return counts
