"""
Cricket ball bounce detection via vertical direction-change analysis.

Why direction-change instead of lowest Y?
-----------------------------------------
YOLO noise creates false local minima in Y. A ball that jitters up/down by a few
pixels can look like it "bounced" at the wrong frame. Requiring sustained downward
motion followed by sustained upward motion is far more stable.

Pipeline
--------
1. track_history   — keep a rolling window of (frame_id, x, y)
2. smooth_positions — moving-average filter removes single-frame spikes
3. detect_bounce   — find the inflection where dy flips from + to −
4. confidence_estimation — score how clean the descent/ascent pattern is
5. draw_prediction — red dot + label + trajectory split at bounce
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class BallPoint:
    frame_id: int
    x: float
    y: float


@dataclass(frozen=True)
class BouncePrediction:
    bounce_frame: int
    bounce_x: float
    bounce_y: float
    confidence: float
    history_index: int


class TrackHistory:
  """Rolling buffer of recent ball centres — limits memory and focuses on current delivery."""

  def __init__(self, maxlen: int = 20) -> None:
    self._pts: Deque[BallPoint] = deque(maxlen=maxlen)

  def add(self, frame_id: int, x: float, y: float) -> None:
    self._pts.append(BallPoint(frame_id, float(x), float(y)))

  def extend(self, points: Iterable[tuple[int, float, float]]) -> None:
    for frame_id, x, y in points:
      self.add(frame_id, x, y)

  def clear(self) -> None:
    self._pts.clear()

  def as_list(self) -> list[BallPoint]:
    return list(self._pts)

  def __len__(self) -> int:
    return len(self._pts)


def track_history(
    points: Sequence[tuple[int, float, float]] | Sequence[BallPoint],
    *,
    maxlen: int = 20,
) -> list[BallPoint]:
  """
  Build a bounded history from raw detections.

  Why: bounce detection only needs the last 10–20 frames; older points add noise
  from release/hand motion and previous deliveries.
  """
  hist = TrackHistory(maxlen=maxlen)
  for p in points:
    if isinstance(p, BallPoint):
      hist.add(p.frame_id, p.x, p.y)
    else:
      hist.add(p[0], p[1], p[2])
  return hist.as_list()


def smooth_positions(
    history: Sequence[BallPoint],
    *,
    window: int = 3,
) -> list[BallPoint]:
  """
  Moving-average smoothing on x and y.

  Why: YOLO centre can jump 2–8 px between frames. Smoothing prevents a single
  outlier from faking a direction reversal.
  """
  if len(history) < 2:
    return list(history)
  w = max(1, int(window))
  xs = np.array([p.x for p in history], dtype=np.float64)
  ys = np.array([p.y for p in history], dtype=np.float64)
  kernel = np.ones(w, dtype=np.float64) / w
  sx = np.convolve(xs, kernel, mode="same")
  sy = np.convolve(ys, kernel, mode="same")
  return [
    BallPoint(history[i].frame_id, float(sx[i]), float(sy[i]))
    for i in range(len(history))
  ]


def _vertical_deltas(history: Sequence[BallPoint]) -> list[float]:
  """dy > 0 means screen-down (toward pitch); dy < 0 means screen-up (after bounce)."""
  return [history[i].y - history[i - 1].y for i in range(1, len(history))]


def confidence_estimation(
    history: Sequence[BallPoint],
    bounce_idx: int,
    *,
    min_descent_frames: int = 3,
    min_ascent_frames: int = 2,
    min_dy_px: float = 1.5,
) -> float:
  """
  Score how convincing the bounce inflection is (0–1).

  Why: low-confidence bounces (weak reversal, gaps, flat motion) are rejected so we
  do not mark false pitch points.
  """
  if bounce_idx < 1 or bounce_idx >= len(history):
    return 0.0

  dys = _vertical_deltas(history)
  di = bounce_idx - 1  # delta index at bounce transition

  before = dys[max(0, di - min_descent_frames + 1): di + 1]
  after = dys[di + 1: di + 1 + min_ascent_frames]

  if not before or not after:
    return 0.0

  descent_ok = sum(1 for d in before if d >= min_dy_px)
  ascent_ok = sum(1 for d in after if d <= -min_dy_px * 0.6)

  descent_ratio = descent_ok / max(len(before), 1)
  ascent_ratio = ascent_ok / max(len(after), 1)

  mean_down = float(np.mean([d for d in before if d > 0] or [0.0]))
  mean_up = float(np.mean([abs(d) for d in after if d < 0] or [0.0]))
  amp_score = min(1.0, (mean_down + mean_up) / (min_dy_px * 4.0))

  # Penalise frame gaps (missed detections)
  gap_pen = 1.0
  for i in range(max(1, bounce_idx - 2), bounce_idx + 2):
    if i < len(history):
      gap = history[i].frame_id - history[i - 1].frame_id
      if gap > 3:
        gap_pen *= 0.75

  conf = 0.45 * descent_ratio + 0.35 * ascent_ratio + 0.20 * amp_score
  return round(min(1.0, max(0.0, conf * gap_pen)), 3)


def detect_bounce(
    points: Sequence[tuple[int, float, float]] | Sequence[BallPoint],
    *,
    history_size: int = 20,
    smooth_window: int = 3,
    min_descent_frames: int = 3,
    min_ascent_frames: int = 2,
    min_dy_px: float = 1.5,
    min_confidence: float = 0.52,
    min_travel_px: float = 18.0,
) -> BouncePrediction | None:
  """
  Find bounce frame where vertical velocity flips from downward to upward.

  Image coordinates: Y grows toward the bottom of the frame, so pre-bounce flight
  gives dy > 0 and post-bounce rebound gives dy < 0.

  Returns None when the pattern is ambiguous — never guess.
  """
  history = track_history(points, maxlen=history_size)
  if len(history) < min_descent_frames + min_ascent_frames + 2:
    return None

  smooth = smooth_positions(history, window=smooth_window)
  dys = _vertical_deltas(smooth)

  best: BouncePrediction | None = None
  best_conf = 0.0

  # Search for inflection: sustained descent then sustained ascent
  for i in range(min_descent_frames, len(dys) - min_ascent_frames + 1):
    before = dys[i - min_descent_frames: i]
    after = dys[i: i + min_ascent_frames]

    if not all(d >= min_dy_px for d in before):
      continue
    if not all(d <= -min_dy_px * 0.55 for d in after):
      continue

    bounce_idx = i  # point *after* last descent delta → landing frame in smooth list
    pt = smooth[bounce_idx]

    travel = math.hypot(pt.x - smooth[0].x, pt.y - smooth[0].y)
    if travel < min_travel_px:
      continue

    conf = confidence_estimation(
      smooth, bounce_idx,
      min_descent_frames=min_descent_frames,
      min_ascent_frames=min_ascent_frames,
      min_dy_px=min_dy_px,
    )
    if conf < min_confidence or conf <= best_conf:
      continue

    best_conf = conf
    best = BouncePrediction(
      bounce_frame=pt.frame_id,
      bounce_x=pt.x,
      bounce_y=pt.y,
      confidence=conf,
      history_index=bounce_idx,
    )

  return best


def draw_prediction(
    frame,
    prediction: BouncePrediction,
    history: Sequence[BallPoint] | Sequence[tuple[int, float, float]],
    *,
    before_color: tuple[int, int, int] = (0, 220, 255),
    after_color: tuple[int, int, int] = (255, 180, 0),
    dot_color: tuple[int, int, int] = (0, 0, 255),
    label: str = "Predicted Bounce",
) -> None:
  """
  Draw bounce marker and trajectory split at the detected inflection.

  Why draw both legs: operators can instantly see whether the algorithm tracked
  release → bounce → post-bounce sensibly.
  """
  pts = track_history(history, maxlen=999)
  if len(pts) < 2:
    return

  smooth = smooth_positions(pts, window=3)
  idx = min(prediction.history_index, len(smooth) - 1)

  def _polyline(segment, color, thickness=2):
    if len(segment) < 2:
      return
    arr = np.array([(int(p.x), int(p.y)) for p in segment], dtype=np.int32)
    cv2.polylines(frame, [arr], False, color, thickness, cv2.LINE_AA)

  _polyline(smooth[: idx + 1], before_color)
  _polyline(smooth[idx:], after_color)

  bx, by = int(prediction.bounce_x), int(prediction.bounce_y)
  h, w = frame.shape[:2]
  if not (0 <= bx < w and 0 <= by < h):
    return

  cv2.circle(frame, (bx, by), 14, dot_color, -1, cv2.LINE_AA)
  cv2.circle(frame, (bx, by), 14, (255, 255, 255), 2, cv2.LINE_AA)
  cv2.putText(
    frame, label, (bx + 16, by - 12),
    cv2.FONT_HERSHEY_SIMPLEX, 0.62, dot_color, 2, cv2.LINE_AA,
  )
  conf_txt = f"{prediction.confidence:.0%}"
  cv2.putText(
    frame, conf_txt, (bx + 16, by + 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
  )
