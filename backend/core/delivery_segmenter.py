"""
Pass 1 — light scan to find delivery clip boundaries (start/end frames).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2

from core.config import CONFIG
from core.delivery_filter import can_start_new_delivery, min_gap_frames

_PROC = CONFIG.get("processing", {})
CLIP_SCAN_STRIDE = int(_PROC.get("clip_scan_stride", 2))
CLIP_START_PADDING = int(_PROC.get("clip_start_padding", 15))
CLIP_END_PADDING = int(_PROC.get("clip_end_padding", 25))
CLIP_LOST_SEC = float(_PROC.get("clip_lost_sec", 0.85))
SCAN_CONF = float(_PROC.get("clip_scan_conf", 0.10))


@dataclass
class DeliveryClip:
    start: int
    end: int
    release_frame: int | None = None


def detect_ball_light(
    model,
    frame,
    width: int,
    height: int,
    *,
    conf_thresh: float,
    imgsz: int,
    max_dim: int,
    use_half: bool,
    yolo_device,
) -> tuple[int, int, float] | None:
    """Fast single-frame ball detection for Pass 1 scanning."""
    infer_scale = min(1.0, max_dim / max(width, height, 1))
    infer_frame = frame
    if infer_scale < 1.0:
        infer_frame = cv2.resize(
            frame,
            (int(width * infer_scale), int(height * infer_scale)),
            interpolation=cv2.INTER_LINEAR,
        )
    inv = 1.0 / infer_scale
    results = model.predict(
        infer_frame,
        conf=conf_thresh,
        imgsz=imgsz,
        max_det=8,
        verbose=False,
        half=use_half,
        device=yolo_device,
        augment=False,
        stream=False,
    )
    best = None
    best_conf = -1.0
    for box in results[0].boxes:
        if int(box.cls[0].item()) != 0:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bw, bh = (x2 - x1) * inv, (y2 - y1) * inv
        area = bw * bh
        if area < 12 or area > 1600:
            continue
        aspect = bw / (bh + 1e-5)
        if not (0.3 < aspect < 3.0):
            continue
        conf = float(box.conf[0].item())
        if conf > best_conf:
            best_conf = conf
            cx = int((x1 + x2) / 2 * inv)
            cy = int((y1 + y2) / 2 * inv)
            best = (cx, cy, conf)
    return best


def segment_deliveries(
    cap,
    model,
    fps: float,
    width: int,
    height: int,
    total_frames: int,
    *,
    conf_thresh: float,
    use_half: bool,
    yolo_device,
    scan_imgsz: int = 640,
    scan_max_dim: int = 640,
    progress_cb=None,
) -> list[DeliveryClip]:
    """
    Light scan (stride=2) to find per-delivery frame ranges.
    Returns 1-based inclusive frame indices.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    lost_threshold = max(12, int(fps * CLIP_LOST_SEC))
    gap_min = min_gap_frames(fps)

    clips: list[DeliveryClip] = []
    current_start: int | None = None
    last_ball_frame = 0
    last_marker_frame = -9999
    frame_idx = 0
    scan_conf = max(conf_thresh, SCAN_CONF)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        should_scan = (frame_idx % CLIP_SCAN_STRIDE == 0) or current_start is not None
        ball = None
        if should_scan and model is not None:
            ball = detect_ball_light(
                model, frame, width, height,
                conf_thresh=scan_conf,
                imgsz=scan_imgsz,
                max_dim=scan_max_dim,
                use_half=use_half,
                yolo_device=yolo_device,
            )

        if ball is not None and current_start is None:
            gap_frames = frame_idx - last_ball_frame if last_ball_frame > 0 else 999
            if can_start_new_delivery(
                frame_idx, last_marker_frame, gap_frames, fps,
                ball[2], from_waiting=True,
            ):
                current_start = max(1, frame_idx - CLIP_START_PADDING)
                last_ball_frame = frame_idx

        if current_start is not None:
            if ball is not None:
                last_ball_frame = frame_idx
            frames_lost = frame_idx - last_ball_frame
            if frames_lost >= lost_threshold:
                end = min(total_frames or frame_idx, frame_idx + CLIP_END_PADDING)
                release = min(frame_idx, current_start + CLIP_START_PADDING)
                clips.append(DeliveryClip(
                    start=current_start,
                    end=end,
                    release_frame=release,
                ))
                last_marker_frame = end
                current_start = None

        if progress_cb and frame_idx % 25 == 0:
            pct = (frame_idx / total_frames * 30) if total_frames > 0 else 0
            progress_cb(pct, frame_idx, total_frames)

    if current_start is not None:
        end = min(total_frames or frame_idx, frame_idx + CLIP_END_PADDING)
        release = min(frame_idx, current_start + CLIP_START_PADDING)
        clips.append(DeliveryClip(start=current_start, end=end, release_frame=release))
        last_marker_frame = end

    # Merge overlapping clips (safety)
    merged: list[DeliveryClip] = []
    for clip in clips:
        if merged and clip.start <= merged[-1].end:
            prev = merged[-1]
            merged[-1] = DeliveryClip(
                start=prev.start,
                end=max(prev.end, clip.end),
                release_frame=prev.release_frame or clip.release_frame,
            )
        else:
            merged.append(clip)

    print(f"[ClipSeg] Pass 1 found {len(merged)} delivery clip(s)")
    for i, c in enumerate(merged, 1):
        print(f"  Clip {i}: frames {c.start}–{c.end} (release ~{c.release_frame})")
    return merged
