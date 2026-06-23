"""
Cricket Ball Tracker — Only Bounce Marker (No Trajectory)
=========================================================
- Trajectory lines removed as per user request.
- Detects the exact bounce (tip) frame.
- Draws a static visual marker at the bounce position on the video frame.
- Keeps tracking session logs for the 2D pitchmap intact.
"""

# --- Venv site-packages auto-inject (works with uv CPython 3.11 binary) ---
import sys as _sys, os as _os
_venv_pkgs = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '.venv', 'Lib', 'site-packages'))
if _os.path.isdir(_venv_pkgs) and _venv_pkgs not in _sys.path:
    _sys.path.insert(0, _venv_pkgs)
# ---------------------------------------------------------------------------

import math, os, threading, time, uuid, subprocess
import cv2, numpy as np
from collections import deque
from scipy.interpolate import splprep, splev
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
import queue
from werkzeug.utils import secure_filename
from pitch_map_renderer import (
    TEMPLATE_CORNERS,     create_hawkeye_template, render_pitch_map,
    draw_light_bounce_dots,
    auto_pitch_quad, paste_hawkeye_panel, paste_hawkeye_panel_centered,
    draw_summary_banner, build_panel_image, blit_panel, classify_length_zone,
    draw_colored_zones_on_video, draw_zone_labels_on_video,
    draw_distance_markers_on_video, draw_zone_boundary_lines_on_video, clear_template_cache,
    is_on_pitch, precompute_pitch_zone_layers, composite_pitch_zones, build_pitch_annotation_layer,
    PITCH_L, PITCH_R, PITCH_TOP, PITCH_BOT,
)
from accuracy_engine import (
    refine_bounce_point, classify_boundary, classify_miss, snap_to_pitch, find_hist_index_near,
)
from core.pitch_calibrator import calibrate_pitch_robust
from core.trajectory_physics import refine_bounce_world, interpolate_track_gaps
from core.hit_detector import score_hit_enhanced
from core.config import CONFIG
from core.pitch_coords import pitchmap_to_world
from core.classifier import classify_bounce
from core.gpu_runtime import init_gpu_runtime, load_yolo_model, infer_settings, ffmpeg_encode_args
from core.pose_estimator import BatsmanPoseEstimator
from core.delivery_filter import (
    can_start_new_delivery, should_register_marker, is_valid_delivery_track,
    MIN_NEW_DET_CONF, min_gap_frames,
)
from core.delivery_segmenter import segment_deliveries
from core.speed_calibrator import compute_bowling_speed_kmh, session_speed_stats

_TRACK = CONFIG.get('tracking', {})
COAST_SECONDS = float(_TRACK.get('coast_seconds', 0.45))


def _coast_limit_frames(fps: float) -> int:
    return max(12, int(fps * COAST_SECONDS))

API_VERSION = 'pitchmap-v36-accuracy-upgrade'
_GPU_CFG = CONFIG.get('gpu', {})

# Overridden at runtime from config.gpu
DETECT_STRIDE_WAITING = int(_GPU_CFG.get('waiting_stride', 3))
DETECT_STRIDE_COAST = 1
INFER_MAX_DIM = int(_GPU_CFG.get('waiting_max_dim', 640))
INFER_MAX_DIM_ACTIVE = int(_GPU_CFG.get('active_max_dim', 960))
MIN_TRACK_FRAMES = 3
SUMMARY_SEC = float(_GPU_CFG.get('summary_seconds', 1))
ENABLE_POSE = bool(_GPU_CFG.get('enable_pose', False))
CALIB_SAMPLES = int(_GPU_CFG.get('calibration_samples', 15))
_VIZ = CONFIG.get('visualization', {})
SHOW_CORNER_PITCH_MAP = bool(_VIZ.get('show_corner_pitch_map', False))
SHOW_SUMMARY_PITCH_MAP = bool(_VIZ.get('show_summary_pitch_map', False))
SHOW_PITCH_ZONE_OVERLAY = bool(_VIZ.get('show_pitch_zone_overlay', True))
_PROC = CONFIG.get('processing', {})
CLIP_BY_CLIP = bool(_PROC.get('clip_by_clip', False))
CLIP_MODE = str(_PROC.get('clip_mode', 'stream'))  # stream | clip
STREAM_DETECT_EVERY_FRAME = bool(_PROC.get('stream_detect_every_frame', True))
CLIP_DEEP_IMGSZ = int(_PROC.get('clip_deep_imgsz', 1280))

_yolo_model = None
_yolo_half = False
_yolo_device = 'cpu'
_gpu_name = None

def _get_yolo():
    global _yolo_model, _yolo_half, _yolo_device, _gpu_name
    if _yolo_model is None:
        if _GPU_CFG.get('require_cuda', True):
            init_gpu_runtime()
        _yolo_model, _yolo_half, _yolo_device, _gpu_name = load_yolo_model(MODEL_PATH)
        print(f"[YOLO] device={_yolo_device} half={_yolo_half} gpu={_gpu_name or 'none'}")
    return _yolo_model, _yolo_half, _yolo_device

def _set_job_progress(job_id, pct, frame_idx=0, total=0, pass_info=None):
    if not job_id:
        return
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]['progress'] = round(pct, 1)
            jobs[job_id]['frame'] = frame_idx
            jobs[job_id]['total_frames'] = total
            if pass_info is not None:
                jobs[job_id]['pass_info'] = pass_info

app = Flask(__name__)
CORS(app)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = CONFIG['model']['path']
POSE_ESTIMATOR = BatsmanPoseEstimator() if ENABLE_POSE else None

jobs      = {}
jobs_lock = threading.Lock()
job_queue = queue.Queue()
_current_job_id = None


def _drain_queue_except(job_id):
    """Drop stale queued jobs so new uploads start immediately."""
    drained = []
    while True:
        try:
            item = job_queue.get_nowait()
            drained.append(item)
            job_queue.task_done()
        except queue.Empty:
            break
    kept = None
    for item in drained:
        old_id = item[0]
        if old_id == job_id:
            kept = item
        else:
            with jobs_lock:
                if old_id in jobs and jobs[old_id]['status'] == 'queued':
                    jobs[old_id]['status'] = 'cancelled'
                    jobs[old_id]['error'] = 'Superseded by newer upload'
    if kept:
        job_queue.put(kept)


# --- PERSISTENT PITCHMAP CONFIGURATION ---
session_bounces = []

# TODO: Calibrate these 4 points based on your camera feed layout
camera_perspective_points = np.array([
    [250, 400],  # Top Left
    [390, 400],  # Top Right
    [100, 700],  # Bottom Left
    [540, 700]   # Bottom Right
], dtype=np.float32)

template_2d_points = TEMPLATE_CORNERS.copy()

H_MATRIX = cv2.getPerspectiveTransform(camera_perspective_points, template_2d_points)
H_INV    = cv2.getPerspectiveTransform(template_2d_points, camera_perspective_points)

def transform_to_pitchmap(cam_x, cam_y, h_matrix=None):
    H = h_matrix if h_matrix is not None else H_MATRIX
    point = np.array([[[cam_x, cam_y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, H)
    return int(transformed[0, 0, 0]), int(transformed[0, 0, 1])

def video_processing_worker():
    global _current_job_id
    while True:
        item = job_queue.get()
        if len(item) == 3:
            job_id, input_path, output_path = item
            options = {}
        else:
            job_id, input_path, output_path, options = item
        with jobs_lock:
            job = jobs.get(job_id)
            if not job or job.get('status') == 'cancelled':
                job_queue.task_done()
                continue
            jobs[job_id]['status'] = 'processing'
        _current_job_id = job_id
        try:
            process_video_async(job_id, input_path, output_path, options=options)
        finally:
            _current_job_id = None
            job_queue.task_done()
threading.Thread(target=video_processing_worker, daemon=True).start()

# ---------- Kalman Filter (tuned for fast ball motion) ----------
class BallKalmanFilter:
    def __init__(self, dt=1.0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        q = np.eye(4, dtype=np.float32)
        q[0,0] = q[1,1] = 3e-1
        q[2,2] = q[3,3] = 8e-1
        self.kf.processNoiseCov = q
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5e-2
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def init(self, x, y):
        self.kf.statePost = np.array([[x],[y],[0],[0]], np.float32)
        self.initialized = True
    def predict(self): return self.kf.predict()
    def correct(self, x, y): self.kf.correct(np.array([[np.float32(x)],[np.float32(y)]]))
    def get_position(self): return (int(self.kf.statePost[0, 0]), int(self.kf.statePost[1, 0]))
    def get_velocity(self):
        return (float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0]))

# ---------- Smoothed History ----------
class SmoothHistory:
    def __init__(self, maxlen=150, smooth_window=5):
        self.queue = deque(maxlen=maxlen)
        self.smooth_window = smooth_window
        self.raw_buffer = deque(maxlen=smooth_window)

    def add(self, raw_point):
        self.raw_buffer.append(raw_point)
        if len(self.raw_buffer) < self.smooth_window:
            smoothed = raw_point
        else:
            weights = np.exp(np.linspace(-1, 0, self.smooth_window))
            weights /= weights.sum()
            pts = np.array(self.raw_buffer)
            smoothed_x = np.dot(pts[:,0], weights)
            smoothed_y = np.dot(pts[:,1], weights)
            smoothed = (int(smoothed_x), int(smoothed_y))
        if len(self.queue) > 0 and self.queue[-1] == smoothed:
            return
        self.queue.append(smoothed)

    def get_list(self): return list(self.queue)
    def clear(self): self.queue.clear(); self.raw_buffer.clear()
    def __len__(self): return len(self.queue)

# ---------- Drawing helpers ----------
def draw_ui_panel(img, title, value, top_left, size=(160,55), value_color=(255,255,255)):
    overlay = img.copy()
    x,y = top_left; w,h = size
    cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,0,0),-1)
    cv2.rectangle(overlay,(x,y),(x+w,y+h),(255,255,255),1)
    cv2.addWeighted(overlay,0.6,img,0.4,0,img)
    cv2.putText(img,title,(x+8,y+18),cv2.FONT_HERSHEY_SIMPLEX,0.42,(200,200,200),1,cv2.LINE_AA)
    cv2.putText(img,value,(x+8,y+44),cv2.FONT_HERSHEY_SIMPLEX,0.65,value_color,2,cv2.LINE_AA)

def bounce_stats(bounces):
    """Count DOT (no run), RUN (hit), 4/6, OUT from tracked bounces."""
    def lbl(b):
        return b.get('type') or b.get('label', 'DOTS')
    dots = sum(1 for b in bounces if lbl(b) == 'DOTS')
    runs = sum(1 for b in bounces if lbl(b) == 'RUNS')
    boundaries = sum(1 for b in bounces if lbl(b) == 'BOUNDARIES')
    wickets = sum(1 for b in bounces if lbl(b) == 'WICKETS')
    return {
        'total': len(bounces),
        'dots': dots,
        'runs': runs,
        'boundaries': boundaries,
        'wickets': wickets,
    }

def draw_ball_stats_panels(frame, stats):
    """Show TOTAL / DOT (not hit) / RUN (hit) on video."""
    draw_ui_panel(frame, 'BALLS', str(stats['total']), (15, 75), size=(100, 48))
    draw_ui_panel(frame, 'DOT', str(stats['dots']), (125, 75), size=(100, 48), value_color=(80, 220, 80))
    draw_ui_panel(frame, 'RUN', str(stats['runs'] + stats['boundaries']), (235, 75), size=(100, 48), value_color=(80, 80, 255))
    if stats['wickets']:
        draw_ui_panel(frame, 'OUT', str(stats['wickets']), (345, 75), size=(100, 48), value_color=(200, 200, 255))


def _estimate_landing_point(raw_pts, height):
    """Best guess of where the ball finished — bounce apex or lowest tracked point."""
    if not raw_pts:
        return None
    bounce = refine_bounce_point(raw_pts, height, lookback=len(raw_pts))
    if bounce is not None:
        return bounce
    if len(raw_pts) >= 3:
        for i in range(1, len(raw_pts) - 1):
            dy_before = raw_pts[i][1] - raw_pts[i - 1][1]
            dy_after = raw_pts[i + 1][1] - raw_pts[i][1]
            if dy_before > 0.8 and dy_after < -0.8:
                return raw_pts[i]
    lower = [p for p in raw_pts if p[1] > height * 0.30]
    if lower:
        return max(lower, key=lambda p: p[1])
    return raw_pts[-1]


def _register_dot_from_track(raw_pts, height, h_matrix, frame_index, job_bounces,
                             session_bounces, persistent_video_bounces, fps,
                             last_marker_frame, last_detection_conf=0.5):
    """Fallback DOT marker when bounce apex wasn't found but track is valid."""
    if not is_valid_delivery_track(raw_pts, height, fps, strict=False):
        return last_marker_frame, False
    landing = _estimate_landing_point(raw_pts, height)
    if landing is None:
        return last_marker_frame, False
    bx, by = landing
    speed = _compute_delivery_speed(raw_pts, fps, height, h_matrix)
    result = _add_delivery_marker(
        bx, by, h_matrix, 'DOTS', frame_index,
        job_bounces, session_bounces, persistent_video_bounces,
        speed_kmh=speed, detection_conf=last_detection_conf,
        raw_pts=raw_pts, height=height, fps=fps,
        last_marker_frame=last_marker_frame, strict=False)
    if result[0] is not None:
        print(f"[Frame {frame_index}] DOT fallback @ ({bx},{by}) | {result[0]} | {speed:.0f} km/h")
        return result[1], True
    return last_marker_frame, False


_CURRENT_STUMP_SCALE = 1.0


def _set_processing_context(stump_scale: float = 1.0):
    global _CURRENT_STUMP_SCALE
    _CURRENT_STUMP_SCALE = stump_scale


def _compute_delivery_speed(raw_pts, fps, height, h_matrix=None, stump_scale=None):
    """Calibrated bowling speed (km/h) — homography-based, realistic 70–180 range."""
    scale = stump_scale if stump_scale is not None else _CURRENT_STUMP_SCALE
    return compute_bowling_speed_kmh(
        raw_pts, fps, h_matrix=h_matrix, height=height, stump_scale=scale,
    )


def _add_delivery_marker(bx, by, h_matrix, label, frame_index,
                         job_bounces, session_bounces, persistent_video_bounces,
                         speed_kmh=0.0, detection_conf=0.9, tracking_conf=0.9,
                         full_toss=False, pose_data=None, raw_pts=None, height=0, fps=25.0,
                         last_marker_frame=0, strict=False):
    """Register one delivery outcome — skipped if track fails validation."""
    if raw_pts is not None and not should_register_marker(
            raw_pts, height, fps, frame_index, last_marker_frame, strict=strict):
        print(f"[Frame {frame_index}] SKIP marker — invalid/false track")
        return None, last_marker_frame
    px_map, py_map = transform_to_pitchmap(bx, by, h_matrix)
    px_map, py_map = snap_to_pitch(px_map, py_map, PITCH_L, PITCH_R, PITCH_TOP, PITCH_BOT)
    x_m, y_m = pitchmap_to_world(px_map, py_map)
    classification = classify_bounce(
        x_m, y_m,
        detection_confidence=detection_conf,
        tracking_confidence=tracking_conf,
        full_toss=full_toss,
    )
    length_zone = classification.length_legacy
    speed_kmh = round(float(speed_kmh or 0), 0)
    bounce_entry = {
        'coords': (px_map, py_map),
        'type': label,
        'length': length_zone,
        'length_type': classification.length_type,
        'line_type': classification.line_type,
        'bounce_x': classification.bounce_x,
        'bounce_y': classification.bounce_y,
        'bounce_confidence': classification.confidence,
        'detection_confidence': detection_conf,
        'tracking_confidence': tracking_conf,
        'frame': frame_index,
        'speed_kmh': speed_kmh,
        'pitch_map_x': px_map,
        'pitch_map_y': py_map,
        'pose_data': pose_data,
    }
    job_bounces.append(bounce_entry)
    session_bounces.append({
        'coords': (px_map, py_map), 'type': label,
        'length': length_zone, 'length_type': classification.length_type,
        'line_type': classification.line_type,
        'bounce_x': classification.bounce_x, 'bounce_y': classification.bounce_y,
        'speed_kmh': speed_kmh,
    })
    persistent_video_bounces.append({
        'coords': (int(bx), int(by)), 'label': label, 'length': length_zone,
        'length_type': classification.length_type, 'line_type': classification.line_type,
        'speed_kmh': speed_kmh,
    })
    return length_zone, frame_index


def _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces, label):
    if session_bounces:
        session_bounces[-1]['type'] = label
    if job_bounces:
        job_bounces[-1]['type'] = label
    if persistent_video_bounces:
        persistent_video_bounces[-1]['label'] = label


def _close_delivery(raw_pts, height, h_matrix, frame_index, hit_occurred, event_status,
                    bounced_this_delivery, job_bounces, session_bounces, persistent_video_bounces,
                    fps=25.0, last_marker_frame=0):
    """Finalize delivery — register DOT/RUN if no bounce marker was placed yet."""
    if bounced_this_delivery:
        return last_marker_frame
    if not is_valid_delivery_track(raw_pts, height, fps, strict=False):
        return last_marker_frame

    landing = _estimate_landing_point(raw_pts, height)
    if landing is None:
        return last_marker_frame

    label = 'RUNS' if hit_occurred else 'DOTS'
    if event_status == 'MISS':
        label = 'WICKETS'

    bx, by = landing
    speed = _compute_delivery_speed(raw_pts, fps, height, h_matrix)
    result = _add_delivery_marker(
        bx, by, h_matrix, label, frame_index,
        job_bounces, session_bounces, persistent_video_bounces, speed_kmh=speed,
        raw_pts=raw_pts, height=height, fps=fps, last_marker_frame=last_marker_frame,
        strict=False)
    if result[0] is not None:
        zone = result[0]
        print(f"[Frame {frame_index}] DELIVERY closed @ ({bx},{by}) | {label} | {zone} | {speed:.0f} km/h")
        return result[1]
    return last_marker_frame


def _finalize_delivery(raw_pts, height, h_matrix, frame_index, hit_occurred, event_status,
                       bounced_this_delivery, job_bounces, session_bounces,
                       persistent_video_bounces, fps, last_marker_frame, last_detection_conf=0.5):
    """
    End one delivery — register DOT if no bounce yet, then close.
    Ensures every tracked ball gets a RUN or DOT marker when possible.
    """
    bounced = bounced_this_delivery
    if not bounced and not hit_occurred and len(raw_pts) >= 3:
        last_marker_frame, registered = _register_dot_from_track(
            raw_pts, height, h_matrix, frame_index, job_bounces, session_bounces,
            persistent_video_bounces, fps, last_marker_frame, last_detection_conf)
        if registered:
            bounced = True
    last_marker_frame = _close_delivery(
        raw_pts, height, h_matrix, frame_index, hit_occurred, event_status, bounced,
        job_bounces, session_bounces, persistent_video_bounces,
        fps=fps, last_marker_frame=last_marker_frame)
    return last_marker_frame


def _reset_delivery_state(kf, dt, history, raw_history):
    """Clear per-delivery trackers for the next ball."""
    history.clear()
    raw_history.clear()
    kf_new = BallKalmanFilter(dt=dt)
    return kf_new, {
        'hit_occurred': False,
        'event_status': 'WAITING',
        'bounce_detected': False,
        'bounce_frame': -1,
        'bounced_this_delivery': False,
        'frames_since_bounce': 999,
        'bounce_hist_idx': None,
        'pre_hit_speed': 0.0,
        'post_hit_max_speed': 0.0,
        'delivery_pose_frames': [],
    }


def _refine_bounce(raw_pts, height, h_matrix):
    """World-coords bounce with pixel-Y fallback."""
    if h_matrix is not None:
        pt = refine_bounce_world(raw_pts, h_matrix, height)
        if pt is not None:
            return pt
    return refine_bounce_point(raw_pts, height)


def _try_detect_bounce(raw_pts, frame_index, last_bounce_frame, fps, h_matrix, persistent_video_bounces, height, width=0):
    """Pitch bounce — apex in trajectory, filtered to pitch area."""
    if len(raw_pts) < 5:
        return None
    if frame_index - last_bounce_frame <= max(6, int(fps * 0.25)):
        return None

    filled = interpolate_track_gaps(raw_pts, fps)
    bounce_pt = _refine_bounce(filled, height, h_matrix)
    if bounce_pt is None:
        return None
    bx, by = bounce_pt
    near_tail = min(math.hypot(bx - p[0], by - p[1]) for p in raw_pts[-8:])
    if near_tail > 45:
        return None

    on_pitch = is_on_pitch(bx, by, h_matrix, margin=50)
    if not on_pitch and width > 0:
        # Fallback: lower-middle corridor on video when homography is imperfect
        in_corridor = (width * 0.15 < bx < width * 0.85 and height * 0.30 < by < height * 0.88)

        
        if not in_corridor:
            return None
    elif not on_pitch:
        return None

    too_close = any(
        math.hypot(bx - b['coords'][0], by - b['coords'][1]) < 22
        for b in persistent_video_bounces)
    if too_close:
        return None
    return (bx, by)


def _clips_to_dicts(clips) -> list[dict]:
    """Serialize DeliveryClip objects or dicts for API/PDF."""
    out = []
    for i, c in enumerate(clips, 1):
        if hasattr(c, 'start'):
            out.append({
                'index': i,
                'start': c.start,
                'end': c.end,
                'release_frame': c.release_frame,
            })
        else:
            out.append(dict(c))
    return out


def _link_bounces_to_clips(bounce_events: list[dict], clips: list[dict]) -> None:
    """Attach clip_index / clip range to each bounce event."""
    for be in bounce_events:
        frame = be.get('frame', 0)
        for clip in clips:
            if clip.get('start', 0) <= frame <= clip.get('end', 0):
                be['clip_index'] = clip.get('index')
                be['clip_start'] = clip.get('start')
                be['clip_end'] = clip.get('end')
                break


def _enrich_clips_with_bounces(clips: list[dict], bounce_events: list[dict], fps: float) -> list[dict]:
    """Merge bounce outcomes into clip summary rows."""
    enriched = []
    for clip in clips:
        row = dict(clip)
        frame = None
        matched = [
            b for b in bounce_events
            if clip.get('start', 0) <= b.get('frame', 0) <= clip.get('end', 0)
        ]
        if matched:
            b = matched[0]
            frame = b.get('frame')
            row.update({
                'bounce_frame': frame,
                'outcome': b.get('type', 'DOTS'),
                'length': b.get('length'),
                'speed_kmh': b.get('speed_kmh', 0),
                'bounce_confidence': b.get('bounce_confidence'),
            })
        else:
            row.setdefault('outcome', 'NO_MARKER')
            row.setdefault('bounce_frame', None)
        row['start_time'] = round(clip.get('start', 0) / fps, 2) if fps else 0
        row['end_time'] = round(clip.get('end', 0) / fps, 2) if fps else 0
        enriched.append(row)
    return enriched


def _synthetic_clips_from_bounces(bounce_events: list[dict], total_frames: int, fps: float) -> list[dict]:
    """Build clip ranges from bounce frames when streaming mode was used."""
    pre = max(10, int(fps * 2))
    post = max(10, int(fps * 1.5))
    clips = []
    for i, b in enumerate(bounce_events, 1):
        f = int(b.get('frame') or 0)
        clips.append({
            'index': i,
            'start': max(1, f - pre),
            'end': min(total_frames or f + post, f + post),
            'release_frame': max(1, f - int(fps * 1.2)),
            'bounce_frame': f,
            'outcome': b.get('type', 'DOTS'),
            'length': b.get('length'),
            'speed_kmh': b.get('speed_kmh', 0),
        })
    return clips

def _yolo_detect_ball(model, frame, width, height, conf_thresh, detect_imgsz, infer_scale_run,
                      use_half, yolo_device, kf=None, frames_since_det=0, coast_limit=18,
                      last_velocity=0.0):
    """YOLO ball detection with optional Kalman proximity scoring."""
    best_coords = None
    best_score = -float('inf')
    last_conf = 0.0
    infer_frame = frame
    if infer_scale_run < 1.0:
        infer_frame = cv2.resize(
            frame,
            (int(width * infer_scale_run), int(height * infer_scale_run)),
            interpolation=cv2.INTER_LINEAR,
        )
    inv = 1.0 / infer_scale_run
    results = model.predict(
        infer_frame, conf=conf_thresh, imgsz=detect_imgsz, max_det=10,
        verbose=False, half=use_half, device=yolo_device,
        augment=False, stream=False,
    )
    for box in results[0].boxes:
        if int(box.cls[0].item()) != 0:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bw, bh = (x2 - x1) * inv, (y2 - y1) * inv
        area = bw * bh
        if area < 12 or area > 15000:
            continue
        aspect = bw / (bh + 1e-5)
        if not (0.15 < aspect < 5.0):
            continue
        cx, cy = int((x1 + x2) / 2 * inv), int((y1 + y2) / 2 * inv)
        score = float(box.conf[0].item())
        if kf is not None and kf.initialized and frames_since_det <= coast_limit:
            px, py = kf.get_position()
            dist = math.hypot(cx - px, cy - py)
            vx, vy = kf.get_velocity()
            speed_est = max(last_velocity, math.hypot(vx, vy), 30.0)
            max_dist = max(280, width * 0.45, speed_est * frames_since_det * 5.5)
            if dist > max_dist:
                continue
            score -= dist * 0.003
        if score > best_score:
            best_score = score
            best_coords = (cx, cy)
            last_conf = float(box.conf[0].item())
    return best_coords, last_conf


def track_delivery_clip(cap, clip, model, fps, width, height, h_matrix, dt,
                        use_half, yolo_device, job_bounces, session_bounces,
                        persistent_video_bounces, last_marker_frame, clip_index=1,
                        stump_scale=1.0):
    """
    Pass 2 — high-accuracy tracking on one delivery clip.
    Fresh Kalman per clip, stride=1 on every frame, larger imgsz.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, clip.start - 1))
    coast_limit = _coast_limit_frames(fps)
    detect_imgsz = min(1280, max(CLIP_DEEP_IMGSZ, max(width, height)))
    infer_scale_run = min(1.0, INFER_MAX_DIM_ACTIVE / max(width, height, 1))
    conf_thresh = CONFIG['model']['confidence']

    kf = BallKalmanFilter(dt=dt)
    history = SmoothHistory(maxlen=200, smooth_window=3)
    raw_history = deque(maxlen=200)
    frames_since_det = 999
    last_velocity = 0.0
    event_status = "BOWLED"
    hit_occurred = False
    last_bounce_frame = -999
    bounced_this_delivery = False
    frames_since_bounce = 999
    last_detection_conf = 0.9
    post_hit_max_speed = 0.0
    pre_hit_speed = 0.0
    delivery_pose_frames = []
    bounce_hist_idx = None
    frame_index = clip.start - 1

    while frame_index < clip.end:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        gap_frames = frames_since_det + 1
        frames_since_det += 1
        best_coords = None
        is_predicted = False

        if model is not None:
            best_coords, last_detection_conf = _yolo_detect_ball(
                model, frame, width, height, conf_thresh, detect_imgsz, infer_scale_run,
                use_half, yolo_device, kf=kf, frames_since_det=frames_since_det,
                coast_limit=coast_limit, last_velocity=last_velocity,
            )

        if best_coords is not None:
            cx, cy = best_coords
            if not kf.initialized:
                kf.init(cx, cy)
                event_status = "BOWLED"
            else:
                kf.predict()
                kf.correct(cx, cy)

            if len(raw_history) > 0:
                lx, ly = raw_history[-1]
                step = math.hypot(cx - lx, cy - ly)
                last_velocity = step
                vx, vy = kf.get_velocity()
                speed_est = max(step, math.hypot(vx, vy))
                jump_limit = max(600, width * 0.65, speed_est * 8.0)
                if step > jump_limit and gap_frames <= int(fps * 0.30):
                    history.clear()
                    raw_history.clear()
                    kf = BallKalmanFilter(dt=dt)
                    kf.init(cx, cy)
                    hit_occurred = False
                    event_status = "BOWLED"
                    bounced_this_delivery = False
                    frames_since_bounce = 999
                    pre_hit_speed = 0.0
                    post_hit_max_speed = 0.0

            history.add((cx, cy))
            raw_history.append((cx, cy))
            frames_since_det = 0
        elif kf.initialized and frames_since_det <= coast_limit:
            kf.predict()
            px, py = kf.get_position()
            best_coords = (px, py)
            is_predicted = True
            history.add((px, py))
            raw_history.append((px, py))
            vx, vy = kf.get_velocity()
            last_velocity = max(last_velocity, math.hypot(vx, vy))
        elif kf.initialized:
            kf.predict()

        raw_list = interpolate_track_gaps(list(raw_history), fps)
        hist = history.get_list()

        pose_samples = None
        if ENABLE_POSE and POSE_ESTIMATOR and delivery_pose_frames:
            pose_samples = POSE_ESTIMATOR.sample_delivery_poses(delivery_pose_frames, max_samples=3)

        # Get dynamic batsman zone to pass to classify_miss
        from core.hit_detector import batsman_zone_from_pose
        _, y_max_val = batsman_zone_from_pose(pose_samples, height)

        if ENABLE_POSE and event_status in ("BOWLED", "POST_HIT") and len(delivery_pose_frames) < 4:
            if frame_index % 4 == 0:
                delivery_pose_frames.append((frame_index, frame))

        if bounced_this_delivery:
            frames_since_bounce += 1

        if event_status == "BOWLED" and not bounced_this_delivery and not hit_occurred:
            bounce_pt = _try_detect_bounce(
                raw_list, frame_index, last_bounce_frame, fps, h_matrix,
                persistent_video_bounces, height, width=width)
            if bounce_pt is not None:
                bx, by = bounce_pt
                bounced_this_delivery = True
                frames_since_bounce = 0
                bounce_hist_idx = find_hist_index_near(hist, (bx, by))
                speed = _compute_delivery_speed(raw_list, fps, height, h_matrix)
                tracking_conf = 0.95 if not is_predicted else 0.6
                if ENABLE_POSE and POSE_ESTIMATOR and not pose_samples:
                    pose_samples = POSE_ESTIMATOR.sample_delivery_poses(delivery_pose_frames, max_samples=2)
                result = _add_delivery_marker(
                    bx, by, h_matrix, 'DOTS', frame_index,
                    job_bounces, session_bounces, persistent_video_bounces,
                    speed_kmh=speed, detection_conf=last_detection_conf,
                    tracking_conf=tracking_conf, pose_data=pose_samples or None,
                    raw_pts=raw_list, height=height, fps=fps,
                    last_marker_frame=last_marker_frame, strict=False)
                if result[0] is not None:
                    last_marker_frame = result[1]
                    last_bounce_frame = frame_index
                    print(f"[Clip {clip.start}-{clip.end}] BOUNCE @ frame {frame_index} | {result[0]} | {speed:.0f} km/h")
                else:
                    bounced_this_delivery = False

        if event_status in ("BOWLED", "MISS") and not hit_occurred:
            is_hit, hit_conf, contact = score_hit_enhanced(
                raw_list, hist, height, fps, bounced_this_delivery, frames_since_bounce,
                bounce_hist_idx=bounce_hist_idx, pose_frames=pose_samples)
            if is_hit:
                hit_occurred = True
                event_status = "POST_HIT"
                pre_hit_speed = math.hypot(hist[-2][0] - hist[-3][0], hist[-2][1] - hist[-3][1]) if len(hist) >= 3 else 0.0
                post_hit_max_speed = math.hypot(hist[-1][0] - hist[-2][0], hist[-1][1] - hist[-2][1])
                if bounced_this_delivery and job_bounces:
                    _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces, 'RUNS')
                else:
                    hp = contact or (hist[-3] if len(hist) >= 3 else hist[-1])
                    speed = _compute_delivery_speed(raw_list, fps, height, h_matrix)
                    result = _add_delivery_marker(
                        hp[0], hp[1], h_matrix, 'RUNS', frame_index,
                        job_bounces, session_bounces, persistent_video_bounces,
                        speed_kmh=speed, raw_pts=raw_list, height=height, fps=fps,
                        last_marker_frame=last_marker_frame, strict=False)
                    if result[0] is not None:
                        last_marker_frame = result[1]
                        bounced_this_delivery = True
                print(f"[Clip {clip.start}-{clip.end}] HIT @ frame {frame_index} (conf={hit_conf:.2f})")

        if event_status == "POST_HIT" and len(hist) >= 2:
            speed = math.hypot(hist[-1][0] - hist[-2][0], hist[-1][1] - hist[-2][1])
            post_hit_max_speed = max(post_hit_max_speed, speed)
            if classify_boundary(hist, post_hit_max_speed, height, pre_hit_speed):
                if persistent_video_bounces and persistent_video_bounces[-1]['label'] == 'RUNS':
                    _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces, 'BOUNDARIES')

        if event_status == "BOWLED" and bounced_this_delivery and classify_miss(hist, height, hit_occurred, bounced_this_delivery, batsman_y_max=y_max_val):
            event_status = "MISS"
            _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces, 'WICKETS')

    if not bounced_this_delivery and not hit_occurred and len(raw_history) >= 4:
        last_marker_frame, registered = _register_dot_from_track(
            list(raw_history), height, h_matrix, min(frame_index, clip.end),
            job_bounces, session_bounces, persistent_video_bounces, fps,
            last_marker_frame, last_detection_conf)
        if registered:
            bounced_this_delivery = True

    last_marker_frame = _close_delivery(
        list(raw_history), height, h_matrix, min(frame_index, clip.end),
        hit_occurred, event_status, bounced_this_delivery,
        job_bounces, session_bounces, persistent_video_bounces,
        fps=fps, last_marker_frame=last_marker_frame)

    clip_bounces = [
        b for b in job_bounces
        if clip.start <= b.get('frame', 0) <= clip.end
    ]
    clip_result = {
        'index': clip_index,
        'start': clip.start,
        'end': clip.end,
        'release_frame': clip.release_frame,
        'track_frames': len(raw_history),
        'bounce_frame': clip_bounces[0].get('frame') if clip_bounces else None,
        'outcome': clip_bounces[0].get('type', 'NO_MARKER') if clip_bounces else 'NO_MARKER',
        'length': clip_bounces[0].get('length') if clip_bounces else None,
        'speed_kmh': clip_bounces[0].get('speed_kmh', 0) if clip_bounces else 0,
    }
    return last_marker_frame, clip_result


def _clip_status_at_frame(frame_index, clips, job_bounces):
    """UI status for render pass — which clip is active at this frame."""
    for clip in clips:
        if clip.start <= frame_index <= clip.end:
            for b in job_bounces:
                if clip.start <= b.get('frame', 0) <= frame_index:
                    lbl = b.get('type', 'DOTS')
                    if lbl == 'RUNS':
                        return "POST_HIT"
                    if lbl == 'WICKETS':
                        return "MISS"
            return "BOWLED"
    return "WAITING"


def _render_video_clip_mode(cap, output_path, fps, width, height, total_frames, h_matrix, h_inv,
                            job_bounces, persistent_video_bounces, clips, job_id,
                            zone_color, zone_mask, pitch_annotations, pitch_alpha):
    """Pass 3 — render full video with markers from clip tracking results."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    panel_w = min(480, max(320, int(width * 0.42)))
    cached_live_panel = None
    cached_bounce_count = -1
    empty_panel = build_panel_image([], 'PITCH MAP', panel_w) if SHOW_CORNER_PITCH_MAP else None

    sorted_bounces = sorted(job_bounces, key=lambda b: b.get('frame', 0))
    visible_job = []
    bounce_idx = 0
    visible_persistent = []
    frame_index = 0
    frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        while bounce_idx < len(sorted_bounces) and sorted_bounces[bounce_idx].get('frame', 0) <= frame_index:
            visible_job.append(sorted_bounces[bounce_idx])
            vb = persistent_video_bounces[bounce_idx] if bounce_idx < len(persistent_video_bounces) else None
            if vb:
                visible_persistent.append(vb)
            bounce_idx += 1

        if SHOW_CORNER_PITCH_MAP:
            if len(visible_job) != cached_bounce_count:
                cached_live_panel = build_panel_image(
                    [{'coords': b['coords'], 'type': b['type'],
                      'length': b.get('length'), 'speed_kmh': b.get('speed_kmh', 0)} for b in visible_job],
                    'PITCH MAP', panel_w)
                cached_bounce_count = len(visible_job)

        if SHOW_PITCH_ZONE_OVERLAY:
            composite_pitch_zones(frame, zone_color, zone_mask, alpha=pitch_alpha)
            nz = pitch_annotations.max(axis=2) > 0
            frame[nz] = pitch_annotations[nz]

        if SHOW_CORNER_PITCH_MAP:
            if cached_live_panel is not None:
                blit_panel(frame, cached_live_panel)
            elif empty_panel is not None:
                blit_panel(frame, empty_panel)
        if visible_persistent:
            draw_light_bounce_dots(frame, visible_persistent, use_video_coords=True, H_matrix=h_matrix)

        event_status = _clip_status_at_frame(frame_index, clips, visible_job)
        stats = bounce_stats(visible_persistent)
        status_color = {"WAITING": (128, 128, 128), "BOWLED": (0, 200, 255),
                        "POST_HIT": (0, 255, 128), "MISS": (0, 80, 255)}.get(event_status, (255, 255, 255))
        draw_ui_panel(frame, "STATUS", event_status, (15, 15), value_color=status_color)
        draw_ui_panel(frame, "BALLS", str(stats['total']), (185, 15))
        draw_ui_panel(frame, "FRAME", str(frame_index), (355, 15))
        draw_ball_stats_panels(frame, stats)

        writer.write(frame)
        if frame_index % 25 == 0:
            pct = 70 + (frame_index / total_frames * 25) if total_frames > 0 else 70
            _set_job_progress(job_id, pct, frame_index, total_frames)

    _set_job_progress(job_id, 95, frame_index, total_frames)

    map_panel_bounces = [{'coords': b['coords'], 'type': b.get('type', 'DOTS'),
                          'length': b.get('length'), 'speed_kmh': b.get('speed_kmh', 0)} for b in job_bounces]
    if not map_panel_bounces:
        map_panel_bounces = [{'coords': (360, 400), 'type': 'DOTS'}]

    summary_count = 0
    if SHOW_SUMMARY_PITCH_MAP and frame is not None:
        end_panel = build_panel_image(map_panel_bounces, 'PITCH MAP', min(440, width // 2))
        for _ in range(int(fps * SUMMARY_SEC)):
            summary = frame.copy()
            summary = draw_summary_banner(summary, 'PITCH MAP - ALL TRACKED DELIVERIES')
            draw_colored_zones_on_video(summary, h_inv, alpha=0.55)
            draw_zone_boundary_lines_on_video(summary, h_inv)
            draw_distance_markers_on_video(summary, h_inv)
            draw_zone_labels_on_video(summary, h_inv, map_panel_bounces)
            draw_light_bounce_dots(summary, persistent_video_bounces, use_video_coords=True, H_matrix=h_matrix)
            draw_ball_stats_panels(summary, bounce_stats(persistent_video_bounces))
            paste_hawkeye_panel_centered(summary, map_panel_bounces, title='PITCH MAP', panel_img=end_panel)
            writer.write(summary)
            summary_count += 1
    print(f"[ClipRender] {summary_count} summary frames, {len(job_bounces)} bounces")
    writer.release()
    return frame_index


def _finalize_video_output(output_path, job_bounces, frame_index, hit_occurred, event_status,
                           *, fps=25.0, clips=None, processing_mode='clip',
                           job_id=None, video_name='', calibration_meta=None):
    """FFmpeg encode + build return dict + optional PDF report."""
    converted = output_path.replace('.mp4', '_converted.mp4')
    cmd = ffmpeg_encode_args(output_path, converted)
    subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if os.path.exists(converted):
        os.remove(output_path)
        os.replace(converted, output_path)

    ball_stats = bounce_stats(job_bounces)
    bounce_events = [
        {
            'coords': list(b['coords']), 'type': b['type'],
            'length': b.get('length'), 'length_type': b.get('length_type'),
            'line_type': b.get('line_type'),
            'bounce_x': b.get('bounce_x'), 'bounce_y': b.get('bounce_y'),
            'bounce_confidence': b.get('bounce_confidence'),
            'detection_confidence': b.get('detection_confidence'),
            'tracking_confidence': b.get('tracking_confidence'),
            'speed_kmh': b.get('speed_kmh', 0), 'frame': b.get('frame', 0),
            'pitch_map_x': b.get('pitch_map_x'), 'pitch_map_y': b.get('pitch_map_y'),
            'pose_data': b.get('pose_data'),
        }
        for b in job_bounces
    ]
    from core.analytics import compute_session_analytics
    analytics = compute_session_analytics(bounce_events)
    speed_stats = session_speed_stats(bounce_events)

    clip_rows = list(clips or [])
    if not clip_rows and bounce_events:
        clip_rows = _synthetic_clips_from_bounces(bounce_events, frame_index, fps)
    _link_bounces_to_clips(bounce_events, clip_rows)
    clip_rows = _enrich_clips_with_bounces(clip_rows, bounce_events, fps)

    result = {
        'frames_processed': frame_index,
        'hit_detected': hit_occurred,
        'event_status': event_status,
        'output_path': output_path,
        'bounce_events': bounce_events,
        'ball_stats': ball_stats,
        'analytics': analytics,
        'speed_stats': speed_stats,
        'clips': clip_rows,
        'clip_count': len(clip_rows),
        'processing_mode': processing_mode,
        'fps': fps,
    }
    if calibration_meta:
        result['calibration'] = calibration_meta

    if job_id:
        try:
            from core.pdf_report import generate_session_pdf
            pdf_path = os.path.join(UPLOAD_FOLDER, f'report_{job_id}.pdf')
            generate_session_pdf(pdf_path, result, video_name=video_name)
            result['report_pdf_url'] = f'/report/{job_id}.pdf'
            result['report_pdf_path'] = pdf_path
            print(f"[Report] PDF saved: {pdf_path}")
        except Exception as exc:
            print(f"[Report] PDF generation failed: {exc}")

    return result


def _process_video_clip_mode(cap, output_path, job_id, model, use_half, yolo_device,
                             fps, width, height, total_frames, h_matrix, h_inv,
                             zone_color, zone_mask, pitch_annotations, pitch_alpha,
                             video_name=''):
    """3-pass clip-by-clip pipeline."""
    global session_bounces
    dt = 1.0 / fps
    conf_thresh = CONFIG['model']['confidence']

    def _pass1_progress(pct, frame_idx, total):
        _set_job_progress(job_id, pct, frame_idx, total, pass_info='Pass 1: Finding delivery clips...')

    print("[ClipMode] Pass 1 — segmenting deliveries...")
    _set_job_progress(job_id, 1, 0, total_frames, pass_info='Pass 1: Finding delivery clips...')
    clips = segment_deliveries(
        cap, model, fps, width, height, total_frames,
        conf_thresh=conf_thresh, use_half=use_half, yolo_device=yolo_device,
        progress_cb=_pass1_progress,
    )

    if not clips:
        print("[ClipMode] No clips detected — using streaming fallback")
        return None

    job_bounces = []
    persistent_video_bounces = []
    last_marker_frame = -9999
    hit_occurred = False
    event_status = "WAITING"
    clip_results = []

    print(f"[ClipMode] Pass 2 — deep tracking {len(clips)} clip(s)...")
    for i, clip in enumerate(clips):
        _set_job_progress(
            job_id, 30 + (i / len(clips)) * 40, clip.start, total_frames,
            pass_info=f'Pass 2: Tracking clip {i + 1}/{len(clips)} (frames {clip.start}–{clip.end})',
        )
        last_marker_frame, clip_result = track_delivery_clip(
            cap, clip, model, fps, width, height, h_matrix, dt,
            use_half, yolo_device, job_bounces, session_bounces,
            persistent_video_bounces, last_marker_frame, clip_index=i + 1,
        )
        clip_results.append(clip_result)
        pct = 30 + (i + 1) / len(clips) * 40
        _set_job_progress(job_id, pct, clip.end, total_frames,
                          pass_info=f'Pass 2: Done clip {i + 1}/{len(clips)}')

    print("[ClipMode] Pass 3 — rendering video...")
    _set_job_progress(job_id, 72, 0, total_frames, pass_info='Pass 3: Rendering video with markers...')
    frame_index = _render_video_clip_mode(
        cap, output_path, fps, width, height, total_frames, h_matrix, h_inv,
        job_bounces, persistent_video_bounces, clips, job_id,
        zone_color, zone_mask, pitch_annotations, pitch_alpha,
    )
    print(f"[PitchMap] clip-mode done, {len(job_bounces)} bounces, version={API_VERSION}")
    return _finalize_video_output(
        output_path, job_bounces, frame_index, hit_occurred, event_status,
        fps=fps, clips=clip_results, processing_mode='clip-by-clip',
        job_id=job_id, video_name=video_name,
    )

# ---------- Core processing ----------
def process_video(input_path, output_path, job_id=None, options=None):
    global session_bounces
    options = options or {}
    manual_quad = options.get('manual_quad')

    try:
        model, use_half, yolo_device = _get_yolo()
    except Exception as e:
        print(f"[ERROR] Model load: {e}")
        model, use_half, yolo_device = None, False, 'cpu'

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dt     = 1.0/fps
    infer_scale = min(1.0, INFER_MAX_DIM / max(width, height, 1))

    # Per-video homography — robust calibration (manual > stump+colour > fallback)
    calib = calibrate_pitch_robust(
        cap, width, height,
        manual_quad=manual_quad,
        max_samples=CALIB_SAMPLES,
    )
    cam_quad = calib.quad
    _set_processing_context(calib.stump_scale)
    print(f"[PitchCalib] source={calib.source} conf={calib.confidence:.2f} stump_scale={calib.stump_scale:.3f}")
    h_matrix = cv2.getPerspectiveTransform(cam_quad, TEMPLATE_CORNERS)
    h_inv    = cv2.getPerspectiveTransform(TEMPLATE_CORNERS, cam_quad)
    clear_template_cache()
    zone_color, zone_mask = precompute_pitch_zone_layers(height, width, h_inv)
    pitch_annotations = build_pitch_annotation_layer(height, width, h_inv)
    pitch_alpha = 0.55
    calibration_meta = {
        'source': calib.source,
        'confidence': calib.confidence,
        'stump_scale': calib.stump_scale,
        'quad': cam_quad.tolist(),
    }

    # --- Primary: frame-by-frame streaming (tracks ALL balls like original app) ---
    use_clip_pipeline = CLIP_BY_CLIP and CLIP_MODE == 'clip' and model is not None
    if use_clip_pipeline:
        clip_result = _process_video_clip_mode(
            cap, output_path, job_id, model, use_half, yolo_device,
            fps, width, height, total_frames, h_matrix, h_inv,
            zone_color, zone_mask, pitch_annotations, pitch_alpha,
            video_name=os.path.basename(input_path),
        )
        cap.release()
        if clip_result is not None:
            return clip_result
        print("[ClipMode] Retrying with streaming pipeline...")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot reopen: {input_path}")

    print("[StreamMode] Frame-by-frame tracking — all deliveries (RUN + DOT)")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    kf = BallKalmanFilter(dt=dt)
    history = SmoothHistory(maxlen=200, smooth_window=3)
    raw_history = deque(maxlen=200)
    frames_since_det = 999
    last_velocity = 0.0
    coast_limit = _coast_limit_frames(fps)

    event_status   = "WAITING"
    hit_occurred   = False
    bounce_detected = False
    bounce_frame   = -1
    last_bounce_frame = -999
    bounced_this_delivery = False
    frames_since_bounce = 999
    frame_index    = 0

    persistent_video_bounces = []
    job_bounces = []
    post_hit_max_speed = 0.0
    pre_hit_speed = 0.0
    last_detection_conf = 0.9
    last_marker_frame = -9999
    delivery_pose_frames = []
    bounce_hist_idx = None
    panel_w = min(480, max(320, int(width * 0.42)))
    cached_live_panel = None
    cached_bounce_count = -1
    empty_panel = build_panel_image([], 'PITCH MAP', panel_w) if SHOW_CORNER_PITCH_MAP else None

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_index += 1
        gap_frames = frames_since_det + 1
        frames_since_det += 1
        best_coords = None
        best_score = -float('inf')
        is_predicted = False

        detect_imgsz = min(1280, max(640, max(width, height)))
        run_detect = model is not None
        ball_active = kf.initialized or event_status != "WAITING"
        gpu_infer = infer_settings(ball_active)
        detect_imgsz = gpu_infer['imgsz']
        infer_scale_run = min(1.0, gpu_infer['max_dim'] / max(width, height, 1))
        conf_thresh = CONFIG['model']['confidence']

        skip_detect = False
        if not STREAM_DETECT_EVERY_FRAME:
            if not ball_active:
                skip_detect = (frame_index % gpu_infer['stride'] != 0)
            elif frames_since_det > 0 and frames_since_det <= coast_limit:
                skip_detect = (frame_index % DETECT_STRIDE_COAST != 0)

        # ---- Detection ----
        if run_detect and not skip_detect:
            infer_frame = frame
            if infer_scale_run < 1.0:
                infer_frame = cv2.resize(frame, (int(width * infer_scale_run), int(height * infer_scale_run)),
                                         interpolation=cv2.INTER_LINEAR)
            inv = 1.0 / infer_scale_run
            results = model.predict(infer_frame, conf=conf_thresh, imgsz=detect_imgsz, max_det=10,
                                    verbose=False, half=use_half, device=yolo_device,
                                    augment=False, stream=False)
            for box in results[0].boxes:
                if int(box.cls[0].item()) != 0: continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bw, bh = (x2 - x1) * inv, (y2 - y1) * inv
                area = bw * bh
                if area < 12 or area > 15000: continue
                aspect = bw / (bh + 1e-5)
                if not (0.15 < aspect < 5.0): continue
                cx, cy = int((x1 + x2) / 2 * inv), int((y1 + y2) / 2 * inv)
                
                score = float(box.conf[0].item())

                if kf.initialized and frames_since_det <= coast_limit:
                    px, py = kf.get_position()
                    dist = math.hypot(cx-px, cy-py)
                    vx, vy = kf.get_velocity()
                    speed_est = max(last_velocity, math.hypot(vx, vy), 30.0)
                    max_dist = max(280, width * 0.45, speed_est * frames_since_det * 5.5)
                    if dist > max_dist: continue
                    score -= dist * 0.003

                if score > best_score:
                    best_score = score
                    best_coords = (cx, cy)
                    last_detection_conf = float(box.conf[0].item())

        # Reject weak detections when no active ball track
        if best_coords is not None and event_status == "WAITING" and not kf.initialized:
            if last_detection_conf < MIN_NEW_DET_CONF:
                best_coords = None

        # ---- Update Kalman & History ----
        if best_coords is not None:
            cx, cy = best_coords
            from_waiting = event_status == "WAITING"

            if gap_frames > int(fps * 0.45) and (kf.initialized or len(raw_history) > 0):
                last_marker_frame = _finalize_delivery(
                    list(raw_history), height, h_matrix, frame_index,
                    hit_occurred, event_status, bounced_this_delivery,
                    job_bounces, session_bounces, persistent_video_bounces,
                    fps=fps, last_marker_frame=last_marker_frame,
                    last_detection_conf=last_detection_conf)
                kf, st = _reset_delivery_state(kf, dt, history, raw_history)
                hit_occurred = st['hit_occurred']
                bounce_detected = st['bounce_detected']
                bounce_frame = st['bounce_frame']
                bounced_this_delivery = st['bounced_this_delivery']
                frames_since_bounce = st['frames_since_bounce']
                bounce_hist_idx = st['bounce_hist_idx']
                post_hit_max_speed = st['post_hit_max_speed']
                pre_hit_speed = st['pre_hit_speed']
                delivery_pose_frames = st['delivery_pose_frames']
                event_status = st['event_status']

            if not kf.initialized:
                if not can_start_new_delivery(
                        frame_index, last_marker_frame, gap_frames, fps,
                        last_detection_conf, from_waiting=True):
                    best_coords = None
                elif best_coords is not None:
                    kf.init(cx, cy)
                    event_status = "BOWLED"
                    print(f"[Frame {frame_index}] New delivery (gap={gap_frames}f, conf={last_detection_conf:.2f})")
            elif best_coords is not None:
                kf.predict()
                kf.correct(cx, cy)

            if best_coords is not None:
                if len(raw_history) > 0:
                    lx, ly = raw_history[-1]
                    step = math.hypot(cx - lx, cy - ly)
                    last_velocity = step
                    vx, vy = kf.get_velocity()
                    speed_est = max(step, math.hypot(vx, vy))
                    jump_limit = max(600, width * 0.65, speed_est * 8.0)
                    if step > jump_limit and gap_frames <= int(fps * 0.30):
                        last_marker_frame = _finalize_delivery(
                            list(raw_history), height, h_matrix, frame_index,
                            hit_occurred, event_status, bounced_this_delivery,
                            job_bounces, session_bounces, persistent_video_bounces,
                            fps=fps, last_marker_frame=last_marker_frame,
                            last_detection_conf=last_detection_conf)
                        history.clear()
                        raw_history.clear()
                        kf = BallKalmanFilter(dt=dt)
                        kf.init(cx, cy)
                        hit_occurred = False
                        event_status = "BOWLED"
                        bounce_detected = False
                        bounce_frame = -1
                        bounced_this_delivery = False
                        frames_since_bounce = 999
                        bounce_hist_idx = None
                        pre_hit_speed = 0.0
                        post_hit_max_speed = 0.0

                history.add((cx, cy))
                raw_history.append((cx, cy))
                frames_since_det = 0
                if event_status == "WAITING":
                    event_status = "BOWLED"
        elif kf.initialized and frames_since_det <= coast_limit:
            kf.predict()
            px, py = kf.get_position()
            best_coords = (px, py)
            is_predicted = True
            history.add((px, py))
            raw_history.append((px, py))
            vx, vy = kf.get_velocity()
            last_velocity = max(last_velocity, math.hypot(vx, vy))
        else:
            if kf.initialized:
                kf.predict()

        raw_list = interpolate_track_gaps(list(raw_history), fps)
        hist = history.get_list()

        pose_samples = None
        if ENABLE_POSE and POSE_ESTIMATOR and delivery_pose_frames:
            pose_samples = POSE_ESTIMATOR.sample_delivery_poses(delivery_pose_frames, max_samples=3)

        # Get dynamic batsman zone to pass to classify_miss
        from core.hit_detector import batsman_zone_from_pose
        _, y_max_val = batsman_zone_from_pose(pose_samples, height)

        if ENABLE_POSE and event_status in ("BOWLED", "POST_HIT") and len(delivery_pose_frames) < 4:
            if frame_index % 4 == 0:
                delivery_pose_frames.append((frame_index, frame))

        if bounced_this_delivery:
            frames_since_bounce += 1

        # ---- Bounce: one marker per delivery on pitch ----
        if event_status == "BOWLED" and not bounced_this_delivery and not hit_occurred:
            bounce_pt = _try_detect_bounce(
                raw_list, frame_index, last_bounce_frame, fps, h_matrix,
                persistent_video_bounces, height, width=width)
            if bounce_pt is not None:
                bx, by = bounce_pt
                bounce_detected = True
                bounce_frame = frame_index
                bounced_this_delivery = True
                frames_since_bounce = 0
                bounce_hist_idx = find_hist_index_near(hist, (bx, by))

                ball_label = 'DOTS'
                video_bounce_coords = (bx, by)
                speed = _compute_delivery_speed(raw_list, fps, height, h_matrix)
                tracking_conf = 0.95 if not is_predicted else 0.6
                if ENABLE_POSE and POSE_ESTIMATOR and not pose_samples:
                    pose_samples = POSE_ESTIMATOR.sample_delivery_poses(delivery_pose_frames, max_samples=2)
                result = _add_delivery_marker(
                    bx, by, h_matrix, ball_label, frame_index,
                    job_bounces, session_bounces, persistent_video_bounces,
                    speed_kmh=speed, detection_conf=last_detection_conf,
                    tracking_conf=tracking_conf, pose_data=pose_samples or None,
                    raw_pts=raw_list, height=height, fps=fps,
                    last_marker_frame=last_marker_frame, strict=False)
                if result[0] is not None:
                    length_zone = result[0]
                    last_marker_frame = result[1]
                    last_bounce_frame = frame_index
                    print(f"[Frame {frame_index}] BOUNCE #{len(persistent_video_bounces)} @ {video_bounce_coords} | {ball_label} | {length_zone} | {speed:.0f} km/h")
                else:
                    bounced_this_delivery = False

        # ---- Reset tracking between deliveries (always, including POST_HIT) ----
        if frames_since_det > int(fps * 0.75):
            if event_status != "WAITING" or len(raw_history) > 0:
                last_marker_frame = _finalize_delivery(
                    list(raw_history), height, h_matrix, frame_index,
                    hit_occurred, event_status, bounced_this_delivery,
                    job_bounces, session_bounces, persistent_video_bounces,
                    fps=fps, last_marker_frame=last_marker_frame,
                    last_detection_conf=last_detection_conf)
                kf, st = _reset_delivery_state(kf, dt, history, raw_history)
                hit_occurred = st['hit_occurred']
                event_status = st['event_status']
                bounce_detected = st['bounce_detected']
                bounce_frame = st['bounce_frame']
                bounced_this_delivery = st['bounced_this_delivery']
                frames_since_bounce = st['frames_since_bounce']
                bounce_hist_idx = st['bounce_hist_idx']
                post_hit_max_speed = st['post_hit_max_speed']
                pre_hit_speed = st['pre_hit_speed']

        # ---- Hit & Miss Status Updates ----
        if event_status in ("BOWLED", "MISS") and not hit_occurred:
            is_hit, hit_conf, contact = score_hit_enhanced(
                raw_list, hist, height, fps, bounced_this_delivery, frames_since_bounce,
                bounce_hist_idx=bounce_hist_idx, pose_frames=pose_samples)
            if is_hit:
                hit_occurred = True
                event_status = "POST_HIT"
                pre_hit_speed = math.hypot(hist[-2][0] - hist[-3][0], hist[-2][1] - hist[-3][1]) if len(hist) >= 3 else 0.0
                post_hit_max_speed = math.hypot(hist[-1][0] - hist[-2][0], hist[-1][1] - hist[-2][1])
                if bounced_this_delivery and job_bounces:
                    _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces, 'RUNS')
                else:
                    hp = contact or (hist[-3] if len(hist) >= 3 else hist[-1])
                    speed = _compute_delivery_speed(raw_list, fps, height, h_matrix)
                    result = _add_delivery_marker(
                        hp[0], hp[1], h_matrix, 'RUNS', frame_index,
                        job_bounces, session_bounces, persistent_video_bounces,
                        speed_kmh=speed, raw_pts=raw_list, height=height, fps=fps,
                        last_marker_frame=last_marker_frame, strict=False)
                    if result[0] is not None:
                        last_marker_frame = result[1]
                        bounced_this_delivery = True
                print(f"[Frame {frame_index}] HIT — RUN (conf={hit_conf:.2f})")

        if event_status == "POST_HIT" and len(hist) >= 2:
            speed = math.hypot(hist[-1][0] - hist[-2][0], hist[-1][1] - hist[-2][1])
            post_hit_max_speed = max(post_hit_max_speed, speed)
            if classify_boundary(hist, post_hit_max_speed, height, pre_hit_speed):
                if persistent_video_bounces and persistent_video_bounces[-1]['label'] == 'RUNS':
                    _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces, 'BOUNDARIES')
                    print(f"[Frame {frame_index}] Boundary")

        if event_status == "BOWLED" and bounced_this_delivery and classify_miss(hist, height, hit_occurred, bounced_this_delivery, batsman_y_max=y_max_val):
            event_status = "MISS"
            _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces, 'WICKETS')
            print(f"[Frame {frame_index}] MISS — wicket/leave")

        if event_status == "POST_HIT" and best_coords:
            lx, ly = best_coords
            if (lx < -20 or lx > width+20 or ly < -20 or ly > height+20) and frames_since_det > int(fps * 0.8):
                last_marker_frame = _finalize_delivery(
                    list(raw_history), height, h_matrix, frame_index,
                    hit_occurred, event_status, bounced_this_delivery,
                    job_bounces, session_bounces, persistent_video_bounces,
                    fps=fps, last_marker_frame=last_marker_frame,
                    last_detection_conf=last_detection_conf)
                kf, st = _reset_delivery_state(kf, dt, history, raw_history)
                hit_occurred = st['hit_occurred']
                event_status = st['event_status']
                bounce_detected = st['bounce_detected']
                bounce_frame = st['bounce_frame']
                bounced_this_delivery = st['bounced_this_delivery']
                frames_since_bounce = st['frames_since_bounce']
                pre_hit_speed = st['pre_hit_speed']
                post_hit_max_speed = st['post_hit_max_speed']

        # ---- DRAWING: bounce dots + optional zone overlay (no corner pitch map) ----
        if SHOW_CORNER_PITCH_MAP:
            if len(job_bounces) != cached_bounce_count:
                cached_live_panel = build_panel_image(
                    [{'coords': b['coords'], 'type': b['type'],
                      'length': b.get('length'), 'speed_kmh': b.get('speed_kmh', 0)} for b in job_bounces],
                    'PITCH MAP', panel_w)
                cached_bounce_count = len(job_bounces)

        if SHOW_PITCH_ZONE_OVERLAY:
            composite_pitch_zones(frame, zone_color, zone_mask, alpha=pitch_alpha)
            nz = pitch_annotations.max(axis=2) > 0
            frame[nz] = pitch_annotations[nz]

        if SHOW_CORNER_PITCH_MAP:
            if cached_live_panel is not None:
                blit_panel(frame, cached_live_panel)
            elif empty_panel is not None:
                blit_panel(frame, empty_panel)
        if persistent_video_bounces:
            draw_light_bounce_dots(frame, persistent_video_bounces, use_video_coords=True, H_matrix=h_matrix)

        stats = bounce_stats(persistent_video_bounces)
        status_color = {"WAITING":(128,128,128),"BOWLED":(0,200,255),
                        "POST_HIT":(0,255,128),"MISS":(0,80,255)}.get(event_status,(255,255,255))
        draw_ui_panel(frame,"STATUS",event_status,(15,15),value_color=status_color)
        draw_ui_panel(frame,"BALLS",str(stats['total']),(185,15))
        draw_ui_panel(frame,"FRAME",str(frame_index),(355,15))
        draw_ball_stats_panels(frame, stats)

        writer.write(frame)
        if frame_index % 25 == 0:
            pct = (frame_index / total_frames * 100) if total_frames > 0 else 0
            _set_job_progress(job_id, pct, frame_index, total_frames)

    cap.release()
    last_marker_frame = _finalize_delivery(
        list(raw_history), height, h_matrix, frame_index,
        hit_occurred, event_status, bounced_this_delivery,
        job_bounces, session_bounces, persistent_video_bounces,
        fps=fps, last_marker_frame=last_marker_frame,
        last_detection_conf=last_detection_conf)
    _set_job_progress(job_id, 95, frame_index, total_frames)

    # ---- END SUMMARY: centred pitch map ----
    map_panel_bounces = [{'coords': b['coords'], 'type': b.get('type', 'DOTS'),
                          'length': b.get('length'), 'speed_kmh': b.get('speed_kmh', 0)} for b in job_bounces]
    # Always show at least empty pitch map template if no bounces detected
    if not map_panel_bounces:
        map_panel_bounces = [{'coords': (360, 400), 'type': 'DOTS'}]

    summary_count = 0
    if SHOW_SUMMARY_PITCH_MAP and frame is not None:
        end_panel = build_panel_image(map_panel_bounces, 'PITCH MAP', min(440, width // 2))
        for _ in range(int(fps * SUMMARY_SEC)):
            summary = frame.copy()
            summary = draw_summary_banner(summary, 'PITCH MAP - ALL TRACKED DELIVERIES')
            draw_colored_zones_on_video(summary, h_inv, alpha=0.55)
            draw_zone_boundary_lines_on_video(summary, h_inv)
            draw_distance_markers_on_video(summary, h_inv)
            draw_zone_labels_on_video(summary, h_inv, map_panel_bounces)
            draw_light_bounce_dots(summary, persistent_video_bounces, use_video_coords=True, H_matrix=h_matrix)
            draw_ball_stats_panels(summary, bounce_stats(persistent_video_bounces))
            paste_hawkeye_panel_centered(summary, map_panel_bounces, title='PITCH MAP', panel_img=end_panel)
            writer.write(summary)
            summary_count += 1
    print(f"[PitchMap] {summary_count} summary frames, {len(job_bounces)} bounces, version={API_VERSION}")
    writer.release()

    return _finalize_video_output(
        output_path, job_bounces, frame_index, hit_occurred, event_status,
        fps=fps, processing_mode='stream', job_id=job_id,
        video_name=os.path.basename(input_path),
        calibration_meta=calibration_meta,
    )

def process_video_async(job_id, input_path, output_path, options=None):
    try:
        result = process_video(input_path, output_path, job_id=job_id, options=options or {})
        with jobs_lock: jobs[job_id] = {'status':'done','result':result,'progress':100}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        with jobs_lock: jobs[job_id] = {'status':'error','error':str(exc)}

# ---------- API Routes ----------
@app.route('/')
def index(): return jsonify({'status':'API running'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files: return jsonify({'error':'No video'}),400
    file = request.files['video']
    if file.filename == '': return jsonify({'error':'Empty filename'}),400
    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    input_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(input_path)
    output_name = f"processed_{os.path.splitext(unique_name)[0]}.mp4"
    output_path = os.path.join(UPLOAD_FOLDER, output_name)
    job_id = str(uuid.uuid4())

    options = {}
    calib_raw = request.form.get('pitch_calibration')
    if calib_raw:
        try:
            import json
            quad = json.loads(calib_raw)
            if isinstance(quad, list) and len(quad) == 4:
                options['manual_quad'] = quad
        except Exception as exc:
            print(f"[PitchCalib] Invalid manual quad: {exc}")

    with jobs_lock:
        jobs[job_id] = {'status': 'queued', 'result': None, 'error': None, 'queued_at': time.time()}
    job_queue.put((job_id, input_path, output_path, options))
    _drain_queue_except(job_id)
    pos = 1 if _current_job_id is None else job_queue.qsize()
    return jsonify({'status': 'queued', 'job_id': job_id, 'queue_position': pos}), 202

@app.route('/status/<job_id>')
def status(job_id):
    with jobs_lock: job = jobs.get(job_id)
    if not job: return jsonify({'error':'Job not found'}),404
    if job['status']=='queued':
        pos = 1 if _current_job_id is None else max(1, job_queue.qsize())
        return jsonify({'status': 'queued', 'queue_position': pos})
    if job['status']=='cancelled':
        return jsonify({'status': 'error', 'error': job.get('error', 'Cancelled')}), 409
    if job['status']=='processing':
        return jsonify({
            'status': 'processing',
            'progress': job.get('progress', 0),
            'frame': job.get('frame', 0),
            'total_frames': job.get('total_frames', 0),
            'pass_info': job.get('pass_info'),
        })
    if job['status']=='error': return jsonify({'status':'error','error':job['error']}),500
    res = job['result']
    return jsonify({
        'status': 'done',
        'video_url': f"/video/{os.path.basename(res['output_path'])}",
        'report_pdf_url': res.get('report_pdf_url'),
        'summary': res,
    })

@app.route('/video/<filename>')
def video(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path): return jsonify({'error':'File not found'}),404
    return send_file(path, mimetype='video/mp4')

@app.route('/report/<job_id>.pdf')
def download_report(job_id):
    path = os.path.join(UPLOAD_FOLDER, f'report_{job_id}.pdf')
    if not os.path.exists(path):
        return jsonify({'error': 'Report not found'}), 404
    return send_file(
        path, mimetype='application/pdf', as_attachment=True,
        download_name=f'cricket_report_{job_id[:8]}.pdf',
    )

@app.route('/get_pitchmap', methods=['GET'])
def get_pitchmap():
    bowler = request.args.get('bowler', 'Bowler')
    output_map_path = os.path.join(UPLOAD_FOLDER, 'session_pitchmap.jpg')
    base = create_hawkeye_template(bowler)
    map_img = render_pitch_map(session_bounces, bowler_name=bowler, base_img=base)
    cv2.imwrite(output_map_path, map_img)
    return send_file(output_map_path, mimetype='image/jpeg')

@app.route('/get_pitchmap_data', methods=['GET'])
def get_pitchmap_data():
    return jsonify({
        'bounces': [
            {'coords': list(b['coords']), 'type': b['type']}
            for b in session_bounces
        ],
        'total': len(session_bounces),
    })

@app.route('/reset_pitchmap', methods=['POST'])
def reset_pitchmap():
    global session_bounces
    session_bounces = []
    return jsonify({'status': 'Session tracking logs cleared successfully'})

@app.route('/health')
def health():
    gpu = _gpu_name
    try:
        import torch
        gpu = gpu or (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:
        pass
    return jsonify({
        'status': 'ok',
        'model_exists': os.path.exists(MODEL_PATH),
        'version': API_VERSION,
        'gpu': gpu,
        'device': str(_yolo_device),
        'gpu_mode': 'cuda_required' if _GPU_CFG.get('require_cuda') else 'auto',
        'half_precision': _yolo_half,
    })

if __name__ == '__main__':
    try:
        _get_yolo()
    except Exception as exc:
        print(f"[WARN] Model preload failed: {exc}")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)