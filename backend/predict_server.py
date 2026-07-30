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

import math, os, threading, time, uuid, subprocess, warnings
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
    draw_light_bounce_dots, draw_predicted_bounce_marker, draw_live_ball_track,
    auto_pitch_quad, paste_hawkeye_panel, paste_hawkeye_panel_centered,
    draw_summary_banner, build_panel_image, blit_panel, classify_length_zone,
    draw_colored_zones_on_video, draw_zone_labels_on_video,
    draw_distance_markers_on_video, draw_zone_boundary_lines_on_video, clear_template_cache,
    is_on_pitch, precompute_pitch_zone_layers, composite_pitch_zones, build_pitch_annotation_layer,
    PITCH_L, PITCH_R, PITCH_TOP, PITCH_BOT,
)
from accuracy_engine import (
    refine_bounce_point, confirm_pitch_bounce, classify_boundary, classify_miss, snap_to_pitch, find_hist_index_near,
)
from core.pitch_calibrator import calibrate_pitch_robust
from core.ball_kalman import BallKalmanFilter, create_ball_kalman
from core.homography import build_homography, is_on_pitch_map as homography_on_pitch, snap_to_pitch_ground
from core.trajectory_physics import refine_bounce_world, interpolate_track_gaps, predict_bounce_landing
from core.hit_detector import score_hit_enhanced
from core.config import CONFIG
from core.pitch_coords import pitchmap_to_world, video_to_pitchmap
from core.classifier import classify_bounce
from core.gpu_runtime import init_gpu_runtime, load_yolo_model, infer_settings, ffmpeg_encode_args
from core.pose_estimator import BatsmanPoseEstimator
from core.delivery_filter import (
    can_start_new_delivery, should_register_marker, is_valid_delivery_track,
    MIN_NEW_DET_CONF, min_gap_frames,
    pending_delivery_confirmed, track_is_static, STATIC_REJECT_PX,
)
from core.ball_detection_filters import (
    ball_candidate_ok, allow_ball_detection, in_batsman_approach_zone, in_bounce_ground_zone,
    ball_bbox_size_ok, is_ball_class, effective_bounce_ground_y_min, effective_approach_y_min,
    in_machine_release_zone,
    is_landscape_frame, TRACK_DELIVERY_IN_FLIGHT, ball_area_limits, DETECT_PITCH_AREA_ONLY,
)
from core.delivery_segmenter import segment_deliveries
from core.speed_calibrator import compute_bowling_speed_kmh, session_speed_stats
from core.video_stabilizer import VideoStabilizer, stabilization_config

_TRACK = CONFIG.get('tracking', {})
COAST_SECONDS = float(_TRACK.get('coast_seconds', 0.45))
_FILT_CFG = CONFIG.get('delivery_filter', {})
PENDING_CLEAR_SEC = float(_FILT_CFG.get('pending_clear_sec', 0.65))


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
SHOW_PITCH_ZONE_OVERLAY = bool(_VIZ.get('show_pitch_zone_overlay', False))
SHOW_PREDICTIONS = bool(_VIZ.get('show_predictions', True))
TRAIL_LENGTH = int(_VIZ.get('trail_length', 16))
_PROC = CONFIG.get('processing', {})
BOUNCE_GROUND_Y_MIN = float(_PROC.get('bounce_ground_y_min', 0.65))
BOUNCE_MIN_TRACK_FRAMES = int(_PROC.get('bounce_min_track_frames', 10))
BOUNCE_LOOKBACK_FRAMES = int(_PROC.get('bounce_lookback_frames', 40))
BOUNCE_MIN_DESCENT_FRAMES = int(_PROC.get('bounce_min_descent_frames', 4))
BOUNCE_MIN_FALL_RATIO = float(_PROC.get('bounce_min_fall_ratio', 0.10))
BOUNCE_MIN_RISE_PX = float(_PROC.get('bounce_min_rise_px', 14))
BOUNCE_SKIP_RELEASE_RATIO = float(_PROC.get('bounce_skip_release_ratio', 0.25))
BOUNCE_MIN_ALONG_RATIO = float(_PROC.get('bounce_min_along_ratio', 0.32))
BOUNCE_MIN_SECONDS_AFTER_LOCK = float(_PROC.get('bounce_min_seconds_after_lock', 0.45))
BOUNCE_MIN_DIST_FROM_LOCK_RATIO = float(_PROC.get('bounce_min_dist_from_lock_ratio', 0.12))
BOUNCE_PREDICT_MIN_FRAMES = int(_PROC.get('bounce_predict_min_frames', 6))
BOUNCE_REFINE_ENABLED = bool(_PROC.get('bounce_refine_enabled', False))
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


def _log(msg: str) -> None:
    """Print immediately to terminal (worker thread + Flask)."""
    print(msg, flush=True)


def _yolo_predict(model, frame, **kwargs):
    """Run YOLO without flooding the terminal with deprecation warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*half.*deprecated.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="ultralytics")
        return model.predict(frame, verbose=False, **kwargs)


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

# Legacy global homography (per-video H computed in process_video)
camera_perspective_points = np.array([
    [250, 400], [390, 400], [100, 700], [540, 700],
], dtype=np.float32)
template_2d_points = TEMPLATE_CORNERS.copy()
H_MATRIX, H_INV, _ = build_homography(camera_perspective_points, template_2d_points)

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
            _log(f"[Process] Started {os.path.basename(input_path)} (job {job_id[:8]}…)")
            process_video_async(job_id, input_path, output_path, options=options)
        finally:
            _current_job_id = None
            job_queue.task_done()
threading.Thread(target=video_processing_worker, daemon=True).start()

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
    draw_ui_panel(frame, 'RUN', str(stats['runs'] + stats['boundaries'] * 4), (235, 75), size=(100, 48), value_color=(80, 80, 255))
    if stats['wickets']:
        draw_ui_panel(frame, 'OUT', str(stats['wickets']), (345, 75), size=(100, 48), value_color=(200, 200, 255))


def _hud_play_status(event_status, *, hit_occurred=False, bounces=None, frame_index=0, clips=None):
    """Batsman played the ball → Hit / Miss / Waiting (new ball in progress)."""
    if hit_occurred or event_status == 'POST_HIT':
        return 'Hit', (0, 255, 128)
    if event_status == 'MISS':
        return 'Miss', (0, 80, 255)
    # New ball approaching or outcome not yet decided
    if event_status in ('WAITING', 'BOWLED'):
        return 'Waiting', (0, 200, 255)

    # Clip render pass — outcome only after it is registered for this clip
    if bounces and clips:
        for clip in clips:
            if clip.start <= frame_index <= clip.end:
                clip_bounces = [
                    b for b in bounces
                    if clip.start <= b.get('frame', 0) <= frame_index
                ]
                if clip_bounces:
                    last = clip_bounces[-1]
                    settled = frame_index > last.get('frame', 0) + 5
                    if settled and last.get('hit') is True:
                        return 'Hit', (0, 255, 128)
                    if settled and last.get('hit') is False:
                        return 'Miss', (0, 80, 255)
                return 'Waiting', (0, 200, 255)

    return 'Waiting', (0, 200, 255)


def _hud_balls_delivered(stats, event_status):
    """Total balls faced — include the delivery currently in flight."""
    total = stats['total']
    if event_status == 'BOWLED':
        return total + 1
    return total


def draw_batsman_stats_hud(frame, event_status, stats, fps, *, hit_occurred=False,
                           frame_index=0, clips=None, bounces=None):
    """Compact top HUD: hit/miss, balls delivered, video FPS."""
    x, y = 10, 8
    box_w, line_h = 310, 18
    pad_x, pad_y = 10, 8
    box_h = pad_y * 2 + line_h * 3 + 4

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (12, 12, 12), -1)
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (180, 180, 180), 1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    play_text, play_color = _hud_play_status(
        event_status, hit_occurred=hit_occurred,
        bounces=bounces, frame_index=frame_index, clips=clips,
    )
    balls = _hud_balls_delivered(stats, event_status)
    fps_text = f'{int(fps)}' if fps >= 100 else (f'{fps:.1f}' if abs(fps - round(fps)) > 0.05 else f'{int(fps)}')

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.45, 1
    ty = y + pad_y + 14
    cv2.putText(frame, f'Batsman Played ball: {play_text}', (x + pad_x, ty),
                font, fs, play_color, th, cv2.LINE_AA)
    cv2.putText(frame, f'Batsman Ball delivered: {balls}', (x + pad_x, ty + line_h + 2),
                font, fs, (220, 220, 220), th, cv2.LINE_AA)
    cv2.putText(frame, f'FPS: {fps_text}', (x + pad_x, ty + 2 * (line_h + 2)),
                font, fs, (180, 220, 255), th, cv2.LINE_AA)


def _estimate_landing_point(raw_pts, height, width=0):
    """Pitch bounce from real detections only — no air fallbacks."""
    if not raw_pts:
        return None
    return confirm_pitch_bounce(
        raw_pts, height,
        lookback=len(raw_pts),
        **_bounce_confirm_kwargs(width, height),
    )


def _bounce_confirm_kwargs(width=0, height=0):
    ground_y = effective_bounce_ground_y_min(width, height) if width > 0 else BOUNCE_GROUND_Y_MIN
    return dict(
        ground_y_ratio=ground_y,
        min_descent=BOUNCE_MIN_DESCENT_FRAMES,
        min_fall_ratio=BOUNCE_MIN_FALL_RATIO,
        min_rise_px=BOUNCE_MIN_RISE_PX,
        width=width,
        skip_release_ratio=BOUNCE_SKIP_RELEASE_RATIO,
        min_along_ratio=BOUNCE_MIN_ALONG_RATIO,
    )


def _register_dot_from_track(raw_pts, height, h_matrix, frame_index, job_bounces,
                             session_bounces, persistent_video_bounces, fps,
                             last_marker_frame, last_detection_conf=0.5, width=0):
    """Fallback DOT marker when bounce apex wasn't found but track is valid."""
    from core.ball_detection_filters import moving_toward_batsman
    if not moving_toward_batsman(raw_pts, height, width=width):
        return last_marker_frame, False
    if not is_valid_delivery_track(raw_pts, height, fps, strict=False, width=width):
        return last_marker_frame, False
    landing = _estimate_landing_point(raw_pts, height, width=width)
    if landing is None:
        return last_marker_frame, False
    bx, by = landing
    # Reject fallback landing points that are still in the air (above pitch bounce band)
    if not in_bounce_ground_zone(by, height, width):
        return last_marker_frame, False
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
                         last_marker_frame=0, strict=False, hit=False, bounce_prediction=None):
    """Register one delivery outcome — bounces already validated by _try_detect_bounce."""
    # Skip extra validation since _try_detect_bounce() already did strict checks
    # if raw_pts is not None and not should_register_marker(
    #         raw_pts, height, fps, frame_index, last_marker_frame, strict=strict):
    #     print(f"[Frame {frame_index}] SKIP marker — invalid/false track")
    #     return None, last_marker_frame
    
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
        'hit': hit,
    }
    if bounce_prediction:
        if bounce_prediction.get('bounce_x') is not None:
            bounce_entry['predicted_bounce_x'] = bounce_prediction['bounce_x']
            bounce_entry['predicted_bounce_y'] = bounce_prediction['bounce_y']
        if bounce_prediction.get('length'):
            bounce_entry['predicted_length'] = bounce_prediction['length']
        if bounce_prediction.get('method'):
            bounce_entry['prediction_method'] = bounce_prediction['method']
        pred_x = bounce_prediction.get('bounce_x')
        pred_y = bounce_prediction.get('bounce_y')
        if pred_x is not None and pred_y is not None:
            err_x = abs(pred_x - classification.bounce_x)
            err_y = abs(pred_y - classification.bounce_y)
            _log(
                f"[BouncePredict] pred=({pred_x:.2f}, {pred_y:.2f}m) "
                f"actual=({classification.bounce_x:.2f}, {classification.bounce_y:.2f}m) "
                f"err=({err_x:.2f}m, {err_y:.2f}m) via {bounce_prediction.get('method', '?')}"
            )
    job_bounces.append(bounce_entry)
    session_bounces.append({
        'coords': (px_map, py_map), 'type': label,
        'length': length_zone, 'length_type': classification.length_type,
        'line_type': classification.line_type,
        'bounce_x': classification.bounce_x, 'bounce_y': classification.bounce_y,
        'speed_kmh': speed_kmh,
        'hit': hit,
    })
    persistent_video_bounces.append({
        'coords': (int(bx), int(by)), 'label': label, 'length': length_zone,
        'length_type': classification.length_type, 'line_type': classification.line_type,
        'speed_kmh': speed_kmh,
        'hit': hit,
    })
    return length_zone, frame_index


def _relocate_last_bounce_coords(bx, by, h_matrix, frame_index, job_bounces, session_bounces,
                                 persistent_video_bounces, label=None):
    """Move the last bounce marker when a deeper pitch contact is confirmed later."""
    if not persistent_video_bounces:
        return False
    old_bx, old_by = persistent_video_bounces[-1]['coords']
    if by < old_by + 12:
        return False
    label = label or persistent_video_bounces[-1].get('label', 'DOTS')
    px_map, py_map = transform_to_pitchmap(bx, by, h_matrix)
    px_map, py_map = snap_to_pitch(px_map, py_map, PITCH_L, PITCH_R, PITCH_TOP, PITCH_BOT)
    x_m, y_m = pitchmap_to_world(px_map, py_map)
    classification = classify_bounce(x_m, y_m)
    length_zone = classification.length_legacy
    persistent_video_bounces[-1].update({
        'coords': (int(bx), int(by)),
        'label': label,
        'length': length_zone,
        'length_type': classification.length_type,
        'line_type': classification.line_type,
    })
    if job_bounces:
        job_bounces[-1].update({
            'coords': (px_map, py_map),
            'length': length_zone,
            'length_type': classification.length_type,
            'line_type': classification.line_type,
            'bounce_x': classification.bounce_x,
            'bounce_y': classification.bounce_y,
            'frame': frame_index,
            'pitch_map_x': px_map,
            'pitch_map_y': py_map,
        })
    if session_bounces:
        session_bounces[-1].update({
            'coords': (px_map, py_map),
            'length': length_zone,
            'length_type': classification.length_type,
            'line_type': classification.line_type,
            'bounce_x': classification.bounce_x,
            'bounce_y': classification.bounce_y,
        })
    print(f"[Bounce] RELOCATED to ({bx:.0f}, {by:.0f}) | was ({old_bx}, {old_by})")
    return True


def _try_refine_bounce_marker(det_pts, height, h_matrix, frame_index, job_bounces, session_bounces,
                              persistent_video_bounces, bounced_this_delivery, hit_occurred,
                              frames_since_bounce, fps, width=0):
    """Optional: shift marker to a later/deeper bounce — off when first-contact-only mode."""
    if not BOUNCE_REFINE_ENABLED:
        return
    if not bounced_this_delivery or hit_occurred:
        return
    if frames_since_bounce > max(12, int(fps * 0.55)):
        return
    if len(det_pts) < BOUNCE_MIN_TRACK_FRAMES:
        return
    lookback = min(len(det_pts), BOUNCE_LOOKBACK_FRAMES)
    bounce_pt = confirm_pitch_bounce(det_pts, height, lookback=lookback, **_bounce_confirm_kwargs(width, height))
    if bounce_pt is None:
        return
    bx, by = bounce_pt
    if not in_bounce_ground_zone(by, height, width):
        return
    _relocate_last_bounce_coords(
        bx, by, h_matrix, frame_index, job_bounces, session_bounces,
        persistent_video_bounces)


def _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces, label, hit=None):
    if hit is None:
        hit = label in ('RUNS', 'BOUNDARIES')
    if session_bounces:
        session_bounces[-1]['type'] = label
        session_bounces[-1]['hit'] = hit
    if job_bounces:
        job_bounces[-1]['type'] = label
        job_bounces[-1]['hit'] = hit
    if persistent_video_bounces:
        persistent_video_bounces[-1]['label'] = label
        persistent_video_bounces[-1]['hit'] = hit


def _close_delivery(raw_pts, height, h_matrix, frame_index, hit_occurred, event_status,
                    bounced_this_delivery, job_bounces, session_bounces, persistent_video_bounces,
                    fps=25.0, last_marker_frame=0, post_hit_max_speed=0.0, width=0):
    """Finalize delivery — register DOT/RUN if no bounce marker was placed yet."""
    if bounced_this_delivery:
        return last_marker_frame
    if not is_valid_delivery_track(raw_pts, height, fps, strict=True):
        return last_marker_frame

    landing = _estimate_landing_point(raw_pts, height, width=width)
    if landing is None:
        return last_marker_frame

    bx, by = landing
    # Reject fallback landing points that are still in the air (above pitch bounce band)
    if not in_bounce_ground_zone(by, height, width):
        return last_marker_frame

    label = 'DOTS'
    if hit_occurred:
        label = 'RUNS' if post_hit_max_speed >= 8.0 else 'DOTS'
        if label == 'RUNS' and post_hit_max_speed > 38.0:
            label = 'BOUNDARIES'
    if event_status == 'MISS':
        label = 'WICKETS'

    speed = _compute_delivery_speed(raw_pts, fps, height, h_matrix)
    result = _add_delivery_marker(
        bx, by, h_matrix, label, frame_index,
        job_bounces, session_bounces, persistent_video_bounces, speed_kmh=speed,
        raw_pts=raw_pts, height=height, fps=fps, last_marker_frame=last_marker_frame,
        strict=False, hit=hit_occurred)
    if result[0] is not None:
        zone = result[0]
        print(f"[Frame {frame_index}] DELIVERY closed @ ({bx},{by}) | {label} | {zone} | {speed:.0f} km/h")
        return result[1]
    return last_marker_frame


def _resolve_batsman_miss(raw_pts, hist, height, hit_occurred, bounced, y_max_val):
    """True when ball passed the batsman without bat contact."""
    if hit_occurred or not bounced:
        return False
    pts = hist if hist and len(hist) >= 4 else raw_pts
    if len(pts) < 4:
        return False
    return classify_miss(pts, height, hit_occurred, bounced, batsman_y_max=y_max_val)


def _finalize_delivery(raw_pts, height, h_matrix, frame_index, hit_occurred, event_status,
                       bounced_this_delivery, job_bounces, session_bounces,
                       persistent_video_bounces, fps, last_marker_frame, last_detection_conf=0.5,
                       post_hit_max_speed=0.0, hist_pts=None, y_max_val=0.92, det_pts=None, width=0):
    """
    End one delivery — register DOT if no bounce yet, then close.
    Ensures every tracked ball gets a RUN or DOT marker when possible.
    """
    bounce_pts = list(det_pts) if det_pts is not None else list(raw_pts)
    bounced = bounced_this_delivery
    if not hit_occurred and _resolve_batsman_miss(
            raw_pts, hist_pts, height, hit_occurred, bounced, y_max_val):
        event_status = 'MISS'
        if bounced and persistent_video_bounces:
            _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces,
                                 'WICKETS', hit=False)

    if not bounced and not hit_occurred and len(bounce_pts) >= BOUNCE_MIN_TRACK_FRAMES:
        last_marker_frame, registered = _register_dot_from_track(
            bounce_pts, height, h_matrix, frame_index, job_bounces, session_bounces,
            persistent_video_bounces, fps, last_marker_frame, last_detection_conf, width=width)
        if registered:
            bounced = True
    last_marker_frame = _close_delivery(
        bounce_pts, height, h_matrix, frame_index, hit_occurred, event_status, bounced,
        job_bounces, session_bounces, persistent_video_bounces,
        fps=fps, last_marker_frame=last_marker_frame, post_hit_max_speed=post_hit_max_speed,
        width=width)
    return last_marker_frame


def _reset_delivery_state(kf, dt, history, raw_history, det_history=None):
    """Clear per-delivery trackers for the next ball."""
    history.clear()
    raw_history.clear()
    if det_history is not None:
        det_history.clear()
    kf_new = create_ball_kalman(1.0 / max(dt, 1e-6))
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
    """
    Finds the exact bounce point by measuring the trajectory's 'sag' due to gravity.
    The bounce is the point furthest below a straight line drawn from release to the current frame.
    """
    if len(raw_pts) < 5:
        return None
    
    # Ignore the first few frames to ensure we are past the bowler's hand release
    segment = raw_pts[3:]
    if len(segment) < 3:
        return None
        
    x1, y1 = segment[0]
    x2, y2 = segment[-1]
    n = len(segment)
    
    max_dist = -float('inf')
    best_p = None
    
    for i in range(1, n - 1):
        p = segment[i]
        # Interpolate Y on a straight line connecting the start and end of the segment
        line_y = y1 + (i / (n - 1)) * (y2 - y1)
        # Calculate how far 'below' (higher Y in pixels) the actual ball is from the straight path
        dist = p[1] - line_y
        
        if dist > max_dist:
            max_dist = dist
            best_p = p
            
    # Require at least a small amount of curvature to confirm a real bounce
    if best_p is not None and max_dist > 2.0:
        return best_p
        
    return None


def _compute_bounce_prediction(det_pts, h_matrix, h_inv, fps, height, width=0):
    """Predict landing point while ball is still in flight (YOLO detections only)."""
    if len(det_pts) < BOUNCE_PREDICT_MIN_FRAMES:
        return None
    if det_pts[-1][1] >= int(height * effective_bounce_ground_y_min(width, height) - 0.04 * height):
        return None
    ground_y = effective_bounce_ground_y_min(width, height) if width > 0 else BOUNCE_GROUND_Y_MIN
    pred = predict_bounce_landing(
        list(det_pts), h_matrix, h_inv, fps, height, width=width, ground_y_ratio=ground_y,
    )
    if pred is None:
        return None
    if pred.get('bounce_x') is not None:
        cls = classify_bounce(pred['bounce_x'], pred['bounce_y'])
        pred['length'] = cls.length_legacy
        pred['length_type'] = cls.length_type
        pred['line_type'] = cls.line_type
    return pred


def _try_detect_bounce(raw_pts, frame_index, last_bounce_frame, fps, h_matrix, persistent_video_bounces, height, width=0, lock_frame=-1, lock_pt=None, cam_quad=None):
    """
    Register bounce only on real YOLO detections: deepest pitch contact (max Y) with rise after.
    """
    from core.delivery_filter import track_is_static, is_valid_delivery_track
    from core.ball_detection_filters import is_landscape_frame

    min_pts = max(18, BOUNCE_MIN_TRACK_FRAMES)
    if len(raw_pts) < min_pts:
        return None
    if lock_frame >= 0:
        min_after_lock = max(20, int(fps * BOUNCE_MIN_SECONDS_AFTER_LOCK))
        if frame_index - lock_frame < min_after_lock:
            return None
    if frame_index - last_bounce_frame <= max(6, int(fps * 0.25)):
        return None
    if track_is_static(raw_pts[-max(6, BOUNCE_MIN_TRACK_FRAMES):]):
        print("[Bounce] REJECTED: Static track (not a moving ball)")
        return None
    if not is_valid_delivery_track(raw_pts, height, fps, strict=False, width=width):
        print("[Bounce] REJECTED: Track lacks real delivery motion")
        return None

    lookback = len(raw_pts)
    bounce_y_min = effective_bounce_ground_y_min(width, height) if width > 0 else BOUNCE_GROUND_Y_MIN
    bounce_kwargs = _bounce_confirm_kwargs(width, height)
    if lock_pt is not None:
        bounce_kwargs['lock_pt'] = lock_pt
    if cam_quad is not None:
        bounce_kwargs['cam_quad'] = cam_quad
    bounce_pt = confirm_pitch_bounce(raw_pts, height, lookback=lookback, **bounce_kwargs)
    if bounce_pt is None:
        return None

    if BOUNCE_REFINE_ENABLED and h_matrix is not None:
        refined = refine_bounce_world(raw_pts, h_matrix, height, lookback=min(lookback, BOUNCE_LOOKBACK_FRAMES))
        if refined is not None:
            rbx, rby = refined
            if in_bounce_ground_zone(rby, height, width):
                bounce_pt = refined

    bx, by = bounce_pt
    if cam_quad is not None:
        snapped = snap_to_pitch_ground(bx, by, cam_quad, height)
        if snapped is None:
            print(f"[Bounce] REJECTED: In air above pitch (by={by:.0f})")
            return None
        bx, by = snapped

    bounce_idx = find_hist_index_near(raw_pts, (bx, by))
    min_bounce_idx = max(18, int(len(raw_pts) * BOUNCE_SKIP_RELEASE_RATIO))
    if bounce_idx is not None and bounce_idx < min_bounce_idx:
        print(f"[Bounce] REJECTED: Too early (idx={bounce_idx}, need>={min_bounce_idx}) — release phase")
        return None
    if lock_pt is not None:
        lx, ly = lock_pt
        dist = math.hypot(bx - lx, by - ly)
        min_dist = height * BOUNCE_MIN_DIST_FROM_LOCK_RATIO
        if width > 0 and is_landscape_frame(width, height):
            min_dist = max(min_dist, width * 0.11)
        if dist < min_dist:
            print(f"[Bounce] REJECTED: Too close to release ({dist:.0f}px < {min_dist:.0f}px)")
            return None
    release_y = lock_pt[1] if lock_pt is not None else raw_pts[0][1]
    if by < release_y + height * 0.08:
        print(f"[Bounce] REJECTED: Still near release (by={by:.0f}, release_y={release_y:.0f})")
        return None
    if not in_bounce_ground_zone(by, height, width):
        print(f"[Bounce] REJECTED: Still in air (by={by:.0f}, need>={height * bounce_y_min:.0f})")
        return None

    travel = math.hypot(raw_pts[-1][0] - raw_pts[0][0], raw_pts[-1][1] - raw_pts[0][1])
    y_span = max(p[1] for p in raw_pts) - min(p[1] for p in raw_pts)
    min_travel = height * (0.04 if width > height * 1.12 else 0.06)
    if travel < min_travel and y_span < min_travel:
        print(f"[Bounce] REJECTED: Insufficient bowler→batsman travel (span={travel:.0f}px)")
        return None

    print(f"[Bounce] X-axis pitch touch toward batsman @ ({bx:.0f}, {by:.0f})")

    if not (height * bounce_y_min <= by < height * 0.88):
        print(f"[Bounce] REJECTED: Outside vertical zone (by={by:.0f})")
        return None

    if h_matrix is not None:
        px, py = video_to_pitchmap(bx, by, h_matrix)
        print(f"[Bounce] Pitchmap: ({px:.1f}, {py:.1f})")
        if not (150 <= py <= 700):
            print(f"[Bounce] REJECTED: Outside pitch (py={py:.1f})")
            return None

    if persistent_video_bounces:
        min_dist = min(math.hypot(bx - b['coords'][0], by - b['coords'][1]) for b in persistent_video_bounces)
        if min_dist < 25:
            print(f"[Bounce] REJECTED: Duplicate (dist={min_dist:.0f})")
            return None

    print(f"[Bounce] ACCEPTED @ ({bx:.0f}, {by:.0f})")
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
                      last_velocity=0.0, raw_history=None, post_contact=False, h_matrix=None):
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
    results = _yolo_predict(
        model,
        infer_frame, conf=conf_thresh, imgsz=detect_imgsz, max_det=10,
        half=use_half, device=yolo_device,
        augment=False, stream=False,
    )
    for box in results[0].boxes:
        if not is_ball_class(int(box.cls[0].item())):
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bw, bh = (x2 - x1) * inv, (y2 - y1) * inv
        area = bw * bh
        if not ball_bbox_size_ok(area, height, width):
            continue
        cx, cy = int((x1 + x2) / 2 * inv), int((y1 + y2) / 2 * inv)
        aspect = bw / (bh + 1e-5)
        round_ok = 0.75 < aspect < 1.35
        if not round_ok and not in_machine_release_zone(cx, cy, width, height):
            continue
        if not round_ok and not (0.55 < aspect < 1.65):
            continue

        ix1, iy1 = max(0, int(x1 * inv)), max(0, int(y1 * inv))
        ix2, iy2 = min(width, int(x2 * inv)), min(height, int(y2 * inv))
        roi = frame[iy1:iy2, ix1:ix2] if ix2 > ix1 and iy2 > iy1 else None
        det_conf = float(box.conf[0].item())
        if not ball_candidate_ok(
                cx, cy, roi, height, width, frame, post_contact=post_contact, det_conf=det_conf,
                track_active=bool(kf and kf.initialized),
                recent_points=list(raw_history) if raw_history else [],
                h_matrix=h_matrix,
        ):
            continue

        score = det_conf
        if kf is not None and kf.initialized and frames_since_det <= coast_limit:
            px, py = kf.get_position()
            dist = math.hypot(cx - px, cy - py)
            vx, vy = kf.get_velocity()
            speed_est = max(last_velocity, math.hypot(vx, vy), 30.0)
            # Dynamic search radius based on vertical zone to optimize tracking stability:
            if py > height * 0.52:
                # In batting zone: slightly wider search area so post-hit or occluded balls are not dropped.
                max_dist = 160 if frames_since_det == 0 else max(420, speed_est * frames_since_det * 4.8)
            else:
                # In the air / bowling zone: generous search area to prevent early tracking loss.
                max_dist = max(320, speed_est * frames_since_det * 4.2)
            if dist > max_dist:
                continue

            # Anti-stick filter for batting zone:
            lx, ly = (raw_history[-1] if (raw_history and len(raw_history) > 0) else (px, py))
            step_from_last = math.hypot(cx - lx, cy - ly)
            if py > height * 0.52 and last_velocity > 15.0:
                if step_from_last < 10.0:
                    score -= 0.30  # Softer penalty to avoid dropping a real ball when it is briefly occluded.
                else:
                    score -= dist * 0.0008
            else:
                score -= dist * 0.0022
        if score > best_score:
            best_score = score
            best_coords = (cx, cy)
            last_conf = float(box.conf[0].item())
    return best_coords, last_conf


def track_delivery_clip(cap, clip, model, fps, width, height, h_matrix, h_inv, dt,
                        use_half, yolo_device, job_bounces, session_bounces,
                        persistent_video_bounces, last_marker_frame, clip_index=1,
                        stump_scale=1.0, cam_quad=None):
    """
    Pass 2 — high-accuracy tracking on one delivery clip.
    Fresh Kalman per clip, stride=1 on every frame, larger imgsz.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, clip.start - 1))
    coast_limit = _coast_limit_frames(fps)
    detect_imgsz = min(1280, max(CLIP_DEEP_IMGSZ, max(width, height)))
    infer_scale_run = min(1.0, INFER_MAX_DIM_ACTIVE / max(width, height, 1))
    conf_thresh = CONFIG['model']['confidence']

    kf = create_ball_kalman(fps)
    history = SmoothHistory(maxlen=200, smooth_window=3)
    raw_history = deque(maxlen=200)
    det_history = deque(maxlen=200)
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
    last_bounce_prediction = None
    delivery_lock_frame = -1
    delivery_lock_pt = None
    frame_index = clip.start - 1
    clip_stab = VideoStabilizer(width, height, stabilization_config())

    while frame_index < clip.end:
        ret, frame = cap.read()
        if not ret:
            break
        frame = clip_stab.process(frame)
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
                raw_history=raw_history,
                post_contact=bounced_this_delivery or hit_occurred,
                h_matrix=h_matrix,
            )

        if best_coords is not None:
            cx, cy = best_coords
            if not kf.initialized:
                kf.init(cx, cy)
                event_status = "BOWLED"
                delivery_lock_frame = frame_index
                delivery_lock_pt = (cx, cy)
            else:
                kf.predict()
                if not kf.correct(cx, cy):
                    cx, cy = kf.get_position()

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
                    det_history.clear()
                    kf = create_ball_kalman(fps)
                    kf.init(cx, cy)
                    delivery_lock_frame = frame_index
                    delivery_lock_pt = (cx, cy)
                    hit_occurred = False
                    event_status = "BOWLED"
                    bounced_this_delivery = False
                    frames_since_bounce = 999
                    pre_hit_speed = 0.0
                    post_hit_max_speed = 0.0

            history.add((cx, cy))
            raw_history.append((cx, cy))
            det_history.append((cx, cy))
            frames_since_det = 0
        elif kf.initialized and frames_since_det <= coast_limit:
            kf.predict()
            px, py = kf.get_position()
            if allow_ball_detection(
                    py, height, post_contact=bounced_this_delivery or hit_occurred, width=width,
                    track_active=True,
            ):
                best_coords = (px, py)
                is_predicted = True
                history.add((px, py))
                raw_history.append((px, py))
                vx, vy = kf.get_velocity()
                last_velocity = max(last_velocity, math.hypot(vx, vy))
        elif kf.initialized:
            kf.predict()

        raw_list = interpolate_track_gaps(list(raw_history), fps)
        det_list = list(det_history)
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

        current_bounce_prediction = None
        if event_status == "BOWLED" and not bounced_this_delivery and not hit_occurred:
            current_bounce_prediction = _compute_bounce_prediction(
                det_list, h_matrix, h_inv, fps, height, width=width)
            if current_bounce_prediction is not None:
                last_bounce_prediction = current_bounce_prediction

        if event_status == "BOWLED" and not bounced_this_delivery and not hit_occurred:
            bounce_pt = _try_detect_bounce(
                det_list, frame_index, last_bounce_frame, fps, h_matrix,
                persistent_video_bounces, height, width=width,
                lock_frame=delivery_lock_frame, lock_pt=delivery_lock_pt, cam_quad=cam_quad)
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
                    last_marker_frame=last_marker_frame, strict=False, hit=False,
                    bounce_prediction=last_bounce_prediction)
                last_bounce_prediction = None
                if result[0] is not None:
                    last_marker_frame = result[1]
                    last_bounce_frame = frame_index
                    print(f"[Clip {clip.start}-{clip.end}] BOUNCE @ frame {frame_index} | {result[0]} | {speed:.0f} km/h")
                else:
                    bounced_this_delivery = False
        elif bounced_this_delivery and not hit_occurred:
            _try_refine_bounce_marker(
                det_list, height, h_matrix, frame_index, job_bounces, session_bounces,
                persistent_video_bounces, bounced_this_delivery, hit_occurred,
                frames_since_bounce, fps, width=width)

        ball_near_batsman = bool(hist) and in_batsman_approach_zone(hist[-1][1], height, width=width)
        if ball_near_batsman and event_status in ("BOWLED", "MISS") and not hit_occurred:
            is_hit, hit_conf, contact = score_hit_enhanced(
                raw_list, hist, height, fps, bounced_this_delivery, frames_since_bounce,
                bounce_hist_idx=bounce_hist_idx, pose_frames=pose_samples)
            if is_hit:
                hit_occurred = True
                event_status = "POST_HIT"
                pre_hit_speed = math.hypot(hist[-2][0] - hist[-3][0], hist[-2][1] - hist[-3][1]) if len(hist) >= 3 else 0.0
                post_hit_max_speed = math.hypot(hist[-1][0] - hist[-2][0], hist[-1][1] - hist[-2][1])
                initial_label = 'RUNS' if post_hit_max_speed >= 8.0 else 'DOTS'
                if bounced_this_delivery and job_bounces:
                    _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces, initial_label, hit=True)
                else:
                    hp = contact or (hist[-3] if len(hist) >= 3 else hist[-1])
                    speed = _compute_delivery_speed(raw_list, fps, height, h_matrix)
                    result = _add_delivery_marker(
                        hp[0], hp[1], h_matrix, initial_label, frame_index,
                        job_bounces, session_bounces, persistent_video_bounces,
                        speed_kmh=speed, raw_pts=raw_list, height=height, fps=fps,
                        last_marker_frame=last_marker_frame, strict=False, hit=True)
                    if result[0] is not None:
                        last_marker_frame = result[1]
                        bounced_this_delivery = True
                print(f"[Clip {clip.start}-{clip.end}] HIT @ frame {frame_index} (conf={hit_conf:.2f})")

        if event_status == "POST_HIT" and len(hist) >= 2:
            speed = math.hypot(hist[-1][0] - hist[-2][0], hist[-1][1] - hist[-2][1])
            post_hit_max_speed = max(post_hit_max_speed, speed)
            label = 'RUNS' if post_hit_max_speed >= 8.0 else 'DOTS'
            if classify_boundary(hist, post_hit_max_speed, height, pre_hit_speed):
                label = 'BOUNDARIES'
            if persistent_video_bounces and persistent_video_bounces[-1]['label'] in ('DOTS', 'RUNS', 'BOUNDARIES'):
                _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces, label, hit=True)

        if event_status == "BOWLED" and bounced_this_delivery:
            miss_ready = frames_since_bounce >= max(6, int(fps * 0.10))
            if miss_ready and not hit_occurred and classify_miss(
                    hist, height, hit_occurred, bounced_this_delivery, batsman_y_max=y_max_val):
                event_status = "MISS"
                _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces, 'WICKETS', hit=False)

    if not bounced_this_delivery and not hit_occurred and len(det_history) >= BOUNCE_MIN_TRACK_FRAMES:
        last_marker_frame, registered = _register_dot_from_track(
            list(det_history), height, h_matrix, min(frame_index, clip.end),
            job_bounces, session_bounces, persistent_video_bounces, fps,
            last_marker_frame, last_detection_conf)
        if registered:
            bounced_this_delivery = True

    last_marker_frame = _close_delivery(
        list(det_history), height, h_matrix, min(frame_index, clip.end),
        hit_occurred, event_status, bounced_this_delivery,
        job_bounces, session_bounces, persistent_video_bounces,
        fps=fps, last_marker_frame=last_marker_frame, post_hit_max_speed=post_hit_max_speed)

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
        'hit_detected': hit_occurred,
    }
    return last_marker_frame, clip_result


def _clip_status_at_frame(frame_index, clips, job_bounces):
    """UI status for render pass — which clip is active at this frame."""
    for clip in clips:
        if clip.start <= frame_index <= clip.end:
            clip_bounces = [
                b for b in job_bounces
                if clip.start <= b.get('frame', 0) <= frame_index
            ]
            if clip_bounces:
                last = clip_bounces[-1]
                if last.get('hit') is True:
                    return "POST_HIT"
                if last.get('hit') is False and frame_index > last.get('frame', 0) + 3:
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
        draw_batsman_stats_hud(
            frame, event_status, stats, fps,
            frame_index=frame_index, clips=clips, bounces=visible_job,
        )

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
        _set_job_progress(job_id, pct, frame_idx, total)

    print("[ClipMode] Pass 1 — segmenting deliveries...")
    _set_job_progress(job_id, 1, 0, total_frames)
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
            cap, clip, model, fps, width, height, h_matrix, h_inv, dt,
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
    print(f"[Homography] quad validated — pitch map aligned (landscape={is_landscape_frame(width, height)})")
    from core.ball_kalman import kalman_config
    kcfg = kalman_config()
    print(f"[Kalman] dt={1.0/fps:.4f}s gate={kcfg['max_gate_px']:.0f}px coast={COAST_SECONDS}s")
    if is_landscape_frame(width, height):
        lo, hi = ball_area_limits(height, width)
        print(f"[Video] landscape {width}x{height} — approach_y={effective_approach_y_min(width, height):.2f} ball_area={lo:.0f}-{hi:.0f}px²")
    if DETECT_PITCH_AREA_ONLY:
        print("[Detect] pitch-area-only — ball search on pitch strip + machine release corridor")
    h_matrix, h_inv, cam_quad = build_homography(calib.quad, TEMPLATE_CORNERS)
    calib_meta_quad = cam_quad.tolist()
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

    print(f"[StreamMode] Frame-by-frame tracking — all deliveries (RUN + DOT)")
    if SHOW_PREDICTIONS:
        print("[BouncePredict] pre-bounce landing prediction enabled")
    if TRACK_DELIVERY_IN_FLIGHT:
        print("[Delivery] flight tracking ON — release→batsman (ground resting balls rejected)")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stab_cfg = stabilization_config()
    if options.get('stabilize') is not None:
        stab_cfg = dict(stab_cfg)
        stab_cfg['mode'] = str(options.get('stabilize')).lower()
        if stab_cfg['mode'] in ('1', 'true', 'yes', 'on'):
            stab_cfg['mode'] = 'always'
        if stab_cfg['mode'] in ('0', 'false', 'no', 'off'):
            stab_cfg['mode'] = 'off'
    stabilizer = VideoStabilizer(width, height, stab_cfg)
    if stab_cfg['mode'] != 'off':
        print(f"[Stabilize] mode={stab_cfg['mode']} (shake auto-threshold={stab_cfg['shake_enable_px']}px)")

    kf = create_ball_kalman(fps)
    history = SmoothHistory(maxlen=200, smooth_window=3)
    raw_history = deque(maxlen=200)
    det_history = deque(maxlen=200)
    pending_lock_pts = deque(maxlen=12)
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
    current_bounce_prediction = None
    last_bounce_prediction = None
    delivery_lock_frame = -1
    delivery_lock_pt = None
    panel_w = min(480, max(320, int(width * 0.42)))
    cached_live_panel = None
    cached_bounce_count = -1
    empty_panel = build_panel_image([], 'PITCH MAP', panel_w) if SHOW_CORNER_PITCH_MAP else None
    last_relable_label = None
    progress_step = max(50, total_frames // 20) if total_frames > 0 else 100

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = stabilizer.process(frame)
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
        if not ball_active and is_landscape_frame(width, height):
            gpu_infer = infer_settings(True)
        detect_imgsz = gpu_infer['imgsz']
        infer_scale_run = min(1.0, gpu_infer['max_dim'] / max(width, height, 1))
        conf_thresh = CONFIG['model']['confidence']
        if event_status == "WAITING" and not kf.initialized:
            conf_thresh = float(_PROC.get('waiting_conf', 0.12))
        post_contact = bounced_this_delivery or hit_occurred

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
            results = _yolo_predict(model, infer_frame, conf=conf_thresh, imgsz=detect_imgsz, max_det=10,
                                    half=use_half, device=yolo_device,
                                    augment=False, stream=False)
            for box in results[0].boxes:
                if not is_ball_class(int(box.cls[0].item())): continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bw, bh = (x2 - x1) * inv, (y2 - y1) * inv
                area = bw * bh
                if not ball_bbox_size_ok(area, height, width):
                    continue
                cx, cy = int((x1 + x2) / 2 * inv), int((y1 + y2) / 2 * inv)
                aspect = bw / (bh + 1e-5)
                round_ok = 0.75 < aspect < 1.35
                if not round_ok and not in_machine_release_zone(cx, cy, width, height):
                    continue
                if not round_ok and not (0.55 < aspect < 1.65):
                    continue

                ix1, iy1 = max(0, int(x1 * inv)), max(0, int(y1 * inv))
                ix2, iy2 = min(width, int(x2 * inv)), min(height, int(y2 * inv))
                roi = frame[iy1:iy2, ix1:ix2] if ix2 > ix1 and iy2 > iy1 else None
                det_conf = float(box.conf[0].item())
                recent_pts = list(pending_lock_pts)
                if not ball_candidate_ok(
                        cx, cy, roi, height, width, frame, post_contact=post_contact, det_conf=det_conf,
                        track_active=kf.initialized,
                        recent_points=recent_pts,
                        h_matrix=h_matrix,
                ):
                    continue
                
                if kf.initialized and frames_since_det <= coast_limit:
                    px, py = kf.get_position()
                    dist = math.hypot(cx-px, cy-py)
                    vx, vy = kf.get_velocity()
                    speed_est = max(last_velocity, math.hypot(vx, vy), 30.0)
                    # Dynamic search radius based on vertical zone to optimize tracking stability:
                    if py > height * 0.52:
                        max_dist = 130 if frames_since_det == 0 else max(380, speed_est * frames_since_det * 4.5)
                    else:
                        max_dist = max(280, speed_est * frames_since_det * 4.0)
                    if dist > max_dist: continue
                    score = det_conf

                    # Anti-stick filter for batting zone:
                    lx, ly = (raw_history[-1] if (raw_history and len(raw_history) > 0) else (px, py))
                    step_from_last = math.hypot(cx - lx, cy - ly)
                    if step_from_last < STATIC_REJECT_PX and last_velocity < 10.0:
                        score -= 0.45
                    elif py > height * 0.52 and last_velocity > 15.0:
                        if step_from_last < 10.0:
                            score -= 0.50  # Heavy penalty to prevent locking onto static batsman/pad/bat
                        else:
                            score -= dist * 0.001  # Reduced penalty for moving candidates to allow post-hit re-acquisition
                    else:
                        score -= dist * 0.003
                else:
                    _, max_ball_area = ball_area_limits(height, width)
                    size_bonus = 0.45 * (1.0 - min(1.0, area / max(max_ball_area, 1.0)))
                    score = det_conf + size_bonus
                    if area > max_ball_area * 0.40:
                        score -= 0.55
                    if not kf.initialized and pending_lock_pts:
                        lx, ly = pending_lock_pts[-1]
                        step_pending = math.hypot(cx - lx, cy - ly)
                        if step_pending < STATIC_REJECT_PX:
                            score -= 0.75
                        elif len(pending_lock_pts) >= 2:
                            fx, fy = pending_lock_pts[0]
                            trail_dx, trail_dy = lx - fx, ly - fy
                            cand_dx, cand_dy = cx - lx, cy - ly
                            if math.hypot(trail_dx, trail_dy) > STATIC_REJECT_PX:
                                align = trail_dx * cand_dx + trail_dy * cand_dy
                                if align > 0:
                                    score += 0.08

                if score > best_score:
                    best_score = score
                    best_coords = (cx, cy)
                    last_detection_conf = float(box.conf[0].item())

        # Reject weak detections when no active ball track
        if best_coords is not None and event_status == "WAITING" and not kf.initialized:
            if last_detection_conf < MIN_NEW_DET_CONF:
                best_coords = None

        if best_coords is None and not kf.initialized and pending_lock_pts:
            if frames_since_det > int(max(8, fps * PENDING_CLEAR_SEC)):
                pending_lock_pts.clear()

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
                    last_detection_conf=last_detection_conf,
                    post_hit_max_speed=post_hit_max_speed,
                    hist_pts=list(raw_history), det_pts=list(det_history), width=width)
                kf, st = _reset_delivery_state(kf, dt, history, raw_history, det_history)
                last_marker_frame = -9999
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
                pending_lock_pts.clear()
                last_bounce_prediction = None
                delivery_lock_frame = -1
                delivery_lock_pt = None

            if not kf.initialized:
                pending_lock_pts.append((cx, cy))
                lock_ready = pending_delivery_confirmed(list(pending_lock_pts), height, width=width)
                if not can_start_new_delivery(
                        frame_index, last_marker_frame, gap_frames, fps,
                        last_detection_conf, from_waiting=True):
                    best_coords = None
                elif not lock_ready:
                    best_coords = None
                    frames_since_det = 0
                elif best_coords is not None:
                    kf.init(cx, cy)
                    event_status = "BOWLED"
                    delivery_lock_frame = frame_index
                    delivery_lock_pt = (cx, cy)
                    pending_lock_pts.clear()
                    last_bounce_prediction = None
                    _log(f"[Frame {frame_index}] New delivery (gap={gap_frames}f, conf={last_detection_conf:.2f}, moving)")
            elif best_coords is not None:
                kf.predict()
                if not kf.correct(cx, cy):
                    cx, cy = kf.get_position()
                    is_predicted = True

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
                            last_detection_conf=last_detection_conf,
                            post_hit_max_speed=post_hit_max_speed,
                            hist_pts=list(raw_history), det_pts=list(det_history), width=width)
                        raw_history.clear()
                        det_history.clear()
                        pending_lock_pts.clear()
                        kf = create_ball_kalman(fps)
                        kf.init(cx, cy)
                        delivery_lock_frame = frame_index
                        delivery_lock_pt = (cx, cy)
                        hit_occurred = False
                        event_status = "BOWLED"
                        bounce_detected = False
                        bounce_frame = -1
                        bounced_this_delivery = False
                        frames_since_bounce = 999
                        bounce_hist_idx = None
                        pre_hit_speed = 0.0
                        post_hit_max_speed = 0.0
                elif kf.initialized and len(raw_history) >= 3 and track_is_static(list(raw_history)):
                    kf = create_ball_kalman(fps)
                    raw_history.clear()
                    det_history.clear()
                    pending_lock_pts.clear()
                    event_status = "WAITING"
                    best_coords = None

                if best_coords is not None:
                    history.add((cx, cy))
                    raw_history.append((cx, cy))
                    det_history.append((cx, cy))
                    frames_since_det = 0
                    if event_status == "WAITING":
                        event_status = "BOWLED"
        elif kf.initialized and frames_since_det <= coast_limit:
            kf.predict()
            px, py = kf.get_position()
            if allow_ball_detection(
                    py, height, post_contact=post_contact, width=width, track_active=True,
            ):
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
        det_list = list(det_history)
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

        current_bounce_prediction = None
        if event_status == "BOWLED" and not bounced_this_delivery and not hit_occurred:
            current_bounce_prediction = _compute_bounce_prediction(
                det_list, h_matrix, h_inv, fps, height, width=width)
            if current_bounce_prediction is not None:
                last_bounce_prediction = current_bounce_prediction

        # ---- Bounce: one marker per delivery on pitch ----
        if event_status == "BOWLED" and not bounced_this_delivery and not hit_occurred:
            bounce_pt = _try_detect_bounce(
                det_list, frame_index, last_bounce_frame, fps, h_matrix,
                persistent_video_bounces, height, width=width,
                lock_frame=delivery_lock_frame, lock_pt=delivery_lock_pt, cam_quad=cam_quad)
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
                    last_marker_frame=last_marker_frame, strict=False, hit=False,
                    bounce_prediction=last_bounce_prediction)
                last_bounce_prediction = None
                if result[0] is not None:
                    length_zone = result[0]
                    last_marker_frame = result[1]
                    last_bounce_frame = frame_index
                    print(f"[Frame {frame_index}] BOUNCE #{len(persistent_video_bounces)} @ {video_bounce_coords} | {ball_label} | {length_zone} | {speed:.0f} km/h")
                else:
                    bounced_this_delivery = False
        elif bounced_this_delivery and not hit_occurred:
            _try_refine_bounce_marker(
                det_list, height, h_matrix, frame_index, job_bounces, session_bounces,
                persistent_video_bounces, bounced_this_delivery, hit_occurred,
                frames_since_bounce, fps, width=width)

        # ---- Hit & Miss: only after ball reaches batsman approach zone ----
        ball_near_batsman = bool(hist) and in_batsman_approach_zone(hist[-1][1], height, width=width)
        if ball_near_batsman and event_status in ("BOWLED", "MISS") and not hit_occurred:
            is_hit, hit_conf, contact = score_hit_enhanced(
                raw_list, hist, height, fps, bounced_this_delivery, frames_since_bounce,
                bounce_hist_idx=bounce_hist_idx, pose_frames=pose_samples)
            if is_hit:
                hit_occurred = True
                event_status = "POST_HIT"
                pre_hit_speed = math.hypot(hist[-2][0] - hist[-3][0], hist[-2][1] - hist[-3][1]) if len(hist) >= 3 else 0.0
                post_hit_max_speed = math.hypot(hist[-1][0] - hist[-2][0], hist[-1][1] - hist[-2][1])
                initial_label = 'RUNS' if post_hit_max_speed >= 8.0 else 'DOTS'
                if bounced_this_delivery and job_bounces:
                    _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces, initial_label, hit=True)
                else:
                    hp = contact or (hist[-3] if len(hist) >= 3 else hist[-1])
                    speed = _compute_delivery_speed(raw_list, fps, height, h_matrix)
                    result = _add_delivery_marker(
                        hp[0], hp[1], h_matrix, initial_label, frame_index,
                        job_bounces, session_bounces, persistent_video_bounces,
                        speed_kmh=speed, raw_pts=raw_list, height=height, fps=fps,
                        last_marker_frame=last_marker_frame, strict=False, hit=True)
                    if result[0] is not None:
                        last_marker_frame = result[1]
                        bounced_this_delivery = True
                print(f"[Frame {frame_index}] HIT — bat contact (conf={hit_conf:.2f})")

        if event_status == "POST_HIT" and len(hist) >= 2:
            speed = math.hypot(hist[-1][0] - hist[-2][0], hist[-1][1] - hist[-2][1])
            post_hit_max_speed = max(post_hit_max_speed, speed)
            label = 'RUNS' if post_hit_max_speed >= 8.0 else 'DOTS'
            if classify_boundary(hist, post_hit_max_speed, height, pre_hit_speed):
                label = 'BOUNDARIES'
            if persistent_video_bounces and persistent_video_bounces[-1]['label'] in ('DOTS', 'RUNS', 'BOUNDARIES'):
                _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces, label, hit=True)
                if label != last_relable_label:
                    _log(f"[Frame {frame_index}] Relabeled to {label}")
                    last_relable_label = label

        if ball_near_batsman and event_status == "BOWLED" and bounced_this_delivery:
            miss_ready = frames_since_bounce >= max(6, int(fps * 0.10))
            if miss_ready and not hit_occurred and classify_miss(
                    hist, height, hit_occurred, bounced_this_delivery, batsman_y_max=y_max_val):
                event_status = "MISS"
                _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces, 'WICKETS', hit=False)
                print(f"[Frame {frame_index}] MISS — batsman left the ball")

        # ---- Reset tracking between deliveries (always, including POST_HIT) ----
        if frames_since_det > int(fps * 0.75):
            if event_status != "WAITING" or len(raw_history) > 0:
                last_marker_frame = _finalize_delivery(
                    list(raw_history), height, h_matrix, frame_index,
                    hit_occurred, event_status, bounced_this_delivery,
                    job_bounces, session_bounces, persistent_video_bounces,
                    fps=fps, last_marker_frame=last_marker_frame,
                    last_detection_conf=last_detection_conf,
                    post_hit_max_speed=post_hit_max_speed,
                    hist_pts=hist, y_max_val=y_max_val, det_pts=list(det_history), width=width)
                kf, st = _reset_delivery_state(kf, dt, history, raw_history, det_history)
                last_marker_frame = -9999
                hit_occurred = st['hit_occurred']
                event_status = st['event_status']
                bounce_detected = st['bounce_detected']
                bounce_frame = st['bounce_frame']
                bounced_this_delivery = st['bounced_this_delivery']
                frames_since_bounce = st['frames_since_bounce']
                bounce_hist_idx = st['bounce_hist_idx']
                post_hit_max_speed = st['post_hit_max_speed']
                pre_hit_speed = st['pre_hit_speed']

        if event_status == "POST_HIT" and best_coords:
            lx, ly = best_coords
            if (lx < -20 or lx > width+20 or ly < -20 or ly > height+20) and frames_since_det > int(fps * 0.8):
                last_marker_frame = _finalize_delivery(
                    list(raw_history), height, h_matrix, frame_index,
                    hit_occurred, event_status, bounced_this_delivery,
                    job_bounces, session_bounces, persistent_video_bounces,
                    fps=fps, last_marker_frame=last_marker_frame,
                    last_detection_conf=last_detection_conf,
                    post_hit_max_speed=post_hit_max_speed,
                    hist_pts=hist, y_max_val=y_max_val, det_pts=list(det_history), width=width)
                kf, st = _reset_delivery_state(kf, dt, history, raw_history, det_history)
                last_marker_frame = -9999
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
        if SHOW_PREDICTIONS and not bounced_this_delivery:
            pred_draw = current_bounce_prediction or last_bounce_prediction
            if pred_draw and pred_draw.get('video'):
                vx, vy = pred_draw['video']
                draw_predicted_bounce_marker(
                    frame, int(vx), int(vy),
                    length_hint=pred_draw.get('length', ''),
                )
        if TRAIL_LENGTH > 0 and event_status in ("BOWLED", "POST_HIT") and hist:
            trail = hist[-TRAIL_LENGTH:] if len(hist) > 1 else hist
            live_pt = best_coords or (hist[-1] if hist else None)
            draw_live_ball_track(frame, live_pt, trail_pts=trail, is_predicted=is_predicted)

        stats = bounce_stats(persistent_video_bounces)
        draw_batsman_stats_hud(
            frame, event_status, stats, fps,
            hit_occurred=hit_occurred, bounces=persistent_video_bounces,
        )

        writer.write(frame)
        if frame_index % progress_step == 0 or frame_index == total_frames:
            pct = (frame_index / total_frames * 100) if total_frames > 0 else 0
            _set_job_progress(job_id, pct, frame_index, total_frames)
            _log(f"[Process] {frame_index}/{total_frames} ({pct:.0f}%) | deliveries={len(job_bounces)}")

    cap.release()
    stab_info = stabilizer.stats()
    _log(f"[Stabilize] done enabled={stab_info['enabled']} applied={stab_info['applied']}/{stab_info['frames']} last_shake={stab_info['last_shake_px']}px")
    calibration_meta = dict(calibration_meta or {})
    calibration_meta['stabilization'] = stab_info
    last_marker_frame = _finalize_delivery(
        list(raw_history), height, h_matrix, frame_index,
        hit_occurred, event_status, bounced_this_delivery,
        job_bounces, session_bounces, persistent_video_bounces,
        fps=fps, last_marker_frame=last_marker_frame,
        last_detection_conf=last_detection_conf,
        post_hit_max_speed=post_hit_max_speed,
        hist_pts=history.get_list(), det_pts=list(det_history), width=width)
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
    _log(f"[PitchMap] {summary_count} summary frames, {len(job_bounces)} bounces, version={API_VERSION}")
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