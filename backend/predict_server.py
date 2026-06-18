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

API_VERSION = 'pitchmap-v26-sidebar-labels'

# Detect every frame — fast balls miss if we skip frames
DETECT_STRIDE_WAITING = 1
DETECT_STRIDE_COAST = 1
INFER_MAX_DIM = 1280
INFER_MAX_DIM_ACTIVE = 1280
WAITING_IMGSZ = 1280
ACTIVE_IMGSZ = 1280
MIN_TRACK_FRAMES = 3
SUMMARY_SEC = 3

_yolo_model = None
_yolo_half = False
_yolo_device = 'cpu'
_gpu_name = None

def _get_yolo():
    global _yolo_model, _yolo_half, _yolo_device, _gpu_name
    if _yolo_model is None:
        try:
            import torch
            if torch.cuda.is_available():
                _yolo_device = 0
                _yolo_half = True
                _gpu_name = torch.cuda.get_device_name(0)
            else:
                _yolo_device = 'cpu'
                _yolo_half = False
                _gpu_name = None
        except Exception:
            _yolo_device = 'cpu'
            _yolo_half = False
            _gpu_name = None
        _yolo_model = YOLO(MODEL_PATH)
        _yolo_model.to(_yolo_device)
        # Warm-up inference so first video job is not delayed
        _yolo_model.predict(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False,
                              device=_yolo_device, half=_yolo_half, imgsz=320)
        print(f"[YOLO] device={_yolo_device} half={_yolo_half} gpu={_gpu_name or 'none'}")
    return _yolo_model, _yolo_half, _yolo_device

def _set_job_progress(job_id, pct, frame_idx=0, total=0):
    if not job_id:
        return
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]['progress'] = round(pct, 1)
            jobs[job_id]['frame'] = frame_idx
            jobs[job_id]['total_frames'] = total

app = Flask(__name__)
CORS(app)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'train5', 'weights', 'best.pt')

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
        job_id, input_path, output_path = job_queue.get()
        with jobs_lock:
            job = jobs.get(job_id)
            if not job or job.get('status') == 'cancelled':
                job_queue.task_done()
                continue
            jobs[job_id]['status'] = 'processing'
        _current_job_id = job_id
        try:
            process_video_async(job_id, input_path, output_path)
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


def _compute_delivery_speed(raw_pts, fps, height):
    """Approximate bowling speed (km/h) from pixel motion along the track."""
    if len(raw_pts) < 2 or fps <= 0:
        return 0.0
    speeds = []
    for i in range(1, len(raw_pts)):
        d = math.hypot(raw_pts[i][0] - raw_pts[i - 1][0], raw_pts[i][1] - raw_pts[i - 1][1])
        speeds.append(d * fps)
    peak_px_s = max(speeds)
    pitch_span_px = max(height * 0.42, 80.0)
    meters_per_px = 18.0 / pitch_span_px
    return round(peak_px_s * meters_per_px * 3.6, 0)


def _add_delivery_marker(bx, by, h_matrix, label, frame_index,
                         job_bounces, session_bounces, persistent_video_bounces,
                         speed_kmh=0.0):
    """Register one delivery outcome on video + pitch map."""
    px_map, py_map = transform_to_pitchmap(bx, by, h_matrix)
    px_map = int(max(PITCH_L + 8, min(PITCH_R - 8, px_map)))
    py_map = int(max(PITCH_TOP + 8, min(PITCH_BOT - 8, py_map)))
    length_zone = classify_length_zone(py_map)
    speed_kmh = round(float(speed_kmh or 0), 0)
    bounce_entry = {
        'coords': (px_map, py_map), 'type': label,
        'length': length_zone, 'frame': frame_index, 'speed_kmh': speed_kmh,
    }
    job_bounces.append(bounce_entry)
    session_bounces.append({
        'coords': (px_map, py_map), 'type': label,
        'length': length_zone, 'speed_kmh': speed_kmh,
    })
    persistent_video_bounces.append({
        'coords': (int(bx), int(by)), 'label': label, 'length': length_zone,
        'speed_kmh': speed_kmh,
    })
    return length_zone


def _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces, label):
    if session_bounces:
        session_bounces[-1]['type'] = label
    if job_bounces:
        job_bounces[-1]['type'] = label
    if persistent_video_bounces:
        persistent_video_bounces[-1]['label'] = label


def _close_delivery(raw_pts, height, h_matrix, frame_index, hit_occurred, event_status,
                    bounced_this_delivery, job_bounces, session_bounces, persistent_video_bounces,
                    fps=25.0):
    """Finalize delivery — add marker even if ball went wide/fast off pitch."""
    if len(raw_pts) < MIN_TRACK_FRAMES:
        return
    if bounced_this_delivery:
        return

    landing = _estimate_landing_point(raw_pts, height)
    if landing is None:
        return

    label = 'RUNS' if hit_occurred else 'DOTS'
    if event_status == 'MISS':
        label = 'WICKETS'

    bx, by = landing
    speed = _compute_delivery_speed(raw_pts, fps, height)
    zone = _add_delivery_marker(
        bx, by, h_matrix, label, frame_index,
        job_bounces, session_bounces, persistent_video_bounces, speed_kmh=speed)
    print(f"[Frame {frame_index}] DELIVERY closed @ ({bx},{by}) | {label} | {zone} | {speed:.0f} km/h")


def _try_detect_bounce(raw_pts, frame_index, last_bounce_frame, fps, h_matrix, persistent_video_bounces):
    """Pitch bounce — V-shape in trajectory, filtered to pitch area."""
    if len(raw_pts) < 3:
        return None
    if frame_index - last_bounce_frame <= max(8, int(fps * 0.32)):
        return None

    dy1 = raw_pts[-1][1] - raw_pts[-2][1]
    dy2 = raw_pts[-2][1] - raw_pts[-3][1]
    bx, by = raw_pts[-2]
    if not (dy1 < -1.2 and dy2 > 1.2):
        return None
    if not is_on_pitch(bx, by, h_matrix, margin=20):
        return None

    too_close = any(
        math.hypot(bx - b['coords'][0], by - b['coords'][1]) < 24
        for b in persistent_video_bounces)
    if too_close:
        return None
    return (bx, by)


def _try_detect_hit(hist_pts, height):
    """Bat hit — sharp angle change in ball path."""
    if len(hist_pts) < 4:
        return False

    p1, p2, p3, p4 = hist_pts[-4], hist_pts[-3], hist_pts[-2], hist_pts[-1]
    v_pre = (p3[0] - p1[0], p3[1] - p1[1])
    v_post = (p4[0] - p2[0], p4[1] - p2[1])
    if math.hypot(*v_pre) < 4 or math.hypot(*v_post) < 4:
        return False

    cos_a = max(-1, min(1, (v_pre[0] * v_post[0] + v_pre[1] * v_post[1]) / (math.hypot(*v_pre) * math.hypot(*v_post))))
    angle = math.degrees(math.acos(cos_a))
    if angle < 20:
        return False
    if p3[1] < int(height * 0.28):
        return False
    return True

# ---------- Core processing ----------
def process_video(input_path, output_path, job_id=None):
    global session_bounces
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

    # Per-video homography (auto-fit pitch to frame)
    cam_quad = auto_pitch_quad(width, height)
    h_matrix = cv2.getPerspectiveTransform(cam_quad, TEMPLATE_CORNERS)
    h_inv    = cv2.getPerspectiveTransform(TEMPLATE_CORNERS, cam_quad)
    clear_template_cache()
    zone_color, zone_mask = precompute_pitch_zone_layers(height, width, h_inv)
    pitch_annotations = build_pitch_annotation_layer(height, width, h_inv)
    pitch_alpha = 0.55

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    kf = BallKalmanFilter(dt=dt)
    history = SmoothHistory(maxlen=200, smooth_window=3)
    raw_history = deque(maxlen=200)
    frames_since_det = 999
    last_velocity = 0.0
    coast_limit = max(18, int(fps * 0.70))

    event_status   = "WAITING"
    hit_occurred   = False
    bounce_detected = False
    bounce_frame   = -1
    last_bounce_frame = -999
    bounced_this_delivery = False
    frame_index    = 0

    persistent_video_bounces = []
    job_bounces = []
    post_hit_max_speed = 0.0
    panel_w = min(480, max(320, int(width * 0.42)))
    cached_live_panel = None
    cached_bounce_count = -1
    empty_panel = build_panel_image([], 'PITCH MAP', panel_w)

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
        infer_scale_run = min(1.0, INFER_MAX_DIM_ACTIVE / max(width, height, 1))
        conf_thresh = 0.06

        skip_detect = False
        if not ball_active:
            skip_detect = (frame_index % DETECT_STRIDE_WAITING != 0)
        elif frames_since_det > 0 and frames_since_det <= coast_limit:
            skip_detect = (frame_index % DETECT_STRIDE_COAST != 0)

        # ---- Detection ----
        if run_detect and not skip_detect:
            infer_frame = frame
            if infer_scale_run < 1.0:
                infer_frame = cv2.resize(frame, (int(width * infer_scale_run), int(height * infer_scale_run)),
                                         interpolation=cv2.INTER_LINEAR)
            inv = 1.0 / infer_scale_run
            results = model.predict(infer_frame, conf=conf_thresh, imgsz=detect_imgsz, max_det=20,
                                    verbose=False, half=use_half, device=yolo_device)
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

        # ---- Update Kalman & History ----
        if best_coords is not None:
            cx, cy = best_coords
            # New ball after gap — fresh delivery (keeps all prior bounce dots)
            if gap_frames > int(fps * 0.30):
                _close_delivery(
                    list(raw_history), height, h_matrix, frame_index,
                    hit_occurred, event_status, bounced_this_delivery,
                    job_bounces, session_bounces, persistent_video_bounces, fps=fps)
                history.clear()
                raw_history.clear()
                kf = BallKalmanFilter(dt=dt)
                hit_occurred = False
                bounce_detected = False
                bounce_frame = -1
                bounced_this_delivery = False
                post_hit_max_speed = 0.0
                event_status = "BOWLED"
                print(f"[Frame {frame_index}] New delivery (gap={gap_frames}f)")

            if not kf.initialized:
                kf.init(cx, cy)
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
                    _close_delivery(
                        list(raw_history), height, h_matrix, frame_index,
                        hit_occurred, event_status, bounced_this_delivery,
                        job_bounces, session_bounces, persistent_video_bounces, fps=fps)
                    history.clear()
                    raw_history.clear()
                    kf = BallKalmanFilter(dt=dt)
                    kf.init(cx, cy)
                    hit_occurred = False
                    event_status = "BOWLED"
                    bounce_detected = False
                    bounce_frame = -1
                    bounced_this_delivery = False

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

        raw_list = list(raw_history)
        hist = history.get_list()

        # ---- Bounce: one marker per delivery on pitch ----
        if event_status == "BOWLED" and not bounced_this_delivery and not hit_occurred:
            bounce_pt = _try_detect_bounce(
                raw_list, frame_index, last_bounce_frame, fps, h_matrix, persistent_video_bounces)
            if bounce_pt is not None:
                bx, by = bounce_pt
                bounce_detected = True
                bounce_frame = frame_index
                bounced_this_delivery = True

                ball_label = 'DOTS'
                video_bounce_coords = (bx, by)
                speed = _compute_delivery_speed(raw_list, fps, height)
                length_zone = _add_delivery_marker(
                    bx, by, h_matrix, ball_label, frame_index,
                    job_bounces, session_bounces, persistent_video_bounces,
                    speed_kmh=speed)
                last_bounce_frame = frame_index
                print(f"[Frame {frame_index}] BOUNCE #{len(persistent_video_bounces)} @ {video_bounce_coords} | {ball_label} | {length_zone} | {speed:.0f} km/h")

        # ---- Reset tracking between deliveries (always, including POST_HIT) ----
        if frames_since_det > int(fps * 0.90):
            if event_status != "WAITING" or len(raw_history) > 0:
                _close_delivery(
                    list(raw_history), height, h_matrix, frame_index,
                    hit_occurred, event_status, bounced_this_delivery,
                    job_bounces, session_bounces, persistent_video_bounces, fps=fps)
                history.clear()
                raw_history.clear()
                kf = BallKalmanFilter(dt=dt)
                hit_occurred = False
                event_status = "WAITING"
                bounce_detected = False
                bounce_frame = -1
                bounced_this_delivery = False
                post_hit_max_speed = 0.0

        # ---- Hit & Miss Status Updates ----
        if event_status == "BOWLED" and not hit_occurred:
            if _try_detect_hit(hist, height):
                hit_occurred = True
                event_status = "POST_HIT"
                post_hit_max_speed = math.hypot(hist[-1][0] - hist[-2][0], hist[-1][1] - hist[-2][1])
                if bounced_this_delivery:
                    _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces, 'RUNS')
                else:
                    hp = hist[-3]
                    speed = _compute_delivery_speed(raw_list, fps, height)
                    _add_delivery_marker(
                        hp[0], hp[1], h_matrix, 'RUNS', frame_index,
                        job_bounces, session_bounces, persistent_video_bounces,
                        speed_kmh=speed)
                    bounced_this_delivery = True
                print(f"[Frame {frame_index}] HIT — marked RUN")

        if event_status == "POST_HIT" and len(hist) >= 2:
            speed = math.hypot(hist[-1][0] - hist[-2][0], hist[-1][1] - hist[-2][1])
            post_hit_max_speed = max(post_hit_max_speed, speed)
            if post_hit_max_speed > 32 or hist[-1][1] < int(height * 0.25):
                if persistent_video_bounces and persistent_video_bounces[-1]['label'] == 'RUNS':
                    _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces, 'BOUNDARIES')
                    print(f"[Frame {frame_index}] Boundary")

        if event_status == "BOWLED" and bounced_this_delivery and len(hist) >= 3:
            if hist[-1][1] > int(height * 0.82) and not hit_occurred:
                event_status = "MISS"
                _relable_last_bounce(job_bounces, session_bounces, persistent_video_bounces, 'WICKETS')
                print(f"[Frame {frame_index}] MISS")

        if event_status == "POST_HIT" and best_coords:
            lx, ly = best_coords
            if (lx < -20 or lx > width+20 or ly < -20 or ly > height+20) and frames_since_det > int(fps * 0.8):
                _close_delivery(
                    list(raw_history), height, h_matrix, frame_index,
                    hit_occurred, event_status, bounced_this_delivery,
                    job_bounces, session_bounces, persistent_video_bounces, fps=fps)
                history.clear()
                raw_history.clear()
                kf = BallKalmanFilter(dt=dt)
                hit_occurred = False
                event_status = "WAITING"
                bounce_detected = False
                bounce_frame = -1
                bounced_this_delivery = False

        # ---- DRAWING: pitch labels + meters + ball dots (always on) ----
        map_bounces_for_overlay = [{'coords': b['coords'], 'type': b['type']} for b in job_bounces]
        if len(job_bounces) != cached_bounce_count:
            cached_live_panel = build_panel_image(
                [{'coords': b['coords'], 'type': b['type'],
                  'length': b.get('length'), 'speed_kmh': b.get('speed_kmh', 0)} for b in job_bounces],
                'PITCH MAP', panel_w)
            cached_bounce_count = len(job_bounces)

        composite_pitch_zones(frame, zone_color, zone_mask, alpha=pitch_alpha)
        nz = pitch_annotations.max(axis=2) > 0
        frame[nz] = pitch_annotations[nz]

        if cached_live_panel is not None:
            blit_panel(frame, cached_live_panel)
        else:
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
    _close_delivery(
        list(raw_history), height, h_matrix, frame_index,
        hit_occurred, event_status, bounced_this_delivery,
        job_bounces, session_bounces, persistent_video_bounces, fps=fps)
    _set_job_progress(job_id, 95, frame_index, total_frames)

    # ---- END SUMMARY: centred pitch map ----
    map_panel_bounces = [{'coords': b['coords'], 'type': b.get('type', 'DOTS'),
                          'length': b.get('length'), 'speed_kmh': b.get('speed_kmh', 0)} for b in job_bounces]
    # Always show at least empty pitch map template if no bounces detected
    if not map_panel_bounces:
        map_panel_bounces = [{'coords': (360, 400), 'type': 'DOTS'}]

    summary_count = 0
    end_panel = build_panel_image(map_panel_bounces, 'PITCH MAP', min(440, width // 2))
    if frame is not None:
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

    converted = output_path.replace('.mp4','_converted.mp4')
    cmd = ['ffmpeg','-y','-i',output_path,'-c:v','libx264','-preset','ultrafast','-crf','23',
           '-pix_fmt','yuv420p','-movflags','+faststart',converted]
    subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if os.path.exists(converted):
        os.remove(output_path)
        os.replace(converted, output_path)

    ball_stats = bounce_stats(job_bounces)
    bounce_events = [
        {'coords': list(b['coords']), 'type': b['type'], 'length': b.get('length'),
         'speed_kmh': b.get('speed_kmh', 0), 'frame': b.get('frame', 0)}
        for b in job_bounces
    ]
    return {
        'frames_processed': frame_index,
        'hit_detected': hit_occurred,
        'event_status': event_status,
        'output_path': output_path,
        'bounce_events': bounce_events,
        'ball_stats': ball_stats,
    }

def process_video_async(job_id, input_path, output_path):
    try:
        result = process_video(input_path, output_path, job_id=job_id)
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
    with jobs_lock:
        jobs[job_id] = {'status': 'queued', 'result': None, 'error': None, 'queued_at': time.time()}
    job_queue.put((job_id, input_path, output_path))
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
        })
    if job['status']=='error': return jsonify({'status':'error','error':job['error']}),500
    res = job['result']
    return jsonify({'status':'done','video_url':f"/video/{os.path.basename(res['output_path'])}",'summary':res})

@app.route('/video/<filename>')
def video(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path): return jsonify({'error':'File not found'}),404
    return send_file(path, mimetype='video/mp4')

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
    })

if __name__ == '__main__':
    try:
        _get_yolo()
    except Exception as exc:
        print(f"[WARN] Model preload failed: {exc}")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)