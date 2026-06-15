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

app = Flask(__name__)
CORS(app)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'train5', 'weights', 'best.pt')

jobs      = {}
jobs_lock = threading.Lock()
job_queue = queue.Queue()

# --- PERSISTENT PITCHMAP CONFIGURATION ---
session_bounces = []

# TODO: Calibrate these 4 points based on your camera feed layout
camera_perspective_points = np.array([
    [250, 400],  # Top Left
    [390, 400],  # Top Right
    [100, 700],  # Bottom Left
    [540, 700]   # Bottom Right
], dtype=np.float32)

template_2d_points = np.array([
    [150, 200],  # Top Left
    [450, 200],  # Top Right
    [150, 800],  # Bottom Left
    [450, 800]   # Bottom Right
], dtype=np.float32)

H_MATRIX = cv2.getPerspectiveTransform(camera_perspective_points, template_2d_points)

def transform_to_pitchmap(cam_x, cam_y):
    point = np.array([[[cam_x, cam_y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, H_MATRIX)
    return int(transformed[0][0][0]), int(transformed[0][0][1])

def video_processing_worker():
    while True:
        job_id, input_path, output_path = job_queue.get()
        with jobs_lock: jobs[job_id]['status'] = 'processing'
        process_video_async(job_id, input_path, output_path)
        job_queue.task_done()
threading.Thread(target=video_processing_worker, daemon=True).start()

# ---------- Kalman Filter (prediction only) ----------
class BallKalmanFilter:
    def __init__(self, dt=1.0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def init(self, x, y):
        self.kf.statePost = np.array([[x],[y],[0],[0]], np.float32)
        self.initialized = True
    def predict(self): return self.kf.predict()
    def correct(self, x, y): self.kf.correct(np.array([[np.float32(x)],[np.float32(y)]]))
    def get_position(self): return (int(self.kf.statePost[0]), int(self.kf.statePost[1]))

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

# ---------- Core processing ----------
def process_video(input_path, output_path):
    global session_bounces
    try: model = YOLO(MODEL_PATH)
    except Exception as e: print(f"[ERROR] Model load: {e}"); model = None

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dt     = 1.0/fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    kf = BallKalmanFilter(dt=dt)
    history = SmoothHistory(maxlen=150, smooth_window=5)
    # Raw (unsmoothed) coords parallel track — exact bounce pixel ke liye
    raw_history = deque(maxlen=150)
    frames_since_det = 999

    event_status   = "WAITING"
    hit_occurred   = False
    bounce_detected = False
    bounce_frame   = -1
    frame_index    = 0

    # Video screen par sabhi bounce points ko accumulate karne ke liye list
    persistent_video_bounces = []

    # Broadcast-style outcome color map (BGR) — image ki legend se match karta hai
    OUTCOME_COLORS = {
        'DOTS'      : (20,  20,  20),   # Black  — dot ball
        'RUNS'      : (235, 170, 50),   # Blue   — runs scored
        'BOUNDARIES': (0,   0,  220),   # Red    — boundary (4/6)
        'WICKETS'   : (230, 230, 230),  # White  — out
    }

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_index += 1
        frames_since_det += 1
        best_coords = None
        best_score = -float('inf')

        # ---- Detection ----
        if model is not None:
            results = model.predict(frame, conf=0.25, imgsz=640, verbose=False)
            for box in results[0].boxes:
                if int(box.cls[0]) != 0: continue
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                bw, bh = x2-x1, y2-y1
                area = bw * bh
                if area < 30 or area > 5000: continue
                aspect = bw / (bh + 1e-5)
                if not (0.3 < aspect < 3.0): continue
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                
                score = float(box.conf[0])

                if kf.initialized and frames_since_det <= 3:
                    px, py = kf.get_position()
                    dist = math.hypot(cx-px, cy-py)
                    if dist > max(150, width*0.15): continue
                    score -= dist * 0.005

                if score > best_score:
                    best_score = score
                    best_coords = (cx, cy)

        # ---- Update Kalman & History ----
        if best_coords is not None:
            cx, cy = best_coords
            if not kf.initialized:
                kf.init(cx, cy)
            else:
                kf.predict()
                kf.correct(cx, cy)

            if len(history) > 0:
                last_pt = history.get_list()[-1]
                if math.hypot(cx - last_pt[0], cy - last_pt[1]) > 250:
                    history.clear()
                    raw_history.clear()
                    kf = BallKalmanFilter(dt=dt)
                    kf.init(cx, cy)
                    hit_occurred = False
                    event_status = "BOWLED"
                    bounce_detected = False; bounce_frame = -1
                    print(f"[Frame {frame_index}] Jump reset")

            history.add((cx, cy))
            raw_history.append((cx, cy))  # Raw unsmoothed coordinate store
            frames_since_det = 0
            if event_status == "WAITING": event_status = "BOWLED"
        else:
            if kf.initialized: kf.predict()

        hist = history.get_list()

        # ---- Bounce Detection (raw_history pe directly — exact pixel) ----
        raw_list = list(raw_history)
        if not bounce_detected and event_status in ("WAITING","BOWLED") and len(raw_list) >= 3:
            # Raw coords pe dy compute karo — smooth history se mismatch nahi hoga
            dy1_raw = raw_list[-1][1] - raw_list[-2][1]  # Current frame: ball UP ho rahi hai (negative)
            dy2_raw = raw_list[-2][1] - raw_list[-3][1]  # Prev frame: ball DOWN aa rahi thi (positive)
            if dy1_raw < -3 and dy2_raw > 3 and raw_list[-2][1] > height * 0.4:
                bounce_detected = True
                bounce_frame = frame_index

                # ball_label pehle define karo — phir video_bounce_label assign karo
                ball_label = 'DOTS'
                if event_status == "POST_HIT" or hit_occurred: ball_label = 'RUNS'
                elif event_status == "MISS": ball_label = 'WICKETS'

                # raw_list[-2] = EXACT bounce pixel — bilkul wahi jahan ball ne ground touch kiya
                video_bounce_coords = raw_list[-2]

                # Global 2D Top-down conversion
                px_map, py_map = transform_to_pitchmap(video_bounce_coords[0], video_bounce_coords[1])

                session_bounces.append({'coords': (px_map, py_map), 'type': ball_label})
                persistent_video_bounces.append({'coords': video_bounce_coords, 'label': ball_label})
                print(f"[Frame {frame_index}] BOUNCE @ raw pixel {video_bounce_coords} | dy_down={dy2_raw:.1f} dy_up={dy1_raw:.1f}")

        # ---- Reset Logic ----
        if frames_since_det > int(fps * 2.5) and event_status != "POST_HIT":
            if event_status != "WAITING" or len(history) > 0:
                history.clear()
                raw_history.clear()
                kf = BallKalmanFilter(dt=dt)
                hit_occurred = False
                event_status = "WAITING"
                bounce_detected = False; bounce_frame = -1

        # ---- Hit & Miss Status Updates ----
        if event_status=="BOWLED" and bounce_detected and len(hist)>=4 and not hit_occurred:
            p1,p2,p3,p4 = hist[-4], hist[-3], hist[-2], hist[-1]
            v_pre  = (p3[0]-p1[0], p3[1]-p1[1])
            v_post = (p4[0]-p2[0], p4[1]-p2[1])
            if math.hypot(*v_pre)>5 and math.hypot(*v_post)>5:
                cos_a = max(-1,min(1, (v_pre[0]*v_post[0]+v_pre[1]*v_post[1])/(math.hypot(*v_pre)*math.hypot(*v_post))))
                if math.degrees(math.acos(cos_a))>20.0 and p3[1]>int(height*0.35) and (p4[1]-p1[1])<8.0:
                    hit_occurred = True
                    event_status = "POST_HIT"
                    if len(session_bounces) > 0: session_bounces[-1]['type'] = 'RUNS'
                    if len(persistent_video_bounces) > 0: persistent_video_bounces[-1]['label'] = 'RUNS'

        if event_status=="BOWLED" and len(hist)>=3 and hist[-1][1] > int(height*0.82):
            event_status = "MISS"
            if len(session_bounces) > 0: session_bounces[-1]['type'] = 'WICKETS'
            if len(persistent_video_bounces) > 0: persistent_video_bounces[-1]['label'] = 'WICKETS'

        if event_status == "POST_HIT" and best_coords:
            lx, ly = best_coords
            if (lx < -20 or lx > width+20 or ly < -20 or ly > height+20) and frames_since_det > int(fps * 1.0):
                history.clear()
                kf = BallKalmanFilter(dt=dt)
                hit_occurred = False
                event_status = "WAITING"
                bounce_detected = False; bounce_frame = -1

        # ---- DRAWING LOGIC ----
        # 1. Ball ke current detected position par small tracking dot
        if best_coords:
            cv2.circle(frame, best_coords, 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, best_coords, 6, (0, 0, 0), 1, cv2.LINE_AA)

        # 2. Broadcast-style bounce markers (Accumulated for all balls)
        for bounce in persistent_video_bounces:
            b_coords = bounce['coords']
            b_label  = bounce['label']
            dot_color = OUTCOME_COLORS.get(b_label, (20, 20, 20))
            
            # Large filled circle — same style as wagonwheel pitch map
            cv2.circle(frame, b_coords, 14, dot_color, -1, cv2.LINE_AA)
            # Dark outline for contrast (broadcast style)
            cv2.circle(frame, b_coords, 14, (10, 10, 10), 2, cv2.LINE_AA)
            # Tiny highlight dot in center (premium look)
            cv2.circle(frame, b_coords, 4, (255, 255, 255), -1, cv2.LINE_AA)

            # Outcome label text next to the dot
            label_text = {'DOTS':'DOT','RUNS':'RUN','BOUNDARIES':'BDRY','WICKETS':'OUT'}.get(b_label, '')
            tx, ty = b_coords[0] + 18, b_coords[1] + 6
            cv2.putText(frame, label_text, (tx+1, ty+1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(frame, label_text, (tx,   ty),   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        status_color = {"WAITING":(128,128,128),"BOWLED":(0,200,255),
                        "POST_HIT":(0,255,128),"MISS":(0,80,255)}.get(event_status,(255,255,255))
        draw_ui_panel(frame,"STATUS",event_status,(15,15),value_color=status_color)
        draw_ui_panel(frame,"BOUNCE","YES" if bounce_detected else "NO",(185,15))
        draw_ui_panel(frame,"FRAME",str(frame_index),(355,15))

        writer.write(frame)

    cap.release()
    writer.release()

    converted = output_path.replace('.mp4','_converted.mp4')
    cmd = ['ffmpeg','-y','-i',output_path,'-c:v','libx264','-preset','veryfast','-crf','22',
           '-pix_fmt','yuv420p','-movflags','+faststart',converted]
    subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if os.path.exists(converted):
        os.remove(output_path)
        os.replace(converted, output_path)

    return {'frames_processed':frame_index,'hit_detected':hit_occurred,'event_status':event_status,'output_path':output_path}

def process_video_async(job_id, input_path, output_path):
    try:
        result = process_video(input_path, output_path)
        with jobs_lock: jobs[job_id] = {'status':'done','result':result}
    except Exception as exc:
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
    with jobs_lock: jobs[job_id] = {'status':'queued','result':None,'error':None}
    job_queue.put((job_id, input_path, output_path))
    return jsonify({'status':'queued','job_id':job_id,'queue_position':job_queue.qsize()}),202

@app.route('/status/<job_id>')
def status(job_id):
    with jobs_lock: job = jobs.get(job_id)
    if not job: return jsonify({'error':'Job not found'}),404
    if job['status']=='queued': return jsonify({'status':'queued','queue_position':job_queue.qsize()})
    if job['status']=='processing': return jsonify({'status':'processing'})
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
    template_path = os.path.join(BASE_DIR, 'pitch_template.jpg')
    output_map_path = os.path.join(UPLOAD_FOLDER, 'session_pitchmap.jpg')
    
    if not os.path.exists(template_path):
        blank_template = np.zeros((900, 600, 3), dtype=np.uint8)
        blank_template[:] = (55, 105, 55)
        
        cv2.rectangle(blank_template, (100, 600), (500, 850), (95, 95, 95), -1)    # Short
        cv2.rectangle(blank_template, (100, 400), (500, 600), (90, 110, 140), -1)  # Good
        cv2.rectangle(blank_template, (100, 150), (500, 400), (90, 145, 90), -1)   # Full
        cv2.rectangle(blank_template, (100, 150), (500, 850), (240, 240, 240), 2)
        
        cv2.putText(blank_template, "Short", (120, 720), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 20, 20), 3)
        cv2.putText(blank_template, "Good", (120, 510), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 40, 80), 3)
        cv2.putText(blank_template, "Full", (120, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 60, 20), 3)
        cv2.imwrite(template_path, blank_template)

    map_img = cv2.imread(template_path)
    color_map = {'DOTS': (0, 0, 255), 'RUNS': (255, 0, 0), 'BOUNDARIES': (0, 255, 255), 'WICKETS': (255, 255, 255)}

    for ball in session_bounces:
        bx, by = ball['coords']
        b_type = ball['type']
        color = color_map.get(b_type, (255, 255, 255))
        bx = max(10, min(590, bx))
        by = max(10, min(890, by))
        cv2.circle(map_img, (bx, by), 9, color, -1)
        cv2.circle(map_img, (bx, by), 9, (20, 20, 20), 1)

    cv2.imwrite(output_map_path, map_img)
    return send_file(output_map_path, mimetype='image/jpeg')

@app.route('/reset_pitchmap', methods=['POST'])
def reset_pitchmap():
    global session_bounces
    session_bounces = []
    return jsonify({'status': 'Session tracking logs cleared successfully'})

@app.route('/health')
def health(): return jsonify({'status':'ok','model_exists':os.path.exists(MODEL_PATH)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)