"""
Cricket Ball Tracker — Robust Reset (No Spider Web)
====================================================
- NEW DELIVERY: if ball undetected >2.5s (except POST_HIT) → history cleared
- Trajectory drawn ONLY from actual YOLO detections (smoothed)
- Kalman prediction green dot shown during occlusion – NOT in trajectory
- Bounce detection, hit detection, post-hit tracking preserved
"""

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

# ---------- Smoothed History (only actual detections) ----------
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

def draw_pitch_line(img):
    h,w = img.shape[:2]
    pts = np.array([[w//2-40,h],[w//2+40,h],[w//2+15,h//2],[w//2-15,h//2]], np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay,[pts],(255,120,50))
    cv2.addWeighted(overlay,0.35,img,0.65,0,img)

def draw_trajectory(frame, pts_list, outer_col, inner_col, outer_thick, inner_thick):
    if len(pts_list) < 2: return
    pts = np.array(pts_list, dtype=np.float32)
    _, idx = np.unique(pts, axis=0, return_index=True)
    pts = pts[np.sort(idx)]
    if len(pts) < 2: return
    try:
        k = min(3, len(pts)-1)
        tck,_ = splprep([pts[:,0], pts[:,1]], s=5, k=k)
        u_new = np.linspace(0,1, max(100, len(pts)*3))
        x_new, y_new = splev(u_new, tck)
        curve = np.vstack((x_new,y_new)).T.astype(np.int32)
        cv2.polylines(frame, [curve], False, outer_col, outer_thick, cv2.LINE_AA)
        cv2.polylines(frame, [curve], False, inner_col, inner_thick, cv2.LINE_AA)
    except:
        cv2.polylines(frame, [pts.astype(np.int32)], False, outer_col, outer_thick, cv2.LINE_AA)

# ---------- Core processing ----------
def process_video(input_path, output_path):
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
    last_centroid = None
    frames_since_det = 999

    event_status   = "WAITING"
    hit_occurred   = False
    hit_split_idx  = -1
    speed_kph      = 0.0
    bounce_detected = False
    bounce_frame   = -1
    frame_index    = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_index += 1
        frames_since_det += 1
        best_coords = None
        best_score = -float('inf')

        # ---- Detection (TIGHTENED) ----
        if model is not None:
            results = model.predict(frame, conf=0.25, imgsz=640, verbose=False)
            for box in results[0].boxes:
                if int(box.cls[0]) != 0: continue
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                bw, bh = x2-x1, y2-y1
                area = bw * bh
                if area < 30 or area > 5000:          # ball size filter
                    continue
                aspect = bw / (bh + 1e-5)
                if not (0.3 < aspect < 3.0):          # circular-ish
                    continue
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                
                score = float(box.conf[0])

                # Proximity guard & tracking score
                if kf.initialized and frames_since_det <= 3:
                    px, py = kf.get_position()
                    dist = math.hypot(cx-px, cy-py)
                    if dist > max(150, width*0.15):
                        continue
                    # Penalize score based on distance from predicted position
                    # This prevents the tracker from jumping to higher-confidence false positives (like gloves/pads)
                    score -= dist * 0.005

                if score > best_score:
                    best_score = score
                    best_coords = (cx, cy)

        # ---- Update Kalman & Smoothed History ----
        if best_coords is not None:
            cx, cy = best_coords
            if not kf.initialized:
                kf.init(cx, cy)
            else:
                kf.predict()
                kf.correct(cx, cy)

            # Jump reset (>250px) – new delivery
            if len(history) > 0:
                last_pt = history.get_list()[-1]
                if math.hypot(cx - last_pt[0], cy - last_pt[1]) > 250:
                    history.clear()
                    kf = BallKalmanFilter(dt=dt)
                    kf.init(cx, cy)
                    hit_occurred = False; hit_split_idx = -1
                    event_status = "BOWLED"
                    speed_kph = 0.0; bounce_detected = False; bounce_frame = -1
                    print(f"[Frame {frame_index}] Jump reset – new delivery")

            history.add((cx, cy))
            last_centroid = (cx, cy)
            frames_since_det = 0
            
            if event_status == "WAITING":
                event_status = "BOWLED"
        else:
            if kf.initialized: kf.predict()

        hist = history.get_list()

        # ---- Bounce Detection ----
        if not bounce_detected and event_status in ("WAITING","BOWLED") and len(hist)>=3:
            dy1 = hist[-1][1] - hist[-2][1]
            dy2 = hist[-2][1] - hist[-3][1]
            if dy1 < -3 and dy2 > 3 and hist[-2][1] > height*0.4:
                bounce_detected = True
                bounce_frame = frame_index
                print(f"[Frame {frame_index}] Bounce detected")

        # ---- New Delivery Reset (gap >2.5s, except POST_HIT) ----
        if frames_since_det > int(fps * 2.5) and event_status != "POST_HIT":
            if event_status != "WAITING" or len(history) > 0:
                history.clear()
                kf = BallKalmanFilter(dt=dt)
                hit_occurred = False; hit_split_idx = -1
                event_status = "WAITING"
                speed_kph = 0.0; bounce_detected = False; bounce_frame = -1
                print(f"[Frame {frame_index}] New delivery (gap >2.5s)")

        # ---- Hit Detection ----
        if event_status=="BOWLED" and bounce_detected and len(hist)>=4 and not hit_occurred:
            p1,p2,p3,p4 = hist[-4], hist[-3], hist[-2], hist[-1]
            v_pre  = (p3[0]-p1[0], p3[1]-p1[1])
            v_post = (p4[0]-p2[0], p4[1]-p2[1])
            mag_pre = math.hypot(*v_pre); mag_post = math.hypot(*v_post)
            if mag_pre>5 and mag_post>5:
                cos_a = max(-1,min(1, (v_pre[0]*v_post[0]+v_pre[1]*v_post[1])/(mag_pre*mag_post)))
                angle = math.degrees(math.acos(cos_a))
                vy_avg = p4[1]-p1[1]
                if angle>20.0 and p3[1]>int(height*0.35) and vy_avg<8.0:
                    hit_occurred = True
                    hit_split_idx = len(hist)-2
                    event_status = "POST_HIT"
                    print(f"[Frame {frame_index}] HIT! angle={angle:.1f}°")

        # ---- MISS Detection ----
        if event_status=="BOWLED" and len(hist)>=3:
            if hist[-1][1] > int(height*0.82):
                event_status = "MISS"
                print(f"[Frame {frame_index}] MISS")

        # ---- POST_HIT Reset ----
        if event_status == "POST_HIT" and last_centroid:
            lx, ly = last_centroid
            ball_outside = (lx < -20 or lx > width+20 or ly < -20 or ly > height+20)
            if ball_outside and frames_since_det > int(fps * 1.0):
                history.clear()
                kf = BallKalmanFilter(dt=dt)
                hit_occurred = False; hit_split_idx = -1
                event_status = "WAITING"
                speed_kph = 0.0; bounce_detected = False; bounce_frame = -1
                print(f"[Frame {frame_index}] Ball left frame – reset")

        # ---- Drawing ----
        # if hit_occurred and 0 < hit_split_idx < len(hist):
        #     pre  = hist[:hit_split_idx+1]
        #     post = hist[hit_split_idx:]
        #     draw_trajectory(frame, pre,  (0,0,0), (0,0,255), 5,2)
        #     draw_trajectory(frame, post, (0,0,0), (0,255,255), 7,3)
        # else:
        #     draw_trajectory(frame, hist, (0,0,0), (0,0,255), 5,2)
        
        if best_coords is not None:
            draw_trajectory(frame, hist, (0,0,0), (0,0,255), 5,2)

        if best_coords:
            cv2.circle(frame, best_coords, 7, (0,255,255), -1)

        status_color = {"WAITING":(128,128,128),"BOWLED":(0,200,255),
                        "POST_HIT":(0,255,128),"MISS":(0,80,255)}.get(event_status,(255,255,255))
        draw_ui_panel(frame,"STATUS",event_status,(15,15),value_color=status_color)
        draw_ui_panel(frame,"PTS",str(len(hist)),(185,15))
        draw_ui_panel(frame,"BOUNCE","YES" if bounce_detected else "NO",(355,15))
        draw_ui_panel(frame,"FRAME",str(frame_index),(525,15))

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

    return {'frames_processed':frame_index,'hit_detected':hit_occurred,
            'event_status':event_status,'output_path':output_path}

def process_video_async(job_id, input_path, output_path):
    try:
        result = process_video(input_path, output_path)
        with jobs_lock: jobs[job_id] = {'status':'done','result':result}
    except Exception as exc:
        with jobs_lock: jobs[job_id] = {'status':'error','error':str(exc)}

# Flask routes (unchanged)
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

@app.route('/health')
def health(): return jsonify({'status':'ok','model_exists':os.path.exists(MODEL_PATH)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)