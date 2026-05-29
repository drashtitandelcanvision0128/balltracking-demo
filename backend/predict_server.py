import math
import os
import threading
import time
import uuid
import subprocess

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'train5', 'weights', 'best.pt')
# Lazy load model inside processing if required, or keep globally safely
model = YOLO(MODEL_PATH)

FRAME_STRIDE = 2
jobs = {}
jobs_lock = threading.Lock()


class FixedSizeQueue:
    def __init__(self, max_size):
        self.queue = []
        self.max_size = max_size

    def add(self, item):
        self.queue.append(item)
        if len(self.queue) > self.max_size:
            self.queue.pop(0)

    def pop(self):
        if self.queue:
            return self.queue.pop(0)
        return None

    def clear(self):
        self.queue.clear()

    def get_queue(self):
        return self.queue

    def __len__(self):
        return len(self.queue)


def angle_between_lines(m1, m2=1):
    try:
        if m1 != -1 / m2:
            return math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
    except ZeroDivisionError:
        pass
    return 90.0

def draw_dashed_line(img, pt1, pt2, color, thickness=1, gap=15):
    dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if dist == 0:
        return
    dashes = max(1, int(dist / gap))
    for i in range(dashes):
        start_t = i / dashes
        end_t = (i + 0.5) / dashes
        start_x = int(pt1[0] + start_t * (pt2[0] - pt1[0]))
        start_y = int(pt1[1] + start_t * (pt2[1] - pt1[1]))
        end_x = int(pt1[0] + end_t * (pt2[0] - pt1[0]))
        end_y = int(pt1[1] + end_t * (pt2[1] - pt1[1]))
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness, cv2.LINE_AA)

def draw_ui_panel(img, text_title, text_value, top_left, size=(120, 55)):
    overlay = img.copy()
    x, y = top_left
    w, h = size
    
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), 1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    cv2.putText(img, text_title, (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, text_value, (x + 10, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

def draw_pitch_line(img):
    height, width = img.shape[:2]
    pt1 = [width // 2 - 30, height]          
    pt2 = [width // 2 + 30, height]          
    pt3 = [width // 2 + 10, height // 2]     
    pt4 = [width // 2 - 10, height // 2]     
    
    pts = np.array([pt1, pt2, pt3, pt4], np.int32).reshape((-1, 1, 2))
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], (255, 100, 50)) 
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)


def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {input_path}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    centroid_history = FixedSizeQueue(100)
    interval = 0.6
    start_time = time.time()
    angle = 0.0
    frame_index = 0
    bounce_events = []
    current_detections = []
    
    speed_mph = 0.0
    spin_deg = 0.0
    swing_sf = 0.0
    frames_since_last_detect = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw_pitch_line(frame)

        current_time = time.time()
        if current_time - start_time >= interval and len(centroid_history) > 0:
            centroid_history.pop()
            start_time = current_time

        if frame_index % FRAME_STRIDE == 0:
            results = model.track(frame, persist=True, conf=0.25, verbose=False)
            current_detections = []

            if results and results[0].boxes and len(results[0].boxes) > 0:
                frames_since_last_detect = 0
                box = results[0].boxes[0] # Most confident element
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                current_detections.append((x1, y1, x2, y2, centroid_x, centroid_y))
                centroid_history.add((centroid_x, centroid_y))
            else:
                frames_since_last_detect += FRAME_STRIDE
                
            if frames_since_last_detect > 30:
                centroid_history.clear()
        
        for x1, y1, x2, y2, _, _ in current_detections:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        if len(centroid_history) > 2:
            centroid_list = list(centroid_history.get_queue())
            pts = np.array(centroid_list)
            
            # Filter consecutive updates smoothly
            _, idx = np.unique(pts, axis=0, return_index=True)
            pts = pts[np.sort(idx)]
            
            if len(pts) >= 4:  # Splprep needs more points than degree k (typically 3+1)
                k = min(3, len(pts) - 1)
                try:
                    tck, u = splprep([pts[:,0], pts[:,1]], s=50, k=k)
                    u_new = np.linspace(0, 1, 200)
                    x_new, y_new = splev(u_new, tck)
                    curve_pts = np.vstack((x_new, y_new)).T.astype(np.int32)
                    
                    cv2.polylines(frame, [curve_pts], isClosed=False, color=(0, 0, 0), thickness=8, lineType=cv2.LINE_AA)
                    cv2.polylines(frame, [curve_pts], isClosed=False, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
                except Exception:
                    pass # Keep going if mathematical anomaly happens

            x_diff = centroid_list[-1][0] - centroid_list[-2][0]
            y_diff = centroid_list[-1][1] - centroid_list[-2][1]
            
            if len(centroid_list) > 5:
                dy = centroid_list[-1][1] - centroid_list[-5][1]
                dx = centroid_list[-1][0] - centroid_list[-5][0]
                if dy > 0:
                    speed_mph = round(min(98.0, max(40.0, dy * 1.8)), 1)
                    swing_sf = round(min(4.5, abs(dx) * 0.12), 1)
                    spin_deg = round(min(6.0, abs(angle) * 0.08), 1)

            if abs(x_diff) > 0.01:
                m1 = y_diff / x_diff
                angle = 90 - angle_between_lines(m1) if m1 != 0 else 90
                if angle >= 45:
                    bounce_events.append((frame_index, round(angle, 2)))

            # Future Prediction Trace
            future_positions = [centroid_list[-1]]
            for i in range(1, 8):
                future_positions.append((int(centroid_list[-1][0] + x_diff * (i * 0.7)), int(centroid_list[-1][1] + y_diff * (i * 0.7))))

            for i in range(1, len(future_positions)):
                draw_dashed_line(frame, future_positions[i - 1], future_positions[i], (0, 0, 0), 4, gap=15)
                draw_dashed_line(frame, future_positions[i - 1], future_positions[i], (0, 255, 0), 2, gap=15)

        draw_ui_panel(frame, "Speed", f"{speed_mph} mph", (20, 60))
        draw_ui_panel(frame, "Spin", f"{spin_deg} deg", (20, 125))
        draw_ui_panel(frame, "Swing", f"{swing_sf} sf", (20, 190))

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    # Web-friendly H.264 Conversion step
    converted_path = output_path.replace('.mp4', '_converted.mp4')
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', output_path,
        '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', converted_path
    ]
    try:
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and os.path.exists(converted_path):
            os.replace(converted_path, output_path)
    except Exception as e:
        print('ffmpeg conversion failed safely:', e)

    return {
        'frames_processed': frame_index,
        'bounce_events': bounce_events,
        'output_path': output_path,
    }


def process_video_async(job_id, input_path, output_path):
    try:
        result = process_video(input_path, output_path)
        with jobs_lock:
            jobs[job_id]['status'] = 'done'
            jobs[job_id]['result'] = {
                'frames_processed': result['frames_processed'],
                'bounce_events': result['bounce_events'],
                'output_path': output_path,
            }
    except Exception as exc:
        with jobs_lock:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error'] = str(exc)


@app.route('/')
def index():
    return jsonify({
        'status': 'API is running',
        'message': 'Please use the Next.js frontend at http://localhost:3000 to interact with the predictor.'
    })


@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    input_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(input_path)

    processed_name = f"processed_{os.path.splitext(unique_name)[0]}.mp4"
    output_path = os.path.join(UPLOAD_FOLDER, processed_name)

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            'status': 'processing',
            'result': None,
            'error': None,
        }

    thread = threading.Thread(target=process_video_async, args=(job_id, input_path, output_path), daemon=True)
    thread.start()

    return jsonify({
        'status': 'processing',
        'job_id': job_id,
        'poll_url': f'/status/{job_id}',
        'input_file': input_path,
        'output_file': output_path,
    }), 202


@app.route('/status/<job_id>')
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        
    if not job:
        return jsonify({'status': 'error', 'error': 'Job not found'}), 404

    if job['status'] == 'processing':
        return jsonify({'status': 'processing'})

    if job['status'] == 'error':
        return jsonify({'status': 'error', 'error': job['error']}), 500

    result = job['result']
    output_path = result['output_path']
    filename = os.path.basename(output_path)
    return jsonify({
        'status': 'done',
        'video_url': f'/video/{filename}',
        'download_url': f'/download/{filename}',
        'summary': {
            'frames_processed': result['frames_processed'],
            'bounce_events': result['bounce_events'],
        },
    })


@app.route('/video/<filename>')
def video(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(file_path, mimetype='video/mp4')


@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(file_path, as_attachment=True, download_name=filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)