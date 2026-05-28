import math
import os
import threading
import time
import uuid

import cv2
from flask import Flask, jsonify, request, send_file
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join('runs', 'detect', 'train5', 'weights', 'best.pt')
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
            self.queue.pop(0)

    def clear(self):
        self.queue.clear()

    def get_queue(self):
        return self.queue

    def __len__(self):
        return len(self.queue)


def angle_between_lines(m1, m2=1):
    if m1 != -1 / m2:
        return math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
    return 90.0


def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {input_path}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Write as mp4v first; then convert with ffmpeg to H.264 for browser preview.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    centroid_history = FixedSizeQueue(10)
    interval = 0.6
    start_time = time.time()
    angle = 0.0
    frame_index = 0
    bounce_events = []
    current_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if current_time - start_time >= interval and len(centroid_history) > 0:
            centroid_history.pop()
            start_time = current_time

        if frame_index % FRAME_STRIDE == 0:
            results = model.track(frame, persist=True, conf=0.35, verbose=False)
            boxes = results[0].boxes
            current_detections = []

            if len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    centroid_x = int((x1 + x2) / 2)
                    centroid_y = int((y1 + y2) / 2)
                    current_detections.append((x1, y1, x2, y2, centroid_x, centroid_y))
                    centroid_history.add((centroid_x, centroid_y))
        
        for x1, y1, x2, y2, centroid_x, centroid_y in current_detections:
            cv2.circle(frame, (centroid_x, centroid_y), radius=3, color=(0, 0, 255), thickness=-1)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        if len(centroid_history) > 1:
            centroid_list = list(centroid_history.get_queue())
            for i in range(1, len(centroid_history)):
                cv2.line(frame, centroid_list[i - 1], centroid_list[i], (255, 0, 0), 4)

            x_diff = centroid_list[-1][0] - centroid_list[-2][0]
            y_diff = centroid_list[-1][1] - centroid_list[-2][1]

            if x_diff != 0:
                m1 = y_diff / x_diff
                if m1 == 1:
                    angle = 90
                elif m1 != 0:
                    angle = 90 - angle_between_lines(m1)
                if angle >= 45:
                    bounce_events.append((frame_index, round(angle, 2)))

            future_positions = [centroid_list[-1]]
            for i in range(1, 5):
                future_positions.append((centroid_list[-1][0] + x_diff * i, centroid_list[-1][1] + y_diff * i))

            for i in range(1, len(future_positions)):
                cv2.line(frame, future_positions[i - 1], future_positions[i], (0, 255, 0), 4)
                cv2.circle(frame, future_positions[i], radius=3, color=(0, 0, 255), thickness=-1)

        cv2.putText(frame, f"Angle: {angle:.2f} degrees", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    # --- ffmpeg H.264 conversion for browser preview ---
    import subprocess
    converted_path = output_path.replace('.mp4', '_converted.mp4')
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', output_path,
        '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', converted_path
    ]
    try:
        result = subprocess.run(
            ffmpeg_cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and os.path.exists(converted_path):
            if os.path.exists(output_path):
                os.remove(output_path)
            os.replace(converted_path, output_path)
        else:
            print('ffmpeg conversion failed:', result.returncode)
            if result.stderr:
                print(result.stderr)
    except Exception as e:
        print('ffmpeg conversion failed:', e)

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
    return """
    <html>
      <head>
        <title>Cricket Ball Trajectory Predictor</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 24px; line-height: 1.5; }
          #app { max-width: 900px; margin: 0 auto; }
          #status { margin: 12px 0; font-weight: bold; }
          video { width: 100%; max-width: 800px; margin-top: 12px; border-radius: 8px; }
          .card { border: 1px solid #ccc; border-radius: 10px; padding: 18px; margin-top: 16px; }
          .summary { margin-top: 10px; color: #444; }
        </style>
      </head>
      <body>
        <div id="app">
          <h1>Cricket Ball Trajectory Predictor</h1>
          <p>Upload a cricket video and the processed prediction video will be shown right on this page.</p>
          <form id="predictForm" class="card" enctype="multipart/form-data">
            <input type="file" name="video" accept="video/*" required>
            <button type="submit">Upload and Predict</button>
          </form>
          <div id="status"></div>
          <div id="result"></div>
        </div>

        <script>
          const form = document.getElementById('predictForm');
          const status = document.getElementById('status');
          const result = document.getElementById('result');

          async function pollJob(jobId) {
            const pollInterval = setInterval(async () => {
              try {
                const response = await fetch(`/status/${jobId}`);
                const data = await response.json();

                if (data.status === 'processing') {
                  status.textContent = 'Processing video...';
                  return;
                }

                clearInterval(pollInterval);

                if (data.status === 'error') {
                  status.textContent = data.error || 'Prediction failed';
                  return;
                }

                const bounceEvents = Array.isArray(data.summary?.bounce_events) ? data.summary.bounce_events.length : 0;
                status.textContent = `Processed ${data.summary?.frames_processed || 0} frames. Detected ${bounceEvents} bounce events.`;

                result.innerHTML = `
                  <div class="card">
                    <h2>Predicted Video</h2>
                    <video controls autoplay playsinline>
                      <source src="${data.video_url}" type="video/mp4">
                    </video>
                    <p class="summary">Processed video is playing inline. Use the download link below if you want to save it.</p>
                    <p><a href="${data.download_url}">Download predicted video</a></p>
                  </div>
                `;
              } catch (error) {
                clearInterval(pollInterval);
                status.textContent = 'Polling failed: ' + error.message;
              }
            }, 2000);
          }

          form.addEventListener('submit', async (event) => {
            event.preventDefault();
            status.textContent = 'Starting upload...';
            result.innerHTML = '';

            const formData = new FormData(form);
            try {
              const response = await fetch('/predict', {
                method: 'POST',
                body: formData
              });

              const data = await response.json();

              if (!response.ok) {
                status.textContent = data.error || 'Prediction failed';
                return;
              }

              if (data.status === 'processing') {
                status.textContent = 'Video queued for processing...';
                pollJob(data.job_id);
                return;
              }

              const bounceEvents = Array.isArray(data.summary?.bounce_events) ? data.summary.bounce_events.length : 0;
              status.textContent = `Processed ${data.summary?.frames_processed || 0} frames. Detected ${bounceEvents} bounce events.`;

              result.innerHTML = `
                <div class="card">
                  <h2>Predicted Video</h2>
                  <video controls autoplay playsinline>
                    <source src="${data.video_url}" type="video/mp4">
                  </video>
                  <p class="summary">Processed video is playing inline. Use the download link below if you want to save it.</p>
                  <p><a href="${data.download_url}">Download predicted video</a></p>
                </div>
              `;
            } catch (error) {
              status.textContent = 'Upload failed: ' + error.message;
            }
          });
        </script>
      </body>
    </html>
    """


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
    app.run(host='0.0.0.0', port=5000)
