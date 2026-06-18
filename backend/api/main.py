"""
FastAPI production API for Cricket Ball Tracking & Pitch Mapping.

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure backend root is on path
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from api.schemas import (
    AnalyticsResponse,
    HeatmapRequest,
    JobStatusResponse,
    MatchCreate,
    MatchResponse,
    PlayerCreate,
    PlayerResponse,
    SessionCreate,
    SessionResponse,
)
from core.analytics import compute_session_analytics, player_statistics
from core.config import CONFIG
from core.heatmap import generate_all_heatmaps, generate_heatmap_grid, heatmap_to_base64, render_heatmap_image
from db.database import get_db, init_db
from db.repository import AnalyticsRepository

# Import video processing from existing server
import predict_server as ps

app = FastAPI(
    title="Cricket Ball Tracking API",
    description="Production-grade ball tracking, bounce detection, pitch mapping, and analytics",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    from core.gpu_runtime import init_gpu_runtime
    init_gpu_runtime()
    init_db()
    try:
        ps._get_yolo()
    except Exception as exc:
        print(f"[WARN] YOLO preload failed: {exc}")


# ---- Health ----
@app.get("/health")
def health():
    gpu = ps._gpu_name
    return {
        "status": "ok",
        "version": "2.0.0",
        "api": "fastapi",
        "model_exists": os.path.exists(ps.MODEL_PATH),
        "gpu": gpu,
        "device": str(ps._yolo_device),
        "features": {
            "ball_detection": True,
            "bounce_detection": True,
            "pitch_mapping": True,
            "line_length_classification": True,
            "pose_estimation": True,
            "heatmaps": True,
            "database": True,
            "trajectory_lines": False,
            "future_prediction": False,
        },
    }


# ---- Video Upload & Processing ----
@app.post("/api/v1/videos/upload", status_code=202)
async def upload_video(
    video: UploadFile = File(...),
    match_id: str | None = None,
    session_id: str | None = None,
    bowler_id: str | None = None,
    pitch_calibration: str | None = None,
    db: Session = Depends(get_db),
):
    """Upload video and queue for ball tracking processing."""
    ext = os.path.splitext(video.filename or "")[1].lower()
    allowed = CONFIG["processing"]["supported_formats"]
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format. Allowed: {allowed}")

    repo = AnalyticsRepository(db)
    if not session_id and match_id:
        session = repo.create_session(match_id, bowler_id=bowler_id)
        session_id = session.id
    elif not session_id:
        match = repo.create_match(name=f"Upload {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")
        session = repo.create_session(match.id, bowler_id=bowler_id)
        session_id = session.id

    safe_name = f"{uuid.uuid4().hex}_{video.filename}"
    input_path = os.path.join(ps.UPLOAD_FOLDER, safe_name)
    os.makedirs(ps.UPLOAD_FOLDER, exist_ok=True)

    content = await video.read()
    with open(input_path, "wb") as f:
        f.write(content)

    output_name = f"processed_{os.path.splitext(safe_name)[0]}.mp4"
    output_path = os.path.join(ps.UPLOAD_FOLDER, output_name)

    repo.update_session_status(session_id, "queued")
    if session := repo.get_session(session_id):
        session.video_path = input_path

    job_id = str(uuid.uuid4())
    options: dict[str, Any] = {}
    if pitch_calibration:
        try:
            import json
            quad = json.loads(pitch_calibration)
            if isinstance(quad, list) and len(quad) == 4:
                options["manual_quad"] = quad
        except Exception as exc:
            print(f"[PitchCalib] Invalid manual quad: {exc}")

    with ps.jobs_lock:
        ps.jobs[job_id] = {
            "status": "queued",
            "result": None,
            "error": None,
            "session_id": session_id,
            "bowler_id": bowler_id,
            "queued_at": __import__("time").time(),
        }
    ps.job_queue.put((job_id, input_path, output_path, options))
    ps._drain_queue_except(job_id)

    return {
        "job_id": job_id,
        "session_id": session_id,
        "status": "queued",
        "queue_position": 1 if ps._current_job_id is None else ps.job_queue.qsize(),
    }


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    with ps.jobs_lock:
        job = ps.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if job["status"] == "queued":
        pos = 1 if ps._current_job_id is None else max(1, ps.job_queue.qsize())
        return JobStatusResponse(status="queued", queue_position=pos)

    if job["status"] == "processing":
        return JobStatusResponse(
            status="processing",
            progress=job.get("progress", 0),
            frame=job.get("frame", 0),
            total_frames=job.get("total_frames", 0),
            pass_info=job.get("pass_info"),
        )

    if job["status"] == "error":
        return JobStatusResponse(status="error", error=job.get("error"))

    res = job["result"]
    session_id = job.get("session_id")
    if session_id and res:
        _persist_job_results(db, session_id, job.get("bowler_id"), res)

    video_url = None
    report_pdf_url = None
    if res and res.get("output_path"):
        video_url = f"/api/v1/videos/{os.path.basename(res['output_path'])}"
    if res and res.get("report_pdf_url"):
        report_pdf_url = res["report_pdf_url"].replace("/report/", "/api/v1/reports/pdf/")

    return JobStatusResponse(
        status="done", video_url=video_url, report_pdf_url=report_pdf_url,
        summary=res, progress=100,
    )


def _persist_job_results(db: Session, session_id: str, bowler_id: str | None, result: dict):
    repo = AnalyticsRepository(db)
    repo.update_session_status(session_id, "completed", result.get("output_path"))
    existing = repo.get_balls(session_id=session_id)
    if existing:
        return
    for i, bounce in enumerate(result.get("bounce_events", []), 1):
        bounce["ball_number"] = i
        bounce["bowler_id"] = bowler_id
        repo.save_ball(session_id, bounce)


@app.get("/api/v1/videos/{filename}")
def stream_video(filename: str):
    path = os.path.join(ps.UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Video not found")
    return FileResponse(path, media_type="video/mp4")


# ---- Matches & Sessions ----
@app.post("/api/v1/matches", response_model=MatchResponse)
def create_match(body: MatchCreate, db: Session = Depends(get_db)):
    repo = AnalyticsRepository(db)
    match = repo.create_match(body.name, body.venue)
    return match


@app.get("/api/v1/matches", response_model=list[MatchResponse])
def list_matches(db: Session = Depends(get_db)):
    repo = AnalyticsRepository(db)
    return repo.list_matches()


@app.get("/api/v1/matches/{match_id}")
def get_match(match_id: str, db: Session = Depends(get_db)):
    repo = AnalyticsRepository(db)
    match = repo.get_match(match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    return match


@app.post("/api/v1/sessions", response_model=SessionResponse)
def create_session(body: SessionCreate, db: Session = Depends(get_db)):
    repo = AnalyticsRepository(db)
    if not repo.get_match(body.match_id):
        raise HTTPException(404, "Match not found")
    session = repo.create_session(body.match_id, body.name, body.bowler_id)
    return session


# ---- Ball & Bounce Data ----
@app.get("/api/v1/balls")
def get_ball_data(
    session_id: str | None = None,
    match_id: str | None = None,
    bowler_id: str | None = None,
    db: Session = Depends(get_db),
):
    repo = AnalyticsRepository(db)
    balls = repo.get_balls(session_id=session_id, match_id=match_id, bowler_id=bowler_id)
    if balls:
        return {"balls": repo.balls_to_dicts(balls), "total": len(balls)}

    # Fallback to in-memory session bounces
    return {
        "balls": [
            {
                "coords": list(b["coords"]),
                "type": b["type"],
                "length": b.get("length"),
                "speed_kmh": b.get("speed_kmh", 0),
            }
            for b in ps.session_bounces
        ],
        "total": len(ps.session_bounces),
        "source": "memory",
    }


@app.get("/api/v1/bounces")
def get_bounce_data(
    session_id: str | None = None,
    match_id: str | None = None,
    bowler_id: str | None = None,
    db: Session = Depends(get_db),
):
    repo = AnalyticsRepository(db)
    balls = repo.get_balls(session_id=session_id, match_id=match_id, bowler_id=bowler_id)
    bounces = []
    for b in repo.balls_to_dicts(balls):
        if b.get("bounce_x") is not None:
            bounces.append({
                "bounce_x": b["bounce_x"],
                "bounce_y": b["bounce_y"],
                "confidence": b.get("bounce_confidence", 0.9),
                "length_type": b.get("length_type"),
                "line_type": b.get("line_type"),
                "frame": b.get("frame"),
            })
    if not bounces and ps.session_bounces:
        from core.pitch_coords import pitchmap_to_world
        for b in ps.session_bounces:
            x_m, y_m = pitchmap_to_world(b["coords"][0], b["coords"][1])
            bounces.append({"bounce_x": x_m, "bounce_y": y_m, "confidence": 0.85})
    return {"bounces": bounces, "total": len(bounces)}


# ---- Analytics ----
@app.get("/api/v1/analytics", response_model=AnalyticsResponse)
def get_analytics(
    session_id: str | None = None,
    match_id: str | None = None,
    bowler_id: str | None = None,
    db: Session = Depends(get_db),
):
    repo = AnalyticsRepository(db)
    balls = repo.get_balls(session_id=session_id, match_id=match_id, bowler_id=bowler_id)
    if balls:
        data = repo.balls_to_dicts(balls)
    else:
        from core.pitch_coords import pitchmap_to_world
        data = []
        for b in ps.session_bounces:
            x_m, y_m = pitchmap_to_world(b["coords"][0], b["coords"][1])
            data.append({**b, "bounce_x": x_m, "bounce_y": y_m, "length": b.get("length")})
    return compute_session_analytics(data)


@app.get("/api/v1/players/{player_id}/statistics")
def get_player_stats(player_id: str, db: Session = Depends(get_db)):
    repo = AnalyticsRepository(db)
    player = repo.get_player(player_id)
    if not player:
        raise HTTPException(404, "Player not found")
    balls = repo.get_balls(bowler_id=player_id)
    return player_statistics(repo.balls_to_dicts(balls), player_id)


# ---- Players ----
@app.post("/api/v1/players", response_model=PlayerResponse)
def create_player(body: PlayerCreate, db: Session = Depends(get_db)):
    repo = AnalyticsRepository(db)
    return repo.create_player(body.name, body.role, body.style)


@app.get("/api/v1/players", response_model=list[PlayerResponse])
def list_players(db: Session = Depends(get_db)):
    repo = AnalyticsRepository(db)
    return repo.list_players()


# ---- Heatmaps ----
@app.get("/api/v1/heatmaps")
def get_heatmaps(
    session_id: str | None = None,
    match_id: str | None = None,
    bowler_id: str | None = None,
    zone_filter: str = Query("all"),
    format: str = Query("json", description="json or image"),
    db: Session = Depends(get_db),
):
    repo = AnalyticsRepository(db)
    balls = repo.get_balls(session_id=session_id, match_id=match_id, bowler_id=bowler_id)
    data = repo.balls_to_dicts(balls) if balls else [
        {"coords": b["coords"], "length": b.get("length"), "type": b["type"]}
        for b in ps.session_bounces
    ]

    if format == "image":
        img = render_heatmap_image(data, zone_filter=zone_filter)
        import cv2
        path = os.path.join(ps.UPLOAD_FOLDER, f"heatmap_{zone_filter}_{uuid.uuid4().hex[:8]}.jpg")
        cv2.imwrite(path, img)
        return FileResponse(path, media_type="image/jpeg")

    if zone_filter == "all":
        return generate_all_heatmaps(data)
    return generate_heatmap_grid(data, zone_filter=zone_filter)


@app.get("/api/v1/heatmaps/image")
def get_heatmap_image(
    session_id: str | None = None,
    zone_filter: str = "all",
    db: Session = Depends(get_db),
):
    repo = AnalyticsRepository(db)
    balls = repo.get_balls(session_id=session_id)
    data = repo.balls_to_dicts(balls) if balls else [
        {"coords": b["coords"], "length": b.get("length")}
        for b in ps.session_bounces
    ]
    img = render_heatmap_image(data, zone_filter=zone_filter)
    return JSONResponse({"image": heatmap_to_base64(img)})


# ---- Pitch Map ----
@app.get("/api/v1/pitchmap")
def get_pitchmap(bowler: str = "Bowler", db: Session = Depends(get_db)):
    import cv2
    from pitch_map_renderer import create_hawkeye_template, render_pitch_map

    output_path = os.path.join(ps.UPLOAD_FOLDER, "session_pitchmap.jpg")
    bounces = [
        {"coords": b["coords"], "type": b["type"]}
        for b in ps.session_bounces
    ]
    base = create_hawkeye_template(bowler)
    map_img = render_pitch_map(bounces, bowler_name=bowler, base_img=base)
    cv2.imwrite(output_path, map_img)
    return FileResponse(output_path, media_type="image/jpeg")


@app.get("/api/v1/pitchmap/data")
def get_pitchmap_data(session_id: str | None = None, db: Session = Depends(get_db)):
    repo = AnalyticsRepository(db)
    if session_id:
        balls = repo.get_balls(session_id=session_id)
        return {"bounces": repo.balls_to_dicts(balls), "total": len(balls)}
    return {
        "bounces": [{"coords": list(b["coords"]), "type": b["type"]} for b in ps.session_bounces],
        "total": len(ps.session_bounces),
    }


@app.post("/api/v1/pitchmap/reset")
def reset_pitchmap():
    ps.session_bounces.clear()
    return {"status": "cleared"}


# ---- Reports Export ----
@app.get("/api/v1/reports/export")
def export_report(
    session_id: str | None = None,
    match_id: str | None = None,
    format: str = Query("json"),
    db: Session = Depends(get_db),
):
    repo = AnalyticsRepository(db)
    balls = repo.get_balls(session_id=session_id, match_id=match_id)
    data = repo.balls_to_dicts(balls)
    analytics = compute_session_analytics(data)
    heatmaps = generate_all_heatmaps(data)
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "match_id": match_id,
        "analytics": analytics,
        "balls": data,
        "heatmaps": heatmaps,
    }
    if format == "csv":
        import csv
        import io
        buf = io.StringIO()
        if data:
            writer = csv.DictWriter(buf, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(buf.getvalue(), media_type="text/csv")
    if format == "pdf":
        from core.pdf_report import generate_session_pdf
        summary = {
            "bounce_events": data,
            "ball_stats": {
                "total": len(data),
                "dots": sum(1 for b in data if b.get("type") == "DOTS"),
                "runs": sum(1 for b in data if b.get("type") == "RUNS"),
                "boundaries": sum(1 for b in data if b.get("type") == "BOUNDARIES"),
                "wickets": sum(1 for b in data if b.get("type") == "WICKETS"),
            },
            "analytics": analytics,
            "speed_stats": {},
            "clips": [],
            "processing_mode": "session-export",
        }
        pdf_path = os.path.join(ps.UPLOAD_FOLDER, f"session_report_{session_id or match_id or 'all'}.pdf")
        generate_session_pdf(pdf_path, summary)
        return FileResponse(pdf_path, media_type="application/pdf", filename="cricket_report.pdf")
    return report


@app.get("/api/v1/reports/pdf/{job_id}")
def download_job_report(job_id: str):
    path = os.path.join(ps.UPLOAD_FOLDER, f"report_{job_id}.pdf")
    if not os.path.exists(path):
        raise HTTPException(404, "Report not found")
    return FileResponse(path, media_type="application/pdf", filename=f"cricket_report_{job_id[:8]}.pdf")


# ---- Legacy Flask-compatible routes ----
@app.post("/predict", status_code=202)
async def legacy_predict(video: UploadFile = File(...)):
    return await upload_video(video)


@app.get("/status/{job_id}")
def legacy_status(job_id: str, db: Session = Depends(get_db)):
    return get_job_status(job_id, db)


@app.get("/video/{filename}")
def legacy_video(filename: str):
    return stream_video(filename)
