"""
Module: main.py
Purpose: Complete Smart CCTV 2.0 microservice with tracking, SQLite persistence, and event buffering.
"""
import asyncio
from contextlib import asynccontextmanager
import os
import shutil
import time
from typing import Generator

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from backend.capture import VideoCaptureDevice
from backend.database import SessionLocal, SurveillanceEvent, init_db
from backend.notifier import AlertDispatcher
from backend.recorder import EventRecorder
from backend.tracker import VisualAnalyticsPipeline

# 1. Initialize SQLite Database Schema
init_db()

# 2. Pipeline Subsystems
DEFAULT_TRIPWIRE = ((320, 0), (320, 480))
camera_device = VideoCaptureDevice(source=0, target_fps=30)
analytics_pipeline = VisualAnalyticsPipeline(tripwire_line=DEFAULT_TRIPWIRE)
event_recorder = EventRecorder(buffer_seconds=5, fps=30)
# Safely reads from environment variables; defaults to None if not yet set
alert_dispatcher = AlertDispatcher(
    telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
    telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID")
)


def record_incident_to_db(event_type: str, identity_label: str, is_resident: bool, clip_path: str = None) -> None:
    """Writes crossing incident metadata to SQLite asynchronously."""
    db = SessionLocal()
    try:
        record = SurveillanceEvent(
            event_type=event_type,
            identity_label=identity_label,
            is_resident=1 if is_resident else 0,
            clip_path=clip_path
        )
        db.add(record)
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[DATABASE LOG ERROR]: {exc}")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        camera_device.start()
        print("Optical ingestion daemon initialized.")
    except Exception as exc:
        print(f"Optical sensor initialization error: {exc}")
    yield
    camera_device.stop()
    print("Optical sensor released cleanly.")


app = FastAPI(
    title="Smart CCTV 2.0 - Core Engine",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def mjpeg_frame_generator() -> Generator[bytes, None, None]:
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
    
    while True:
        success, raw_frame = camera_device.read_frame()
        if not success or raw_frame is None:
            time.sleep(0.01)  # Frame starvation protection
            continue

        # Step A: Maintain continuous 5s pre-event RAM buffer
        event_recorder.push_frame(raw_frame)

        # Step B: Run tracking and extract crossing events
        # Note: If your process_frame currently only returns the frame, update tracker.py
        # to return (annotated_frame, event_info) or inspect the tripwire directly.
        processed_frame = analytics_pipeline.process_frame(raw_frame)

        # Step C: Check if a crossing event was logged by the tripwire
        # (Assuming your pipeline exposes the latest triggered crossing)
        latest_event = getattr(analytics_pipeline, "latest_event", None)
        if latest_event:
            event_type = latest_event.get("type")           # e.g., "INGRESS" or "EGRESS"
            identity = latest_event.get("identity", "STRANGER")
            is_resident = latest_event.get("is_resident", False)

            # Clear event so it only triggers once per crossing
            analytics_pipeline.latest_event = None

            # If stranger enters or an ingress violation occurs, dump ring buffer clip
            clip_path = None
            if not is_resident:
                clip_path = event_recorder.trigger_event_recording(
                    event_label=f"{identity}_{event_type}",
                    post_roll_seconds=5
                )
                
                # Only attempt notification if Telegram credentials exist
                if getattr(alert_dispatcher, "tg_token", None) and getattr(alert_dispatcher, "tg_chat_id", None):
                    import threading
                    threading.Thread(
                        target=lambda: asyncio.run(
                            alert_dispatcher.send_stranger_alert(
                                event_name=f"{identity} {event_type}",
                                clip_path=clip_path
                            )
                        ),
                        daemon=True
                    ).start()
                else:
                    # Safe fallback: logs to terminal without raising unawaited coroutine warnings
                    print(f"[SECURITY EVENT LOGGED]: {identity} {event_type} | Clip: {clip_path}")

            # Persist event in SQLite
            record_incident_to_db(
                event_type=event_type,
                identity_label=identity,
                is_resident=is_resident,
                clip_path=clip_path
            )

        # Step D: Compress and yield multipart stream
        ret, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n'
            b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' +
            frame_bytes + b'\r\n'
        )


@app.get("/")
def serve_dashboard():
    html_path = os.path.join("frontend", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="frontend/index.html not found.")
    return FileResponse(html_path)


@app.get("/api/v1/stream")
def get_live_stream():
    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/v1/metrics")
def get_tripwire_metrics():
    ingress = analytics_pipeline.tripwire.ingress_count
    egress = analytics_pipeline.tripwire.egress_count
    return {
        "ingress": ingress,
        "egress": egress,
        "net_occupancy": max(0, ingress - egress)
    }


@app.post("/api/v1/metrics/reset")
def reset_tripwire_metrics():
    analytics_pipeline.tripwire.reset_counts()
    return {"status": "SUCCESS", "message": "Counts reset."}


@app.get("/api/v1/telemetry")
def get_camera_telemetry():
    metrics = camera_device.get_source_metrics()
    if not metrics.get("connected"):
        raise HTTPException(status_code=503, detail="Camera sensor unavailable")
    return metrics


@app.get("/api/v1/events/recent")
def get_recent_events(limit: int = 8):
    """Provides recent incident history for the web dashboard."""
    db = SessionLocal()
    try:
        events = (
            db.query(SurveillanceEvent)
            .order_by(SurveillanceEvent.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": ev.id,
                "timestamp": ev.timestamp.strftime("%H:%M:%S") if ev.timestamp else "--",
                "event_type": ev.event_type,
                "identity": ev.identity_label,
                "is_resident": bool(ev.is_resident),
                "clip_path": ev.clip_path
            }
            for ev in events
        ]
    finally:
        db.close()


@app.post("/api/v1/residents/enroll")
async def enroll_resident(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    """Enrolls an authorized resident into data/residents/<name>/."""
    clean_name = name.strip().replace(" ", "_").lower()
    target_dir = os.path.join("data", "residents", clean_name)
    os.makedirs(target_dir, exist_ok=True)

    file_extension = os.path.splitext(file.filename)[1] or ".jpg"
    dest_path = os.path.join(target_dir, f"profile_{int(time.time())}{file_extension}")

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if hasattr(analytics_pipeline, "authenticator"):
        analytics_pipeline.authenticator._load_resident_database()

    return {
        "status": "SUCCESS",
        "message": f"Successfully enrolled resident: {clean_name}",
        "stored_file": dest_path
    }


@app.get("/api/v1/residents")
def list_enrolled_residents():
    """Returns all authorized identities currently stored in the database."""
    base_dir = os.path.join("data", "residents")
    if not os.path.exists(base_dir):
        return {"residents": []}
    
    residents = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    return {"residents": residents}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)