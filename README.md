# 👁️ AVSS: Autonomous Visual Surveillance System

An asynchronous, edge-optimized surveillance node built with **FastAPI**, **YOLOv8**, and **ByteTrack**. It replaces heuristic pixel-differencing algorithms with 2D vector tripwires, deep multi-target spatial tracking, rolling RAM pre-roll recording buffers, and privacy-first facial allowlisting.

---

## 🚀 Key Features

* **Producer-Consumer Frame Pipeline:** Dedicated background daemon thread handles camera acquisition into a single-frame ring buffer, preventing OS-level frame drops.
* **Vector Tripwire Line-Crossing:** Evaluates 2D planar cross-products to determine **Ingress vs. Egress** direction with zero coordinate brittleness.
* **Pre/Post-Roll RAM Ring Buffer:** Maintains rolling footage in memory. On verified intrusion, it dumps a 10-second clip (5s pre-event + 5s post-event) without blocking real-time video streams.
* **Allowlist Identity Verification:** Uses lightweight OpenCV deep learning models (`YuNet` + `SFace`) to distinguish authorized household residents from strangers.
* **Secure Telemetry Alerts:** Dispatches encrypted incident snapshots to private Telegram channels using an isolated two-factor authorization model.
* **Local Audit Logging:** High-concurrency event telemetry cataloged using SQLite in WAL mode.

---

## 🏗️ System Architecture

```text
[ Optical Sensor (USB / RTSP) ]
               │
               ▼ (Asynchronous Daemon Thread)
[ Threaded Ring-Buffer Ingestion ]
               │
               ├─────────────────────────────────────────┐
               ▼                                         ▼
[ YOLOv8 Deep Inference (ONNX) ]         [ In-Memory Pre-Roll Ring Buffer ]
               │                                         │
               ▼                                         │
[ ByteTrack Multi-Target Association ]                   │
               │                                         │
               ▼                                         │
[ 2D Vector Tripwire Engine ]                            │
               │                                         │
      (Crossing Triggered)                               │
               │                                         │
               ├── (If Stranger) ──► Dump 10s MP4 Clip ──┘
               │
               ├── Persist Event to SQLite (surveillance.db)
               │
               └── Asynchronous Telegram Push Notification

## 🚀 Quickstart Guide

Follow these steps to run the surveillance node locally on your machine.

### 1. Prerequisites
* **Python 3.10+**
* Connected webcam (ID: `0`) or an active RTSP feed

### 2. Clone the Repository
```bash
git clone [https://github.com/Jag-riti-Pal/AVSS.git](https://github.com/Jag-riti-Pal/AVSS.git)
cd AVSS

### 3. Install Dependencies
Bash
pip install -r requirements.txt

### 4. Launch the Surveillance Engine
Bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

### 5. Access the Dashboard
Plaintext
[http://127.0.0.1:8000/](http://127.0.0.1:8000/)
