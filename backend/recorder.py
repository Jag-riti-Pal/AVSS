"""
Module: recorder.py
Purpose: Rolling memory ring buffer and asynchronous MP4 event-clip generation.
"""
"""
In backend/recorder.py
"""
import os
import time
import threading
from datetime import datetime
import cv2

class EventRecorder:
    def __init__(self, buffer_seconds: int = 5, fps: int = 30, output_dir: str = "recordings"):
        self.fps = fps
        self.buffer_size = buffer_seconds * fps
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        from collections import deque
        self._frame_buffer = deque(maxlen=self.buffer_size)
        self._lock = threading.Lock()
        self._is_recording = False
        self._active_writer = None
        self._event_end_time = 0.0

    def push_frame(self, frame):
        with self._lock:
            self._frame_buffer.append(frame.copy())
            if self._is_recording and self._active_writer is not None:
                self._active_writer.write(frame)
                if time.time() >= self._event_end_time:
                    self._stop_active_recording()

    def _write_buffer_worker(self, frames_snapshot, filepath, post_roll_seconds):
        """Dedicated background worker that handles disk I/O without blocking streaming."""
        if not frames_snapshot:
            return
        h, w = frames_snapshot[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filepath, fourcc, float(self.fps), (w, h))

        for f in frames_snapshot:
            writer.write(f)

        with self._lock:
            self._active_writer = writer
            self._is_recording = True
            self._event_end_time = time.time() + post_roll_seconds

    def trigger_event_recording(self, event_label: str, post_roll_seconds: int = 5) -> str:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{event_label}_{timestamp_str}.mp4"
        filepath = os.path.join(self.output_dir, filename)

        with self._lock:
            frames_copy = list(self._frame_buffer)

        # Offload file creation and frame dumping to a background thread
        threading.Thread(
            target=self._write_buffer_worker,
            args=(frames_copy, filepath, post_roll_seconds),
            daemon=True
        ).start()

        return filepath

    def _stop_active_recording(self):
        self._is_recording = False
        if self._active_writer is not None:
            self._active_writer.release()
            self._active_writer = None