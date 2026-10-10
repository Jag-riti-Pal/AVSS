"""
Module: capture.py
Purpose: Thread-isolated hardware frame ingestion with thread-safe atomic swaps.
"""
import threading
import time
from typing import Optional, Tuple, Union
import cv2
import numpy as np


class VideoCaptureDevice:
    def __init__(self, source: Union[int, str] = 0, target_fps: int = 30):
        self.source = source
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps
        
        self.cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._is_running: bool = False
        
        # Concurrency primitives
        self._lock: threading.Lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "VideoCaptureDevice":
        """Initializes the optical sensor and starts the ingestion daemon."""
        if self._is_running:
            return self

        self.cap = cv2.VideoCapture(self.source)
        # Force a minimal internal buffer at the driver level to reduce frame lag
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to bind optical stream at source: {self.source}")

        self._is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        """Daemon worker continually draining the hardware sensor buffer."""
        while self._is_running:
            start_time = time.monotonic()
            
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue

            success, frame = self.cap.read()
            if success and frame is not None:
                with self._lock:
                    # Atomic reference pointer update
                    self._latest_frame = frame
            
            # Rate limit to avoid pegging CPU cores unnecessarily
            elapsed = time.monotonic() - start_time
            sleep_time = max(0.0, self.frame_delay - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Safely retrieves the most recent frame pointer from the buffer."""
        with self._lock:
            if self._latest_frame is None:
                return False, None
            # Return an isolated copy to prevent memory race conditions
            return True, self._latest_frame.copy()

    def get_source_metrics(self) -> dict:
        """Extracts low-level optical metadata from the hardware driver."""
        if not self.cap or not self.cap.isOpened():
            return {"connected": False}
        return {
            "connected": True,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "hardware_fps": float(self.cap.get(cv2.CAP_PROP_FPS)),
            "source": str(self.source)
        }

    def stop(self) -> None:
        """Gracefully halts the ingestion thread and releases device descriptors."""
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self._latest_frame = None