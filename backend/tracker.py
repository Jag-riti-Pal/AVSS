"""
Module: tracker.py
Purpose: Deep learning perception pipeline combining YOLOv8 inference with ByteTrack.
"""
from typing import Tuple
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from backend.tripwire import VectorTripwire


class VisualAnalyticsPipeline:
    def __init__(self, tripwire_line: Tuple[Tuple[int, int], Tuple[int, int]]):
        # 1. YOLOv8 Object Detector (ONNX model for CPU inference)
        self.detector = YOLO("yolov8n.onnx", task="detect")

        # 2. ByteTrack Multi-Target Tracker (Clean instantiation)
        # Using default supervision ByteTrack arguments to avoid deprecation warnings
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )

        # 3. Tripwire geometry engine
        self.tripwire = VectorTripwire(tripwire_line[0], tripwire_line[1])

        # 4. Supervision Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)

        # State cache for passing latest crossing event to main.py
        self.latest_event = None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Run YOLO inference filtering for person class (ID 0)
        results = self.detector(frame, classes=[0], verbose=False, conf=0.3)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Update ByteTrack tracking
        tracked_detections = self.tracker.update_with_detections(detections)

        active_track_ids = set()

        if len(tracked_detections) > 0 and tracked_detections.tracker_id is not None:
            for i in range(len(tracked_detections)):
                track_id = tracked_detections.tracker_id[i]
                if track_id is None:
                    continue

                active_track_ids.add(track_id)
                x1, y1, x2, y2 = tracked_detections.xyxy[i]

                # Bottom-center ground contact point
                bottom_center = (int((x1 + x2) / 2), int(y2))

                # Check tripwire crossing
                crossing = self.tripwire.update(track_id, bottom_center)
                if crossing:
                    # Cache event for main.py to trigger recording & alerts
                    self.latest_event = {
                        "type": crossing,
                        "identity": f"Person_{track_id}",
                        "is_resident": False
                    }

        # Purge stale tracking IDs
        self.tripwire.purge_stale_tracks(active_track_ids)

        # Draw overlays
        annotated_frame = frame.copy()

        # Render tripwire line
        cv2.line(
            annotated_frame,
            (int(self.tripwire.p1.x), int(self.tripwire.p1.y)),
            (int(self.tripwire.p2.x), int(self.tripwire.p2.y)),
            (0, 215, 255), 3, cv2.LINE_AA
        )
        cv2.putText(
            annotated_frame, "TRIPWIRE",
            (int(self.tripwire.p1.x) + 5, int(self.tripwire.p1.y) + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1, cv2.LINE_AA
        )

        # Annotate detections
        if len(tracked_detections) > 0 and tracked_detections.tracker_id is not None:
            labels = [f"ID #{tid}" for tid in tracked_detections.tracker_id]
            annotated_frame = self.trace_annotator.annotate(annotated_frame, detections=tracked_detections)
            annotated_frame = self.box_annotator.annotate(annotated_frame, detections=tracked_detections)
            annotated_frame = self.label_annotator.annotate(annotated_frame, detections=tracked_detections, labels=labels)

        # On-screen HUD Card
        cv2.rectangle(annotated_frame, (10, 10), (250, 90), (15, 23, 42), -1)
        cv2.rectangle(annotated_frame, (10, 10), (250, 90), (51, 65, 85), 1)

        cv2.putText(
            annotated_frame, f"INGRESS (IN):  {self.tripwire.ingress_count}",
            (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (52, 211, 153), 2, cv2.LINE_AA
        )
        cv2.putText(
            annotated_frame, f"EGRESS  (OUT): {self.tripwire.egress_count}",
            (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (248, 113, 113), 2, cv2.LINE_AA
        )

        return annotated_frame