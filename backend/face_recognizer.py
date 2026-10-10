"""
Module: face_recognizer.py
Purpose: Resident authentication using OpenCV's built-in YuNet (detector) + SFace (recognizer).
Dependencies: opencv-python-headless, numpy (Zero C++ compilers / Zero dlib required).
"""
import os
import glob
from typing import Dict, List, Tuple
import cv2
import numpy as np


class ResidentAuthenticator:
    def __init__(
        self,
        yunet_model_path: str = "face_detection_yunet.onnx",
        sface_model_path: str = "face_recognition_sface.onnx",
        database_dir: str = "data/residents",
        cosine_threshold: float = 0.363  # SFace default threshold for cosine similarity
    ):
        self.database_dir = database_dir
        self.cosine_threshold = cosine_threshold

        # Initialize YuNet detector
        self.detector = cv2.FaceDetectorYN.create(
            model=yunet_model_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )

        # Initialize SFace embedding extractor & matcher
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=sface_model_path,
            config=""
        )

        # In-memory resident embeddings: { "person_name": [embedding_vector, ...] }
        self.enrolled_embeddings: Dict[str, List[np.ndarray]] = {}
        self._load_resident_database()

    def _load_resident_database(self) -> None:
        """Indexes resident profile folders and computes SFace embeddings."""
        if not os.path.exists(self.database_dir):
            os.makedirs(self.database_dir, exist_ok=True)
            return

        for person_name in os.listdir(self.database_dir):
            person_path = os.path.join(self.database_dir, person_name)
            if not os.path.isdir(person_path):
                continue

            self.enrolled_embeddings[person_name] = []
            img_paths = glob.glob(os.path.join(person_path, "*.*"))

            for path in img_paths:
                img = cv2.imread(path)
                if img is None:
                    continue

                h, w = img.shape[:2]
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(img)

                if faces is not None and len(faces) > 0:
                    aligned_face = self.recognizer.alignCrop(img, faces[0])
                    feature = self.recognizer.feature(aligned_face)
                    self.enrolled_embeddings[person_name].append(feature)

        print(f"Indexed {len(self.enrolled_embeddings)} resident profiles via OpenCV SFace.")

    def identify_person(self, frame_bgr: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> Tuple[str, bool]:
        """
        Extracts face within tracker bbox and matches against enrolled residents.
        Returns: (Name/Label, is_resident)
        """
        x1, y1, x2, y2 = bbox_xyxy
        h_frame, w_frame = frame_bgr.shape[:2]

        # Crop target region (focusing on upper half for face detection)
        crop_y2 = min(y2, y1 + int((y2 - y1) * 0.6))
        crop = frame_bgr[max(0, y1):crop_y2, max(0, x1):min(w_frame, x2)]

        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return "UNKNOWN", False

        h_crop, w_crop = crop.shape[:2]
        self.detector.setInputSize((w_crop, h_crop))
        _, faces = self.detector.detect(crop)

        if faces is None or len(faces) == 0:
            return "UNKNOWN (No Face Detected)", False

        # Extract 128D feature vector for detected face
        aligned_face = self.recognizer.alignCrop(crop, faces[0])
        query_feature = self.recognizer.feature(aligned_face)

        best_match_name = "STRANGER"
        max_similarity = -1.0

        # Compare against enrolled database via Cosine Similarity
        for name, enc_list in self.enrolled_embeddings.items():
            for ref_feature in enc_list:
                similarity = self.recognizer.match(
                    query_feature, ref_feature, cv2.FaceRecognizerSF_FR_COSINE
                )
                if similarity > max_similarity:
                    max_similarity = similarity
                    if similarity >= self.cosine_threshold:
                        best_match_name = name

        is_resident = (best_match_name != "STRANGER")
        return best_match_name, is_resident