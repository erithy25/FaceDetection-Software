"""
Face Detection & Preprocessing Module — MediaPipe Face Mesh.

Extracts faces from video frames using MediaPipe's 468-landmark
face mesh. Handles face cropping with padding, normalization to
EfficientNet input format, and feature extraction (blink detection,
head pose estimation).
"""

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# EfficientNet-B0 input dimensions
CROP_SIZE = 224
# Padding around the detected face bounding box (20% as per spec)
FACE_PADDING = 0.20
# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Number of consecutive no-face frames before entering standby
NO_FACE_THRESHOLD = 10

# Eye landmark indices for blink detection (MediaPipe Face Mesh)
# Left eye: top=159, bottom=145; Right eye: top=386, bottom=374
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Blink detection threshold (eye aspect ratio)
BLINK_EAR_THRESHOLD = 0.21


class FaceDetector:
    """MediaPipe-based face detection with landmark extraction."""

    def __init__(self) -> None:
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._no_face_count = 0
        self._blink_state = False  # True = eyes closed
        self._blink_frames_since_last = 0
        self._blink_rate = 0.0

    def detect(
        self, frame: np.ndarray
    ) -> Optional[tuple[np.ndarray, np.ndarray, dict]]:
        """Detect a face and extract features from a single frame.

        Args:
            frame: BGR image from OpenCV (640x480).

        Returns:
            Tuple of (face_crop, landmarks, features) or None if no face found.
            - face_crop: normalized 224x224 RGB array (float32, ImageNet-normalized)
            - landmarks: array of 468 (x, y, z) landmark coordinates
            - features: dict with blink rate, head pose, etc.
        """
        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._no_face_count += 1
            if self._no_face_count > NO_FACE_THRESHOLD:
                return None  # Enter standby mode
            return None

        self._no_face_count = 0
        face_landmarks = results.multi_face_landmarks[0]

        h, w = frame.shape[:2]

        # Extract landmark coordinates as numpy array
        landmarks = np.array(
            [(lm.x * w, lm.y * h, lm.z * w) for lm in face_landmarks.landmark],
            dtype=np.float32,
        )

        # Compute face bounding box from landmarks
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        x_min, x_max = float(x_coords.min()), float(x_coords.max())
        y_min, y_max = float(y_coords.min()), float(y_coords.max())

        # Add padding (20%)
        pad_w = (x_max - x_min) * FACE_PADDING
        pad_h = (y_max - y_min) * FACE_PADDING
        x_min = max(0, int(x_min - pad_w))
        y_min = max(0, int(y_min - pad_h))
        x_max = min(w, int(x_max + pad_w))
        y_max = min(h, int(y_max + pad_h))

        # Crop and resize face
        face_bgr = frame[y_min:y_max, x_min:x_max]
        if face_bgr.size == 0:
            return None

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(
            face_rgb, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA
        )

        # Normalize to ImageNet mean/std
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_normalized = (face_normalized - IMAGENET_MEAN) / IMAGENET_STD

        # Extract features
        features = self._extract_features(landmarks)

        return face_normalized, landmarks, features

    def _extract_features(self, landmarks: np.ndarray) -> dict:
        """Extract auxiliary features from facial landmarks.

        Returns dict with:
            - blink: blink rate (higher = more natural)
            - head_yaw, head_pitch: head rotation angles
        """
        # Eye aspect ratio for blink detection
        left_ear = self._eye_aspect_ratio(
            landmarks[LEFT_EYE_TOP], landmarks[LEFT_EYE_BOTTOM]
        )
        right_ear = self._eye_aspect_ratio(
            landmarks[RIGHT_EYE_TOP], landmarks[RIGHT_EYE_BOTTOM]
        )
        avg_ear = (left_ear + right_ear) / 2.0

        # Track blink state transitions
        eyes_closed = avg_ear < BLINK_EAR_THRESHOLD
        self._blink_frames_since_last += 1

        if eyes_closed and not self._blink_state:
            # Blink started
            self._blink_state = True
        elif not eyes_closed and self._blink_state:
            # Blink ended — compute rate
            self._blink_state = False
            if self._blink_frames_since_last > 0:
                # Natural blink interval: 15–40 frames at 30 FPS (0.5–1.3s)
                interval = self._blink_frames_since_last
                if 10 <= interval <= 120:
                    self._blink_rate = min(1.0, 0.5 + 0.5 * (1.0 - abs(interval - 45) / 45))
                else:
                    self._blink_rate = max(0.0, self._blink_rate - 0.1)
            self._blink_frames_since_last = 0

        # Simple head pose from nose and chin landmarks
        nose = landmarks[1]
        chin = landmarks[152]
        forehead = landmarks[10]

        # Approximate yaw from nose-to-center offset
        face_center_x = (landmarks[:, 0].min() + landmarks[:, 0].max()) / 2
        yaw = (nose[0] - face_center_x) / (landmarks[:, 0].max() - landmarks[:, 0].min() + 1e-6)

        # Approximate pitch from vertical nose position
        face_center_y = (forehead[1] + chin[1]) / 2
        pitch = (nose[1] - face_center_y) / (chin[1] - forehead[1] + 1e-6)

        return {
            "blink": self._blink_rate,
            "head_yaw": float(yaw),
            "head_pitch": float(pitch),
        }

    @staticmethod
    def _eye_aspect_ratio(top: np.ndarray, bottom: np.ndarray) -> float:
        """Compute simplified eye aspect ratio from top and bottom landmarks."""
        return float(np.linalg.norm(top[:2] - bottom[:2]))
