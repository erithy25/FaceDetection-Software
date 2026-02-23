"""
Deepfake Classifier — EfficientNet-B0 (frame-level) + LSTM (temporal).

Handles ONNX-based inference for both the frame-level EfficientNet
classifier and the temporal LSTM layer. Manages the embedding buffer
for temporal analysis and provides Grad-CAM heatmap generation.
"""

import os
import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger("silentwitness.classifier")

# Model file paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
EFFICIENTNET_MODEL = os.path.join(MODEL_DIR, "efficientnet_b0_deepfake.onnx")
LSTM_MODEL = os.path.join(MODEL_DIR, "temporal_lstm.onnx")

# Architecture constants
EMBEDDING_DIM = 256
LSTM_HIDDEN_DIM = 128
TEMPORAL_WINDOW = 16  # Number of frame embeddings for LSTM input
TEMPORAL_STRIDE = 8   # Process every 8 frames (50% overlap)


class DeepfakeClassifier:
    """EfficientNet-B0 + LSTM deepfake classifier with ONNX Runtime inference."""

    def __init__(self) -> None:
        self._efficientnet_session = None
        self._lstm_session = None

        # Embedding buffer for temporal analysis
        self._embedding_buffer: deque[np.ndarray] = deque(maxlen=TEMPORAL_WINDOW)
        self._frame_counter = 0

        # Cache last results for frame-skipping
        self.last_frame_score: float = 0.5
        self.last_embedding: Optional[np.ndarray] = None
        self._last_temporal_score: float = 0.5

        # LSTM hidden state (carried between windows)
        self._lstm_hidden: Optional[np.ndarray] = None
        self._lstm_cell: Optional[np.ndarray] = None

    def load_models(self) -> None:
        """Load ONNX models for inference.

        Falls back to dummy inference if models are not found (development mode).
        """
        try:
            import onnxruntime as ort

            if os.path.exists(EFFICIENTNET_MODEL):
                self._efficientnet_session = ort.InferenceSession(
                    EFFICIENTNET_MODEL,
                    providers=["CPUExecutionProvider"],
                )
                logger.info("EfficientNet model loaded")
            else:
                logger.warning(
                    f"EfficientNet model not found at {EFFICIENTNET_MODEL}, "
                    "using dummy inference"
                )

            if os.path.exists(LSTM_MODEL):
                self._lstm_session = ort.InferenceSession(
                    LSTM_MODEL,
                    providers=["CPUExecutionProvider"],
                )
                logger.info("LSTM model loaded")
            else:
                logger.warning(
                    f"LSTM model not found at {LSTM_MODEL}, "
                    "using dummy inference"
                )

        except ImportError:
            logger.warning("ONNX Runtime not installed, using dummy inference")

    def classify_frame(self, face_crop: np.ndarray) -> tuple[float, np.ndarray]:
        """Classify a single face crop as real or fake.

        Args:
            face_crop: Normalized 224x224 RGB face image (float32).

        Returns:
            Tuple of (frame_score, embedding).
            - frame_score: probability of being real [0.0–1.0]
            - embedding: 256-dim feature vector for temporal analysis
        """
        if self._efficientnet_session is not None:
            # Prepare input: (1, 3, 224, 224) — NCHW format
            input_tensor = np.transpose(face_crop, (2, 0, 1))[np.newaxis].astype(
                np.float32
            )

            input_name = self._efficientnet_session.get_inputs()[0].name
            outputs = self._efficientnet_session.run(None, {input_name: input_tensor})

            # outputs[0] = classification score, outputs[1] = embedding
            frame_score = float(_sigmoid(outputs[0][0, 0]))
            embedding = outputs[1][0].astype(np.float32)
        else:
            # Dummy inference for development without trained models
            frame_score = 0.5 + np.random.normal(0, 0.05)
            frame_score = float(np.clip(frame_score, 0.0, 1.0))
            embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)

        self.last_frame_score = frame_score
        self.last_embedding = embedding
        self._embedding_buffer.append(embedding)
        self._frame_counter += 1

        return frame_score, embedding

    def analyze_temporal(self, embedding: Optional[np.ndarray]) -> float:
        """Run temporal analysis on the embedding buffer using the LSTM.

        The LSTM processes a window of 16 embeddings every 8 frames
        (50% overlap) to detect temporal inconsistencies.

        Args:
            embedding: Current frame embedding (may be None if skipped).

        Returns:
            Temporal confidence score [0.0–1.0].
        """
        # Only run LSTM every TEMPORAL_STRIDE frames and when buffer is full
        if (
            self._frame_counter % TEMPORAL_STRIDE != 0
            or len(self._embedding_buffer) < TEMPORAL_WINDOW
        ):
            return self._last_temporal_score

        # Stack embeddings into sequence: (1, 16, 256)
        sequence = np.stack(list(self._embedding_buffer), axis=0)[np.newaxis].astype(
            np.float32
        )

        if self._lstm_session is not None:
            input_name = self._lstm_session.get_inputs()[0].name
            outputs = self._lstm_session.run(None, {input_name: sequence})
            temporal_score = float(_sigmoid(outputs[0][0, 0]))
        else:
            # Dummy temporal score for development
            temporal_score = 0.5 + np.random.normal(0, 0.03)
            temporal_score = float(np.clip(temporal_score, 0.0, 1.0))

        self._last_temporal_score = temporal_score
        return temporal_score

    def generate_gradcam(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """Generate a Grad-CAM heatmap for the current face crop.

        This is only available when using PyTorch models (not ONNX).
        For ONNX inference, returns None. The Grad-CAM module handles
        this separately using a PyTorch model copy.

        Args:
            face_crop: Normalized 224x224 RGB face image.

        Returns:
            BGR heatmap overlay image or None.
        """
        # Grad-CAM requires PyTorch model with gradient computation.
        # Delegated to gradcam.py module when available.
        return None

    def reset(self) -> None:
        """Reset classifier state (embedding buffer, counters, hidden states)."""
        self._embedding_buffer.clear()
        self._frame_counter = 0
        self.last_frame_score = 0.5
        self.last_embedding = None
        self._last_temporal_score = 0.5
        self._lstm_hidden = None
        self._lstm_cell = None


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
