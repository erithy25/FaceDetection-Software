"""
Deepfake Classifier — EfficientNet-B0 (frame-level) + LSTM (temporal).

Handles ONNX-based inference for both the frame-level EfficientNet
classifier and the temporal LSTM layer. Manages the embedding buffer
for temporal analysis.

Model outputs:
  - EfficientNet ONNX: output[0] = logit (1,1), output[1] = embedding (1,256)
  - LSTM ONNX: output[0] = logit (1,1)

Both logits are passed through sigmoid to get [0,1] scores.
Score meaning: 1.0 = definitely real, 0.0 = definitely fake.
"""

import os
import logging
import time
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger("silentwitness.classifier")

# Model file paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
EFFICIENTNET_MODEL = os.path.join(MODEL_DIR, "efficientnet_b0_deepfake.onnx")
EFFICIENTNET_MODEL_INT8 = os.path.join(MODEL_DIR, "efficientnet_b0_deepfake_int8.onnx")
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

        # Performance tracking
        self._inference_times: deque[float] = deque(maxlen=100)

    def load_models(self) -> None:
        """Load ONNX models for inference.

        Prefers INT8 quantized model if available, falls back to FP32,
        then falls back to dummy inference if no models are found.
        """
        try:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            # Use all available CPU threads
            sess_options.intra_op_num_threads = 0

            # Load EfficientNet (prefer INT8)
            eff_path = None
            if os.path.exists(EFFICIENTNET_MODEL_INT8):
                eff_path = EFFICIENTNET_MODEL_INT8
                logger.info("Using INT8 quantized EfficientNet model")
            elif os.path.exists(EFFICIENTNET_MODEL):
                eff_path = EFFICIENTNET_MODEL
                logger.info("Using FP32 EfficientNet model")

            if eff_path:
                self._efficientnet_session = ort.InferenceSession(
                    eff_path,
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )
                # Validate model outputs
                outputs = self._efficientnet_session.get_outputs()
                logger.info(
                    f"EfficientNet loaded: {len(outputs)} outputs "
                    f"({', '.join(o.name for o in outputs)})"
                )
            else:
                logger.warning(
                    f"No EfficientNet model found in {MODEL_DIR}, "
                    "using dummy inference (random scores)"
                )

            # Load LSTM
            if os.path.exists(LSTM_MODEL):
                self._lstm_session = ort.InferenceSession(
                    LSTM_MODEL,
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )
                logger.info("LSTM model loaded")
            else:
                logger.warning(
                    f"LSTM model not found at {LSTM_MODEL}, "
                    "using dummy temporal inference"
                )

        except ImportError:
            logger.warning(
                "ONNX Runtime not installed. Install with: "
                "pip install onnxruntime. Using dummy inference."
            )

    def classify_frame(self, face_crop: np.ndarray) -> tuple[float, np.ndarray]:
        """Classify a single face crop as real or fake.

        Args:
            face_crop: Normalized 224x224 RGB face image (float32, ImageNet-normalized).

        Returns:
            Tuple of (frame_score, embedding).
            - frame_score: probability of being real [0.0-1.0]
            - embedding: 256-dim feature vector for temporal analysis
        """
        t0 = time.perf_counter()

        if self._efficientnet_session is not None:
            # Prepare input: (1, 3, 224, 224) — NCHW format
            input_tensor = np.transpose(face_crop, (2, 0, 1))[np.newaxis].astype(
                np.float32
            )

            input_name = self._efficientnet_session.get_inputs()[0].name
            outputs = self._efficientnet_session.run(None, {input_name: input_tensor})

            # outputs[0] = classification logit (1, 1), outputs[1] = embedding (1, 256)
            frame_score = float(_sigmoid(outputs[0][0, 0]))
            embedding = outputs[1][0].astype(np.float32)
        else:
            # Dummy inference for development without trained models
            frame_score = 0.5 + np.random.normal(0, 0.05)
            frame_score = float(np.clip(frame_score, 0.0, 1.0))
            embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            # Simulate inference latency
            time.sleep(0.005)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._inference_times.append(elapsed_ms)

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
            Temporal confidence score [0.0-1.0].
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

    @property
    def avg_inference_ms(self) -> float:
        """Average inference time in milliseconds (last 100 frames)."""
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)

    def reset(self) -> None:
        """Reset classifier state (embedding buffer, counters)."""
        self._embedding_buffer.clear()
        self._frame_counter = 0
        self.last_frame_score = 0.5
        self.last_embedding = None
        self._last_temporal_score = 0.5
        self._inference_times.clear()


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
