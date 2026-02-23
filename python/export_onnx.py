"""
ONNX Export & INT8 Quantization — EfficientNet-B0 and LSTM models.

Exports trained PyTorch models to ONNX format with optional INT8
post-training static quantization via ONNX Runtime. Validates that
the exported ONNX models produce outputs consistent with the original
PyTorch models.

Usage:
    python export_onnx.py \
        --efficientnet_model path/to/efficientnet.pt \
        --lstm_model path/to/lstm.pt \
        --calibration_dir path/to/face_crops/ \
        --output_dir path/to/output/
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Shared model architecture (single source of truth)
from models import (
    EfficientNetDeepfake,
    EfficientNetExportWrapper,
    TemporalLSTM,
    load_efficientnet_deepfake,
    load_temporal_lstm,
    EMBEDDING_DIM,
    LSTM_HIDDEN_DIM,
    TEMPORAL_WINDOW,
    EFFICIENTNET_INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

logger = logging.getLogger("silentwitness.export_onnx")

# Validation tolerance
ATOL_FP32 = 1e-5
ATOL_INT8 = 0.05

# Calibration settings
CALIBRATION_SAMPLE_COUNT = 500


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_efficientnet(
    model_path: str,
    output_path: str,
) -> str:
    """Export the EfficientNet-B0 model to ONNX.

    Args:
        model_path: Path to the trained PyTorch .pt checkpoint.
        output_path: Path for the output .onnx file.

    Returns:
        The path to the written ONNX file.
    """
    import torch

    logger.info("Loading EfficientNet-B0 from %s", model_path)
    base_model = load_efficientnet_deepfake(model_path)
    export_model = EfficientNetExportWrapper(base_model)

    dummy_input = torch.randn(1, 3, EFFICIENTNET_INPUT_SIZE, EFFICIENTNET_INPUT_SIZE)

    logger.info("Exporting EfficientNet-B0 to %s", output_path)
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["score", "embedding"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "score": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
    )
    logger.info("EfficientNet-B0 ONNX export complete: %s", output_path)
    _verify_onnx(output_path)
    return output_path


def export_lstm(
    model_path: str,
    output_path: str,
) -> str:
    """Export the temporal LSTM model to ONNX.

    Args:
        model_path: Path to the trained PyTorch .pt checkpoint.
        output_path: Path for the output .onnx file.

    Returns:
        The path to the written ONNX file.
    """
    import torch

    logger.info("Loading temporal LSTM from %s", model_path)
    model = load_temporal_lstm(model_path)

    dummy_input = torch.randn(1, TEMPORAL_WINDOW, EMBEDDING_DIM)

    logger.info("Exporting LSTM to %s", output_path)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["temporal_score"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "temporal_score": {0: "batch_size"},
        },
    )
    logger.info("LSTM ONNX export complete: %s", output_path)
    _verify_onnx(output_path)
    return output_path


def _verify_onnx(onnx_path: str) -> None:
    """Run onnx.checker on an exported model to verify structural validity."""
    import onnx

    logger.info("Verifying ONNX model: %s", onnx_path)
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    logger.info("ONNX model verification passed: %s", onnx_path)


# ---------------------------------------------------------------------------
# Validation: PyTorch vs ONNX output comparison
# ---------------------------------------------------------------------------

def validate_efficientnet(
    pytorch_model_path: str,
    onnx_model_path: str,
    num_samples: int = 10,
) -> bool:
    """Compare EfficientNet ONNX outputs against PyTorch outputs.

    Generates random input tensors and checks that both backends produce
    numerically close results (within ATOL_FP32).

    Returns:
        True if all samples pass the tolerance check.
    """
    import torch
    import onnxruntime as ort

    logger.info("Validating EfficientNet: PyTorch vs ONNX (FP32)")

    # Load PyTorch model via wrapper
    base_model = _build_efficientnet()
    state_dict = torch.load(pytorch_model_path, map_location="cpu", weights_only=True)
    base_model.load_state_dict(state_dict)
    base_model.eval()
    export_model = _make_efficientnet_export_module(base_model)

    # Load ONNX model
    session = ort.InferenceSession(
        onnx_model_path, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    all_passed = True
    max_score_diff = 0.0
    max_embed_diff = 0.0

    for i in range(num_samples):
        x = torch.randn(1, 3, EFFICIENTNET_INPUT_SIZE, EFFICIENTNET_INPUT_SIZE)

        with torch.no_grad():
            pt_score, pt_embed = export_model(x)
        pt_score = pt_score.numpy()
        pt_embed = pt_embed.numpy()

        onnx_outputs = session.run(None, {input_name: x.numpy()})
        onnx_score = onnx_outputs[0]
        onnx_embed = onnx_outputs[1]

        score_diff = float(np.max(np.abs(pt_score - onnx_score)))
        embed_diff = float(np.max(np.abs(pt_embed - onnx_embed)))
        max_score_diff = max(max_score_diff, score_diff)
        max_embed_diff = max(max_embed_diff, embed_diff)

        if score_diff > ATOL_FP32 or embed_diff > ATOL_FP32:
            logger.warning(
                "Sample %d: score diff=%.6e, embedding diff=%.6e (FAIL)",
                i, score_diff, embed_diff,
            )
            all_passed = False
        else:
            logger.debug(
                "Sample %d: score diff=%.6e, embedding diff=%.6e (OK)",
                i, score_diff, embed_diff,
            )

    logger.info(
        "EfficientNet validation: max_score_diff=%.6e, max_embed_diff=%.6e — %s",
        max_score_diff, max_embed_diff, "PASSED" if all_passed else "FAILED",
    )
    return all_passed


def validate_lstm(
    pytorch_model_path: str,
    onnx_model_path: str,
    num_samples: int = 10,
) -> bool:
    """Compare LSTM ONNX outputs against PyTorch outputs.

    Returns:
        True if all samples pass the tolerance check.
    """
    import torch
    import onnxruntime as ort

    logger.info("Validating LSTM: PyTorch vs ONNX (FP32)")

    model = _build_lstm()
    state_dict = torch.load(pytorch_model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    session = ort.InferenceSession(
        onnx_model_path, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    all_passed = True
    max_diff = 0.0

    for i in range(num_samples):
        x = torch.randn(1, TEMPORAL_WINDOW, EMBEDDING_DIM)

        with torch.no_grad():
            pt_out = model(x).numpy()

        onnx_out = session.run(None, {input_name: x.numpy()})[0]

        diff = float(np.max(np.abs(pt_out - onnx_out)))
        max_diff = max(max_diff, diff)

        if diff > ATOL_FP32:
            logger.warning("Sample %d: diff=%.6e (FAIL)", i, diff)
            all_passed = False
        else:
            logger.debug("Sample %d: diff=%.6e (OK)", i, diff)

    logger.info(
        "LSTM validation: max_diff=%.6e — %s",
        max_diff, "PASSED" if all_passed else "FAILED",
    )
    return all_passed


# ---------------------------------------------------------------------------
# INT8 Quantization
# ---------------------------------------------------------------------------

class _EfficientNetCalibrationReader:
    """CalibrationDataReader for EfficientNet INT8 quantization.

    Loads face crop images from a directory, applies ImageNet normalization,
    and yields them one at a time as NCHW float32 tensors.

    The calibration dataset consists of representative face crops randomly
    sampled from the training set (up to CALIBRATION_SAMPLE_COUNT images).
    """

    def __init__(self, calibration_dir: str, input_name: str) -> None:
        self.input_name = input_name
        self.image_paths: list[str] = []
        self._index = 0

        calibration_path = Path(calibration_dir)
        if not calibration_path.is_dir():
            raise FileNotFoundError(
                f"Calibration directory not found: {calibration_dir}"
            )

        # Collect image files (common formats)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        all_images = sorted(
            p
            for p in calibration_path.rglob("*")
            if p.suffix.lower() in extensions and p.is_file()
        )

        if not all_images:
            raise FileNotFoundError(
                f"No image files found in calibration directory: {calibration_dir}"
            )

        # Randomly sample up to CALIBRATION_SAMPLE_COUNT images
        rng = np.random.default_rng(seed=42)
        if len(all_images) > CALIBRATION_SAMPLE_COUNT:
            indices = rng.choice(
                len(all_images), size=CALIBRATION_SAMPLE_COUNT, replace=False
            )
            self.image_paths = [str(all_images[i]) for i in sorted(indices)]
        else:
            self.image_paths = [str(p) for p in all_images]

        logger.info(
            "Calibration dataset: %d images from %s",
            len(self.image_paths), calibration_dir,
        )

    def get_next(self) -> Optional[dict]:
        """Return the next calibration sample or None when exhausted."""
        if self._index >= len(self.image_paths):
            return None

        image_path = self.image_paths[self._index]
        self._index += 1

        try:
            tensor = self._load_and_preprocess(image_path)
            return {self.input_name: tensor}
        except Exception as exc:
            logger.warning(
                "Skipping calibration image %s: %s", image_path, exc
            )
            # Recurse to skip bad images
            return self.get_next()

    def _load_and_preprocess(self, image_path: str) -> np.ndarray:
        """Load an image and preprocess it to EfficientNet input format.

        Returns:
            NCHW float32 array of shape (1, 3, 224, 224).
        """
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Resize to 224x224
        img = cv2.resize(
            img, (EFFICIENTNET_INPUT_SIZE, EFFICIENTNET_INPUT_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1] then ImageNet mean/std
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        # HWC -> NCHW
        tensor = np.transpose(img, (2, 0, 1))[np.newaxis].astype(np.float32)
        return tensor


class _LSTMCalibrationReader:
    """CalibrationDataReader for LSTM INT8 quantization.

    Generates synthetic calibration data by running face crops through
    the FP32 EfficientNet ONNX model to produce realistic embedding
    sequences. Falls back to random normal data if no calibration
    images are available.
    """

    def __init__(
        self,
        efficientnet_onnx_path: str,
        calibration_dir: str,
        input_name: str,
    ) -> None:
        self.input_name = input_name
        self._sequences: list[np.ndarray] = []
        self._index = 0

        self._generate_sequences(efficientnet_onnx_path, calibration_dir)

    def _generate_sequences(
        self, efficientnet_onnx_path: str, calibration_dir: str
    ) -> None:
        """Generate embedding sequences from calibration images."""
        import onnxruntime as ort

        session = ort.InferenceSession(
            efficientnet_onnx_path, providers=["CPUExecutionProvider"]
        )
        enet_input_name = session.get_inputs()[0].name

        # Reuse the EfficientNet calibration reader to get preprocessed images
        enet_reader = _EfficientNetCalibrationReader(calibration_dir, enet_input_name)

        # Collect all embeddings
        embeddings: list[np.ndarray] = []
        while True:
            sample = enet_reader.get_next()
            if sample is None:
                break
            outputs = session.run(None, sample)
            # outputs[1] is the embedding (1, 256)
            embeddings.append(outputs[1][0])

        if len(embeddings) < TEMPORAL_WINDOW:
            logger.warning(
                "Not enough calibration embeddings (%d) for LSTM sequences "
                "(need at least %d). Padding with random data.",
                len(embeddings), TEMPORAL_WINDOW,
            )
            rng = np.random.default_rng(seed=42)
            while len(embeddings) < TEMPORAL_WINDOW:
                embeddings.append(
                    rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
                )

        # Build overlapping sequences of length TEMPORAL_WINDOW
        stride = max(1, TEMPORAL_WINDOW // 2)
        for start in range(0, len(embeddings) - TEMPORAL_WINDOW + 1, stride):
            seq = np.stack(
                embeddings[start : start + TEMPORAL_WINDOW], axis=0
            )[np.newaxis].astype(np.float32)
            self._sequences.append(seq)

        # Limit to a reasonable number of calibration sequences
        max_seqs = CALIBRATION_SAMPLE_COUNT
        if len(self._sequences) > max_seqs:
            rng = np.random.default_rng(seed=42)
            indices = rng.choice(
                len(self._sequences), size=max_seqs, replace=False
            )
            self._sequences = [self._sequences[i] for i in sorted(indices)]

        logger.info(
            "LSTM calibration: %d sequences of length %d",
            len(self._sequences), TEMPORAL_WINDOW,
        )

    def get_next(self) -> Optional[dict]:
        """Return the next calibration sample or None when exhausted."""
        if self._index >= len(self._sequences):
            return None
        seq = self._sequences[self._index]
        self._index += 1
        return {self.input_name: seq}


def quantize_efficientnet(
    fp32_onnx_path: str,
    int8_output_path: str,
    calibration_dir: str,
) -> str:
    """Apply INT8 post-training static quantization to the EfficientNet ONNX model.

    Uses ONNX Runtime's quantization toolkit with a calibration dataset
    of representative face crops.

    Args:
        fp32_onnx_path: Path to the FP32 ONNX model.
        int8_output_path: Path for the quantized INT8 ONNX model.
        calibration_dir: Directory containing calibration face crop images.

    Returns:
        Path to the quantized model.
    """
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantType,
        quantize_static,
    )

    logger.info("Quantizing EfficientNet to INT8: %s", int8_output_path)

    # Determine the input name from the FP32 model
    import onnxruntime as ort

    session = ort.InferenceSession(
        fp32_onnx_path, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    del session

    # Build calibration reader
    calibration_reader = _EfficientNetCalibrationReader(
        calibration_dir, input_name
    )

    quantize_static(
        model_input=fp32_onnx_path,
        model_output=int8_output_path,
        calibration_data_reader=calibration_reader,
        quant_format=_get_quant_format(),
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=True,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )

    logger.info("EfficientNet INT8 quantization complete: %s", int8_output_path)
    return int8_output_path


def quantize_lstm(
    fp32_onnx_path: str,
    int8_output_path: str,
    efficientnet_onnx_path: str,
    calibration_dir: str,
) -> str:
    """Apply INT8 post-training static quantization to the LSTM ONNX model.

    Generates calibration sequences by running face crops through the
    FP32 EfficientNet model to produce realistic embedding sequences.

    Args:
        fp32_onnx_path: Path to the FP32 LSTM ONNX model.
        int8_output_path: Path for the quantized INT8 ONNX model.
        efficientnet_onnx_path: Path to the FP32 EfficientNet ONNX model
            (used to generate embedding sequences for calibration).
        calibration_dir: Directory containing calibration face crop images.

    Returns:
        Path to the quantized model.
    """
    from onnxruntime.quantization import (
        QuantType,
        quantize_static,
    )

    logger.info("Quantizing LSTM to INT8: %s", int8_output_path)

    import onnxruntime as ort

    session = ort.InferenceSession(
        fp32_onnx_path, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    del session

    calibration_reader = _LSTMCalibrationReader(
        efficientnet_onnx_path, calibration_dir, input_name
    )

    quantize_static(
        model_input=fp32_onnx_path,
        model_output=int8_output_path,
        calibration_data_reader=calibration_reader,
        quant_format=_get_quant_format(),
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=False,  # LSTM layers do not benefit from per-channel
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )

    logger.info("LSTM INT8 quantization complete: %s", int8_output_path)
    return int8_output_path


def _get_quant_format():
    """Return the preferred quantization format from onnxruntime.quantization."""
    from onnxruntime.quantization import QuantFormat

    return QuantFormat.QDQ


# ---------------------------------------------------------------------------
# INT8 vs FP32 accuracy comparison
# ---------------------------------------------------------------------------

def compare_int8_vs_fp32_efficientnet(
    fp32_path: str,
    int8_path: str,
    num_samples: int = 50,
) -> dict:
    """Compare INT8 quantized EfficientNet against FP32 on a test batch.

    Generates random test inputs and reports max absolute difference,
    mean absolute difference, and whether all outputs are within
    acceptable tolerance.

    Returns:
        Dict with comparison statistics.
    """
    import onnxruntime as ort

    logger.info("Comparing EfficientNet INT8 vs FP32 (%d samples)", num_samples)

    fp32_session = ort.InferenceSession(
        fp32_path, providers=["CPUExecutionProvider"]
    )
    int8_session = ort.InferenceSession(
        int8_path, providers=["CPUExecutionProvider"]
    )

    fp32_input = fp32_session.get_inputs()[0].name
    int8_input = int8_session.get_inputs()[0].name

    score_diffs = []
    embed_diffs = []

    for _ in range(num_samples):
        x = np.random.randn(
            1, 3, EFFICIENTNET_INPUT_SIZE, EFFICIENTNET_INPUT_SIZE
        ).astype(np.float32)

        fp32_out = fp32_session.run(None, {fp32_input: x})
        int8_out = int8_session.run(None, {int8_input: x})

        score_diffs.append(float(np.max(np.abs(fp32_out[0] - int8_out[0]))))
        embed_diffs.append(float(np.max(np.abs(fp32_out[1] - int8_out[1]))))

    stats = {
        "score_max_diff": float(np.max(score_diffs)),
        "score_mean_diff": float(np.mean(score_diffs)),
        "embed_max_diff": float(np.max(embed_diffs)),
        "embed_mean_diff": float(np.mean(embed_diffs)),
        "within_tolerance": float(np.max(score_diffs)) < ATOL_INT8
        and float(np.max(embed_diffs)) < ATOL_INT8,
    }

    logger.info(
        "EfficientNet INT8 vs FP32: score_max=%.4f score_mean=%.4f "
        "embed_max=%.4f embed_mean=%.4f — %s",
        stats["score_max_diff"],
        stats["score_mean_diff"],
        stats["embed_max_diff"],
        stats["embed_mean_diff"],
        "PASSED" if stats["within_tolerance"] else "FAILED",
    )
    return stats


def compare_int8_vs_fp32_lstm(
    fp32_path: str,
    int8_path: str,
    num_samples: int = 50,
) -> dict:
    """Compare INT8 quantized LSTM against FP32 on a test batch.

    Returns:
        Dict with comparison statistics.
    """
    import onnxruntime as ort

    logger.info("Comparing LSTM INT8 vs FP32 (%d samples)", num_samples)

    fp32_session = ort.InferenceSession(
        fp32_path, providers=["CPUExecutionProvider"]
    )
    int8_session = ort.InferenceSession(
        int8_path, providers=["CPUExecutionProvider"]
    )

    fp32_input = fp32_session.get_inputs()[0].name
    int8_input = int8_session.get_inputs()[0].name

    diffs = []
    for _ in range(num_samples):
        x = np.random.randn(
            1, TEMPORAL_WINDOW, EMBEDDING_DIM
        ).astype(np.float32)

        fp32_out = fp32_session.run(None, {fp32_input: x})[0]
        int8_out = int8_session.run(None, {int8_input: x})[0]

        diffs.append(float(np.max(np.abs(fp32_out - int8_out))))

    stats = {
        "max_diff": float(np.max(diffs)),
        "mean_diff": float(np.mean(diffs)),
        "within_tolerance": float(np.max(diffs)) < ATOL_INT8,
    }

    logger.info(
        "LSTM INT8 vs FP32: max=%.4f mean=%.4f — %s",
        stats["max_diff"],
        stats["mean_diff"],
        "PASSED" if stats["within_tolerance"] else "FAILED",
    )
    return stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Export trained PyTorch deepfake detection models to ONNX "
            "with optional INT8 quantization."
        ),
    )
    parser.add_argument(
        "--efficientnet_model",
        type=str,
        required=True,
        help="Path to the trained EfficientNet-B0 PyTorch checkpoint (.pt).",
    )
    parser.add_argument(
        "--lstm_model",
        type=str,
        required=True,
        help="Path to the trained temporal LSTM PyTorch checkpoint (.pt).",
    )
    parser.add_argument(
        "--calibration_dir",
        type=str,
        required=True,
        help=(
            "Directory containing representative face crop images for "
            "INT8 calibration (500 images recommended)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write the exported ONNX models.",
    )
    parser.add_argument(
        "--skip_quantization",
        action="store_true",
        default=False,
        help="Skip INT8 quantization (export FP32 only).",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        default=False,
        help="Skip output validation against PyTorch.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the full export pipeline.

    Steps:
        1. Export EfficientNet-B0 to ONNX (FP32)
        2. Export LSTM to ONNX (FP32)
        3. Validate ONNX outputs against PyTorch
        4. INT8 quantization with calibration data
        5. Compare INT8 vs FP32 accuracy

    Returns:
        0 on success, 1 on failure.
    """
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Validate inputs
    if not os.path.isfile(args.efficientnet_model):
        logger.error(
            "EfficientNet model not found: %s", args.efficientnet_model
        )
        return 1

    if not os.path.isfile(args.lstm_model):
        logger.error("LSTM model not found: %s", args.lstm_model)
        return 1

    if not args.skip_quantization and not os.path.isdir(args.calibration_dir):
        logger.error(
            "Calibration directory not found: %s", args.calibration_dir
        )
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output file paths
    enet_fp32_path = str(output_dir / "efficientnet_b0_deepfake.onnx")
    lstm_fp32_path = str(output_dir / "temporal_lstm.onnx")
    enet_int8_path = str(output_dir / "efficientnet_b0_deepfake_int8.onnx")
    lstm_int8_path = str(output_dir / "temporal_lstm_int8.onnx")

    success = True

    # --- Step 1: Export EfficientNet to ONNX ---
    try:
        export_efficientnet(args.efficientnet_model, enet_fp32_path)
    except Exception:
        logger.exception("Failed to export EfficientNet to ONNX")
        return 1

    # --- Step 2: Export LSTM to ONNX ---
    try:
        export_lstm(args.lstm_model, lstm_fp32_path)
    except Exception:
        logger.exception("Failed to export LSTM to ONNX")
        return 1

    # --- Step 3: Validate ONNX vs PyTorch ---
    if not args.skip_validation:
        try:
            enet_valid = validate_efficientnet(
                args.efficientnet_model, enet_fp32_path
            )
            if not enet_valid:
                logger.error("EfficientNet ONNX validation FAILED")
                success = False
        except Exception:
            logger.exception("EfficientNet validation error")
            success = False

        try:
            lstm_valid = validate_lstm(args.lstm_model, lstm_fp32_path)
            if not lstm_valid:
                logger.error("LSTM ONNX validation FAILED")
                success = False
        except Exception:
            logger.exception("LSTM validation error")
            success = False

    # --- Step 4: INT8 Quantization ---
    if not args.skip_quantization:
        try:
            quantize_efficientnet(
                enet_fp32_path, enet_int8_path, args.calibration_dir
            )
        except Exception:
            logger.exception("EfficientNet INT8 quantization failed")
            success = False

        try:
            quantize_lstm(
                lstm_fp32_path, lstm_int8_path,
                enet_fp32_path, args.calibration_dir,
            )
        except Exception:
            logger.exception("LSTM INT8 quantization failed")
            success = False

        # --- Step 5: Compare INT8 vs FP32 ---
        if not args.skip_validation:
            if os.path.isfile(enet_int8_path):
                try:
                    enet_cmp = compare_int8_vs_fp32_efficientnet(
                        enet_fp32_path, enet_int8_path
                    )
                    if not enet_cmp["within_tolerance"]:
                        logger.warning(
                            "EfficientNet INT8 accuracy degradation "
                            "exceeds tolerance (%.4f > %.4f)",
                            max(
                                enet_cmp["score_max_diff"],
                                enet_cmp["embed_max_diff"],
                            ),
                            ATOL_INT8,
                        )
                except Exception:
                    logger.exception("EfficientNet INT8 vs FP32 comparison failed")

            if os.path.isfile(lstm_int8_path):
                try:
                    lstm_cmp = compare_int8_vs_fp32_lstm(
                        lstm_fp32_path, lstm_int8_path
                    )
                    if not lstm_cmp["within_tolerance"]:
                        logger.warning(
                            "LSTM INT8 accuracy degradation exceeds "
                            "tolerance (%.4f > %.4f)",
                            lstm_cmp["max_diff"],
                            ATOL_INT8,
                        )
                except Exception:
                    logger.exception("LSTM INT8 vs FP32 comparison failed")

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("Export summary:")
    logger.info("  EfficientNet FP32: %s", enet_fp32_path)
    logger.info("  LSTM FP32:         %s", lstm_fp32_path)
    if not args.skip_quantization:
        logger.info("  EfficientNet INT8: %s", enet_int8_path)
        logger.info("  LSTM INT8:         %s", lstm_int8_path)
    logger.info("=" * 60)

    if success:
        logger.info("All exports completed successfully.")
        return 0
    else:
        logger.error("Some steps failed. Check the log output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
