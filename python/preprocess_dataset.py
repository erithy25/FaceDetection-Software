#!/usr/bin/env python3
"""
FaceForensics++ Dataset Preprocessor — Face extraction and train/val/test splitting.

Processes the FaceForensics++ dataset by extracting faces from video frames
using MediaPipe Face Mesh, cropping them to 224x224 with 20% padding, and
organizing output into train/val/test splits at the VIDEO level to prevent
data leakage. Supports all three compression levels (raw, c23, c40) and all
four manipulation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures).

Supports both the legacy MediaPipe ``solutions`` API and the newer
``mediapipe.tasks`` API (v0.10.21+). The appropriate backend is selected
automatically at import time. When using the tasks API the required
``face_landmarker.task`` model file is downloaded on first run.

Output structure::

    output_dir/
        train/
            real/
                {video_id}_frame_{idx:04d}.jpg
            fake/
                {video_id}_{method}_frame_{idx:04d}.jpg
        val/
            real/ ...
            fake/ ...
        test/
            real/ ...
            fake/ ...
        metadata.json

Usage::

    python preprocess_dataset.py \\
        --dataset_root /path/to/FaceForensics++ \\
        --output_dir /path/to/output \\
        --compression c23 \\
        --max_frames_per_video 50
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import urllib.request
from functools import partial
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    print(
        "ERROR: tqdm is required. Install it with: pip install tqdm",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# MediaPipe backend detection
# ---------------------------------------------------------------------------

_USE_TASKS_API: bool = False

try:
    # Prefer the newer tasks-based API (mediapipe >= 0.10.21)
    import mediapipe as _mp
    from mediapipe.tasks.python import vision as _mp_vision  # noqa: F401
    from mediapipe.tasks.python.core.base_options import BaseOptions as _BaseOptions  # noqa: F401

    _USE_TASKS_API = True
except (ImportError, AttributeError):
    pass

if not _USE_TASKS_API:
    try:
        import mediapipe as _mp  # noqa: F811
        # Verify legacy solutions API is accessible
        _ = _mp.solutions.face_mesh
    except (ImportError, AttributeError):
        print(
            "ERROR: mediapipe is required. Install it with: pip install mediapipe",
            file=sys.stderr,
        )
        sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EfficientNet-B0 input size (matches detector.py)
CROP_SIZE = 224
# Padding around detected face bounding box (matches detector.py)
FACE_PADDING = 0.20

# FaceForensics++ manipulation methods
MANIPULATION_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

# Valid compression levels
VALID_COMPRESSIONS = ["raw", "c23", "c40"]

# Train / Val / Test split ratios (video-level)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# JPEG save quality
JPEG_QUALITY = 95

# MediaPipe tasks API — model download
_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_FACE_LANDMARKER_CACHE_DIR = Path.home() / ".cache" / "mediapipe"
_FACE_LANDMARKER_PATH = _FACE_LANDMARKER_CACHE_DIR / "face_landmarker.task"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess_dataset")


# ---------------------------------------------------------------------------
# MediaPipe model management
# ---------------------------------------------------------------------------


def _ensure_face_landmarker_model() -> str:
    """Download the face landmarker model if it is not already cached.

    Returns:
        Absolute path to the cached model file.
    """
    model_path = _FACE_LANDMARKER_PATH
    if model_path.exists():
        return str(model_path)

    _FACE_LANDMARKER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading face_landmarker.task to %s (one-time download)...",
        model_path,
    )
    try:
        urllib.request.urlretrieve(_FACE_LANDMARKER_URL, str(model_path))
        logger.info("Download complete (%d bytes)", model_path.stat().st_size)
    except Exception as exc:
        # Clean up partial download
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(
            f"Failed to download face_landmarker.task: {exc}\n"
            f"You can manually download from:\n  {_FACE_LANDMARKER_URL}\n"
            f"and place it at:\n  {model_path}"
        ) from exc

    return str(model_path)


# ---------------------------------------------------------------------------
# Face detector abstraction — hides API differences
# ---------------------------------------------------------------------------


class _FaceDetectorLegacy:
    """Face detector using the legacy ``mediapipe.solutions`` API."""

    def __init__(self) -> None:
        self._mesh = _mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def detect_landmarks(
        self, rgb_image: np.ndarray
    ) -> Optional[list[tuple[float, float]]]:
        """Return normalized (x, y) landmark pairs or None if no face."""
        results = self._mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return None
        face = results.multi_face_landmarks[0]
        return [(lm.x, lm.y) for lm in face.landmark]

    def close(self) -> None:
        self._mesh.close()


class _FaceDetectorTasks:
    """Face detector using the newer ``mediapipe.tasks`` API."""

    def __init__(self, model_path: str) -> None:
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def detect_landmarks(
        self, rgb_image: np.ndarray
    ) -> Optional[list[tuple[float, float]]]:
        """Return normalized (x, y) landmark pairs or None if no face."""
        import mediapipe as mp_mod

        mp_image = mp_mod.Image(
            image_format=mp_mod.ImageFormat.SRGB, data=rgb_image
        )
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        face = result.face_landmarks[0]
        return [(lm.x, lm.y) for lm in face]

    def close(self) -> None:
        self._landmarker.close()


def _create_face_detector(model_path: Optional[str] = None) -> object:
    """Factory that returns the appropriate face detector backend.

    Args:
        model_path: Path to face_landmarker.task (only used by tasks API).

    Returns:
        An object with ``detect_landmarks(rgb)`` and ``close()`` methods.
    """
    if _USE_TASKS_API:
        if model_path is None:
            model_path = _ensure_face_landmarker_model()
        return _FaceDetectorTasks(model_path)
    return _FaceDetectorLegacy()


# ---------------------------------------------------------------------------
# Video discovery
# ---------------------------------------------------------------------------


def discover_videos(
    dataset_root: Path,
    compression: str,
) -> tuple[list[dict], list[dict]]:
    """Discover all real and fake videos in the FaceForensics++ directory tree.

    Args:
        dataset_root: Root directory of the FaceForensics++ dataset.
        compression: Compression level — one of "raw", "c23", "c40".

    Returns:
        Tuple of (real_videos, fake_videos) where each entry is a dict
        with keys: path, video_id, label, method.
    """
    real_videos: list[dict] = []
    fake_videos: list[dict] = []

    # --- Real videos ---
    real_dir = dataset_root / "original_sequences" / "youtube" / compression / "videos"
    if real_dir.is_dir():
        for video_path in sorted(real_dir.glob("*.mp4")):
            real_videos.append({
                "path": str(video_path),
                "video_id": video_path.stem,
                "label": 1,
                "method": "original",
            })
        logger.info("Found %d real videos in %s", len(real_videos), real_dir)
    else:
        logger.warning("Real video directory not found: %s", real_dir)

    # --- Fake videos ---
    for method in MANIPULATION_METHODS:
        method_dir = (
            dataset_root
            / "manipulated_sequences"
            / method
            / compression
            / "videos"
        )
        if not method_dir.is_dir():
            logger.warning("Method directory not found: %s", method_dir)
            continue

        method_videos = sorted(method_dir.glob("*.mp4"))
        for video_path in method_videos:
            fake_videos.append({
                "path": str(video_path),
                "video_id": video_path.stem,
                "label": 0,
                "method": method,
            })
        logger.info(
            "Found %d fake videos for method '%s' in %s",
            len(method_videos),
            method,
            method_dir,
        )

    return real_videos, fake_videos


# ---------------------------------------------------------------------------
# Video-level splitting
# ---------------------------------------------------------------------------


def split_videos(
    videos: list[dict],
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split a list of video entries into train/val/test at the VIDEO level.

    Uses a fixed random seed for reproducibility. The split is performed
    on videos (not frames) to prevent data leakage between splits.

    Args:
        videos: List of video dicts (path, video_id, label, method).
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping split name to list of video dicts.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(videos))

    n_train = int(len(videos) * TRAIN_RATIO)
    n_val = int(len(videos) * VAL_RATIO)
    # Remaining goes to test (handles rounding)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return {
        "train": [videos[i] for i in train_idx],
        "val": [videos[i] for i in val_idx],
        "test": [videos[i] for i in test_idx],
    }


# ---------------------------------------------------------------------------
# Face extraction from a single video
# ---------------------------------------------------------------------------


def extract_faces_from_video(
    video_info: dict,
    output_base: Path,
    split: str,
    max_frames: int,
    model_path: Optional[str] = None,
) -> dict:
    """Extract face crops from a single video and save as JPEGs.

    Frames are sampled at even intervals across the video duration.
    For each sampled frame, MediaPipe detects the face, applies 20%
    bounding-box padding, crops and resizes to 224x224, and saves as
    a JPEG file.

    Args:
        video_info: Dict with keys: path, video_id, label, method.
        output_base: Root output directory.
        split: One of "train", "val", "test".
        max_frames: Maximum number of frames to extract per video.
        model_path: Path to MediaPipe face landmarker model (tasks API only).

    Returns:
        Dict with processing results: video_id, split, label, method,
        num_frames_extracted, num_frames_failed, error (if any).
    """
    video_path = video_info["path"]
    video_id = video_info["video_id"]
    label = video_info["label"]
    method = video_info["method"]

    class_name = "real" if label == 1 else "fake"
    out_dir = output_base / split / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build filename prefix: for fake videos, include method for traceability
    if label == 1:
        prefix = video_id
    else:
        prefix = f"{video_id}_{method}"

    result = {
        "video_id": video_id,
        "video_path": video_path,
        "split": split,
        "label": label,
        "method": method,
        "num_frames_extracted": 0,
        "num_frames_failed": 0,
        "error": None,
    }

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result["error"] = f"Cannot open video: {video_path}"
        return result

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        result["error"] = f"Video has no frames: {video_path}"
        return result

    # Compute evenly-spaced frame indices
    n_to_sample = min(max_frames, total_frames)
    if n_to_sample <= 0:
        cap.release()
        result["error"] = f"No frames to sample from: {video_path}"
        return result

    frame_indices = np.linspace(0, total_frames - 1, n_to_sample, dtype=int)
    # Remove duplicates that can occur with very short videos
    frame_indices = sorted(set(frame_indices))

    # Initialize face detector for this worker
    detector = _create_face_detector(model_path)

    extracted = 0
    failed = 0

    for target_idx in frame_indices:
        # Seek to target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            failed += 1
            continue

        h, w = frame.shape[:2]

        # Detect face — returns normalized (x, y) pairs
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = detector.detect_landmarks(rgb)

        if landmarks is None:
            failed += 1
            continue

        # Compute bounding box from normalized landmarks
        xs = [lx * w for lx, _ in landmarks]
        ys = [ly * h for _, ly in landmarks]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Apply 20% padding
        pad_w = (x_max - x_min) * FACE_PADDING
        pad_h = (y_max - y_min) * FACE_PADDING
        x_min = max(0, int(x_min - pad_w))
        y_min = max(0, int(y_min - pad_h))
        x_max = min(w, int(x_max + pad_w))
        y_max = min(h, int(y_max + pad_h))

        # Validate crop dimensions
        if (x_max - x_min) < 10 or (y_max - y_min) < 10:
            failed += 1
            continue

        face_crop = frame[y_min:y_max, x_min:x_max]
        if face_crop.size == 0:
            failed += 1
            continue

        # Resize to 224x224
        face_resized = cv2.resize(
            face_crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA
        )

        # Save as JPEG
        filename = f"{prefix}_frame_{extracted:04d}.jpg"
        save_path = out_dir / filename
        cv2.imwrite(
            str(save_path),
            face_resized,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        extracted += 1

    cap.release()
    detector.close()

    result["num_frames_extracted"] = extracted
    result["num_frames_failed"] = failed
    return result


# ---------------------------------------------------------------------------
# Worker wrapper for multiprocessing
# ---------------------------------------------------------------------------


def _process_video_worker(
    task: tuple[dict, str],
    output_base: Path,
    max_frames: int,
    model_path: Optional[str] = None,
) -> dict:
    """Multiprocessing-compatible wrapper around extract_faces_from_video.

    Args:
        task: Tuple of (video_info, split_name).
        output_base: Root output directory.
        max_frames: Maximum frames per video.
        model_path: Path to MediaPipe face landmarker model (tasks API only).

    Returns:
        Processing result dict.
    """
    video_info, split = task
    try:
        return extract_faces_from_video(
            video_info, output_base, split, max_frames, model_path
        )
    except Exception as exc:
        return {
            "video_id": video_info.get("video_id", "unknown"),
            "video_path": video_info.get("path", "unknown"),
            "split": split,
            "label": video_info.get("label", -1),
            "method": video_info.get("method", "unknown"),
            "num_frames_extracted": 0,
            "num_frames_failed": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------------


def build_metadata(
    real_splits: dict[str, list[dict]],
    fake_splits: dict[str, list[dict]],
    processing_results: list[dict],
    args: argparse.Namespace,
) -> dict:
    """Build the metadata.json content with video-to-split mapping and counts.

    Args:
        real_splits: Real videos split into train/val/test.
        fake_splits: Fake videos split into train/val/test.
        processing_results: List of per-video processing result dicts.
        args: Parsed command-line arguments.

    Returns:
        Metadata dict ready for JSON serialization.
    """
    # Build video-to-split mapping
    video_mapping: dict[str, dict] = {}
    for result in processing_results:
        vid = result["video_id"]
        key = vid if result["label"] == 1 else f"{vid}_{result['method']}"
        video_mapping[key] = {
            "video_id": vid,
            "split": result["split"],
            "label": result["label"],
            "method": result["method"],
            "num_frames_extracted": result["num_frames_extracted"],
            "num_frames_failed": result["num_frames_failed"],
            "source_path": result["video_path"],
            "error": result["error"],
        }

    # Aggregate counts
    counts: dict[str, dict[str, int]] = {}
    for split_name in ["train", "val", "test"]:
        split_results = [r for r in processing_results if r["split"] == split_name]
        real_results = [r for r in split_results if r["label"] == 1]
        fake_results = [r for r in split_results if r["label"] == 0]
        counts[split_name] = {
            "total_videos": len(split_results),
            "real_videos": len(real_results),
            "fake_videos": len(fake_results),
            "total_frames_extracted": sum(
                r["num_frames_extracted"] for r in split_results
            ),
            "real_frames_extracted": sum(
                r["num_frames_extracted"] for r in real_results
            ),
            "fake_frames_extracted": sum(
                r["num_frames_extracted"] for r in fake_results
            ),
        }

    # Per-method counts
    method_counts: dict[str, dict[str, int]] = {}
    for method in ["original"] + MANIPULATION_METHODS:
        method_results = [r for r in processing_results if r["method"] == method]
        if method_results:
            method_counts[method] = {
                "total_videos": len(method_results),
                "total_frames_extracted": sum(
                    r["num_frames_extracted"] for r in method_results
                ),
                "videos_with_errors": sum(
                    1 for r in method_results if r["error"] is not None
                ),
            }

    # Error summary
    errors = [r for r in processing_results if r["error"] is not None]

    metadata = {
        "dataset": "FaceForensics++",
        "compression": args.compression,
        "max_frames_per_video": args.max_frames_per_video,
        "crop_size": CROP_SIZE,
        "face_padding": FACE_PADDING,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "split_seed": 42,
        "split_level": "video",
        "counts": counts,
        "method_counts": method_counts,
        "total_videos_processed": len(processing_results),
        "total_frames_extracted": sum(
            r["num_frames_extracted"] for r in processing_results
        ),
        "total_errors": len(errors),
        "video_mapping": video_mapping,
    }

    return metadata


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_preprocessing(args: argparse.Namespace) -> None:
    """Execute the full preprocessing pipeline.

    1. Discover videos in the FaceForensics++ directory structure.
    2. Split real and fake video lists independently at the video level.
    3. Create output directory structure.
    4. Ensure the MediaPipe model is available (tasks API only).
    5. Process all videos with multiprocessing (face extraction + cropping).
    6. Save metadata.json.

    Args:
        args: Parsed CLI arguments.
    """
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    compression = args.compression
    max_frames = args.max_frames_per_video
    num_workers = args.num_workers

    logger.info("=" * 70)
    logger.info("FaceForensics++ Dataset Preprocessor")
    logger.info("=" * 70)
    logger.info("Dataset root   : %s", dataset_root)
    logger.info("Output dir     : %s", output_dir)
    logger.info("Compression    : %s", compression)
    logger.info("Max frames/vid : %d", max_frames)
    logger.info("Workers        : %d", num_workers)
    logger.info("MediaPipe API  : %s", "tasks" if _USE_TASKS_API else "legacy")
    logger.info("=" * 70)

    # Validate dataset root
    if not dataset_root.is_dir():
        logger.error("Dataset root does not exist: %s", dataset_root)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1: Discover videos
    # ------------------------------------------------------------------
    logger.info("Discovering videos...")
    real_videos, fake_videos = discover_videos(dataset_root, compression)

    total_videos = len(real_videos) + len(fake_videos)
    if total_videos == 0:
        logger.error(
            "No videos found. Verify dataset_root and compression level. "
            "Expected structure: "
            "<root>/original_sequences/youtube/%s/videos/*.mp4",
            compression,
        )
        sys.exit(1)

    logger.info(
        "Total: %d videos (%d real, %d fake)",
        total_videos,
        len(real_videos),
        len(fake_videos),
    )

    # ------------------------------------------------------------------
    # Step 2: Split at the video level (independently for real and fake)
    # ------------------------------------------------------------------
    logger.info("Splitting videos (70/15/15 at VIDEO level)...")
    real_splits = split_videos(real_videos, seed=42)
    fake_splits = split_videos(fake_videos, seed=42)

    for split_name in ["train", "val", "test"]:
        n_real = len(real_splits[split_name])
        n_fake = len(fake_splits[split_name])
        logger.info(
            "  %-5s : %d real, %d fake (%d total)",
            split_name,
            n_real,
            n_fake,
            n_real + n_fake,
        )

    # ------------------------------------------------------------------
    # Step 3: Create output directory structure
    # ------------------------------------------------------------------
    for split_name in ["train", "val", "test"]:
        for class_name in ["real", "fake"]:
            (output_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 4: Ensure MediaPipe model is cached (download once in main
    #         process so workers don't race to download simultaneously)
    # ------------------------------------------------------------------
    model_path: Optional[str] = None
    if _USE_TASKS_API:
        model_path = _ensure_face_landmarker_model()
        logger.info("MediaPipe model : %s", model_path)

    # ------------------------------------------------------------------
    # Step 5: Build task list and process with multiprocessing
    # ------------------------------------------------------------------
    tasks: list[tuple[dict, str]] = []
    for split_name in ["train", "val", "test"]:
        for video_info in real_splits[split_name]:
            tasks.append((video_info, split_name))
        for video_info in fake_splits[split_name]:
            tasks.append((video_info, split_name))

    logger.info("Processing %d videos across %d workers...", len(tasks), num_workers)
    start_time = time.time()

    worker_fn = partial(
        _process_video_worker,
        output_base=output_dir,
        max_frames=max_frames,
        model_path=model_path,
    )

    results: list[dict] = []

    if num_workers <= 1:
        # Single-process mode (useful for debugging)
        for task in tqdm(tasks, desc="Processing videos", unit="video"):
            results.append(worker_fn(task))
    else:
        # Multiprocessing with progress bar
        with mp.Pool(processes=num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(worker_fn, tasks),
                total=len(tasks),
                desc="Processing videos",
                unit="video",
            ):
                results.append(result)

    elapsed = time.time() - start_time

    # ------------------------------------------------------------------
    # Step 6: Report results
    # ------------------------------------------------------------------
    total_extracted = sum(r["num_frames_extracted"] for r in results)
    total_failed = sum(r["num_frames_failed"] for r in results)
    errors = [r for r in results if r["error"] is not None]

    logger.info("=" * 70)
    logger.info("Processing complete in %.1f seconds", elapsed)
    logger.info("Total frames extracted : %d", total_extracted)
    logger.info("Total frames failed    : %d", total_failed)
    logger.info("Videos with errors     : %d / %d", len(errors), len(results))

    if errors:
        logger.warning("Errors encountered:")
        for err in errors[:20]:  # Show first 20 errors
            logger.warning(
                "  [%s] %s: %s", err["video_id"], err["method"], err["error"]
            )
        if len(errors) > 20:
            logger.warning("  ... and %d more errors", len(errors) - 20)

    # ------------------------------------------------------------------
    # Step 7: Save metadata.json
    # ------------------------------------------------------------------
    metadata = build_metadata(real_splits, fake_splits, results, args)
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Metadata saved to %s", metadata_path)

    # Final summary per split
    logger.info("-" * 70)
    for split_name in ["train", "val", "test"]:
        split_results = [r for r in results if r["split"] == split_name]
        split_frames = sum(r["num_frames_extracted"] for r in split_results)
        real_frames = sum(
            r["num_frames_extracted"] for r in split_results if r["label"] == 1
        )
        fake_frames = sum(
            r["num_frames_extracted"] for r in split_results if r["label"] == 0
        )
        logger.info(
            "  %-5s : %d frames (%d real, %d fake) from %d videos",
            split_name,
            split_frames,
            real_frames,
            fake_frames,
            len(split_results),
        )
    logger.info("=" * 70)
    logger.info("Done. Output directory: %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed Namespace object.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess the FaceForensics++ dataset: extract face crops from "
            "videos using MediaPipe Face Mesh, organize into train/val/test "
            "splits at the video level, and generate metadata."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help=(
            "Root directory of the FaceForensics++ dataset. Expected to contain "
            "original_sequences/ and manipulated_sequences/ subdirectories."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for preprocessed face crops and metadata.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="c23",
        choices=VALID_COMPRESSIONS,
        help="Video compression level.",
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=50,
        help="Maximum number of frames to extract per video (evenly spaced).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of parallel worker processes.",
    )

    args = parser.parse_args(argv)

    # Validate
    if args.max_frames_per_video < 1:
        parser.error("--max_frames_per_video must be at least 1")
    if args.num_workers < 1:
        parser.error("--num_workers must be at least 1")

    return args


def main() -> None:
    """Entry point."""
    args = parse_args()
    run_preprocessing(args)


if __name__ == "__main__":
    main()
