#!/usr/bin/env python3
"""
Generate synthetic demo data for the SilentWitness ML pipeline.

Creates a small dataset of synthetic 224x224 face-crop images organized
into the exact directory structure and metadata format expected by
train_efficientnet.py and train_lstm.py.  This allows the full pipeline
(train -> export) to run end-to-end **without** downloading the real
FaceForensics++ dataset.

Output structure::

    output_dir/
        train/
            real/   (80 images from 4 synthetic videos)
            fake/   (80 images from 4 synthetic videos)
        val/
            real/   (20 images from 1 synthetic video)
            fake/   (20 images from 1 synthetic video)
        test/
            real/   (20 images from 1 synthetic video)
            fake/   (20 images from 1 synthetic video)
        metadata.json

Usage::

    python generate_demo_data.py --output_dir data/processed
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Image size matching EfficientNet-B0 input
CROP_SIZE = 224
JPEG_QUALITY = 95

# Demo dataset layout: (num_videos, frames_per_video) per split
SPLIT_CONFIG = {
    "train": 4,   # videos per class
    "val":   1,
    "test":  1,
}
FRAMES_PER_VIDEO = 20


def generate_real_face(rng: np.random.Generator, frame_idx: int) -> np.ndarray:
    """Generate a synthetic 'real' face image.

    Uses smooth skin-tone gradients with natural noise to simulate
    a genuine face crop.  The pattern is consistent within a video
    (seeded by the generator state) but varies across frames.
    """
    h, w = CROP_SIZE, CROP_SIZE

    # Smooth skin-tone base (warm beige gradient)
    y_grad = np.linspace(0.55, 0.75, h).reshape(h, 1, 1)
    x_grad = np.linspace(0.45, 0.65, w).reshape(1, w, 1)

    # RGB channels with natural skin variation
    base = np.zeros((h, w, 3), dtype=np.float64)
    base[:, :, 0] = (y_grad[:, :, 0] * 0.85 + x_grad[:, :, 0] * 0.15)  # R
    base[:, :, 1] = (y_grad[:, :, 0] * 0.65 + x_grad[:, :, 0] * 0.20)  # G
    base[:, :, 2] = (y_grad[:, :, 0] * 0.50 + x_grad[:, :, 0] * 0.15)  # B

    # Elliptical face shape mask
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    mask = ((X - cx) / (w * 0.38)) ** 2 + ((Y - cy) / (h * 0.45)) ** 2
    face_mask = np.clip(1.0 - mask, 0, 1)[:, :, np.newaxis]

    # Background (darker, cooler tone)
    bg = np.full((h, w, 3), [0.25, 0.28, 0.32], dtype=np.float64)

    img = base * face_mask + bg * (1 - face_mask)

    # Natural Gaussian noise (low amplitude)
    noise = rng.normal(0, 0.015, (h, w, 3))
    img = np.clip(img + noise, 0, 1)

    # Subtle temporal variation (simulates slight head movement)
    shift = np.sin(frame_idx * 0.3) * 0.02
    img = np.clip(img + shift, 0, 1)

    return (img * 255).astype(np.uint8)


def generate_fake_face(rng: np.random.Generator, frame_idx: int) -> np.ndarray:
    """Generate a synthetic 'fake' face image.

    Similar to real faces but with detectable artifacts that a model
    can learn to distinguish:
      - Slightly different color distribution (cooler tones)
      - Subtle grid/block artifacts (simulating compression from generation)
      - Sharper edges at the face boundary (blending artifacts)
      - Higher noise in specific frequency bands
    """
    h, w = CROP_SIZE, CROP_SIZE

    # Slightly different skin tone (shifted toward pink/cool)
    y_grad = np.linspace(0.58, 0.72, h).reshape(h, 1, 1)
    x_grad = np.linspace(0.48, 0.62, w).reshape(1, w, 1)

    base = np.zeros((h, w, 3), dtype=np.float64)
    base[:, :, 0] = (y_grad[:, :, 0] * 0.88 + x_grad[:, :, 0] * 0.12)  # R (higher)
    base[:, :, 1] = (y_grad[:, :, 0] * 0.58 + x_grad[:, :, 0] * 0.18)  # G (lower)
    base[:, :, 2] = (y_grad[:, :, 0] * 0.55 + x_grad[:, :, 0] * 0.18)  # B (higher)

    # Face shape with sharper boundary (blending artifact)
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    mask = ((X - cx) / (w * 0.36)) ** 2 + ((Y - cy) / (h * 0.43)) ** 2
    face_mask = np.clip(1.0 - mask * 1.5, 0, 1)[:, :, np.newaxis]

    bg = np.full((h, w, 3), [0.22, 0.26, 0.35], dtype=np.float64)
    img = base * face_mask + bg * (1 - face_mask)

    # Block artifacts (8x8 grid, simulating GAN checkerboard)
    block_size = 8
    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            block_shift = rng.normal(0, 0.008)
            img[by:by + block_size, bx:bx + block_size] += block_shift

    # Higher-amplitude noise (fake images tend to have different noise profile)
    noise = rng.normal(0, 0.025, (h, w, 3))
    img = np.clip(img + noise, 0, 1)

    # Subtle periodic pattern (GAN fingerprint)
    freq = rng.uniform(0.08, 0.12)
    pattern = np.sin(np.arange(w) * freq)[np.newaxis, :, np.newaxis] * 0.01
    img = np.clip(img + pattern, 0, 1)

    # Temporal jitter (less smooth than real)
    shift = np.sin(frame_idx * 0.5) * 0.03 + rng.normal(0, 0.01)
    img = np.clip(img + shift, 0, 1)

    return (img * 255).astype(np.uint8)


def generate_demo_dataset(output_dir: Path, seed: int = 42) -> dict:
    """Generate the complete demo dataset.

    Args:
        output_dir: Root output directory.
        seed: Random seed for reproducibility.

    Returns:
        Metadata dict.
    """
    rng = np.random.default_rng(seed)

    videos: dict = {}
    split_mapping: dict = {"train": [], "val": [], "test": []}

    video_counter = 0

    for split_name, num_videos_per_class in SPLIT_CONFIG.items():
        for class_label in ["real", "fake"]:
            class_dir = output_dir / split_name / class_label
            class_dir.mkdir(parents=True, exist_ok=True)

            for v in range(num_videos_per_class):
                video_id = f"demo_{class_label}_{video_counter:03d}"
                video_counter += 1

                frames_meta = []

                for f_idx in range(FRAMES_PER_VIDEO):
                    # Generate image
                    if class_label == "real":
                        img = generate_real_face(rng, f_idx)
                    else:
                        img = generate_fake_face(rng, f_idx)

                    # Save
                    filename = f"{video_id}_frame_{f_idx:04d}.jpg"
                    filepath = class_dir / filename
                    cv2.imwrite(
                        str(filepath),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                    )

                    # Metadata entry (relative path from output_dir)
                    rel_path = f"{split_name}/{class_label}/{filename}"
                    frames_meta.append({
                        "frame_idx": f_idx,
                        "face_crop": rel_path,
                    })

                # Video metadata
                videos[video_id] = {
                    "label": class_label,
                    "source": "original" if class_label == "real" else "Deepfakes",
                    "frames": frames_meta,
                }
                split_mapping[split_name].append(video_id)

    # Build metadata in the format expected by train_lstm.py
    metadata = {
        "dataset": "SilentWitness-Demo",
        "description": "Synthetic demo data for pipeline testing",
        "crop_size": CROP_SIZE,
        "frames_per_video": FRAMES_PER_VIDEO,
        "videos": videos,
        "split": split_mapping,
        "counts": {},
    }

    # Compute counts per split
    for split_name in ["train", "val", "test"]:
        split_vids = split_mapping[split_name]
        real_count = sum(1 for vid in split_vids if "real" in vid)
        fake_count = sum(1 for vid in split_vids if "fake" in vid)
        real_frames = real_count * FRAMES_PER_VIDEO
        fake_frames = fake_count * FRAMES_PER_VIDEO
        metadata["counts"][split_name] = {
            "total_videos": len(split_vids),
            "real_videos": real_count,
            "fake_videos": fake_count,
            "total_frames_extracted": real_frames + fake_frames,
            "real_frames_extracted": real_frames,
            "fake_frames_extracted": fake_frames,
        }

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic demo data for the SilentWitness ML pipeline.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for synthetic face crops and metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()

    print(f"Generating demo dataset in {output_dir} ...")

    total_images = sum(SPLIT_CONFIG.values()) * 2 * FRAMES_PER_VIDEO
    print(f"  {total_images} synthetic face crops "
          f"({sum(SPLIT_CONFIG.values())} videos/class, "
          f"{FRAMES_PER_VIDEO} frames/video)")

    metadata = generate_demo_dataset(output_dir, seed=args.seed)

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Summary
    for split_name in ["train", "val", "test"]:
        c = metadata["counts"][split_name]
        print(f"  {split_name:5s}: {c['real_frames_extracted']} real + "
              f"{c['fake_frames_extracted']} fake frames "
              f"({c['total_videos']} videos)")

    print(f"  Metadata: {metadata_path}")
    print("Done.")


if __name__ == "__main__":
    main()
