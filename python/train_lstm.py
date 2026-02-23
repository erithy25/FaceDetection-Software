#!/usr/bin/env python3
"""
Train the Temporal LSTM layer for deepfake detection.

This script:
  1. Loads a trained EfficientNet-B0 model (backbone + embedding layer)
  2. Extracts 256-dim embeddings from all preprocessed face crops
  3. Groups embeddings by source video using metadata.json
  4. Creates sliding-window sequences of 16 consecutive frame embeddings
     (stride=8, 50% overlap)
  5. Trains a single-layer LSTM (input=256, hidden=128, output=1 scalar)
     with BCE loss for binary classification (real vs fake)
  6. Saves the best LSTM checkpoint based on validation loss

Usage:
    python train_lstm.py \
        --data_dir /path/to/preprocessed_dataset \
        --efficientnet_model /path/to/efficientnet_b0.pt \
        --output_dir /path/to/output \
        --epochs 50 \
        --batch_size 32 \
        --patience 7
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Shared model architecture (single source of truth)
from models import (
    EfficientNetEmbedder,
    TemporalLSTM,
    load_efficientnet_embedder,
    EMBEDDING_DIM,
    LSTM_HIDDEN_DIM,
    TEMPORAL_WINDOW,
    TEMPORAL_STRIDE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_lstm")


# EfficientNetEmbedder, TemporalLSTM imported from models.py (single source of truth)


# ===================================================================
# Dataset
# ===================================================================

class EmbeddingSequenceDataset(Dataset):
    """Dataset of fixed-length embedding sequences for LSTM training.

    Each sample is a tuple of:
        - sequence: (TEMPORAL_WINDOW, EMBEDDING_DIM) float32 tensor
        - label:    scalar float32 (1.0 = real, 0.0 = fake)
    """

    def __init__(
        self,
        sequences: list[np.ndarray],
        labels: list[float],
    ) -> None:
        assert len(sequences) == len(labels), (
            f"Mismatch: {len(sequences)} sequences vs {len(labels)} labels"
        )
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = torch.from_numpy(self.sequences[idx]).float()  # (T, 256)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq, label


# load_efficientnet_embedder imported from models.py


# ===================================================================
# Helper: face-crop image dataset for batch embedding extraction
# ===================================================================

class FaceCropDataset(Dataset):
    """Simple image dataset that reads face crop PNGs/JPGs from disk.

    Returns (image_tensor, image_path) so the caller can associate
    embeddings back with their source files.
    """

    def __init__(self, image_paths: list[str]) -> None:
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        import cv2

        path = self.image_paths[idx]
        # Read as BGR, convert to RGB, resize, normalise — same as detector.py
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            # Return a zero tensor for corrupt images; will be filtered later
            return torch.zeros(3, 224, 224, dtype=torch.float32), path

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        img_norm = img_resized.astype(np.float32) / 255.0
        img_norm = (img_norm - IMAGENET_MEAN) / IMAGENET_STD

        # HWC -> CHW
        tensor = torch.from_numpy(np.transpose(img_norm, (2, 0, 1))).float()
        return tensor, path


# ===================================================================
# Core pipeline functions
# ===================================================================

def discover_face_crops(data_dir: Path) -> tuple[list[str], dict]:
    """Read metadata.json and discover all face crop image paths.

    Expected metadata.json structure (produced by preprocess_dataset.py):
    {
        "videos": {
            "<video_id>": {
                "label": "real" | "fake",
                "source": "original" | "Deepfakes" | ...,
                "frames": [
                    {"frame_idx": 0, "face_crop": "relative/path/to/crop.png"},
                    ...
                ]
            },
            ...
        },
        "split": {
            "train": ["vid1", "vid2", ...],
            "val": ["vid3", ...],
            "test": ["vid4", ...]
        }
    }

    Returns:
        (all_image_paths, metadata_dict)
    """
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {metadata_path}. "
            "Run preprocess_dataset.py first."
        )

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    all_paths: list[str] = []
    videos = metadata.get("videos", {})

    for video_id, info in videos.items():
        for frame in info.get("frames", []):
            crop_rel = frame.get("face_crop", "")
            if crop_rel:
                full_path = str(data_dir / crop_rel)
                all_paths.append(full_path)

    logger.info(
        f"Discovered {len(all_paths)} face crops across "
        f"{len(videos)} videos"
    )
    return all_paths, metadata


@torch.no_grad()
def extract_embeddings(
    embedder: EfficientNetEmbedder,
    image_paths: list[str],
    batch_size: int,
    device: torch.device,
    num_workers: int = 4,
) -> dict[str, np.ndarray]:
    """Extract 256-dim embeddings for every face crop.

    Args:
        embedder: Loaded EfficientNetEmbedder in eval mode.
        image_paths: List of absolute paths to face crop images.
        batch_size: Batch size for inference.
        device: Torch device.
        num_workers: DataLoader workers.

    Returns:
        Dict mapping image_path -> (256,) numpy embedding.
    """
    logger.info(f"Extracting embeddings for {len(image_paths)} face crops ...")

    dataset = FaceCropDataset(image_paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    path_to_embedding: dict[str, np.ndarray] = {}

    for batch_imgs, batch_paths in tqdm(loader, desc="Extracting embeddings"):
        batch_imgs = batch_imgs.to(device)
        embeddings = embedder(batch_imgs).cpu().numpy()  # (B, 256)

        for i, path in enumerate(batch_paths):
            path_to_embedding[path] = embeddings[i]

    logger.info(f"Extracted {len(path_to_embedding)} embeddings")
    return path_to_embedding


def build_sequences(
    metadata: dict,
    path_to_embedding: dict[str, np.ndarray],
    data_dir: Path,
    seq_len: int = TEMPORAL_WINDOW,
    stride: int = TEMPORAL_STRIDE,
) -> tuple[list[np.ndarray], list[float]]:
    """Group embeddings by video and build sliding-window sequences.

    For each video with N frames, generates sequences starting at
    indices [0, stride, 2*stride, ...] each of length ``seq_len``.
    Videos with fewer than ``seq_len`` frames are skipped.

    Args:
        metadata: Parsed metadata.json.
        path_to_embedding: Mapping from crop path to embedding vector.
        data_dir: Root of the preprocessed dataset.
        seq_len: Number of frames per sequence.
        stride: Sliding-window step size.

    Returns:
        (sequences, labels) where each sequence is (seq_len, 256) ndarray
        and each label is 1.0 (real) or 0.0 (fake).
    """
    videos = metadata.get("videos", {})
    sequences: list[np.ndarray] = []
    labels: list[float] = []
    skipped_videos = 0

    for video_id, info in videos.items():
        label_str = info.get("label", "").lower()
        label = 1.0 if label_str == "real" else 0.0

        # Collect embeddings for this video's frames in order
        frame_embeddings: list[np.ndarray] = []
        for frame in info.get("frames", []):
            crop_rel = frame.get("face_crop", "")
            if not crop_rel:
                continue
            full_path = str(data_dir / crop_rel)
            emb = path_to_embedding.get(full_path)
            if emb is not None:
                frame_embeddings.append(emb)

        if len(frame_embeddings) < seq_len:
            skipped_videos += 1
            continue

        # Build sliding-window sequences
        for start in range(0, len(frame_embeddings) - seq_len + 1, stride):
            window = frame_embeddings[start : start + seq_len]
            sequences.append(np.stack(window, axis=0))  # (seq_len, 256)
            labels.append(label)

    logger.info(
        f"Built {len(sequences)} sequences from {len(videos)} videos "
        f"({skipped_videos} videos skipped — fewer than {seq_len} frames)"
    )
    return sequences, labels


def split_sequences(
    metadata: dict,
    all_sequences: list[np.ndarray],
    all_labels: list[float],
    data_dir: Path,
    path_to_embedding: dict[str, np.ndarray],
) -> tuple[
    tuple[list[np.ndarray], list[float]],
    tuple[list[np.ndarray], list[float]],
]:
    """Split sequences into train/val sets using metadata split info.

    If metadata contains a ``split`` key with ``train`` and ``val``
    video lists, those are used. Otherwise, we fall back to a simple
    80/20 random split at the *video* level to avoid data leakage
    (no sequences from the same video appear in both splits).

    Returns:
        ((train_seqs, train_labels), (val_seqs, val_labels))
    """
    split_info = metadata.get("split", {})
    train_ids = set(split_info.get("train", []))
    val_ids = set(split_info.get("val", []))

    videos = metadata.get("videos", {})

    if train_ids and val_ids:
        logger.info(
            f"Using metadata split: {len(train_ids)} train videos, "
            f"{len(val_ids)} val videos"
        )
        return _build_split_sequences(
            videos, train_ids, val_ids, data_dir, path_to_embedding,
        )

    # Fallback: 80/20 random split at video level
    logger.info("No split info in metadata; performing 80/20 video-level split")
    video_ids = list(videos.keys())
    rng = np.random.RandomState(42)
    rng.shuffle(video_ids)
    split_point = int(0.8 * len(video_ids))
    train_ids = set(video_ids[:split_point])
    val_ids = set(video_ids[split_point:])

    return _build_split_sequences(
        videos, train_ids, val_ids, data_dir, path_to_embedding,
    )


def _build_split_sequences(
    videos: dict,
    train_ids: set[str],
    val_ids: set[str],
    data_dir: Path,
    path_to_embedding: dict[str, np.ndarray],
    seq_len: int = TEMPORAL_WINDOW,
    stride: int = TEMPORAL_STRIDE,
) -> tuple[
    tuple[list[np.ndarray], list[float]],
    tuple[list[np.ndarray], list[float]],
]:
    """Build sequences separately for train and val video sets."""
    train_seqs: list[np.ndarray] = []
    train_labels: list[float] = []
    val_seqs: list[np.ndarray] = []
    val_labels: list[float] = []

    for video_id, info in videos.items():
        label_str = info.get("label", "").lower()
        label = 1.0 if label_str == "real" else 0.0

        frame_embeddings: list[np.ndarray] = []
        for frame in info.get("frames", []):
            crop_rel = frame.get("face_crop", "")
            if not crop_rel:
                continue
            full_path = str(data_dir / crop_rel)
            emb = path_to_embedding.get(full_path)
            if emb is not None:
                frame_embeddings.append(emb)

        if len(frame_embeddings) < seq_len:
            continue

        for start in range(0, len(frame_embeddings) - seq_len + 1, stride):
            window = frame_embeddings[start : start + seq_len]
            seq = np.stack(window, axis=0)

            if video_id in train_ids:
                train_seqs.append(seq)
                train_labels.append(label)
            elif video_id in val_ids:
                val_seqs.append(seq)
                val_labels.append(label)

    logger.info(
        f"Train: {len(train_seqs)} sequences | Val: {len(val_seqs)} sequences"
    )
    return (train_seqs, train_labels), (val_seqs, val_labels)


# ===================================================================
# Training loop
# ===================================================================

def train_one_epoch(
    model: TemporalLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for a single epoch.

    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in tqdm(loader, desc="  Train", leave=False):
        sequences = sequences.to(device)  # (B, T, 256)
        labels = labels.to(device)        # (B,)

        optimizer.zero_grad()
        logits = model(sequences)          # (B,)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * sequences.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += sequences.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: TemporalLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on the validation set.

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in tqdm(loader, desc="  Val  ", leave=False):
        sequences = sequences.to(device)
        labels = labels.to(device)

        logits = model(sequences)
        loss = criterion(logits, labels)

        total_loss += loss.item() * sequences.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += sequences.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train(
    model: TemporalLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 7,
) -> Path:
    """Full training loop with early stopping on validation loss.

    Args:
        model: The TemporalLSTM model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Torch device.
        output_dir: Directory to save checkpoints.
        epochs: Maximum number of epochs.
        lr: Learning rate for Adam.
        patience: Early-stopping patience (epochs without improvement).

    Returns:
        Path to the best saved checkpoint.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_checkpoint_path = output_dir / "temporal_lstm_best.pt"

    logger.info(
        f"Starting training: {epochs} epochs, lr={lr}, patience={patience}, "
        f"device={device}"
    )

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            # Save best checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "architecture": {
                        "input_size": EMBEDDING_DIM,
                        "hidden_size": LSTM_HIDDEN_DIM,
                        "num_layers": 1,
                        "seq_len": TEMPORAL_WINDOW,
                    },
                },
                best_checkpoint_path,
            )
            logger.info(
                f"  -> New best model saved (val_loss={val_loss:.4f})"
            )
        else:
            epochs_without_improvement += 1
            logger.info(
                f"  -> No improvement for {epochs_without_improvement}/{patience} epochs"
            )

        if epochs_without_improvement >= patience:
            logger.info(
                f"Early stopping triggered after {epoch} epochs "
                f"(best val_loss={best_val_loss:.4f})"
            )
            break

    logger.info(f"Training complete. Best checkpoint: {best_checkpoint_path}")
    return best_checkpoint_path


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the temporal LSTM layer for deepfake detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the preprocessed dataset directory (contains metadata.json "
             "and face crop images produced by preprocess_dataset.py).",
    )
    parser.add_argument(
        "--efficientnet_model",
        type=str,
        required=True,
        help="Path to the trained EfficientNet-B0 .pt checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_lstm",
        help="Directory to save model checkpoints and logs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for both embedding extraction and LSTM training.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early-stopping patience (epochs without val loss improvement).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Setup -------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: discover face crops from metadata -------------------
    logger.info("=" * 60)
    logger.info("Step 1/4: Discovering face crops from metadata.json")
    logger.info("=" * 60)

    all_image_paths, metadata = discover_face_crops(data_dir)
    if not all_image_paths:
        logger.error("No face crops found. Aborting.")
        sys.exit(1)

    # ---- Step 2: extract embeddings with EfficientNet ----------------
    logger.info("=" * 60)
    logger.info("Step 2/4: Extracting embeddings with EfficientNet-B0")
    logger.info("=" * 60)

    embedder = load_efficientnet_embedder(args.efficientnet_model, device)
    path_to_embedding = extract_embeddings(
        embedder,
        all_image_paths,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
    )

    # Free GPU memory used by EfficientNet
    del embedder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Step 3: build sequences with sliding window -----------------
    logger.info("=" * 60)
    logger.info("Step 3/4: Building temporal sequences (sliding window)")
    logger.info("=" * 60)

    (train_seqs, train_labels), (val_seqs, val_labels) = split_sequences(
        metadata, [], [], data_dir, path_to_embedding
    )

    if not train_seqs:
        logger.error(
            "No training sequences could be built. Ensure videos have "
            f"at least {TEMPORAL_WINDOW} frames each."
        )
        sys.exit(1)
    if not val_seqs:
        logger.warning(
            "No validation sequences found. Falling back to using 20% "
            "of training sequences for validation."
        )
        # Emergency split at the sequence level (not ideal but usable)
        rng = np.random.RandomState(args.seed)
        indices = rng.permutation(len(train_seqs))
        split_pt = int(0.8 * len(indices))
        val_indices = indices[split_pt:]
        train_indices = indices[:split_pt]
        val_seqs = [train_seqs[i] for i in val_indices]
        val_labels = [train_labels[i] for i in val_indices]
        train_seqs = [train_seqs[i] for i in train_indices]
        train_labels = [train_labels[i] for i in train_indices]

    # Log class balance
    n_train_real = sum(1 for l in train_labels if l == 1.0)
    n_train_fake = sum(1 for l in train_labels if l == 0.0)
    n_val_real = sum(1 for l in val_labels if l == 1.0)
    n_val_fake = sum(1 for l in val_labels if l == 0.0)
    logger.info(
        f"Train: {len(train_seqs)} sequences "
        f"({n_train_real} real, {n_train_fake} fake)"
    )
    logger.info(
        f"Val:   {len(val_seqs)} sequences "
        f"({n_val_real} real, {n_val_fake} fake)"
    )

    # Create DataLoaders
    train_dataset = EmbeddingSequenceDataset(train_seqs, train_labels)
    val_dataset = EmbeddingSequenceDataset(val_seqs, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Sequences are in-memory; no IO needed
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # ---- Step 4: train LSTM ------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 4/4: Training Temporal LSTM")
    logger.info("=" * 60)

    lstm_model = TemporalLSTM(
        input_size=EMBEDDING_DIM,
        hidden_size=LSTM_HIDDEN_DIM,
        num_layers=1,
    ).to(device)

    logger.info(f"LSTM architecture:\n{lstm_model}")
    total_params = sum(p.numel() for p in lstm_model.parameters())
    trainable_params = sum(
        p.numel() for p in lstm_model.parameters() if p.requires_grad
    )
    logger.info(
        f"Parameters: {total_params:,} total, {trainable_params:,} trainable"
    )

    best_ckpt = train(
        model=lstm_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )

    # ---- Summary -----------------------------------------------------
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    logger.info(f"Best epoch:     {ckpt['epoch']}")
    logger.info(f"Best val loss:  {ckpt['val_loss']:.4f}")
    logger.info(f"Best val acc:   {ckpt['val_acc']:.4f}")
    logger.info(f"Checkpoint:     {best_ckpt}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
