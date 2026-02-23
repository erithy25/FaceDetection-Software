#!/usr/bin/env python3
"""
EfficientNet-B0 Fine-Tuning for Deepfake Detection.

Fine-tunes a pretrained EfficientNet-B0 on face crops organized into
{train,val,test}/{real,fake}/ directory structure (produced by
preprocess_dataset.py).  The model outputs both a binary classification
logit and a 256-dim embedding that is consumed downstream by the
temporal LSTM module.

Architecture
------------
  EfficientNet-B0 backbone (ImageNet)
    -> avgpool -> 1280-dim
    -> Linear(1280, 256) + ReLU   (embedding)
    -> Dropout(0.3)
    -> Linear(256, 1)             (logit)

Usage
-----
  python train_efficientnet.py \
      --data_dir /path/to/dataset \
      --output_dir /path/to/checkpoints \
      --epochs 30 --batch_size 32

The script logs AUC, accuracy and loss per epoch, saves the best model
checkpoint (by validation AUC), and writes a JSON training history.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("silentwitness.train")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 256
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEED = 42


# ===================================================================
# Reproducibility
# ===================================================================

def seed_everything(seed: int = SEED) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms can be slower but ensure exact reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set to %d", seed)


# ===================================================================
# Custom JPEG compression transform
# ===================================================================

class RandomJPEGCompression:
    """Apply random JPEG compression as a data-augmentation transform.

    Simulates the artefacts introduced by social-media re-encoding and
    video compression, which are common in deepfake distribution pipelines.

    Args:
        quality_min: Minimum JPEG quality (inclusive).
        quality_max: Maximum JPEG quality (inclusive).
    """

    def __init__(self, quality_min: int = 70, quality_max: int = 100) -> None:
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, img: Image.Image) -> Image.Image:
        import io
        quality = random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(quality_min={self.quality_min}, quality_max={self.quality_max})"
        )


# ===================================================================
# Dataset
# ===================================================================

class DeepfakeFaceDataset(Dataset):
    """Dataset for loading face crops from a directory hierarchy.

    Expected layout (produced by ``preprocess_dataset.py``)::

        root/
            real/
                img_0001.png
                img_0002.jpg
                ...
            fake/
                img_0001.png
                ...

    Labels: real = 1, fake = 0.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        # Collect real images (label=1)
        real_dir = self.root_dir / "real"
        if real_dir.is_dir():
            for fp in sorted(real_dir.iterdir()):
                if fp.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((fp, 1))

        # Collect fake images (label=0)
        fake_dir = self.root_dir / "fake"
        if fake_dir.is_dir():
            for fp in sorted(fake_dir.iterdir()):
                if fp.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((fp, 0))

        if len(self.samples) == 0:
            logger.warning(
                "No images found under %s — expected real/ and fake/ "
                "subdirectories containing image files.",
                self.root_dir,
            )

        # Class balance info
        n_real = sum(1 for _, lbl in self.samples if lbl == 1)
        n_fake = sum(1 for _, lbl in self.samples if lbl == 0)
        logger.info(
            "Loaded %d samples from %s  (real=%d, fake=%d)",
            len(self.samples),
            self.root_dir,
            n_real,
            n_fake,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


# ===================================================================
# Model
# ===================================================================

class EfficientNetDeepfake(nn.Module):
    """EfficientNet-B0 for binary deepfake detection.

    Architecture
    ------------
    The standard torchvision EfficientNet-B0 backbone extracts a 1280-dim
    feature vector after global average pooling.  The custom classification
    head maps this to a 256-dim embedding (used later by the temporal LSTM)
    and then to a single logit for binary classification::

        features (1280) -> Linear(1280, 256) -> ReLU -> Dropout -> Linear(256, 1)

    The ``classifier`` attribute is stored as an ``nn.Sequential`` so that
    the saved ``state_dict`` is directly loadable by the inference code in
    ``gradcam.py`` and by the ONNX export pipeline, both of which expect
    ``model.classifier = Sequential(Linear, ReLU, Linear)``.

    Forward returns ``(logit, embedding)`` so both the binary prediction
    and the 256-dim temporal feature are available during training.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        # Load the full EfficientNet-B0 from torchvision
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            base = models.efficientnet_b0(weights=weights)
            logger.info("Loaded EfficientNet-B0 with ImageNet pretrained weights")
        else:
            base = models.efficientnet_b0(weights=None)
            logger.info("Loaded EfficientNet-B0 without pretrained weights")

        # Keep the convolutional backbone and pooling layer
        self.features = base.features
        self.avgpool = base.avgpool

        # Custom classification head.
        # We store it as model.classifier so the state_dict keys match
        # the structure expected by gradcam.py and ONNX export:
        #   classifier.0  -> Linear(1280, 256)
        #   classifier.1  -> ReLU
        #   classifier.2  -> Linear(256, 1)
        # A Dropout layer is inserted at index 2 during training only;
        # at export / inference time the dropout is a no-op (eval mode).
        self.classifier = nn.Sequential(
            nn.Linear(1280, EMBEDDING_DIM),       # 0
            nn.ReLU(inplace=True),                # 1
            nn.Dropout(p=0.3),                    # 2
            nn.Linear(EMBEDDING_DIM, 1),          # 3
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, 3, 224, 224)``.

        Returns:
            logit: Raw classification logit of shape ``(B, 1)``.
            embedding: 256-dim embedding of shape ``(B, 256)``.
        """
        x = self.features(x)                     # (B, 1280, 7, 7)
        x = self.avgpool(x)                      # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)                  # (B, 1280)

        # Run through the head piece-by-piece so we can tap the embedding
        embedding = self.classifier[1](           # ReLU
            self.classifier[0](x)                 # Linear(1280, 256)
        )                                         # (B, 256)
        logit = self.classifier[3](               # Linear(256, 1)
            self.classifier[2](embedding)         # Dropout
        )                                         # (B, 1)

        return logit, embedding


# ===================================================================
# Data augmentation helpers
# ===================================================================

def build_train_transforms() -> transforms.Compose:
    """Build the training-time augmentation pipeline.

    Augmentations (per spec):
      - Random horizontal flip
      - Random rotation +/-5 degrees
      - Color jitter (brightness, contrast, saturation +/-10%)
      - Random JPEG compression (quality 70-100)
      - Random Gaussian blur (sigma 0-0.5)
      - Resize to 224x224 and ImageNet normalisation
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
        ),
        RandomJPEGCompression(quality_min=70, quality_max=100),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.001, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_val_transforms() -> transforms.Compose:
    """Build the validation / test augmentation pipeline (deterministic)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ===================================================================
# Training & evaluation loops
# ===================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Run a single training epoch.

    Returns:
        Dictionary with ``loss``, ``accuracy``, and ``auc`` for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels: list[float] = []
    all_probs: list[float] = []

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False, unit="batch")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits, _ = model(images)
        logits = logits.squeeze(1)  # (B,)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        # --- Metrics ---
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(np.float32)
        labels_np = labels.detach().cpu().numpy()

        correct += (preds == labels_np).sum()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels_np.tolist())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)

    # AUC requires both classes present
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return {"loss": avg_loss, "accuracy": float(accuracy), "auc": float(auc)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    split: str = "Val",
) -> dict[str, float]:
    """Evaluate the model on a validation or test set.

    Returns:
        Dictionary with ``loss``, ``accuracy``, and ``auc``.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels: list[float] = []
    all_probs: list[float] = []

    pbar = tqdm(loader, desc=f"{split} Epoch {epoch}", leave=False, unit="batch")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, _ = model(images)
        logits = logits.squeeze(1)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(np.float32)
        labels_np = labels.cpu().numpy()

        correct += (preds == labels_np).sum()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels_np.tolist())

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return {"loss": avg_loss, "accuracy": float(accuracy), "auc": float(auc)}


# ===================================================================
# Early stopping
# ===================================================================

class EarlyStopping:
    """Early stopping based on validation loss.

    Training is halted when the validation loss does not improve for
    ``patience`` consecutive epochs.
    """

    def __init__(self, patience: int = 5) -> None:
        self.patience = patience
        self.best_loss: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Update state with the latest validation loss.

        Returns:
            ``True`` if training should be stopped.
        """
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ===================================================================
# Main training procedure
# ===================================================================

def train(args: argparse.Namespace) -> None:
    """Full training pipeline: data loading, training loop, evaluation."""

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    seed_everything(SEED)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info(
            "GPU: %s  |  CUDA %s",
            torch.cuda.get_device_name(0),
            torch.version.cuda,
        )

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    train_dataset = DeepfakeFaceDataset(train_dir, transform=build_train_transforms())
    val_dataset = DeepfakeFaceDataset(val_dir, transform=build_val_transforms())

    if len(train_dataset) == 0:
        logger.error("Training set is empty.  Aborting.")
        sys.exit(1)
    if len(val_dataset) == 0:
        logger.error("Validation set is empty.  Aborting.")
        sys.exit(1)

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    logger.info(
        "Data loaders ready  (train=%d batches, val=%d batches)",
        len(train_loader),
        len(val_loader),
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = EfficientNetDeepfake(pretrained=True).to(device)
    logger.info(
        "Model parameters: %.2fM",
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    # ------------------------------------------------------------------
    # Optimizer — differential learning rates
    # ------------------------------------------------------------------
    backbone_params = list(model.features.parameters())
    head_params = list(model.classifier.parameters())

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": args.learning_rate * 0.01},  # 1e-5
            {"params": head_params, "lr": args.learning_rate},             # 1e-3
        ],
        weight_decay=1e-4,
    )
    logger.info(
        "AdamW optimizer  (backbone lr=%.1e, head lr=%.1e)",
        args.learning_rate * 0.01,
        args.learning_rate,
    )

    # ------------------------------------------------------------------
    # Scheduler — cosine annealing with warm restarts
    # ------------------------------------------------------------------
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,       # Restart period in epochs
        T_mult=2,     # Double the period after each restart
        eta_min=1e-7,
    )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    early_stopping = EarlyStopping(patience=args.patience)
    best_val_auc = 0.0
    history: list[dict[str, Any]] = []

    checkpoint_path = output_dir / "best_efficientnet_b0_deepfake.pt"

    logger.info("=" * 60)
    logger.info("Starting training for %d epochs", args.epochs)
    logger.info("=" * 60)

    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # --- Train ---
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # --- Validate ---
        val_metrics = evaluate(
            model, val_loader, criterion, device, epoch, split="Val"
        )

        # --- Scheduler step ---
        scheduler.step(epoch)

        epoch_time = time.time() - epoch_start

        # --- Logging ---
        current_backbone_lr = optimizer.param_groups[0]["lr"]
        current_head_lr = optimizer.param_groups[1]["lr"]

        logger.info(
            "Epoch %02d/%02d [%.0fs]  "
            "Train  loss=%.4f  acc=%.4f  auc=%.4f  |  "
            "Val  loss=%.4f  acc=%.4f  auc=%.4f  |  "
            "LR backbone=%.2e head=%.2e",
            epoch,
            args.epochs,
            epoch_time,
            train_metrics["loss"],
            train_metrics["accuracy"],
            train_metrics["auc"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["auc"],
            current_backbone_lr,
            current_head_lr,
        )

        # --- Record history ---
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "lr_backbone": current_backbone_lr,
            "lr_head": current_head_lr,
            "epoch_time_s": round(epoch_time, 2),
        })

        # --- Save best model based on val AUC ---
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(
                "  -> New best val AUC=%.4f  — model saved to %s",
                best_val_auc,
                checkpoint_path,
            )

        # --- Early stopping on val loss ---
        if early_stopping.step(val_metrics["loss"]):
            logger.info(
                "Early stopping triggered after %d epochs without "
                "validation loss improvement (patience=%d).",
                early_stopping.counter,
                args.patience,
            )
            break

    total_time = time.time() - total_start
    logger.info("=" * 60)
    logger.info(
        "Training complete in %.1f minutes.  Best val AUC: %.4f",
        total_time / 60.0,
        best_val_auc,
    )
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Test evaluation (if test split exists)
    # ------------------------------------------------------------------
    if test_dir.is_dir():
        test_dataset = DeepfakeFaceDataset(
            test_dir, transform=build_val_transforms()
        )
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )

            # Load the best checkpoint for test evaluation
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logger.info("Loaded best checkpoint for test evaluation")

            test_metrics = evaluate(
                model, test_loader, criterion, device, epoch=0, split="Test"
            )
            logger.info(
                "Test results  —  loss=%.4f  acc=%.4f  auc=%.4f",
                test_metrics["loss"],
                test_metrics["accuracy"],
                test_metrics["auc"],
            )

            # Append test metrics to history
            history.append({
                "split": "test",
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_auc": test_metrics["auc"],
            })
        else:
            logger.warning("Test directory exists but contains no images.")
    else:
        logger.info("No test split found at %s — skipping test evaluation.", test_dir)

    # ------------------------------------------------------------------
    # Save training history
    # ------------------------------------------------------------------
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(
            {
                "config": {
                    "data_dir": str(args.data_dir),
                    "output_dir": str(args.output_dir),
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "patience": args.patience,
                    "num_workers": args.num_workers,
                    "device": str(device),
                    "seed": SEED,
                    "best_val_auc": best_val_auc,
                    "total_time_s": round(total_time, 2),
                },
                "history": history,
            },
            f,
            indent=2,
        )
    logger.info("Training history saved to %s", history_path)
    logger.info("Best model checkpoint saved to %s", checkpoint_path)


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune EfficientNet-B0 for deepfake detection.  "
            "Expects data organised as data_dir/{train,val,test}/{real,fake}/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing train/, val/, and test/ splits.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory for saving model checkpoints and training history.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size for training and evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=(
            "Learning rate for the classification head.  "
            "The backbone uses lr * 0.01 (i.e. 1e-5 by default)."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience (epochs without val loss improvement).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data-loading worker processes.",
    )

    return parser.parse_args()


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)
