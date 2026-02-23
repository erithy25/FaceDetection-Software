"""
SilentWitness — Shared Model Architecture Definitions.

Single source of truth for all neural network architectures used across
training, export, inference, and visualization. Every script that needs
a model imports from here to guarantee consistency.

Architecture Summary:
    EfficientNetDeepfake:
        EfficientNet-B0 backbone (ImageNet pretrained)
        -> AdaptiveAvgPool2d -> Flatten -> 1280-dim
        -> Linear(1280, 256) + ReLU     (embedding layer)
        -> Dropout(0.3)                  (regularization — no-op in eval)
        -> Linear(256, 1)                (classification logit)
        Returns: (logit, embedding) tuple

    TemporalLSTM:
        LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        -> last hidden state h_n[-1] -> Linear(128, 1)
        Returns: logit scalar per sequence
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger("silentwitness.models")

# ───────────────────────────────────────────────────────────────────────
# Architecture Constants — used by all scripts
# ───────────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 256
LSTM_HIDDEN_DIM = 128
TEMPORAL_WINDOW = 16
TEMPORAL_STRIDE = 8
EFFICIENTNET_INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ───────────────────────────────────────────────────────────────────────
# EfficientNet-B0 for Frame-Level Deepfake Detection
# ───────────────────────────────────────────────────────────────────────
class EfficientNetDeepfake(nn.Module):
    """EfficientNet-B0 fine-tuned for binary deepfake detection.

    The forward pass returns both the classification logit and the
    256-dim intermediate embedding (used by the temporal LSTM).

    The classifier head structure is:
        classifier[0] = Linear(1280, 256)
        classifier[1] = ReLU
        classifier[2] = Dropout(0.3)    <-- no-op in eval mode
        classifier[3] = Linear(256, 1)

    IMPORTANT: Dropout MUST always be present to keep state_dict keys
    consistent between training checkpoints and inference/export code.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision import models

        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            base = models.efficientnet_b0(weights=weights)
        else:
            base = models.efficientnet_b0(weights=None)

        self.features = base.features
        self.avgpool = base.avgpool

        # Custom classification head — 4 layers, fixed indexing
        self.classifier = nn.Sequential(
            nn.Linear(1280, EMBEDDING_DIM),   # [0] embedding projection
            nn.ReLU(inplace=True),            # [1] activation
            nn.Dropout(p=0.3),                # [2] regularization
            nn.Linear(EMBEDDING_DIM, 1),      # [3] classification logit
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both logit and embedding.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).

        Returns:
            logit:     Raw classification logit, shape (B, 1).
            embedding: 256-dim feature vector, shape (B, 256).
        """
        x = self.features(x)           # (B, 1280, 7, 7)
        x = self.avgpool(x)            # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)        # (B, 1280)

        # Execute head layer-by-layer to extract the embedding
        embedding = self.classifier[1](
            self.classifier[0](x)
        )                               # (B, 256) — after Linear+ReLU
        logit = self.classifier[3](
            self.classifier[2](embedding)
        )                               # (B, 1) — after Dropout+Linear

        return logit, embedding


class EfficientNetExportWrapper(nn.Module):
    """Thin wrapper for ONNX export that exposes (logit, embedding).

    Wraps an EfficientNetDeepfake instance to ensure the ONNX graph
    returns named outputs compatible with classifier.py inference:
        output[0] = "logit"     (B, 1)
        output[1] = "embedding" (B, 256)
    """

    def __init__(self, model: EfficientNetDeepfake) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)


# ───────────────────────────────────────────────────────────────────────
# EfficientNet Embedding Extractor (for LSTM training)
# ───────────────────────────────────────────────────────────────────────
class EfficientNetEmbedder(nn.Module):
    """Embedding-only extractor from EfficientNet-B0.

    Strips the classification head (Linear(256,1)) and keeps only
    the backbone + embedding projection (Linear(1280,256) + ReLU).
    Used by train_lstm.py to extract embeddings for temporal sequences.
    """

    def __init__(self) -> None:
        super().__init__()
        from torchvision import models

        base = models.efficientnet_b0(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        self.embed_fc = nn.Linear(1280, EMBEDDING_DIM)
        self.embed_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 256-dim embedding from face crop.

        Args:
            x: (B, 3, 224, 224) normalized face crops.

        Returns:
            (B, 256) embedding tensor.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)       # (B, 1280)
        x = self.embed_relu(self.embed_fc(x))  # (B, 256)
        return x


def load_efficientnet_embedder(
    model_path: str,
    device: Optional[torch.device] = None,
) -> EfficientNetEmbedder:
    """Load a trained EfficientNetDeepfake checkpoint into an EfficientNetEmbedder.

    Handles the key remapping between the full model's classifier head
    (Sequential with 4 layers) and the embedder's flat structure:

        classifier.0.weight -> embed_fc.weight
        classifier.0.bias   -> embed_fc.bias
        classifier.1-3.*    -> ignored (ReLU has no params, Dropout/Linear discarded)

    Args:
        model_path: Path to a .pt checkpoint from train_efficientnet.py.
        device: Target device (default: CPU).

    Returns:
        EfficientNetEmbedder in eval mode on the specified device.
    """
    if device is None:
        device = torch.device("cpu")

    state_dict = torch.load(model_path, map_location=device, weights_only=False)

    # Unwrap nested checkpoint formats
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    embedder = EfficientNetEmbedder()
    mapped_dict: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.startswith("features.") or key.startswith("avgpool."):
            mapped_dict[key] = value
        elif key.startswith("classifier.0."):
            # classifier.0 = Linear(1280, 256) → embed_fc
            new_key = key.replace("classifier.0.", "embed_fc.")
            mapped_dict[new_key] = value
        # classifier.1 = ReLU (no params)
        # classifier.2 = Dropout (no params)
        # classifier.3 = Linear(256, 1) — discarded for embedder

    missing, unexpected = embedder.load_state_dict(mapped_dict, strict=False)
    if missing:
        logger.warning(f"Embedder missing keys: {missing}")
    if unexpected:
        logger.warning(f"Embedder unexpected keys: {unexpected}")

    embedder.to(device)
    embedder.eval()
    return embedder


def load_efficientnet_deepfake(
    model_path: str,
    device: Optional[torch.device] = None,
) -> EfficientNetDeepfake:
    """Load a trained EfficientNetDeepfake from a checkpoint.

    Handles multiple checkpoint formats:
        - Raw state_dict
        - {"model_state_dict": ...}
        - {"state_dict": ...}

    Args:
        model_path: Path to a .pt checkpoint.
        device: Target device (default: CPU).

    Returns:
        EfficientNetDeepfake in eval mode on the specified device.
    """
    if device is None:
        device = torch.device("cpu")

    model = EfficientNetDeepfake(pretrained=False)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ───────────────────────────────────────────────────────────────────────
# Temporal LSTM for Sequence-Level Deepfake Detection
# ───────────────────────────────────────────────────────────────────────
class TemporalLSTM(nn.Module):
    """Single-layer LSTM for temporal deepfake detection.

    Takes a sequence of 16 frame embeddings (256-dim each) and outputs
    a single scalar logit indicating real vs fake.

    Architecture:
        LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        -> last hidden state h_n[-1]
        -> Linear(128, 1) -> logit
    """

    def __init__(
        self,
        input_size: int = EMBEDDING_DIM,
        hidden_size: int = LSTM_HIDDEN_DIM,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce a per-sequence logit.

        Args:
            x: (B, seq_len, input_size) embedding sequences.

        Returns:
            (B,) raw logits. Apply sigmoid externally for probabilities.
        """
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])     # (B, 1)
        return out.squeeze(-1)     # (B,)


def load_temporal_lstm(
    model_path: str,
    device: Optional[torch.device] = None,
) -> TemporalLSTM:
    """Load a trained TemporalLSTM from a checkpoint.

    Args:
        model_path: Path to a .pt checkpoint.
        device: Target device (default: CPU).

    Returns:
        TemporalLSTM in eval mode on the specified device.
    """
    if device is None:
        device = torch.device("cpu")

    model = TemporalLSTM()
    state_dict = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
