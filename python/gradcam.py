"""
Grad-CAM Module — Gradient-weighted Class Activation Mapping.

Generates visual heatmaps showing which regions of the face the
EfficientNet classifier considers most suspicious. Hooks into the
last convolutional layer to capture activations and gradients.

Note: Requires PyTorch model (not ONNX) for gradient computation.
This module is used only for the analysis view, not the main
inference pipeline, to maintain FPS targets.
"""

import os
import logging
from typing import Optional

import cv2
import numpy as np

from models import load_efficientnet_deepfake, IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger("silentwitness.gradcam")

HEATMAP_ALPHA = 0.4  # Blend opacity


class GradCAM:
    """Grad-CAM visualization for the EfficientNet-B0 classifier."""

    def __init__(self) -> None:
        self._model = None
        self._target_layer = None
        self._activations: Optional[np.ndarray] = None
        self._gradients: Optional[np.ndarray] = None
        self._hooks: list = []

    def load_model(self, model_path: str) -> bool:
        """Load a PyTorch EfficientNet model for Grad-CAM computation.

        Uses the shared EfficientNetDeepfake from models.py to ensure
        the architecture (including Dropout layer) matches the training
        checkpoint exactly.

        Args:
            model_path: Path to the PyTorch .pt model file.

        Returns:
            True if model was loaded successfully.
        """
        try:
            if not os.path.exists(model_path):
                logger.warning(f"PyTorch model not found at {model_path}")
                return False

            model = load_efficientnet_deepfake(model_path)
            self._model = model
            # Hook into the last convolutional layer: features[-1]
            self._target_layer = model.features[-1]
            self._register_hooks()

            logger.info("Grad-CAM model loaded")
            return True

        except Exception as e:
            logger.error(f"Failed to load Grad-CAM model: {e}")
            return False

    def generate(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """Generate a Grad-CAM heatmap for the given face crop.

        Args:
            face_crop: Normalized 224x224 RGB face image (float32, ImageNet-normalized).

        Returns:
            BGR heatmap overlay image (224x224) or None if model is not loaded.
        """
        if self._model is None:
            return None

        try:
            import torch

            # Prepare input tensor: (1, 3, 224, 224)
            input_tensor = torch.from_numpy(
                np.transpose(face_crop, (2, 0, 1))[np.newaxis]
            ).float()
            input_tensor.requires_grad_(True)

            # Forward pass — EfficientNetDeepfake returns (logit, embedding)
            logit, _ = self._model(input_tensor)

            # Backward pass with respect to the "fake" class
            self._model.zero_grad()
            logit.backward()

            if self._activations is None or self._gradients is None:
                return None

            # Compute Grad-CAM weights: global average pooling of gradients
            weights = np.mean(self._gradients, axis=(2, 3), keepdims=True)

            # Weighted sum of activations
            cam = np.sum(weights * self._activations, axis=1, keepdims=False)[0]

            # Apply ReLU (only positive contributions)
            cam = np.maximum(cam, 0)

            # Normalize to [0, 1]
            cam_max = cam.max()
            if cam_max > 0:
                cam = cam / cam_max

            # Resize to face crop dimensions
            cam_resized = cv2.resize(cam, (224, 224))

            # Apply colormap (JET: blue -> red)
            heatmap = cv2.applyColorMap(
                np.uint8(255 * cam_resized), cv2.COLORMAP_JET
            )

            # Blend with original face crop
            mean = np.array(IMAGENET_MEAN)
            std = np.array(IMAGENET_STD)
            face_display = ((face_crop * std + mean) * 255).clip(0, 255).astype(np.uint8)
            face_bgr = cv2.cvtColor(face_display, cv2.COLOR_RGB2BGR)

            overlay = cv2.addWeighted(face_bgr, 1 - HEATMAP_ALPHA, heatmap, HEATMAP_ALPHA, 0)
            return overlay

        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}")
            return None

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        if self._target_layer is None:
            return

        def forward_hook(module, input, output):
            self._activations = output.detach().numpy()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach().numpy()

        self._hooks.append(self._target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self._target_layer.register_full_backward_hook(backward_hook))

    def cleanup(self) -> None:
        """Remove hooks and release model."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._model = None
