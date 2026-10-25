"""Vision Transformer (ViT) feature extraction service.

Uses google/vit-base-patch16-224-in21k to extract 768-dimensional normalized
semantic visual embeddings from input images.

Implements a safe initialization fallback that allows the Flask application,
web UI, and non-model features to run cleanly even when Windows Application
Control or system policies restrict PyTorch native DLLs (e.g., WinError 4551 on shm.dll).
"""

import logging
from typing import Optional, Any
import numpy as np
from PIL import Image

from config import MODEL_NAME, DEVICE, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class VisionTransformerService:
    """Manages loading and inference for the Vision Transformer model with safe error handling."""

    _instance: Optional["VisionTransformerService"] = None

    def __init__(self):
        self.device = DEVICE
        self.model_name = MODEL_NAME
        self.processor: Optional[Any] = None
        self.model: Optional[Any] = None
        self._is_ready = False
        self._error_reason: Optional[str] = None
        self._torch_module: Optional[Any] = None
        # Attempt safe initialization at startup without raising unhandled exceptions
        self._try_initialize()

    @classmethod
    def get_instance(cls) -> "VisionTransformerService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _try_initialize(self):
        """Attempts to import PyTorch and load the Vision Transformer model safely."""
        try:
            # Lazy import to avoid crashing at application startup if DLLs are blocked
            import torch
            from transformers import ViTImageProcessor, ViTModel

            self._torch_module = torch
            actual_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = actual_device

            logger.info("Initializing Vision Transformer (%s) on %s...", self.model_name, self.device)
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self._is_ready = True
            self._error_reason = None
            logger.info("Vision Transformer successfully loaded.")

        except OSError as exc:
            # Captures WinError 4551: "An Application Control policy has blocked this file (torch/lib/shm.dll)"
            self._is_ready = False
            self.processor = None
            self.model = None
            self._error_reason = f"PyTorch native library blocked by Windows Application Control (WinError 4551): {exc}"
            logger.warning("Vision Transformer disabled: %s", self._error_reason)

        except ImportError as exc:
            self._is_ready = False
            self.processor = None
            self.model = None
            self._error_reason = f"Required machine learning package missing: {exc}"
            logger.warning("Vision Transformer disabled: %s", self._error_reason)

        except Exception as exc:
            self._is_ready = False
            self.processor = None
            self.model = None
            self._error_reason = f"Failed to initialize model: {exc}"
            logger.warning("Vision Transformer disabled: %s", self._error_reason)

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def error_reason(self) -> Optional[str]:
        return self._error_reason

    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract a 768-D L2-normalized embedding for a single PIL image.

        Args:
            image: PIL Image object.

        Returns:
            1D numpy array of shape (768,) with dtype float32, L2-normalized.

        Raises:
            RuntimeError: If Vision Transformer cannot be loaded.
        """
        if not self._is_ready:
            # Retry initialization in case environment has been adjusted
            self._try_initialize()

        if not self._is_ready or self.processor is None or self.model is None or self._torch_module is None:
            raise RuntimeError(
                f"Vision Transformer is unavailable: {self._error_reason or 'Model failed to load'}. "
                "Windows Application Control has blocked PyTorch native binaries. "
                "Non-model features (browsing, DNA inspection, pipeline documentation) remain fully operational."
            )

        torch = self._torch_module

        # Ensure image is in RGB format (handles PNG alpha, grayscale, palettes)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # ViT processor resizes to 224x224 and normalizes with ImageNet statistics
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract the representation of the [CLS] classification token (index 0)
        cls_token = outputs.last_hidden_state[:, 0, :]
        normalized_vector = torch.nn.functional.normalize(cls_token, p=2, dim=1)

        embedding = normalized_vector.cpu().numpy()[0].astype(np.float32)
        assert embedding.shape == (EMBEDDING_DIM,), f"Expected shape ({EMBEDDING_DIM},), got {embedding.shape}"
        return embedding


# Global singleton helper
def get_vit_service() -> VisionTransformerService:
    return VisionTransformerService.get_instance()
