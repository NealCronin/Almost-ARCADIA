"""
sam3_server_helper.py

Wraps the SAM3 (Segment Anything Model 3) inference pipeline.

Responsibilities
----------------
* Load a ``sam3.pt`` checkpoint into a model instance.
* Run segmentation inference on a single frame given points or boxes.
* Provide a ``_Sam3MockModel`` fallback when torch / the real model
  is unavailable.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DEFAULT_CHECKPOINT_PATH = ""


# ------------------------------------------------------------------
# Exception
# ------------------------------------------------------------------
class Sam3Error(Exception):
    """Raised when SAM3 operations fail."""


# ------------------------------------------------------------------
# Mock fallback
# ------------------------------------------------------------------
class _Sam3MockModel:
    """
    Lightweight fallback that returns empty results when the real
    SAM3 model cannot be loaded (e.g. no GPU / missing deps).
    """

    def __init__(self) -> None:
        self.device = "cpu"
        logger.info("Using _Sam3MockModel — real SAM3 not available")

    def predict(
        self,
        frame: Any,
        input_points: Optional[list] = None,
        input_boxes: Optional[Any] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        h, w = self._frame_shape(frame)
        n = len(input_boxes) if input_boxes is not None else len(input_points or [])
        if n == 0:
            n = 1
        return {
            "boxes": [[0, 0, w, h]] * n,
            "masks": [self._empty_mask(h, w)] * n,
            "scores": [0.0] * n,
        }

    # -- helpers --------------------------------------------------

    @staticmethod
    def _frame_shape(frame: Any) -> tuple[int, int]:
        if hasattr(frame, "shape"):
            return frame.shape[:2]
        return (480, 640)

    @staticmethod
    def _empty_mask(h: int, w: int) -> list[list[int]]:
        return [[0] * w for _ in range(h)]


# ------------------------------------------------------------------
# Main helper
# ------------------------------------------------------------------
class Sam3ServerHelper:
    """
    Wrapper around the SAM3 segmentation model.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``sam3.pt`` checkpoint file.
    device : str
        Torch device string (``"cuda"`` or ``"cpu"``).
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load_model(self) -> bool:
        """
        Load the SAM3 checkpoint.

        Returns
        -------
        bool
            ``True`` if loaded successfully.
        """
        if self._model is not None:
            logger.info("Model already loaded")
            return True

        try:
            self._model = self._load_real_model()
            logger.info(
                "SAM3 model loaded from %s on %s",
                self.checkpoint_path,
                self.device,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load real SAM3 model: %s — falling back", exc)
            self._model = _Sam3MockModel()
            return True

    # -- internal --------------------------------------------------

    def _load_real_model(self) -> Any:
        """Attempt to load the real SAM3 model via torch."""
        import torch

        # Import the SAM3 model class — adjust import path as your project requires.
        try:
            from sam3.models import build_sam3  # type: ignore[import-not-found,import-untyped]
        except ImportError:
            raise Sam3Error("sam3 package not installed")

        model = build_sam3(self.checkpoint_path, device=self.device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def evaluate(
        self,
        frame: Any,
        points: Optional[list] = None,
        boxes: Optional[Any] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run segmentation on *frame*.

        Parameters
        ----------
        frame : numpy.ndarray or similar
            Input image (H, W, C) or (H, W).
        points : list of [x, y], optional
            Click points for promptable segmentation.
        boxes : list of [x1, y1, x2, y2], optional
            Bounding-box prompts.
        **kwargs
            Extra model-specific flags (e.g. ``multiscale``, ``npoints_per_point``).

        Returns
        -------
        dict
            ``{"boxes": [...], "masks": [[...]], "scores": [...]}``
        """
        if self._model is None:
            raise Sam3Error("Model not loaded. Call load_model() first.")

        result = self._model.predict(
            frame,
            input_points=points,
            input_boxes=boxes,
            **kwargs,
        )

        # Normalise keys so callers always get the same shape
        return {
            "boxes": result.get("boxes", []),
            "masks": result.get("masks", []),
            "scores": result.get("scores", []),
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def status(self) -> dict:
        info: dict = {
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "loaded": self._model is not None,
            "is_mock": isinstance(self._model, _Sam3MockModel),
        }
        return info
