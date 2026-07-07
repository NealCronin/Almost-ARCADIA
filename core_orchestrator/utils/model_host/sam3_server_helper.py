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
    Mock SAM3 model that returns empty masks.

    Used when torch or the real model is unavailable.
    """

    def __init__(self) -> None:
        self.device = "cpu"
        logger.info("Using _Sam3MockModel — real SAM3 not available")

    def predict(
        self,
        frame: Any,
        input_points: Optional[list[list[float]]] = None,
        input_labels: Optional[list[int]] = None,
        input_boxes: Optional[list[list[float]]] = None,
    ) -> dict[str, Any]:
        """Return empty segmentation results."""
        h, w = self._frame_shape(frame)
        return {
            "masks": [self._empty_mask(h, w)],
            "scores": [0.0],
            "bbox": [0.0, 0.0, 0.0, 0.0],
            "point_coords": input_points or [],
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
    Helper class for SAM3 model inference.

    Attributes
    ----------
    checkpoint_path : str
        Path to the SAM3 checkpoint file.
    device : str
        Device to run inference on (e.g., "cpu", "cuda").
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model: Optional[Any] = None
        self._initialized = False

    def initialize(self) -> bool:
        """
        Load the SAM3 model from the checkpoint.

        Returns True if initialization succeeded.
        """
        if self._initialized:
            return True

        if not self.checkpoint_path:
            logger.warning("No checkpoint path provided, using mock model")
            self._model = _Sam3MockModel()
            self._initialized = True
            return True

        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry

            # Determine model type from checkpoint or use default
            model_type = "vit_h"  # Default to ViT-H

            logger.info("Loading SAM3 model from %s", self.checkpoint_path)
            sam = sam_model_registry[model_type](checkpoint=self.checkpoint_path)
            sam.to(device=self.device)
            sam.eval()

            self._model = SamPredictor(sam)
            self._initialized = True
            logger.info("SAM3 model loaded successfully")
            return True

        except ImportError as exc:
            logger.warning(
                "Failed to import segment_anything or torch: %s. Using mock model.",
                exc,
            )
            self._model = _Sam3MockModel()
            self._initialized = True
            return True
        except Exception as exc:
            logger.exception("Failed to load SAM3 model: %s", exc)
            self._model = _Sam3MockModel()
            self._initialized = True
            return False

    def predict(
        self,
        image: Any,
        input_points: Optional[list[list[float]]] = None,
        input_labels: Optional[list[int]] = None,
        input_boxes: Optional[list[list[float]]] = None,
    ) -> dict[str, Any]:
        """
        Run segmentation prediction on an image.

        Parameters
        ----------
        image : Any
            Input image as a numpy array (H, W, 3) in RGB format.
        input_points : list[list[float]] | None
            Point prompts as [[x1, y1], [x2, y2], ...]
        input_labels : list[int] | None
            Point labels (1 for foreground, 0 for background).
        input_boxes : list[list[float]] | None
            Box prompts as [[x1, y1, x2, y2], ...]

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - masks: list of binary masks
            - scores: list of confidence scores
            - bbox: bounding box [x1, y1, x2, y2]
        """
        if not self._initialized:
            self.initialize()

        if self._model is None:
            raise Sam3Error("Model not initialized")

        try:
            # Set image if using SamPredictor
            if hasattr(self._model, "set_image"):
                self._model.set_image(image)

            # Run prediction
            if hasattr(self._model, "predict"):
                masks, scores, boxes = self._model.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    box=input_boxes,
                    multimask_output=False,
                )

                return {
                    "masks": masks.tolist() if hasattr(masks, "tolist") else masks,
                    "scores": scores.tolist() if hasattr(scores, "tolist") else scores,
                    "bbox": boxes[0].tolist() if len(boxes) > 0 else [0.0, 0.0, 0.0, 0.0],
                }
            else:
                # Mock model path
                return self._model.predict(
                    image, input_points, input_labels, input_boxes
                )

        except Exception as exc:
            logger.exception("Prediction failed: %s", exc)
            raise Sam3Error(f"Prediction failed: {exc}") from exc

    def predict_from_points(
        self,
        image: Any,
        points: list[list[float]],
        labels: list[int],
    ) -> dict[str, Any]:
        """
        Convenience method for point-based prediction.

        Parameters
        ----------
        image : Any
            Input image as numpy array.
        points : list[list[float]]
            Point coordinates [[x1, y1], [x2, y2], ...]
        labels : list[int]
            Point labels [1, 0, ...] (1=foreground, 0=background)

        Returns
        -------
        dict[str, Any]
            Segmentation results.
        """
        return self.predict(image, input_points=points, input_labels=labels)

    def predict_from_box(
        self,
        image: Any,
        box: list[float],
    ) -> dict[str, Any]:
        """
        Convenience method for box-based prediction.

        Parameters
        ----------
        image : Any
            Input image as numpy array.
        box : list[float]
            Bounding box [x1, y1, x2, y2]

        Returns
        -------
        dict[str, Any]
            Segmentation results.
        """
        return self.predict(image, input_boxes=[box])

    def get_target_coordinates(
        self,
        image: Any,
        input_points: Optional[list[list[float]]] = None,
    ) -> list[tuple[float, float]]:
        """
        Extract target pixel coordinates from segmentation.

        Returns the centroid of each detected mask.

        Parameters
        ----------
        image : Any
            Input image as numpy array.
        input_points : list[list[float]] | None
            Initial point prompts for detection.

        Returns
        -------
        list[tuple[float, float]]
            List of (x, y) centroid coordinates.
        """
        result = self.predict(image, input_points=input_points)
        masks = result.get("masks", [])
        centroids = []

        for mask in masks:
            if isinstance(mask, list) and len(mask) > 0:
                # Convert to numpy for easier processing
                import numpy as np

                mask_arr = np.array(mask)
                ys, xs = np.where(mask_arr > 0)
                if len(xs) > 0:
                    centroid_x = float(np.mean(xs))
                    centroid_y = float(np.mean(ys))
                    centroids.append((centroid_x, centroid_y))

        return centroids

    def status(self) -> dict:
        """Return the current model status."""
        return {
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "initialized": self._initialized,
            "model_type": type(self._model).__name__ if self._model else None,
        }
