"""
sam3_server_helper.py

Wraps the SAM3 (Segment Anything Model 3) inference pipeline using Ultralytics.

Responsibilities
----------------
* Load a ``sam3.pt`` checkpoint into a model instance (singleton pattern).
* Run segmentation inference on a single frame given points or boxes.
* Maintain persistent model state in memory (warm model, no reload per request).
* Provide a ``_Sam3MockModel`` fallback when ultralytics / the real model
  is unavailable.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DEFAULT_CHECKPOINT_PATH = ""
# Class-level singleton state for persistent model instance
_singleton_instance: Optional["Sam3ServerHelper"] = None
_singleton_model: Optional[Any] = None
_singleton_weights_path: str = ""
_singleton_lock = threading.Lock()


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

    Used when ultralytics or the real model is unavailable.
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
    Helper class for SAM3 model inference using Ultralytics.

    Implements a singleton pattern to maintain persistent model state
    in memory, avoiding reload overhead on every request.

    Attributes
    ----------
    checkpoint_path : str
        Path to the SAM3 checkpoint file.
    device : str
        Device to run inference on (e.g., "cpu", "cuda").
    """
    def __new__(cls, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH, device: str = "cpu"):
        """
        Singleton pattern: return the same instance if already created.
        This ensures the model stays warm in memory across requests.
        """
        global _singleton_instance
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = super().__new__(cls)
        return _singleton_instance

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the SAM3 helper.

        Note: Due to singleton pattern, __init__ may be called multiple times
        but only the first call initializes the model.
        """
        global _singleton_weights_path

        # Only initialize once (thread-safe check)
        if hasattr(self, "_initialized") and self._initialized:
            return

        with _singleton_lock:
            # Double-check after acquiring lock
            if hasattr(self, "_initialized") and self._initialized:
                return

            self.checkpoint_path = checkpoint_path or _singleton_weights_path
            self.device = device
            self._model: Optional[Any] = None
            self._initialized = False

            # Store weights path for singleton reuse
            if checkpoint_path:
                _singleton_weights_path = checkpoint_path

    @classmethod
    def get_model(cls) -> Optional[Any]:
        """
        Get the persistent model instance.

        Returns
        -------
        Any | None
            The loaded Ultralytics SAM model, or None if not initialized.
        """
        return _singleton_model

    @classmethod
    def reset_singleton(cls) -> None:
        """
        Reset the singleton state.

        Use this to force a model reload (e.g., if weights path changes).
        WARNING: This will unload the current model from memory.
        """
        global _singleton_instance, _singleton_model, _singleton_weights_path
        with _singleton_lock:
            _singleton_instance = None
            _singleton_model = None
            _singleton_weights_path = ""
        logger.info("SAM3 singleton state reset")

    def initialize(self) -> bool:
        """
        Load the SAM3 model from the checkpoint (only once).

        Returns True if initialization succeeded.
        """
        global _singleton_model

        if self._initialized:
            return True

        # Check if model already loaded by another instance
        if _singleton_model is not None:
            self._model = _singleton_model
            self._initialized = True
            logger.info("Using existing persistent SAM3 model")
            return True

        if not self.checkpoint_path:
            logger.warning("No checkpoint path provided, using mock model")
            self._model = _Sam3MockModel()
            self._initialized = True
            return True

        try:
            from ultralytics import SAM

            logger.info("Loading SAM3 model from %s on %s", self.checkpoint_path, self.device)
            model = SAM(self.checkpoint_path)
            model.to(self.device)

            self._model = model
            _singleton_model = model
            self._initialized = True
            logger.info("SAM3 model loaded successfully and cached in memory")
            return True

        except ImportError as exc:
            logger.warning(
                "Failed to import ultralytics: %s. Using mock model.",
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
        Run segmentation prediction on an image using Ultralytics SAM.

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
            - point_coords: list of point coordinates used
        """
        if not self._initialized:
            self.initialize()

        if self._model is None:
            raise Sam3Error("Model not initialized")

        try:
            # Ultralytics SAM inference API
            # Prepare kwargs for the predict method
            predict_kwargs = {}

            if input_points:
                # Ultralytics expects points as list of [x, y] and labels as list
                predict_kwargs["points"] = input_points
            if input_labels:
                predict_kwargs["labels"] = input_labels
            if input_boxes:
                predict_kwargs["boxes"] = input_boxes

            # Run prediction with the persistent model
            results = self._model(image, **predict_kwargs)

            # Extract results from Ultralytics Results object
            # results is a list of Results objects; we use the first one
            if len(results) == 0:
                logger.warning("No results returned from model")
                return {
                    "masks": [],
                    "scores": [],
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                    "point_coords": input_points or [],
                }

            result = results[0]

            # Extract masks
            masks = []
            if hasattr(result, "masks") and result.masks is not None:
                # result.masks.data is a tensor of shape (n, H, W)
                if hasattr(result.masks, "data") and len(result.masks.data) > 0:
                    masks_tensor = result.masks.data
                    # Convert to list of 2D arrays
                    for i in range(len(masks_tensor)):
                        mask = masks_tensor[i].cpu().numpy().tolist()
                        masks.append(mask)

            # Extract scores (confidence)
            scores = []
            if hasattr(result, "boxes") and result.boxes is not None:
                if hasattr(result.boxes, "conf"):
                    scores = result.boxes.conf.cpu().numpy().tolist()

            # Extract bounding box if available
            bbox = [0.0, 0.0, 0.0, 0.0]
            if hasattr(result, "boxes") and result.boxes is not None:
                if hasattr(result.boxes, "xyxy") and len(result.boxes.xyxy) > 0:
                    bbox = result.boxes.xyxy[0].cpu().numpy().tolist()

            return {
                "masks": masks,
                "scores": scores,
                "bbox": bbox,
                "point_coords": input_points or [],
            }

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
            List of (x, y) centroid coordinates. Returns empty list
            if no valid targets are detected (prevents divide-by-zero).
        """
        result = self.predict(image, input_points=input_points)
        masks = result.get("masks", [])
        centroids: list[tuple[float, float]] = []

        # Import numpy once at function level for efficiency
        import numpy as np

        for mask in masks:
            # Validate mask structure
            if not isinstance(mask, list) or len(mask) == 0:
                logger.debug("Skipping empty or invalid mask")
                continue

            # Convert to numpy for easier processing
            try:
                mask_arr = np.array(mask)

                # CRITICAL: Check if mask is entirely zeros (no targets detected)
                # This prevents divide-by-zero errors when computing centroids
                if mask_arr.size == 0:
                    logger.debug("Mask array is empty, skipping")
                    continue

                # Find all non-zero pixel coordinates
                ys, xs = np.where(mask_arr > 0)

                # Validate that we found at least one valid target pixel
                if len(xs) == 0 or len(ys) == 0:
                    logger.debug("No valid target pixels found in mask")
                    continue

                # Compute centroid safely (now guaranteed to have non-zero elements)
                centroid_x = float(np.mean(xs))
                centroid_y = float(np.mean(ys))
                centroids.append((centroid_x, centroid_y))

            except (ValueError, TypeError) as exc:
                # Handle malformed mask data gracefully
                logger.warning("Failed to process mask: %s", exc)
                continue
            except Exception as exc:
                # Catch any other unexpected errors to prevent pipeline crash
                logger.error("Unexpected error processing mask: %s", exc)
                continue

        return centroids

    def status(self) -> dict:
        """Return the current model status."""
        return {
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "initialized": self._initialized,
            "model_type": type(self._model).__name__ if self._model else None,
            "singleton_active": _singleton_model is not None,
        }
