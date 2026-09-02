"""Attaches the existing dish-recognition models (YOLO detector + ConvNeXt
classifier, from the Food-Image-Modelling repo) onto the portion-estimation
backbone.

Two independent pieces, used for two different situations:

  * `ConvNeXtDishClassifier` (a `FoodClassifier`) — takes the FoodInstance
    masks the backbone already found (SAM/compartment-floor based for the
    Ashoka tray) and labels each one with a real dish name instead of just
    "solid"/"liquid". This is the normal path for Ashoka-tray photos.

  * `YoloBoxFoodSegmenter` (a `FoodSegmenter`) — for photos with no known
    compartment layout (general thali images, "Task 3" style with no fixed
    pattern/angle): runs YOLO to find dish regions directly, since there's
    no tray profile to align floor polygons against. Feed its output through
    `ConvNeXtDishClassifier` the same way as the Ashoka path.

Both degrade gracefully (log + return input unchanged / empty) if their
weight files aren't present yet, so a repo checkout without weights still
imports and the rest of the pipeline still runs.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..interfaces import FoodClassifier, FoodSegmenter
from ..types import Calibration, FoodInstance

LOGGER = logging.getLogger(__name__)


def _resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class ConvNeXtDishClassifier(FoodClassifier):
    def __init__(self, weights_path: str, labels_path: str, device: str = "auto",
                 min_crop_px: int = 8) -> None:
        self.weights_path = weights_path
        self.labels_path = labels_path
        self._device_str = device
        self.min_crop_px = min_crop_px
        self._model = None
        self._labels: list[str] | None = None
        self._transform = None
        self._device = None
        self._unavailable_reason: str | None = None

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._unavailable_reason is not None:
            return False
        if not Path(self.weights_path).exists() or not Path(self.labels_path).exists():
            self._unavailable_reason = (
                f"classifier weights/labels not found ({self.weights_path}, {self.labels_path})"
            )
            LOGGER.warning("ConvNeXtDishClassifier: %s — dish names will stay solid/liquid.", self._unavailable_reason)
            return False
        try:
            import timm
            import torch
            import torch.nn as nn
            import torchvision.transforms as T
        except ImportError as exc:
            self._unavailable_reason = f"timm/torch/torchvision not installed ({exc})"
            LOGGER.warning("ConvNeXtDishClassifier: %s", self._unavailable_reason)
            return False

        with open(self.labels_path, "r", encoding="utf-8") as handle:
            self._labels = [line.strip() for line in handle if line.strip()]

        class _Head(nn.Module):
            def __init__(self, backbone, feature_dim, num_classes, drop_rate=0.2):
                super().__init__()
                self.backbone = backbone
                self.head = nn.Sequential(
                    nn.LayerNorm(feature_dim), nn.Dropout(p=drop_rate), nn.Linear(feature_dim, num_classes),
                )

            def forward(self, x):
                return self.head(self.backbone(x))

        self._device = _resolve_device(self._device_str)
        backbone = timm.create_model("convnext_base", pretrained=False, num_classes=0,
                                      global_pool="avg", drop_path_rate=0.2)
        model = _Head(backbone, backbone.num_features, len(self._labels))
        checkpoint = torch.load(self.weights_path, map_location=self._device, weights_only=False)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)
        model.to(self._device).eval()
        self._model = model

        resolution = 224
        resize_to = int(resolution * 512 / 448)
        self._transform = T.Compose([
            T.Resize(resize_to, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return True

    def _predict_crop(self, crop_bgr: np.ndarray) -> tuple[str, float] | None:
        import torch
        from PIL import Image

        if crop_bgr.size == 0:
            return None
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs).item())
        return self._labels[idx], float(probs[idx].item())

    def classify(self, image: np.ndarray, foods: list[FoodInstance]) -> list[FoodInstance]:
        if not foods:
            return foods
        if not self._ensure_loaded():
            return foods

        results: list[FoodInstance] = []
        for food in foods:
            ys, xs = np.where(food.mask)
            if len(xs) < self.min_crop_px:
                results.append(food)
                continue
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            crop = image[y1:y2, x1:x2]
            prediction = self._predict_crop(crop)
            if prediction is None:
                results.append(food)
                continue
            label, confidence = prediction
            results.append(FoodInstance(
                mask=food.mask, score=food.score, label=label, class_confidence=confidence,
                physical_type=food.physical_type, compartment_coverage=food.compartment_coverage,
            ))
        return results


class YoloBoxFoodSegmenter(FoodSegmenter):
    """YOLO-box-based food segmenter for general thali photos with no known
    compartment layout (no tray profile to align floor polygons against).

    Produces rectangular FoodInstance masks straight from YOLO detections,
    intersected with the plate mask so nothing outside the plate counts.
    Solid/liquid is decided the same way as `ColorComponentFoodSegmenter`
    (saturation + colour uniformity within the box), since there's no known
    floor polygon to compute compartment coverage against here.
    """

    def __init__(self, weights_path: str, conf_threshold: float = 0.10, iou_threshold: float = 0.30,
                 device: str = "auto", min_area_px: int = 500, saturation_threshold: int = 55,
                 liquid_gray_std_threshold: float = 42.0) -> None:
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._device_str = device
        self.min_area_px = min_area_px
        self.saturation_threshold = saturation_threshold
        self.liquid_gray_std_threshold = liquid_gray_std_threshold
        self._model = None
        self._unavailable_reason: str | None = None

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._unavailable_reason is not None:
            return False
        if not Path(self.weights_path).exists():
            self._unavailable_reason = f"YOLO weights not found: {self.weights_path}"
            LOGGER.warning("YoloBoxFoodSegmenter: %s", self._unavailable_reason)
            return False
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self._unavailable_reason = f"ultralytics not installed ({exc})"
            LOGGER.warning("YoloBoxFoodSegmenter: %s", self._unavailable_reason)
            return False
        self._model = YOLO(self.weights_path)
        return True

    def segment(self, image: np.ndarray, plate_mask: np.ndarray,
                calibration: Calibration | None = None) -> list[FoodInstance]:
        if not self._ensure_loaded():
            return []
        h, w = image.shape[:2]
        resized = cv2.resize(image, (500, 500))
        results = self._model.predict(source=resized, save=False, conf=self.conf_threshold,
                                       iou=self.iou_threshold, agnostic_nms=True, verbose=False)
        scale_x, scale_y = w / 500.0, h / 500.0
        foods: list[FoodInstance] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1o, y1o = max(0, int(x1 * scale_x)), max(0, int(y1 * scale_y))
            x2o, y2o = min(w, int(x2 * scale_x)), min(h, int(y2 * scale_y))
            box_mask = np.zeros((h, w), dtype=bool)
            box_mask[y1o:y2o, x1o:x2o] = True
            mask = box_mask & plate_mask
            if int(mask.sum()) < self.min_area_px:
                continue
            crop_hsv = cv2.cvtColor(image[y1o:y2o, x1o:x2o], cv2.COLOR_BGR2HSV)
            crop_gray = cv2.cvtColor(image[y1o:y2o, x1o:x2o], cv2.COLOR_BGR2GRAY)
            is_liquid = (
                float(np.std(crop_gray)) <= self.liquid_gray_std_threshold
                and float(np.mean(crop_hsv[..., 1])) < self.saturation_threshold
            )
            foods.append(FoodInstance(
                mask=mask, score=float(box.conf[0].item()) if hasattr(box, "conf") else 1.0,
                label="liquid" if is_liquid else "solid", class_confidence=0.35,
                physical_type="liquid" if is_liquid else "solid", compartment_coverage=1.0,
            ))
        return foods
