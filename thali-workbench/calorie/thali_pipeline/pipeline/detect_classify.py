"""YOLO detect -> crop -> classify. Shared by Task 2 (direct) and Task 3
(after the BEV warp) — both just call this on whatever image they hand it.
"""

from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image

from thali_pipeline.models.classifiers import classify_crop
from thali_pipeline.models.yolo_detector import YoloDishDetector


@dataclass
class ClassifierBundle:
    """Holds whichever classifier profile(s) are loaded, keyed by name
    ("dino", "convnext", or any other profile added to
    configs/weights.yaml). `classify` below picks one or averages two,
    based on `mode`.
    """
    models: dict = field(default_factory=dict)     # name -> nn.Module
    transforms: dict = field(default_factory=dict)  # name -> torchvision transform
    labels: List[str] = field(default_factory=list)
    device: torch.device = torch.device("cpu")

    def classify(self, crop_rgb: Image.Image, mode: str = "dino"):
        """Returns (label, confidence, detail) where detail is a dict of
        per-classifier {label, confidence} when mode == "ensemble".
        """
        if mode == "ensemble":
            if "dino" not in self.models or "convnext" not in self.models:
                raise ValueError(
                    "mode='ensemble' requires both 'dino' and 'convnext' "
                    "classifiers to be loaded."
                )
            _, _, probs_dino = classify_crop(
                crop_rgb, self.models["dino"], self.transforms["dino"], self.labels, self.device
            )
            _, _, probs_conv = classify_crop(
                crop_rgb, self.models["convnext"], self.transforms["convnext"], self.labels, self.device
            )
            avg_probs = (probs_dino + probs_conv) / 2.0
            conf, idx = torch.max(avg_probs, dim=0)
            label = self.labels[idx.item()]
            detail = {
                "dino": {"label": self.labels[int(probs_dino.argmax())], "confidence": float(probs_dino.max())},
                "convnext": {"label": self.labels[int(probs_conv.argmax())], "confidence": float(probs_conv.max())},
            }
            return label, float(conf.item()), detail

        if mode not in self.models:
            raise ValueError(f"classifier mode '{mode}' is not loaded (loaded: {list(self.models.keys())})")

        label, conf, _ = classify_crop(
            crop_rgb, self.models[mode], self.transforms[mode], self.labels, self.device
        )
        return label, conf, {mode: {"label": label, "confidence": conf}}


def detect_and_classify(
    img_bgr: np.ndarray,
    detector: YoloDishDetector,
    classifier_bundle: ClassifierBundle,
    classifier_mode: str = "dino",
    min_box_px: int = 20,
    crop_padding_frac: float = 0.15,
) -> List[dict]:
    """Runs YOLO on img_bgr, classifies each valid crop.

    `crop_padding_frac`: expands each detected box by this fraction of its
    own width/height (per side) before cropping for classification/calories,
    to give the crop some visible plate/context instead of food filling the
    entire frame edge-to-edge. The tight (unpadded) box is still what's
    reported/annotated -- only the crop fed downstream is padded.

    Returns a list of dicts:
        {"bbox": [x1,y1,x2,y2], "det_conf": float,
         "class": str, "class_confidence": float, "classifier_detail": dict,
         "crop_rgb": PIL.Image}   # crop kept in-memory for the calorie step
    """
    detections = detector.detect(img_bgr)
    if not detections:
        return []

    h, w = img_bgr.shape[:2]
    results = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        x1_i, y1_i = max(0, int(x1)), max(0, int(y1))
        x2_i, y2_i = min(w, int(x2)), min(h, int(y2))

        if (x2_i - x1_i) < min_box_px or (y2_i - y1_i) < min_box_px:
            continue

        pad_x = int((x2_i - x1_i) * crop_padding_frac)
        pad_y = int((y2_i - y1_i) * crop_padding_frac)
        px1, py1 = max(0, x1_i - pad_x), max(0, y1_i - pad_y)
        px2, py2 = min(w, x2_i + pad_x), min(h, y2_i + pad_y)

        crop_bgr = img_bgr[py1:py2, px1:px2]
        if crop_bgr.size == 0:
            continue

        crop_rgb_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        label, conf, detail = classifier_bundle.classify(crop_rgb_pil, mode=classifier_mode)

        results.append({
            "bbox": [x1_i, y1_i, x2_i, y2_i],
            "det_conf": det["det_conf"],
            "class": label,
            "class_confidence": conf,
            "classifier_detail": detail,
            "crop_rgb": crop_rgb_pil,
        })

    return results
