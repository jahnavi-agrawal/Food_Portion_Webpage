"""The one dish classifier, loaded once and shared by every tab.

The Dish tab and the portion pipeline both want the same ConvNeXt checkpoint,
so it lives here behind `st.cache_resource` rather than being loaded twice.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Matches `classifiers.convnext` in the calorie repo's weights.yaml, which is
# the resolution/normalization this checkpoint was trained at. (DishDetect.py
# used Resize((200, 200)) with no normalization — see README.)
RESIZE = 438
CROP_SIZE = 384


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


@st.cache_resource(show_spinner=False)
def load_labels(labels_path: str) -> list[str]:
    lines = Path(labels_path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


class _ConvNextFoodClassifier(nn.Module):
    """Same architecture as both repos' ConvNeXt wrapper (backbone + head)."""

    def __init__(self, num_classes: int, drop_rate: float = 0.2, drop_path_rate: float = 0.2):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "convnext_base", pretrained=False, num_classes=0,
            global_pool="avg", drop_path_rate=drop_path_rate,
        )
        feature_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(p=drop_rate),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


@st.cache_resource(show_spinner="Loading dish classifier...")
def load_dish_classifier(weights_path: str, labels_path: str, device_preference: str = "auto"):
    """Returns (model, transform, labels, device).

    Handles both checkpoint shapes: a pickled nn.Module (what DishDetect.py's
    torch.load + .eval() expects) and a plain state_dict. The two repos assume
    different ones, so accept either rather than guessing.
    """
    labels = load_labels(labels_path)
    device = resolve_device(device_preference)

    obj = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        model = obj
    else:
        state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
        model = _ConvNextFoodClassifier(len(labels))
        result = model.load_state_dict(state, strict=False)
        missing, unexpected = getattr(result, "missing_keys", []), getattr(result, "unexpected_keys", [])
        if missing or unexpected:
            st.warning(
                f"Classifier checkpoint didn't match cleanly — {len(missing)} missing / "
                f"{len(unexpected)} unexpected keys. Predictions may be wrong."
            )

    model.to(device).eval()

    transform = T.Compose([
        T.Resize(RESIZE, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(CROP_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return model, transform, labels, device


@torch.no_grad()
def classify_pil(image: Image.Image, model, transform, labels, device, top_k: int = 5):
    """Returns [(label, confidence), ...] sorted best first."""
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1)[0]
    k = min(top_k, len(labels))
    confidences, indices = torch.topk(probs, k)
    return [(labels[int(i)], float(c)) for c, i in zip(confidences, indices)]


class DishNameClassifier:
    """Adapter matching the portion pipeline's FoodClassifier protocol.

    Replaces each FoodInstance's "solid"/"liquid" label with a real dish name,
    using the same cached model as the Dish tab. `physical_type` is preserved,
    since the volume estimator still needs it to pick the liquid fill prior.
    """

    def __init__(self, model, transform, labels, device, min_crop_px: int = 8):
        self.model, self.transform, self.labels, self.device = model, transform, labels, device
        self.min_crop_px = min_crop_px

    def classify(self, image: np.ndarray, foods: list) -> list:
        if not foods:
            return foods
        from portion_estimation.types import FoodInstance

        labelled = []
        for food in foods:
            ys, xs = np.where(food.mask)
            if len(xs) < self.min_crop_px:
                labelled.append(food)
                continue
            crop = image[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1]
            if crop.size == 0:
                labelled.append(food)
                continue
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            label, confidence = classify_pil(
                pil_crop, self.model, self.transform, self.labels, self.device, top_k=1
            )[0]
            labelled.append(FoodInstance(
                mask=food.mask,
                score=food.score,
                label=label,
                class_confidence=confidence,
                physical_type=food.physical_type,
                compartment_coverage=food.compartment_coverage,
            ))
        return labelled
