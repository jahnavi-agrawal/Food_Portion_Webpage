"""Dish classifiers.

Two architectures, matching your original repo exactly (parameter names and
shapes must match a checkpoint's state_dict for `load_state_dict` to work):

  ConvNextFoodClassifier  -- self.backbone (timm convnext_base) + self.head
                             (LayerNorm -> Dropout -> Linear). This is the
                             SAME architecture in both
                             thali_detection/task1_clf_convnext.py (224px)
                             and dish_detection/models/veer_convnext/predict.py
                             (384px) — only the input resolution and dropout
                             rate differ, and dropout is a no-op in eval mode.
                             One class covers both; resolution/dropout come
                             from configs/weights.yaml.

  DinoV2FoodClassifier    -- self.backbone (torch.hub dinov2_vits14) + self.head
                             (LayerNorm -> Linear -> GELU -> Dropout -> Linear).
                             Matches thali_detection/task1_clf_dinov2.py.
                             dish_detection/models/saransh_dino/predict.py wraps
                             the identical layers in a bare nn.Sequential
                             instead, which saves a checkpoint with numeric
                             keys ("0.weight", "1.weight", ...) rather than
                             named ones. `load_classifier_checkpoint` below
                             detects and remaps that automatically.
"""

from typing import List

import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


# ============================================================================
# Architectures
# ============================================================================

class ConvNextFoodClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        timm_name: str = "convnext_base",
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            timm_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=drop_path_rate,
        )
        self.feature_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(p=drop_rate),
            nn.Linear(self.feature_dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class DinoV2FoodClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dino_variant: str = "dinov2_vits14",
        feature_dim: int = 384,
        hidden_dim: int = 512,
        drop_rate: float = 0.3,
    ):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", dino_variant)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ============================================================================
# Transforms
# ============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_convnext_transform(resize: int, crop_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_dino_transform(resize: int = 384, crop_size: int = 336) -> T.Compose:
    return T.Compose([
        T.Resize(resize),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================================================
# Checkpoint loading (handles both key formats seen in your repo)
# ============================================================================

def _unwrap_checkpoint(ckpt):
    """Both training scripts save checkpoints in slightly different shapes."""
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
    return ckpt


def _remap_sequential_dino_keys(state_dict):
    """dish_detection/models/saransh_dino/predict.py saves the model as a bare
    nn.Sequential(backbone, LayerNorm, Linear, GELU, Dropout, Linear), so keys
    look like "0.weight", "1.bias", etc. Map them onto our named
    backbone./head.N. structure. GELU (index 3) and Dropout (index 4) have no
    parameters, so there's nothing to remap for them.
    """
    index_map = {"0": "backbone", "1": "head.0", "2": "head.1", "5": "head.4"}
    remapped = {}
    for k, v in state_dict.items():
        prefix, _, rest = k.partition(".")
        if prefix in index_map:
            remapped[f"{index_map[prefix]}.{rest}"] = v
        else:
            remapped[k] = v
    return remapped


def _looks_like_sequential_keys(state_dict) -> bool:
    return any(k.split(".")[0].isdigit() for k in state_dict.keys())


def load_classifier_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    device: torch.device,
    strict: bool = False,
) -> nn.Module:
    """Load a classifier checkpoint, auto-remapping nn.Sequential-numeric keys
    if needed. Uses strict=False by default and PRINTS any missing/unexpected
    keys instead of raising — with the checkpoint <-> role mapping in
    configs/weights.yaml being a best guess, a loud warning is far more useful
    than a silent wrong-shape crash. Run scripts/check_weights.py to see this
    up front before running real inference.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = _unwrap_checkpoint(ckpt)

    if _looks_like_sequential_keys(state_dict):
        state_dict = _remap_sequential_dino_keys(state_dict)

    result = model.load_state_dict(state_dict, strict=strict)
    missing = getattr(result, "missing_keys", [])
    unexpected = getattr(result, "unexpected_keys", [])
    if missing or unexpected:
        print(f"[classifiers] WARNING loading {ckpt_path}:")
        if missing:
            print(f"  missing_keys: {missing}")
        if unexpected:
            print(f"  unexpected_keys: {unexpected}")
        print("  -> this checkpoint may not match this classifier's config profile.")

    return model


# ============================================================================
# Convenience builders (read a single classifier profile from weights.yaml)
# ============================================================================

def build_convnext_classifier(profile_cfg: dict, num_classes: int, device: torch.device):
    model = ConvNextFoodClassifier(
        num_classes=num_classes,
        timm_name=profile_cfg.get("timm_name", "convnext_base"),
        drop_rate=profile_cfg.get("drop_rate", 0.2),
        drop_path_rate=profile_cfg.get("drop_path_rate", 0.2),
    )
    load_classifier_checkpoint(model, profile_cfg["weights"], device)
    model.to(device).eval()
    transform = build_convnext_transform(profile_cfg["resize"], profile_cfg["crop_size"])
    return model, transform


def build_dino_classifier(profile_cfg: dict, num_classes: int, device: torch.device):
    model = DinoV2FoodClassifier(
        num_classes=num_classes,
        dino_variant=profile_cfg.get("dino_variant", "dinov2_vits14"),
        feature_dim=profile_cfg.get("feature_dim", 384),
        hidden_dim=profile_cfg.get("hidden_dim", 512),
    )
    load_classifier_checkpoint(model, profile_cfg["weights"], device)
    model.to(device).eval()
    transform = build_dino_transform(profile_cfg.get("resize", 384), profile_cfg.get("crop_size", 336))
    return model, transform


@torch.no_grad()
def classify_crop(
    crop_rgb: Image.Image,
    model: nn.Module,
    transform,
    labels: List[str],
    device: torch.device,
):
    """Returns (label: str, confidence: float, probs: torch.Tensor[num_classes])."""
    tensor = transform(crop_rgb.convert("RGB")).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    return labels[idx.item()], float(conf.item()), probs.cpu()
