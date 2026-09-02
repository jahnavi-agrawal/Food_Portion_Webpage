"""CalorieCLIP: image -> estimated calories.

Reproduces https://huggingface.co/jc-builds/CalorieCLIP exactly (RegressionHead
architecture and checkpoint keys "clip_state" / "regressor_state"), so
calorie_clip.pt loads unmodified. We attach this per detected-dish CROP rather
than on the whole thali, since the model card itself flags: "Single-dish
focused; complex multi-item plates may have higher error." A whole-image call
is still available (see calorie_attach.py) as a sanity-check total, per your
call — just don't treat it as more trustworthy than the per-dish sum.
"""

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from PIL import Image


class RegressionHead(nn.Module):
    """Matches jc-builds/CalorieCLIP's calorie_clip.py exactly."""

    def __init__(self, input_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


class CalorieCLIPModel:
    def __init__(
        self,
        weights_path: str,
        base_model: str = "ViT-B-32",
        pretrained: str = "openai",
        device: torch.device = torch.device("cpu"),
    ):
        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "CalorieCLIP requires open-clip-torch: pip install open-clip-torch"
            ) from e

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"CalorieCLIP weights not found: {weights_path}")

        self.device = device
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            base_model, pretrained=pretrained
        )
        self.head = RegressionHead(input_dim=512)

        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        if "clip_state" in checkpoint:
            self.clip_model.load_state_dict(checkpoint["clip_state"], strict=False)
        else:
            print("[calorie_clip] WARNING: checkpoint has no 'clip_state' key; "
                  "using base pretrained CLIP weights only (predictions will be inaccurate).")

        if "regressor_state" in checkpoint:
            self.head.load_state_dict(checkpoint["regressor_state"])
        elif "head_state" in checkpoint:
            self.head.load_state_dict(checkpoint["head_state"])
        else:
            raise KeyError(
                f"{weights_path} has neither 'regressor_state' nor 'head_state' — "
                f"this doesn't look like a CalorieCLIP checkpoint."
            )

        self.clip_model.to(device).eval()
        self.head.to(device).eval()

    @torch.no_grad()
    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # Note: CalorieCLIP's own code does NOT L2-normalize features before
        # the regression head — training didn't use normalization either.
        return self.clip_model.encode_image(image_tensor).float()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> float:
        image = image.convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.encode_image(tensor)
        calories = self.head(features).item()
        return max(0.0, calories)

    @torch.no_grad()
    def predict_batch(self, images: List[Image.Image]) -> List[float]:
        if not images:
            return []
        tensors = torch.stack(
            [self.preprocess(img.convert("RGB")) for img in images]
        ).to(self.device)
        features = self.encode_image(tensors)
        calories = self.head(features).squeeze(-1)
        return [max(0.0, float(c)) for c in calories.cpu().tolist()]
