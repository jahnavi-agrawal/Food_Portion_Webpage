"""Every path, weight file and pipeline config the app needs, in one place.

Override any location with an environment variable:

    THALI_MODELS_DIR     folder holding the .pt/.pth checkpoints
    THALI_CALORIE_REPO   root of the calorie repo (contains thali_pipeline/)
    THALI_PORTION_REPO   root of the portion repo (contains src/)
    THALI_LABELS         labels.txt (defaults to one inside the models folder)
    THALI_CALORIE_LOOKUP calorie_lookup.csv
    THALI_TRAY_PROFILE   ashoka.yaml
    THALI_EMPTY_TRAY     empty-tray reference photo
    THALI_DEVICE         auto | cpu | cuda | mps
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
GENERATED_DIR = APP_DIR / ".generated"


def _from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


MODELS_DIR = _from_env("THALI_MODELS_DIR", PROJECT_ROOT / "models")
CALORIE_REPO = _from_env("THALI_CALORIE_REPO", PROJECT_ROOT / "calorie")
PORTION_REPO = _from_env("THALI_PORTION_REPO", PROJECT_ROOT / "portion")

WEIGHTS = {
    "yolo": MODELS_DIR / "yolo_best.pt",
    "convnext": MODELS_DIR / "convnext_base_ep018_macro_f1=0.9425.pt",
    "dino": MODELS_DIR / "save_best_dino_v2.pth",
    "sam2_base": MODELS_DIR / "sam2.1_hiera_tiny.pt",
    "sam2_finetuned": MODELS_DIR / "sam2_finetuned.pth",
    "calorie_clip": MODELS_DIR / "calorie_clip.pt",
}

LABELS_PATH = _from_env("THALI_LABELS", MODELS_DIR / "labels.txt")
CALORIE_LOOKUP_CSV = _from_env("THALI_CALORIE_LOOKUP", CALORIE_REPO / "configs" / "calorie_lookup.csv")
TRAY_PROFILE = _from_env("THALI_TRAY_PROFILE", PORTION_REPO / "configs" / "tray" / "ashoka.yaml")
EMPTY_TRAY_IMAGE = _from_env("THALI_EMPTY_TRAY", PORTION_REPO / "images" / "empty_tray.jpg")

DEVICE = os.environ.get("THALI_DEVICE", "auto")

SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

BEV_SIZE = (600, 450)       # tray size on the warped canvas, in pixels
BEV_MARGIN_PX = 40          # blank border around the tray (see tab_portion._bev)


def add_repos_to_path() -> None:
    """Make `thali_pipeline` and `portion_estimation` importable."""
    for path in (CALORIE_REPO, PORTION_REPO / "src"):
        entry = str(path)
        if path.exists() and entry not in sys.path:
            sys.path.insert(0, entry)


def missing_files() -> dict[str, Path]:
    """Checkpoints/configs the app expects but can't find, for the sidebar."""
    expected = {name: path for name, path in WEIGHTS.items()}
    expected["labels.txt"] = LABELS_PATH
    expected["ashoka.yaml"] = TRAY_PROFILE
    return {name: path for name, path in expected.items() if not path.exists()}


# Calorie pipeline

def calorie_config(classifier_default: str = "dino") -> dict:
    """Mirrors configs/weights.yaml, minus the tray profile.

    `tray_profiles.active` is None on purpose: this tab is the general case, so
    portion scaling falls back to each dish's box area relative to the other
    dishes in the same photo (see calorie_attach.compute_portion_scale_factors).
    BEV stays available through `routing` — Task 2 runs first and Task 3
    (SAM2 -> homography -> CLAHE) takes over when direct detection finds nothing.
    """
    return {
        "device": DEVICE,
        "labels_path": str(LABELS_PATH),
        "yolo": {
            "weights": str(WEIGHTS["yolo"]),
            "conf_threshold": 0.03,
            "iou_threshold": 0.30,
            "agnostic_nms": True,
            "max_detections": 8,
            "min_box_px": 20,
            "crop_padding_frac": 0.15,
        },
        "classifiers": {
            "default": classifier_default,
            "dino": {
                "weights": str(WEIGHTS["dino"]),
                "dino_variant": "dinov2_vits14",
                "feature_dim": 384,
                "hidden_dim": 512,
                "resize": 384,
                "crop_size": 336,
            },
            "convnext": {
                "weights": str(WEIGHTS["convnext"]),
                "timm_name": "convnext_base",
                "drop_path_rate": 0.2,
                "drop_rate": 0.2,
                "resize": 438,
                "crop_size": 384,
            },
        },
        "sam2": {
            "base_checkpoint": str(WEIGHTS["sam2_base"]),
            "finetuned_checkpoint": str(WEIGHTS["sam2_finetuned"]),
            "model_cfg": SAM2_MODEL_CFG,
        },
        "calorie_clip": {
            "weights": str(WEIGHTS["calorie_clip"]),
            "base_model": "ViT-B-32",
            "pretrained": "openai",
            "min_crop_px": 20,
        },
        "calorie_lookup": {
            # Disabled automatically when the CSV isn't there, so the tab still
            # runs on raw CalorieCLIP output instead of failing to build.
            "enabled": CALORIE_LOOKUP_CSV.exists(),
            "path": str(CALORIE_LOOKUP_CSV),
            "max_weight": 0.6,
            "portion_scaling": {"enabled": True, "min_scale": 0.4, "max_scale": 2.5},
        },
        "tray_profiles": {"directory": str(GENERATED_DIR), "active": None},
        "routing": {"mode": "auto", "min_direct_detections": 1, "bev_out_size": list(BEV_SIZE)},
        "output": {"dir": str(GENERATED_DIR / "calorie")},
    }


# Portion pipeline for Ashoka six-compartment tray


def materialize_tray_profile() -> Path | None:
 
    if not TRAY_PROFILE.exists():
        return None

    profile = yaml.safe_load(TRAY_PROFILE.read_text(encoding="utf-8")) or {}

    masking = profile.setdefault("food_masking", {})
    if EMPTY_TRAY_IMAGE.exists():
        masking["reference_image_path"] = str(EMPTY_TRAY_IMAGE)
    else:
        masking.pop("reference_image_path", None)

    sam = profile.setdefault("sam_registration", {})
    sam["sam2_config"] = SAM2_MODEL_CFG
    sam["sam2_base_checkpoint"] = str(WEIGHTS["sam2_base"])
    sam["sam2_finetuned_checkpoint"] = str(WEIGHTS["sam2_finetuned"])
    sam.setdefault("bev_size", list(BEV_SIZE))
    sam.setdefault("box_prompt_margin_frac", 0.05)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GENERATED_DIR / "ashoka_resolved.yaml"
    out_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return out_path


def has_empty_tray_reference() -> bool:
    return EMPTY_TRAY_IMAGE.exists()


def sam_registration_kwargs() -> dict:
    """Constructor arguments for SamTrayRegistrar (used by the BEV step)."""
    return {
        "sam2_config": SAM2_MODEL_CFG,
        "sam2_base_checkpoint": str(WEIGHTS["sam2_base"]),
        "sam2_finetuned_checkpoint": str(WEIGHTS["sam2_finetuned"]),
        "device": DEVICE,
        "box_prompt_margin_frac": 0.05,
    }


def portion_config() -> dict:
    """Mirrors configs/baseline.yaml, pinned to the Ashoka tray.

    `food_classifier` is left unset here — the app attaches its own adapter
    (models.DishNameClassifier) after build_components() so the Dish tab and
    this tab share one cached ConvNeXt instead of loading it twice.
    """
    profile_path = materialize_tray_profile()
    food: dict = {
        "min_area_px": 500,
        "saturation_threshold": 55,
        "boundary_margin_px": 12,
        "reference_difference_threshold": 18,
        "solid_min_coverage": 0.025,
        "use_sam_registration": True,
    }
    if profile_path is not None:
        food["tray_profile_path"] = str(profile_path)

    return {
        "pipeline": {
            "plate_segmenter": "contour",
            "food_segmenter": "color_components",
            "depth_estimator": "depth_anything",
            "calibrator": "known_geometry",
            "volume_estimator": "area_height",
            "weight_estimator": "density",
            "food_classifier": None,
        },
        "plate": {"min_area_fraction": 0.10},
        "food": food,
        # Measured exterior of the Ashoka tray (configs/tray/ashoka.yaml).
        "calibration": {
            "reference_width_mm": 380.0,
            "reference_height_mm": 280.0,
            "reference_shape": "rectangle",
        },
        "volume": {
            "min_height_cm": 0.45,
            "area_height_slope": 0.06,
            "liquid_fill_height_cm": 1.75,
        },
        "weight": {
            "default_density_g_ml": 0.85,
            "solid_density_g_ml": 0.85,
            "liquid_density_g_ml": 1.0,
            "densities_g_ml": {"liquid": 1.0},
        },
        "depth": {
            "model_id": "depth-anything/Depth-Anything-V2-Small-hf",
            "device": DEVICE,
            "min_floor_reference_px": 200,
        },
        "output": {"min_label_weight_g": 3.0},
    }
