from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

from .calibration import KnownGeometryCalibrator
from .depth import LuminanceRelativeDepthEstimator, MetricMonoDepthEstimator
from .geometry import ContourPlateSegmenter, SimplePlateGeometryEstimator
from .segmentation import ColorComponentFoodSegmenter, SamTrayRegistrar
from .weight import AreaHeightVolumeEstimator, DensityWeightEstimator

try:
    from .classification import ConvNeXtDishClassifier, YoloBoxFoodSegmenter
except ImportError:  # torch/timm/ultralytics not installed — these stay unavailable
    ConvNeXtDishClassifier = None  # type: ignore[assignment]
    YoloBoxFoodSegmenter = None  # type: ignore[assignment]


def _load_tray_profile(food_config: dict[str, Any]) -> dict[str, Any]:
    path = food_config.get("tray_profile_path")
    if not path or not Path(path).exists():
        return {}
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _compartment_depth_cm(tray_profile: dict[str, Any], default_cm: float = 2.5) -> float:
    """Single source of truth for the physical compartment ceiling.

    Reads `layout.compartment_depth_mm` from the tray profile (2.5 cm for the
    Ashoka tray) so `AreaHeightVolumeEstimator.max_height_cm` and
    `MetricMonoDepthEstimator.compartment_depth_cm` can never drift apart the
    way `baseline.yaml`'s old standalone `max_height_cm: 3.0` /
    `depth_range_cm: 2.0` did.
    """
    depth_mm = tray_profile.get("layout", {}).get("compartment_depth_mm")
    if depth_mm is None:
        return default_cm
    return float(depth_mm) / 10.0


def _build_sam_registrar(food_config: dict[str, Any], tray_profile: dict[str, Any]) -> SamTrayRegistrar | None:
    sam_cfg = tray_profile.get("sam_registration")
    if not sam_cfg or not food_config.get("use_sam_registration", True):
        return None
    return SamTrayRegistrar(
        sam2_config=sam_cfg.get("sam2_config", "configs/sam2.1/sam2.1_hiera_t.yaml"),
        sam2_base_checkpoint=sam_cfg.get("sam2_base_checkpoint", "models/sam2.1_hiera_tiny.pt"),
        sam2_finetuned_checkpoint=sam_cfg.get("sam2_finetuned_checkpoint"),
        box_prompt_margin_frac=float(sam_cfg.get("box_prompt_margin_frac", 0.05)),
    )


def build_components(config: dict[str, Any]) -> dict[str, Any]:
    p = config.get('pipeline', {})
    plate = config.get('plate', {})
    food = dict(config.get('food', {}))
    cal = config.get('calibration', {})
    vol = dict(config.get('volume', {}))
    weight = config.get('weight', {})
    depth_cfg = config.get('depth', {})
    clf_cfg = config.get('classification', {})

    if p.get('plate_segmenter', 'contour') != 'contour':
        raise ValueError('Unsupported plate_segmenter')
    food_segmenter_choice = p.get('food_segmenter', 'color_components')
    if food_segmenter_choice not in ('color_components', 'yolo_box'):
        raise ValueError(f'Unsupported food_segmenter: {food_segmenter_choice!r}')

    tray_profile = _load_tray_profile(food)
    compartment_depth_cm = _compartment_depth_cm(tray_profile)

    if food_segmenter_choice == 'yolo_box':
        # Task 3 style: general thali photo, no known fixed compartment
        # layout to align floor polygons against. Detect dish regions with
        # YOLO directly instead. See classification/dish_classifier.py.
        if YoloBoxFoodSegmenter is None:
            raise ImportError('yolo_box food_segmenter requires ultralytics to be installed')
        food_segmenter = YoloBoxFoodSegmenter(
            weights_path=clf_cfg.get('yolo_weights', 'models/best.pt'),
            conf_threshold=float(clf_cfg.get('conf_threshold', 0.10)),
            iou_threshold=float(clf_cfg.get('iou_threshold', 0.30)),
            device=clf_cfg.get('device', 'auto'),
            min_area_px=food.get('min_area_px', 500),
            saturation_threshold=food.get('saturation_threshold', 55),
        )
    else:
        use_sam = food.pop('use_sam_registration', True)
        sam_registrar = _build_sam_registrar({**food, 'use_sam_registration': use_sam}, tray_profile)
        food_segmenter = ColorComponentFoodSegmenter(use_sam_registration=use_sam, sam_registrar=sam_registrar, **food)

    depth_choice = p.get('depth_estimator', 'luminance_relative')
    luminance_fallback = LuminanceRelativeDepthEstimator()
    if depth_choice == 'depth_anything':
        depth_estimator: Any = MetricMonoDepthEstimator(
            model_id=depth_cfg.get('model_id', 'depth-anything/Depth-Anything-V2-Small-hf'),
            device=depth_cfg.get('device', 'auto'),
            compartment_depth_cm=compartment_depth_cm,
            min_floor_reference_px=int(depth_cfg.get('min_floor_reference_px', 200)),
            fallback=luminance_fallback,
            manual_scale=float(depth_cfg.get('manual_scale', 1.0)),
            manual_shift=float(depth_cfg.get('manual_shift', 0.0)),
        )
        depth_is_metric_cm = True
    elif depth_choice == 'luminance_relative':
        depth_estimator = luminance_fallback
        depth_is_metric_cm = False
    else:
        raise ValueError(f'Unsupported depth_estimator: {depth_choice!r}')

    # Ceiling clamp always comes from the tray file, never from a number
    # duplicated in baseline.yaml's `volume:` block (see docstring above).
    vol.pop('max_height_cm', None)
    vol.pop('depth_range_cm', None)
    volume_estimator = AreaHeightVolumeEstimator(
        max_height_cm=compartment_depth_cm,
        depth_range_cm=1.0 if depth_is_metric_cm else 2.0,
        depth_is_metric_cm=depth_is_metric_cm,
        **vol,
    )

    components: dict[str, Any] = {
        'plate_segmenter': ContourPlateSegmenter(**plate),
        'geometry_estimator': SimplePlateGeometryEstimator(),
        'food_segmenter': food_segmenter,
        'depth_estimator': depth_estimator,
        'calibrator': KnownGeometryCalibrator(cal['reference_width_mm'], cal['reference_height_mm']),
        'volume_estimator': volume_estimator,
        'weight_estimator': DensityWeightEstimator(
            weight.get('default_density_g_ml', .85), weight.get('densities_g_ml'),
            weight.get('solid_density_g_ml'), weight.get('liquid_density_g_ml', 1.0),
        ),
    }

    classifier_choice = p.get('food_classifier')
    if classifier_choice == 'yolo_convnext' and ConvNeXtDishClassifier is not None:
        components['food_classifier'] = ConvNeXtDishClassifier(
            weights_path=clf_cfg.get('convnext_weights', 'models/convnext_base.pt'),
            labels_path=clf_cfg.get('labels_path', 'models/labels.txt'),
            device=clf_cfg.get('device', 'auto'),
        )
    else:
        components['food_classifier'] = None

    return components
