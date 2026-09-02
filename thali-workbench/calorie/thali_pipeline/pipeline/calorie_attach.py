"""Attach a CalorieCLIP calorie estimate to each detected/classified crop,
then adjust it toward a class-informed prior (the calorie lookup table),
weighted by how confident the classifier was.
"""

from typing import List, Optional, Tuple

from thali_pipeline.models.calorie_clip import CalorieCLIPModel
from thali_pipeline.models.calorie_lookup import CalorieLookupTable
from thali_pipeline.models.tray_profile import TrayProfile


def attach_calories_to_detections(
    detections: List[dict],
    calorie_model: CalorieCLIPModel,
    min_crop_px: int = 20,
) -> List[dict]:
    """Mutates+returns `detections`, adding a "calories_clip_raw" float to
    each dict (CalorieCLIP's raw, label-agnostic output). Crops below
    `min_crop_px` on either side get None (too small/noisy a region for
    CalorieCLIP to say anything meaningful).
    """
    for det in detections:
        crop = det["crop_rgb"]
        if min(crop.size) < min_crop_px:
            det["calories_clip_raw"] = None
        else:
            det["calories_clip_raw"] = calorie_model.predict(crop)
        # Default "calories" (the field everything downstream reads) to the
        # raw CLIP value; apply_class_informed_calories() below may adjust it.
        det["calories"] = det["calories_clip_raw"]
    return detections


def compute_portion_scale_factors(
    detections: List[dict],
    image_shape: Optional[Tuple[int, int]] = None,
    tray_profile: Optional[TrayProfile] = None,
    min_scale: float = 0.4,
    max_scale: float = 2.5,
) -> Tuple[List[float], List[Optional[int]]]:
    """Returns (scale_factors, matched_compartment_indices) -- one pair per
    detection.

    If `tray_profile` is given and has exactly 6 compartments, each
    detection's bbox is matched to its nearest known compartment (by
    centroid distance), and its scale factor is that compartment's FIXED
    relative_size (real floor area / mean floor area across the tray's 6
    compartments -- computed once from measured geometry, independent of
    this photo). A detection that doesn't match any compartment confidently
    (e.g. a stray false positive) gets scale=1.0 and index=None.

    Otherwise, falls back to the original relative-to-other-detections
    approach: this box's area / the mean box area across every OTHER
    detection in this same photo. There's no absolute size reference in that
    case (camera distance varies per photo), so it's relative within one
    image only, and degrades to a no-op (scale=1.0) if there's just one
    detection. The clip range keeps one bad detection from blowing up an
    estimate either way.
    """
    if not detections:
        return [], []

    if tray_profile is not None and tray_profile.compartment_count == 6 and image_shape is not None:
        scales, indices = [], []
        for d in detections:
            idx = tray_profile.match_compartment(d["bbox"], image_shape)
            if idx is None:
                scales.append(1.0)
            else:
                scales.append(max(min_scale, min(max_scale, tray_profile.relative_size_for(idx))))
            indices.append(idx)
        return scales, indices

    areas = [
        max(1, (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        for d in detections
    ]
    mean_area = sum(areas) / len(areas)
    scales = [max(min_scale, min(max_scale, a / mean_area if mean_area > 0 else 1.0)) for a in areas]
    return scales, [None] * len(detections)


def apply_class_informed_calories(
    detections: List[dict],
    lookup_table: CalorieLookupTable,
    max_lookup_weight: float = 0.6,
    portion_scaling: bool = True,
    portion_scale_min: float = 0.4,
    portion_scale_max: float = 2.5,
    image_shape: Optional[Tuple[int, int]] = None,
    tray_profile: Optional[TrayProfile] = None,
) -> List[dict]:
    """Blends each dish's raw CalorieCLIP estimate toward the class-based
    lookup estimate, since CalorieCLIP itself never sees the predicted label
    -- it's purely a visual guess. The blend weight scales with the
    classifier's own confidence, so a low-confidence label barely moves the
    number, while a confident, table-known label pulls it toward the
    class prior:

        blend_factor = class_confidence * max_lookup_weight
        calories     = (1 - blend_factor) * clip_raw + blend_factor * lookup_estimate

    If `portion_scaling` is True, the lookup estimate itself is first scaled
    per compute_portion_scale_factors() above -- tray-geometry-based if
    `tray_profile` is a 6-compartment profile, otherwise relative to the
    other dishes detected in this same photo.

    Classes not present in the lookup table are left as pure CalorieCLIP
    output (lookup_estimate=None, blend_factor=0, portion_scale_factor=None).

    Mutates+returns `detections`, adding "calories_lookup_estimate",
    "calories_blend_factor", "portion_scale_factor", "portion_scale_source"
    ("tray_profile" | "relative_area" | None), and "tray_compartment" (the
    matched compartment's label, or None) to each dict, and overwriting
    "calories" with the blended value.
    """
    if portion_scaling:
        scale_factors, compartment_indices = compute_portion_scale_factors(
            detections, image_shape, tray_profile, portion_scale_min, portion_scale_max
        )
        using_tray_profile = tray_profile is not None and tray_profile.compartment_count == 6 and image_shape is not None
    else:
        scale_factors = [1.0] * len(detections)
        compartment_indices = [None] * len(detections)
        using_tray_profile = False

    for det, scale, comp_idx in zip(detections, scale_factors, compartment_indices):
        clip_raw = det.get("calories_clip_raw")
        if clip_raw is None:
            det["calories_lookup_estimate"] = None
            det["calories_blend_factor"] = 0.0
            det["portion_scale_factor"] = None
            det["portion_scale_source"] = None
            det["tray_compartment"] = None
            continue

        lookup_estimate: Optional[float] = lookup_table.estimate(det["class"], portion_scale=scale)
        det["portion_scale_source"] = (
            "tray_profile" if (using_tray_profile and comp_idx is not None)
            else "relative_area" if portion_scaling
            else None
        )
        det["tray_compartment"] = tray_profile.label_for(comp_idx) if (using_tray_profile and comp_idx is not None) else None

        if lookup_estimate is None:
            det["calories_lookup_estimate"] = None
            det["calories_blend_factor"] = 0.0
            det["portion_scale_factor"] = None
            # det["calories"] stays as clip_raw (set in attach_calories_to_detections)
            continue

        blend_factor = det["class_confidence"] * max_lookup_weight
        blended = (1.0 - blend_factor) * clip_raw + blend_factor * lookup_estimate

        det["calories_lookup_estimate"] = lookup_estimate
        det["calories_blend_factor"] = round(blend_factor, 4)
        det["portion_scale_factor"] = round(scale, 3)
        det["calories"] = blended

    return detections
