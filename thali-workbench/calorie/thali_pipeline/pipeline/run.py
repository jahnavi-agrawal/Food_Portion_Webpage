"""End-to-end orchestration.

    ctx = build_context(config, classifier_modes=["dino"])
    result, annotated_bgr = process_image("plate.jpg", ctx, classifier_mode="dino")

`build_context` loads every model ONCE (YOLO, requested classifier profile(s),
CalorieCLIP; SAM2 stays lazy until a Task-3 path is actually hit). Reuse the
same `ctx` across many images — that's the whole point of the split.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

from thali_pipeline.annotate import annotate_image
from thali_pipeline.config import REPO_ROOT
from thali_pipeline.device import resolve_device
from thali_pipeline.io_utils import load_image_bgr
from thali_pipeline.labels import load_labels
from thali_pipeline.models.calorie_clip import CalorieCLIPModel
from thali_pipeline.models.calorie_lookup import CalorieLookupTable, load_calorie_lookup
from thali_pipeline.models.classifiers import build_convnext_classifier, build_dino_classifier
from thali_pipeline.models.sam2_segmenter import Sam2TraySegmenter
from thali_pipeline.models.tray_profile import TrayProfile, load_tray_profile
from thali_pipeline.models.yolo_detector import YoloDishDetector
from thali_pipeline.pipeline.calorie_attach import (
    apply_class_informed_calories,
    attach_calories_to_detections,
)
from thali_pipeline.pipeline.detect_classify import ClassifierBundle
from thali_pipeline.pipeline.router import route_and_detect


@dataclass
class PipelineContext:
    config: dict
    device: torch.device
    labels: List[str]
    detector: YoloDishDetector
    classifier_bundle: ClassifierBundle
    segmenter: Sam2TraySegmenter
    calorie_model: CalorieCLIPModel
    calorie_lookup: Optional[CalorieLookupTable]
    tray_profile: Optional[TrayProfile]


def build_context(
    config: dict,
    classifier_modes: Optional[List[str]] = None,
    tray_profile_override: Optional[str] = None,
) -> PipelineContext:
    """classifier_modes: which classifier profile(s) to load — any key
    present under `classifiers` in configs/weights.yaml (currently "dino"
    and "convnext"). Pass both if you plan to use classifier_mode="ensemble"
    later. Defaults to just `classifiers.default` from the config.

    tray_profile_override: a tray profile id (filename stem under
    tray_profiles.directory), or "none" to force pure relative-area scaling.
    Defaults to `tray_profiles.active` from the config.
    """
    device = resolve_device(config.get("device", "auto"))
    labels = load_labels(config["labels_path"])

    if classifier_modes is None:
        classifier_modes = [config["classifiers"].get("default", "dino")]

    models, transforms = {}, {}
    for mode in classifier_modes:
        profile_cfg = config["classifiers"][mode]
        if mode == "dino":
            model, transform = build_dino_classifier(profile_cfg, len(labels), device)
        elif mode in ("convnext", "standalone_convnext"):
            model, transform = build_convnext_classifier(profile_cfg, len(labels), device)
        else:
            raise ValueError(f"unknown classifier mode: {mode}")
        models[mode] = model
        transforms[mode] = transform

    classifier_bundle = ClassifierBundle(models=models, transforms=transforms, labels=labels, device=device)

    yolo_cfg = config["yolo"]
    detector = YoloDishDetector(
        weights_path=yolo_cfg["weights"],
        conf_threshold=yolo_cfg.get("conf_threshold", 0.10),
        iou_threshold=yolo_cfg.get("iou_threshold", 0.30),
        agnostic_nms=yolo_cfg.get("agnostic_nms", True),
        max_detections=yolo_cfg.get("max_detections", 8),
    )

    sam2_cfg = config["sam2"]
    segmenter = Sam2TraySegmenter(
        base_checkpoint=sam2_cfg["base_checkpoint"],
        finetuned_checkpoint=sam2_cfg.get("finetuned_checkpoint"),
        model_cfg=sam2_cfg["model_cfg"],
        device=device,
    )

    cc_cfg = config["calorie_clip"]
    calorie_model = CalorieCLIPModel(
        weights_path=cc_cfg["weights"],
        base_model=cc_cfg.get("base_model", "ViT-B-32"),
        pretrained=cc_cfg.get("pretrained", "openai"),
        device=device,
    )

    lookup_cfg = config.get("calorie_lookup", {})
    calorie_lookup = None
    if lookup_cfg.get("enabled", True):
        calorie_lookup = load_calorie_lookup(lookup_cfg["path"])

    tray_cfg = config.get("tray_profiles", {})
    tray_profile_id = tray_profile_override if tray_profile_override is not None else tray_cfg.get("active")
    tray_profile = None
    if tray_profile_id and tray_profile_id.lower() != "none":
        tray_dir = Path(tray_cfg.get("directory", "configs/tray_profiles"))
        if not tray_dir.is_absolute():
            tray_dir = REPO_ROOT / tray_dir
        tray_profile = load_tray_profile(str(tray_dir / f"{tray_profile_id}.yaml"))

    return PipelineContext(
        config=config,
        device=device,
        labels=labels,
        detector=detector,
        classifier_bundle=classifier_bundle,
        segmenter=segmenter,
        calorie_model=calorie_model,
        calorie_lookup=calorie_lookup,
        tray_profile=tray_profile,
    )


def process_image(
    image_path: str,
    ctx: PipelineContext,
    classifier_mode: str = "dino",
    task_mode: Optional[str] = None,
    manual_task: Optional[int] = None,
):
    """Runs the full pipeline on one image.

    Returns (result: dict, annotated_bgr: np.ndarray).
    `result` is exactly what gets written to the per-image JSON.
    """
    routing_cfg = ctx.config["routing"]
    mode = task_mode or routing_cfg.get("mode", "auto")
    min_box_px = ctx.config["yolo"].get("min_box_px", 20)
    crop_padding_frac = ctx.config["yolo"].get("crop_padding_frac", 0.15)
    min_crop_px = ctx.config["calorie_clip"].get("min_crop_px", 20)

    img_bgr = load_image_bgr(image_path)

    detections, base_img_bgr, task_used = route_and_detect(
        img_bgr=img_bgr,
        detector=ctx.detector,
        classifier_bundle=ctx.classifier_bundle,
        classifier_mode=classifier_mode,
        min_box_px=min_box_px,
        crop_padding_frac=crop_padding_frac,
        segmenter=ctx.segmenter,
        mode=mode,
        min_direct_detections=routing_cfg.get("min_direct_detections", 1),
        bev_out_size=tuple(routing_cfg.get("bev_out_size", [600, 450])),
        manual_task=manual_task,
    )

    detections = attach_calories_to_detections(detections, ctx.calorie_model, min_crop_px=min_crop_px)

    if ctx.calorie_lookup is not None:
        lookup_cfg = ctx.config.get("calorie_lookup", {})
        portion_cfg = lookup_cfg.get("portion_scaling", {})
        detections = apply_class_informed_calories(
            detections,
            ctx.calorie_lookup,
            max_lookup_weight=lookup_cfg.get("max_weight", 0.6),
            portion_scaling=portion_cfg.get("enabled", True),
            portion_scale_min=portion_cfg.get("min_scale", 0.4),
            portion_scale_max=portion_cfg.get("max_scale", 2.5),
            image_shape=base_img_bgr.shape[:2],
            tray_profile=ctx.tray_profile,
        )

    total_calories = sum(d["calories"] for d in detections if d["calories"] is not None)

    annotated_bgr = annotate_image(
        base_img_bgr, detections, task_used, total_calories, classifier_mode
    )

    result = {
        "image": Path(image_path).name,
        "task_used": task_used,
        "classifier_mode": classifier_mode,
        "tray_profile": ctx.tray_profile.profile_id if ctx.tray_profile is not None else None,
        "num_dishes": len(detections),
        "total_calories_summed": round(total_calories, 1),
        "detections": [
            {
                "id": i,
                "class": d["class"],
                "class_confidence": round(d["class_confidence"], 4),
                "detector_confidence": round(d["det_conf"], 4),
                "bbox_xyxy": d["bbox"],
                "calories": round(d["calories"], 1) if d["calories"] is not None else None,
                "calories_clip_raw": (
                    round(d["calories_clip_raw"], 1) if d.get("calories_clip_raw") is not None else None
                ),
                "calories_lookup_estimate": (
                    round(d["calories_lookup_estimate"], 1)
                    if d.get("calories_lookup_estimate") is not None else None
                ),
                "calories_blend_factor": d.get("calories_blend_factor", 0.0),
                "portion_scale_factor": d.get("portion_scale_factor"),
                "portion_scale_source": d.get("portion_scale_source"),
                "tray_compartment": d.get("tray_compartment"),
            }
            for i, d in enumerate(detections)
        ],
    }

    return result, annotated_bgr
