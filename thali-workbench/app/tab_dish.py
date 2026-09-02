"""Tab 1 — dish detection.

Finds every dish in the photo rather than assuming one per image: YOLO locates
the regions, then the ConvNeXt classifier names each crop. A close-up with no
detections falls back to classifying the whole frame.
"""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import settings
from models import classify_pil, load_dish_classifier

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


@st.cache_resource(show_spinner="Loading detector...")
def _detector():
    from thali_pipeline.models.yolo_detector import YoloDishDetector

    yolo = settings.calorie_config()["yolo"]
    return YoloDishDetector(
        weights_path=yolo["weights"],
        conf_threshold=yolo["conf_threshold"],
        iou_threshold=yolo["iou_threshold"],
        agnostic_nms=yolo["agnostic_nms"],
        max_detections=yolo["max_detections"],
    )


@st.cache_resource(show_spinner="Loading dish classifier...")
def _bundle():
    """The classifier wrapped in the shape detect_and_classify expects."""
    from thali_pipeline.pipeline.detect_classify import ClassifierBundle

    model, transform, labels, device = load_dish_classifier(
        str(settings.WEIGHTS["convnext"]), str(settings.LABELS_PATH), settings.DEVICE
    )
    return ClassifierBundle(
        models={"convnext": model}, transforms={"convnext": transform},
        labels=labels, device=device,
    )


def _draw(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"{detection['class']} ({detection['class_confidence']:.0%})"
        (width, height), baseline = cv2.getTextSize(label, FONT, 0.55, 1)
        top = max(0, y1 - height - baseline - 4)
        cv2.rectangle(out, (x1, top), (x1 + width + 6, top + height + baseline + 4), BOX_COLOR, -1)
        cv2.putText(out, label, (x1 + 3, top + height + 1), FONT, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _run(image_bgr: np.ndarray) -> dict:
    from thali_pipeline.pipeline.detect_classify import detect_and_classify

    yolo = settings.calorie_config()["yolo"]
    detections = detect_and_classify(
        image_bgr, _detector(), _bundle(), "convnext",
        min_box_px=yolo["min_box_px"], crop_padding_frac=yolo["crop_padding_frac"],
    )
    if detections:
        return {"detections": detections, "image": _draw(image_bgr, detections), "whole_frame": None}

    # Nothing found? Treat it as a single-dish close-up.
    model, transform, labels, device = load_dish_classifier(
        str(settings.WEIGHTS["convnext"]), str(settings.LABELS_PATH), settings.DEVICE
    )
    pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    predictions = classify_pil(pil_image, model, transform, labels, device)
    return {
        "detections": [],
        "image": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        "whole_frame": predictions,
    }


def render(image_bgr: np.ndarray | None) -> None:
    st.subheader("Dish detection")
    st.caption("Finds every dish in the photo and labels it.")

    if image_bgr is None:
        st.info("Upload an image in the sidebar to get started.")
        return

    if st.button("Detect dishes", type="primary", key="dish_run"):
        with st.spinner("Detecting..."):
            st.session_state["dish_payload"] = _run(image_bgr)

    payload = st.session_state.get("dish_payload")
    if not payload:
        return

    st.image(payload["image"], use_container_width=True)

    if payload["detections"]:
        st.success(f"Found **{len(payload['detections'])}** dishes.")
        st.dataframe(
            [
                {"dish": d["class"], "confidence": f"{d['class_confidence']:.1%}"}
                for d in payload["detections"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        predictions = payload["whole_frame"]
        label, confidence = predictions[0]
        st.info("No separate dishes found, so classified the whole image instead.")
        st.success(f"Predicted dish: **{label}**")
        st.write(f"Confidence: **{confidence:.2%}**")
        with st.expander("Other candidates"):
            for name, score in predictions[1:]:
                st.write(f"{name} — {score:.2%}")
