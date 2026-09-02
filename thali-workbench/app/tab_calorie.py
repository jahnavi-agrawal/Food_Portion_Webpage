"""Tab 2 : calorie estimation
Wraps the calorie repo's `process_image`: YOLO finds the dishes, the chosen
classifier names each crop, CalorieCLIP regresses kcal per crop, and the result
is blended toward the class lookup in proportion to classifier confidence.

"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import settings

ROUTES = {
    "Auto": (None, None),
    "Direct only": ("manual", 2),
    "BEV only": ("manual", 3),
}


@st.cache_resource(show_spinner="Loading calorie pipeline...")
def _context(classifier_mode: str):
    from thali_pipeline.pipeline.run import build_context

    modes = ["dino", "convnext"] if classifier_mode == "ensemble" else [classifier_mode]
    config = settings.calorie_config("dino" if classifier_mode == "ensemble" else classifier_mode)
    # "none" forces relative-area portion scaling regardless of config 
    return build_context(config, classifier_modes=modes, tray_profile_override="none")


def _run(image_bgr: np.ndarray, classifier: str, route: str) -> dict:
    from thali_pipeline.pipeline.run import process_image

    task_mode, manual_task = ROUTES[route]
    tmp_path = Path(tempfile.gettempdir()) / "thali_calorie_input.png"
    cv2.imwrite(str(tmp_path), image_bgr)

    result, annotated_bgr = process_image(
        str(tmp_path),
        _context(classifier),
        classifier_mode=classifier,
        task_mode=task_mode,
        manual_task=manual_task,
    )
    return {"result": result, "annotated": cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)}


def _show(payload: dict) -> None:
    result = payload["result"]
    st.image(payload["annotated"], caption="Detected dishes", use_container_width=True)

    columns = st.columns(2)
    columns[0].metric("Dishes", result["num_dishes"])
    columns[1].metric("Total kcal", f"{result['total_calories_summed']:.0f}")

    if result["detections"]:
        st.dataframe(
            [
                {
                    "dish": d["class"],
                    "confidence": round(d["class_confidence"], 3),
                    "kcal": d["calories"],
                    "CalorieCLIP raw": d["calories_clip_raw"],
                    "lookup kcal": d["calories_lookup_estimate"],
                }
                for d in result["detections"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No dishes detected. Try the BEV route if the photo is taken at an angle.")


def render(image_bgr: np.ndarray | None) -> None:
    st.subheader("Calorie estimation")
    st.caption(
        "Detects and classifies using CalorieCLIP per crop, blended toward the class lookup by "
        "classifier confidence. Portion size is judged relative to the other dishes in the photo."
    )

    if image_bgr is None:
        st.info("Upload an image in the sidebar to get started.")
        return

    if not settings.CALORIE_LOOKUP_CSV.exists():
        st.warning(
            f"calorie_lookup.csv not found at {settings.CALORIE_LOOKUP_CSV} and running on raw "
            "CalorieCLIP output with no class-informed blending."
        )

    left, right = st.columns(2)
    classifier = left.selectbox("Classifier", ["dino", "convnext", "ensemble"], key="cal_clf")
    route = right.selectbox("Routing", list(ROUTES), key="cal_route")

    if st.button("Estimate calories", type="primary", key="cal_run"):
        with st.spinner("Running pipeline..."):
            st.session_state["calorie_payload"] = _run(image_bgr, classifier, route)

    payload = st.session_state.get("calorie_payload")
    if payload:
        _show(payload)
