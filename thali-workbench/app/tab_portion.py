"""
Portion estimation for the Ashoka six-compartment tray.

Runs the portion repo's pipeline pinned to configs/tray/ashoka.yaml: contour
plate mask and  known-geometry calibration to  compartment-floor food masks to
hardcoded heights which gives volume that is converted to grams, using density priors. 
.
"""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

import settings


@st.cache_resource(show_spinner="Loading portion pipeline...")
def _pipeline(name_dishes: bool):
    from portion_estimation.factory import build_components
    from portion_estimation.pipeline import PortionEstimationPipeline

    components = build_components(settings.portion_config())
    if name_dishes:
        from models import DishNameClassifier, load_dish_classifier

        model, transform, labels, device = load_dish_classifier(
            str(settings.WEIGHTS["convnext"]), str(settings.LABELS_PATH), settings.DEVICE
        )
        components["food_classifier"] = DishNameClassifier(model, transform, labels, device)
    return PortionEstimationPipeline(**components)


@st.cache_resource(show_spinner=False)
def _registrar():
    from portion_estimation.segmentation import SamTrayRegistrar

    return SamTrayRegistrar(**settings.sam_registration_kwargs())


def _bev(image_bgr: np.ndarray) -> tuple[np.ndarray | None, str | None]:

    registrar = _registrar()
    if not registrar.available:
        return None, f"SAM2 checkpoint not found ({settings.WEIGHTS['sam2_base']}) and BEV unavailable."

    corners = registrar.tray_corners(image_bgr)
    if corners is None:
        return None, "SAM2 could not find the tray outline and BEV unavailable."

    width, height = settings.BEV_SIZE
    margin = settings.BEV_MARGIN_PX
    destination = np.float32([
        [margin, margin],
        [margin + width, margin],
        [margin + width, margin + height],
        [margin, margin + height],
    ])
    homography, _ = cv2.findHomography(corners, destination)
    if homography is None:
        return None, "Could not solve the BEV homography and BEV unavailable."

    warped = registrar.warp_to_bev(image_bgr, homography, (width + 2 * margin, height + 2 * margin))
    return warped, None


def _run(image_bgr: np.ndarray, use_bev: bool, name_dishes: bool, min_label_weight_g: float) -> dict:
    from portion_estimation.visualization import render_overlay

    notice, bev_rgb = None, None
    working = image_bgr
    if use_bev:
        warped, notice = _bev(image_bgr)
        if warped is not None:
            working = warped
            bev_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    result = _pipeline(name_dishes).predict(working)
    overlay = render_overlay(working, result, min_label_weight_g=min_label_weight_g)

    return {
        "data": result.to_dict(),
        "overlay": cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
        "bev": bev_rgb,
        "notice": notice,
    }


def _show(payload: dict) -> None:
    if payload["notice"]:
        st.warning(payload["notice"])

    if payload["bev"] is not None:
        left, right = st.columns(2)
        left.image(payload["bev"], caption="Bird's-eye view", use_container_width=True)
        right.image(payload["overlay"], caption="Estimated portions", use_container_width=True)
    else:
        st.image(payload["overlay"], caption="Estimated portions", use_container_width=True)

    data = payload["data"]
    calibration = data["calibration"]
    columns = st.columns(3)
    columns[0].metric("Foods", len(data["foods"]))
    columns[1].metric("Total weight", f"{data['total_weight_g']:.0f} g")
    columns[2].metric("Scale", f"{calibration['mm_per_pixel_x']:.3f} mm/px")

    if data["foods"]:
        st.dataframe(
            [
                {
                    "dish": row["label"],
                    "type": row["physical_type"],
                    "coverage": round(row["compartment_coverage"], 3),
                    "area cm²": round(row["area_cm2"], 1),
                    "height cm": round(row["mean_height_cm"], 2),
                    "volume ml": round(row["volume_ml"], 1),
                    "weight g": round(row["weight_g"], 1),
                }
                for row in data["foods"]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Heights are clamped by the tray's measured 2.5 cm floor-to-rim depth, so a "
            "compartment can never report food standing taller than its own wall."
        )
    else:
        st.warning(
            "No food found. Check that the whole tray rim is visible, and try BEV correction "
            "if the photo isn't top-down."
        )


def render(image_bgr: np.ndarray | None) -> None:
    st.subheader("Ashoka thali portion estimation")
    st.caption(
        "Six-compartment tray, 380 × 280 mm, 2.5 cm deep. It is a measured geometry from "
        "configs/tray/ashoka.yaml that supplies the scale, so no marker or reference object is needed :)"
    )

    if image_bgr is None:
        st.info("Upload an image in the sidebar to get started.")
        return

    if not settings.TRAY_PROFILE.exists():
        st.error(f"Tray profile not found: {settings.TRAY_PROFILE}")
        return
    if not settings.has_empty_tray_reference():
        st.warning(
            f"Empty-tray reference not found at {settings.EMPTY_TRAY_IMAGE} so food masks fall back "
            "to saturation thresholding instead of reference differencing, which is less accurate."
        )

    left, middle, right = st.columns(3)
    use_bev = left.checkbox("BEV correction", value=True, key="por_bev",
                            help="Warp an angled photo to top-down with SAM2 before measuring.")
    name_dishes = middle.checkbox("Name dishes", value=True, key="por_names",
                                  help="Label each compartment with a dish name instead of solid/liquid.")
    min_label_weight_g = right.number_input("Hide labels under (g)", 0.0, 100.0, 3.0, 0.5, key="por_minw")

    if st.button("Estimate portions", type="primary", key="por_run"):
        with st.spinner("Running pipeline..."):
            st.session_state["portion_payload"] = _run(image_bgr, use_bev, name_dishes, min_label_weight_g)

    payload = st.session_state.get("portion_payload")
    if payload:
        _show(payload)
