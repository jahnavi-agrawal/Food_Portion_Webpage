"""Thali workbench — dish detection, calorie estimation and portion estimation.

Run with:  streamlit run app/DishDetect.py

The image is uploaded once in the sidebar and shared by all three tabs. Each
tab does its work behind a button, because Streamlit re-runs every tab's code
on every interaction and none of these pipelines are cheap.
"""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import settings

settings.add_repos_to_path()

import tab_calorie  # noqa: E402  (must come after the repos are on sys.path)
import tab_dish  # noqa: E402
import tab_portion  # noqa: E402

st.set_page_config(page_title="Thali Workbench", layout="wide")
st.title("Thali Workbench")

with st.sidebar:
    st.header("Image")
    uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

    st.header("Setup")
    st.caption(f"Models: `{settings.MODELS_DIR}`")
    st.caption(f"Device: `{settings.DEVICE}`")

    missing = settings.missing_files()
    if missing:
        st.error("Missing files:\n" + "\n".join(f"- `{name}`" for name in missing))
        with st.expander("Expected locations"):
            for name, path in missing.items():
                st.code(str(path), language=None)
        st.caption("Point the app elsewhere with THALI_MODELS_DIR / THALI_CALORIE_REPO / "
                   "THALI_PORTION_REPO — see the README.")
    else:
        st.success("All checkpoints found.")

image_rgb = None
image_bgr = None
if uploaded_file is not None:
    image_rgb = np.array(Image.open(uploaded_file).convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    with st.sidebar:
        st.image(image_rgb, caption="Current image", use_container_width=True)

dish_tab, calorie_tab, portion_tab = st.tabs(
    ["Dish detection", "Calorie estimation", "Ashoka thali portion"]
)

with dish_tab:
    tab_dish.render(image_rgb)

with calorie_tab:
    tab_calorie.render(image_bgr)

with portion_tab:
    tab_portion.render(image_bgr)
