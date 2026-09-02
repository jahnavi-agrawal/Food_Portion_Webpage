# Thali Workbench

One Streamlit app over the three existing pieces of work: dish classification,
calorie estimation and Ashoka-tray portion estimation. Upload a photo once in
the sidebar; each tab runs its own pipeline on it.

```
app/
  DishDetect.py     entrypoint — sidebar, shared image, three tabs
  settings.py       all paths + the config dicts both pipelines already accept
  models.py         the one ConvNeXt classifier, cached and shared
  tab_dish.py       Tab 1 — multi-dish detection
  tab_calorie.py    Tab 2 — calorie estimation (general case)
  tab_portion.py    Tab 3 — portion estimation (Ashoka tray, BEV)
  requirements.txt
```

No model logic is copied here. The calorie repo's `process_image()` and the
portion repo's `build_components()` / `PortionEstimationPipeline` are imported
and called as libraries, so fixes in either project flow through automatically.

## Running

```bash
pip install -r app/requirements.txt
pip install "git+https://github.com/facebookresearch/sam2.git"   # BEV / Task 3
streamlit run app/DishDetect.py
```

The app expects this layout by default, with both repos and the checkpoints
beside the `app/` folder:

```
<parent>/
  app/
  models/          yolo_best.pt, convnext_base_ep018_macro_f1=0.9425.pt,
                   save_best_dino_v2.pth, sam2.1_hiera_tiny.pt,
                   sam2_finetuned.pth, calorie_clip.pt, labels.txt
  calorie/         the calorie repo root (contains thali_pipeline/)
  portion/         the portion repo root (contains src/portion_estimation/)
```

Anything can be moved with environment variables — the sidebar shows what's
missing and where it was looked for:

| Variable | Points at |
| --- | --- |
| `THALI_MODELS_DIR` | folder holding the checkpoints |
| `THALI_CALORIE_REPO` | calorie repo root |
| `THALI_PORTION_REPO` | portion repo root |
| `THALI_LABELS` | `labels.txt` |
| `THALI_CALORIE_LOOKUP` | `configs/calorie_lookup.csv` |
| `THALI_TRAY_PROFILE` | `configs/tray/ashoka.yaml` |
| `THALI_EMPTY_TRAY` | empty-tray reference photo |
| `THALI_DEVICE` | `auto` \| `cpu` \| `cuda` \| `mps` |

```bash
THALI_MODELS_DIR=~/weights THALI_PORTION_REPO=~/code/portion streamlit run app/DishDetect.py
```

First run downloads DINOv2 (torch.hub) and Depth Anything V2 (HuggingFace), so
it needs network access once.

## What each tab does

**Dish detection** — YOLO finds every dish, ConvNeXt names each one; falls back
to whole-frame classification when nothing is detected.

**Calorie estimation (general case)** — YOLO finds the dishes, the chosen
classifier (DINOv2, ConvNeXt or both) names each crop, CalorieCLIP regresses
kcal per crop, and each estimate is pulled toward the class lookup in
proportion to classifier confidence. No tray profile is used, so portion size
is judged from each box's area relative to the other dishes in the same photo.
Routing is selectable: Auto tries direct detection and falls back to BEV
(SAM2 tray corners → homography → CLAHE) when it finds nothing, or force
either path. `calorie_lookup.csv` is optional — without it the tab reports raw
CalorieCLIP output and says so.

**Ashoka thali portion** — the measured tray profile supplies the scale, so no
marker or reference object is needed. Compartment-floor polygons find the food,
Depth Anything V2 gives per-pixel heights self-calibrated against the exposed
floor and clamped by the tray's 2.5 cm depth, then volume × density gives
grams.

### BEV in the portion tab

For angled photos, SAM2 segments the tray and the app warps it onto a canonical
top-down canvas before the pipeline runs. The tray is placed inside a 40 px
blank margin rather than filling the canvas: `ContourPlateSegmenter` throws away
any contour touching the frame (it assumes that's the table), so the margin is
what makes the plate mask come out as exactly the tray. That in turn makes
calibration exact — 380 mm over a known 600 px — instead of depending on camera
distance. Expect roughly 0.633 mm/px in the readout when BEV is on; the sidebar
metric is a quick sanity check.

Turn BEV off for photos already shot square-on.

## Two things worth knowing

**Classifier preprocessing.** The original `DishDetect.py` used
`Resize((200, 200))` with no normalization, `weights.yaml` specifies resize 438
/ crop 384 with ImageNet normalization, and the portion repo's classifier used
224. Those can't all be right for one checkpoint. The app standardizes on the
`weights.yaml` values, since that's the documented training config. To go back
to the old behaviour, change `RESIZE` / `CROP_SIZE` in `models.py`.

**Checkpoint shape.** `DishDetect.py` treated the ConvNeXt file as a pickled
`nn.Module`; both repos treat it as a `state_dict`. `models.load_dish_classifier`
accepts either, and the portion pipeline gets its dish names through
`models.DishNameClassifier` so both tabs share one loaded model.
