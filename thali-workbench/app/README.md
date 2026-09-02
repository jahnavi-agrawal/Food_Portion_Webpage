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