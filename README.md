# Clause Lens — Passport Field Detection & Extraction

A computer-vision pipeline that **detects** passport fields using [YOLO26](https://docs.ultralytics.com/) and **extracts** their text with OCR, outputting structured JSON.

```
Passport image  ──▶  YOLO26 Detection  ──▶  Field Crops  ──▶  OCR  ──▶  Structured JSON
```

## ✨ Key Highlights

- **YOLO26 (Sep 2025)** — latest Ultralytics model with NMS-free inference, improved small-object detection (ProgLoss + STAL), and ~31% faster CPU inference than YOLO11.
- **Nano variant** (`yolo26n.pt`, ~3M params) — deliberately small to avoid overfitting on a 50-image dataset, fast to iterate, easy to deploy.
- **Transfer learning** from COCO pretrained weights — universal low-level features for free; only final layers are fine-tuned.
- **End-to-end notebooks** — from data labelling to model training, fully explained for AI students and engineers.

## 📂 Project Structure

```
clause-ade/
├── api/
│   ├── notebooks/
│   │   ├── vision-v0.ipynb         # Notebook 0 — LabelImg setup & Python 3.12 fix
│   │   ├── vision-v1.ipynb         # Notebook 1 — Data annotation guide (YOLO format)
│   │   ├── vision-v2.ipynb         # Notebook 2 — Train/val split + data.yaml creation
│   │   ├── vision-v3.ipynb         # Notebook 3 — YOLO26 training (hyperparams explained)
│   │   ├── rename_images.ipynb     # Utility — batch rename images to img_00, img_01, …
│   │   ├── raw_data/
│   │   │   ├── images/             # 50 original passport images (.jpeg)
│   │   │   └── labels/             # YOLO-format annotations (.txt per image)
│   │   ├── dataset/
│   │   │   ├── data.yaml           # YOLO dataset config (paths, classes)
│   │   │   ├── images/
│   │   │   │   ├── train/          # 40 training images (80% split)
│   │   │   │   └── val/            # 10 validation images (20% split)
│   │   │   └── labels/
│   │   │       ├── train/          # Corresponding training labels
│   │   │       └── val/            # Corresponding validation labels
│   │   └── yolo26n.pt              # Pretrained YOLO26 nano weights
│   ├── pyproject.toml              # Python project config (dependencies)
│   ├── uv.lock                     # Locked dependency versions
│   └── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- [**uv**](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or `pip`

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd clause-ade/api

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLO26 model training & inference |
| `torch` / `torchvision` | Deep learning backend |
| `numpy` | Array operations |
| `labelimg` | Bounding-box annotation tool |
| `pyqt5` | GUI framework (for LabelImg) |

## 📓 Notebook Pipeline

The project is structured as a 4-notebook pipeline. Run them in order:

| # | Notebook | What it does |
|---|----------|-------------|
| 0 | `vision-v0.ipynb` | Sets up **LabelImg** for annotation, including an auto-fix for the Python 3.12 `float→int` crash |
| 1 | `vision-v1.ipynb` | Guides you through **annotating** passport fields in YOLO format with LabelImg |
| 2 | `vision-v2.ipynb` | Splits raw data into **train/val** sets (80/20) and generates `data.yaml` |
| 3 | `vision-v3.ipynb` | **Trains YOLO26n** on the dataset — every hyperparameter and metric explained in depth |

> **Notebook 4** (inference + OCR → structured JSON) is planned next.

## 🏷️ Dataset Details

| Property | Value |
|----------|-------|
| Total images | 50 |
| Train / Val split | 40 / 10 (80/20) |
| Annotation format | YOLO (normalized `x_center y_center width height`) |
| Image resolution | Variable (letterboxed to 640×640 for training) |
| Country coverage | Single nationality |
| Image quality | Mixed — clean scans, blurry phone shots, varied lighting |

### Target Fields

The model detects passport fields such as:
- `first_name`
- `last_name`
- `passport_no`
- `nationality`
- `dob` (date of birth, when visible)

## 🏋️ Training

From `api/notebooks/`, open and run `vision-v3.ipynb`, or use the CLI:

```bash
cd api
uv run yolo detect train \
    data=notebooks/dataset/data.yaml \
    model=notebooks/yolo26n.pt \
    imgsz=640 \
    batch=8 \
    epochs=100 \
    patience=20 \
    project=runs/detect \
    name=passport_v1
```

### Training Outputs

```
api/runs/detect/passport_v1/
├── weights/
│   ├── best.pt              ← Best val mAP checkpoint (use for inference)
│   └── last.pt              ← Final epoch checkpoint
├── results.png              ← Loss & mAP curves
├── confusion_matrix.png     ← Per-class confusion matrix
├── val_batch0_pred.jpg       ← Predictions on validation images
└── results.csv              ← Per-epoch metrics (open with pandas)
```

### Key Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `imgsz` | 640 | Matches COCO pretrain resolution; balances accuracy vs. speed |
| `batch` | 8 | Fits in most laptop GPUs (2–4 GB VRAM) |
| `epochs` | 100 | Sufficient with augmentation for 50 images |
| `patience` | 20 | Early stopping — prevents overfitting |
| Model | `yolo26n.pt` | Nano variant — lower capacity matches small dataset |

## 📊 Metrics Guide

| Metric | What it means |
|--------|---------------|
| `box_loss` | How far off predicted box coordinates are (lower = better) |
| `cls_loss` | How wrong the class predictions are (lower = better) |
| `mAP50` | Mean Average Precision at IoU ≥ 0.5 (the easier metric) |
| `mAP50-95` | Mean AP averaged over IoU 0.5–0.95 (the standard benchmark) |

> For document AI with consistent layouts, expect **mAP50-95 > 0.50** to be production-viable, and **> 0.70** to be excellent.

## 🛠️ LabelImg — Python 3.12 Fix

LabelImg crashes on Python 3.10+ due to `float→int` type errors in PyQt5 calls. Notebook `vision-v0.ipynb` includes an **automatic patch** that wraps affected calls in `int(...)`. Re-run the fix cell after every `uv sync` or `pip install`.

## 🗺️ Roadmap

- [x] Data collection (50 passport images)
- [x] LabelImg setup with Python 3.12 fix
- [x] Annotation in YOLO format
- [x] Train/val split & data.yaml generation
- [x] YOLO26n training pipeline
- [ ] Inference notebook + OCR extraction → structured JSON
- [ ] FastAPI backend for serving predictions
- [ ] Scale dataset to 500+ images & retrain with `yolo26s.pt`

## 👤 Author

**Abdullah Ahmed** — [abd1962964@gmail.com](mailto:abd1962964@gmail.com)

## 📄 License

This project is for educational purposes as part of an AI / Computer Vision course.
