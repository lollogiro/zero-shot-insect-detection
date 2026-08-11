#!/usr/bin/env python3
"""YOLO supervised baseline inference on the shipped example dataset."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repo root: examples/ -> ..
sys.path.insert(0, str(ROOT))                # importable without the editable install
os.environ.setdefault("MODEL_CACHE_DIR", str(ROOT / "models-cache"))

from utils.models import FlatBugDetector
from utils.inference import run_inference, has_binary_classifier

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# SLICE_SIZE and OVERLAP used here for dynamic output naming
SLICE_SIZE = None
OVERLAP = None

# ---------------------------------------------------------------------------
# Config: runs on the shipped example_dataset
# ---------------------------------------------------------------------------
COCO_JSON_PATH = f"{ROOT}/example_dataset/detections_val_20.json"
IMAGES_DIR = f"{ROOT}/example_dataset"
OUTPUT_PATH = f"{ROOT}/results/flatbug/example_dataset_slice{SLICE_SIZE if SLICE_SIZE is not None else 'nan'}_overlap{int(OVERLAP * 100) if OVERLAP is not None else 'nan'}.json"
DESCRIPTION = "Example run: FlatBug on example_dataset"

CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.1

# Binary classifier (optional): insect vs background filter
USE_BINARY_CLASSIFIER = False
DISCRIMINATOR_MODE = "mlp"  # "mlp" or "svm"
DISCRIMINATOR_NAME = "dinov3"  # "dinov3" or "bioclip2_5"
BINARY_SCORE_MODE = "joint"  # "joint" (product) or "mean" (average)
FIXRES = 0.3  # 30% margin around the box before classification
DISCRIMINATOR_FILTERING = None

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    detector = FlatBugDetector(
        model_name=f"{ROOT}/models-cache/flat_bug_M.pt",
        slice_size=SLICE_SIZE,
        overlap=OVERLAP,
    )

    # Binary classifier: active only if enabled and weights are present
    if USE_BINARY_CLASSIFIER and has_binary_classifier(DISCRIMINATOR_MODE, DISCRIMINATOR_NAME):
        discriminator_mode = DISCRIMINATOR_MODE
        discriminator_name = DISCRIMINATOR_NAME
        print(f"Binary classifier: {DISCRIMINATOR_MODE} / {DISCRIMINATOR_NAME}")
    else:
        discriminator_mode = None
        discriminator_name = None

    run_inference(
        detector=detector,
        coco_json_path=COCO_JSON_PATH,
        images_dir=IMAGES_DIR,
        output_path=OUTPUT_PATH,
        description=DESCRIPTION,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        slice_size=SLICE_SIZE,
        overlap=OVERLAP,
        discriminator_mode=discriminator_mode,
        discriminator_name=discriminator_name,
        fixres=FIXRES,
        discriminator_filtering=DISCRIMINATOR_FILTERING,
        binary_score_mode=BINARY_SCORE_MODE,
    )
