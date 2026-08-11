"""Unified inference pipeline.

Replaces the 5 duplicated standalone scripts (``grounding_dino.py``,
``sam3.py``, …) with a single ``run_inference()`` function.

Usage
-----
::

    from utils.models import GroundingDINODetector
    from utils.inference import run_inference

    detector = GroundingDINODetector()
    results = run_inference(
        detector=detector,
        coco_json_path="annotations.json",
        images_dir="images/",
        prompt=["beetle", "ant"],
        output_path="./results.json",
        description="EXP3 on Stefan dataset",
        conf_threshold=0.1,
    )
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional

from PIL import Image
from tqdm import tqdm

from utils.config import BINARY_CLASS_THRESHOLD, DEFAULT_MODEL_CACHE_DIR, INSECT_CATEGORY_ID
from utils.models import (
    BaseDetector,
    BinaryClassification,
    MLPBinaryClassificator,
    load_feature_extractor,
)

# ---------------------------------------------------------------------------
# Ready-made zero-shot prompts
# ---------------------------------------------------------------------------

PROMPTS = {
    # P1: zero context, single broad term
    "P1": ["insect"],
    # P2: more context, describes the expected background
    "P2": [
        "an insect on a natural background",
        "an insect on a flower",
        "an insect on the ground",
    ],
    # P3: taxonomic grouping, common names
    # (13 terms = 32 tokens with the "," join, the SAM3 context limit;
    #  "centipede" is left out as it is partially covered by "millipede")
    "P3": [
        "beetle", "fly", "bee", "ant", "wasp", "butterfly", "moth",
        "spider", "bug", "dragonfly", "earwig", "millipede", "pillbug",
    ],
    # Order-level scientific names (exceeds the SAM3 context limit of 32 tokens),
    # kept here for reference and for detectors with longer context
    "P4_scientific": [
        "coleoptera", "diptera", "hymenoptera", "lepidoptera", "araneae",
        "hemiptera", "odonata", "dermaptera", "myriapoda", "isopoda", "opiliones",
    ],
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference(
    detector: BaseDetector,
    coco_json_path: Optional[str],
    images_dir: str,
    prompt: Optional[List[str]] = None,
    negative_prompt: Optional[List[str]] = None,
    output_path: str = "./results.json",
    description: str = "",
    conf_threshold: float = 0.1,
    iou_threshold: float = 0.5,
    slice_size: Optional[int] = None,
    overlap: float = 0.2,
    # Binary classifier (optional)
    discriminator_mode: Optional[str] = None,
    discriminator_name: Optional[str] = None,
    fixres: Optional[float] = 0.3,
    discriminator_filtering: Optional[float] = None,
    binary_score_mode: str = "joint",
    # Additional kwargs passed to detector.predict()
    **detector_kwargs,
) -> dict:
    """Run inference with *detector* on a dataset and save COCO-format results.

    Parameters
    ----------
    detector : BaseDetector
        Initialised detector instance.
    coco_json_path : str or None
        Path to COCO-format ground-truth JSON.  If ``None``, a flat list of
        images is built from *images_dir* (every image file found).
    images_dir : str
        Directory containing the source images.
    prompt : list of str or None
        Category names / descriptions for zero-shot detectors.  Ignored by
        supervised (YOLO) detectors.
    negative_prompt : list of str or None
        Terms to suppress (only supported by some detectors).
    output_path : str
        Where to write the result JSON.
    description : str
        Human-readable experiment description (stored in output metadata).
    conf_threshold : float
        Detection confidence threshold.
    iou_threshold : float
        NMS IoU threshold for fusing detections across tiles.
    slice_size : int or None
        Tile size for sliced inference on large images.  ``None`` = full image.
    overlap : float
        Fractional overlap between adjacent tiles.
    discriminator_mode : {"svm", "mlp", None}
        Binary classifier type for post-hoc false-positive filtering.
    discriminator_name : {"bioclip2_5", "dinov3", …} or None
        Embedding model used by the binary classifier.
    fixres : float or None
        Fraction by which to enlarge each detection box before passing it to
        the binary classifier (e.g. ``0.3`` = 30% margin).
    discriminator_filtering : float or None
        If set, detections whose binary-classifier score falls below this
        threshold are dropped entirely.
    binary_score_mode : {"joint", "mean"}
        How to combine detection and binary-classifier scores: ``"joint"``
        multiplies them, ``"mean"`` averages them.
    **detector_kwargs
        Forwarded to ``detector.predict()``.

    Returns
    -------
    dict
        The assembled output dictionary (also written to *output_path*).
    """
    # ------------------------------------------------------------------
    # 1. Load image list from COCO JSON, or scan directory
    # ------------------------------------------------------------------
    coco_data = _load_coco_or_scan(coco_json_path, images_dir)
    images_list: list = coco_data["images"]
    print(f"Found {len(images_list)} images to process.")

    # ------------------------------------------------------------------
    # 2. Load binary classifier (optional)
    # ------------------------------------------------------------------
    classifier = load_binary_classifier(discriminator_mode, discriminator_name)

    # ------------------------------------------------------------------
    # 3. Inference loop
    # ------------------------------------------------------------------
    all_annotations: list = []
    ann_id = 1
    start_time = time.time()

    detector_prompt: List[str] = prompt or []
    model_has_prompt = len(detector_prompt) > 0

    for img_info in tqdm(images_list, desc=f"[{detector.model_name}] inference"):
        # Resolve image path
        img_path = _resolve_image_path(img_info, images_dir)
        if not img_path.exists():
            print(f"  Warning: image not found, skipping: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")

        # -- 3a. Run the detector -------------------------------------------
        # Determine whether to pass prompt (zero-shot models) or not (YOLO)
        if model_has_prompt:
            detections = detector.predict(
                image,
                prompt=detector_prompt,
                negative_prompt=negative_prompt,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                slice_size=slice_size,
                overlap=overlap,
                **detector_kwargs,
            )
        else:
            # YOLO-style detectors: no prompt arg
            detections = detector.predict(
                image,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                slice_size=slice_size,
                overlap=overlap,
                **detector_kwargs,
            )

        # -- 3b. Binary classifier scoring + FIXRES -----------------------
        for det in detections:
            if classifier is not None:
                # Enlarge box for context (FIXRES)
                crop_box = _expand_box(det.box, fixres, image.width, image.height) \
                    if fixres else det.box
                crop = image.crop((int(crop_box[0]), int(crop_box[1]),
                                   int(crop_box[2]), int(crop_box[3])))

                binary_result = classifier.predict(crop)
                insect_proba = binary_result.score

                # Combine scores: "joint" = product, "mean" = average
                if binary_score_mode == "joint":
                    det.score = det.score * insect_proba
                elif binary_score_mode == "mean":
                    det.score = (det.score + insect_proba) / 2
                else:
                    raise ValueError(
                        f"Unknown binary_score_mode: {binary_score_mode!r}. "
                        "Use 'joint' or 'mean'."
                    )

                # Filter by binary score
                if discriminator_filtering is not None and insect_proba < discriminator_filtering:
                    continue

            # -- 3c. Convert to COCO format --------------------------------
            coco_bbox = box_to_coco_bbox(det.box)
            record: dict = {
                "id": ann_id,
                "image_id": img_info["id"],
                "category_id": INSECT_CATEGORY_ID,
                "bbox": coco_bbox,
                "score": det.score,
                "label": det.label,
            }

            if det.mask_rle is not None:
                record["segmentation"] = mask_rle_to_coco_rle(det.mask_rle)

            all_annotations.append(record)
            ann_id += 1

    elapsed = time.time() - start_time

    # ------------------------------------------------------------------
    # 4. Assemble output
    # ------------------------------------------------------------------
    parameters: dict = {
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "slice_size": slice_size,
        "overlap": overlap,
        "prompt": detector_prompt,
        "negative_prompt": negative_prompt,
    }
    # Only record the options of features that were actually used
    if discriminator_mode is not None:
        parameters["discriminator_mode"] = discriminator_mode
        parameters["discriminator_name"] = discriminator_name
        parameters["binary_score_mode"] = binary_score_mode
        if fixres is not None:
            parameters["fixres"] = fixres
        if discriminator_filtering is not None:
            parameters["discriminator_filtering"] = discriminator_filtering

    output_data = {
        "images": images_list,
        "results": all_annotations,
        "info": {
            "description": description,
            "detector": detector.model_name,
            "num_images": len(images_list),
            "num_detections": len(all_annotations),
            "inference_time_sec": round(elapsed, 2),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": parameters,
        },
    }

    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"\nDone in {elapsed:.1f}s.  {len(all_annotations)} detections saved to: {output_path}")
    return output_data


def _classifier_weights_dir() -> Path:
    """Directory holding trained binary-classifier weights.

    Weights live in ``<MODEL_CACHE_DIR>/binary_classificator`` (default
    ``models-cache/`` at the repo root) so the lookup works from any cwd.
    """
    cache = os.environ.get("MODEL_CACHE_DIR")
    if not cache:
        cache = str(Path(__file__).resolve().parents[1] / DEFAULT_MODEL_CACHE_DIR)
    return Path(cache) / "binary_classificator"


class _SVMClassifierAdapter:
    """Adapt an SVM pipeline (trained on embeddings) to the image interface.

    ``predict(image)`` returns a :class:`BinaryClassification`, mirroring
    :class:`~utils.models.MLPBinaryClassificator`.
    """

    def __init__(self, pipeline, extractor_name: str) -> None:
        self.pipeline = pipeline
        self.extractor = load_feature_extractor(extractor_name)

    def predict(self, image: Image.Image) -> BinaryClassification:
        embedding = self.extractor(image).numpy().reshape(1, -1)
        proba = self.pipeline.predict_proba(embedding)[0][1]
        return BinaryClassification(
            # Label is informational only: "joint"/"mean" score modes combine
            # the raw probability (score) and never read this label
            label="insect" if proba >= BINARY_CLASS_THRESHOLD else "background",
            score=proba,
        )


def has_binary_classifier(mode: Optional[str], name: Optional[str]) -> bool:
    """Return ``True`` if a trained binary classifier (``mode``/``name``) is available.

    Checks ``<MODEL_CACHE_DIR>/binary_classificator`` for the matching weight
    file without loading it.
    """
    if mode is None or name is None:
        return False
    ext = ".joblib" if mode == "svm" else ".pth"
    return (_classifier_weights_dir() / f"insect_classifier_ecostack_{mode}_{name}{ext}").exists()


def load_binary_classifier(
    mode: Optional[str],
    name: Optional[str],
) -> Optional[object]:
    """Load a binary classifier (SVM or MLP) by embedding name.

    Parameters
    ----------
    mode : {"svm", "mlp", None}
        Classifier type.
    name : str or None
        Embedding model name (e.g. ``"bioclip2_5"``, ``"dinov3"``).

    Returns
    -------
    object or None
        A classifier with a ``predict(image) -> BinaryClassification`` method,
        or ``None`` if *mode* is ``None``.
    """
    if mode is None or name is None:
        return None

    weights_dir = _classifier_weights_dir()

    if mode == "svm":
        import joblib
        weight_path = weights_dir / f"insect_classifier_ecostack_svm_{name}.joblib"
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Binary classifier weights not found: {weight_path}\n"
                "Train them first with binary_classificator/train_svm.ipynb."
            )
        return _SVMClassifierAdapter(joblib.load(weight_path), name)

    elif mode == "mlp":
        weight_path = weights_dir / f"insect_classifier_ecostack_mlp_{name}.pth"
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Binary classifier weights not found: {weight_path}\n"
                "Train them first with binary_classificator/train_mlp.ipynb."
            )
        return MLPBinaryClassificator(
            extractor_name=name, pth_weights_path=str(weight_path)
        )

    else:
        raise ValueError(f"Unknown binary classifier mode: {mode!r}. Use 'svm' or 'mlp'.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_coco_or_scan(
    coco_json_path: Optional[str],
    images_dir: str,
) -> dict:
    """Load a COCO JSON or, if *coco_json_path* is ``None``, scan *images_dir*."""
    if coco_json_path is not None:
        with open(coco_json_path) as f:
            data = json.load(f)
        images = data.get("images", [])
        # Filter to images that actually exist on disk
        existing = []
        for img in images:
            candidate = _resolve_image_path(img, images_dir)
            if candidate.exists():
                existing.append(img)
        if len(existing) < len(images):
            print(f"  Filtered to {len(existing)}/{len(images)} images that exist on disk.")
        return {"images": existing, "categories": data.get("categories", [])}
    else:
        # Scan directory for images
        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        image_files = sorted(
            p for p in Path(images_dir).iterdir() if p.suffix in exts
        )
        images = [
            {"id": i + 1, "file_name": p.name, "width": 0, "height": 0}
            for i, p in enumerate(image_files)
        ]
        return {"images": images}


def _resolve_image_path(img_info: dict, images_dir: str) -> Path:
    """Build the full filesystem path for an image, trying several strategies."""
    candidates = [
        Path(images_dir) / img_info["file_name"],
        Path(img_info.get("file_name", "")),
    ]
    # Also try with the raw id as filename
    if not candidates[0].exists():
        candidates.append(Path(images_dir) / str(img_info.get("id", "")))

    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # return first candidate even if missing (caller handles)


def _expand_box(
    box: List[float],
    fraction: Optional[float],
    img_w: int,
    img_h: int,
) -> List[float]:
    """Enlarge a bounding box by *fraction* on each side, clamped to image edges."""
    if fraction is None:
        return box
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = w * fraction, h * fraction
    return [
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(img_w, x2 + dx),
        min(img_h, y2 + dy),
    ]


def box_to_coco_bbox(box: List[float]) -> List[float]:
    """Convert ``[x1, y1, x2, y2]`` to COCO ``[x, y, w, h]``."""
    return [round(box[0], 1), round(box[1], 1),
            round(box[2] - box[0], 1), round(box[3] - box[1], 1)]


def mask_rle_to_coco_rle(mask_rle: dict) -> dict:
    """Convert an RLE dict with ``bytes`` counts into a JSON-serializable dict."""
    rle = dict(mask_rle)
    if isinstance(rle.get("counts"), bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle
