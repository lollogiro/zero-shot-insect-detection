"""Visualisation utilities for detection results.

All public functions accept either a ``list`` of
:class:`~utils.models.Detection` objects or a path/results dict (via
:func:`~utils.eval_utils._load_predictions`).
"""

from __future__ import annotations

from typing import List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.eval_utils import _load_predictions
from utils.models import Detection, rle_to_mask


def plot_results(
    img_path: str,
    results: Union[str, dict, list, List[Detection]],
    image: Optional[Image.Image] = None,
    figsize: tuple = (12, 12),
) -> None:
    """Overlay bounding boxes on an image and display.

    Parameters
    ----------
    img_path : str
        Path to the source image.
    results : str or dict or list
        Detections: either a JSON path, a dict wrapping an ``annotations`` /
        ``results`` key, a raw list, or a ``list[Detection]``.
    image : PIL.Image or None
        Pre-loaded image (avoids a second PIL open).
    figsize : tuple
        Figure dimensions ``(width, height)`` in inches.
    """
    if image is None:
        img = Image.open(img_path).convert("RGB")
    else:
        img = image

    dets = _normalise_detection_list(results)

    plt.figure(figsize=figsize)
    plt.imshow(img)

    for det in dets:
        x1, y1, x2, y2 = det.box
        plt.gca().add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                edgecolor="red", facecolor="none", linewidth=2,
            )
        )
        plt.text(x1, y1 - 10, f"({det.score:.2f})", color="red", fontsize=12)

    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_crop_zoomed(
    img_path: str,
    results: Union[str, dict, list, List[Detection]],
    image: Optional[Image.Image] = None,
    cols: int = 3,
) -> None:
    """Show individual detection crops in a grid.

    Parameters
    ----------
    img_path : str
        Path to the source image.
    results : str or dict or list
        Detections (see :func:`plot_results`).
    image : PIL.Image or None
        Pre-loaded image.
    cols : int
        Number of columns in the grid.
    """
    if image is None:
        img = Image.open(img_path).convert("RGB")
    else:
        img = image

    dets = _normalise_detection_list(results)
    n = len(dets)

    if n == 0:
        print("No detections to display.")
        return

    rows = (n + cols - 1) // cols
    _, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i, det in enumerate(dets):
        x1, y1, x2, y2 = det.box
        crop = img.crop((x1, y1, x2, y2))
        axes[i].imshow(crop)
        axes[i].set_title(f"({det.score:.2f})")
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_results_masks(
    img_path: str,
    results: Union[str, dict, list, List[Detection]],
    figsize: tuple = (12, 12),
) -> None:
    """Overlay segmentation masks and bounding boxes on an image.

    Parameters
    ----------
    img_path : str
        Path to the source image.
    results : str or dict or list
        Detections (see :func:`plot_results`).
    figsize : tuple
        Figure dimensions in inches.
    """
    img = np.array(Image.open(img_path).convert("RGB"))
    overlay = img.copy()
    dets = _normalise_detection_list(results)

    colors = plt.cm.tab10.colors
    for i, det in enumerate(dets):
        color = tuple(int(c * 255) for c in colors[i % len(colors)])

        if det.mask_rle is not None:
            mask = rle_to_mask(det.mask_rle).astype(bool)
            overlay[mask] = (
                np.array(overlay[mask], dtype=np.float32) * 0.5
                + np.array(color, dtype=np.float32) * 0.5
            ).astype(np.uint8)

        x1, y1, x2, y2 = [int(v) for v in det.box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            overlay, f"({det.score:.2f})",
            (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

    plt.figure(figsize=figsize)
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_masks_zoomed(
    img_path: str,
    results: Union[str, dict, list, List[Detection]],
    figsize: tuple = (12, 6),
) -> None:
    """Show individual mask crops in a grid.

    Parameters
    ----------
    img_path : str
        Path to the source image.
    results : str or dict or list
        Detections (see :func:`plot_results`).
    figsize : tuple
        Base figure dimensions in inches (height is scaled by row count).
    """
    img = np.array(Image.open(img_path).convert("RGB"))
    dets = _normalise_detection_list(results)

    num_detections = len(dets)
    if num_detections == 0:
        return

    num_cols = min(2, num_detections)
    num_rows = (num_detections + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize[0], figsize[1] * num_rows))
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1 or num_cols == 1:
        axes = axes.reshape(num_rows, num_cols)

    for idx, det in enumerate(dets):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col]

        if det.mask_rle is not None:
            mask = rle_to_mask(det.mask_rle).astype(bool)

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if rows.any() and cols.any():
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]

                padding = 20
                y_min = max(0, y_min - padding)
                y_max = min(mask.shape[0], y_max + padding)
                x_min = max(0, x_min - padding)
                x_max = min(mask.shape[1], x_max + padding)

                cropped = np.ones((y_max - y_min, x_max - x_min, 3), dtype=np.uint8) * 255
                mask_crop = mask[y_min:y_max, x_min:x_max]
                img_crop = img[y_min:y_max, x_min:x_max]
                cropped[mask_crop] = img_crop[mask_crop]

                ax.imshow(cropped)
                ax.set_title(f"({det.score:.2f})")

        ax.axis("off")

    for idx in range(num_detections, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_detection_list(
    results: Union[str, dict, list, List[Detection]],
) -> List[Detection]:
    """Convert results (path / dict / raw list / Detection list) to ``list[Detection]``."""
    if results is None or (isinstance(results, list) and len(results) == 0):
        return []

    # Already a list of Detection objects
    if isinstance(results, list) and results and isinstance(results[0], Detection):
        return results

    # Raw dict / path: load from COCO-format JSON
    raw = _load_predictions(results)  # type: ignore[arg-type]
    out: List[Detection] = []
    for r in raw:
        b = r["bbox"]  # COCO format: [x, y, w, h]
        out.append(Detection(
            box=[b[0], b[1], b[0] + b[2], b[1] + b[3]],  # → xyxy
            score=r.get("score", 1.0),
            label=r.get("label", ""),
        ))
    return out
