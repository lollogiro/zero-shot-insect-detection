"""Shared sliced-inference helper and tile-coordinate generator.

All detectors follow the same tiled-inference pattern:
  1. Slice the image into overlapping tiles (or use the full image).
  2. Run model-specific inference on each tile.
  3. Offset tile-local coordinates back to full-image space.
  4. Fuse duplicates across tiles with NMS.
  5. Sort by score descending.

``run_sliced_inference()`` handles steps 1, 4, and 5.
Each detector implements ``process_tile()`` (steps 2 + 3).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torchvision.ops as ops
from PIL import Image


def run_sliced_inference(
    image: Image.Image,
    slice_size: Optional[int],
    overlap: float,
    process_tile: Callable,
    iou_threshold: float = 0.5,
    disable_progress: bool = True,
    **process_kwargs,
) -> List:
    """Run sliced inference with a tile-processing callback.

    Parameters
    ----------
    image : PIL.Image
        Full-resolution input image.
    slice_size : int or None
        Tile side length in pixels.  ``None`` processes the full image at once.
    overlap : float
        Fractional overlap between adjacent tiles (0 = no overlap).
    process_tile : Callable[[PIL.Image, int, int], List]
        ``process_tile(tile, x1, y1, **process_kwargs) -> list[Detection]``
        *tile* is the cropped region, *(x1, y1)* is its top-left corner in
        full-image coordinates.  Must return a list of detection-like objects
        that have ``.box`` (list[float], [x1,y1,x2,y2]) and ``.score`` (float).
    iou_threshold : float
        NMS threshold for fusing detections across tile boundaries.
    disable_progress : bool
        If True (default), suppress the tqdm progress bar.
    **process_kwargs
        Forwarded verbatim to *process_tile*.

    Returns
    -------
    list
        Sorted list of detection objects (highest score first).
    """
    from tqdm import tqdm

    width, height = image.size
    coords = (
        [(0, 0, width, height)]
        if slice_size is None
        else _get_slices(width, height, slice_size=slice_size, overlap=overlap)
    )

    raw: List = []
    for (x1, y1, x2, y2) in tqdm(
        coords, desc="Processing slices", disable=disable_progress or len(coords) == 1
    ):
        tile = image.crop((x1, y1, x2, y2))
        tile_dets = process_tile(tile, x1, y1, iou_threshold=iou_threshold, **process_kwargs)
        raw.extend(tile_dets)

    if not raw:
        return []

    # NMS across tiles: ensure float tensors (NMS requires float, not long)
    boxes_t = torch.tensor([d.box for d in raw], dtype=torch.float)
    scores_t = torch.tensor([d.score for d in raw])
    keep = ops.nms(boxes_t, scores_t, iou_threshold)

    final = [raw[k] for k in keep.tolist()]
    return sorted(final, key=lambda d: d.score, reverse=True)


def _get_slices(
    image_w: int,
    image_h: int,
    slice_size: int = 1024,
    overlap: float = 0.3,
) -> List[Tuple[int, int, int, int]]:
    """Generate overlapping tile coordinates for a large image.

    Parameters
    ----------
    image_w, image_h : int
        Full image dimensions.
    slice_size : int
        Tile side length in pixels.
    overlap : float
        Fractional overlap between adjacent tiles.

    Returns
    -------
    list of (x1, y1, x2, y2)
        Tile coordinates in pixel space.  Adjacent tiles overlap by the
        specified fraction.  Tiles near the right/bottom edge are clamped
        so that they stay within the image boundaries.
    """
    if slice_size is None:
        return [(0, 0, image_w, image_h)]

    padded_w = (
        image_w
        if image_w % slice_size == 0
        else ((image_w // slice_size) + 1) * slice_size
    )
    padded_h = (
        image_h
        if image_h % slice_size == 0
        else ((image_h // slice_size) + 1) * slice_size
    )

    step = int(slice_size * (1 - overlap))
    slices = []
    for y in range(0, padded_h - slice_size + 1, step):
        for x in range(0, padded_w - slice_size + 1, step):
            x1 = max(0, min(x, image_w - slice_size))
            y1 = max(0, min(y, image_h - slice_size))
            x2 = min(x1 + slice_size, image_w)
            y2 = min(y1 + slice_size, image_h)
            slices.append((x1, y1, x2, y2))
    return slices
