import json
from collections import defaultdict
from functools import lru_cache
import os
from pathlib import Path
import random
from typing import Dict, Union
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util
import copy

def plot_dataset_distribution(annotations_file):
    """
    Plot the distribution of categories in a dataset from a COCO-formatted annotations file.
    This function reads a JSON annotations file, counts the occurrences of each category,
    and displays a bar chart showing the distribution. Categories are sorted in descending
    order by count, and the exact count is displayed on top of each bar.
    Args:
        annotations_file (str): Path to the COCO-formatted JSON annotations file.
            Expected to contain 'annotations' and 'categories' keys.
    Returns:
        pd.DataFrame: A DataFrame containing the category distribution with columns:
            - 'category' (str): Category name
            - 'count' (int): Number of annotations for the category
            Sorted by count in descending order.
    """
    with open(annotations_file, 'r') as f:
        data = json.load(f)
        
    counts = defaultdict(int)
    for ann in data['annotations']:
        counts[ann['category_id']] += 1
        
    names = {cat['id']: cat['name'] for cat in data['categories']}
    df = pd.DataFrame([
        {'category': names.get(cid, str(cid)), 'count': counts[cid]} 
        for cid in names
    ]).sort_values('count', ascending=False)
    
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df, x='category', y='count', palette='Blues_d', hue='category', legend=False)
    for i, v in enumerate(df['count']):
        ax.text(i, v, str(v), ha='center', va='bottom')
        
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return df


def _load_predictions(preds: Union[str, dict, list]) -> list:
    """Normalise *preds* (str path / dict / list) into a list of prediction dicts.

    Many functions in this module accept predictions in three forms: a JSON
    file path, a dict wrapping an ``annotations`` or ``results`` key, or a raw
    list.  This helper centralises the conversion.
    """
    if isinstance(preds, str):
        with open(preds, 'r') as f:
            preds = json.load(f)
    if isinstance(preds, dict):
        preds = preds.get("annotations", preds.get("results", []))
    return preds


def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) metric between two bounding boxes.
    
    The IoU is computed as the ratio of the intersection area to the union area
    of two bounding boxes. This metric is commonly used in object detection to
    measure the overlap between predicted and ground truth boxes.
    
    Args:
        boxA (tuple or list): First bounding box in format (x, y, width, height)
                             where (x, y) is the top-left corner.
        boxB (tuple or list): Second bounding box in format (x, y, width, height)
                             where (x, y) is the top-left corner.
    
    Returns:
        float: IoU score between 0 and 1, where:
               - 0 means no overlap between boxes
               - 1 means perfect overlap (boxes are identical)
    """
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    return interArea / float(boxA[2] * boxA[3] + boxB[2] * boxB[3] - interArea + 1e-6)


def evaluate_detections(coco_gt, preds, tested_img_ids, iou_threshold=0.1):
    """Match predictions to ground truth and return scored results.

    This is a convenience wrapper around :func:`match_predictions_to_gt` that
    accepts a COCO ground-truth object and a raw prediction list.

    Parameters
    ----------
    coco_gt : COCO
        Ground-truth annotations.
    preds : list
        Prediction dicts, each with ``"image_id"``, ``"bbox"`` (xywh), and
        optionally ``"score"``.
    tested_img_ids : list
        Image IDs to consider.
    iou_threshold : float
        Minimum IoU for a match.

    Returns
    -------
    tuple
        ``(evaluated_preds, total_gt)``; see docstring above.
    """
    # Build gt_by_img and cat mapping
    gt_by_img: Dict[int, list] = defaultdict(list)
    cat_for_gt: dict = {}
    for img_id in tested_img_ids:
        for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id, iscrowd=False)):
            gt_by_img[img_id].append(ann["bbox"])
            cat_for_gt[(img_id, tuple(ann["bbox"]))] = ann["category_id"]

    # Build detections list (img_id, bbox, score) sorted by score descending
    detections: list = []
    for p in preds:
        if p["image_id"] in tested_img_ids:
            detections.append((p["image_id"], p["bbox"], p.get("score", 0)))
    detections.sort(key=lambda x: x[2], reverse=True)

    tp, fp, scores = match_predictions_to_gt(detections, gt_by_img, iou_threshold)

    total_gt = sum(len(v) for v in gt_by_img.values())
    evaluated_preds = [
        {"score": float(s), "is_tp": bool(t), "cat_id": None}
        for t, s in zip(tp, scores)
    ]
    return evaluated_preds, total_gt


def detection_evaluation(coco_gt, preds, iou_type='bbox', images_to_remove: list = None):
    """
    Evaluate object detection predictions against ground truth annotations using COCO metrics.
    This function computes various Average Precision (AP) and Average Recall (AR) metrics
    at different Intersection over Union (IoU) thresholds. It supports filtering images
    based on annotation criteria and handles both bounding box and segmentation evaluation.
    Args:
        coco_gt (COCO): COCO ground truth object containing the annotated dataset.
        preds (str or list or dict): Detection predictions. Can be:
            - A file path (str) to a JSON file containing predictions
            - A list of annotations
            - A dict with 'annotations' or 'results' key containing the predictions
        iou_type (str, optional): Type of evaluation metric. Defaults to 'bbox'.
            Supported values: 'bbox' for bounding box, 'segm' for segmentation.
        images_to_remove (list, optional): List of (key, value) tuples to filter out images
            from evaluation. Images with annotations matching the criteria will be excluded.
            Defaults to None (no filtering).
    Returns:
        COCOeval: Evaluation object containing:
            - Standard COCO metrics (precision, recall)
            - extra_metrics (dict): Additional computed metrics including:
                - mAP@10, mAP@30, mAP@50: Mean AP at specific IoU thresholds
                - mAR@10, mAR@30, mAR@50: Mean AR at specific IoU thresholds
                - mAP@10:95: Mean AP across all IoU thresholds (0.1 to 0.95)
                - mAP@50:95: Mean AP for IoU thresholds >= 0.5
                - mAR@10:95: Mean AR across all IoU thresholds
                - mAR@50:95: Mean AR for IoU thresholds >= 0.5
    """
    preds = _load_predictions(preds)
    anns = preds
    img_ids = coco_gt.getImgIds()

    if images_to_remove is not None:
        for key, value in images_to_remove:
            img_ids_to_remove = set()
            for ann in coco_gt.anns.values():
                if ann.get(key) == value:
                    img_ids_to_remove.add(ann['image_id'])
            img_ids = [img_id for img_id in img_ids if img_id not in img_ids_to_remove]
            print(f"Removed {len(img_ids_to_remove)} images based on the criteria {key}={value}")

    dummy_gt_anns = copy.deepcopy(coco_gt.dataset.get('annotations', coco_gt.dataset.get('results', [])))
    for ann in dummy_gt_anns: 
        ann['category_id'] = 1
        
    eval_coco_gt = COCO()
    eval_coco_gt.dataset = {
        'images': coco_gt.dataset.get('images', []),
        'categories': [{'id': 1, 'name': 'insect'}],
        'annotations': dummy_gt_anns
    }
    eval_coco_gt.createIndex()

    coco_dt = eval_coco_gt.loadRes(anns)
    for ann in coco_dt.anns.values(): 
        ann['category_id'] = 1

    if iou_type == 'segm':
        for ann in coco_dt.anns.values():
            segm = ann.get('segmentation')
            if isinstance(segm, dict) and isinstance(segm.get('counts'), list):
                ann['segmentation'] = mask_util.frPyObjects(segm, segm['size'][0], segm['size'][1])

    full_thrs = np.linspace(0.1, 0.95, 18)
    eval_obj = COCOeval(eval_coco_gt, coco_dt, iou_type)
    eval_obj.params.imgIds = img_ids
    eval_obj.params.catIds = [1]
    eval_obj.params.iouThrs = full_thrs
    eval_obj.evaluate()
    eval_obj.accumulate()
    
    def get_ap(ev, thr_val):
        idx = np.where(np.isclose(ev.params.iouThrs, thr_val))[0]
        if len(idx) == 0: return 0.0
        p = ev.eval['precision'][idx[0], :, :, 0, 2]
        valid = p[p > -1]
        return np.mean(valid) if len(valid) > 0 else 0.0

    def get_ar(ev, thr_val):
        idx = np.where(np.isclose(ev.params.iouThrs, thr_val))[0]
        if len(idx) == 0: return 0.0
        r = ev.eval['recall'][idx[0], :, 0, 2]
        valid = r[r > -1]
        return np.mean(valid) if len(valid) > 0 else 0.0

    eval_obj.extra_metrics = {}
    
    for t in [0.1, 0.3, 0.5]:
        eval_obj.extra_metrics[f'mAP@{int(t*100)}'] = get_ap(eval_obj, t)
        eval_obj.extra_metrics[f'mAR@{int(t*100)}'] = get_ar(eval_obj, t)
        
    eval_obj.extra_metrics['mAP@10:95'] = np.mean([get_ap(eval_obj, t) for t in full_thrs])
    eval_obj.extra_metrics['mAP@50:95'] = np.mean([get_ap(eval_obj, t) for t in full_thrs if t >= 0.5])
    eval_obj.extra_metrics['mAR@10:95'] = np.mean([get_ar(eval_obj, t) for t in full_thrs])
    eval_obj.extra_metrics['mAR@50:95'] = np.mean([get_ar(eval_obj, t) for t in full_thrs if t >= 0.5])

    return eval_obj


def plot_coco_performance(coco_eval, test_name="default"):
    """
    Plot COCO evaluation metrics for precision (AP) and recall (AR).
    This function visualizes the mean Average Precision (mAP) and mean Average Recall (mAR)
    metrics from COCO evaluation results using a two-subplot bar chart layout. The top subplot
    displays precision metrics using a 'magma' color palette, while the bottom subplot displays
    recall metrics using a 'viridis' color palette. Values are annotated on top of each bar.
    Args:
        coco_eval: COCO evaluation object containing extra_metrics attribute with mAP and mAR metrics.
        test_name (str, optional): Name of the test/dataset to display in the plot titles. 
                                   Defaults to "default".
    Returns:
        None. Displays the plot.
    """
    m = coco_eval.extra_metrics
    
    map_data = {k: v for k, v in m.items() if k.startswith('mAP')}
    mar_data = {k: v for k, v in m.items() if k.startswith('mAR')}
    
    df_map = pd.DataFrame({'Metric': list(map_data.keys()), 'Value': list(map_data.values())})
    df_mar = pd.DataFrame({'Metric': list(mar_data.keys()), 'Value': list(mar_data.values())})
    
    _, axes = plt.subplots(2, 1, figsize=(15, 10), sharey=True)
    
    sns.barplot(x='Metric', y='Value', data=df_map, palette='magma', hue='Metric', legend=False, ax=axes[0])
    axes[0].set_title(f'Precision (AP) - {test_name}', fontsize=14, fontweight='bold')
    for i, v in enumerate(df_map['Value']):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('Score')
    axes[0].set_xlabel('')
    
    sns.barplot(x='Metric', y='Value', data=df_mar, palette='viridis', hue='Metric', legend=False, ax=axes[1])
    axes[1].set_title(f'Recall (AR) - {test_name}', fontsize=14, fontweight='bold')
    for i, v in enumerate(df_mar['Value']):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylabel('Score')
    axes[1].set_xlabel('')
    
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curves(coco_eval, thresholds=[0.1, 0.5, 0.75]):
    """
    Plot precision-recall curves for different Intersection over Union (IoU) thresholds.
    This function visualizes the precision-recall relationship at multiple IoU threshold
    levels using COCO evaluation metrics. It creates a line plot where each line represents
    the precision-recall curve for a specific IoU threshold.
    Args:
        coco_eval: A COCO evaluation object containing computed metrics and precision-recall data.
        thresholds (list, optional): List of IoU thresholds to plot. Defaults to [0.1, 0.5, 0.75].
    Returns:
        None. Displays the plot.
    """
    plt.figure(figsize=(10, 7))
    recall_steps = np.linspace(0, 1, 101)
    
    for thr in thresholds:
        idx = np.where(np.isclose(coco_eval.params.iouThrs, thr))[0]
        if len(idx) > 0:
            pr_data = coco_eval.eval['precision'][idx[0], :, :, 0, 2]
            masked_pr = np.ma.masked_where(pr_data < 0, pr_data)
            mean_pr = masked_pr.mean(axis=1).filled(0)
            
            plt.plot(recall_steps, mean_pr, label=f'IoU={thr:.2f}', linewidth=2)
            plt.fill_between(recall_steps, mean_pr, alpha=0.05)
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.0)
    plt.show()


def plot_performance(coco_eval, test_name="default"):
    """
    Visualize COCO evaluation metrics and precision-recall curves for model performance analysis.

    This function generates two types of plots:
    1. Overall COCO performance metrics (mAP, mAP50, mAP75, etc.)
    2. Precision-recall curves at multiple IoU thresholds

    Args:
        coco_eval (COCOeval): COCO evaluation object containing computed metrics.
        test_name (str, optional): Name identifier for the test/model being evaluated.
            Used in plot titles and file naming. Defaults to "default".

    Returns:
        None
    """
    plot_coco_performance(coco_eval, test_name)
    plot_precision_recall_curves(coco_eval, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5])


def match_predictions_to_gt(detections, gt_by_img, iou_threshold=0.1):
    matched_gt = defaultdict(set)
    tp, fp, scores = [], [], []

    for img_id, pb, score in detections:
        gt_boxes = gt_by_img.get(img_id, [])
        best_iou = 0
        best_gt = -1

        for g_idx, gb in enumerate(gt_boxes):
            # skip if this gt box is already matched to a previous prediction
            if g_idx in matched_gt[img_id]:
                continue
            iou = calculate_iou(gb, pb)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt = g_idx

        if best_gt >= 0:
            matched_gt[img_id].add(best_gt)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
            
        scores.append(score)

    return np.array(tp), np.array(fp), np.array(scores)


@lru_cache(maxsize=None)
def normalize_id(raw_id):
    """
    Normalize a raw identifier by extracting the filename and removing its extension.
    
    This function takes a raw identifier (typically a file path), extracts the base filename,
    removes the file extension, and attempts to convert it to an integer. If conversion fails,
    it returns the cleaned string as-is.
    
    Args:
        raw_id: A file path or identifier string to normalize.
    
    Returns:
        int: The identifier as an integer if the cleaned filename is numeric.
        str: The cleaned filename as a string if it cannot be converted to an integer.
    """
    clean = os.path.basename(str(raw_id))
    clean = os.path.splitext(clean)[0]
    try:
        return int(clean)
    except ValueError:
        return clean


def ensemble_coco(json_paths, output_path, iou_thr=0.5, num_models=3):
    """Fuse predictions from multiple JSON result files into one ensemble result.

    Clusters detections across models by image and IoU, averages their boxes
    and scores (penalised by ``num_models``), and writes a fused COCO-format
    JSON to *output_path*.

    Parameters
    ----------
    json_paths : list of str
        Paths to model-specific result JSON files.
    output_path : str
        Where to write the fused result JSON.
    iou_thr : float
        IoU threshold for clustering detections of the same object.
    num_models : int
        Denominator for score averaging (typically ``len(json_paths)``).
    """
    preds_by_img = defaultdict(list)
    images_dict = {}

    for path in json_paths:
        with open(path, 'r') as f:
            data = json.load(f)

        imgs = data.get('images', []) if isinstance(data, dict) else []
        for img in imgs:
            images_dict[str(img.get('id', img.get('file_name')))] = img

        anns = _load_predictions(data)
        for ann in anns:
            preds_by_img[str(ann['image_id'])].append(ann)

    fused_annotations = []
    ann_id_counter = 1

    for _, boxes in preds_by_img.items():
        boxes.sort(key=lambda x: x['score'], reverse=True)

        while boxes:
            seed_box = boxes.pop(0)
            cluster = [seed_box]

            remaining_boxes = []
            for box in boxes:
                if calculate_iou(seed_box['bbox'], box['bbox']) >= iou_thr:
                    cluster.append(box)
                else:
                    remaining_boxes.append(box)

            avg_bbox = np.mean([b['bbox'] for b in cluster], axis=0).tolist()
            fused_score = sum(b['score'] for b in cluster) / num_models

            fused_ann = {
                "id": ann_id_counter,
                "image_id": seed_box['image_id'],
                "category_id": seed_box.get('category_id', 1),
                "bbox": [round(coord, 1) for coord in avg_bbox],
                "score": float(fused_score),
                "label": seed_box.get('label', 'ensemble_fused'),
            }
            if 'segmentation' in seed_box:
                fused_ann['segmentation'] = seed_box['segmentation']

            fused_annotations.append(fused_ann)
            ann_id_counter += 1
            boxes = remaining_boxes

    if not images_dict:
        unique_ids = {ann['image_id'] for ann in fused_annotations}
        images_list = [{"id": uid, "file_name": str(uid)} for uid in unique_ids]
    else:
        images_list = list(images_dict.values())

    final_coco_format = {"images": images_list, "results": fused_annotations}
    with open(output_path, 'w') as f:
        json.dump(final_coco_format, f, indent=4)


def plot_f1_confidence_curve(coco_gt, preds, tested_img_ids, iou_threshold=0.5, conf=None):
    """
    Plot F1-score as a function of confidence threshold and find optimal confidence value.

    Uses greedy IoU matching (via match_predictions_to_gt) and cumulative TP/FP to
    compute micro-F1 at every detection confidence threshold, the same approach as
    the original implementation (pre-refactoring). Pre-dates the adoption of COCOeval
    for this specific plot because COCOeval's interpolated precision gives different
    (less intuitive) F1 values for threshold tuning.

    Args:
        coco_gt (COCO): COCO ground truth object containing dataset with annotations.
        preds (str or list or dict): Predictions data. Can be:
            - Path to JSON file containing predictions
            - Dictionary with 'annotations' or 'results' key
            - List of prediction dictionaries
        tested_img_ids (list or set): Image IDs to evaluate on (filters both gt and predictions).
        iou_threshold (float, optional): IoU threshold for matching predictions to ground truth.
            Defaults to 0.5.
        conf (float, optional): Fixed confidence threshold to evaluate. If provided, uses this
            threshold instead of finding the best one. Defaults to None.
    Returns:
        tuple: A tuple containing:
            - best_conf (float): Optimal confidence threshold that maximizes F1-score
              (or the provided confidence if conf parameter is specified).
            - best_f1 (float): Maximum F1-score achieved (or F1 at given confidence).
            Returns (None, None) if no predictions or ground truths are found.
    """
    preds = _load_predictions(preds)

    tested_img_ids_norm = {normalize_id(i) for i in tested_img_ids}

    gt_by_img = defaultdict(list)
    total_gt = 0
    for ann in coco_gt.dataset.get('annotations', []):
        ann_img_id = normalize_id(ann.get('image_id'))
        if ann_img_id in tested_img_ids_norm and ann.get('iscrowd', 0) == 0:
            gt_by_img[ann_img_id].append(ann['bbox'])
            total_gt += 1

    preds_by_img = defaultdict(list)
    for p in preds:
        p_img_id = normalize_id(p.get("image_id"))
        if p_img_id in tested_img_ids_norm:
            preds_by_img[p_img_id].append(p)

    detections = []
    for img_id in tested_img_ids_norm:
        for p in preds_by_img.get(img_id, []):
            detections.append((img_id, p["bbox"], p.get("score", 0)))

    detections.sort(key=lambda x: x[2], reverse=True)

    tp, fp, scores = match_predictions_to_gt(detections, gt_by_img, iou_threshold)

    if total_gt == 0 or len(tp) == 0:
        print("No predictions or ground truths found.")
        return None, None

    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / total_gt

    f1 = np.zeros_like(precision)
    denom = precision + recall
    valid = denom > 0
    f1[valid] = 2 * precision[valid] * recall[valid] / denom[valid]

    if conf is not None:
        valid_indices = np.where(scores >= conf)[0]
        best_idx = valid_indices[-1] if len(valid_indices) > 0 else 0
    else:
        best_idx = np.argmax(f1)

    best_f1, best_conf = f1[best_idx], scores[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(scores, f1, linewidth=2)

    label_text = f"Given Conf = {best_conf:.3f} (F1={best_f1:.3f})" if conf else f"Best F1 Conf = {best_conf:.3f} (F1={best_f1:.3f})"

    plt.axvline(best_conf, linestyle="--", color="red", label=label_text)
    plt.xlabel("Confidence threshold")
    plt.ylabel("F1-score")
    plt.title(f"F1 vs Confidence (IoU={iou_threshold})\nTarget F1: {best_f1:.3f} | P: {precision[best_idx]:.3f} | R: {recall[best_idx]:.3f}")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return best_conf, best_f1

def plot_fppi_vs_recall(coco_gt, preds, tested_img_ids, iou_threshold=0.5, max_fppi=10, best_conf=None):
    """
    Plot a FROC (Free-Response Operating Characteristic) curve showing recall vs FPPI.
    This function evaluates object detection predictions against ground truth annotations
    and generates a plot of recall versus false positives per image (FPPI) at a specified
    IoU threshold. It can optionally highlight an optimal confidence threshold on the curve.
    Args:
        coco_gt (COCO): COCO ground truth object containing annotations and image metadata.
        preds (str or list or dict): Object detection predictions. Can be:
            - A file path (str) to a JSON file containing predictions
            - A list of prediction dictionaries
            - A dictionary with 'annotations' or 'results' key containing predictions
        tested_img_ids (list): List of image IDs to evaluate on.
        iou_threshold (float, optional): IoU threshold for matching predictions to ground truth.
            Defaults to 0.5.
        max_fppi (int, optional): Maximum FPPI value to display on x-axis. Defaults to 10.
        best_conf (float, optional): Confidence threshold to highlight on the plot.
            If provided, marks the optimal operating point. Defaults to None.
    Returns:
        None: Displays the plot.
    """
    preds = _load_predictions(preds)

    tested_img_ids_norm = {normalize_id(i) for i in tested_img_ids}

    gt_by_img = defaultdict(list)
    total_gt = 0
    for ann in coco_gt.dataset.get('annotations', []):
        ann_img_id = normalize_id(ann.get('image_id'))
        if ann_img_id in tested_img_ids_norm and ann.get('iscrowd', 0) == 0:
            gt_by_img[ann_img_id].append(ann['bbox'])
            total_gt += 1

    preds_by_img = defaultdict(list)
    for p in preds:
        p_img_id = normalize_id(p.get("image_id"))
        if p_img_id in tested_img_ids_norm:
            preds_by_img[p_img_id].append(p)

    detections = []
    for img_id in tested_img_ids_norm:
        for p in preds_by_img.get(img_id, []):
            detections.append((img_id, p["bbox"], p.get("score", 0)))
            
    detections.sort(key=lambda x: x[2], reverse=True)

    tp, fp, scores = match_predictions_to_gt(detections, gt_by_img, iou_threshold)

    if total_gt == 0 or len(tp) == 0:
        print("No predictions or ground truths found.")
        return

    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
    recalls = cum_tp / total_gt
    fppi = cum_fp / len(tested_img_ids_norm)

    plt.figure(figsize=(10,6))
    plt.plot(fppi, recalls, linewidth=2, label="FROC Curve")

    if best_conf is not None:
        valid_indices = np.where(scores >= best_conf)[0]
        if len(valid_indices) > 0:
            idx = valid_indices[-1]
            plt.axvline(x=fppi[idx], color='red', linestyle='--', alpha=0.7, 
                        label=f'Optimal Conf ({best_conf:.2f})\nFPPI: {fppi[idx]:.2f}')
            plt.plot(fppi[idx], recalls[idx], marker='o', color='red', markersize=6)

    plt.title(f"Recall vs FPPI (IoU={iou_threshold})")
    plt.xlabel("False positives per image (FPPI)")
    plt.ylabel("Recall")
    plt.xlim(0, max_fppi)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.show()

def class_aware_recall_report(coco_gt, preds, tested_img_ids, iou_threshold=0.1, conf_threshold=None):
    """
    Generate a class-aware recall report for object detection predictions.
    This function computes recall metrics for each object class by comparing
    predicted detections against ground truth annotations. It filters results
    based on IoU threshold and optional confidence threshold, then produces
    a sorted report showing recall performance per class.
    Args:
        coco_gt (COCO): COCO dataset object containing ground truth annotations and category information.
        preds (str, dict, or list): Predictions data. Can be:
            - str: Path to a JSON file containing predictions
            - dict: Dictionary with 'annotations' or 'results' key
            - list: List of prediction dictionaries
        tested_img_ids (list or set): Collection of image IDs to evaluate.
        iou_threshold (float, optional): Intersection over Union threshold for matching predictions to ground truth.
            Defaults to 0.1.
        conf_threshold (float, optional): Confidence score threshold for filtering predictions.
            Predictions with score below this threshold are excluded. Defaults to None (no filtering).
    Returns:
        None: Prints a formatted table with class-aware recall statistics to stdout.
              Table includes columns: Class, GTs (ground truth count), TPs (true positives), Recall.
    """
    preds = _load_predictions(preds)

    tested_img_ids_norm = {normalize_id(i) for i in tested_img_ids}

    gt_by_img_and_class = defaultdict(lambda: defaultdict(list))
    gt_counts_by_class = defaultdict(int)

    for ann in coco_gt.dataset.get('annotations', []):
        ann_img_id = normalize_id(ann.get('image_id'))
        if ann_img_id in tested_img_ids_norm and ann.get('iscrowd', 0) == 0:
            cid = ann['category_id']
            gt_by_img_and_class[cid][ann_img_id].append(ann['bbox'])
            gt_counts_by_class[cid] += 1

    preds_by_img = defaultdict(list)
    
    for p in preds:
        p_img_id = normalize_id(p.get("image_id"))
        if p_img_id in tested_img_ids_norm:
            if conf_threshold is None or p.get("score", 0) >= conf_threshold:
                preds_by_img[p_img_id].append(p)

    results = []

    for cid, cat in coco_gt.cats.items():
        total_gt = gt_counts_by_class.get(cid, 0)
        if total_gt == 0:
            continue

        gt_by_img = gt_by_img_and_class[cid]
        
        detections = []
        for img_id in tested_img_ids_norm:
            img_preds = preds_by_img.get(img_id, [])
            img_preds = sorted(img_preds, key=lambda x: x.get("score", 0), reverse=True)
            for p in img_preds:
                detections.append((img_id, p["bbox"], p.get("score", 0)))

        detections.sort(key=lambda x: x[2], reverse=True)
        
        tp, _, _ = match_predictions_to_gt(detections, gt_by_img, iou_threshold)

        total_hits = np.sum(tp) if len(tp) > 0 else 0
        recall = total_hits / total_gt

        results.append({
            "Class": cat["name"],
            "GTs": total_gt,
            "TPs": int(total_hits),
            "Recall": round(recall, 2)
        })

    df = pd.DataFrame(results).sort_values("GTs", ascending=False)
    print(f"\nClass-Aware recall report (Conf Threshold: {f'{conf_threshold:.3f}' if conf_threshold is not None else 'None'})")
    print(df.to_string(index=False))
    

def visualize_json_results(coco_gt, preds, images_dir, conf_threshold=None, image_id=None, plot_masks=True):
    """
    Visualize COCO dataset predictions and ground truth annotations on images.
    This function loads a COCO dataset and predictions, displays an image with both
    ground truth bounding boxes (in blue) and predicted bounding boxes (in red),
    and optionally shows cropped individual detections with their confidence scores
    and IoU values relative to ground truth boxes.
    Args:
        coco_gt (COCO): COCO dataset object containing ground truth annotations.
        preds (str, dict, or list): Predictions data. Can be:
            - Path to a JSON file containing predictions
            - Dictionary with 'annotations' or 'results' key
            - List of prediction dictionaries
        images_dir (str or Path): Directory path containing the image files.
        conf_threshold (float, optional): Confidence score threshold for filtering predictions.
            Predictions below this threshold are not displayed. Defaults to None (no filtering).
        image_id (int or str, optional): ID of the image to visualize. If None, a random
            image from the dataset is selected. Defaults to None.
        plot_masks (bool, optional): Whether to extract and display individual cropped
            detections below the main image. Defaults to True.
    Returns:
        list: List of tuples containing:
            - cropped_insect (np.ndarray): Cropped region around the detection
            - score (float): Prediction confidence score
            - best_iou (float): Best IoU value with ground truth boxes
    """
    preds = _load_predictions(preds)
    
    if image_id is None:
        # take one random image from the dataset
        image_id = coco_gt.getImgIds()[random.randint(0, len(coco_gt.getImgIds()) - 1)]

    img_preds = [p for p in preds if str(p.get('image_id')) == str(image_id)]
    
    try:
        img_info = coco_gt.loadImgs([int(image_id)])[0]
    except ValueError:
        img_info = coco_gt.loadImgs([str(image_id)])[0]
        
    image_path = Path(images_dir) / img_info['file_name']
    image = Image.open(image_path).convert("RGB")
    
    gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_info['id']])
    gt_anns = coco_gt.loadAnns(gt_ann_ids)

    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    for ann in gt_anns:
        x, y, w, h = ann['bbox']
        draw.rectangle([x, y, x + w, y + h], outline="blue", width=4)

    for p in img_preds:
        x, y, w, h = p['bbox']
        score = p.get('score', 1.0)
        if conf_threshold is not None and score < conf_threshold:
            continue
        draw.rectangle([x, y, x + w, y + h], outline="red", width=4)
        # draw.text((x, y - 10), f"{score:.2f}", fill="red")

    individual_insects = []
    
    for p in img_preds:
        segm = p.get('segmentation')
        if conf_threshold is not None and p.get('score', 0) < conf_threshold:
            continue
        if segm and plot_masks:
            if isinstance(segm, list):
                # Polygon (list of lists)
                h, w = img_info.get('height', image.height), img_info.get('width', image.width)
                rles = mask_util.frPyObjects(segm, h, w)
                rle = mask_util.merge(rles)
                mask = mask_util.decode(rle)
            elif isinstance(segm, dict) and isinstance(segm.get('counts'), list):
                # Uncompressed RLE
                h, w = segm['size'][0], segm['size'][1]
                rle = mask_util.frPyObjects(segm, h, w)
                mask = mask_util.decode(rle)
            else:
                # Compressed RLE (SAM3 output)
                mask = mask_util.decode(segm)

            masked_insect = np.array(image) * np.expand_dims(mask, axis=2)
            
            x, y, w, h = map(int, p['bbox'])
            padding = 20
            x_start, y_start = max(0, x - padding), max(0, y - padding)
            x_end, y_end = min(image.width, x + w + padding), min(image.height, y + h + padding)
            
            cropped_insect = masked_insect[y_start:y_end, x_start:x_end]
            
            best_iou = max([calculate_iou(p['bbox'], gt_ann['bbox']) for gt_ann in gt_anns], default=0.0)
            
            individual_insects.append((cropped_insect, p.get('score', 1.0), best_iou))
        elif plot_masks:
            x, y, w, h = map(int, p['bbox'])
            cropped_insect = np.array(image.crop((x, y, x + w, y + h)))
            best_iou = max([calculate_iou(p['bbox'], gt_ann['bbox']) for gt_ann in gt_anns], default=0.0)
            individual_insects.append((cropped_insect, p.get('score', 1.0), best_iou))

    num_insects = len(individual_insects)
    
    if num_insects == 0:
        plt.figure(figsize=(15, 10))
        plt.imshow(image_with_boxes)
        plt.title(f"Image: {image_id} | GT (Blue): {len(gt_anns)} | Preds (Red): {num_insects} (Total: {len(img_preds)})", fontsize=16)
        plt.axis('off')
    else:
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, num_insects, height_ratios=[2, 1])

        ax_main = fig.add_subplot(gs[0, :])
        ax_main.imshow(image_with_boxes)
        ax_main.set_title(f"Image: {image_id} | GT (Blue): {len(gt_anns)} | Preds (Red): {num_insects} (Total: {len(img_preds)})", fontsize=16)
        ax_main.axis('off')

        for i, (insect, score, iou) in enumerate(individual_insects):
            ax_crop = fig.add_subplot(gs[1, i])
            ax_crop.imshow(insect.astype(np.uint8))
            ax_crop.set_title(f"Pred {i+1} | Conf: {score:.2f} | Box IoU: {iou:.2f}", fontsize=12)
            ax_crop.axis('off')
            
    plt.tight_layout()
    plt.show()