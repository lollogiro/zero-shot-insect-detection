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

def class_aware_recall_report(coco_gt, preds, tested_img_ids, iou_threshold=0.1, conf_threshold=None, images_to_remove: list = None, show_plot=False):
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
        images_to_remove (list, optional): List of (key, value) tuples to filter out images
            from the report. Images with annotations matching the criteria are excluded.
            Defaults to None (no filtering).
        show_plot (bool, optional): If True, also draw a horizontal bar chart of recall
            per class (class on the y axis, recall percentage on the x axis).
            Defaults to False.
    Returns:
        pd.DataFrame: Per-class recall statistics, with columns Class, GTs
              (ground truth count), TPs (true positives) and Recall. The same
              table is also printed to stdout.
    """
    preds = _load_predictions(preds)

    tested_img_ids_norm = {normalize_id(i) for i in tested_img_ids}

    if images_to_remove is not None:
        for key, value in images_to_remove:
            img_ids_to_remove = {
                normalize_id(ann['image_id'])
                for ann in coco_gt.dataset.get('annotations', [])
                if ann.get(key) == value
            }
            tested_img_ids_norm -= img_ids_to_remove
            print(f"Removed {len(img_ids_to_remove)} images based on the criteria {key}={value}")

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

    if show_plot:
        plot_df = df.sort_values("Recall", ascending=True)
        pct = plot_df["Recall"].to_numpy() * 100

        fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(plot_df) + 1)))
        ax.barh(plot_df["Class"], pct, height=0.6, color="#2a78d6", zorder=2)
        for y, value in enumerate(pct):
            ax.text(value + 1, y, f"{value:.0f}%", va="center", ha="left",
                    fontsize=9, color="#52514e")

        ax.set_xlim(0, 110)
        ax.set_xlabel("Recall (%)")
        ax.set_ylabel("Class")
        title = f"Class-aware recall (IoU={iou_threshold}"
        if conf_threshold is not None:
            title += f", conf={conf_threshold:.3f}"
        else:
            title += ", conf=None"
        ax.set_title(title + ")")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors="#52514e", length=0)
        fig.tight_layout()
        plt.show()

    return df


def plot_class_area_vs_recall(coco_gt, preds, tested_img_ids, iou_threshold=0.1, conf_threshold=None, images_to_remove: list = None, correlation=False):
    """
    Plot the relative bounding-box area per class next to class-aware recall.

    The relative area is the fraction of the image covered by a ground-truth
    box, so it does not depend on the camera resolution:

        relative_area = (bbox_width * bbox_height) / (image_width * image_height)

    It is expressed as a percentage of the image. Classes are ordered by their
    median relative area (smallest at the bottom), and the recall panel uses
    the same order, so the two panels can be read row by row. The area axis is
    logarithmic because relative areas span more than two orders of magnitude.

    If correlation=True, a third panel shows the class-level relationship
    between median relative area and recall at conf_threshold, with a Spearman
    rank correlation and its p-value. Recall is computed from the unrounded
    TPs/GTs ratio, so tied rounded recalls do not distort the ranks. A
    significant p-value indicates a monotonic association between class size
    and recall; it does not by itself prove that size is the causal driver.

    Args:
        coco_gt (COCO): COCO dataset object containing ground truth annotations and category information.
        preds (str, dict, or list): Predictions data, in any form accepted by
            class_aware_recall_report.
        tested_img_ids (list or set): Collection of image IDs to evaluate.
        iou_threshold (float, optional): Intersection over Union threshold for matching predictions to ground truth.
            Defaults to 0.1.
        conf_threshold (float, optional): Confidence score threshold for filtering predictions.
            Predictions with score below this threshold are excluded. Defaults to None (no filtering).
            This is the operating point used for the correlation if enabled.
        images_to_remove (list, optional): List of (key, value) tuples to filter out images
            from the plot. Images with annotations matching the criteria are excluded.
            Defaults to None (no filtering).
        correlation (bool, optional): If True, add a scatter panel with the
            Spearman rank correlation between per-class median relative area
            and recall, and print rho and the p-value. Defaults to False.
    Returns:
        pd.DataFrame: Per-class statistics with columns Class, GTs, TPs, Recall
            and MedianArea (percentage of the image), sorted by median area.
            When correlation=True, the DataFrame attrs also contain
            'spearman_rho' and 'spearman_p'.
    """
    df = class_aware_recall_report(
        coco_gt, preds, tested_img_ids, iou_threshold=iou_threshold,
        conf_threshold=conf_threshold, images_to_remove=images_to_remove,
    )

    img_ids_norm = {normalize_id(i) for i in tested_img_ids}
    if images_to_remove is not None:
        for key, value in images_to_remove:
            img_ids_norm -= {
                normalize_id(ann['image_id'])
                for ann in coco_gt.dataset.get('annotations', [])
                if ann.get(key) == value
            }

    images = {normalize_id(img['id']): img for img in coco_gt.dataset.get('images', [])}
    areas_by_name = defaultdict(list)
    for ann in coco_gt.dataset.get('annotations', []):
        if ann.get('iscrowd', 0) != 0:
            continue
        img = images.get(normalize_id(ann.get('image_id')))
        if img is None or not img.get('width') or not img.get('height'):
            continue
        if normalize_id(ann.get('image_id')) not in img_ids_norm:
            continue
        _, _, w, h = ann['bbox']
        rel_area = 100 * w * h / (img['width'] * img['height'])
        if rel_area > 0:
            areas_by_name[coco_gt.cats[ann['category_id']]['name']].append(rel_area)

    stats = df[['Class', 'GTs', 'TPs', 'Recall']].copy()
    stats['MedianArea'] = stats['Class'].map(
        lambda name: np.median(areas_by_name[name]) if areas_by_name.get(name) else np.nan
    )
    stats = stats.sort_values('MedianArea', ascending=True).reset_index(drop=True)
    if stats.empty:
        raise ValueError("No ground-truth annotations found for the selected images.")

    order = stats['Class'].tolist()
    box_data = [areas_by_name[name] for name in order]
    if any(len(vals) == 0 for vals in box_data):
        raise ValueError("No ground-truth boxes found for some classes; check tested_img_ids and images_to_remove.")

    height = max(4, 0.55 * len(order) + 1.8)
    fig, axes = plt.subplots(
        1, 3 if correlation else 2,
        figsize=(17 if correlation else 13, height),
        gridspec_kw={'width_ratios': [1.2, 1, 1.1] if correlation else [1.2, 1]},
    )
    ax_area, ax_recall = axes[0], axes[1]
    ax_recall.sharey(ax_area)
    ax_corr = axes[2] if correlation else None

    ax_area.boxplot(
        box_data, orientation='horizontal', positions=range(len(order)),
        widths=0.6, patch_artist=True, showfliers=True,
        boxprops={'facecolor': '#cde2fb', 'edgecolor': '#2a78d6', 'linewidth': 1},
        medianprops={'color': '#184f95', 'linewidth': 1.5},
        whiskerprops={'color': '#2a78d6', 'linewidth': 1},
        capprops={'color': '#2a78d6', 'linewidth': 1},
        flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': '#898781',
                    'markeredgecolor': 'none', 'alpha': 0.6},
    )
    for y, median_area in enumerate(stats['MedianArea']):
        ax_area.annotate(
            f"{median_area:.2f}%", xy=(median_area, y), xytext=(5, 0),
            textcoords='offset points', va='center', fontsize=8,
            fontweight='bold', color='#184f95',
        )

    ax_area.set_xscale('log')
    ax_area.set_xticks([0.1, 0.5, 1, 5, 10, 50])
    ax_area.set_xticklabels(['0.1', '0.5', '1', '5', '10', '50'])
    ax_area.minorticks_off()
    ax_area.set_xlabel('Relative bbox area (% of image, log scale)')
    ax_area.set_yticks(range(len(order)))
    ax_area.set_yticklabels(order)
    ax_area.set_ylabel('Class')
    ax_area.set_title('Relative bbox area', fontsize=11)

    recall_pct = stats['Recall'].to_numpy() * 100
    ax_recall.barh(range(len(order)), recall_pct, height=0.6, color='#2a78d6', zorder=2)
    for y, value in enumerate(recall_pct):
        ax_recall.text(value + 1, y, f"{value:.0f}%", va='center', ha='left',
                       fontsize=9, color='#52514e')
    ax_recall.set_xlim(0, 110)
    ax_recall.set_xlabel('Recall (%)')
    ax_recall.set_title('Class-aware recall', fontsize=11)

    if correlation:
        from scipy.stats import spearmanr

        corr_df = stats.dropna(subset=['MedianArea']).copy()
        corr_df['RecallExact'] = 100 * corr_df['TPs'] / corr_df['GTs']
        x = corr_df['MedianArea'].to_numpy()
        y = corr_df['RecallExact'].to_numpy()
        rho, p_value = spearmanr(x, y)

        ax_corr.scatter(x, y, s=55, color='#2a78d6', edgecolor='#fcfcfb',
                        linewidth=1.2, zorder=3)
        for i, row in enumerate(corr_df.itertuples()):
            dy = 7 if i % 2 == 0 else -13
            ax_corr.annotate(
                row.Class, (row.MedianArea, row.RecallExact), xytext=(6, dy),
                textcoords='offset points', fontsize=8, color='#52514e',
            )

        if len(x) >= 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(np.log10(x), y, 1)
            xs = np.geomspace(x.min(), x.max(), 100)
            ax_corr.plot(xs, slope * np.log10(xs) + intercept, color='#898781',
                         linewidth=1.5, zorder=2, label='linear fit (log area)')
            ax_corr.legend(frameon=False, fontsize=8, loc='lower right')

        ax_corr.set_xscale('log')
        ax_corr.set_xticks([0.1, 0.5, 1, 5, 10, 50])
        ax_corr.set_xticklabels(['0.1', '0.5', '1', '5', '10', '50'])
        ax_corr.minorticks_off()
        ax_corr.set_xlabel('Median relative bbox area (% of image, log scale)')
        ax_corr.set_ylabel('Recall (%)')
        ax_corr.set_ylim(0, 105)
        ax_corr.set_xlim(x.min() * 0.7, x.max() * 1.4)
        if np.isfinite(p_value):
            verdict = 'p < 0.05' if p_value < 0.05 else 'p >= 0.05'
        else:
            verdict = 'p = n/a'
        ax_corr.set_title(
            f"Spearman rho={rho:.2f}, p={p_value:.3f} ({verdict}), n={len(corr_df)}",
            fontsize=10,
        )
        print(
            f"Spearman correlation (median area vs recall at conf={conf_threshold}): "
            f"rho={rho:.3f}, p={p_value:.4f}, n={len(corr_df)}"
        )
        stats.attrs['spearman_rho'] = rho
        stats.attrs['spearman_p'] = p_value

    style_axes = (ax_area, ax_recall, ax_corr) if correlation else (ax_area, ax_recall)
    for ax in style_axes:
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#e1e0d9', linewidth=0.8)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#c3c2b7')
        ax.tick_params(colors='#52514e', length=0)

    title = f"Class-aware recall vs object size (IoU={iou_threshold}"
    if conf_threshold is not None:
        title += f", conf={conf_threshold:.3f}"
    fig.suptitle(title + ")")

    smallest = stats.iloc[0]
    fig.text(
        0.5, 0.02,
        f"Smallest median area: {smallest['Class']} ({smallest['MedianArea']:.2f}% of image)",
        ha='center', fontsize=10, fontweight='bold', color='#52514e',
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    plt.show()

    return stats


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

    n_drawn = 0
    for _p in img_preds:
        _score = _p.get("score", 1.0)
        if conf_threshold is not None and _score < conf_threshold:
            continue
        n_drawn += 1

    num_insects = len(individual_insects)
    
    if num_insects == 0:
        plt.figure(figsize=(15, 10))
        plt.imshow(image_with_boxes)
        plt.title(f"Image: {image_id} | GT (Blue): {len(gt_anns)} | Preds (Red): {n_drawn} (Total: {len(img_preds)})", fontsize=16)
        plt.axis('off')
    else:
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, num_insects, height_ratios=[2, 1])

        ax_main = fig.add_subplot(gs[0, :])
        ax_main.imshow(image_with_boxes)
        ax_main.set_title(f"Image: {image_id} | GT (Blue): {len(gt_anns)} | Preds (Red): {n_drawn} (Total: {len(img_preds)})", fontsize=16)
        ax_main.axis('off')

        for i, (insect, score, iou) in enumerate(individual_insects):
            ax_crop = fig.add_subplot(gs[1, i])
            ax_crop.imshow(insect.astype(np.uint8))
            ax_crop.set_title(f"Pred {i+1} | Conf: {score:.2f} | Box IoU: {iou:.2f}", fontsize=12)
            ax_crop.axis('off')

    plt.tight_layout()
    plt.show()


def _class_matching_curves(coco_gt, preds, tested_img_ids, iou_threshold=0.1, images_to_remove: list = None):
    """Match predictions to ground truth once per class and return scoring curves.

    This mirrors the image and annotation filtering of
    :func:`class_aware_recall_report`, but keeps the per-detection match outcome
    instead of only the aggregate counts. The detections are sorted by score,
    and because filtering to ``score >= thr`` keeps a prefix of that ordering,
    the greedy matching of the surviving detections is unchanged. The TP count
    at any threshold is therefore the sum of ``is_tp`` over that prefix.

    Returns a dict with the filtered image ids, per-class GT counts, the
    per-class GT boxes grouped by image, and a ``curves`` entry per class with
    aligned ``image_ids``, ``scores`` and ``is_tp`` arrays.
    """
    preds = _load_predictions(preds)

    tested_img_ids_norm = {normalize_id(i) for i in tested_img_ids}

    if images_to_remove is not None:
        for key, value in images_to_remove:
            img_ids_to_remove = {
                normalize_id(ann['image_id'])
                for ann in coco_gt.dataset.get('annotations', [])
                if ann.get(key) == value
            }
            tested_img_ids_norm -= img_ids_to_remove

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
            preds_by_img[p_img_id].append(p)

    detections = []
    for img_id in tested_img_ids_norm:
        for p in preds_by_img.get(img_id, []):
            detections.append((img_id, p["bbox"], p.get("score", 0)))
    detections.sort(key=lambda x: x[2], reverse=True)

    det_image_ids = np.array([d[0] for d in detections], dtype=object)
    det_scores = np.array([float(d[2]) for d in detections], dtype=float)

    curves = {}
    for cid, gt_by_img in gt_by_img_and_class.items():
        if gt_counts_by_class.get(cid, 0) == 0:
            continue
        tp, _, _ = match_predictions_to_gt(detections, gt_by_img, iou_threshold)
        curves[cid] = {
            'image_ids': det_image_ids,
            'scores': det_scores,
            'is_tp': np.asarray(tp, dtype=bool),
        }

    return {
        'img_ids': tested_img_ids_norm,
        'gt_counts': dict(gt_counts_by_class),
        'gt_by_img_and_class': gt_by_img_and_class,
        'curves': curves,
    }


def _tp_mask(curve, conf_threshold=None):
    """Boolean mask of TP detections surviving a confidence threshold."""
    if conf_threshold is None:
        return curve['is_tp']
    return curve['is_tp'] & (curve['scores'] >= conf_threshold)


def _tp_counts_at_threshold(ctx, class_ids, conf_threshold=None):
    """Per-class TP counts from the matching curves at a confidence threshold."""
    counts = {}
    for cid in class_ids:
        counts[cid] = int(_tp_mask(ctx['curves'][cid], conf_threshold).sum())
    return counts


def _community_metrics(n_gt, tp):
    """Ecological metrics comparing the GT and detected communities.

    ``n_gt`` and ``tp`` are per-class count arrays. ``P_AI`` uses the matched
    true positives only: false positives carry no taxon and cannot be assigned
    to a class. Bray-Curtis is computed on relative abundances, so it measures
    compositional distortion rather than the overall detection shortfall.
    """
    n_gt = np.asarray(n_gt, dtype=float)
    tp = np.asarray(tp, dtype=float)

    total_gt = float(n_gt.sum())
    total_tp = float(tp.sum())
    s_gt = int((n_gt > 0).sum())
    s_ai = int((tp > 0).sum())

    result = {
        'bray_curtis': np.nan,
        'shannon_gt': np.nan,
        'shannon_ai': np.nan,
        'shannon_diff': np.nan,
        'pielou_gt': np.nan,
        'pielou_ai': np.nan,
        'pielou_diff': np.nan,
        'richness_gt': s_gt,
        'richness_ai': s_ai,
        'gt_total': total_gt,
        'tp_total': total_tp,
    }
    if total_gt == 0 or total_tp == 0:
        return result

    p_gt = n_gt / total_gt
    p_ai = tp / total_tp

    h_gt = -np.sum(p_gt[p_gt > 0] * np.log(p_gt[p_gt > 0]))
    h_ai = -np.sum(p_ai[p_ai > 0] * np.log(p_ai[p_ai > 0]))
    j_gt = float(h_gt / np.log(s_gt)) if s_gt > 1 else np.nan
    j_ai = float(h_ai / np.log(s_ai)) if s_ai > 1 else np.nan

    result.update({
        'bray_curtis': float(0.5 * np.abs(p_gt - p_ai).sum()),
        'shannon_gt': float(h_gt),
        'shannon_ai': float(h_ai),
        'shannon_diff': float(h_ai - h_gt),
        'pielou_gt': j_gt,
        'pielou_ai': j_ai,
        'pielou_diff': float(j_ai - j_gt) if np.isfinite(j_gt) and np.isfinite(j_ai) else np.nan,
    })
    return result


def _community_count_matrices(ctx, class_ids, conf_threshold=None):
    """Build per-image GT and TP count matrices for paired bootstrap."""
    img_ids = sorted(ctx['img_ids'], key=lambda x: str(x))
    img_index = {img_id: i for i, img_id in enumerate(img_ids)}

    n_gt = np.zeros((len(img_ids), len(class_ids)), dtype=float)
    tp = np.zeros((len(img_ids), len(class_ids)), dtype=float)

    for j, cid in enumerate(class_ids):
        for img_id, boxes in ctx['gt_by_img_and_class'][cid].items():
            n_gt[img_index[img_id], j] = len(boxes)

        curve = ctx['curves'][cid]
        for img_id in curve['image_ids'][_tp_mask(curve, conf_threshold)]:
            tp[img_index[img_id], j] += 1

    return n_gt, tp


def _format_ci(value, ci):
    """Format a point estimate with an optional percentile confidence interval."""
    if not np.isfinite(value):
        return "n/a"
    if ci is None or not np.isfinite(ci[0]) or not np.isfinite(ci[1]):
        return f"{value:.3f}"
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def community_composition_report(coco_gt, preds, tested_img_ids, iou_threshold=0.1,
                                 conf_threshold=None, images_to_remove: list = None,
                                 show_plot=False, n_bootstrap=1000, random_state=0):
    """Compare the real and AI-perceived taxon communities.

    Ground truth abundance is ``N_GT,c``, the number of annotations of taxon
    ``c``. The detected abundance is ``TP_c``, the number of class-aware true
    positives. False positives are excluded because they cannot be assigned a
    taxon. Relative abundances are:

        P_GT(c) = N_GT,c / sum_k N_GT,k
        P_AI(c) = TP_c / sum_k TP_k

    If recall were the same for every taxon, ``P_AI`` would equal ``P_GT``.
    Class-dependent recall distorts it.

    Metrics:

        BC = sum_c |P_GT(c) - P_AI(c)| / sum_c (P_GT(c) + P_AI(c))
        H' = -sum_{p>0} p ln p
        J' = H' / ln(S)

    ``BC`` is computed on relative abundances, so it isolates compositional
    distortion from the overall detection shortfall. ``S`` is the number of
    taxa with a non-zero count, which is reported separately for GT and AI
    because a taxon with zero TP drops out of the AI community. ``J'`` is NaN
    when ``S < 2``.

    Args:
        coco_gt (COCO): COCO dataset object containing ground truth annotations and category information.
        preds (str, dict, or list): Predictions data, in any form accepted by
            class_aware_recall_report.
        tested_img_ids (list or set): Collection of image IDs to evaluate.
        iou_threshold (float, optional): Intersection over Union threshold for matching predictions to ground truth.
            Defaults to 0.1.
        conf_threshold (float, optional): Operating-point confidence threshold. Defaults to None (all detections).
        images_to_remove (list, optional): List of (key, value) tuples. Every image containing a
            matching annotation is excluded. Defaults to None.
        show_plot (bool, optional): If True, draw paired horizontal bars of P_GT and P_AI. Defaults to False.
        n_bootstrap (int, optional): Number of paired image-bootstrap replicates for
            percentile confidence intervals. Set to 0 to skip. Defaults to 1000.
        random_state (int, optional): Seed for the bootstrap. Defaults to 0.

    Returns:
        pd.DataFrame: Per-class table with columns Class, GTs, TPs, Recall,
            P_GT and P_AI, sorted by GTs descending. ``attrs`` contains
            'bray_curtis', 'shannon_gt', 'shannon_ai', 'shannon_diff',
            'pielou_gt', 'pielou_ai', 'pielou_diff', 'richness_gt',
            'richness_ai' and, when bootstrap runs, the matching '_ci' pairs.
    """
    ctx = _class_matching_curves(
        coco_gt, preds, tested_img_ids,
        iou_threshold=iou_threshold, images_to_remove=images_to_remove,
    )

    class_ids = sorted(ctx['gt_counts'].keys())
    tp_counts = _tp_counts_at_threshold(ctx, class_ids, conf_threshold)

    results = []
    for cid in class_ids:
        total_gt = ctx['gt_counts'][cid]
        total_tp = tp_counts[cid]
        results.append({
            "Class": coco_gt.cats[cid]["name"],
            "GTs": int(total_gt),
            "TPs": int(total_tp),
            "Recall": total_tp / total_gt,
        })

    df = pd.DataFrame(results).sort_values("GTs", ascending=False).reset_index(drop=True)

    n_gt = df["GTs"].to_numpy(dtype=float)
    tp = df["TPs"].to_numpy(dtype=float)
    metrics = _community_metrics(n_gt, tp)
    total_tp = metrics['tp_total']

    if total_tp > 0:
        df["P_GT"] = n_gt / metrics['gt_total']
        df["P_AI"] = tp / total_tp
    else:
        df["P_GT"] = n_gt / metrics['gt_total'] if metrics['gt_total'] > 0 else np.nan
        df["P_AI"] = np.nan

    if total_tp == 0:
        print("Warning: no true positives at this operating point; AI metrics are undefined.")

    for key, value in metrics.items():
        df.attrs[key] = value
    df.attrs['conf_threshold'] = conf_threshold
    df.attrs['iou_threshold'] = iou_threshold

    if n_bootstrap and n_bootstrap > 0 and total_tp > 0:
        n_gt_mat, tp_mat = _community_count_matrices(ctx, class_ids, conf_threshold)
        rng = np.random.default_rng(random_state)
        n_images = n_gt_mat.shape[0]
        rep_keys = ['bray_curtis', 'shannon_gt', 'shannon_ai', 'shannon_diff',
                    'pielou_gt', 'pielou_ai', 'pielou_diff']
        replicates = {key: [] for key in rep_keys}
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_images, n_images)
            rep = _community_metrics(n_gt_mat[idx].sum(axis=0), tp_mat[idx].sum(axis=0))
            if not np.isfinite(rep['bray_curtis']):
                continue
            if not np.isfinite(rep['pielou_gt']) or not np.isfinite(rep['pielou_ai']):
                continue
            for key in rep_keys:
                replicates[key].append(rep[key])

        for key in rep_keys:
            values = np.asarray(replicates[key], dtype=float)
            if len(values) > 0:
                df.attrs[f'{key}_ci'] = (float(np.percentile(values, 2.5)),
                                         float(np.percentile(values, 97.5)))
            else:
                df.attrs[f'{key}_ci'] = (np.nan, np.nan)
        df.attrs['n_bootstrap'] = n_bootstrap
        df.attrs['n_bootstrap_effective'] = len(replicates['bray_curtis'])

    print(f"\nCommunity composition report (IoU={iou_threshold}, "
          f"conf={f'{conf_threshold:.3f}' if conf_threshold is not None else 'None'})")
    print(df.round(3).to_string(index=False))

    bc_ci = df.attrs.get('bray_curtis_ci')
    h_gt_ci = df.attrs.get('shannon_gt_ci')
    h_ai_ci = df.attrs.get('shannon_ai_ci')
    h_diff_ci = df.attrs.get('shannon_diff_ci')
    j_gt_ci = df.attrs.get('pielou_gt_ci')
    j_ai_ci = df.attrs.get('pielou_ai_ci')
    j_diff_ci = df.attrs.get('pielou_diff_ci')

    print(f"Bray-Curtis (relative abundance): {_format_ci(metrics['bray_curtis'], bc_ci)}")
    print(f"Shannon H': GT {_format_ci(metrics['shannon_gt'], h_gt_ci)} | "
          f"AI {_format_ci(metrics['shannon_ai'], h_ai_ci)} | "
          f"delta {_format_ci(metrics['shannon_diff'], h_diff_ci)}")
    print(f"Pielou J': GT {_format_ci(metrics['pielou_gt'], j_gt_ci)} | "
          f"AI {_format_ci(metrics['pielou_ai'], j_ai_ci)} | "
          f"delta {_format_ci(metrics['pielou_diff'], j_diff_ci)}")
    print(f"Richness S: GT {metrics['richness_gt']} | AI {metrics['richness_ai']} | "
          f"GT total {int(metrics['gt_total'])} | TP total {int(metrics['tp_total'])}")
    if 'n_bootstrap_effective' in df.attrs:
        print(f"Bootstrap: {df.attrs['n_bootstrap_effective']}/{df.attrs['n_bootstrap']} "
              f"valid replicates (paired over images)")

    if show_plot:
        plot_df = df.sort_values("P_GT", ascending=True)

        fig, ax = plt.subplots(figsize=(9, max(3, 0.55 * len(plot_df) + 1.8)))
        y = np.arange(len(plot_df))
        height = 0.36
        gt_vals = plot_df["P_GT"].to_numpy() * 100
        ai_vals = plot_df["P_AI"].to_numpy() * 100

        ax.barh(y - height / 2 - 0.02, gt_vals, height=height,
                color="#898781", label="Ground truth", zorder=2)
        ax.barh(y + height / 2 + 0.02, ai_vals, height=height,
                color="#2a78d6", label="AI detected", zorder=2)

        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["Class"])
        ax.set_xlabel("Relative abundance (%)")
        ax.set_ylabel("Class")
        ax.set_title(f"Community composition (IoU={iou_threshold}, "
                     f"conf={f'{conf_threshold:.3f}' if conf_threshold is not None else 'None'})\n"
                     f"BC={_format_ci(metrics['bray_curtis'], bc_ci)}",
                     fontsize=11)
        ax.legend(frameon=False, loc="lower right", fontsize=9)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors="#52514e", length=0)
        fig.text(
            0.5, 0.02,
            f"H': GT {metrics['shannon_gt']:.3f} vs AI {metrics['shannon_ai']:.3f} | "
            f"J': GT {metrics['pielou_gt']:.3f} vs AI {metrics['pielou_ai']:.3f} | "
            f"S: GT {metrics['richness_gt']} vs AI {metrics['richness_ai']}",
            ha='center', fontsize=9, color='#52514e',
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        plt.show()

    return df


def plot_community_metrics_vs_confidence(coco_gt, preds, tested_img_ids, iou_threshold=0.1,
                                         conf_thresholds=None, images_to_remove: list = None):
    """Plot community metrics as a function of the operating confidence threshold.

    The matching is computed once per class over all predictions. Because the
    greedy match processes detections by descending score, the TP count at
    ``score >= thr`` is the cumulative TP count over that score-sorted prefix,
    so the whole sweep needs a single matching pass.

    Args:
        coco_gt (COCO): COCO dataset object containing ground truth annotations and category information.
        preds (str, dict, or list): Predictions data, in any form accepted by
            class_aware_recall_report.
        tested_img_ids (list or set): Collection of image IDs to evaluate.
        iou_threshold (float, optional): Intersection over Union threshold for matching predictions to ground truth.
            Defaults to 0.1.
        conf_thresholds (array-like, optional): Confidence values to sweep.
            Defaults to 0.05, 0.10, ..., 0.95.
        images_to_remove (list, optional): List of (key, value) tuples. Every image containing a
            matching annotation is excluded. Defaults to None.

    Returns:
        pd.DataFrame: One row per threshold with columns bray_curtis,
            shannon_gt, shannon_ai, shannon_diff, pielou_gt, pielou_ai,
            pielou_diff, richness_gt, richness_ai, gt_total and tp_total.
    """
    if conf_thresholds is None:
        conf_thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)

    ctx = _class_matching_curves(
        coco_gt, preds, tested_img_ids,
        iou_threshold=iou_threshold, images_to_remove=images_to_remove,
    )

    class_ids = sorted(ctx['gt_counts'].keys())
    n_gt = np.array([ctx['gt_counts'][cid] for cid in class_ids], dtype=float)

    rows = []
    for thr in conf_thresholds:
        tp_counts = _tp_counts_at_threshold(ctx, class_ids, thr)
        tp = np.array([tp_counts[cid] for cid in class_ids], dtype=float)
        metrics = _community_metrics(n_gt, tp)
        metrics['conf_threshold'] = float(thr)
        rows.append(metrics)

    sweep_df = pd.DataFrame(rows).set_index('conf_threshold')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_div, ax_bc = axes[0], axes[1]
    x = sweep_df.index.to_numpy()

    ax_div.plot(x, sweep_df['shannon_gt'], color='#898781', linestyle='--',
                linewidth=1.8, label="H' GT")
    ax_div.plot(x, sweep_df['shannon_ai'], color='#2a78d6', linewidth=2,
                label="H' AI")
    ax_div.plot(x, sweep_df['pielou_gt'], color='#898781', linestyle=':',
                linewidth=1.8, label="J' GT")
    ax_div.plot(x, sweep_df['pielou_ai'], color='#2a78d6', linestyle='-.',
                linewidth=1.8, label="J' AI")
    ax_div.set_xlabel('Confidence threshold')
    ax_div.set_ylabel("Diversity index")
    ax_div.set_title("Shannon H' and Pielou J'", fontsize=11)
    ax_div.set_xlim(0, 1)
    ax_div.legend(frameon=False, fontsize=8)
    ax_div.grid(axis='y', color='#e1e0d9', linewidth=0.8)

    ax_bc.plot(x, sweep_df['bray_curtis'], color='#2a78d6', linewidth=2,
               label='Bray-Curtis')
    ax_bc.set_xlabel('Confidence threshold')
    ax_bc.set_ylabel('Bray-Curtis dissimilarity')
    ax_bc.set_ylim(0, 1.05)
    ax_bc.set_xlim(0, 1)
    ax_bc.set_title('Compositional distortion and richness', fontsize=11)

    ax_rich = ax_bc.twinx()
    ax_rich.plot(x, sweep_df['richness_ai'], color='#898781', linestyle='--',
                 linewidth=1.8, label='Richness AI')
    ax_rich.set_ylabel('AI richness S')
    ax_rich.set_ylim(0, max(2, sweep_df['richness_gt'].max() + 1))

    lines = ax_bc.get_lines() + ax_rich.get_lines()
    ax_bc.legend(lines, [line.get_label() for line in lines],
                 frameon=False, fontsize=8, loc='lower left')
    ax_bc.grid(axis='y', color='#e1e0d9', linewidth=0.8)

    for ax in (ax_div, ax_bc, ax_rich):
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#c3c2b7')
        ax.tick_params(colors='#52514e', length=0)

    fig.suptitle(f"Community metrics vs confidence (IoU={iou_threshold})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()

    return sweep_df