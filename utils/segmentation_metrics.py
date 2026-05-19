from __future__ import annotations

import numpy as np


def binary_confusion(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, int]:
    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    counts = binary_confusion(pred_mask, gt_mask)
    denom = counts["tp"] + counts["fp"] + counts["fn"]
    return 0.0 if denom == 0 else counts["tp"] / denom


def dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    counts = binary_confusion(pred_mask, gt_mask)
    denom = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    return 0.0 if denom == 0 else 2 * counts["tp"] / denom


def precision_recall(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float]:
    counts = binary_confusion(pred_mask, gt_mask)
    precision_den = counts["tp"] + counts["fp"]
    recall_den = counts["tp"] + counts["fn"]
    precision = 0.0 if precision_den == 0 else counts["tp"] / precision_den
    recall = 0.0 if recall_den == 0 else counts["tp"] / recall_den
    return precision, recall
