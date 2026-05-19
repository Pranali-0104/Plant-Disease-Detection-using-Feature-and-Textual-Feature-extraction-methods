from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SeverityResult:
    infected_area_px: int
    leaf_area_px: int
    severity_percent: float
    severity_class: str


def severity_class(severity_percent: float, thresholds: dict[str, float]) -> str:
    mild = float(thresholds.get("mild", 5))
    moderate = float(thresholds.get("moderate", 20))
    severe = float(thresholds.get("severe", 45))
    if severity_percent < mild:
        return "Mild"
    if severity_percent < moderate:
        return "Moderate"
    if severity_percent < severe:
        return "Severe"
    return "Critical"


def combine_masks(masks: list[np.ndarray], threshold: float = 0.5, min_area_px: int = 0) -> np.ndarray:
    if not masks:
        raise ValueError("At least one lesion mask is required for severity estimation.")
    combined = np.zeros(masks[0].shape[-2:], dtype=bool)
    for mask in masks:
        binary = np.asarray(mask) >= threshold
        if int(binary.sum()) >= min_area_px:
            combined |= binary
    return combined


def estimate_severity(
    lesion_masks: list[np.ndarray],
    leaf_mask: np.ndarray | None,
    image_shape: tuple[int, int],
    config: dict[str, Any],
) -> SeverityResult:
    severity_cfg = config.get("severity", {})
    threshold = float(severity_cfg.get("infected_mask_threshold", 0.5))
    min_area_px = int(severity_cfg.get("min_lesion_area_px", 0))
    infected_mask = combine_masks(lesion_masks, threshold=threshold, min_area_px=min_area_px)

    if leaf_mask is None:
        leaf_area = int(image_shape[0] * image_shape[1])
    else:
        leaf_area = int(np.asarray(leaf_mask).astype(bool).sum())

    infected_area = int(infected_mask.sum())
    severity_percent = 0.0 if leaf_area == 0 else infected_area / leaf_area * 100.0
    return SeverityResult(
        infected_area_px=infected_area,
        leaf_area_px=leaf_area,
        severity_percent=severity_percent,
        severity_class=severity_class(severity_percent, severity_cfg.get("thresholds", {})),
    )
