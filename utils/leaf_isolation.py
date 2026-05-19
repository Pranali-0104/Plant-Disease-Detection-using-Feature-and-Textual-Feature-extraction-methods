from __future__ import annotations

from typing import Any

import numpy as np


def isolate_leaf(image_rgb: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    severity_cfg = config.get("severity", {})
    leaf_cfg = severity_cfg.get("leaf_isolation", {})
    if not leaf_cfg.get("enabled", False):
        return np.ones(image_rgb.shape[:2], dtype=bool)

    try:
        import cv2
    except ImportError:
        return np.ones(image_rgb.shape[:2], dtype=bool)

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower_green = np.array([20, 20, 20], dtype=np.uint8)
    upper_green = np.array([100, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel_size = int(leaf_cfg.get("morph_kernel_size", 5))
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if leaf_cfg.get("largest_contour_only", True):
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cleaned = np.zeros_like(mask)
            if cv2.contourArea(largest) >= int(leaf_cfg.get("min_leaf_area_px", 500)):
                cv2.drawContours(cleaned, [largest], -1, 255, thickness=-1)
                mask = cleaned

    return mask.astype(bool)
