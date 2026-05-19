from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def mask_to_rgba(mask: np.ndarray, color: tuple[int, int, int] = (255, 0, 0), alpha: int = 170) -> Image.Image:
    binary = np.asarray(mask).astype(bool)
    rgba = np.zeros((*binary.shape, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = np.where(binary, alpha, 0)
    return Image.fromarray(rgba, mode="RGBA")


def blend_mask(image_path: Path, mask: np.ndarray, output_path: Path, alpha: float = 0.45) -> None:
    image = Image.open(image_path).convert("RGBA")
    overlay = mask_to_rgba(mask, alpha=int(alpha * 255))
    blended = Image.alpha_composite(image, overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blended.convert("RGB").save(output_path, quality=95)


def draw_combined(
    image_path: Path,
    mask: np.ndarray,
    bbox: list[float] | None,
    label: str,
    output_path: Path,
    alpha: float = 0.45,
) -> None:
    image = Image.open(image_path).convert("RGBA")
    overlay = mask_to_rgba(mask, alpha=int(alpha * 255))
    combined = Image.alpha_composite(image, overlay).convert("RGB")
    draw = ImageDraw.Draw(combined)
    font = ImageFont.load_default()

    if bbox is not None:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_height - 4)
        draw.rectangle((x1, label_y, x1 + text_width + 6, label_y + text_height + 4), fill="red")
        draw.text((x1 + 3, label_y + 2), label, fill="white", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path, quality=95)


def save_side_by_side(image_path: Path, overlay_path: Path, output_path: Path) -> None:
    original = Image.open(image_path).convert("RGB")
    overlay = Image.open(overlay_path).convert("RGB")
    canvas = Image.new("RGB", (original.width + overlay.width, max(original.height, overlay.height)), "white")
    canvas.paste(original, (0, 0))
    canvas.paste(overlay, (original.width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def extract_contours(mask: np.ndarray) -> list[list[list[int]]]:
    try:
        import cv2
    except ImportError:
        return []
    binary = np.asarray(mask).astype("uint8") * 255
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour.squeeze(1).astype(int).tolist() for contour in contours if len(contour) >= 3]
