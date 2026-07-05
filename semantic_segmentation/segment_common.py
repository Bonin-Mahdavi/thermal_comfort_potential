"""Shared layout, I/O, and visualization helpers for segmentation scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Input / photo slot (original image size — not scaled in outputs).
PHOTO_WIDTH = 1366
PHOTO_HEIGHT = 607

# White margins around the photo (left, top, bottom only; right is the legend panel).
MARGIN_LEFT = 16
MARGIN_TOP = 16
MARGIN_BOTTOM = 16

# Fixed legend column on the right.
LEGEND_PANEL_WIDTH = 320
LEGEND_FONT_SIZE = 14
LEGEND_SWATCH = 16
LEGEND_PAD = 12
LEGEND_ROW_GAP = 6
LEGEND_SWATCH_GAP = 8
LEGEND_PANEL_MARGIN = 12

OUTPUT_WIDTH = MARGIN_LEFT + PHOTO_WIDTH + LEGEND_PANEL_WIDTH
OUTPUT_HEIGHT = MARGIN_TOP + PHOTO_HEIGHT + MARGIN_BOTTOM
LEGEND_PANEL_HEIGHT = OUTPUT_HEIGHT

CITYSCAPES_CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

CITYSCAPES_COLORS = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [150, 95, 55],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)

RESULTS_DIR_NAME = "results"
NVIDIA_SOLID_DIR = "nvidia-solid"
NVIDIA_STATS_FILE = "nvidia-stats.xlsx"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def palette_for_model(num_labels: int) -> np.ndarray:
    if num_labels == len(CITYSCAPES_COLORS):
        return CITYSCAPES_COLORS
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(num_labels, 3), dtype=np.uint8)


def colorize(pred: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return palette[pred]


def fit_photo_size(photo: Image.Image) -> tuple[Image.Image, bool]:
    """Resize to the standard photo slot when the source differs (e.g. D-3.jpg)."""
    photo = photo.convert("RGB")
    target = (PHOTO_WIDTH, PHOTO_HEIGHT)
    if photo.size == target:
        return photo, False
    return photo.resize(target, Image.Resampling.LANCZOS), True


def clamp_pred(pred: np.ndarray, num_labels: int) -> np.ndarray:
    pred = pred.astype(np.int64, copy=False)
    pred[pred < 0] = 0
    pred[pred >= num_labels] = 0
    return pred


def class_pixel_fractions(pred: np.ndarray, num_labels: int) -> list[float]:
    """Per-class pixel share in [0, 1] (not percent)."""
    total_px = int(pred.size)
    if total_px == 0:
        return [0.0] * num_labels
    return [int(np.sum(pred == i)) / total_px for i in range(num_labels)]


def write_stats_excel(rows: list[dict[str, object]], path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "pixel_fractions"
    headers = ["image", *CITYSCAPES_CLASSES]
    ws.append(headers)
    for row in rows:
        ws.append([row["image"], *[row[name] for name in CITYSCAPES_CLASSES]])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def load_segment_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def truncate_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> str:
    if draw.textlength(text, font=font) <= max_w:
        return text
    ellipsis = "…"
    trimmed = text
    while trimmed and draw.textlength(trimmed + ellipsis, font=font) > max_w:
        trimmed = trimmed[:-1]
    return (trimmed + ellipsis) if trimmed else ellipsis


def _legend_row_height(font: ImageFont.ImageFont) -> int:
    dummy = Image.new("RGB", (8, 8), (255, 255, 255))
    measure = ImageDraw.Draw(dummy)
    bbox = measure.textbbox((0, 0), "Ag", font=font)
    return max(LEGEND_SWATCH, bbox[3] - bbox[1])


def render_legend_panel(
    class_ids: list[int],
    palette: np.ndarray,
    id2label: dict[int, str],
    pred: np.ndarray,
) -> Image.Image:
    """Draw legend into a fixed-size panel (same dimensions for every image)."""
    total_px = int(pred.size)
    panel_w = LEGEND_PANEL_WIDTH
    panel_h = LEGEND_PANEL_HEIGHT
    inner_w = panel_w - 2 * LEGEND_PANEL_MARGIN

    font = load_segment_font(LEGEND_FONT_SIZE)
    dummy = Image.new("RGB", (8, 8), (255, 255, 255))
    measure = ImageDraw.Draw(dummy)
    text_max_w = inner_w - 2 * LEGEND_PAD - LEGEND_SWATCH - LEGEND_SWATCH_GAP
    row_h = _legend_row_height(font)

    fg = (28, 28, 28)
    swatch_border = (90, 90, 90)
    leg = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
    draw = ImageDraw.Draw(leg)

    cy = LEGEND_PANEL_MARGIN + LEGEND_PAD
    bottom_limit = panel_h - LEGEND_PANEL_MARGIN - LEGEND_PAD

    for cid in class_ids:
        if cy + row_h > bottom_limit:
            break

        base_name = id2label.get(cid, str(cid))
        n = int(np.sum(pred == cid))
        pct = 100.0 * n / total_px if total_px else 0.0
        label = truncate_text(
            measure,
            f"{base_name} ({pct:.1f} %)",
            font,
            text_max_w,
        )
        rgb = tuple(int(x) for x in palette[cid])

        sx0 = LEGEND_PANEL_MARGIN + LEGEND_PAD
        sy0 = cy + (row_h - LEGEND_SWATCH) // 2
        draw.rectangle(
            (sx0, sy0, sx0 + LEGEND_SWATCH, sy0 + LEGEND_SWATCH),
            fill=rgb,
            outline=swatch_border,
            width=1,
        )
        tx = sx0 + LEGEND_SWATCH + LEGEND_SWATCH_GAP
        _l, _t, _r, _b = draw.textbbox((0, 0), label, font=font)
        th = _b - _t
        ty = cy + (row_h - th) // 2 - _t
        draw.text((tx, ty), label, fill=fg, font=font)
        cy += row_h + LEGEND_ROW_GAP

    return leg


def compose_framed_layout(photo: Image.Image, legend_panel: Image.Image) -> Image.Image:
    """Fixed canvas: margins L/T/B, photo at native size, legend column on the right."""
    photo = photo.convert("RGB")
    if photo.size != (PHOTO_WIDTH, PHOTO_HEIGHT):
        raise ValueError(
            f"Expected photo size {PHOTO_WIDTH}x{PHOTO_HEIGHT}, got {photo.size[0]}x{photo.size[1]}"
        )
    if legend_panel.size != (LEGEND_PANEL_WIDTH, LEGEND_PANEL_HEIGHT):
        raise ValueError(
            f"Expected legend panel {LEGEND_PANEL_WIDTH}x{LEGEND_PANEL_HEIGHT}, "
            f"got {legend_panel.size[0]}x{legend_panel.size[1]}"
        )

    canvas = Image.new("RGB", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (255, 255, 255))
    canvas.paste(photo, (MARGIN_LEFT, MARGIN_TOP))
    canvas.paste(legend_panel, (MARGIN_LEFT + PHOTO_WIDTH, 0))
    return canvas


def collect_input_paths(explicit: list[Path], input_dir: Path) -> list[Path]:
    if explicit:
        paths: list[Path] = []
        for p in explicit:
            r = p.expanduser().resolve()
            if not r.is_file():
                raise FileNotFoundError(f"Not a file: {p}")
            if r.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported type (use {IMAGE_SUFFIXES}): {p}")
            paths.append(r)
        return paths
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"No input folder {input_dir!s} (create it or pass image paths)."
        )
    return sorted(
        p.resolve()
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )

