"""
NVIDIA SegFormer-B5 (Cityscapes): solid segmentation + Excel class pixel fractions.

Writes under ``results/``:
  - ``nvidia-solid/``
  - ``nvidia-stats.xlsx``
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from segment_common import (
    CITYSCAPES_CLASSES,
    IMAGE_SUFFIXES,
    NVIDIA_SOLID_DIR,
    NVIDIA_STATS_FILE,
    PHOTO_HEIGHT,
    PHOTO_WIDTH,
    clamp_pred,
    class_pixel_fractions,
    collect_input_paths,
    colorize,
    compose_framed_layout,
    fit_photo_size,
    palette_for_model,
    pick_device,
    render_legend_panel,
    write_stats_excel,
)

_TMP_DIR = Path(__file__).resolve().parent / ".tmp"
_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(_TMP_DIR)
os.environ["TMP"] = str(_TMP_DIR)
os.environ["TEMP"] = str(_TMP_DIR)

with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

MODEL_ID = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
MODEL_NAME = "NVIDIA SegFormer-B5"


def load_model(device: torch.device):
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    return processor, model


@torch.inference_mode()
def predict_mask(
    processor,
    model,
    image: Image.Image,
    device: torch.device,
) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    _, _, h, w = pixel_values.shape

    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits
    upsampled = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    pred = upsampled.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()

    mask = Image.fromarray(pred, mode="L").resize(image.size, Image.Resampling.NEAREST)
    return np.array(mask, dtype=np.int64)


def process_image(
    image_path: Path,
    solid_dir: Path,
    device: torch.device,
    processor,
    model,
) -> dict[str, object]:
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size
    image, resized = fit_photo_size(image)
    if resized:
        print(
            f"  warning: {image_path.name} resized "
            f"{orig_size[0]}x{orig_size[1]} -> {PHOTO_WIDTH}x{PHOTO_HEIGHT}"
        )
    pred = predict_mask(processor, model, image, device)

    palette = palette_for_model(model.config.num_labels)
    num_labels = int(palette.shape[0])
    pred = clamp_pred(pred, num_labels)

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    colored = colorize(pred, palette)
    labels_present = sorted(int(x) for x in np.unique(pred))
    legend_panel = render_legend_panel(labels_present, palette, id2label, pred)

    stem = image_path.stem
    out_name = f"{stem}_segmented.png"
    segment_img = Image.fromarray(colored)
    solid_path = solid_dir / out_name
    compose_framed_layout(segment_img, legend_panel).save(solid_path)

    fractions = class_pixel_fractions(pred, len(CITYSCAPES_CLASSES))
    row: dict[str, object] = {"image": image_path.name}
    for name, frac in zip(CITYSCAPES_CLASSES, fractions, strict=True):
        row[name] = round(frac, 6)
    return row


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME}: solid segmentation (batch)")
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--input-dir", type=Path, default=root / "images")
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    args = parser.parse_args(argv)

    try:
        inputs = collect_input_paths(list(args.images), args.input_dir.expanduser())
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 1

    if not inputs:
        print(
            f"No images found in {args.input_dir.resolve()} "
            f"(supported: {', '.join(sorted(IMAGE_SUFFIXES))}).",
            file=sys.stderr,
        )
        return 1

    results_dir = args.results_dir.expanduser().resolve()
    solid_dir = results_dir / NVIDIA_SOLID_DIR
    solid_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device() if args.device == "auto" else torch.device(args.device)

    print(f"{MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Results: {results_dir}")
    print(f"  solid: {solid_dir}")

    processor, model = load_model(device)
    excel_rows: list[dict[str, object]] = []

    for image_path in inputs:
        row = process_image(image_path, solid_dir, device, processor, model)
        excel_rows.append(row)
        print(f"  {image_path.name}")

    stats_path = results_dir / NVIDIA_STATS_FILE
    write_stats_excel(excel_rows, stats_path)
    print(f"\nStats saved: {stats_path}")
    print(f"Done ({len(inputs)} image(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
