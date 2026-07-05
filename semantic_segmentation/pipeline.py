#!/usr/bin/env python3
"""
Run NVIDIA SegFormer segmentation on ``images/`` and write outputs to ``results/``.

Layout:
  results/
    nvidia-solid/
    nvidia-stats.xlsx
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEGMENT_SCRIPT = ROOT / "segment_nvidia.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run NVIDIA Cityscapes segmentation.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "images",
        help="Input images folder (default: images/)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Output root folder (default: results/)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    args = parser.parse_args(argv)

    input_dir = args.input_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        str(SEGMENT_SCRIPT),
        "--input-dir",
        str(input_dir),
        "--results-dir",
        str(results_dir),
        "--device",
        args.device,
    ]

    print(f"\n{'=' * 60}\nRunning {SEGMENT_SCRIPT.name}\n{'=' * 60}")
    code = subprocess.call(cmd, cwd=ROOT)
    if code == 0:
        print(f"\n{'=' * 60}\nPipeline complete. Results: {results_dir}\n{'=' * 60}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
