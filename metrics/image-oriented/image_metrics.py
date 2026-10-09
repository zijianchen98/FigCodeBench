#!/usr/bin/env python3
"""Evaluate one language's image outputs against data_gt.

The default run computes PSNR, SSIM, and mean pixel-wise CIEDE2000 for
exemplary and user_generated, split by image_base/image_variant1/
image_variant2.  LPIPS is implemented behind --with-lpips because its
pretrained AlexNet weights may need to be downloaded on first use.
"""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


SPLITS = ("exemplary", "user_generated")
VARIANTS = ("image_base", "image_variant1", "image_variant2")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class MetricAccumulator:
    expected: int = 0
    succeeded: int = 0
    psnr_sum: float = 0.0
    ssim_sum: float = 0.0
    ciede2000_sum: float = 0.0
    lpips_sum: float = 0.0
    errors: list[str] = field(default_factory=list)

    def add(self, values: tuple[float, float, float, float | None]) -> None:
        psnr, ssim, ciede2000, lpips_value = values
        self.succeeded += 1
        self.psnr_sum += psnr
        self.ssim_sum += ssim
        self.ciede2000_sum += ciede2000
        if lpips_value is not None:
            self.lpips_sum += lpips_value

    def merge(self, other: "MetricAccumulator") -> None:
        self.expected += other.expected
        self.succeeded += other.succeeded
        self.psnr_sum += other.psnr_sum
        self.ssim_sum += other.ssim_sum
        self.ciede2000_sum += other.ciede2000_sum
        self.lpips_sum += other.lpips_sum
        self.errors.extend(other.errors)

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.expected if self.expected else 0.0

    def raw(self, metric_sum: float) -> float | None:
        return metric_sum / self.succeeded if self.succeeded else None

    @property
    def psnr_raw(self) -> float | None:
        return self.raw(self.psnr_sum)

    @property
    def ssim_raw(self) -> float | None:
        return self.raw(self.ssim_sum)

    @property
    def ciede2000_raw(self) -> float | None:
        return self.raw(self.ciede2000_sum)

    @property
    def lpips_raw(self) -> float | None:
        return self.raw(self.lpips_sum)

    @property
    def psnr_weighted(self) -> float:
        # Equivalent to assigning 0 dB to every missing/failed image.
        return self.psnr_sum / self.expected if self.expected else 0.0

    @property
    def ssim_weighted(self) -> float:
        # Equivalent to assigning SSIM=0 to every missing/failed image.
        return self.ssim_sum / self.expected if self.expected else 0.0

    @property
    def ciede2000_weighted(self) -> float:
        # Missing/failed images receive a conservative Delta-E penalty of 100.
        failures = self.expected - self.succeeded
        return (self.ciede2000_sum + 100.0 * failures) / self.expected if self.expected else 100.0

    @property
    def lpips_weighted(self) -> float:
        # LPIPS is a distance; use 1.0 as the missing/failed-image penalty.
        failures = self.expected - self.succeeded
        return (self.lpips_sum + failures) / self.expected if self.expected else 1.0


@dataclass(frozen=True)
class ReferenceImage:
    relative_path: Path
    rgb: np.ndarray
    lab: np.ndarray


def list_images(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        image = image.resize((size, size), Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.float32) / 255.0


def make_lpips_calculator() -> Callable[[np.ndarray, np.ndarray], float]:
    """Return a standard AlexNet LPIPS calculator for RGB arrays in [0, 1]."""
    import lpips
    import torch

    model = lpips.LPIPS(net="alex").eval()

    def calculate(reference: np.ndarray, candidate: np.ndarray) -> float:
        def to_tensor(image: np.ndarray):
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
            return tensor.mul(2.0).sub(1.0)

        with torch.inference_mode():
            value = model(to_tensor(reference), to_tensor(candidate))
        return float(value.item())

    return calculate


def calculate_metrics(
    reference: np.ndarray,
    reference_lab: np.ndarray,
    candidate: np.ndarray,
    lpips_calculator: Callable[[np.ndarray, np.ndarray], float] | None,
) -> tuple[float, float, float, float | None]:
    psnr = float(peak_signal_noise_ratio(reference, candidate, data_range=1.0))
    # Preserve a finite aggregate if a candidate happens to be pixel-identical.
    if math.isinf(psnr):
        psnr = 100.0
    ssim = float(structural_similarity(reference, candidate, data_range=1.0, channel_axis=2))
    candidate_lab = rgb2lab(candidate)
    ciede2000 = float(np.mean(deltaE_ciede2000(reference_lab, candidate_lab)))
    lpips_value = lpips_calculator(reference, candidate) if lpips_calculator else None
    return psnr, ssim, ciede2000, lpips_value


def prepare_references(gt_root: Path, size: int) -> list[ReferenceImage]:
    references = []
    for gt_path in list_images(gt_root):
        rgb = load_rgb(gt_path, size)
        references.append(ReferenceImage(gt_path.relative_to(gt_root), rgb, rgb2lab(rgb)))
    return references


def evaluate_group(
    references: list[ReferenceImage],
    output_root: Path,
    size: int,
    lpips_calculator: Callable[[np.ndarray, np.ndarray], float] | None,
    workers: int,
) -> MetricAccumulator:
    accumulator = MetricAccumulator(expected=len(references))

    def evaluate_one(reference: ReferenceImage):
        output_path = output_root / reference.relative_path
        if not output_path.is_file():
            return None
        try:
            candidate = load_rgb(output_path, size)
            values = calculate_metrics(reference.rgb, reference.lab, candidate, lpips_calculator)
            return values
        except Exception as error:
            # A present but unreadable image is an execution failure.
            return f"{output_path}: {type(error).__name__}: {error}"

    if workers == 1:
        results = map(evaluate_one, references)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(evaluate_one, references)
    try:
        for result in results:
            if isinstance(result, tuple):
                accumulator.add(result)
            elif isinstance(result, str):
                accumulator.errors.append(result)
    finally:
        if workers != 1:
            executor.shutdown()
    return accumulator


def evaluate_variant_all_models(
    gt_root: Path,
    model_dirs: list[Path],
    language: str,
    split: str,
    variant: str,
    size: int,
    lpips_calculator: Callable[[np.ndarray, np.ndarray], float] | None,
    workers: int,
) -> dict[str, MetricAccumulator]:
    """Stream GT images and evaluate all models without caching a full 1024px set."""
    gt_files = list_images(gt_root)
    accumulators = {
        model_dir.name: MetricAccumulator(expected=len(gt_files)) for model_dir in model_dirs
    }
    active_model_dirs = [
        model_dir for model_dir in model_dirs
        if (model_dir / language / split / variant).is_dir()
    ]
    if not active_model_dirs:
        print(
            f"[{split}/{variant}] no selected model has an output image directory",
            flush=True,
        )
        return accumulators

    executor = ThreadPoolExecutor(max_workers=workers) if workers != 1 else None
    try:
        for gt_index, gt_path in enumerate(gt_files, start=1):
            relative_path = gt_path.relative_to(gt_root)
            output_paths = {
                model_dir.name: model_dir / language / split / variant / relative_path
                for model_dir in active_model_dirs
            }
            present_model_dirs = [
                model_dir for model_dir in active_model_dirs
                if output_paths[model_dir.name].is_file()
            ]

            # A missing output is already represented by expected - succeeded.
            # Avoid decoding a potentially huge GT when no selected model produced it.
            if present_model_dirs:
                reference = load_rgb(gt_path, size)
                reference_lab = rgb2lab(reference)

            def evaluate_model(model_dir: Path):
                output_path = output_paths[model_dir.name]
                try:
                    candidate = load_rgb(output_path, size)
                    values = calculate_metrics(
                        reference, reference_lab, candidate, lpips_calculator
                    )
                    return model_dir.name, values
                except Exception as error:
                    message = f"{output_path}: {type(error).__name__}: {error}"
                    return model_dir.name, message

            results = (
                executor.map(evaluate_model, present_model_dirs)
                if executor is not None
                else map(evaluate_model, present_model_dirs)
            )
            for model_name, result in results:
                if isinstance(result, tuple):
                    accumulators[model_name].add(result)
                elif isinstance(result, str):
                    accumulators[model_name].errors.append(result)

            if gt_index % 10 == 0 or gt_index == len(gt_files):
                print(
                    f"[{split}/{variant}] GT {gt_index}/{len(gt_files)}",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()
    return accumulators


def fmt(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def table_header(include_lpips: bool) -> str:
    columns = [
        "model", "scope", "success", "rate",
        "PSNR_raw", "PSNR_weighted", "SSIM_raw", "SSIM_weighted",
        "CIEDE2000_raw", "CIEDE2000_weighted",
    ]
    if include_lpips:
        columns.extend(("LPIPS_raw", "LPIPS_weighted"))
    return " | ".join(columns)


def table_row(model: str, scope: str, acc: MetricAccumulator, include_lpips: bool) -> str:
    values = [
        model,
        scope,
        f"{acc.succeeded}/{acc.expected}",
        f"{100.0 * acc.success_rate:.2f}%",
        fmt(acc.psnr_raw),
        fmt(acc.psnr_weighted),
        fmt(acc.ssim_raw),
        fmt(acc.ssim_weighted),
        fmt(acc.ciede2000_raw),
        fmt(acc.ciede2000_weighted),
    ]
    if include_lpips:
        values.extend((fmt(acc.lpips_raw), fmt(acc.lpips_weighted)))
    return " | ".join(values)


def write_report(
    report_path: Path,
    records: dict[str, dict[tuple[str, str], MetricAccumulator]],
    language: str,
    size: int,
    include_lpips: bool,
) -> None:
    model_totals: dict[str, MetricAccumulator] = {}
    split_totals: dict[str, dict[str, MetricAccumulator]] = defaultdict(dict)
    for model, groups in records.items():
        total = MetricAccumulator()
        for acc in groups.values():
            total.merge(acc)
        model_totals[model] = total
        for split in SPLITS:
            split_acc = MetricAccumulator()
            for variant in VARIANTS:
                split_acc.merge(groups[(split, variant)])
            split_totals[model][split] = split_acc

    lines = [
        f"{language} image similarity evaluation",
        "=================================",
        "",
        f"Scope: {language}/{{exemplary,user_generated}}/"
        "{image_base,image_variant1,image_variant2}",
        f"Models evaluated: {len(records)}",
        f"Preprocessing: RGB, both GT and output resized to {size}x{size} with bicubic interpolation.",
        "Pairing: exact relative path beneath each image_* directory; non-image files are ignored.",
        "Raw metrics: arithmetic mean over successfully paired and decoded outputs only.",
        "Success rate: successful pairs / valid GT images.",
        "Weighted PSNR: sum(successful PSNR) / all GT (missing/failed output = 0 dB).",
        "Weighted SSIM: sum(successful SSIM) / all GT (missing/failed output = 0).",
        "Weighted CIEDE2000: (sum(successful Delta-E) + 100 * failures) / all GT.",
        "Directions: PSNR and SSIM higher is better; CIEDE2000 lower is better.",
        "Identical-image PSNR is capped at 100 dB for finite aggregation.",
    ]
    if include_lpips:
        lines.extend([
            "LPIPS: AlexNet LPIPS; lower is better; missing/failed output penalty = 1.0.",
            "LPIPS was computed in this report.",
        ])
    else:
        lines.extend([
            "LPIPS was not computed in this report. The implementation is in make_lpips_calculator()",
            "and is enabled by rerunning this script with --with-lpips.",
        ])

    lines.extend(["", "GT counts", "---------"])
    for split in SPLITS:
        counts = [records[next(iter(records))][(split, variant)].expected for variant in VARIANTS]
        lines.append(
            f"{split}: image_base={counts[0]}, image_variant1={counts[1]}, "
            f"image_variant2={counts[2]}, total={sum(counts)}"
        )
    lines.append(f"combined total: {next(iter(model_totals.values())).expected}")

    lines.extend(["", "Overall (both versions and all variants)", "----------------------------------------", table_header(include_lpips)])
    ranked_models = sorted(
        model_totals,
        key=lambda model: (-model_totals[model].ssim_weighted, model),
    )
    for model in ranked_models:
        lines.append(table_row(model, "all", model_totals[model], include_lpips))

    lines.extend(["", "By version", "----------", table_header(include_lpips)])
    for model in sorted(records):
        for split in SPLITS:
            lines.append(table_row(model, split, split_totals[model][split], include_lpips))

    lines.extend(["", "By version and image variant", "----------------------------", table_header(include_lpips)])
    for model in sorted(records):
        for split in SPLITS:
            for variant in VARIANTS:
                scope = f"{split}/{variant}"
                lines.append(table_row(model, scope, records[model][(split, variant)], include_lpips))

    errors = [error for groups in records.values() for acc in groups.values() for error in acc.errors]
    lines.extend(["", "Decode/metric errors", "--------------------"])
    lines.extend(errors or ["None"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-gt", type=Path, default=Path("data_gt"))
    parser.add_argument("--results", type=Path, default=Path("results_final"))
    parser.add_argument("--language", default="Matlab")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional model directory names to evaluate (for example: internvl35-38B)",
    )
    parser.add_argument("--output", type=Path, default=Path("matlab_image_metrics.txt"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--with-lpips", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size < 16:
        raise ValueError("--size must be at least 16")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    lpips_calculator = make_lpips_calculator() if args.with_lpips else None
    if lpips_calculator and args.workers != 1:
        print("LPIPS enabled: forcing --workers=1 for deterministic model inference", flush=True)
        args.workers = 1
    model_dirs = sorted(path for path in args.results.iterdir() if path.is_dir())
    if not model_dirs:
        raise FileNotFoundError(f"No model directories found beneath {args.results}")
    if args.models:
        model_by_name = {path.name: path for path in model_dirs}
        missing_models = sorted(set(args.models) - set(model_by_name))
        if missing_models:
            raise FileNotFoundError(f"Model directories not found: {', '.join(missing_models)}")
        model_dirs = [model_by_name[name] for name in args.models]

    records: dict[str, dict[tuple[str, str], MetricAccumulator]] = defaultdict(dict)
    for split in SPLITS:
        for variant in VARIANTS:
            print(f"Evaluating: {split}/{variant}", flush=True)
            gt_root = args.data_gt / args.language / split / variant
            group_results = evaluate_variant_all_models(
                gt_root,
                model_dirs,
                args.language,
                split,
                variant,
                args.size,
                lpips_calculator,
                args.workers,
            )
            for model_name, accumulator in group_results.items():
                records[model_name][(split, variant)] = accumulator

    write_report(args.output, records, args.language, args.size, args.with_lpips)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
