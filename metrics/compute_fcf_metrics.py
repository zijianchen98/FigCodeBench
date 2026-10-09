#!/usr/bin/env python3
"""Compute Fig--Code Fidelity (FCF) for FigCodeBench outputs.

For every expected model/image pair, this script assigns

    FCF_i = E_i * (q_PSNR q_SSIM q_LPIPS q_DeltaE)^(1/4) * C_i,

where E_i is one when both the rendered image and generated source are
available (zero otherwise) and C_i=min(1, L_GT / L_gen) is the one-sided
effective-token compactness factor.  Images are composited on white and
resized to 1024 x 1024 before PSNR, SSIM, and mean CIEDE2000 are computed.

The Delta-E term follows Yang, Ming, and Yu (2012, Eq. 2): their OSCSP score
Q in [0, 5] is evaluated from the mean CIEDE2000 value and q_DeltaE = Q / 5.
This preserves the paper's JND-aware, piecewise perceptual scale.

The output contains one mean FCF value for each model and each of Python,
Matlab, R, and Latex.  Failed or absent samples are retained in the
denominator with FCF=0, so execution/availability is reflected directly.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


CANONICAL_LANGUAGES = ("python", "matlab", "r", "latex")
DISPLAY_LANGUAGE = {
    "python": "python",
    "matlab": "Matlab",
    "r": "R",
    "latex": "latex",
}
IMAGE_TO_VARIANT = {
    "image_base": "base",
    "image_variant1": "variant1",
    "image_variant2": "variant2",
}

# Benchmark figures can legitimately exceed Pillow's conservative safety limit.
Image.MAX_IMAGE_PIXELS = None
CODE_TO_VARIANT = {
    "code_base": "base",
    "code_variant1": "variant1",
    "code_variant2": "variant2",
}


@dataclass(frozen=True)
class ImagePair:
    model: str
    language: str
    source: str
    variant: str
    item: str
    gt_path: Path
    output_path: Path
    lpips: float
    gt_tokens: int
    generated_tokens: int | None


def canonical_language(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized not in CANONICAL_LANGUAGES:
        raise ValueError(f"Unsupported language: {value!r}")
    return normalized


def normalized_stem(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    return path.with_suffix("").as_posix().casefold()


def parse_gt_key(code_file: str, language: str, source: str, variant: str) -> tuple[str, str, str, str]:
    """Build a language/source/variant/item key from a GT code-file path."""
    path = PurePosixPath(code_file.replace("\\", "/"))
    parts = path.parts
    try:
        language_index = next(i for i, part in enumerate(parts) if part.casefold() == language)
        code_variant = CODE_TO_VARIANT[parts[language_index + 2]]
        item = normalized_stem(PurePosixPath(*parts[language_index + 3 :]).as_posix())
    except (StopIteration, KeyError, IndexError) as error:
        raise ValueError(f"Cannot parse GT code path: {code_file}") from error
    if code_variant != variant:
        raise ValueError(f"Variant mismatch in GT row: {code_file}")
    return language, source.casefold(), variant, item


def parse_generated_key(relative_path: str, language: str) -> tuple[str, str, str, str, str]:
    """Return model plus the same key fields used by image and GT records."""
    parts = PurePosixPath(relative_path.replace("\\", "/")).parts
    if len(parts) < 5:
        raise ValueError(f"Cannot parse generated-code path: {relative_path}")
    model, path_language, source, code_variant = parts[:4]
    canonical = canonical_language(path_language)
    if canonical != language:
        raise ValueError(f"Language mismatch in generated-code row: {relative_path}")
    if code_variant not in CODE_TO_VARIANT:
        raise ValueError(f"Unknown generated-code variant: {relative_path}")
    return model, canonical, source.casefold(), CODE_TO_VARIANT[code_variant], normalized_stem(
        PurePosixPath(*parts[4:]).as_posix()
    )


def parse_image_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    language = canonical_language(row["language"])
    variant = IMAGE_TO_VARIANT[row["variant"]]
    return (
        row["model"],
        language,
        row["split"].casefold(),
        variant,
        normalized_stem(row["relative_path"]),
    )


def parse_token_count(value: str) -> int | None:
    try:
        tokens = int(value)
    except (TypeError, ValueError):
        return None
    return tokens if tokens > 0 else None


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            white = Image.new("RGBA", rgba.size, "white")
            image = Image.alpha_composite(white, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        image = image.resize((size, size), Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.float32) / 255.0


def yang_delta_e_similarity(delta_e: float) -> float:
    """Map mean CIEDE2000 to Yang et al.'s OSCSP Q/5 (Eq. 2)."""
    if not math.isfinite(delta_e) or delta_e > 24.0:
        return 0.0
    if delta_e < 0.5:
        return 1.0
    # (lower bound, upper bound, score at lower bound), for k=2,...,6.
    for lower, upper, score in (
        (0.5, 1.5, 5.0),
        (1.5, 3.0, 4.0),
        (3.0, 6.0, 3.0),
        (6.0, 12.0, 2.0),
        (12.0, 24.0, 1.0),
    ):
        if delta_e <= upper:
            quality = score - (delta_e - lower) / (upper - lower)
            return float(np.clip(quality / 5.0, 0.0, 1.0))
    return 0.0


def visual_quality(reference: np.ndarray, reference_lab: np.ndarray, candidate_path: Path, lpips_value: float, size: int) -> float:
    candidate = load_rgb(candidate_path, size)
    psnr = float(peak_signal_noise_ratio(reference, candidate, data_range=1.0))
    if math.isinf(psnr):
        psnr = 100.0
    ssim = float(structural_similarity(reference, candidate, data_range=1.0, channel_axis=2))
    delta_e = float(np.mean(deltaE_ciede2000(reference_lab, rgb2lab(candidate))))

    q_psnr = 1.0 - 10.0 ** (-max(psnr, 0.0) / 20.0)
    q_ssim = float(np.clip(ssim, 0.0, 1.0))
    q_lpips = 1.0 / (1.0 + max(lpips_value, 0.0))
    q_delta_e = yang_delta_e_similarity(delta_e)
    return float((q_psnr * q_ssim * q_lpips * q_delta_e) ** 0.25)


def process_item(task: tuple[tuple[str, str, str, str], list[ImagePair]], size: int) -> list[tuple[str, str, float]]:
    """Evaluate every model output belonging to one GT image, once loaded."""
    _, pairs = task
    if not pairs:
        return []
    try:
        reference = load_rgb(pairs[0].gt_path, size)
        reference_lab = rgb2lab(reference)
    except Exception:
        return []

    results: list[tuple[str, str, float]] = []
    for pair in pairs:
        # Missing/empty/unreadable generated source gives FCF=0 by construction.
        if pair.generated_tokens is None or not pair.output_path.is_file():
            continue
        try:
            visual = visual_quality(reference, reference_lab, pair.output_path, pair.lpips, size)
            # One-sided compactness: longer generated scripts are penalized, shorter
            # scripts receive no bonus above the figure-fidelity ceiling.
            compactness = min(1.0, pair.gt_tokens / pair.generated_tokens)
            results.append((pair.model, pair.language, visual * compactness))
        except Exception:
            # Any image decoding/metric failure remains the pre-initialized zero.
            continue
    return results


def make_pairs(
    gt_rows: Iterable[dict[str, str]],
    generated_rows: Iterable[dict[str, str]],
    lpips_rows: Iterable[dict[str, str]],
    root: Path,
) -> tuple[dict[tuple[str, str, str, str], list[ImagePair]], dict[tuple[str, str], int], dict[tuple[str, str, str, str], int]]:
    gt_tokens: dict[tuple[str, str, str, str], int] = {}
    for row in gt_rows:
        if row.get("token_status") != "ok":
            continue
        language = canonical_language(row["language"])
        tokens = parse_token_count(row.get("effective_tokens_llama3_8b", ""))
        if tokens is None:
            continue
        key = parse_gt_key(row["code_file"], language, row["source"], row["variant"])
        if key in gt_tokens:
            raise ValueError(f"Duplicate GT key: {key}")
        gt_tokens[key] = tokens

    generated_tokens: dict[tuple[str, str, str, str, str], int] = {}
    for row in generated_rows:
        if row.get("token_status") != "ok":
            continue
        language = canonical_language(row["language"])
        tokens = parse_token_count(row.get("effective_tokens_llama3_8b", ""))
        if tokens is None:
            continue
        key = parse_generated_key(row["relative_path"], language)
        # Retain the first duplicate path deterministically; duplicated source files
        # are not expected, but should not inflate a sample's compactness.
        generated_tokens.setdefault(key, tokens)

    by_gt: dict[tuple[str, str, str, str], list[ImagePair]] = defaultdict(list)
    expected: dict[tuple[str, str], int] = defaultdict(int)
    encountered_gt: dict[tuple[str, str, str, str], int] = {}
    for row in lpips_rows:
        model, language, source, variant, item = parse_image_key(row)
        expected[(model, language)] += 1
        base_key = (language, source, variant, item)
        encountered_gt.setdefault(base_key, 0)
        if row.get("status") != "ok":
            continue
        try:
            lpips_value = float(row["LPIPS"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(lpips_value):
            continue
        generated = generated_tokens.get((model, language, source, variant, item))
        gt_count = gt_tokens.get(base_key)
        if generated is None or gt_count is None:
            continue
        pair = ImagePair(
            model=model,
            language=language,
            source=source,
            variant=variant,
            item=item,
            gt_path=root / row["gt_path"].replace("\\", "/"),
            output_path=root / row["output_path"].replace("\\", "/"),
            lpips=lpips_value,
            gt_tokens=gt_count,
            generated_tokens=generated,
        )
        by_gt[base_key].append(pair)
        encountered_gt[base_key] += 1
    return by_gt, expected, encountered_gt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-tokens", type=Path, default=Path("generated_code_effective_tokens.csv"))
    parser.add_argument("--gt-tokens", type=Path, default=Path("llama3_8b_effective_tokens.csv"))
    parser.add_argument("--lpips-details", type=Path, default=Path("lpips_results_1024x1024/lpips_pair_details.csv"))
    parser.add_argument("--output", type=Path, default=Path("_fcf_metrics_by_model_language_unchecked.csv"))
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    root = Path.cwd()
    gt_rows = load_csv(args.gt_tokens)
    generated_rows = load_csv(args.generated_tokens)
    lpips_rows = load_csv(args.lpips_details)
    by_gt, expected, _ = make_pairs(gt_rows, generated_rows, lpips_rows, root)

    totals: dict[tuple[str, str], float] = defaultdict(float)
    tasks = sorted(by_gt.items())
    print(f"Computing image metrics for {sum(len(pairs) for _, pairs in tasks):,} successful image/code pairs "
          f"across {len(tasks):,} GT images at {args.size}x{args.size}.")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for result in tqdm(executor.map(lambda task: process_item(task, args.size), tasks), total=len(tasks), desc="FCF image pairs"):
            for model, language, fcf in result:
                totals[(model, language)] += fcf

    models = sorted({model for model, _ in expected})
    output_rows = []
    for model in models:
        record = {"model": model}
        for language in CANONICAL_LANGUAGES:
            denominator = expected.get((model, language), 0)
            mean = totals[(model, language)] / denominator if denominator else 0.0
            record[f"FCF_{DISPLAY_LANGUAGE[language]}"] = f"{mean:.6f}"
        output_rows.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "FCF_python", "FCF_Matlab", "FCF_R", "FCF_latex"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} model rows to {args.output}")
    for model in models:
        coverage = ", ".join(
            f"{DISPLAY_LANGUAGE[language]}={expected.get((model, language), 0)}"
            for language in CANONICAL_LANGUAGES
        )
        print(f"  {model}: {coverage}")


if __name__ == "__main__":
    main()
