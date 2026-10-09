#!/usr/bin/env python3
"""Compute code-oriented FigCodeBench metrics and grouped CSV statistics.

Protocol
--------
* Ground truth and generated code are paired by their exact relative path under
  each language directory.
* CrystalBLEU uses the official implementation, 1--4 grams, and the 500 most
  frequent language-specific n-grams learned exclusively from data_gt.
* Normalized AST similarity is the multiset Dice coefficient over normalized
  AST node labels and parent--child productions.  This yields a stable [0, 1]
  structural score across Python, Matlab, R, and LaTeX without the prohibitive
  cubic cost of full tree-edit distance on large plotting programs.
* Plotting-API F1 is a multiset F1 over namespace-normalized plotting API calls.
* CodeBERTScore is computed only for Python with the official
  neulab/codebert-python model and punctuation filtering enabled.

The final CSV contains variant, source-level, and language-level aggregates.
For every score it reports both the mean over matched files and an all-target
mean in which a missing generated file contributes zero.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from crystalbleu import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.util import ngrams
from pygments import lex
from pygments.lexers import get_lexer_for_filename
from pygments.token import Token
from pylatexenc.latexwalker import (
    LatexCharsNode,
    LatexCommentNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexSpecialsNode,
    LatexWalker,
)
from tqdm import tqdm
from tree_sitter import Language, Parser
import tree_sitter_matlab

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data_gt"
RESULTS_ROOT = PROJECT_ROOT / "results_final"
OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = OUTPUT_DIR / "code_oriented_metrics.csv"
CHECKPOINT_DIR = OUTPUT_DIR / "codebert_checkpoints"
R_HELPER = OUTPUT_DIR / "extract_r_ast_features.R"

LANGUAGES = ("python", "Matlab", "R", "latex")
SOURCES = ("exemplary", "user_generated")
VARIANTS = ("base", "variant1", "variant2")
SUFFIXES = {"python": ".py", "Matlab": ".m", "R": ".R", "latex": ".tex"}
CRYSTAL_K = 500

sys.path.insert(0, str(PROJECT_ROOT))
PYTHON_PLOT_NAMES = {
    "figure", "axes", "subplot", "subplots", "subplot2grid", "subplot_mosaic",
    "twinx", "twiny", "add_axes", "add_subplot", "subfigures", "add_subfigure",
    "add_gridspec", "subplots_adjust", "tight_layout", "align_labels",
    "align_xlabels", "align_ylabels", "inset_axes", "secondary_xaxis",
    "secondary_yaxis", "plot", "plot_date", "step", "stairs", "loglog",
    "semilogx", "semilogy", "scatter", "errorbar", "bar", "barh", "bar_label",
    "stem", "eventplot", "pie", "stackplot", "broken_barh", "fill",
    "fill_between", "fill_betweenx", "hist", "hist2d", "boxplot", "violinplot",
    "imshow", "matshow", "pcolor", "pcolormesh", "pcolorfast", "hexbin",
    "specgram", "spy", "contour", "contourf", "tricontour", "tricontourf",
    "tripcolor", "triplot", "quiver", "quiverkey", "barbs", "streamplot",
    "plot3D", "scatter3D", "plot_surface", "plot_wireframe", "plot_trisurf",
    "contour3D", "bar3d", "text3D", "text", "annotate", "title", "suptitle",
    "xlabel", "ylabel", "figtext", "legend", "table", "set_title",
    "set_xlabel", "set_ylabel", "supxlabel", "supylabel", "axis", "grid", "box",
    "margins", "autoscale", "tick_params", "ticklabel_format", "minorticks_on",
    "minorticks_off", "xlim", "ylim", "xscale", "yscale", "xticks", "yticks",
    "set_xlim", "set_ylim", "set_zlim", "set_xscale", "set_yscale", "set_zscale",
    "set_xticks", "set_yticks", "set_zticks", "set_xticklabels", "set_yticklabels",
    "set_zticklabels", "set_aspect", "set_adjustable", "set_axis_off",
    "set_axis_on", "set_rmax", "set_rmin", "set_rorigin", "set_rticks",
    "set_rlabel_position", "set_thetamin", "set_thetamax", "axhline", "axvline",
    "axline", "axhspan", "axvspan", "hlines", "vlines", "arrow", "Line2D", "Arc",
    "Arrow", "Circle", "ConnectionPatch", "Ellipse", "FancyArrowPatch",
    "FancyBboxPatch", "PathPatch", "Polygon", "Rectangle", "Wedge", "add_artist",
    "add_collection", "add_container", "add_image", "add_line", "add_patch",
    "add_table", "colorbar", "colormap", "set_cmap", "clim", "setp",
    "set_facecolor", "set_prop_cycle", "set_label", "set_theme", "set_style",
    "set_context", "despine", "move_legend", "scatterplot", "lineplot", "relplot",
    "histplot", "kdeplot", "ecdfplot", "rugplot", "displot", "stripplot",
    "swarmplot", "boxenplot", "pointplot", "barplot", "countplot", "catplot",
    "regplot", "residplot", "lmplot", "heatmap", "clustermap", "pairplot",
    "jointplot", "FacetGrid", "PairGrid", "JointGrid", "map", "map_dataframe",
    "map_diag", "map_lower", "map_upper",
}
PYTHON_EXCLUDED_CALLS = {"show", "savefig", "load_dataset", "color_palette"}

MATLAB_PLOT_NAMES = {
    "figure", "uifigure", "axes", "uiaxes", "polaraxes", "geoaxes", "subplot",
    "tiledlayout", "nexttile", "yyaxis", "hold", "plot", "plot3", "fplot",
    "fplot3", "fimplicit", "fimplicit3", "loglog", "semilogx", "semilogy",
    "stairs", "area", "stackedplot", "errorbar", "scatter", "scatter3",
    "binscatter", "scatterhistogram", "swarmchart", "swarmchart3", "bubblechart",
    "bubblechart3", "bubblecloud", "bar", "barh", "bar3", "bar3h", "stem",
    "stem3", "pareto", "histogram", "histogram2", "boxchart", "boxplot",
    "violinplot", "raincloudplot", "pie", "pie3", "piechart", "donutchart",
    "dotchart", "wordcloud", "polarplot", "polarscatter", "polarhistogram",
    "polarbubblechart", "fpolarplot", "compass", "compassplot", "geoplot",
    "geoscatter", "geobubble", "geodensityplot", "geolimits", "geobasemap",
    "contour", "contourf", "contour3", "fcontour", "contourslice", "surf",
    "surfc", "surfl", "fsurf", "mesh", "meshc", "meshz", "fmesh", "waterfall",
    "ribbon", "pcolor", "slice", "isosurface", "isonormals", "isocaps", "quiver",
    "quiver3", "feather", "streamline", "streamslice", "streamtube",
    "streamribbon", "streamparticles", "coneplot", "image", "imagesc", "imshow",
    "montage", "imtile", "line", "patch", "surface", "fill", "fill3",
    "rectangle", "viscircles", "title", "subtitle", "sgtitle", "xlabel", "ylabel",
    "zlabel", "legend", "bubblelegend", "text", "annotation", "xline", "yline",
    "xregion", "yregion", "constantplane", "datatip", "texlabel", "axis", "xlim",
    "ylim", "zlim", "xticks", "yticks", "zticks", "xticklabels", "yticklabels",
    "zticklabels", "grid", "box", "colorbar", "colormap", "clim", "caxis",
    "view", "daspect", "pbaspect", "camlight", "lighting", "material", "shading",
    "alpha", "fontname", "fontsize", "set",
}
MATLAB_EXCLUDED_CALLS = {
    "drawnow", "saveas", "savefig", "exportgraphics", "print", "close", "openfig"
}

R_BASE_PLOT_NAMES = {
    "plot.new", "plot.window", "frame", "par", "layout", "split.screen", "screen",
    "close.screen", "plot", "curve", "matplot", "pairs", "coplot", "barplot",
    "dotchart", "hist", "boxplot", "bxp", "pie", "mosaicplot", "assocplot",
    "fourfoldplot", "spineplot", "stripchart", "sunflowerplot", "smoothScatter",
    "contour", "filled.contour", "image", "persp", "stars", "symbols", "stem",
    "points", "lines", "segments", "arrows", "abline", "polygon", "polypath",
    "rect", "text", "mtext", "title", "axis", "box", "grid", "legend", "rug",
    "rasterImage", "xspline", "ggplot", "qplot", "quickplot", "aes", "aes_",
    "guides", "labs", "xlab", "ylab", "ggtitle", "annotate", "annotation_custom",
    "annotation_logticks", "annotation_map", "theme", "theme_gray", "theme_grey",
    "theme_bw", "theme_linedraw", "theme_light", "theme_dark", "theme_minimal",
    "theme_classic", "theme_void", "theme_set", "theme_update", "theme_replace",
    "element_text", "element_line", "element_rect", "element_blank", "element_point",
    "margin", "grid.newpage", "grid.layout", "viewport", "pushViewport",
    "popViewport", "upViewport", "downViewport", "seekViewport", "grid.draw",
    "grid.points", "grid.lines", "grid.polyline", "grid.segments", "grid.arrows",
    "grid.rect", "grid.roundrect", "grid.circle", "grid.ellipse", "grid.polygon",
    "grid.xspline", "grid.curve", "grid.text", "grid.raster", "grid.grill", "xyplot",
    "bwplot", "barchart", "histogram", "densityplot", "stripplot", "qq", "qqmath",
    "splom", "levelplot", "contourplot", "cloud", "wireframe", "parallelplot",
    "panel.xyplot", "panel.points", "panel.lines", "panel.abline", "panel.grid",
    "panel.text", "panel.polygon", "panel.rect", "panel.segments", "panel.arrows",
    "panel.smooth", "panel.loess", "panel.levelplot", "panel.contourplot",
    "trellis.par.set", "update.trellis",
}
R_PLOT_PREFIXES = ("geom_", "stat_", "scale_", "coord_", "facet_", "guide_")
R_EXCLUDED_CALLS = {
    "png", "jpeg", "tiff", "bmp", "pdf", "svg", "dev.new", "dev.off", "ggsave",
    "print.ggplot", "plotly_build", "library", "require", "source",
}

LATEX_PLOT_MACROS = {
    "draw", "path", "fill", "filldraw", "shade", "shadedraw", "clip", "pattern",
    "node", "coordinate", "matrix", "graph", "pic", "addplot", "addplot+",
    "addplot3", "addplot3+", "addlegendentry", "addlegendimage", "legend",
    "pgfplotsset", "pgfplotstabletypeset", "pgfplotstablevertcat", "pgfplotstablesort",
    "tikzset", "tikzstyle", "pgfkeys", "pgfkeysalso", "pgftransformshift",
    "pgftransformxshift", "pgftransformyshift", "pgftransformscale",
    "pgftransformxscale", "pgftransformyscale", "pgftransformrotate", "pgfpathmoveto",
    "pgfpathlineto", "pgfpathcurveto", "pgfpathquadraticcurveto", "pgfpathrectangle",
    "pgfpathcircle", "pgfpathellipse", "pgfpatharc", "pgfpathclose", "pgfusepath",
    "pgftext", "pgfnode", "pgfnodeconnline", "pgfshadepath", "pgfimage",
    "includegraphics", "resizebox", "scalebox", "rotatebox", "reflectbox",
}
LATEX_PLOT_ENVIRONMENTS = {
    "tikzpicture", "axis", "semilogxaxis", "semilogyaxis", "loglogaxis", "polaraxis",
    "ternaryaxis", "smithchart", "groupplot", "pgfpicture",
}

@dataclass(frozen=True)
class Target:
    record_id: str
    language: str
    source: str
    variant: str
    tag: str
    gt_path: Path
    relative_path: Path


@dataclass
class PairMetric:
    target: Target
    matched: bool
    gt_tokens: list[str] = field(default_factory=list)
    pred_tokens: list[str] = field(default_factory=list)
    ast_similarity: float | None = None
    gt_ast_ok: bool = False
    pred_ast_ok: bool = False
    api_precision: float | None = None
    api_recall: float | None = None
    api_f1: float | None = None
    api_intersection: int = 0
    gt_api_count: int = 0
    pred_api_count: int = 0
    codebert_p: float | None = None
    codebert_r: float | None = None
    codebert_f1: float | None = None
    codebert_f3: float | None = None


def read_code(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"\A\s*```(?:python|matlab|r|latex|tex)?\s*\n", "", text, flags=re.I)
    text = re.sub(r"\n\s*```\s*\Z", "\n", text)
    return text


def code_without_comments(code: str, filename: str) -> str:
    try:
        lexer = get_lexer_for_filename(filename)
        return "".join(value for kind, value in lex(code, lexer) if kind not in Token.Comment)
    except Exception:
        return code


def code_tokens(code: str, filename: str) -> list[str]:
    try:
        lexer = get_lexer_for_filename(filename)
        out = []
        for kind, value in lex(code, lexer):
            if kind in Token.Comment or kind in Token.Text.Whitespace:
                continue
            out.extend(re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\w\s]", value, flags=re.UNICODE))
        return out
    except Exception:
        return re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\w\s]", code)


def collect_targets() -> list[Target]:
    targets: list[Target] = []
    for language in LANGUAGES:
        for source in SOURCES:
            language_source_root = DATA_ROOT / language / source
            for variant in VARIANTS:
                root = language_source_root / f"code_{variant}"
                if not root.exists():
                    continue
                suffix = SUFFIXES[language]
                for path in sorted(root.rglob(f"*{suffix}")):
                    rel = path.relative_to(root)
                    tag = rel.parent.as_posix() if rel.parent.as_posix() != "." else "Conceptual" if language == "latex" else "All"
                    overall_rel = Path(source) / f"code_{variant}" / rel
                    targets.append(Target(
                        record_id=f"{language}/{source}/{variant}/{rel.with_suffix('').as_posix()}",
                        language=language,
                        source=source,
                        variant=variant,
                        tag=tag,
                        gt_path=path,
                        relative_path=overall_rel,
                    ))
    return targets


def build_ignored_ngrams(targets: list[Target], gt_cache: dict[Path, tuple[str, list[str]]]) -> dict[str, dict]:
    frequencies: dict[str, Counter] = {lang: Counter() for lang in LANGUAGES}
    for target in tqdm(targets, desc="Learning CrystalBLEU n-grams"):
        _, tokens = gt_cache[target.gt_path]
        for n in range(1, 5):
            frequencies[target.language].update(ngrams(tokens, n))
    return {lang: dict(counter.most_common(CRYSTAL_K)) for lang, counter in frequencies.items()}


def add_tree_features(label: str, children: Iterable, features: Counter, child_fn) -> None:
    # Use an explicit stack: malformed/generated programs can contain more
    # than Python's default 1,000 recursive levels.
    stack = [(label, list(children))]
    while stack:
        parent_label, current_children = stack.pop()
        features[f"N:{parent_label}"] += 1
        for child in current_children:
            child_label, grandchildren = child_fn(child)
            if child_label in {"comment", "whitespace"}:
                continue
            features[f"E:{parent_label}>{child_label}"] += 1
            stack.append((child_label, list(grandchildren)))


def python_ast_features(code: str) -> tuple[Counter, bool]:
    try:
        root = ast.parse(code)
    except Exception:
        return Counter(), False
    features: Counter = Counter()

    def info(node):
        return type(node).__name__, list(ast.iter_child_nodes(node))

    label, children = info(root)
    add_tree_features(label, children, features, info)
    return features, True


MATLAB_LANGUAGE = Language(tree_sitter_matlab.language())
try:
    MATLAB_PARSER = Parser(MATLAB_LANGUAGE)
except TypeError:
    MATLAB_PARSER = Parser()
    MATLAB_PARSER.language = MATLAB_LANGUAGE


def matlab_ast_features(code: str) -> tuple[Counter, bool]:
    root = MATLAB_PARSER.parse(code.encode("utf-8", errors="ignore")).root_node
    features: Counter = Counter()

    def info(node):
        return node.type, list(node.named_children)

    label, children = info(root)
    add_tree_features(label, children, features, info)
    return features, not root.has_error


def latex_ast_features(code: str) -> tuple[Counter, bool]:
    try:
        nodes, _, _ = LatexWalker(code).get_latex_nodes(pos=0)
    except Exception:
        return Counter(), False
    features: Counter = Counter()

    def info(node):
        children = []
        if isinstance(node, LatexMacroNode):
            label = f"macro:{node.macroname}"
            if node.nodeargd:
                children = [x for x in node.nodeargd.argnlist if x is not None]
        elif isinstance(node, LatexEnvironmentNode):
            label = f"environment:{node.environmentname}"
            if node.nodeargd:
                children.extend(x for x in node.nodeargd.argnlist if x is not None)
            children.extend(node.nodelist or [])
        elif isinstance(node, LatexGroupNode):
            label = "group"
            children = list(node.nodelist or [])
        elif isinstance(node, LatexMathNode):
            label = f"math:{node.displaytype}"
            children = list(node.nodelist or [])
        elif isinstance(node, LatexSpecialsNode):
            label = f"special:{node.specials_chars}"
            if node.nodeargd:
                children = [x for x in node.nodeargd.argnlist if x is not None]
        elif isinstance(node, LatexCommentNode):
            label = "comment"
        elif isinstance(node, LatexCharsNode):
            label = "text" if node.chars.strip() else "whitespace"
        else:
            label = type(node).__name__
            children = list(getattr(node, "nodelist", None) or [])
        return label, children

    features["N:document"] += 1
    for node in nodes:
        label, children = info(node)
        if label in {"comment", "whitespace"}:
            continue
        features[f"E:document>{label}"] += 1
        add_tree_features(label, children, features, info)
    return features, True


def parse_counter(encoded: str) -> Counter:
    counter: Counter = Counter()
    if not encoded:
        return counter
    for item in encoded.split("|"):
        key, count = item.rsplit("=", 1)
        counter[unquote(key)] = int(count)
    return counter


def extract_r_features(paths: list[Path], cache_path: Path) -> dict[Path, tuple[Counter, bool]]:
    if cache_path.exists():
        result = {}
        with cache_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                result[Path(row["path"])] = (parse_counter(row["features"]), row["parse_ok"] == "1")
        if all(p in result for p in paths):
            return result
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as manifest:
        for path in paths:
            manifest.write(str(path) + "\n")
        manifest_path = Path(manifest.name)
    try:
        subprocess.run(["Rscript", str(R_HELPER), str(manifest_path), str(cache_path)], check=True)
    finally:
        manifest_path.unlink(missing_ok=True)
    return extract_r_features(paths, cache_path)


def multiset_scores(reference: Counter, candidate: Counter) -> tuple[float, float, float, int, int, int]:
    ref_n, cand_n = sum(reference.values()), sum(candidate.values())
    overlap = sum((reference & candidate).values())
    if ref_n == 0 and cand_n == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0
    precision = overlap / cand_n if cand_n else 0.0
    recall = overlap / ref_n if ref_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, overlap, ref_n, cand_n


def api_calls(language: str, code: str) -> Counter:
    calls: list[str] = []
    if language == "python":
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = None
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if name in PYTHON_PLOT_NAMES and name not in PYTHON_EXCLUDED_CALLS:
                    calls.append(name.lower())
        except Exception:
            names = re.findall(r"(?<![\w.])(?:[A-Za-z_]\w*\.)*([A-Za-z_]\w*)\s*\(", code)
            calls.extend(n.lower() for n in names if n in PYTHON_PLOT_NAMES and n not in PYTHON_EXCLUDED_CALLS)
    elif language == "Matlab":
        names = re.findall(r"(?<![\w.])([A-Za-z]\w*)\s*\(", code)
        calls.extend(n.lower() for n in names if n.lower() in {x.lower() for x in MATLAB_PLOT_NAMES} and n.lower() not in {x.lower() for x in MATLAB_EXCLUDED_CALLS})
    elif language == "R":
        names = re.findall(r"(?<![\w.])((?:[A-Za-z.][\w.]*:::{0,1})?[A-Za-z.][\w.]*)\s*\(", code)
        for raw in names:
            name = raw.split("::")[-1]
            if (name in R_BASE_PLOT_NAMES or name.startswith(R_PLOT_PREFIXES)) and name not in R_EXCLUDED_CALLS:
                calls.append(name.lower())
    else:
        macros = re.findall(r"\\([A-Za-z@]+\+?)", code)
        calls.extend(f"macro:{m.lower()}" for m in macros if m.lower() in {x.lower() for x in LATEX_PLOT_MACROS})
        envs = re.findall(r"\\begin\s*\{([^}]+)\}", code)
        calls.extend(f"environment:{e.lower()}" for e in envs if e.lower() in {x.lower() for x in LATEX_PLOT_ENVIRONMENTS})
    return Counter(calls)


def get_ast(language: str, code: str, path: Path, r_features: dict[Path, tuple[Counter, bool]]) -> tuple[Counter, bool]:
    if language == "python":
        return python_ast_features(code)
    if language == "Matlab":
        return matlab_ast_features(code)
    if language == "R":
        return r_features.get(path, (Counter(), False))
    return latex_ast_features(code)


def group_keys(target: Target):
    yield (target.language, target.source, target.variant, "variant")
    yield (target.language, target.source, "all", "source_all")
    yield (target.language, "all", "all", "language_all")


def mean_or_blank(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def zero_filled_mean(values: list[float], expected: int) -> float | str:
    return sum(values) / expected if expected else ""


def compute_codebert(metrics_by_model: dict[str, list[PairMetric]], model_path: Path, batch_size: int, chunk_size: int) -> None:
    import code_bert_score

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, pairs in metrics_by_model.items():
        python_pairs = [p for p in pairs if p.target.language == "python" and p.matched]
        checkpoint = CHECKPOINT_DIR / f"{model_name}.jsonl"
        completed: dict[str, dict] = {}
        if checkpoint.exists():
            for line in checkpoint.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    completed[row["record_id"]] = row
        pending = [p for p in python_pairs if p.target.record_id not in completed]
        if pending:
            print(f"CodeBERTScore {model_name}: {len(pending)} pending / {len(python_pairs)} matched")
            for start in tqdm(range(0, len(pending), chunk_size), desc=f"CodeBERTScore {model_name}"):
                chunk = pending[start:start + chunk_size]
                cands = [code_without_comments(read_code(RESULTS_ROOT / model_name / p.target.language / p.target.relative_path), p.target.relative_path.name) for p in chunk]
                refs = [code_without_comments(read_code(p.target.gt_path), p.target.gt_path.name) for p in chunk]
                scores = code_bert_score.score(
                    cands=cands,
                    refs=refs,
                    lang="python",
                    model_type=str(model_path),
                    device="cpu",
                    batch_size=batch_size,
                    no_punc=True,
                    chunk_overlap=0.5,
                    verbose=False,
                )
                rows = []
                for pair, p, r, f1, f3 in zip(chunk, *[x.tolist() for x in scores]):
                    row = {"record_id": pair.target.record_id, "p": p, "r": r, "f1": f1, "f3": f3}
                    completed[pair.target.record_id] = row
                    rows.append(row)
                with checkpoint.open("a", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        for pair in python_pairs:
            row = completed.get(pair.target.record_id)
            if row:
                pair.codebert_p = float(row["p"])
                pair.codebert_r = float(row["r"])
                pair.codebert_f1 = float(row["f1"])
                pair.codebert_f3 = float(row["f3"])


def summarize(metrics_by_model: dict[str, list[PairMetric]], ignored: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    smoothing = SmoothingFunction().method3
    for model_name, pairs in metrics_by_model.items():
        groups: dict[tuple, list[PairMetric]] = defaultdict(list)
        for pair in pairs:
            for key in group_keys(pair.target):
                groups[key].append(pair)
        for (language, source, variant, scope), items in sorted(groups.items()):
            matched = [p for p in items if p.matched]
            expected_n, matched_n = len(items), len(matched)
            refs = [[p.gt_tokens] for p in matched]
            hyps = [p.pred_tokens for p in matched]
            crystal = corpus_bleu(refs, hyps, ignoring=ignored[language], smoothing_function=smoothing) if matched else ""
            ast_values = [p.ast_similarity for p in matched if p.ast_similarity is not None]
            api_values = [p.api_f1 for p in matched if p.api_f1 is not None]
            cbp = [p.codebert_p for p in matched if p.codebert_p is not None]
            cbr = [p.codebert_r for p in matched if p.codebert_r is not None]
            cb1 = [p.codebert_f1 for p in matched if p.codebert_f1 is not None]
            cb3 = [p.codebert_f3 for p in matched if p.codebert_f3 is not None]
            if matched:
                api_overlap = sum(p.api_intersection for p in matched)
                gt_api = sum(p.gt_api_count for p in matched)
                pred_api = sum(p.pred_api_count for p in matched)
                micro_p = api_overlap / pred_api if pred_api else (1.0 if gt_api == 0 else 0.0)
                micro_r = api_overlap / gt_api if gt_api else (1.0 if pred_api == 0 else 0.0)
                micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
            else:
                micro_p = micro_r = micro_f1 = ""
            row = {
                "model": model_name,
                "language": language,
                "source": source,
                "variant": variant,
                "scope": scope,
                "expected_files": expected_n,
                "matched_files": matched_n,
                "coverage_rate": matched_n / expected_n if expected_n else "",
                "crystalbleu_corpus_matched": crystal,
                "crystalbleu_corpus_all": (float(crystal) * matched_n / expected_n) if matched and expected_n else "",
                "normalized_ast_similarity_mean_matched": mean_or_blank(ast_values),
                "normalized_ast_similarity_mean_all": zero_filled_mean(ast_values, expected_n),
                "ast_gt_parse_rate": sum(p.gt_ast_ok for p in matched) / matched_n if matched_n else "",
                "ast_prediction_parse_rate": sum(p.pred_ast_ok for p in matched) / matched_n if matched_n else "",
                "plotting_api_precision_micro": micro_p,
                "plotting_api_recall_micro": micro_r,
                "plotting_api_f1_micro": micro_f1,
                "plotting_api_f1_mean_matched": mean_or_blank(api_values),
                "plotting_api_f1_mean_all": zero_filled_mean(api_values, expected_n),
                "codebertscore_precision_mean_matched": mean_or_blank(cbp) if language == "python" else "",
                "codebertscore_recall_mean_matched": mean_or_blank(cbr) if language == "python" else "",
                "codebertscore_f1_mean_matched": mean_or_blank(cb1) if language == "python" else "",
                "codebertscore_f3_mean_matched": mean_or_blank(cb3) if language == "python" else "",
                "codebertscore_f1_mean_all": zero_filled_mean(cb1, expected_n) if language == "python" else "",
                "codebertscore_f3_mean_all": zero_filled_mean(cb3, expected_n) if language == "python" else "",
                "crystalbleu_k": CRYSTAL_K,
                "ast_similarity_definition": "multiset Dice over normalized AST nodes and parent-child productions",
                "plotting_api_f1_definition": "multiset exact API-name F1 after namespace normalization",
            }
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=Path.home() / ".cache/huggingface/hub/models--neulab--codebert-python/snapshots/c4ff459e2a91d002d9bb4ab8cc2b7ee7125ef213")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--codebert-chunk-size", type=int, default=128)
    parser.add_argument("--skip-codebert", action="store_true")
    args = parser.parse_args()

    targets = collect_targets()
    models = sorted(p.name for p in RESULTS_ROOT.iterdir() if p.is_dir())
    print(f"Targets: {len(targets)}; models: {len(models)}")

    gt_cache: dict[Path, tuple[str, list[str]]] = {}
    for target in tqdm(targets, desc="Reading ground truth"):
        code = read_code(target.gt_path)
        gt_cache[target.gt_path] = (code, code_tokens(code, target.gt_path.name))
    ignored = build_ignored_ngrams(targets, gt_cache)

    r_paths = [t.gt_path for t in targets if t.language == "R"]
    for model in models:
        r_paths.extend(RESULTS_ROOT / model / t.language / t.relative_path for t in targets if t.language == "R" and (RESULTS_ROOT / model / t.language / t.relative_path).exists())
    r_paths = sorted(set(r_paths))
    print(f"Extracting/caching R ASTs for {len(r_paths)} files")
    r_features = extract_r_features(r_paths, OUTPUT_DIR / "cache" / "r_ast_features.tsv")

    gt_struct: dict[Path, tuple[Counter, bool, Counter]] = {}
    for target in tqdm(targets, desc="Parsing ground-truth structure"):
        code, _ = gt_cache[target.gt_path]
        ast_features, ok = get_ast(target.language, code, target.gt_path, r_features)
        gt_struct[target.gt_path] = (ast_features, ok, api_calls(target.language, code))

    metrics_by_model: dict[str, list[PairMetric]] = {}
    for model in models:
        pairs: list[PairMetric] = []
        for target in tqdm(targets, desc=f"Comparing {model}", leave=False):
            pred_path = RESULTS_ROOT / model / target.language / target.relative_path
            if not pred_path.exists():
                pairs.append(PairMetric(target=target, matched=False))
                continue
            pred_code = read_code(pred_path)
            pred_tokens = code_tokens(pred_code, pred_path.name)
            gt_code, gt_tokens = gt_cache[target.gt_path]
            gt_ast, gt_ok, gt_api = gt_struct[target.gt_path]
            pred_ast, pred_ok = get_ast(target.language, pred_code, pred_path, r_features)
            _, _, ast_f1, _, _, _ = multiset_scores(gt_ast, pred_ast)
            pred_api = api_calls(target.language, pred_code)
            api_p, api_r, api_f1, api_i, gt_n, pred_n = multiset_scores(gt_api, pred_api)
            pairs.append(PairMetric(
                target=target,
                matched=True,
                gt_tokens=gt_tokens,
                pred_tokens=pred_tokens,
                ast_similarity=ast_f1,
                gt_ast_ok=gt_ok,
                pred_ast_ok=pred_ok,
                api_precision=api_p,
                api_recall=api_r,
                api_f1=api_f1,
                api_intersection=api_i,
                gt_api_count=gt_n,
                pred_api_count=pred_n,
            ))
        metrics_by_model[model] = pairs

    if not args.skip_codebert:
        if not args.model_path.exists():
            raise FileNotFoundError(f"CodeBERT model not found: {args.model_path}")
        compute_codebert(metrics_by_model, args.model_path, args.batch_size, args.codebert_chunk_size)

    rows = summarize(metrics_by_model, ignored)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
