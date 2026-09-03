"""Build the standalone, static results dashboard for a completed sweep.

One HTML file, no server, no CDN, no build step -- open it from disk or
commit it to a repository and it renders. Everything it needs (charts as
inline SVG, sample images as data URIs) is embedded. This is a *reporting*
tool over a finished sweep, distinct from ``training.legacy``'s
``dashboard_app.py`` (a live FastAPI app for uploading and scoring a single
image against the production inference path).

The page is arranged as an argument rather than a dump of tables:

1. the headline -- one model, every category, and what it costs;
2. the regime comparison that justifies that headline;
3. the accuracy/latency frontier, which is what makes it an engineering
   choice rather than a leaderboard entry;
4. per-category detail, so a reader can find where it is weak;
5. localization evidence -- heatmaps against ground truth, including failures.

Organized in two parts:

1. Chart primitives (:func:`bar_chart`, :func:`scatter_pareto`,
   :func:`heatmap`, :func:`roc_curves`, :func:`histogram`) -- inline-SVG
   marks, since a charting library would break the single-file constraint.
2. The dashboard itself: aggregation, the image/heatmap gallery, and the
   page template.

Usage:
    python -m training.report
    python -m training.report --open
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from training.store import artifact_path, load_artifact

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "training" / "benchmark_results"

# -----------------------------------------------------------------------------
# Inline-SVG chart primitives
# -----------------------------------------------------------------------------
#
# The dashboard has to be one file that opens from disk with no server, no
# CDN and no build step -- something to drop in a repository and open in
# front of someone. That rules out a charting library, so the marks are
# emitted as SVG directly.
#
# Colors come from the data-viz reference palette unmodified, and each form
# stays inside that palette's documented caps: scatter and other all-pairs
# forms use only the first three categorical slots, bars and lines may use
# the adjacent order, and magnitude uses the single-hue blue ramp. Every
# color is emitted as a CSS custom property so light and dark modes swap in
# one place.

#: Categorical slots, in the palette's fixed order. Never cycled: a ninth
#: series folds into "Other" rather than inventing a hue.
SERIES = ("--series-1", "--series-2", "--series-3", "--series-4", "--series-5")

#: Blue ramp used for magnitude (heatmap cells), light to dark.
SEQUENTIAL = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
)


@dataclass(frozen=True)
class Point:
    """One scatter mark.

    Attributes:
        x: Horizontal value.
        y: Vertical value.
        label: Name used for the tooltip and any direct label.
        highlight: Whether this mark is on the Pareto frontier.
    """

    x: float
    y: float
    label: str
    highlight: bool = False


def esc(text: object) -> str:
    """HTML-escape a value for safe inclusion in markup."""
    return html.escape(str(text), quote=True)


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Pick round tick values spanning ``[low, high]``."""
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return [low]
    raw_step = (high - low) / max(1, count)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for multiple in (1, 2, 2.5, 5, 10):
        step = magnitude * multiple
        if step >= raw_step:
            break
    start = math.floor(low / step) * step
    ticks = []
    value = start
    while value <= high + step * 0.5:
        if value >= low - step * 0.5:
            ticks.append(round(value, 10))
        value += step
    return ticks


def bar_chart(
    categories: list[str],
    series: list[tuple[str, list[float]]],
    *,
    width: int = 760,
    height: int = 320,
    value_format: str = "{:.3f}",
    y_label: str = "",
) -> str:
    """Grouped bar chart.

    Bars use the adjacent categorical order, which the palette validates for
    up to eight series. Data-ends are rounded and anchored to the baseline, and
    adjacent fills carry a 2px surface gap.

    Args:
        categories: One group per category.
        series: ``(name, values)`` pairs, one per series.
        width: SVG width.
        height: SVG height.
        value_format: Format string for direct labels.
        y_label: Axis caption.

    Returns:
        SVG markup.
    """
    if not categories or not series:
        return ""

    pad_left, pad_right, pad_top, pad_bottom = 56, 16, 24, 56
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    values = [v for _, vals in series for v in vals if math.isfinite(v)]
    if not values:
        return ""
    top = max(values)
    bottom = min(min(values), 0.0)
    # Anomaly-detection scores cluster near 1.0; a zero-based axis would make
    # every bar look identical. Start just below the data instead, and say so
    # in the caption rather than hiding the truncation.
    if bottom >= 0.5 and top <= 1.0:
        bottom = max(0.0, min(values) - 0.05)
    span = max(top - bottom, 1e-9)

    group_w = plot_w / len(categories)
    bar_gap = 2.0
    bar_w = max(3.0, (group_w * 0.72 - bar_gap * (len(series) - 1)) / len(series))

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
        f'preserveAspectRatio="xMidYMid meet">'
    ]

    for tick in _nice_ticks(bottom, top):
        y = pad_top + plot_h - (tick - bottom) / span * plot_h
        parts.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" '
            f'class="grid"/>'
            f'<text x="{pad_left - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">'
            f"{tick:.2f}</text>"
        )

    for group_index, category in enumerate(categories):
        group_x = pad_left + group_index * group_w
        for series_index, (name, vals) in enumerate(series):
            value = vals[group_index] if group_index < len(vals) else float("nan")
            if not math.isfinite(value):
                continue
            bar_h = max(0.0, (value - bottom) / span * plot_h)
            x = group_x + group_w * 0.14 + series_index * (bar_w + bar_gap)
            y = pad_top + plot_h - bar_h
            color = f"var({SERIES[series_index % len(SERIES)]})"
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                f'rx="4" fill="{color}"><title>{esc(name)} / {esc(category)}: '
                f"{value_format.format(value)}</title></rect>"
            )
        parts.append(
            f'<text x="{group_x + group_w / 2:.1f}" y="{height - pad_bottom + 18}" '
            f'class="tick" text-anchor="middle">{esc(category)}</text>'
        )

    parts.append(
        f'<line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{width - pad_right}" '
        f'y2="{pad_top + plot_h}" class="axis"/>'
    )
    if y_label:
        parts.append(
            f'<text x="{pad_left}" y="{pad_top - 8}" class="axis-label">{esc(y_label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def scatter_pareto(
    points: list[Point],
    *,
    width: int = 760,
    height: int = 400,
    x_label: str = "",
    y_label: str = "",
    x_log: bool = True,
) -> str:
    """Accuracy-versus-cost scatter with the Pareto frontier picked out.

    Only one hue is used. Frontier marks take categorical slot 1 and are
    direct-labeled; everything else is muted ink. That sidesteps the all-pairs
    color cap entirely and puts the emphasis where the decision is -- a reader
    should be able to see the shippable options without decoding a legend.

    Args:
        points: Marks to plot; ``highlight`` flags frontier membership.
        width: SVG width.
        height: SVG height.
        x_label: Horizontal axis caption.
        y_label: Vertical axis caption.
        x_log: Whether to use a log scale on x, which latency usually needs.

    Returns:
        SVG markup.
    """
    finite = [p for p in points if math.isfinite(p.x) and math.isfinite(p.y) and p.x > 0]
    if not finite:
        return ""

    pad_left, pad_right, pad_top, pad_bottom = 60, 120, 24, 52
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def fx(value: float) -> float:
        return math.log10(value) if x_log else value

    xs = [fx(p.x) for p in finite]
    ys = [p.y for p in finite]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    x_span = max(x_hi - x_lo, 1e-9)
    y_pad = max((y_hi - y_lo) * 0.12, 1e-4)
    y_lo, y_hi = y_lo - y_pad, min(1.0, y_hi + y_pad)
    y_span = max(y_hi - y_lo, 1e-9)

    def px(value: float) -> float:
        return pad_left + (fx(value) - x_lo) / x_span * plot_w

    def py(value: float) -> float:
        return pad_top + plot_h - (value - y_lo) / y_span * plot_h

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
        f'preserveAspectRatio="xMidYMid meet">'
    ]

    for tick in _nice_ticks(y_lo, y_hi):
        y = py(tick)
        if pad_top - 1 <= y <= pad_top + plot_h + 1:
            parts.append(
                f'<line x1="{pad_left}" y1="{y:.1f}" x2="{pad_left + plot_w}" y2="{y:.1f}" '
                f'class="grid"/><text x="{pad_left - 8}" y="{y + 4:.1f}" class="tick" '
                f'text-anchor="end">{tick:.3f}</text>'
            )

    for tick in _nice_ticks(x_lo, x_hi, 4):
        value = 10**tick if x_log else tick
        x = px(value)
        if pad_left - 1 <= x <= pad_left + plot_w + 1:
            shown = f"{value:.0f}" if value >= 10 else f"{value:.1f}"
            parts.append(
                f'<text x="{x:.1f}" y="{pad_top + plot_h + 20}" class="tick" '
                f'text-anchor="middle">{shown}</text>'
            )

    frontier = sorted((p for p in finite if p.highlight), key=lambda p: p.x)
    if len(frontier) > 1:
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{px(p.x):.1f},{py(p.y):.1f}" for i, p in enumerate(frontier)
        )
        parts.append(f'<path d="{path}" class="frontier"/>')

    for point in finite:
        if point.highlight:
            continue
        parts.append(
            f'<circle cx="{px(point.x):.1f}" cy="{py(point.y):.1f}" r="5" class="dot-muted">'
            f"<title>{esc(point.label)}: {point.y:.4f} @ {point.x:.1f} ms</title></circle>"
        )
    for point in frontier:
        cx, cy = px(point.x), py(point.y)
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" class="dot-front">'
            f"<title>{esc(point.label)}: {point.y:.4f} @ {point.x:.1f} ms</title></circle>"
            f'<text x="{cx + 11:.1f}" y="{cy + 4:.1f}" class="point-label">'
            f"{esc(point.label)}</text>"
        )

    parts.append(
        f'<line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{pad_left + plot_w}" '
        f'y2="{pad_top + plot_h}" class="axis"/>'
    )
    if x_label:
        parts.append(
            f'<text x="{pad_left + plot_w / 2:.1f}" y="{height - 8}" class="axis-label" '
            f'text-anchor="middle">{esc(x_label)}</text>'
        )
    if y_label:
        parts.append(
            f'<text x="{pad_left}" y="{pad_top - 8}" class="axis-label">{esc(y_label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def heatmap(
    row_labels: list[str],
    column_labels: list[str],
    values: list[list[float]],
    *,
    cell: int = 34,
    label_width: int = 210,
) -> str:
    """Magnitude grid on the single-hue blue ramp.

    Args:
        row_labels: One label per row (methods).
        column_labels: One label per column (categories).
        values: ``values[row][column]``; ``nan`` renders as an empty cell.
        cell: Cell edge length in pixels.
        label_width: Width reserved for row labels.

    Returns:
        SVG markup.
    """
    if not row_labels or not column_labels:
        return ""

    top_pad = 96
    width = label_width + len(column_labels) * cell + 12
    height = top_pad + len(row_labels) * cell + 8

    flat = [v for row in values for v in row if math.isfinite(v)]
    if not flat:
        return ""
    low, high = min(flat), max(flat)
    span = max(high - low, 1e-9)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart heatmap" '
        f'preserveAspectRatio="xMinYMin meet">'
    ]

    for column_index, label in enumerate(column_labels):
        x = label_width + column_index * cell + cell / 2
        parts.append(
            f'<text x="{x:.1f}" y="{top_pad - 10}" class="tick" text-anchor="start" '
            f'transform="rotate(-60 {x:.1f} {top_pad - 10})">{esc(label)}</text>'
        )

    for row_index, row_label in enumerate(row_labels):
        y = top_pad + row_index * cell
        parts.append(
            f'<text x="{label_width - 10}" y="{y + cell / 2 + 4:.1f}" class="tick" '
            f'text-anchor="end">{esc(row_label)}</text>'
        )
        for column_index in range(len(column_labels)):
            value = (
                values[row_index][column_index]
                if column_index < len(values[row_index])
                else float("nan")
            )
            x = label_width + column_index * cell
            if not math.isfinite(value):
                parts.append(
                    f'<rect x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" '
                    f'rx="3" class="cell-empty"/>'
                )
                continue
            step = SEQUENTIAL[min(len(SEQUENTIAL) - 1, int((value - low) / span * len(SEQUENTIAL)))]
            # Ink flips on the darker half of the ramp so the value stays legible.
            ink = "#ffffff" if (value - low) / span > 0.55 else "#0b0b0b"
            parts.append(
                f'<rect x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" '
                f'rx="3" fill="{step}"><title>{esc(row_label)} / '
                f"{esc(column_labels[column_index])}: {value:.4f}</title></rect>"
                f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 3.5:.1f}" '
                f'text-anchor="middle" class="cell-text" fill="{ink}">'
                f"{value * 100:.0f}</text>"
            )

    parts.append("</svg>")
    return "".join(parts)


def roc_curves(
    curves: list[tuple[str, list[float], list[float]]],
    *,
    width: int = 520,
    height: int = 400,
) -> str:
    """ROC curves for a handful of methods.

    Lines use the adjacent categorical order, which the palette validates for
    this form. Kept to a few series and direct-labeled at the curve end, so
    identity never rests on color alone.

    Args:
        curves: ``(name, fpr, tpr)`` triples.
        width: SVG width.
        height: SVG height.

    Returns:
        SVG markup.
    """
    if not curves:
        return ""

    pad_left, pad_right, pad_top, pad_bottom = 52, 16, 20, 48
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
        f'preserveAspectRatio="xMidYMid meet">'
    ]

    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = pad_top + plot_h - tick * plot_h
        x = pad_left + tick * plot_w
        parts.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{pad_left + plot_w}" y2="{y:.1f}" '
            f'class="grid"/>'
            f'<text x="{pad_left - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">'
            f"{tick:.2f}</text>"
            f'<text x="{x:.1f}" y="{pad_top + plot_h + 18}" class="tick" '
            f'text-anchor="middle">{tick:.2f}</text>'
        )

    # Chance line: the reference every ROC is read against.
    parts.append(
        f'<line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{pad_left + plot_w}" '
        f'y2="{pad_top}" class="chance"/>'
    )

    for index, (name, fpr, tpr) in enumerate(curves[: len(SERIES)]):
        color = f"var({SERIES[index % len(SERIES)]})"
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{pad_left + f * plot_w:.1f},"
            f"{pad_top + plot_h - t * plot_h:.1f}"
            for i, (f, t) in enumerate(zip(fpr, tpr, strict=True))
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2">'
            f"<title>{esc(name)}</title></path>"
        )

    parts.append(
        f'<line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{pad_left + plot_w}" '
        f'y2="{pad_top + plot_h}" class="axis"/>'
        f'<text x="{pad_left + plot_w / 2:.1f}" y="{height - 8}" class="axis-label" '
        f'text-anchor="middle">False positive rate</text>'
        f'<text x="{pad_left}" y="{pad_top - 6}" class="axis-label">True positive rate</text>'
        "</svg>"
    )
    return "".join(parts)


def histogram(
    normal: list[float],
    anomalous: list[float],
    *,
    width: int = 520,
    height: int = 300,
    bins: int = 36,
    threshold: float | None = None,
) -> str:
    """Overlaid score distributions for normal and anomalous images.

    This is the chart that shows *why* a model works or does not: two
    separated humps mean a threshold exists, overlap means it does not, and
    the shape of the overlap says whether the failures are scrap or escapes.

    Args:
        normal: Scores of normal images.
        anomalous: Scores of anomalous images.
        width: SVG width.
        height: SVG height.
        bins: Histogram bin count.
        threshold: Optional decision threshold to mark.

    Returns:
        SVG markup.
    """
    values = [v for v in (*normal, *anomalous) if math.isfinite(v)]
    if not values:
        return ""

    low, high = min(values), max(values)
    span = max(high - low, 1e-9)
    pad_left, pad_right, pad_top, pad_bottom = 44, 16, 24, 46
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def histify(data: list[float]) -> list[int]:
        counts = [0] * bins
        for value in data:
            if math.isfinite(value):
                index = min(bins - 1, int((value - low) / span * bins))
                counts[index] += 1
        return counts

    normal_counts = histify(normal)
    anomalous_counts = histify(anomalous)
    peak = max(1, max((*normal_counts, *anomalous_counts)))
    bar_w = plot_w / bins

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
        f'preserveAspectRatio="xMidYMid meet">'
    ]

    for counts, slot, name in (
        (normal_counts, SERIES[0], "normal"),
        (anomalous_counts, SERIES[1], "anomalous"),
    ):
        for index, count in enumerate(counts):
            if count == 0:
                continue
            bar_h = count / peak * plot_h
            x = pad_left + index * bar_w
            parts.append(
                f'<rect x="{x + 0.5:.1f}" y="{pad_top + plot_h - bar_h:.1f}" '
                f'width="{max(1.0, bar_w - 1):.1f}" height="{bar_h:.1f}" rx="2" '
                f'fill="var({slot})" fill-opacity="0.62">'
                f"<title>{name}: {count} images</title></rect>"
            )

    if threshold is not None and math.isfinite(threshold):
        x = pad_left + (threshold - low) / span * plot_w
        if pad_left <= x <= pad_left + plot_w:
            parts.append(
                f'<line x1="{x:.1f}" y1="{pad_top}" x2="{x:.1f}" y2="{pad_top + plot_h}" '
                f'class="threshold"/>'
                f'<text x="{x:.1f}" y="{pad_top - 8}" class="tick" text-anchor="middle">'
                f"threshold</text>"
            )

    parts.append(
        f'<line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{pad_left + plot_w}" '
        f'y2="{pad_top + plot_h}" class="axis"/>'
        f'<text x="{pad_left}" y="{height - 8}" class="tick">{low:.3g}</text>'
        f'<text x="{pad_left + plot_w:.1f}" y="{height - 8}" class="tick" '
        f'text-anchor="end">{high:.3g}</text>'
        f'<text x="{pad_left + plot_w / 2:.1f}" y="{height - 8}" class="axis-label" '
        f'text-anchor="middle">anomaly score</text>'
        "</svg>"
    )
    return "".join(parts)


# -----------------------------------------------------------------------------
# The dashboard: aggregation, the gallery, and the page itself
# -----------------------------------------------------------------------------

#: Methods shown in the per-category heatmap and ROC panel.
_TOP_METHODS = 12
_ROC_METHODS = 4

#: Gallery size. Enough to be convincing, small enough to keep the file
#: portable -- every image is embedded, so this directly sets the page weight.
_GALLERY_HITS = 6
_GALLERY_MISSES = 6
_GALLERY_PX = 190


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping malformed lines."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last row per ``(regime, method, config)``."""
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("regime", "oneclass"), row.get("method", ""), row.get("config", ""))
        latest[key] = row
    return list(latest.values())


def _finite(value: Any) -> float:
    """Coerce to float, mapping missing or NaN to ``nan``."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def _mean(values: list[float]) -> float:
    """Mean over finite values, or ``nan`` if there are none."""
    finite = [v for v in values if v == v]
    return float(np.mean(finite)) if finite else float("nan")


def shared_configs(rows: list[dict[str, Any]], regimes: list[str]) -> set[str]:
    """Configurations that every named regime actually covers.

    Multi-class is only defined for a family with more than one category, so
    it never covers the single-category corpora that one-class does.
    Comparing means over different config sets would attribute to the regime a
    difference that is partly just a different set of categories.
    """
    per_regime = [
        {r["config"] for r in rows if r.get("status") == "ok" and r.get("regime") == regime}
        for regime in regimes
    ]
    populated = [configs for configs in per_regime if configs]
    return set.intersection(*populated) if populated else set()


def aggregate(
    rows: list[dict[str, Any]], regime: str, only_configs: set[str] | None = None
) -> list[dict[str, Any]]:
    """Summarize one regime into a per-method ranking.

    Args:
        rows: All result rows.
        regime: Regime to summarize.
        only_configs: Restrict to these configurations, for cross-regime
            comparisons that must be like-for-like.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok" or row.get("regime") != regime:
            continue
        if only_configs is not None and row.get("config") not in only_configs:
            continue
        grouped[row["method"]].append(row)

    total_configs = len({r["config"] for r in rows if r.get("regime") == regime})
    summary: list[dict[str, Any]] = []
    for method, entries in grouped.items():
        aurocs = [_finite(e.get("auroc")) for e in entries]
        summary.append(
            {
                "method": method,
                "family": entries[0].get("family", ""),
                "backend": entries[0].get("backend", ""),
                "configs": len({e["config"] for e in entries}),
                "complete": len({e["config"] for e in entries}) == total_configs,
                "auroc": _mean(aurocs),
                "min_auroc": min((v for v in aurocs if v == v), default=float("nan")),
                "ap": _mean([_finite(e.get("average_precision")) for e in entries]),
                "f1": _mean([_finite(e.get("f1_max")) for e in entries]),
                "scrap": _mean([_finite(e.get("fpr_at_95tpr")) for e in entries]),
                "escape": _mean([_finite(e.get("fnr_at_1fpr")) for e in entries]),
                "aupro": _mean([_finite(e.get("aupro")) for e in entries]),
                "pixel_auroc": _mean([_finite(e.get("pixel_auroc")) for e in entries]),
                "ms": _mean([_finite(e.get("ms_per_image")) for e in entries]),
                "vram": _mean([_finite(e.get("peak_vram_gb")) for e in entries]),
                "fit_s": _mean([_finite(e.get("fit_seconds")) for e in entries]),
            }
        )
    summary.sort(
        key=lambda row: (
            not row["complete"],
            -(row["auroc"] if row["auroc"] == row["auroc"] else -1),
        )
    )
    return summary


def pareto_front(points: list[Point]) -> list[Point]:
    """Mark the points that no other point beats on both axes.

    A model is on the frontier when nothing is simultaneously faster *and*
    more accurate. Those are the only defensible choices; everything else is
    dominated, and picking one means accepting a worse model for no gain.
    """
    ordered = sorted(points, key=lambda p: (p.x, -p.y))
    best_y = -float("inf")
    marked: list[Point] = []
    for point in ordered:
        if point.y > best_y:
            best_y = point.y
            marked.append(Point(point.x, point.y, point.label, highlight=True))
        else:
            marked.append(point)
    return marked


def _roc_points(scores: np.ndarray, labels: np.ndarray, max_points: int = 120):
    """Compute a decimated ROC curve suitable for drawing."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, np.nan_to_num(scores, nan=0.0))
    if len(fpr) > max_points:
        index = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr, tpr = fpr[index], tpr[index]
    return [float(v) for v in fpr], [float(v) for v in tpr]


def _encode_image(path: Path, size: int) -> str | None:
    """Read an image, resize it, and return a JPEG data URI."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    scale = size / max(height, width)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def _encode_heatmap(image_path: Path, anomaly_map: np.ndarray, size: int) -> str | None:
    """Render an anomaly map over its source image as a JPEG data URI."""
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    scale = size / max(height, width)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    resized = cv2.resize(
        anomaly_map.astype(np.float32),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    low, high = float(resized.min()), float(resized.max())
    normalized = (resized - low) / max(high - low, 1e-9)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    blended = cv2.addWeighted(image, 0.55, colored, 0.45, 0)

    ok, buffer = cv2.imencode(".jpg", blended, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def build_gallery(artifact_root: Path, regime: str, method: str, config_key: str) -> str:
    """Build the heatmap gallery for one method on one config.

    Deliberately shows the misses as well as the hits. A page of successes is
    a sales deck; the failures are what tell a reviewer whether the model is
    trustworthy, and being able to explain them is the point.
    """
    # This legacy static dashboard predates seeded artifacts; seed 0 is the
    # only one it knows to look for.
    artifact = load_artifact(artifact_path(artifact_root, regime, method, config_key, seed=0))
    if artifact is None or artifact.maps is None:
        return "<p class='note'>No stored heatmaps for this configuration.</p>"

    scores, labels, paths, maps = artifact.scores, artifact.labels, artifact.paths, artifact.maps
    if not labels.any() or labels.all():
        return "<p class='note'>Configuration has only one class; no gallery.</p>"

    # Threshold at the point that maximizes F1, so "hit" and "miss" mean what
    # they would mean in production rather than being read off the ranking.
    from training.evaluate import compute_metrics

    threshold = compute_metrics(scores, labels).f1_threshold
    predicted = scores >= threshold

    hits = [i for i in range(len(scores)) if labels[i] and predicted[i]]
    misses = [i for i in range(len(scores)) if labels[i] != predicted[i]]
    hits.sort(key=lambda i: -scores[i])
    misses.sort(key=lambda i: -abs(scores[i] - threshold))

    cards: list[str] = []
    for group, indices, tone in (
        ("caught", hits[:_GALLERY_HITS], "good"),
        ("missed", misses[:_GALLERY_MISSES], "bad"),
    ):
        for index in indices:
            path = Path(str(paths[index]))
            if not path.exists():
                continue
            original = _encode_image(path, _GALLERY_PX)
            overlay = _encode_heatmap(path, maps[index], _GALLERY_PX)
            if original is None or overlay is None:
                continue
            kind = (
                "false alarm"
                if not labels[index]
                else ("detected" if predicted[index] else "escape")
            )
            cards.append(
                f'<figure class="shot {tone}">'
                f'<div class="pair">'
                f'<img src="{original}" alt="source image" loading="lazy"/>'
                f'<img src="{overlay}" alt="anomaly heatmap" loading="lazy"/>'
                f"</div>"
                f'<figcaption><span class="tag {tone}">{esc(kind)}</span>'
                f"<span class='mono'>{esc(path.name)}</span>"
                f"<span class='mono'>score {scores[index]:.3f}</span></figcaption>"
                f"</figure>"
            )
            _ = group

    if not cards:
        return "<p class='note'>No renderable gallery images found on disk.</p>"
    return f'<div class="gallery">{"".join(cards)}</div>'


def _fmt(value: float, digits: int = 4) -> str:
    """Format a float, rendering ``nan`` as an em dash."""
    return "—" if value != value else f"{value:.{digits}f}"


def _pct(value: float) -> str:
    """Format a rate as a percentage."""
    return "—" if value != value else f"{value * 100:.1f}%"


def ranking_table(summary: list[dict[str, Any]], table_id: str) -> str:
    """Render a sortable ranking table."""
    if not summary:
        return "<p class='note'>No runs recorded for this regime.</p>"

    header = (
        "<tr>"
        "<th data-sort='num'>#</th><th data-sort='str'>Method</th>"
        "<th data-sort='str'>Family</th><th data-sort='num'>Cfgs</th>"
        "<th data-sort='num'>AUROC</th><th data-sort='num'>Worst</th>"
        "<th data-sort='num'>AP</th><th data-sort='num'>F1</th>"
        "<th data-sort='num'>Scrap@95</th><th data-sort='num'>Escape@1</th>"
        "<th data-sort='num'>AUPRO</th><th data-sort='num'>ms/img</th>"
        "<th data-sort='num'>VRAM GB</th>"
        "</tr>"
    )
    body: list[str] = []
    for index, row in enumerate(summary, start=1):
        flag = "" if row["complete"] else " incomplete"
        body.append(
            f'<tr class="{flag.strip()}">'
            f"<td>{index}</td>"
            f"<td class='mono'>{esc(row['method'])}"
            + ("" if row["complete"] else " <span class='warn-chip'>partial</span>")
            + "</td>"
            f"<td>{esc(row['family'])}</td>"
            f"<td>{row['configs']}</td>"
            f"<td class='strong'>{_fmt(row['auroc'])}</td>"
            f"<td>{_fmt(row['min_auroc'])}</td>"
            f"<td>{_fmt(row['ap'])}</td>"
            f"<td>{_fmt(row['f1'])}</td>"
            f"<td>{_pct(row['scrap'])}</td>"
            f"<td>{_pct(row['escape'])}</td>"
            f"<td>{_fmt(row['aupro'])}</td>"
            f"<td>{_fmt(row['ms'], 1)}</td>"
            f"<td>{_fmt(row['vram'], 2)}</td>"
            "</tr>"
        )
    return (
        f'<div class="scroll"><table id="{table_id}" class="rank">'
        f"<thead>{header}</thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def build_html(
    rows: list[dict[str, Any]],
    deployment: list[dict[str, Any]],
    ensembles: list[dict[str, Any]],
    artifact_root: Path,
) -> str:
    """Assemble the whole dashboard."""
    regimes = sorted({row.get("regime", "oneclass") for row in rows})
    multiclass = aggregate(rows, "multiclass")
    oneclass = aggregate(rows, "oneclass")
    fewshot_regimes = [r for r in regimes if r.startswith("fewshot")]

    headline = next((r for r in multiclass if r["complete"]), None) or (
        multiclass[0] if multiclass else None
    )
    baseline = next((r for r in oneclass if r["complete"]), None) or (
        oneclass[0] if oneclass else None
    )

    sections: list[str] = []
    sections.append(_hero(headline, baseline, rows))
    sections.append(_regime_section(rows, multiclass, oneclass, fewshot_regimes))
    sections.append(_frontier_section(multiclass or oneclass, deployment))
    sections.append(_ranking_section(multiclass, oneclass))
    sections.append(_per_category_section(rows))
    sections.append(_diagnostics_section(rows, artifact_root, headline))
    sections.append(_gallery_section(rows, artifact_root, headline))
    if ensembles:
        sections.append(_ensemble_section(ensembles))
    sections.append(_methodology_section(rows))

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return _PAGE.replace("{{CONTENT}}", "".join(sections)).replace("{{GENERATED}}", generated)


def _hero(headline: dict | None, baseline: dict | None, rows: list[dict]) -> str:
    """The headline claim and the numbers behind it."""
    configs = len({r["config"] for r in rows if r.get("status") == "ok"})
    methods = len({r["method"] for r in rows if r.get("status") == "ok"})

    if headline is None:
        return (
            "<section class='hero'><h1>Anomaly-detection benchmark</h1>"
            "<p class='note'>No multi-class results yet. Run the sweep with "
            "<code>--regimes oneclass multiclass</code>.</p></section>"
        )

    # The comparison is restricted to configurations both regimes cover, or
    # the gap would conflate the regime with the different set of categories
    # each one includes.
    delta = ""
    common = shared_configs(rows, ["multiclass", "oneclass"])
    if baseline is not None and common:
        multi = aggregate(rows, "multiclass", common)
        single = aggregate(rows, "oneclass", common)
        if multi and single:
            best_multi, best_single = multi[0], single[0]
            gap = best_multi["auroc"] - best_single["auroc"]
            direction = "above" if gap >= 0 else "below"
            delta = (
                f"<p class='hero-sub'>On the {len(common)} configurations both regimes cover, "
                f"that lands {abs(gap) * 100:.2f} points {direction} the best "
                f"<em>per-category</em> model "
                f"(<span class='mono'>{esc(best_single['method'])}</span>, "
                f"{_fmt(best_single['auroc'])}) &mdash; which needs one checkpoint per "
                f"category rather than one per dataset family.</p>"
            )

    # The two error rates sit beside AUROC deliberately: they are the numbers
    # a plant actually pays for, and burying them in a table reads as though
    # the ranking metric were the result.
    tiles = (
        ("Mean AUROC", _fmt(headline["auroc"]), f"worst config {_fmt(headline['min_auroc'])}"),
        ("Best F1", _fmt(headline["f1"]), "at the F1-optimal threshold"),
        ("Scrap @ 95% recall", _pct(headline["scrap"]), "good parts rejected to catch 95%"),
        ("Escape @ 1% alarms", _pct(headline["escape"]), "defects missed within a 1% budget"),
        ("Latency", f"{_fmt(headline['ms'], 1)} ms", "per image, batch 1"),
        ("Localization", _fmt(headline["aupro"]), "AUPRO, per-region overlap"),
    )
    stats = "".join(
        f'<div class="stat"><span class="k">{esc(key)}</span>'
        f'<span class="v">{value}</span>'
        f'<span class="s">{esc(sub)}</span></div>'
        for key, value, sub in tiles
    )

    return f"""
<section class="hero">
  <p class="eyebrow">Industrial visual anomaly detection &middot; {methods} methods &middot;
     {configs} dataset configurations</p>
  <h1>One model. Every category.</h1>
  <p class="hero-lead">
    <span class="mono big">{esc(headline["method"])}</span> reaches
    <strong>{_fmt(headline["auroc"])} mean AUROC</strong> across all
    {headline["configs"]} configurations from a <em>single</em> fitted model per dataset
    family &mdash; no per-category training, no per-category checkpoint.
  </p>
  {delta}
  <div class="stats">{stats}</div>
</section>
"""


def _regime_section(
    rows: list[dict], multiclass: list[dict], oneclass: list[dict], fewshot: list[str]
) -> str:
    """Compare the deployment regimes head to head."""
    methods = [r["method"] for r in multiclass[:6]] or [r["method"] for r in oneclass[:6]]
    if not methods:
        return ""

    # Every bar is measured on the same configurations, so bar heights differ
    # by regime alone rather than by which categories the regime covered.
    regimes = ["oneclass", "multiclass", *fewshot]
    common = shared_configs(rows, regimes) or None

    def series_for(regime: str) -> list[float]:
        summary = {r["method"]: r for r in aggregate(rows, regime, common)}
        return [summary.get(m, {}).get("auroc", float("nan")) for m in methods]

    series = [("one-class", series_for("oneclass")), ("multi-class", series_for("multiclass"))]
    if fewshot:
        best_shot = sorted(fewshot, key=lambda r: int(r.replace("fewshot", "") or 0))[-1]
        series.append((best_shot.replace("fewshot", "few-shot k="), series_for(best_shot)))

    legend = "".join(
        f'<span class="key"><i style="background:var(--series-{i + 1})"></i>{esc(name)}</span>'
        for i, (name, _) in enumerate(series)
    )
    short = [m.replace("patchcore_", "pc/").replace("dinomaly_", "dino/") for m in methods]
    chart = bar_chart(short, series, y_label="mean AUROC (axis truncated)")
    n_common = len(common) if common else len({r["config"] for r in rows})

    return f"""
<section>
  <h2>The regime decides the architecture</h2>
  <p class="lede">
    One-class fits a model per category &mdash; the setting nearly every paper reports, and the
    easiest to win, because each model only has to represent one product. Multi-class fits
    <em>one</em> model for a whole dataset family. Few-shot fits on a handful of images, which is
    the cold-start case when a new product arrives. A method that only wins on the left is a method
    that ships twenty-nine checkpoints.
  </p>
  <p class="lede">Measured on the {n_common} configurations every regime covers.</p>
  <div class="legend">{legend}</div>
  {chart}
</section>
"""


def _frontier_section(summary: list[dict], deployment: list[dict]) -> str:
    """Accuracy against latency, with the Pareto frontier marked."""
    latency: dict[str, float] = {}
    for row in deployment:
        if row.get("status") == "ok":
            value = row.get("cuda_latency_p95_ms") or row.get("cpu_latency_p95_ms")
            if value:
                latency[row["method"]] = float(value)

    points = [
        Point(latency.get(row["method"], row["ms"]), row["auroc"], row["method"])
        for row in summary
        if row["auroc"] == row["auroc"] and (latency.get(row["method"], row["ms"]) or 0) > 0
    ]
    if not points:
        return ""

    source = "measured p95 (batch 1)" if latency else "mean sweep throughput"
    chart = scatter_pareto(
        pareto_front(points),
        x_label="latency per image (ms, log)",
        y_label="mean AUROC",
    )
    return f"""
<section>
  <h2>Accuracy is only half the decision</h2>
  <p class="lede">
    Every model that is both slower <em>and</em> less accurate than another is dominated &mdash;
    choosing it means accepting a worse model for nothing. The labelled points are the ones that
    are not dominated; the right choice is whichever of them clears the station's cycle time.
    Latency here is {source}, log scale.
  </p>
  {chart}
</section>
"""


def _ranking_section(multiclass: list[dict], oneclass: list[dict]) -> str:
    """Full sortable rankings for both primary regimes."""
    return f"""
<section>
  <h2>Full ranking</h2>
  <p class="lede">
    Click any column to sort. <strong>Scrap@95</strong> is the share of good parts rejected when
    the line is tuned to catch 95% of defects; <strong>Escape@1</strong> is the share of defects
    missed when held to a 1% false-alarm budget. Those two are what a plant pays for; AUROC is
    just how the models rank. Methods that did not complete every configuration are marked
    <span class="warn-chip">partial</span> &mdash; their averages cover an easier subset and are
    not comparable to a complete row.
  </p>
  <h3>Multi-class &mdash; one model per dataset family</h3>
  {ranking_table(multiclass, "rank-multi")}
  <h3>One-class &mdash; one model per category</h3>
  {ranking_table(oneclass, "rank-one")}
</section>
"""


def _per_category_section(rows: list[dict]) -> str:
    """Heatmap of every method against every category."""
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return ""

    regime = "multiclass" if any(r.get("regime") == "multiclass" for r in ok) else "oneclass"
    subset = [r for r in ok if r.get("regime") == regime]
    ranked = [r["method"] for r in aggregate(rows, regime)[:_TOP_METHODS]]
    configs = sorted({r["config"] for r in subset})

    lookup = {(r["method"], r["config"]): _finite(r.get("auroc")) for r in subset}
    values = [[lookup.get((m, c), float("nan")) for c in configs] for m in ranked]
    labels = [c.split("/", 1)[-1] for c in configs]

    return f"""
<section>
  <h2>Where each model is weak</h2>
  <p class="lede">
    AUROC &times;100 per category, {esc(regime)} regime. A high average hides the one category that
    will produce every escape in production &mdash; this is where to look for it. Empty cells are
    runs that failed or were not attempted.
  </p>
  <div class="scroll">{heatmap(ranked, labels, values)}</div>
</section>
"""


def _diagnostics_section(rows: list[dict], artifact_root: Path, headline: dict | None) -> str:
    """ROC curves and score distributions from the stored artifacts."""
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok or headline is None:
        return ""

    regime = headline_regime = (
        "multiclass" if any(r.get("regime") == "multiclass" for r in ok) else "oneclass"
    )

    # Pick the config where the leading method struggles most: the interesting
    # diagnostic is the hard case, not a category everything already solves.
    candidates = [r for r in ok if r.get("regime") == regime and r["method"] == headline["method"]]
    if not candidates:
        return ""
    hardest = min(candidates, key=lambda r: _finite(r.get("auroc")))
    config_key = hardest["config"]

    curves = []
    top_methods = [r["method"] for r in aggregate(rows, regime)[:_ROC_METHODS]]
    for method in top_methods:
        artifact = load_artifact(
            artifact_path(artifact_root, headline_regime, method, config_key, seed=0)
        )
        if artifact is None or not artifact.labels.any() or artifact.labels.all():
            continue
        fpr, tpr = _roc_points(artifact.scores, artifact.labels)
        curves.append((method, fpr, tpr))

    lead = load_artifact(
        artifact_path(artifact_root, headline_regime, headline["method"], config_key, seed=0)
    )
    hist = ""
    if lead is not None and lead.labels.any() and not lead.labels.all():
        from training.evaluate import compute_metrics

        threshold = compute_metrics(lead.scores, lead.labels).f1_threshold
        hist = histogram(
            [float(s) for s, y in zip(lead.scores, lead.labels, strict=True) if not y],
            [float(s) for s, y in zip(lead.scores, lead.labels, strict=True) if y],
            threshold=threshold,
        )

    if not curves and not hist:
        return ""

    legend = "".join(
        f'<span class="key"><i style="background:var(--series-{i + 1})"></i>'
        f'<span class="mono">{esc(name)}</span></span>'
        for i, (name, _, _) in enumerate(curves)
    )

    return f"""
<section>
  <h2>The hardest configuration: <span class="mono">{esc(config_key)}</span></h2>
  <p class="lede">
    The leading model's weakest category, shown two ways. The ROC curve says how well the ranking
    separates; the score histogram says whether a usable threshold actually exists. Two separated
    humps mean yes. Overlap means any threshold trades scrap against escapes, and the shape of the
    overlap says which.
  </p>
  <div class="legend">{legend}</div>
  <div class="two-up">
    <figure>{roc_curves(curves)}<figcaption>ROC &mdash; dashed line is chance</figcaption></figure>
    <figure>{hist}<figcaption>Score distribution &mdash;
      <span class="swatch s1"></span>normal
      <span class="swatch s2"></span>anomalous</figcaption></figure>
  </div>
</section>
"""


def _gallery_section(rows: list[dict], artifact_root: Path, headline: dict | None) -> str:
    """Heatmap gallery, including the failures."""
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok or headline is None:
        return ""
    regime = "multiclass" if any(r.get("regime") == "multiclass" for r in ok) else "oneclass"
    candidates = [r for r in ok if r.get("regime") == regime and r["method"] == headline["method"]]
    if not candidates:
        return ""

    with_maps = [r for r in candidates if _finite(r.get("aupro")) == _finite(r.get("aupro"))]
    target = (with_maps or candidates)[0]["config"]

    return f"""
<section>
  <h2>Localization &mdash; what the model actually saw</h2>
  <p class="lede">
    <span class="mono">{esc(headline["method"])}</span> on
    <span class="mono">{esc(target)}</span>. Source image left, anomaly heatmap right, thresholded
    at the F1-optimal operating point. The failures are included on purpose: a page of successes
    proves nothing, and being able to explain the misses is the difference between a demo and a
    system someone will sign off on.
  </p>
  {build_gallery(artifact_root, regime, headline["method"], target)}
</section>
"""


def _ensemble_section(ensembles: list[dict]) -> str:
    """Whether fusing methods is worth the latency it costs."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in ensembles:
        grouped[row["method"]].append(row)

    summary = sorted(
        (
            {
                "method": method,
                "configs": len(entries),
                "auroc": _mean([_finite(e.get("auroc")) for e in entries]),
                "f1": _mean([_finite(e.get("f1_max")) for e in entries]),
                "scrap": _mean([_finite(e.get("fpr_at_95tpr")) for e in entries]),
            }
            for method, entries in grouped.items()
        ),
        key=lambda row: -(row["auroc"] if row["auroc"] == row["auroc"] else -1),
    )[:12]

    def describe(name: str) -> tuple[str, str]:
        """Split ``ensemble[rule]:a+b+c`` into a rule and a member list.

        The raw key runs to 80-odd characters and blows the table's first
        column past the viewport; the members belong on their own line.
        """
        rule, _, members = name.partition(":")
        rule = rule.replace("ensemble[", "").rstrip("]")
        pretty = "<br/>".join(
            f"<span class='mono'>{esc(m)}</span>" for m in members.split("+") if m
        )
        return rule, pretty

    rows_html: list[str] = []
    for entry in summary:
        rule, members = describe(entry["method"])
        rows_html.append(
            f"<tr><td class='members'>{members}</td><td>{esc(rule)}</td>"
            f"<td>{entry['configs']}</td>"
            f"<td class='strong'>{_fmt(entry['auroc'])}</td><td>{_fmt(entry['f1'])}</td>"
            f"<td>{_pct(entry['scrap'])}</td></tr>"
        )
    body = "".join(rows_html)
    return f"""
<section>
  <h2>Is an ensemble worth it?</h2>
  <p class="lede">
    Members are fused on <em>ranks</em>, not raw scores &mdash; Mahalanobis distances and cosine
    errors differ by orders of magnitude, so averaging them directly would just return whichever
    member has the biggest numbers. An ensemble costs the sum of its members' latency, so it only
    earns its place if the accuracy gain clears the cycle-time budget.
  </p>
  <div class="scroll"><table class="rank ensemble"><thead><tr><th>Members</th><th>Rule</th>
    <th>Cfgs</th><th>AUROC</th><th>F1</th><th>Scrap@95</th></tr></thead>
    <tbody>{body}</tbody></table></div>
</section>
"""


def _methodology_section(rows: list[dict]) -> str:
    """How the numbers were produced, and what they do not cover."""
    failed = [r for r in rows if r.get("status") != "ok"]
    fail_note = (
        f"<li><strong>{len(failed)} runs failed</strong> and are recorded as rows with their "
        f"tracebacks rather than dropped &mdash; a model that OOMs on a dataset has told you "
        f"something about deployability.</li>"
        if failed
        else ""
    )
    return f"""
<section class="method">
  <h2>Method &amp; caveats</h2>
  <ul>
    <li><strong>Fitting never sees a label.</strong> Every method is fitted on normal images only.
      Thresholds are chosen on held-out <em>normal</em> data, so the operating point does not leak
      the evaluation set.</li>
    <li><strong>Identical geometry per config.</strong> Input sizes follow each corpus's real
      aspect ratio &mdash; Severstal strips are ~6:1, Kolektor panels ~2.8:1 &mdash; and every
      method sees the same one, so the comparison measures the method, not the resize.</li>
    <li><strong>Metrics are computed here, not read from any backend.</strong> Both halves of the
      zoo go through the same code on raw scores.</li>
    <li><strong>AUPRO alongside pixel AUROC.</strong> Pixel AUROC is dominated by large defects,
      because a big region simply contributes more pixels. AUPRO weights every defect region
      equally, which is the right weighting when the subtle defects are the expensive ones.</li>
    <li><strong>Severstal's positive rate is ~92%</strong>, unlike the others. Read its average
      precision with that in mind; balanced error rate is the fairer comparison there.</li>
    {fail_note}
  </ul>
</section>
"""


#: The page shell. Colors are the data-viz reference palette, unmodified, with
#: dark values declared under both the media query and the theme attribute so a
#: viewer's explicit choice wins in both directions.
_PAGE = """<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Anomaly Detection Benchmark</title>
<style>
:root{
  color-scheme: light;
  --surface-1:#fcfcfb; --plane:#f9f9f7;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  --series-4:#eda100; --series-5:#e87ba4;
  --good:#0ca30c; --critical:#d03b3b; --warning:#fab219;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --surface-1:#1a1a19; --plane:#0d0d0d;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
    --series-4:#c98500; --series-5:#d55181;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --surface-1:#1a1a19; --plane:#0d0d0d;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  --series-4:#c98500; --series-5:#d55181;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--plane); color:var(--text-primary);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
}
.wrap{max-width:1180px;margin:0 auto;padding:32px 20px 80px}
section{
  background:var(--surface-1); border:1px solid var(--border); border-radius:14px;
  padding:26px 28px; margin:0 0 22px;
}
h1{font-size:2.4rem;line-height:1.12;margin:.2em 0 .3em;letter-spacing:-0.02em}
h2{font-size:1.32rem;margin:0 0 .5em;letter-spacing:-0.01em}
h3{font-size:1rem;margin:1.6em 0 .5em;color:var(--text-secondary);font-weight:600}
p{margin:0 0 1em}
.eyebrow{color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.09em;margin:0}
.hero-lead{font-size:1.1rem;max-width:75ch}
.hero-sub{color:var(--text-secondary);max-width:75ch}
.lede{color:var(--text-secondary);max-width:82ch}
.note{color:var(--muted);font-style:italic}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.92em}
.big{font-size:1.06em;font-weight:600}
.strong{font-weight:650}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(184px,1fr));gap:12px;margin-top:22px}
.stat{
  background:var(--plane); border:1px solid var(--border); border-radius:10px;
  padding:13px 15px; display:flex; flex-direction:column; gap:3px;
}
.stat .k{color:var(--muted);font-size:.74rem;text-transform:uppercase;letter-spacing:.06em}
.stat .v{font-size:1.7rem;font-weight:640;letter-spacing:-0.02em}
.stat .s{color:var(--text-secondary);font-size:.76rem}
.chart{width:100%;height:auto;display:block}
.heatmap{min-width:640px}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
.chance{stroke:var(--axis);stroke-width:1.5;stroke-dasharray:4 4}
.frontier{fill:none;stroke:var(--series-1);stroke-width:1.5;stroke-dasharray:5 4;opacity:.55}
.threshold{stroke:var(--text-primary);stroke-width:1.5;stroke-dasharray:4 3}
.tick{fill:var(--muted);font-size:11px;font-family:inherit}
.axis-label{fill:var(--text-secondary);font-size:11.5px;font-family:inherit}
.cell-text{font-size:10.5px;font-family:inherit;font-variant-numeric:tabular-nums}
.cell-empty{fill:var(--grid);opacity:.45}
.dot-muted{fill:var(--muted);opacity:.42;stroke:var(--surface-1);stroke-width:1.5}
.dot-front{fill:var(--series-1);stroke:var(--surface-1);stroke-width:2}
.point-label{fill:var(--text-secondary);font-size:11px;font-family:inherit}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin:0 0 12px}
.key{display:inline-flex;align-items:center;gap:7px;font-size:.83rem;color:var(--text-secondary)}
.key i{width:11px;height:11px;border-radius:3px;display:inline-block}
.swatch{width:10px;height:10px;border-radius:3px;display:inline-block;margin:0 4px 0 10px}
.swatch.s1{background:var(--series-1)} .swatch.s2{background:var(--series-2)}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table.rank{border-collapse:collapse;width:100%;font-size:.85rem;min-width:900px}
table.rank th,table.rank td{
  padding:7px 10px;text-align:right;border-bottom:1px solid var(--border);
  font-variant-numeric:tabular-nums;white-space:nowrap;
}
table.rank th:nth-child(2),table.rank td:nth-child(2),
table.rank th:nth-child(3),table.rank td:nth-child(3){text-align:left}
table.rank th{
  color:var(--text-secondary);font-weight:600;cursor:pointer;user-select:none;
  position:sticky;top:0;background:var(--surface-1);
}
table.rank th:hover{color:var(--text-primary)}
table.rank tbody tr:hover{background:var(--plane)}
tr.incomplete{opacity:.62}
table.ensemble{min-width:640px}
table.ensemble td.members{text-align:left;white-space:normal;line-height:1.35}
table.ensemble th:first-child,table.ensemble td:first-child{text-align:left}
.warn-chip{
  font-size:.68rem;padding:1px 6px;border-radius:99px;background:var(--warning);
  color:#0b0b0b;font-weight:600;
}
.two-up{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:22px}
.two-up figure{margin:0}
.two-up figcaption,.gallery figcaption{
  color:var(--muted);font-size:.78rem;margin-top:6px;display:flex;
  align-items:center;gap:6px;flex-wrap:wrap;
}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:16px}
.shot{margin:0;border:1px solid var(--border);border-radius:10px;padding:9px}
.shot{background:var(--plane)}
.shot .pair{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.shot img{width:100%;height:auto;border-radius:5px;display:block}
.tag{font-size:.68rem;padding:1px 7px;border-radius:99px;font-weight:600;color:#0b0b0b}
.tag.good{background:var(--good)} .tag.bad{background:var(--critical);color:#fff}
.method ul{margin:0;padding-left:1.15em;color:var(--text-secondary);max-width:88ch}
.method li{margin-bottom:.55em}
footer{color:var(--muted);font-size:.78rem;text-align:center;padding:8px 0 0}
@media (max-width:640px){
  h1{font-size:1.8rem} section{padding:20px 16px} .wrap{padding:20px 12px 60px}
}
</style>
<div class="wrap">
{{CONTENT}}
<footer>Generated {{GENERATED}} &middot; every number reproducible from
<span class="mono">results.jsonl</span></footer>
</div>
<script>
// Click-to-sort on any ranking table. Numeric columns sort numerically and
// treat the em-dash placeholder as missing, so empty cells sink rather than
// sorting as text.
document.querySelectorAll('table.rank').forEach(function (table) {
  table.querySelectorAll('th').forEach(function (th, index) {
    th.addEventListener('click', function () {
      var body = table.tBodies[0];
      var rows = Array.prototype.slice.call(body.rows);
      var numeric = th.dataset.sort !== 'str';
      var asc = th.dataset.dir !== 'asc';
      table.querySelectorAll('th').forEach(function (o) { delete o.dataset.dir; });
      th.dataset.dir = asc ? 'asc' : 'desc';
      rows.sort(function (a, b) {
        var x = a.cells[index] ? a.cells[index].textContent.trim() : '';
        var y = b.cells[index] ? b.cells[index].textContent.trim() : '';
        if (numeric) {
          var nx = parseFloat(x.replace('%', ''));
          var ny = parseFloat(y.replace('%', ''));
          if (isNaN(nx)) nx = Infinity;
          if (isNaN(ny)) ny = Infinity;
          return asc ? nx - ny : ny - nx;
        }
        return asc ? x.localeCompare(y) : y.localeCompare(x);
      });
      rows.forEach(function (r) { body.appendChild(r); });
    });
  });
});
</script>
"""


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and write the dashboard."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR / "results.jsonl")
    parser.add_argument("--artifacts", type=Path, default=RESULTS_DIR / "artifacts")
    parser.add_argument("--deployment", type=Path, default=RESULTS_DIR / "deployment.jsonl")
    parser.add_argument("--ensembles", type=Path, default=RESULTS_DIR / "ensembles.jsonl")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "dashboard.html")
    parser.add_argument("--open", action="store_true", help="Open the file when done.")
    args = parser.parse_args(argv)

    rows = dedupe(read_jsonl(args.results))
    if not rows:
        msg = f"No results at {args.results}. Run `python -m training.sweep` first."
        raise SystemExit(msg)

    html_text = build_html(
        rows, read_jsonl(args.deployment), read_jsonl(args.ensembles), args.artifacts
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    size_mb = args.output.stat().st_size / 1e6
    print(f"wrote {args.output} ({size_mb:.2f} MB)")

    if args.open:
        import webbrowser

        webbrowser.open(args.output.resolve().as_uri())


if __name__ == "__main__":
    main()
