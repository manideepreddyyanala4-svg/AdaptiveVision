"""Inline-SVG chart primitives for the dashboard.

The dashboard has to be one file that opens from disk with no server, no CDN
and no build step -- something to drop in a repository and open in front of
someone. That rules out a charting library, so the marks are emitted as SVG
directly.

Colors come from the data-viz reference palette unmodified, and each form
stays inside that palette's documented caps: scatter and other all-pairs forms
use only the first three categorical slots, bars and lines may use the adjacent
order, and magnitude uses the single-hue blue ramp. Every color is emitted as a
CSS custom property so light and dark modes swap in one place.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass

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
