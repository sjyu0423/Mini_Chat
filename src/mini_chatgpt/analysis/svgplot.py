"""Dependency-free static SVG line charts for experiment reports.

The charts are embedded in Markdown, so they are static: no scripts and a
single light theme. Each point carries a native ``<title>`` tooltip, which
browsers show on hover when the SVG is opened directly.

Style follows the project's chart conventions: one y-axis per chart, 2 px
lines, recessive grid, a legend whenever there are two or more series, text in
ink colors (never series colors), and categorical colors in fixed slot order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from html import escape

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
AXIS = "#b9b8b2"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")


@dataclass
class Series:
    label: str
    x: list[float]
    y: list[float]
    slot: int = 0  # categorical slot, 0-based; fixed per entity across charts
    width: float = 2.0
    opacity: float = 1.0
    dash: str | None = None
    markers: bool = False
    legend: bool = True


@dataclass
class Chart:
    title: str
    x_label: str
    y_label: str
    series: list[Series] = field(default_factory=list)
    subtitle: str = ""
    width: int = 760
    height: int = 380
    y_min: float | None = None
    y_max: float | None = None
    y_format: str = "{:.2f}"
    x_format: str = "{:,.0f}"
    hlines: list[tuple[float, str]] = field(default_factory=list)  # (y, label)
    vlines: list[tuple[float, str]] = field(default_factory=list)  # (x, label), e.g. phase changes
    x_min: float | None = None
    x_max: float | None = None


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round tick values covering [lo, hi]."""
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
    start = math.ceil(lo / step - 1e-9) * step
    ticks = []
    v = start
    while v <= hi + 1e-9 * step:
        ticks.append(round(v, 12))
        v += step
    return ticks


def render(chart: Chart) -> str:
    w, h = chart.width, chart.height
    legend = [s for s in chart.series if s.legend]
    show_legend = len(legend) >= 2
    left, right, bottom = 80, 24, 52
    # Legend rows above the plot area (wrapped to the chart width), so it never covers data.
    legend_y0 = (64 if chart.subtitle else 48) + 6
    legend_pos: list[tuple[float, float]] = []
    lx, row = float(left), 0
    for s in legend:
        item_w = 24 + 6.6 * len(s.label) + 20
        if lx > left and lx + item_w - 20 > w - right:
            lx, row = float(left), row + 1
        legend_pos.append((lx, legend_y0 + 18 * row))
        lx += item_w
    top = legend_y0 + 18 * row + 18 if show_legend else legend_y0 - 6
    pw, ph = w - left - right, h - top - bottom
    xs = [x for s in chart.series for x in s.x]
    ys = [y for s in chart.series for y in s.y if math.isfinite(y)]
    x0 = chart.x_min if chart.x_min is not None else min(xs)
    x1 = chart.x_max if chart.x_max is not None else max(xs)
    y0 = chart.y_min if chart.y_min is not None else min(ys)
    y1 = chart.y_max if chart.y_max is not None else max(ys)
    if chart.y_min is None or chart.y_max is None:
        pad = 0.05 * (y1 - y0 or 1.0)
        y0 = y0 - pad if chart.y_min is None else y0
        y1 = y1 + pad if chart.y_max is None else y1

    def px(x: float) -> float:
        return left + (x - x0) / ((x1 - x0) or 1.0) * pw

    def py(y: float) -> float:
        y = min(max(y, y0), y1)
        return top + (1 - (y - y0) / ((y1 - y0) or 1.0)) * ph

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
        f'height="{h}" font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif" '
        f'role="img" aria-label="{escape(chart.title)}">',
        f"<title>{escape(chart.title)}</title>",
        f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>',
        f'<text x="{left}" y="24" font-size="15" font-weight="600" '
        f'fill="{TEXT_PRIMARY}">{escape(chart.title)}</text>',
    ]
    if chart.subtitle:
        out.append(
            f'<text x="{left}" y="42" font-size="12" fill="{TEXT_SECONDARY}">'
            f"{escape(chart.subtitle)}</text>"
        )
    for t in nice_ticks(y0, y1):
        y = py(t)
        out.append(
            f'<line x1="{left}" x2="{left + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" '
            'stroke-width="1"/>'
        )
        out.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" '
            f'fill="{TEXT_SECONDARY}">{escape(chart.y_format.format(t))}</text>'
        )
    for t in nice_ticks(x0, x1, 6):
        x = px(t)
        out.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top + ph}" y2="{top + ph + 4}" stroke="{AXIS}"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{top + ph + 18}" font-size="11" text-anchor="middle" '
            f'fill="{TEXT_SECONDARY}">{escape(chart.x_format.format(t))}</text>'
        )
    out.append(
        f'<line x1="{left}" x2="{left + pw}" y1="{top + ph}" y2="{top + ph}" stroke="{AXIS}"/>'
    )
    out.append(
        f'<text x="{left + pw / 2}" y="{h - 12}" font-size="12" text-anchor="middle" '
        f'fill="{TEXT_SECONDARY}">{escape(chart.x_label)}</text>'
    )
    out.append(
        f'<text transform="translate(16 {top + ph / 2}) rotate(-90)" font-size="12" '
        f'text-anchor="middle" fill="{TEXT_SECONDARY}">{escape(chart.y_label)}</text>'
    )
    for xv, label in chart.vlines:
        x = px(xv)
        out.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + ph}" stroke="{TEXT_SECONDARY}" '
            'stroke-width="1.5" stroke-dasharray="3 3"/>'
        )
        out.append(
            f'<text x="{x + 6:.1f}" y="{top + 14}" font-size="11" fill="{TEXT_SECONDARY}">'
            f"{escape(label)}</text>"
        )
    for yv, label in chart.hlines:
        y = py(yv)
        out.append(
            f'<line x1="{left}" x2="{left + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{AXIS}" '
            'stroke-dasharray="4 4"/>'
        )
        out.append(
            f'<text x="{left + pw - 4}" y="{y - 5:.1f}" font-size="11" text-anchor="end" '
            f'fill="{TEXT_SECONDARY}">{escape(label)}</text>'
        )

    for s in chart.series:
        color = SERIES[s.slot % len(SERIES)]
        pts = [(px(x), py(y)) for x, y in zip(s.x, s.y, strict=True) if math.isfinite(y)]
        if not pts:
            continue
        d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts)
        dash = f' stroke-dasharray="{s.dash}"' if s.dash else ""
        out.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{s.width}" '
            f'stroke-linejoin="round" stroke-linecap="round" opacity="{s.opacity}"{dash}/>'
        )
        if s.markers:
            for (a, b), x, y in zip(pts, s.x, s.y, strict=False):
                out.append(
                    f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" fill="{color}" '
                    f'stroke="{SURFACE}" stroke-width="2"><title>{escape(s.label)} — step '
                    f"{x:,.0f}: {escape(chart.y_format.format(y))}</title></circle>"
                )

    if show_legend:
        for s, (lx, ly) in zip(legend, legend_pos, strict=True):
            color = SERIES[s.slot % len(SERIES)]
            if s.width == 0:  # marker-only series
                out.append(
                    f'<circle cx="{lx + 9}" cy="{ly}" r="4" fill="{color}" stroke="{SURFACE}" '
                    'stroke-width="2"/>'
                )
            else:
                dash = f' stroke-dasharray="{s.dash}"' if s.dash else ""
                out.append(
                    f'<line x1="{lx}" x2="{lx + 18}" y1="{ly}" y2="{ly}" stroke="{color}" '
                    f'stroke-width="{max(s.width, 2)}" opacity="{max(s.opacity, 0.4)}"{dash}/>'
                )
            out.append(
                f'<text x="{lx + 24}" y="{ly + 4}" font-size="12" fill="{TEXT_PRIMARY}">'
                f"{escape(s.label)}</text>"
            )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def ema(values: list[float], alpha: float = 0.02) -> list[float]:
    """Exponential moving average with bias correction (for noisy train loss)."""
    out, m = [], 0.0
    for i, v in enumerate(values, 1):
        m = (1 - alpha) * m + alpha * v
        out.append(m / (1 - (1 - alpha) ** i))
    return out
