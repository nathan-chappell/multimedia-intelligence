"""Demo-only chart rendering helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

import jmespath  # type: ignore[import-untyped]
from PIL import Image, ImageDraw, ImageFont

ChartType = Literal["line", "grouped-bar", "scatter"]

MAX_INPUT_ROWS = 5_000
MAX_SERIES = 12
MAX_CATEGORIES = 50
MAX_PNG_BYTES = 512 * 1024
WIDTH = 1_200
HEIGHT = 675


@dataclass(frozen=True, slots=True)
class ChartSpec:
    expression: str
    chart_type: ChartType
    x_field: str
    y_field: str
    series_field: str | None
    title: str
    x_label: str | None = None
    y_label: str | None = None


@dataclass(frozen=True, slots=True)
class ChartResult:
    png: bytes
    row_count: int
    plotted_points: int
    series: tuple[str, ...]
    categories: tuple[str, ...]


def render_chart(value: object, spec: ChartSpec) -> ChartResult:
    selected = jmespath.search(spec.expression, value)
    if not isinstance(selected, list):
        raise ValueError("Chart expression must return an array of row objects")
    if len(selected) > MAX_INPUT_ROWS:
        raise ValueError(f"Chart expressions are limited to {MAX_INPUT_ROWS:,} rows")
    if not selected:
        raise ValueError("Chart expression returned no rows")
    if not all(isinstance(row, dict) for row in selected):
        raise ValueError("Chart expression must return only row objects")

    points: list[tuple[object, float, str]] = []
    for row in selected:
        assert isinstance(row, dict)
        x = row.get(spec.x_field)
        y = row.get(spec.y_field)
        if x is None or isinstance(x, bool):
            continue
        if isinstance(y, bool) or not isinstance(y, (int, float)):
            raise ValueError(f"Field {spec.y_field!r} must contain numeric values")
        series_value = row.get(spec.series_field) if spec.series_field else "Value"
        if series_value is None:
            series_value = "(missing)"
        points.append((x, float(y), str(series_value)))
    if not points:
        raise ValueError("No plottable rows remain after validating chart fields")

    series = tuple(dict.fromkeys(point[2] for point in points))
    if len(series) > MAX_SERIES:
        raise ValueError(f"Charts are limited to {MAX_SERIES} series")
    category_values = {str(point[0]): point[0] for point in points}
    categories = tuple(
        sorted(category_values, key=lambda label: _category_sort_key(category_values[label]))
    )
    if spec.chart_type == "grouped-bar" and len(categories) > MAX_CATEGORIES:
        raise ValueError(f"Grouped bar charts are limited to {MAX_CATEGORIES} categories")

    png = _draw_chart(points, series, categories, spec)
    if len(png) > MAX_PNG_BYTES:
        raise ValueError("Rendered chart exceeds the 512 KiB output limit")
    return ChartResult(
        png=png,
        row_count=len(selected),
        plotted_points=len(points),
        series=series,
        categories=categories,
    )


def _draw_chart(
    points: list[tuple[object, float, str]],
    series: tuple[str, ...],
    categories: tuple[str, ...],
    spec: ChartSpec,
) -> bytes:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#fbfcfe")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    plot = (100, 90, 940, 570)
    left, top, right, bottom = plot
    palette = (
        "#2563eb",
        "#dc2626",
        "#059669",
        "#7c3aed",
        "#d97706",
        "#0891b2",
        "#db2777",
        "#4f46e5",
        "#65a30d",
        "#9333ea",
        "#ea580c",
        "#0f766e",
    )
    color_by_series = dict(zip(series, palette, strict=False))

    draw.text((left, 28), spec.title, fill="#111827", font=font)
    draw.line((left, bottom, right, bottom), fill="#64748b", width=2)
    draw.line((left, top, left, bottom), fill="#64748b", width=2)

    ys = [point[1] for point in points]
    y_min, y_max = min(ys), max(ys)
    if y_min == y_max:
        padding = abs(y_min) * 0.1 or 1.0
        y_min -= padding
        y_max += padding
    elif spec.chart_type != "scatter" and y_min > 0:
        y_min = 0.0
    elif spec.chart_type != "scatter" and y_max < 0:
        y_max = 0.0

    def y_pixel(value: float) -> float:
        return bottom - ((value - y_min) / (y_max - y_min)) * (bottom - top)

    for tick in range(6):
        ratio = tick / 5
        y = bottom - ratio * (bottom - top)
        value = y_min + ratio * (y_max - y_min)
        draw.line((left, y, right, y), fill="#e2e8f0", width=1)
        draw.text((25, y - 6), f"{value:,.2f}", fill="#475569", font=font)

    if spec.chart_type == "scatter":
        numeric_x = [_numeric(point[0], spec.x_field) for point in points]
        x_min, x_max = min(numeric_x), max(numeric_x)
        if x_min == x_max:
            x_min -= 1
            x_max += 1

        def x_pixel(value: float) -> float:
            return left + ((value - x_min) / (x_max - x_min)) * (right - left)

        for (point, x_value) in zip(points, numeric_x, strict=True):
            x, y, name = point
            del x
            px, py = x_pixel(x_value), y_pixel(y)
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color_by_series[name])
    else:
        category_index = {name: index for index, name in enumerate(categories)}
        step = (right - left) / max(1, len(categories))
        if spec.chart_type == "line":
            grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
            for x, y, name in points:
                grouped[name].append((category_index[str(x)], y))
            for name, series_points in grouped.items():
                coordinates = [
                    (left + (index + 0.5) * step, y_pixel(value))
                    for index, value in sorted(series_points)
                ]
                if len(coordinates) > 1:
                    draw.line(coordinates, fill=color_by_series[name], width=4)
                for px, py in coordinates:
                    draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color_by_series[name])
        else:
            grouped_values: dict[tuple[str, str], list[float]] = defaultdict(list)
            for x, y, name in points:
                grouped_values[(str(x), name)].append(y)
            width = min(42.0, step * 0.8 / len(series))
            for category, category_position in category_index.items():
                center = left + (category_position + 0.5) * step
                for series_position, name in enumerate(series):
                    values = grouped_values.get((category, name))
                    if not values:
                        continue
                    value = sum(values) / len(values)
                    x0 = center + (series_position - len(series) / 2) * width
                    x1 = x0 + width * 0.85
                    draw.rectangle((x0, y_pixel(value), x1, y_pixel(0)), fill=color_by_series[name])

        label_stride = max(1, (len(categories) + 11) // 12)
        for index, category in enumerate(categories):
            if index % label_stride == 0:
                label = category if len(category) <= 14 else f"{category[:12]}…"
                draw.text(
                    (left + (index + 0.5) * step - 18, bottom + 10),
                    label,
                    fill="#475569",
                    font=font,
                )

    if spec.x_label:
        draw.text(((left + right) / 2 - 20, 625), spec.x_label, fill="#334155", font=font)
    if spec.y_label:
        draw.text((20, 62), spec.y_label, fill="#334155", font=font)
    legend_x = 975
    for index, name in enumerate(series):
        y = 105 + index * 30
        draw.rectangle((legend_x, y, legend_x + 16, y + 16), fill=color_by_series[name])
        label = name if len(name) <= 25 else f"{name[:23]}…"
        draw.text((legend_x + 24, y + 2), label, fill="#334155", font=font)

    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _numeric(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Scatter chart field {field!r} must contain numeric values")
    return float(value)


def _category_sort_key(value: object) -> tuple[int, float | str]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value))
    return (1, str(value).casefold())
