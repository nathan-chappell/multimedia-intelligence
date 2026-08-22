from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from statistics import fmean

from PIL import Image, ImageDraw

type Cell = str | int | float | bool | None
type Row = dict[str, Cell]

NULL_MARKERS = frozenset({"", "null", "none", "na", "n/a"})
TRUE_MARKERS = frozenset({"true", "yes"})
FALSE_MARKERS = frozenset({"false", "no"})


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    column: str
    label: str | None = None


@dataclass(frozen=True, slots=True)
class ColumnSummary:
    name: str
    inferred_type: str
    nullable: bool


@dataclass(frozen=True, slots=True)
class TableSlice:
    columns: tuple[ColumnSummary, ...]
    rows: tuple[Row, ...]
    start: int
    count: int


@dataclass(frozen=True, slots=True)
class NumericStats:
    column: str
    count: int
    null_count: int
    invalid_count: int
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float | None
    quantiles: dict[str, float]
    approximate_quantiles: bool


@dataclass(frozen=True, slots=True)
class PlotArtifact:
    content: bytes
    media_type: str
    suggested_filename: str
    description: str


@dataclass(slots=True)
class _RunningStats:
    sample_limit: int
    rng: random.Random
    count: int = 0
    null_count: int = 0
    invalid_count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    sample: list[float] = field(default_factory=list)

    def add(self, raw_value: str) -> None:
        value = raw_value.strip()
        if value.casefold() in NULL_MARKERS:
            self.null_count += 1
            return
        try:
            number = float(value)
        except ValueError:
            self.invalid_count += 1
            return
        if not math.isfinite(number):
            self.invalid_count += 1
            return

        self.count += 1
        delta = number - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (number - self.mean)
        self.minimum = min(self.minimum, number)
        self.maximum = max(self.maximum, number)

        if len(self.sample) < self.sample_limit:
            self.sample.append(number)
        else:
            replacement = self.rng.randrange(self.count)
            if replacement < self.sample_limit:
                self.sample[replacement] = number

    def finish(self, column: str) -> NumericStats:
        if self.count == 0:
            raise ValueError(f"Column {column!r} has no finite numeric values")
        ordered = sorted(self.sample)
        return NumericStats(
            column=column,
            count=self.count,
            null_count=self.null_count,
            invalid_count=self.invalid_count,
            minimum=self.minimum,
            maximum=self.maximum,
            mean=self.mean,
            standard_deviation=(math.sqrt(self.m2 / (self.count - 1)) if self.count > 1 else None),
            quantiles={
                "p25": _quantile(ordered, 0.25),
                "p50": _quantile(ordered, 0.50),
                "p75": _quantile(ordered, 0.75),
            },
            approximate_quantiles=self.count > self.sample_limit,
        )


class CsvAnalyzer:
    """Read-only CSV tools with explicit response and memory bounds."""

    def __init__(
        self,
        path: Path,
        *,
        encoding: str = "utf-8-sig",
        inference_rows: int = 200,
        max_rows_per_call: int = 200,
        quantile_sample_size: int = 10_000,
    ) -> None:
        self.path = path
        self.encoding = encoding
        self.inference_rows = inference_rows
        self.max_rows_per_call = max_rows_per_call
        self.quantile_sample_size = quantile_sample_size

    def head(self, count: int = 10) -> TableSlice:
        return self.rows(start=0, count=min(count, 10))

    def rows(
        self,
        start: int,
        count: int,
        columns: tuple[str, ...] | None = None,
    ) -> TableSlice:
        if start < 0:
            raise ValueError("start must be non-negative")
        if not 1 <= count <= self.max_rows_per_call:
            raise ValueError(f"count must be between 1 and {self.max_rows_per_call}")

        headers, inferred = self._schema()
        selected = columns or headers
        _validate_columns(headers, selected)
        result: list[Row] = []
        with self.path.open("r", encoding=self.encoding, newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, raw_row in enumerate(reader):
                if row_index < start:
                    continue
                if len(result) >= count:
                    break
                result.append(
                    {
                        column: _coerce(raw_row.get(column, ""), inferred[column])
                        for column in selected
                    }
                )

        summaries = tuple(
            ColumnSummary(
                name=column,
                inferred_type=inferred[column][0],
                nullable=inferred[column][1],
            )
            for column in selected
        )
        return TableSlice(
            columns=summaries,
            rows=tuple(result),
            start=start,
            count=len(result),
        )

    def stats(self, columns: tuple[str, ...] | None = None) -> tuple[NumericStats, ...]:
        headers, inferred = self._schema()
        selected = columns or tuple(
            column for column in headers if inferred[column][0] in {"integer", "number"}
        )
        _validate_columns(headers, selected)
        if not selected:
            raise ValueError("No numeric columns were inferred")

        accumulators = {
            column: _RunningStats(
                sample_limit=self.quantile_sample_size,
                rng=random.Random(f"csv-stats:{column}"),
            )
            for column in selected
        }
        with self.path.open("r", encoding=self.encoding, newline="") as handle:
            for row in csv.DictReader(handle):
                for column, accumulator in accumulators.items():
                    accumulator.add(row.get(column, ""))
        return tuple(accumulators[column].finish(column) for column in selected)

    def plot(
        self,
        x: ColumnSpec,
        y: ColumnSpec,
        *,
        max_points: int = 2_000,
    ) -> PlotArtifact:
        if not 10 <= max_points <= 10_000:
            raise ValueError("max_points must be between 10 and 10000")
        headers, inferred = self._schema()
        _validate_columns(headers, (x.column, y.column))
        if inferred[y.column][0] not in {"integer", "number"}:
            raise ValueError("The y column must be numeric")

        x_kind = inferred[x.column][0]
        points = self._sample_plot_points(
            x.column,
            y.column,
            max_points,
            numeric_x=x_kind in {"integer", "number"},
        )
        if not points:
            raise ValueError("No plottable rows were found")

        image = Image.new("RGB", (1200, 675), "white")
        drawing = ImageDraw.Draw(image)
        plot_box = (90, 55, 1160, 590)
        drawing.line(
            ((plot_box[0], plot_box[1]), (plot_box[0], plot_box[3])),
            fill="#334155",
            width=2,
        )
        drawing.line(
            ((plot_box[0], plot_box[3]), (plot_box[2], plot_box[3])),
            fill="#334155",
            width=2,
        )
        if x_kind in {"integer", "number"}:
            _draw_scatter(drawing, plot_box, points)
            plot_kind = "scatter"
        else:
            grouped: dict[str, list[float]] = defaultdict(list)
            for category, value in points:
                grouped[str(category)].append(value)
            top_groups = sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)[:30]
            _draw_bars(drawing, plot_box, top_groups)
            plot_kind = "category mean bar"

        title = f"{y.label or y.column} by {x.label or x.column}"
        drawing.text((90, 20), title, fill="#0f172a")
        drawing.text((550, 630), x.label or x.column, fill="#334155")
        drawing.text((10, 55), y.label or y.column, fill="#334155")
        output = BytesIO()
        image.save(output, format="PNG", optimize=True)
        return PlotArtifact(
            content=output.getvalue(),
            media_type="image/png",
            suggested_filename=f"plot-{x.column}-{y.column}.png",
            description=(
                f"Automatic {plot_kind} plot of {y.column!r} against {x.column!r} "
                f"using {len(points)} sampled rows."
            ),
        )

    def _schema(self) -> tuple[tuple[str, ...], dict[str, tuple[str, bool]]]:
        samples: dict[str, list[str]] = {}
        with self.path.open("r", encoding=self.encoding, newline="") as handle:
            reader = csv.DictReader(handle)
            headers = tuple(reader.fieldnames or ())
            _validate_headers(headers)
            samples = {header: [] for header in headers}
            for row_index, row in enumerate(reader):
                if row_index >= self.inference_rows:
                    break
                for header in headers:
                    samples[header].append(row.get(header, ""))
        return headers, {header: _infer_type(samples[header]) for header in headers}

    def _sample_plot_points(
        self,
        x_column: str,
        y_column: str,
        limit: int,
        *,
        numeric_x: bool,
    ) -> list[tuple[str | float, float]]:
        sample: list[tuple[str | float, float]] = []
        rng = random.Random(f"csv-plot:{x_column}:{y_column}")
        seen = 0
        with self.path.open("r", encoding=self.encoding, newline="") as handle:
            for row in csv.DictReader(handle):
                x_raw = row.get(x_column, "").strip()
                y_raw = row.get(y_column, "").strip()
                if x_raw.casefold() in NULL_MARKERS or y_raw.casefold() in NULL_MARKERS:
                    continue
                try:
                    y_value = float(y_raw)
                except ValueError:
                    continue
                if not math.isfinite(y_value):
                    continue
                if numeric_x:
                    try:
                        numeric_value = float(x_raw)
                    except ValueError:
                        continue
                    if not math.isfinite(numeric_value):
                        continue
                    x_value: str | float = numeric_value
                else:
                    x_value = x_raw

                seen += 1
                point = (x_value, y_value)
                if len(sample) < limit:
                    sample.append(point)
                else:
                    replacement = rng.randrange(seen)
                    if replacement < limit:
                        sample[replacement] = point
        return sample


def _validate_headers(headers: tuple[str, ...]) -> None:
    if not headers:
        raise ValueError("CSV has no header row")
    if any(not header.strip() for header in headers):
        raise ValueError("CSV contains an empty header")
    if len(set(headers)) != len(headers):
        raise ValueError("CSV headers must be unique")


def _validate_columns(headers: tuple[str, ...], columns: tuple[str, ...]) -> None:
    unknown = set(columns).difference(headers)
    if unknown:
        raise ValueError(f"Unknown columns: {', '.join(sorted(unknown))}")


def _infer_type(values: list[str]) -> tuple[str, bool]:
    present = [value.strip() for value in values if value.strip().casefold() not in NULL_MARKERS]
    nullable = len(present) != len(values)
    if not present:
        return "unknown", True
    if all(value.casefold() in TRUE_MARKERS | FALSE_MARKERS for value in present):
        return "boolean", nullable
    if all(_is_integer(value) for value in present):
        return "integer", nullable
    if all(_is_number(value) for value in present):
        return "number", nullable
    if all(_is_datetime(value) for value in present):
        return "datetime", nullable
    return "string", nullable


def _coerce(value: str | None, inferred: tuple[str, bool]) -> Cell:
    normalized = (value or "").strip()
    if normalized.casefold() in NULL_MARKERS:
        return None
    kind = inferred[0]
    if kind == "boolean":
        if normalized.casefold() in TRUE_MARKERS:
            return True
        if normalized.casefold() in FALSE_MARKERS:
            return False
        return normalized
    if kind == "integer":
        try:
            return int(normalized)
        except ValueError:
            return normalized
    if kind == "number":
        try:
            number = float(normalized)
        except ValueError:
            return normalized
        return number if math.isfinite(number) else normalized
    return normalized


def _is_integer(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _is_number(value: str) -> bool:
    try:
        number = float(value)
    except ValueError:
        return False
    return math.isfinite(number)


def _is_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _quantile(ordered: list[float], probability: float) -> float:
    if not ordered:
        raise ValueError("Cannot calculate a quantile from an empty sample")
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


def _draw_scatter(
    drawing: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    points: list[tuple[str | float, float]],
) -> None:
    numeric_points = [(float(x), y) for x, y in points]
    x_values = [point[0] for point in numeric_points]
    y_values = [point[1] for point in numeric_points]
    for x_value, y_value in numeric_points:
        px = _scale(x_value, min(x_values), max(x_values), box[0], box[2])
        py = _scale(y_value, min(y_values), max(y_values), box[3], box[1])
        drawing.ellipse((px - 4, py - 4, px + 4, py + 4), fill="#2563eb")


def _draw_bars(
    drawing: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    groups: list[tuple[str, list[float]]],
) -> None:
    means = [fmean(values) for _group, values in groups]
    low = min(0.0, min(means))
    high = max(0.0, max(means))
    slot_width = (box[2] - box[0]) / max(1, len(groups))
    zero_y = _scale(0.0, low, high, box[3], box[1])
    for index, ((label, _values), mean) in enumerate(zip(groups, means, strict=True)):
        left = box[0] + index * slot_width + 2
        right = box[0] + (index + 1) * slot_width - 2
        value_y = _scale(mean, low, high, box[3], box[1])
        drawing.rectangle(
            (left, min(zero_y, value_y), right, max(zero_y, value_y)),
            fill="#0f766e",
        )
        if len(groups) <= 12:
            drawing.text((left, box[3] + 8), label[:12], fill="#475569")


def _scale(
    value: float,
    low: float,
    high: float,
    output_low: int,
    output_high: int,
) -> int:
    if math.isclose(low, high):
        return round((output_low + output_high) / 2)
    ratio = (value - low) / (high - low)
    return round(output_low + ratio * (output_high - output_low))
