from __future__ import annotations

import pytest

from multimedia_intelligence.demo.charts import ChartSpec, render_chart


def test_render_line_grouped_bar_and_scatter_png() -> None:
    rows = [
        {"year": year, "language": language, "value": value}
        for year, values in [(2025, (15, 22)), (2024, (10, 20))]
        for language, value in zip(("TypeScript", "Python"), values, strict=True)
    ]
    for chart_type in ("line", "grouped-bar"):
        result = render_chart(
            rows,
            ChartSpec(
                expression="@",
                chart_type=chart_type,
                x_field="year",
                y_field="value",
                series_field="language",
                title="Language trends",
            ),
        )
        assert result.png.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(result.png) <= 512 * 1024
        assert result.plotted_points == 4
        assert result.series == ("TypeScript", "Python")
        assert result.categories == ("2024", "2025")

    scatter = render_chart(
        [{"x": 1, "y": 2}, {"x": 2, "y": 4}],
        ChartSpec("@", "scatter", "x", "y", None, "Scatter"),
    )
    assert scatter.plotted_points == 2


def test_render_chart_rejects_bad_inputs_and_bounds() -> None:
    spec = ChartSpec("@", "line", "x", "y", "series", "Invalid")
    with pytest.raises(ValueError, match="array"):
        render_chart({"x": 1, "y": 2}, spec)
    with pytest.raises(ValueError, match="numeric"):
        render_chart([{"x": 1, "y": "two", "series": "a"}], spec)
    with pytest.raises(ValueError, match="12 series"):
        render_chart(
            [{"x": 1, "y": 2, "series": str(index)} for index in range(13)], spec
        )
    with pytest.raises(ValueError, match="5,000"):
        render_chart([{"x": index, "y": 1, "series": "a"} for index in range(5_001)], spec)
