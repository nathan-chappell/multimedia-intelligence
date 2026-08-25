from pathlib import Path

import pytest

from tests.support.csv_analysis import ColumnSpec, CsvAnalyzer


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "measurements.csv"
    path.write_text(
        "city,temperature,visits,active\nZagreb,10.0,2,true\nSplit,20.0,4,false\nRijeka,,6,true\n",
        encoding="utf-8",
    )
    return path


def test_head_infers_types_and_coerces_values(csv_file: Path) -> None:
    result = CsvAnalyzer(csv_file).head()
    assert [column.inferred_type for column in result.columns] == [
        "string",
        "number",
        "integer",
        "boolean",
    ]
    assert result.rows[0]["visits"] == 2
    assert result.rows[2]["temperature"] is None


def test_head_identifies_iso_datetime_values(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    path.write_text(
        "occurred_at,event\n2026-08-20T10:00:00Z,opened\n2026-08-21T11:30:00Z,closed\n",
        encoding="utf-8",
    )

    result = CsvAnalyzer(path).head()

    assert result.columns[0].inferred_type == "datetime"
    assert result.rows[0]["occurred_at"] == "2026-08-20T10:00:00Z"


def test_rows_are_bounded_and_project_columns(csv_file: Path) -> None:
    analyzer = CsvAnalyzer(csv_file, max_rows_per_call=2)
    result = analyzer.rows(1, 2, ("city", "visits"))
    assert result.rows == (
        {"city": "Split", "visits": 4},
        {"city": "Rijeka", "visits": 6},
    )
    with pytest.raises(ValueError, match="between 1 and 2"):
        analyzer.rows(0, 3)


def test_stats_stream_numeric_columns(csv_file: Path) -> None:
    stats = CsvAnalyzer(csv_file).stats(("temperature", "visits"))
    assert stats[0].count == 2
    assert stats[0].null_count == 1
    assert stats[0].mean == 15
    assert stats[1].quantiles["p50"] == 4


def test_plot_produces_png(csv_file: Path) -> None:
    plot = CsvAnalyzer(csv_file).plot(ColumnSpec("visits"), ColumnSpec("temperature"))
    assert plot.media_type == "image/png"
    assert plot.content.startswith(b"\x89PNG")
