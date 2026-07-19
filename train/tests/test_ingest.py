from __future__ import annotations

import csv
import tempfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from pipeline.ingest import ingest_all, parse_single_csv


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "TEST.csv"
    rows = [
        ["Price", "Close", "High", "Low", "Open", "Volume"],
        ["Ticker", "TEST", "", "", "", ""],
        ["Date", "", "", "", "", ""],
    ]
    base = date(2024, 1, 1)
    for i in range(30):
        d = (base + timedelta(days=i)).isoformat()
        c = 100 + i * 0.5
        h = c + 1
        lo = c - 1
        o = c + 0.1
        v = 10000 + i * 100
        rows.append([d, str(c), str(h), str(lo), str(o), str(v)])

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return csv_file


@pytest.fixture
def sample_csvs(tmp_path: Path) -> Path:
    for ticker in ["AAA", "BBB"]:
        csv_file = tmp_path / f"{ticker}.csv"
        rows = [
            ["Price", "Close", "High", "Low", "Open", "Volume"],
            ["Ticker", ticker, "", "", "", ""],
            ["Date", "", "", "", "", ""],
        ]
        base = date(2024, 1, 1)
        for i in range(20):
            d = (base + timedelta(days=i)).isoformat()
            c = 100 + i
            h = c + 2
            lo = c - 2
            o = c + 0.5
            v = 5000 + i * 200
            rows.append([d, str(c), str(h), str(lo), str(o), str(v)])
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    return tmp_path


def test_parse_single_csv(sample_csv: Path) -> None:
    df = parse_single_csv(sample_csv)
    assert isinstance(df, pl.DataFrame)
    assert df["ticker"][0] == "TEST"
    assert "date" in df.columns
    assert "close" in df.columns
    assert len(df) == 30


def test_ingest_all(sample_csvs: Path, tmp_path: Path) -> None:
    output = tmp_path / "prices.parquet"
    df = ingest_all(sample_csvs, output)
    assert df["ticker"].n_unique() == 2
    assert output.exists()


def test_date_sorting(sample_csv: Path) -> None:
    df = parse_single_csv(sample_csv)
    dates = df["date"].to_list()
    assert dates == sorted(dates)


def test_no_null_close(sample_csv: Path) -> None:
    df = parse_single_csv(sample_csv)
    assert df["close"].null_count() == 0
