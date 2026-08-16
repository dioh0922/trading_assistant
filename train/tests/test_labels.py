from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from pipeline.labels import triple_barrier_label, build_labels_task1, build_labels_task2


def _make_prices_trending_up(n_days: int = 100, ticker: str = "UP") -> pl.DataFrame:
  base = date(2024, 1, 1)
  dates = [base + timedelta(days=i) for i in range(n_days)]
  close = 100.0 + np.arange(n_days, dtype=float) * 0.5
  return pl.DataFrame(
    {
      "ticker": [ticker] * n_days,
      "date": dates,
      "open": close,
      "high": close + 1,
      "low": close - 0.5,
      "close": close,
      "volume": [10000] * n_days,
    }
  )


def _make_prices_trending_down(n_days: int = 100, ticker: str = "DOWN") -> pl.DataFrame:
  base = date(2024, 1, 1)
  dates = [base + timedelta(days=i) for i in range(n_days)]
  close = 100.0 - np.arange(n_days, dtype=float) * 0.5
  return pl.DataFrame(
    {
      "ticker": [ticker] * n_days,
      "date": dates,
      "open": close,
      "high": close + 0.5,
      "low": close - 1,
      "close": close,
      "volume": [10000] * n_days,
    }
  )


def _make_prices_flat(n_days: int = 100, ticker: str = "FLAT") -> pl.DataFrame:
  base = date(2024, 1, 1)
  dates = [base + timedelta(days=i) for i in range(n_days)]
  close = np.full(n_days, 100.0)
  return pl.DataFrame(
    {
      "ticker": [ticker] * n_days,
      "date": dates,
      "open": close,
      "high": close + 0.1,
      "low": close - 0.1,
      "close": close,
      "volume": [10000] * n_days,
    }
  )


def test_triple_barrier_uptrend() -> None:
  prices = _make_prices_trending_up(100, "UP")
  result = triple_barrier_label(prices, upper=0.10, lower=-0.05, max_days=60)

  assert "label" in result.columns
  assert "days_to_hit" in result.columns
  assert "ticker" in result.columns
  assert "date" in result.columns

  first_label = result[0, "label"]
  assert first_label in ("upper", "lower", "timeout")
  assert result[0, "ticker"] == "UP"


def test_triple_barrier_downtrend() -> None:
  prices = _make_prices_trending_down(100, "DOWN")
  result = triple_barrier_label(prices, upper=0.10, lower=-0.05, max_days=60)

  labels = result["label"].to_list()
  assert "lower" in labels


def test_triple_barrier_flat() -> None:
  prices = _make_prices_flat(100, "FLAT")
  result = triple_barrier_label(prices, upper=0.10, lower=-0.05, max_days=60)

  labels = result["label"].to_list()
  assert all(lbl == "timeout" for lbl in labels)


def test_days_to_hit_within_max() -> None:
  prices = _make_prices_trending_up(100, "UP")
  result = triple_barrier_label(prices, upper=0.10, lower=-0.05, max_days=60)

  for row in result.iter_rows(named=True):
    if row["label"] == "timeout":
      assert row["days_to_hit"] is None or np.isnan(row["days_to_hit"])
    else:
      assert row["days_to_hit"] is not None
      assert row["days_to_hit"] <= 60


def test_upper_none_no_upper_label() -> None:
  prices = _make_prices_trending_up(100, "UP")
  result = triple_barrier_label(prices, upper=None, lower=-0.05, max_days=60)

  labels = result["label"].to_list()
  assert "upper" not in labels


def test_lower_none_no_lower_label() -> None:
  prices = _make_prices_trending_down(100, "DOWN")
  result = triple_barrier_label(prices, upper=0.10, lower=None, max_days=60)

  labels = result["label"].to_list()
  assert "lower" not in labels


def test_build_labels_task1(tmp_path: Path) -> None:
  base = date(2024, 1, 1)
  dates = [base + timedelta(days=i) for i in range(60)]
  close = [100.0 + (i % 10 - 5) * 0.3 for i in range(60)]
  prices = pl.DataFrame(
    {
      "ticker": ["UP"] * 60,
      "date": dates,
      "open": close,
      "high": [c + 1 for c in close],
      "low": [c - 1 for c in close],
      "close": close,
      "volume": [10000] * 60,
    }
  )
  prices.write_parquet(tmp_path / "prices.parquet")
  output = tmp_path / "labels_task1.parquet"
  result = build_labels_task1(
    tmp_path / "prices.parquet", output, horizon_days=5, up_threshold=0.01
  )
  assert output.exists()
  assert result["label"].n_unique() >= 2


def test_build_labels_task2(tmp_path: Path) -> None:
  prices = _make_prices_trending_up(80, "UP")
  prices.write_parquet(tmp_path / "prices.parquet")
  output = tmp_path / "labels_task2.parquet"
  result = build_labels_task2(
    tmp_path / "prices.parquet", output, upper=0.10, lower=-0.05, max_days=60
  )
  assert output.exists()
  assert "sample_weight" in result.columns


def test_manual_hand_calculation() -> None:
  base = date(2024, 1, 1)
  dates = [base + timedelta(days=i) for i in range(10)]
  close = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
  prices = pl.DataFrame(
    {
      "ticker": ["M"] * 10,
      "date": dates,
      "open": close,
      "high": [c + 1 for c in close],
      "low": [c - 1 for c in close],
      "close": close,
      "volume": [1000] * 10,
    }
  )

  result = triple_barrier_label(prices, upper=0.05, lower=-0.05, max_days=5)

  assert result[0, "label"] == "upper"
  assert result[0, "days_to_hit"] == 5.0

  result2 = triple_barrier_label(prices, upper=0.05, lower=-0.05, max_days=3)
  assert result2[0, "label"] == "timeout"
