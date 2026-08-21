from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from pipeline.features_single import compute_features_single
from pipeline.features_cross import compute_cross_sectional


def _make_prices(n_days: int = 100, ticker: str = "TEST") -> pl.DataFrame:
  np.random.seed(42)
  base = date(2024, 1, 1)
  dates = [base + timedelta(days=i) for i in range(n_days)]
  close = 100.0 + np.cumsum(np.random.randn(n_days) * 0.5)
  return pl.DataFrame(
    {
      "ticker": [ticker] * n_days,
      "date": dates,
      "open": close + 0.1,
      "high": close + np.abs(np.random.randn(n_days)),
      "low": close - np.abs(np.random.randn(n_days)),
      "close": close,
      "volume": [10000 + int(x * 100) for x in np.abs(np.random.randn(n_days))],
    }
  )


def test_feature_columns_exist() -> None:
  prices = _make_prices(100)
  feat = compute_features_single(prices, windows=[5, 20, 60])
  expected_cols = [
    "return_5d",
    "return_20d",
    "return_60d",
    "ma_dev_5",
    "ma_dev_20",
    "ma_dev_60",
    "volatility_5d",
    "volatility_20d",
    "volatility_60d",
    "atr_14",
    "rsi_14",
    "volume_ratio_20d",
    "range_position_60d",
  ]
  for c in expected_cols:
    assert c in feat.columns, f"Missing column: {c}"


def test_no_future_leakage() -> None:
  prices = _make_prices(100)
  feat = compute_features_single(prices, windows=[5, 20, 60])

  close = prices["close"].to_list()

  for w in [5, 20, 60]:
    for t_idx in range(w + 1, 100):
      return_val = feat[f"return_{w}d"][t_idx]
      if return_val is None or (isinstance(return_val, float) and np.isnan(return_val)):
        continue
      expected = (close[t_idx] - close[t_idx - w]) / close[t_idx - w]
      assert (
        abs(return_val - expected) < 1e-10
      ), f"return_{w}d at row {t_idx}: got {return_val}, expected {expected}"


def test_warmup_nulls() -> None:
  prices = _make_prices(100)
  feat = compute_features_single(prices, windows=[5, 20, 60])

  assert feat["return_5d"].null_count() > 0
  assert feat["return_60d"].null_count() > 0
  assert feat["return_5d"].null_count() < feat["return_60d"].null_count()


def test_rsi_range() -> None:
  prices = _make_prices(100)
  feat = compute_features_single(prices, windows=[5, 20, 60])
  rsi_valid = feat["rsi_14"].drop_nulls()
  assert (rsi_valid >= 0).all()
  assert (rsi_valid <= 100).all()


def _make_multi_ticker_prices(
  n_days: int = 30, tickers: list[str] | None = None
) -> pl.DataFrame:
  if tickers is None:
    tickers = ["A", "B", "C"]
  np.random.seed(42)
  base = date(2024, 1, 1)
  frames = []
  for t in tickers:
    dates = [base + timedelta(days=i) for i in range(n_days)]
    close = 100.0 + np.cumsum(np.random.randn(n_days) * 0.5)
    vol = [10000 + int(x * 100) for x in np.abs(np.random.randn(n_days))]
    frames.append(
      pl.DataFrame(
        {
          "ticker": [t] * n_days,
          "date": dates,
          "open": close + 0.1,
          "high": close + np.abs(np.random.randn(n_days)),
          "low": close - np.abs(np.random.randn(n_days)),
          "close": close,
          "volume": vol,
        }
      )
    )
  return pl.concat(frames)


def test_cross_sectional_columns_exist() -> None:
  prices = _make_multi_ticker_prices(100, ["A", "B", "C"])
  feat_single = compute_features_single(prices, windows=[5, 20, 60])
  feat_cross = compute_cross_sectional(feat_single, prices)
  for c in ["return_rank_pct", "volume_rank_pct", "sector_relative_return"]:
    assert c in feat_cross.columns, f"Missing column: {c}"


def test_cross_sectional_row_count_preserved() -> None:
  prices = _make_multi_ticker_prices(60, ["A", "B"])
  feat_single = compute_features_single(prices, windows=[5, 20, 60])
  feat_cross = compute_cross_sectional(feat_single, prices)
  assert len(feat_cross) == len(feat_single)


def test_return_rank_pct_distribution() -> None:
  prices = _make_multi_ticker_prices(60, ["A", "B", "C", "D", "E"])
  feat_single = compute_features_single(prices, windows=[5, 20, 60])
  feat_cross = compute_cross_sectional(feat_single, prices)

  day = date(2024, 1, 15)
  subset = feat_cross.filter(pl.col("date") == day)
  ranks = subset["return_rank_pct"].drop_nulls()

  assert len(ranks) == 5
  assert ranks.min() > 0
  assert ranks.max() <= 1.0


def test_volume_rank_pct_distribution() -> None:
  prices = _make_multi_ticker_prices(60, ["A", "B", "C"])
  feat_single = compute_features_single(prices, windows=[5, 20, 60])
  feat_cross = compute_cross_sectional(feat_single, prices)

  day = date(2024, 1, 10)
  subset = feat_cross.filter(pl.col("date") == day)
  ranks = subset["volume_rank_pct"].drop_nulls()

  assert len(ranks) == 3
  assert ranks.min() > 0
  assert ranks.max() <= 1.0


def test_sector_relative_return_centered() -> None:
  prices = _make_multi_ticker_prices(60, ["A", "B", "C"])
  feat_single = compute_features_single(prices, windows=[5, 20, 60])
  feat_cross = compute_cross_sectional(feat_single, prices)

  day = date(2024, 1, 10)
  subset = feat_cross.filter(pl.col("date") == day)
  rel = subset["sector_relative_return"].drop_nulls()

  assert len(rel) == 3
  assert abs(rel.sum()) < 1e-10
