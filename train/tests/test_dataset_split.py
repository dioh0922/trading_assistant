from __future__ import annotations

import datetime
import numpy as np
import polars as pl
import pytest

from pipeline.config import SplitConfig
from pipeline.dataset_split import assign_splits


def test_assign_splits():
  # Create mock features and labels
  # We want to check:
  # 1. stock_holdout
  # 2. time_holdout
  # 3. Purged K-Fold

  # 10 tickers
  tickers = [f"TICKER_{i}" for i in range(10)]
  # Dates from 2026-01-01 to 2026-07-01 (182 days)
  start_date = datetime.date(2026, 1, 1)
  dates = [start_date + datetime.timedelta(days=i) for i in range(180)]

  rows = []
  for t in tickers:
    for d in dates:
      rows.append(
        {
          "ticker": t,
          "date": d,
          "feature_1": np.random.rand(),
          "label": "timeout" if np.random.rand() > 0.5 else "upper",
        }
      )

  df = pl.DataFrame(rows)

  config = SplitConfig(
    embargo_days=2,
    purge_days=5,
    time_holdout_months=2,
    stock_holdout_ratio=0.2,  # 2 tickers should be holdout
    n_folds=3,
  )

  # features_df has ticker, date, feature_1
  features_df = df.select(["ticker", "date", "feature_1"])
  # labels_df has ticker, date, label
  labels_df = df.select(["ticker", "date", "label"])

  res = assign_splits(features_df, labels_df, config)

  assert len(res) > 0

  # 1. Stock holdout verification
  stock_holdout_subset = res.filter(pl.col("split_type") == "stock_holdout")
  stock_holdout_tickers = stock_holdout_subset["ticker"].unique().to_list()
  assert len(stock_holdout_tickers) == 2

  # Check that stock_holdout tickers do not appear in other splits
  other_subset = res.filter(pl.col("split_type") != "stock_holdout")
  other_tickers = other_subset["ticker"].unique().to_list()
  for t in stock_holdout_tickers:
    assert t not in other_tickers

  # 2. Time holdout verification
  time_holdout_subset = res.filter(pl.col("split_type") == "time_holdout")
  assert len(time_holdout_subset) > 0
  # Check they are in cv_tickers (which is other_tickers) and date >= time_holdout_start
  # 2 months before 2026-06-29 is approx 2026-04-29
  # The minimum date in time_holdout should be >= 2026-04-28 (roughly)
  assert time_holdout_subset["date"].min() >= datetime.date(2026, 4, 28)

  # 3. Fold purging verification
  # For each fold, check there is no overlap in dates between train and valid with less than purge_days
  for k in range(config.n_folds):
    fold_k = res.filter(pl.col("fold") == k)
    train_k = fold_k.filter(pl.col("split_type") == "train")
    valid_k = fold_k.filter(pl.col("split_type") == "valid")

    assert len(train_k) > 0
    assert len(valid_k) > 0

    valid_dates = valid_k["date"].unique().sort().to_list()
    val_start = min(valid_dates)
    val_end = max(valid_dates)

    # Check train dates are purged
    train_dates = train_k["date"].unique().to_list()
    for td in train_dates:
      # If train date is before valid start, it must be < val_start - purge_days
      if td < val_start:
        assert (val_start - td).days >= config.purge_days
      # If train date is after valid end, it must be > val_end + purge_days + embargo_days
      elif td > val_end:
        assert (td - val_end).days >= (config.purge_days + config.embargo_days)
      else:
        pytest.fail(f"Train date {td} is inside valid range [{val_start}, {val_end}]")
