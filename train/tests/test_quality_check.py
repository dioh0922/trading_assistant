from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from pipeline.quality_check import flag_quality_issues


def _make_prices(n_days: int = 30, ticker: str = "TEST") -> pl.DataFrame:
    base = date(2024, 1, 1)
    dates = [base + timedelta(days=i) for i in range(n_days)]
    close = 100.0 + np.arange(n_days, dtype=float)
    return pl.DataFrame({
        "ticker": [ticker] * n_days,
        "date": dates,
        "open": close + 0.1,
        "high": close + 2.0,
        "low": close - 2.0,
        "close": close,
        "volume": [10000] * n_days,
    })


def test_no_flags_on_clean_data() -> None:
    prices = _make_prices(30)
    clean, flags = flag_quality_issues(prices)
    assert len(flags) == 0
    assert clean["is_flagged"].sum() == 0


def test_ohlc_inconsistency_flagged() -> None:
    prices = _make_prices(30)
    row = pl.DataFrame({
        "ticker": ["TEST"],
        "date": [date(2024, 1, 11)],
        "open": [100.0],
        "high": [99.0],
        "low": [101.0],
        "close": [100.0],
        "volume": [10000],
    })
    prices = pl.concat([prices, row]).unique(subset=["ticker", "date"], keep="last").sort("date")
    clean, flags = flag_quality_issues(prices)
    assert len(flags) > 0
    assert "ohlc_inconsistency" in flags["reason"].to_list()


def test_volume_zero_flagged() -> None:
    prices = _make_prices(30)
    row = pl.DataFrame({
        "ticker": ["TEST"],
        "date": [date(2024, 1, 16)],
        "open": [100.0],
        "high": [102.0],
        "low": [98.0],
        "close": [100.0],
        "volume": [0],
    })
    prices = pl.concat([prices, row]).unique(subset=["ticker", "date"], keep="last").sort("date")
    clean, flags = flag_quality_issues(prices)
    assert "volume_zero" in flags["reason"].to_list()
