from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)


def compute_features_single(df: pl.DataFrame, windows: list[int], rsi_period: int = 14, atr_period: int = 14) -> pl.DataFrame:
    out = df.sort("date")

    exprs: list[pl.Expr] = []
    for w in windows:
        exprs.extend([
            (pl.col("close").pct_change(w)).alias(f"return_{w}d"),
            (pl.col("close") / pl.col("close").rolling_mean(window_size=w) - 1).alias(f"ma_dev_{w}"),
            (
                (pl.col("close").log() - pl.col("close").shift(1).log())
            ).rolling_std(window_size=w).alias(f"volatility_{w}d"),
        ])

    atr_high_low = (pl.col("high") - pl.col("low"))
    atr_high_close = (pl.col("high") - pl.col("close").shift(1)).abs()
    atr_low_close = (pl.col("low") - pl.col("close").shift(1)).abs()
    true_range = pl.max_horizontal(atr_high_low, atr_high_close, atr_low_close)
    atr = true_range.rolling_mean(window_size=atr_period).alias("atr_14")
    exprs.append(atr)

    out = out.with_columns(exprs)

    delta = pl.col("close").diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    avg_gain = gain.rolling_mean(window_size=rsi_period)
    avg_loss = loss.rolling_mean(window_size=rsi_period)
    rs = avg_gain / avg_loss
    rsi_expr = pl.when(avg_loss == 0).then(pl.lit(100.0)).otherwise(100.0 - (100.0 / (1.0 + rs)))
    out = out.with_columns(rsi_expr.alias("rsi_14"))

    out = out.with_columns([
        (pl.col("volume") / pl.col("volume").rolling_mean(window_size=20)).alias("volume_ratio_20d"),
        (
            (pl.col("close") - pl.col("low").rolling_min(window_size=60))
            / (pl.col("high").rolling_max(window_size=60) - pl.col("low").rolling_min(window_size=60))
        ).alias("range_position_60d"),
    ])

    return out


def build_features_single(prices_path: Path, output_path: Path, windows: list[int] | None = None, rsi_period: int = 14, atr_period: int = 14) -> pl.DataFrame:
    prices = pl.read_parquet(prices_path)
    log.info("loaded %s (%d rows)", prices_path, len(prices))

    if windows is None:
        windows = [5, 20, 60]

    parts: list[pl.DataFrame] = []
    for ticker, group in prices.group_by("ticker"):
        feat = compute_features_single(group, windows, rsi_period, atr_period)
        parts.append(feat)

    features = pl.concat(parts)
    log.info("features: %d rows, %d columns", len(features), len(features.columns))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(output_path)
    log.info("wrote %s", output_path)
    return features


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3a: Single-stock features")
    parser.add_argument("--prices", type=Path, default=Path("data/processed/prices_clean.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/features/features_single.parquet"))
    parser.add_argument("--windows", type=int, nargs="+", default=[5, 20, 60])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_features_single(args.prices, args.output, windows=args.windows)
