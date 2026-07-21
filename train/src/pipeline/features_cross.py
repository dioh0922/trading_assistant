from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)


def compute_cross_sectional(
    features_single: pl.DataFrame,
    prices_clean: pl.DataFrame,
) -> pl.DataFrame:
    daily_prices = prices_clean.group_by(["ticker", "date"]).agg([
        pl.col("close").last(),
        pl.col("volume").last(),
    ])

    daily = daily_prices.with_columns([
        pl.col("close").pct_change().over("ticker").alias("daily_return"),
    ])

    cross = daily.group_by("date").agg([
        pl.col("ticker"),
        pl.col("daily_return"),
        pl.col("volume"),
    ]).explode(["ticker", "daily_return", "volume"])

    ranked = cross.with_columns([
        pl.col("daily_return").rank(method="average", descending=True).over("date").alias("_ret_rank"),
        pl.col("volume").rank(method="average", descending=True).over("date").alias("_vol_rank"),
    ])

    count_per_date = ranked.group_by("date").agg(pl.len().alias("_count"))

    ranked = ranked.join(count_per_date, on="date", how="left")
    ranked = ranked.with_columns([
        (pl.col("_ret_rank") / pl.col("_count")).alias("return_rank_pct"),
        (pl.col("_vol_rank") / pl.col("_count")).alias("volume_rank_pct"),
    ])

    sector_return = daily.group_by("date").agg([
        pl.col("daily_return").mean().alias("_market_avg_return"),
    ])

    ranked = ranked.join(sector_return, on="date", how="left")
    ranked = ranked.with_columns([
        (pl.col("daily_return") - pl.col("_market_avg_return")).alias("sector_relative_return"),
    ])

    cross_features = ranked.select([
        "ticker", "date", "return_rank_pct", "volume_rank_pct", "sector_relative_return",
    ])

    features_cross = features_single.join(
        cross_features, on=["ticker", "date"], how="left",
    )

    log.info("cross-sectional features added: %d rows", len(features_cross))
    return features_cross


def build_features_cross(
    features_single_path: Path,
    prices_clean_path: Path,
    output_path: Path,
) -> pl.DataFrame:
    features_single = pl.read_parquet(features_single_path)
    prices_clean = pl.read_parquet(prices_clean_path)

    log.info("loaded features_single (%d rows)", len(features_single))
    log.info("loaded prices_clean (%d rows)", len(prices_clean))

    features = compute_cross_sectional(features_single, prices_clean)
    # 入力DataFrameを解放（compute_cross_sectional内で参照が残っていないため安全）
    del features_single, prices_clean

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(output_path)
    log.info("wrote %s (%d rows)", output_path, len(features))
    return features


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3b: Cross-sectional features")
    parser.add_argument("--features-single", type=Path, default=Path("data/features/features_single.parquet"))
    parser.add_argument("--prices-clean", type=Path, default=Path("data/processed/prices_clean.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/features/features_cross.parquet"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_features_cross(args.features_single, args.prices_clean, args.output)
