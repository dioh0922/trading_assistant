from __future__ import annotations

import datetime
import logging
from pathlib import Path
import numpy as np
import polars as pl

from pipeline.config import SplitConfig, load_config

log = logging.getLogger(__name__)


def get_time_holdout_start(max_date: datetime.date, months: int) -> datetime.date:
    y = max_date.year
    m = max_date.month - months
    while m <= 0:
        m += 12
        y -= 1
    # Safe day mapping (handling short months, e.g. Feb 30th)
    d = min(max_date.day, 28)
    return datetime.date(y, m, d)


def assign_splits(
    features_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    config: SplitConfig,
) -> pl.DataFrame:
    # 1. Join features and labels
    # Use 'ticker' and 'date' as key
    df = features_df.join(labels_df, on=["ticker", "date"], how="inner")
    if len(df) == 0:
        log.warning("Joined dataset is empty.")
        return pl.DataFrame()

    # 2. Stock Holdout Split
    tickers = df["ticker"].unique().to_list()
    tickers.sort()  # Sort to ensure reproducibility with seed
    rng = np.random.default_rng(42)
    rng.shuffle(tickers)

    n_holdout = int(len(tickers) * config.stock_holdout_ratio)
    stock_holdout_tickers = set(tickers[:n_holdout])
    cv_tickers = set(tickers[n_holdout:])
    log.info(
        "Stock holdout: %d tickers, CV: %d tickers (Total: %d)",
        len(stock_holdout_tickers),
        len(cv_tickers),
        len(tickers),
    )

    # 3. Time Holdout Split
    max_date = df["date"].max()
    if isinstance(max_date, str):
        # Convert if string
        df = df.with_columns(pl.col("date").str.to_date())
        max_date = df["date"].max()
    
    time_holdout_start = get_time_holdout_start(max_date, config.time_holdout_months)
    log.info("Time holdout start date: %s (Max date: %s)", time_holdout_start, max_date)

    # Label the whole dataset based on stock & time holdout first
    # We will replicate data for folds later
    stock_holdout_mask = pl.col("ticker").is_in(list(stock_holdout_tickers))
    time_holdout_mask = (pl.col("ticker").is_in(list(cv_tickers))) & (pl.col("date") >= time_holdout_start)
    cv_mask = (pl.col("ticker").is_in(list(cv_tickers))) & (pl.col("date") < time_holdout_start)

    stock_holdout_df = df.filter(stock_holdout_mask).with_columns([
        pl.lit(-1).alias("fold"),
        pl.lit("stock_holdout").alias("split_type"),
    ])

    time_holdout_df = df.filter(time_holdout_mask).with_columns([
        pl.lit(-1).alias("fold"),
        pl.lit("time_holdout").alias("split_type"),
    ])

    cv_base_df = df.filter(cv_mask)
    if len(cv_base_df) == 0:
        log.warning("No data left for CV after holdout splits.")
        return pl.concat([stock_holdout_df, time_holdout_df])

    # 4. Purged K-Fold
    unique_dates = cv_base_df["date"].unique().sort().to_list()
    n_dates = len(unique_dates)
    if n_dates < config.n_folds:
        raise ValueError(f"Too few unique dates ({n_dates}) for {config.n_folds} folds.")

    step = n_dates // config.n_folds
    fold_dfs = []

    for k in range(config.n_folds):
        val_start_idx = k * step
        val_end_idx = (k + 1) * step - 1 if k < config.n_folds - 1 else n_dates - 1
        
        val_start_date = unique_dates[val_start_idx]
        val_end_date = unique_dates[val_end_idx]

        # Valid mask
        val_mask = (pl.col("date") >= val_start_date) & (pl.col("date") <= val_end_date)
        valid_chunk = cv_base_df.filter(val_mask).with_columns([
            pl.lit(k).alias("fold"),
            pl.lit("valid").alias("split_type"),
        ])
        fold_dfs.append(valid_chunk)

        # Train mask with purging and embargo
        # train date must be:
        # date < val_start_date - purge_days OR date > val_end_date + purge_days + embargo_days
        train_before_limit = val_start_date - datetime.timedelta(days=config.purge_days)
        train_after_limit = val_end_date + datetime.timedelta(days=config.purge_days + config.embargo_days)

        train_mask = (pl.col("date") < train_before_limit) | (pl.col("date") > train_after_limit)
        train_chunk = cv_base_df.filter(train_mask).with_columns([
            pl.lit(k).alias("fold"),
            pl.lit("train").alias("split_type"),
        ])
        fold_dfs.append(train_chunk)

    result_df = pl.concat([stock_holdout_df, time_holdout_df] + fold_dfs)
    return result_df


def build_dataset_split(
    features_path: Path,
    labels_path: Path,
    output_path: Path,
    config: SplitConfig,
) -> None:
    log.info("Loading features from %s", features_path)
    features_df = pl.read_parquet(features_path)
    log.info("Loading labels from %s", labels_path)
    labels_df = pl.read_parquet(labels_path)

    log.info("Splitting dataset...")
    result_df = assign_splits(features_df, labels_df, config)

    if len(result_df) > 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.write_parquet(output_path)
        log.info("Saved split dataset to %s (%d rows)", output_path, len(result_df))
    else:
        log.error("Failed to split dataset (empty result)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 5: Purged K-Fold dataset split")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"), help="config file path")
    parser.add_argument("--features", type=Path, default=Path("data/features/features_cross.parquet"))
    parser.add_argument("--labels", type=Path, required=True, help="path to labels parquet")
    parser.add_argument("--output", type=Path, required=True, help="path to output parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    app_config = load_config(args.config)
    build_dataset_split(args.features, args.labels, args.output, app_config.split)
