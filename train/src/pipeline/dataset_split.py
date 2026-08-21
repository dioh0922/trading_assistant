from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Generator
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


def _compute_split_meta(
  df: pl.DataFrame,
  config: SplitConfig,
) -> dict:
  """結合済みDataFrameからスプリット定義（メタ情報のみ）を計算する。
  大きなDataFrameのコピーは作らず、ティッカーリストと日付境界だけを返す。
  """
  tickers = df["ticker"].unique().to_list()
  tickers.sort()
  rng = np.random.default_rng(42)
  rng.shuffle(tickers)

  n_holdout = int(len(tickers) * config.stock_holdout_ratio)
  stock_holdout_tickers = list(tickers[:n_holdout])
  cv_tickers = list(tickers[n_holdout:])
  log.info(
    "Stock holdout: %d tickers, CV: %d tickers (Total: %d)",
    len(stock_holdout_tickers),
    len(cv_tickers),
    len(tickers),
  )

  max_date = df["date"].max()
  if isinstance(max_date, str):
    max_date = datetime.date.fromisoformat(max_date)
  time_holdout_start = get_time_holdout_start(max_date, config.time_holdout_months)
  log.info("Time holdout start date: %s (Max date: %s)", time_holdout_start, max_date)

  # CV用の日付リストを取得（dfのサブセットから）
  cv_dates = (
    df.filter(
      pl.col("ticker").is_in(cv_tickers) & (pl.col("date") < time_holdout_start)
    )["date"]
    .unique()
    .sort()
    .to_list()
  )

  fold_boundaries = []
  n_dates = len(cv_dates)
  if n_dates >= config.n_folds:
    step = n_dates // config.n_folds
    for k in range(config.n_folds):
      val_start_idx = k * step
      val_end_idx = (k + 1) * step - 1 if k < config.n_folds - 1 else n_dates - 1
      fold_boundaries.append(
        {
          "k": k,
          "val_start": cv_dates[val_start_idx],
          "val_end": cv_dates[val_end_idx],
        }
      )
  else:
    log.warning(
      "Too few unique dates (%d) for %d folds; skipping CV folds.",
      n_dates,
      config.n_folds,
    )

  return {
    "stock_holdout_tickers": stock_holdout_tickers,
    "cv_tickers": cv_tickers,
    "time_holdout_start": time_holdout_start,
    "fold_boundaries": fold_boundaries,
    "purge_days": config.purge_days,
    "embargo_days": config.embargo_days,
  }


def _iter_splits(
  df: pl.DataFrame | pl.LazyFrame,
  meta: dict,
) -> Generator[tuple[str, pl.DataFrame], None, None]:
  """スプリット名とDataFrameを1件ずつyieldする（同時に複数を保持しない）。

  DataFrame / LazyFrame の両方を受け付け、LazyFrameの場合はcollectしてからyieldする。
  """
  _collect = (lambda x: x.collect()) if isinstance(df, pl.LazyFrame) else (lambda x: x)

  stock_holdout_tickers = meta["stock_holdout_tickers"]
  cv_tickers = meta["cv_tickers"]
  time_holdout_start = meta["time_holdout_start"]

  # stock holdout
  chunk = _collect(
    df.filter(pl.col("ticker").is_in(stock_holdout_tickers)).with_columns(
      [pl.lit(-1).alias("fold"), pl.lit("stock_holdout").alias("split_type")]
    )
  )
  yield "stock_holdout", chunk
  del chunk

  # time holdout
  chunk = _collect(
    df.filter(
      pl.col("ticker").is_in(cv_tickers) & (pl.col("date") >= time_holdout_start)
    ).with_columns(
      [pl.lit(-1).alias("fold"), pl.lit("time_holdout").alias("split_type")]
    )
  )
  yield "time_holdout", chunk
  del chunk

  # CV base（fold計算用）- 日付境界だけ使い、データ自体はここでは保持しない
  for fold_info in meta["fold_boundaries"]:
    k = fold_info["k"]
    val_start = fold_info["val_start"]
    val_end = fold_info["val_end"]
    purge = meta["purge_days"]
    embargo = meta["embargo_days"]

    # CVティッカー × time_holdout前 のデータのみ対象
    cv_base_filter = pl.col("ticker").is_in(cv_tickers) & (
      pl.col("date") < time_holdout_start
    )

    # valid
    val_chunk = _collect(
      df.filter(
        cv_base_filter & (pl.col("date") >= val_start) & (pl.col("date") <= val_end)
      ).with_columns([pl.lit(k).alias("fold"), pl.lit("valid").alias("split_type")])
    )
    yield f"fold_{k}_valid", val_chunk
    del val_chunk

    # train (purge + embargo)
    train_before = val_start - datetime.timedelta(days=purge)
    train_after = val_end + datetime.timedelta(days=purge + embargo)
    train_chunk = _collect(
      df.filter(
        cv_base_filter
        & ((pl.col("date") < train_before) | (pl.col("date") > train_after))
      ).with_columns([pl.lit(k).alias("fold"), pl.lit("train").alias("split_type")])
    )
    yield f"fold_{k}_train", train_chunk
    del train_chunk


def generate_splits(
  features_df: pl.DataFrame,
  labels_df: pl.DataFrame,
  config: SplitConfig,
) -> dict[str, pl.DataFrame]:
  """後方互換のためgenerate_splitsを残す（テスト・小規模用）。
  大規模データにはbuild_dataset_splitを使うこと。
  """
  df = features_df.join(labels_df, on=["ticker", "date"], how="inner")
  if len(df) == 0:
    log.warning("Joined dataset is empty.")
    return {}
  meta = _compute_split_meta(df, config)
  return {name: chunk for name, chunk in _iter_splits(df, meta)}


def assign_splits(
  features_df: pl.DataFrame,
  labels_df: pl.DataFrame,
  config: SplitConfig,
) -> pl.DataFrame:
  splits = generate_splits(features_df, labels_df, config)
  if not splits:
    return pl.DataFrame()
  return pl.concat(list(splits.values()))


def build_dataset_split(
  features_path: Path,
  labels_path: Path,
  output_path: Path,
  config: SplitConfig,
) -> None:
  """メモリ効率化版のデータセット分割。

  features と labels を join した後、スプリットを1件ずつ計算・書き込み・解放する。
  splitsの辞書に全スプリットを同時保持しないため、ピークメモリを大幅に削減できる。
  """
  log.info("Loading features from %s (lazy)", features_path)
  # scan_parquetで必要列のみフィルタしてからcollect
  features_lazy = pl.scan_parquet(features_path)
  log.info("Loading labels from %s (lazy)", labels_path)
  labels_lazy = pl.scan_parquet(labels_path)

  log.info("Joining features and labels (lazy)...")
  joined_lazy = features_lazy.join(labels_lazy, on=["ticker", "date"], how="inner")

  # メタ情報計算のために最小限のカラムだけcollect
  log.info("Collecting ticker/date metadata for split planning...")
  meta_df = joined_lazy.select(["ticker", "date"]).collect()
  if len(meta_df) == 0:
    log.error("Joined dataset is empty. Aborting.")
    return

  # スプリット定義（メタ情報のみ）を計算
  meta = _compute_split_meta(meta_df, config)
  del meta_df  # メタ情報計算後は不要

  if output_path.exists() and output_path.is_file():
    output_path.unlink()
  output_path.mkdir(parents=True, exist_ok=True)

  # スプリットを1件ずつcollect→書き込み（全データを一括collectせずLazyFrame→個別collectでピークメモリ削減）
  for name, split_df in _iter_splits(joined_lazy, meta):
    split_file = output_path / f"{name}.parquet"
    split_df.write_parquet(split_file)
    log.info("Saved split '%s' to %s (%d rows)", name, split_file, len(split_df))

  log.info("Saved split dataset directory to %s", output_path)


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Phase 5: Purged K-Fold dataset split")
  parser.add_argument(
    "--config", type=Path, default=Path("config/config.yaml"), help="config file path"
  )
  parser.add_argument(
    "--features", type=Path, default=Path("data/features/features_cross.parquet")
  )
  parser.add_argument(
    "--labels", type=Path, required=True, help="path to labels parquet"
  )
  parser.add_argument(
    "--output", type=Path, required=True, help="path to output parquet"
  )
  args = parser.parse_args()

  logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )

  app_config = load_config(args.config)
  build_dataset_split(args.features, args.labels, args.output, app_config.split)
