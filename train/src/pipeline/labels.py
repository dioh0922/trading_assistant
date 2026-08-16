from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

log = logging.getLogger(__name__)


def triple_barrier_label(
  prices: pl.DataFrame,
  upper: float | None,
  lower: float | None,
  max_days: int,
) -> pl.DataFrame:
  close = prices["close"].to_numpy()
  n = len(close)

  labels = np.empty(n, dtype="U10")
  days_to_hit = np.full(n, np.nan)

  for i in range(n):
    entry_price = close[i]
    end_idx = min(i + max_days + 1, n)

    if i + 1 >= end_idx:
      labels[i] = "timeout"
      continue

    window = close[i + 1 : end_idx]

    first_upper = np.inf
    first_lower = np.inf

    if upper is not None:
      upper_price = entry_price * (1 + upper)
      hits = np.where(window >= upper_price)[0]
      if len(hits):
        first_upper = hits[0]

    if lower is not None:
      lower_price = entry_price * (1 + lower)
      hits = np.where(window <= lower_price)[0]
      if len(hits):
        first_lower = hits[0]

    if first_upper == np.inf and first_lower == np.inf:
      labels[i] = "timeout"
      days_to_hit[i] = np.nan
    elif first_upper <= first_lower:
      labels[i] = "upper"
      days_to_hit[i] = first_upper + 1
    else:
      labels[i] = "lower"
      days_to_hit[i] = first_lower + 1

  result = prices.select(["ticker", "date"]).clone()
  result = result.with_columns(
    [
      pl.Series("label", labels),
      pl.Series("days_to_hit", days_to_hit),
    ]
  )
  return result


def compute_sample_weights(labels_df: pl.DataFrame, max_days: int) -> pl.DataFrame:
  """
  ラベルの時間窓（起点日 ～ 起点日+max_days）が重なっているサンプルほど
  重みを下げる（Lopez de Prado の overlapping-label weighting の簡易版）。

  修正点（元実装からの変更）:
    元の実装は「銘柄内の全サンプル同士を総当たり」する O(n^2) ループだったため、
    4000銘柄・数百万行規模では実質的に終わらなかった（O(n^2) は 1銘柄 2000行でも
    400万回、4000銘柄分で計算量が破綻する）。

    2つの時間窓 [entry_i, entry_i+max_days) と [entry_j, entry_j+max_days) が
    重なる条件は、単純に |entry_i - entry_j| < max_days に等しい（窓の長さが
    同じため）。日付でソート済みなら、この条件を満たす件数は
    np.searchsorted で O(log n) で求まるので、銘柄ごとに O(m log m) で済む。
    結果は元のO(n^2)実装と完全に同じ値になることを検証済み。
  """
  tickers = labels_df["ticker"].to_numpy()
  dates = labels_df["date"].to_numpy().astype("datetime64[D]")

  weights = np.ones(len(labels_df))

  # 銘柄→日付の順にソートしておく（searchsortedは配列がソート済みである必要があるため）
  order = np.lexsort((dates, tickers))
  sorted_tickers = tickers[order]
  sorted_dates = dates[order]

  max_days_td = np.timedelta64(max_days, "D")
  sorted_weights = np.ones(len(labels_df))

  n = len(sorted_tickers)
  start = 0
  while start < n:
    end = start
    while end < n and sorted_tickers[end] == sorted_tickers[start]:
      end += 1

    t_dates = sorted_dates[start:end]  # この銘柄区間はすでに日付昇順

    # |t_dates[i] - t_dates[j]| < max_days を満たすjの個数を一括で求める
    left_idx = np.searchsorted(t_dates, t_dates - max_days_td, side="right")
    right_idx = np.searchsorted(t_dates, t_dates + max_days_td, side="left")
    overlap_count = (right_idx - left_idx) - 1  # 自分自身を除く
    overlap_count = np.clip(overlap_count, 0, None)

    sorted_weights[start:end] = 1.0 / (overlap_count + 1)
    start = end

  weights[order] = sorted_weights

  labels_df = labels_df.with_columns(pl.Series("sample_weight", weights))
  return labels_df


def build_labels_task1(
  prices_path: Path,
  output_path: Path,
  horizon_days: int = 20,
  up_threshold: float = 0.03,
  down_threshold: float = -0.03,
) -> None:
  prices = pl.read_parquet(prices_path)
  log.info("task1: loaded %d rows", len(prices))

  output_path.parent.mkdir(parents=True, exist_ok=True)
  writer: pq.ParquetWriter | None = None
  total_rows = 0
  try:
    for ticker, group in prices.group_by("ticker"):
      g = group.sort("date")
      ret = g["close"].pct_change(horizon_days).to_numpy()
      n = len(ret)

      labels = np.empty(n, dtype="U10")
      for i in range(n):
        if np.isnan(ret[i]):
          labels[i] = "unknown"
        elif ret[i] >= up_threshold:
          labels[i] = "up"
        elif ret[i] <= down_threshold:
          labels[i] = "down"
        else:
          labels[i] = "range"

      df = g.select(["ticker", "date"]).clone()
      df = df.with_columns(
        [
          pl.Series("label", labels),
          pl.lit(horizon_days).alias("horizon_days"),
        ]
      )
      df = df.filter(pl.col("label") != "unknown")
      if len(df) == 0:
        continue
      arrow_table = df.to_arrow()
      if writer is None:
        writer = pq.ParquetWriter(str(output_path), arrow_table.schema)
      writer.write_table(arrow_table)
      total_rows += len(df)
  finally:
    if writer is not None:
      writer.close()

  log.info("task1: wrote %s (%d rows)", output_path, total_rows)


def build_labels_task2(
  prices_path: Path,
  output_path: Path,
  upper: float = 0.10,
  lower: float = -0.05,
  max_days: int = 60,
) -> None:
  prices = pl.read_parquet(prices_path)
  log.info("task2: loaded %d rows", len(prices))

  output_path.parent.mkdir(parents=True, exist_ok=True)
  writer: pq.ParquetWriter | None = None
  total_rows = 0
  try:
    for ticker, group in prices.group_by("ticker"):
      g = group.sort("date")
      lbl = triple_barrier_label(g, upper=upper, lower=lower, max_days=max_days)
      lbl = compute_sample_weights(lbl, max_days)
      arrow_table = lbl.to_arrow()
      if writer is None:
        writer = pq.ParquetWriter(str(output_path), arrow_table.schema)
      writer.write_table(arrow_table)
      total_rows += len(lbl)
  finally:
    if writer is not None:
      writer.close()

  log.info("task2: wrote %s (%d rows)", output_path, total_rows)


def build_labels_task3(
  prices_path: Path,
  output_path: Path,
  upper_pcts: list[float] | None = None,
  max_days: int = 60,
) -> None:
  if upper_pcts is None:
    upper_pcts = [0.05, 0.10, 0.15, 0.20, 0.30]

  prices = pl.read_parquet(prices_path)
  log.info("task3: loaded %d rows, upper_pcts=%s", len(prices), upper_pcts)

  output_path.parent.mkdir(parents=True, exist_ok=True)

  # target_returnごとに流れ作業して逐次書き込む。
  # 全target_return・全銘柄をall_partsに蓄積する元の実装は
  # 「銘柄数 × 5」分のDataFrameが同時メモリに乗るため非常に重い。
  writer: pq.ParquetWriter | None = None
  total_rows = 0

  try:
    for pct in upper_pcts:
      for ticker, group in prices.group_by("ticker"):
        g = group.sort("date")
        lbl = triple_barrier_label(g, upper=pct, lower=None, max_days=max_days)
        lbl = lbl.with_columns(pl.lit(pct).alias("target_return"))
        lbl = compute_sample_weights(lbl, max_days)

        arrow_table = lbl.to_arrow()
        if writer is None:
          writer = pq.ParquetWriter(str(output_path), arrow_table.schema)
        writer.write_table(arrow_table)
        total_rows += len(lbl)

      log.info("task3: pct=%.2f done (%d rows)", pct, total_rows)
  finally:
    if writer is not None:
      writer.close()

  log.info("task3: wrote %s (%d rows)", output_path, total_rows)


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(
    description="Phase 4: Triple-barrier label generation"
  )
  parser.add_argument(
    "--prices", type=Path, default=Path("data/processed/prices_clean.parquet")
  )
  parser.add_argument("--labels-dir", type=Path, default=Path("data/labels"))
  parser.add_argument(
    "--task", choices=["task1", "task2", "task3", "all"], default="all"
  )
  args = parser.parse_args()

  logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )

  if args.task in ("task1", "all"):
    build_labels_task1(args.prices, args.labels_dir / "labels_task1.parquet")
  if args.task in ("task2", "all"):
    build_labels_task2(args.prices, args.labels_dir / "labels_task2.parquet")
  if args.task in ("task3", "all"):
    build_labels_task3(args.prices, args.labels_dir / "labels_task3.parquet")
