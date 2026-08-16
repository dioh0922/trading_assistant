from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

log = logging.getLogger(__name__)


def compute_features_single(
  df: pl.DataFrame, windows: list[int], rsi_period: int = 14, atr_period: int = 14
) -> pl.DataFrame:
  out = df.sort("date")

  exprs: list[pl.Expr] = []
  for w in windows:
    exprs.extend(
      [
        (pl.col("close").pct_change(w)).alias(f"return_{w}d"),
        (pl.col("close") / pl.col("close").rolling_mean(window_size=w) - 1).alias(
          f"ma_dev_{w}"
        ),
        (pl.col("close").log() - pl.col("close").shift(1).log())
        .rolling_std(window_size=w)
        .alias(f"volatility_{w}d"),
      ]
    )

  atr_high_low = pl.col("high") - pl.col("low")
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
  rsi_expr = (
    pl.when(avg_loss == 0).then(pl.lit(100.0)).otherwise(100.0 - (100.0 / (1.0 + rs)))
  )
  out = out.with_columns(rsi_expr.alias("rsi_14"))

  out = out.with_columns(
    [
      (pl.col("volume") / pl.col("volume").rolling_mean(window_size=20)).alias(
        "volume_ratio_20d"
      ),
      (
        (pl.col("close") - pl.col("low").rolling_min(window_size=60))
        / (
          pl.col("high").rolling_max(window_size=60)
          - pl.col("low").rolling_min(window_size=60)
        )
      ).alias("range_position_60d"),
    ]
  )

  return out


def build_features_single(
  prices_path: Path,
  output_path: Path,
  windows: list[int] | None = None,
  rsi_period: int = 14,
  atr_period: int = 14,
  chunk_size: int = 200,
) -> None:
  """銘柄ごとにfeatureを計算し逐次parquetへ書き込む（メモリ節約版）。

  全銘柄分のDataFrameをpartsリストに蓄積してからpl.concatする実装は
  ピーク時に「全銘柄の計算結果 × 2」のメモリを消費する。
  本実装は chunk_size 銘柄ずつ処理してpyarrow ParquetWriterで
  逐次アペンドするためピークメモリを大幅に削減できる。
  """
  if windows is None:
    windows = [5, 20, 60]

  # --- スキーマ確認用に先頭1銘柄だけ読んでスキーマを確定する ---
  tickers_all = (
    pl.scan_parquet(prices_path)
    .select("ticker")
    .collect()["ticker"]
    .unique()
    .sort()
    .to_list()
  )
  log.info("loaded ticker list: %d tickers from %s", len(tickers_all), prices_path)

  output_path.parent.mkdir(parents=True, exist_ok=True)

  writer: pq.ParquetWriter | None = None
  total_rows = 0

  try:
    for chunk_start in range(0, len(tickers_all), chunk_size):
      chunk_tickers = tickers_all[chunk_start : chunk_start + chunk_size]
      # チャンク分だけロード
      chunk_df = (
        pl.scan_parquet(prices_path)
        .filter(pl.col("ticker").is_in(chunk_tickers))
        .collect()
      )

      for ticker in chunk_tickers:
        group = chunk_df.filter(pl.col("ticker") == ticker)
        if len(group) == 0:
          continue
        feat = compute_features_single(group, windows, rsi_period, atr_period)
        arrow_table = feat.to_arrow()
        if writer is None:
          writer = pq.ParquetWriter(str(output_path), arrow_table.schema)
        writer.write_table(arrow_table)
        total_rows += len(feat)

      log.info(
        "features_single: processed %d/%d tickers, total rows so far: %d",
        min(chunk_start + chunk_size, len(tickers_all)),
        len(tickers_all),
        total_rows,
      )
      # チャンクを明示的に解放
      del chunk_df

  finally:
    if writer is not None:
      writer.close()

  log.info("wrote %s (%d rows)", output_path, total_rows)


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Phase 3a: Single-stock features")
  parser.add_argument(
    "--prices", type=Path, default=Path("data/processed/prices_clean.parquet")
  )
  parser.add_argument(
    "--output", type=Path, default=Path("data/features/features_single.parquet")
  )
  parser.add_argument("--windows", type=int, nargs="+", default=[5, 20, 60])
  args = parser.parse_args()

  logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )
  build_features_single(args.prices, args.output, windows=args.windows)
