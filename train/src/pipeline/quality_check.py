from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)


def flag_quality_issues(prices: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
  flags: list[pl.DataFrame] = []

  prices = prices.sort(["ticker", "date"])

  merged = prices.with_columns(
    [
      pl.col("close").pct_change().over("ticker").alias("_ret"),
      pl.col("volume").rolling_mean(window_size=20).over("ticker").alias("_vol_ma20"),
    ]
  )

  flag_return_spike = merged.filter(
    (pl.col("_ret").abs() > 0.50)
    & (pl.col("_vol_ma20").is_not_null())
    & (pl.col("_vol_ma20") > 0)
    & (pl.col("volume") < pl.col("_vol_ma20") * 0.01)
  ).select(
    pl.col("ticker"),
    pl.col("date"),
    pl.lit("return_spike_low_volume").alias("reason"),
  )
  flags.append(flag_return_spike)

  flag_volume_zero = merged.filter(
    (pl.col("volume") == 0) | (pl.col("volume").is_null())
  ).select(
    pl.col("ticker"),
    pl.col("date"),
    pl.lit("volume_zero").alias("reason"),
  )
  flags.append(flag_volume_zero)

  flag_hlc = merged.filter(
    (pl.col("high") < pl.col("low"))
    | (pl.col("close") > pl.col("high"))
    | (pl.col("close") < pl.col("low"))
  ).select(
    pl.col("ticker"),
    pl.col("date"),
    pl.lit("ohlc_inconsistency").alias("reason"),
  )
  flags.append(flag_hlc)

  if flags:
    flag_df = pl.concat(flags).unique(subset=["ticker", "date", "reason"])
  else:
    flag_df = pl.DataFrame(
      schema={"ticker": pl.Utf8, "date": pl.Date, "reason": pl.Utf8}
    )

  log.info("quality_check: %d flagged rows", len(flag_df))

  flagged_dates = flag_df.select(
    pl.col("ticker"),
    pl.col("date"),
    pl.lit(True).alias("is_flagged"),
  ).unique(subset=["ticker", "date"])

  prices_clean = prices.join(flagged_dates, on=["ticker", "date"], how="left")
  prices_clean = prices_clean.with_columns(pl.col("is_flagged").fill_null(False))

  return prices_clean, flag_df


def run_quality_check(
  prices_path: Path,
  output_clean_path: Path,
  output_flags_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
  prices = pl.read_parquet(prices_path)
  log.info("loaded %s (%d rows)", prices_path, len(prices))

  prices_clean, flag_df = flag_quality_issues(prices)

  output_clean_path.parent.mkdir(parents=True, exist_ok=True)
  output_flags_path.parent.mkdir(parents=True, exist_ok=True)

  prices_clean.write_parquet(output_clean_path)
  log.info("wrote %s (%d rows)", output_clean_path, len(prices_clean))

  flag_df.write_csv(output_flags_path)
  log.info("wrote %s (%d rows)", output_flags_path, len(flag_df))

  return prices_clean, flag_df


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Phase 2: Quality check")
  parser.add_argument(
    "--prices", type=Path, default=Path("data/processed/prices.parquet")
  )
  parser.add_argument(
    "--output-clean", type=Path, default=Path("data/processed/prices_clean.parquet")
  )
  parser.add_argument(
    "--output-flags", type=Path, default=Path("reports/quality_check_flags.csv")
  )
  args = parser.parse_args()

  logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )
  run_quality_check(args.prices, args.output_clean, args.output_flags)
