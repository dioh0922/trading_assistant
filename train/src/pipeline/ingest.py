from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)


def parse_single_csv(csv_path: Path) -> pl.DataFrame:
    ticker = csv_path.stem
    raw = pl.read_csv(csv_path, has_header=True, infer_schema=False)

    date_col = raw.columns[0]
    raw = raw.rename({date_col: "date"})

    meta_rows = {"Ticker", "Date", "date"}
    raw = raw.filter(~pl.col("date").is_in(meta_rows))

    col_map: dict[str, str] = {}
    for c in raw.columns:
        cl = c.strip().lower()
        if cl == "close":
            col_map[c] = "close"
        elif cl == "high":
            col_map[c] = "high"
        elif cl == "low":
            col_map[c] = "low"
        elif cl == "open":
            col_map[c] = "open"
        elif cl == "volume":
            col_map[c] = "volume"
    raw = raw.rename(col_map)

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{csv_path.name}: missing columns {missing}")

    df = raw.select(["date", "open", "high", "low", "close", "volume"])
    df = df.with_columns(
        pl.col("date").str.to_date("%Y-%m-%d", strict=False).alias("date"),
        pl.col("close").cast(pl.Float64, strict=False),
        pl.col("high").cast(pl.Float64, strict=False),
        pl.col("low").cast(pl.Float64, strict=False),
        pl.col("open").cast(pl.Float64, strict=False),
        pl.col("volume").cast(pl.Int64, strict=False),
        pl.lit(ticker).alias("ticker"),
    )

    df = df.drop_nulls(subset=["close"])
    df = df.sort("date")

    n_dup = df.select("date").is_duplicated().sum()
    if n_dup > 0:
        log.warning("%s: %d duplicate dates found (keeping first)", csv_path.name, n_dup)
        df = df.unique(subset=["date", "ticker"], keep="first")

    return df


def ingest_all(raw_dir: Path, output_path: Path, sample_tickers: int | None = None) -> pl.DataFrame:
    csv_paths = sorted(raw_dir.glob("*.csv"))
    if sample_tickers is not None:
        csv_paths = csv_paths[:sample_tickers]
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames: list[pl.DataFrame] = []
    for p in csv_paths:
        try:
            frames.append(parse_single_csv(p))
        except Exception as e:
            log.warning("Failed to parse %s: %s", p.name, e)

    if not frames:
        raise RuntimeError("No CSV files were successfully parsed")

    prices = pl.concat(frames)
    n_tickers = prices["ticker"].n_unique()
    log.info("ingest: %d tickers, %d rows", n_tickers, len(prices))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.write_parquet(output_path)
    log.info("wrote %s", output_path)
    return prices


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: CSV -> Parquet ingest")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/prices.parquet"))
    parser.add_argument("--sample-tickers", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ingest_all(args.raw_dir, args.output, sample_tickers=args.sample_tickers)
