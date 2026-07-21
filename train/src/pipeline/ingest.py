from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import polars as pl
import pyarrow.parquet as pq

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


def ingest_all(
    raw_dir: Path,
    output_path: Path,
    sample_tickers: Optional[int] = None,
    extract_ratio: float = 0.0,
    seed: int = 42,
) -> tuple[None, list[str]]:
    """全CSVを読み込みparquetへ変換する（メモリ節約版）。

    全銘柄をframesリストに蓄積してからpl.concatするパターンは
    ピーク時に「全データ × 2」のメモリを消費する。
    本実装はpyarrow ParquetWriterで銀柄ごとに逐次アペンドするため
    同時メモリ使用量を大幅に削減できる。

    Args:
        raw_dir: CSVファイルが格納されたディレクトリ。
        output_path: 出力先parquetパス。
        sample_tickers: 先頭N銘柄に絞る（Noneで全件）。
        extract_ratio: ランダムに除外する銘柄の割合 (0.0〜1.0)。
                       除外された銘柄はパイプラインの学習に使われず
                       extract.txt に書き出される。
        seed: ランダムシード（再現性確保用）。

    Returns:
        (None, extracted_tickers list)
    """
    csv_paths = sorted(raw_dir.glob("*.csv"))
    if sample_tickers is not None:
        csv_paths = csv_paths[:sample_tickers]
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    # extract_ratio に応じて銘柄をランダム除外
    extracted_tickers: list[str] = []
    if extract_ratio > 0.0:
        all_stems = [p.stem for p in csv_paths]
        rng = random.Random(seed)
        n_extract = max(1, round(len(all_stems) * extract_ratio))
        extracted_tickers = sorted(rng.sample(all_stems, n_extract))
        extracted_set = set(extracted_tickers)
        csv_paths = [p for p in csv_paths if p.stem not in extracted_set]
        log.info(
            "ingest: extract_ratio=%.2f → %d tickers excluded, %d tickers used",
            extract_ratio,
            len(extracted_tickers),
            len(csv_paths),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # pyarrow ParquetWriter で銘柄ごとに逐次書き込み（フレームリストに全銘柄を蓄積しない）
    writer: pq.ParquetWriter | None = None
    n_tickers = 0
    total_rows = 0
    try:
        for p in csv_paths:
            try:
                df = parse_single_csv(p)
            except Exception as e:
                log.warning("Failed to parse %s: %s", p.name, e)
                continue
            arrow_table = df.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), arrow_table.schema)
            writer.write_table(arrow_table)
            n_tickers += 1
            total_rows += len(df)
    finally:
        if writer is not None:
            writer.close()

    if n_tickers == 0:
        raise RuntimeError("No CSV files were successfully parsed")

    log.info("ingest: %d tickers, %d rows", n_tickers, total_rows)
    log.info("wrote %s", output_path)
    return None, extracted_tickers


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: CSV -> Parquet ingest")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/prices.parquet"))
    parser.add_argument("--sample-tickers", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ingest_all(args.raw_dir, args.output, sample_tickers=args.sample_tickers)
