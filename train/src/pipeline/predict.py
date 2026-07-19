from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from pipeline.config import load_config
from pipeline.ingest import parse_single_csv
from pipeline.features_single import compute_features_single

log = logging.getLogger("predict")


def load_ensemble_models(models_dir: Path) -> tuple[list[joblib.load], dict]:
    metadata_path = models_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    n_folds = metadata["train_info"]["n_folds"]
    models = []
    for k in range(n_folds):
        model_path = models_dir / f"fold{k}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        models.append(joblib.load(model_path))

    return models, metadata


def get_latest_features(
    config,
    ticker: str,
    data_dir: Path | None = None,
    feature_cols: list[str] = [],
) -> pl.DataFrame:
    # 1. Check if CSV file exists in data_dir or config raw_dir
    csv_dir = data_dir if data_dir is not None else Path(config.data.raw_dir)
    csv_path = csv_dir / f"{ticker}.csv"
    
    if csv_path.exists():
        log.info("Loading latest price data from CSV: %s", csv_path)
        prices_df = parse_single_csv(csv_path)
        # Compute single features dynamically
        windows = config.features.windows
        rsi_period = config.features.rsi_period
        atr_period = config.features.atr_period
        
        feats_df = compute_features_single(prices_df, windows, rsi_period, atr_period)
        latest = feats_df.tail(1)
        
        # Fill missing features (like cross-sectional ones) with sensible defaults
        for col in feature_cols:
            if col not in latest.columns:
                if "rank" in col or "pct" in col:
                    default_val = 0.5
                else:
                    default_val = 0.0
                latest = latest.with_columns(pl.lit(default_val).alias(col))
        
        return latest

    # 2. Fallback: Load from processed Parquet files
    features_path = None
    if data_dir is not None:
        features_path = data_dir / "features_cross.parquet"
        if not features_path.exists():
            features_path = data_dir / "features_single.parquet"
    else:
        features_path = Path(config.data.features_dir) / "features_cross.parquet"
        if not features_path.exists():
            features_path = Path(config.data.features_dir) / "features_single.parquet"

    if features_path is None or not features_path.exists():
        raise FileNotFoundError(
            f"No price CSV found at {csv_path} and no features Parquet found. "
            "Please run run_pipeline.py or place price CSVs appropriately."
        )

    log.info("Loading features from Parquet Cache: %s", features_path)
    df = pl.read_parquet(features_path)
    ticker_df = df.filter(pl.col("ticker") == ticker).sort("date")
    if len(ticker_df) == 0:
        raise ValueError(f"No feature data found for ticker: {ticker}")

    # Get the latest row
    latest = ticker_df.tail(1)
    return latest


def predict_trend(
    config_path: Path,
    ticker: str,
    data_dir: Path | None = None,
) -> None:
    config = load_config(config_path)
    models_dir = Path("models") / "task1"

    models, metadata = load_ensemble_models(models_dir)
    feature_cols = metadata["feature_columns"]
    classes = metadata["label_classes"]

    latest_df = get_latest_features(config, ticker, data_dir, feature_cols)
    latest_date = latest_df["date"][0]
    latest_close = latest_df["close"][0]

    X = latest_df.select(feature_cols).to_pandas()

    # Ensemble prediction
    probas = [m.predict_proba(X) for m in models]
    avg_proba = np.mean(probas, axis=0)[0]

    print(f"\n[トレンド予測 (Task 1)]")
    print(f"銘柄: {ticker} (最新日付: {latest_date}, 終値: {latest_close:.1f}円)")
    print("-" * 50)
    for cls, p in sorted(zip(classes, avg_proba), key=lambda x: -x[1]):
        print(f"  {cls:<10s}: {p * 100:5.1f}%")
    print("-" * 50)


def predict_exit(
    config_path: Path,
    ticker: str,
    entry_price: float | None,
    data_dir: Path | None = None,
) -> None:
    config = load_config(config_path)
    models_dir = Path("models") / "task2"

    models, metadata = load_ensemble_models(models_dir)
    feature_cols = metadata["feature_columns"]
    classes = metadata["label_classes"]
    barrier = metadata["barrier_config"]

    latest_df = get_latest_features(config, ticker, data_dir, feature_cols)
    latest_date = latest_df["date"][0]
    latest_close = latest_df["close"][0]

    X = latest_df.select(feature_cols).to_pandas()

    # Ensemble prediction
    probas = [m.predict_proba(X) for m in models]
    avg_proba = np.mean(probas, axis=0)[0]

    # Calculate actual price levels if entry price provided
    ref_price = entry_price if entry_price is not None else latest_close

    upper_price = ref_price * (1 + barrier["upper_barrier"])
    lower_price = ref_price * (1 + barrier["lower_barrier"])

    print(f"\n[利確/損切り判定 (Task 2)]")
    print(f"銘柄: {ticker} (最新日付: {latest_date}, 終値: {latest_close:.1f}円)")
    print(f"基準価格: {ref_price:.1f}円 (利確価格 (+{barrier['upper_barrier']*100:.1f}%): {upper_price:.1f}円, 損切り価格 ({barrier['lower_barrier']*100:.1f}%): {lower_price:.1f}円)")
    print("-" * 50)
    
    LABEL_MAP = {"upper": "利確 (upper)", "lower": "損切り (lower)", "timeout": "時間切れ (timeout)"}
    
    for cls, p in sorted(zip(classes, avg_proba), key=lambda x: -x[1]):
        label = LABEL_MAP.get(cls, cls)
        print(f"  {label:<15s}: {p * 100:5.1f}%")
    print("-" * 50)


def predict_target(
    config_path: Path,
    ticker: str,
    target_price: float,
    data_dir: Path | None = None,
) -> None:
    config = load_config(config_path)
    
    # 1. Load hit classification model
    hit_models, hit_metadata = load_ensemble_models(Path("models") / "task3_hit")
    feature_cols = hit_metadata["feature_columns"]

    # 2. Load days regression model
    days_models, _ = load_ensemble_models(Path("models") / "task3_days")

    latest_df = get_latest_features(config, ticker, data_dir, feature_cols)
    latest_date = latest_df["date"][0]
    latest_close = latest_df["close"][0]

    # Calculate target return
    target_return = (target_price - latest_close) / latest_close

    # Add target_return to features
    latest_df = latest_df.with_columns(pl.lit(target_return).alias("target_return"))
    X = latest_df.select(feature_cols).to_pandas()

    # Ensemble prediction for hit
    probas = [m.predict_proba(X)[:, 1] for m in hit_models]
    hit_prob = np.mean(probas, axis=0)[0]

    # Ensemble prediction for days
    preds_days = [m.predict(X) for m in days_models]
    predicted_days = np.mean(preds_days, axis=0)[0]

    print(f"\n[目標価格到達予測 (Task 3)]")
    print(f"銘柄: {ticker} (最新日付: {latest_date}, 終値: {latest_close:.1f}円)")
    print(f"目標価格: {target_price:.1f}円 (必要リターン: {target_return * 100:+.2f}%)")
    print("-" * 50)
    print(f"  到達確率: {hit_prob * 100:5.1f}%")
    if hit_prob >= 0.5:
        print(f"  目標到達時の予想日数: {predicted_days:.1f} 日")
    else:
        print(f"  目標到達時の予想日数: {predicted_days:.1f} 日 (※到達確率は 50% 未満です)")
    print("-" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8: Ensemble Inference CLI")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"), help="Config file path")
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    trend_parser = subparsers.add_parser("trend", help="Predict trend (Task 1)")
    trend_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker code")
    trend_parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing feature data")

    exit_parser = subparsers.add_parser("exit", help="Predict exit decision (Task 2)")
    exit_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker code")
    exit_parser.add_argument("--entry-price", type=float, default=None, help="Position entry price")
    exit_parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing feature data")

    target_parser = subparsers.add_parser("target", help="Predict target return hit (Task 3)")
    target_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker code")
    target_parser.add_argument("--target-price", type=float, required=True, help="Target price to hit")
    target_parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing feature data")

    args = parser.parse_args()

    if args.command == "trend":
        predict_trend(args.config, args.ticker, args.data_dir)
    elif args.command == "exit":
        predict_exit(args.config, args.ticker, args.entry_price, args.data_dir)
    elif args.command == "target":
        predict_target(args.config, args.ticker, args.target_price, args.data_dir)


if __name__ == "__main__":
    main()
