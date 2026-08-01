from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import datetime
import numpy as np
import polars as pl

from pipeline.config import load_config
from pipeline.predict import load_ensemble_models, get_latest_features
from pipeline.reliability import load_reliability_table, lookup_reliability

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("analyze_ticker")


def load_feature_importance(model_dir: Path, feature_cols: list[str], models: list) -> dict[str, float]:
    csv_path = model_dir / "feature_importance.csv"
    if csv_path.exists():
        log.info("Loading feature importance from %s", csv_path)
        df = pl.read_csv(csv_path)
        return {row["feature"]: float(row["importance"]) for row in df.iter_rows(named=True)}

    log.warning("feature_importance.csv not found. Calculating from model objects.")
    importances = np.zeros(len(feature_cols))
    valid_models = 0
    for m in models:
        if hasattr(m, "feature_importances_"):
            importances += m.feature_importances_
            valid_models += 1

    if valid_models > 0:
        importances /= valid_models
        return {col: float(imp) for col, imp in zip(feature_cols, importances)}

    return {col: 0.0 for col in feature_cols}


def _analyze_single_ticker(
    ticker: str,
    entry_price: float | None,
    models: list,
    metadata: dict,
    reliability_table: dict,
    config,
    model_dir: Path,
    data_dir: Path | None,
    out_json_path: Path,
    no_timestamp: bool = False,
) -> None:
    feature_cols = metadata["feature_columns"]
    classes = metadata["label_classes"]
    barrier_config = metadata.get("barrier_config", {})

    log.info("Fetching latest features for ticker %s", ticker)
    latest_df = get_latest_features(config, ticker, data_dir, feature_cols)
    latest_date_raw = latest_df["date"][0]
    latest_date = latest_date_raw.isoformat() if hasattr(latest_date_raw, "isoformat") else str(latest_date_raw)
    latest_close = float(latest_df["close"][0])

    X = latest_df.select(feature_cols).to_pandas()
    probas = [m.predict_proba(X) for m in models]
    avg_proba = np.mean(probas, axis=0)[0]

    max_idx = np.argmax(avg_proba)
    predicted_label = classes[max_idx]
    confidence = float(avg_proba[max_idx])

    proba_map = {cls: float(p) for cls, p in zip(classes, avg_proba)}

    precision, support = lookup_reliability(reliability_table, predicted_label, confidence)

    unrealized_return = None
    if entry_price is not None:
        unrealized_return = (latest_close - entry_price) / entry_price

    importances = load_feature_importance(model_dir, feature_cols, models)

    feature_details = []
    for col in feature_cols:
        val = latest_df[col][0]
        if hasattr(val, "item"):
            val = val.item()
        feature_details.append({
            "feature": col,
            "value": val,
            "global_importance": importances.get(col, 0.0)
        })

    feature_details = sorted(feature_details, key=lambda x: x["global_importance"], reverse=True)

    analysis_result = {
        "ticker": ticker,
        "latest_date": latest_date,
        "latest_close": latest_close,
        "entry_price": entry_price,
        "unrealized_return": unrealized_return,
        "prediction": {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": proba_map,
            "historical_precision": precision,
            "historical_support": support,
        },
        "barrier_config": barrier_config,
        "features": feature_details,
        "analyzed_at": datetime.datetime.now().isoformat()
    }

    if no_timestamp:
        actual_out_path = out_json_path
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file_name = f"{out_json_path.stem}_{timestamp}{out_json_path.suffix}"
        actual_out_path = out_json_path.with_name(out_file_name)

    actual_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(actual_out_path, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)

    log.info("Analysis detail JSON saved at %s", actual_out_path)


def analyze_ticker(
    ticker: str,
    entry_price: float | None,
    model_dir: Path,
    reliability_path: Path,
    config_path: Path,
    data_dir: Path | None,
    out_json_path: Path,
    no_timestamp: bool = False,
) -> None:
    log.info("Loading models from %s", model_dir)
    models, metadata = load_ensemble_models(model_dir)

    reliability_table = load_reliability_table(reliability_path)

    config = load_config(config_path)

    _analyze_single_ticker(
        ticker=ticker,
        entry_price=entry_price,
        models=models,
        metadata=metadata,
        reliability_table=reliability_table,
        config=config,
        model_dir=model_dir,
        data_dir=data_dir,
        out_json_path=out_json_path,
        no_timestamp=no_timestamp,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3: Detailed Ticker Analyzer for LLM Context")
    parser.add_argument("--ticker", type=str, default=None, help="Stock ticker code")
    parser.add_argument("--entry-price", type=float, default=None, help="Position entry price (optional)")
    parser.add_argument("--positions-csv", type=Path, default=None, help="Positions CSV path (batch mode)")
    parser.add_argument("--model-dir", type=Path, default=Path("models/task2"), help="Model directory")
    parser.add_argument("--reliability-table", type=Path, default=Path("reliability_table.json"), help="Reliability table JSON path")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"), help="Config file path")
    parser.add_argument("--data-dir", type=Path, default=None, help="Feature data directory (optional)")
    parser.add_argument("--out-json", type=Path, default=Path("reports/json/ticker_detail.json"), help="Output detailed JSON path")
    parser.add_argument("--no-timestamp", action="store_true", help="Skip timestamp suffix in output filename")

    args = parser.parse_args()

    if args.positions_csv:
        log.info("Batch mode: loading positions from %s", args.positions_csv)
        models, metadata = load_ensemble_models(args.model_dir)
        reliability_table = load_reliability_table(args.reliability_table)
        config = load_config(args.config)

        df = pl.read_csv(args.positions_csv)
        for row in df.iter_rows(named=True):
            ticker = str(row["code"])
            entry_price = float(row["entry_price"])
            out_path = Path(f"reports/json/{ticker}_detail.json")
            _analyze_single_ticker(
                ticker=ticker,
                entry_price=entry_price,
                models=models,
                metadata=metadata,
                reliability_table=reliability_table,
                config=config,
                model_dir=args.model_dir,
                data_dir=args.data_dir,
                out_json_path=out_path,
                no_timestamp=args.no_timestamp,
            )

    else:
        if args.ticker is None:
            parser.error("Either --ticker or --positions-csv is required")

        out_path = args.out_json
        if out_path == Path("reports/json/ticker_detail.json"):
            out_path = Path(f"reports/json/{args.ticker}_detail.json")

        analyze_ticker(
            ticker=args.ticker,
            entry_price=args.entry_price,
            model_dir=args.model_dir,
            reliability_path=args.reliability_table,
            config_path=args.config,
            data_dir=args.data_dir,
            out_json_path=out_path,
            no_timestamp=args.no_timestamp,
        )


analyze_positions = analyze_ticker

if __name__ == "__main__":
    main()