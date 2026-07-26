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

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("analyze_ticker")


def get_nearest_reliability(
    table: dict, label: str, confidence: float
) -> tuple[float | None, int]:
    """
    reliability_table.json から指定したラベルと確信度に最も近い閾値の precision と support を取得する。
    """
    if label not in table:
        return None, 0
    
    label_data = table[label]
    thresholds = []
    for k in label_data.keys():
        try:
            thresholds.append(float(k))
        except ValueError:
            continue
            
    if not thresholds:
        return None, 0
        
    nearest_threshold = min(thresholds, key=lambda x: abs(x - confidence))
    threshold_str = f"{nearest_threshold:.2f}"
    
    if threshold_str not in label_data:
        closest_key = min(label_data.keys(), key=lambda k: abs(float(k) - confidence))
        threshold_str = closest_key
        
    data = label_data[threshold_str]
    return data.get("precision"), data.get("support", 0)


def load_feature_importance(model_dir: Path, feature_cols: list[str], models: list) -> dict[str, float]:
    """
    feature_importance.csv をロードする。存在しない場合は、アンサンブルモデルから動的に算出する。
    """
    csv_path = model_dir / "feature_importance.csv"
    if csv_path.exists():
        log.info("Loading feature importance from %s", csv_path)
        df = pl.read_csv(csv_path)
        return {row["feature"]: float(row["importance"]) for row in df.iter_rows(named=True)}
        
    log.warning("feature_importance.csv not found. Calculating from model objects.")
    importances = np.zeros(len(feature_cols))
    valid_models = 0
    for m in models:
        # LightGBM や XGBoost などの feature_importances_ 属性を確認
        if hasattr(m, "feature_importances_"):
            importances += m.feature_importances_
            valid_models += 1
            
    if valid_models > 0:
        importances /= valid_models
        return {col: float(imp) for col, imp in zip(feature_cols, importances)}
        
    # フォールバックとしてすべて 0.0 を返す
    return {col: 0.0 for col in feature_cols}


def analyze_ticker(
    ticker: str,
    entry_price: float | None,
    model_dir: Path,
    reliability_path: Path,
    config_path: Path,
    data_dir: Path | None,
    out_json_path: Path,
) -> None:
    # 1. モデルとメタデータ、校正テーブルのロード
    log.info("Loading models from %s", model_dir)
    models, metadata = load_ensemble_models(model_dir)
    feature_cols = metadata["feature_columns"]
    classes = metadata["label_classes"]
    barrier_config = metadata.get("barrier_config", {})
    
    log.info("Loading reliability table from %s", reliability_path)
    if reliability_path.exists():
        with open(reliability_path, "r", encoding="utf-8") as f:
            reliability_table = json.load(f)
    else:
        log.warning("Reliability table not found.")
        reliability_table = {}
        
    config = load_config(config_path)
    
    # 2. 最新特徴量の取得
    log.info("Fetching latest features for ticker %s", ticker)
    latest_df = get_latest_features(config, ticker, data_dir, feature_cols)
    latest_date_raw = latest_df["date"][0]
    latest_date = latest_date_raw.isoformat() if hasattr(latest_date_raw, "isoformat") else str(latest_date_raw)
    latest_close = float(latest_df["close"][0])
    
    # 3. 推論
    X = latest_df.select(feature_cols).to_pandas()
    probas = [m.predict_proba(X) for m in models]
    avg_proba = np.mean(probas, axis=0)[0]
    
    max_idx = np.argmax(avg_proba)
    predicted_label = classes[max_idx]
    confidence = float(avg_proba[max_idx])
    
    # 全クラスの予測確率マップ
    proba_map = {cls: float(p) for cls, p in zip(classes, avg_proba)}
    
    # 過去精度の取得
    precision, support = get_nearest_reliability(reliability_table, predicted_label, confidence)
    
    # 4. 含み損益状況の計算
    unrealized_return = None
    if entry_price is not None:
        unrealized_return = (latest_close - entry_price) / entry_price
        
    # 5. 特徴量生値と重要度の抽出
    importances = load_feature_importance(model_dir, feature_cols, models)
    
    feature_details = []
    for col in feature_cols:
        val = latest_df[col][0]
        # float の場合は Python float に変換、polars オブジェクトをシリアライズ可能にするため
        if hasattr(val, "item"):
            val = val.item()
        
        feature_details.append({
            "feature": col,
            "value": val,
            "global_importance": importances.get(col, 0.0)
        })
        
    # 重要度順にソート
    feature_details = sorted(feature_details, key=lambda x: x["global_importance"], reverse=True)
    
    # 6. JSON 構造化データの作成
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
    
    # タイムスタンプをファイル名に追加
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file_name = f"{out_json_path.stem}_{timestamp}{out_json_path.suffix}"
    actual_out_path = out_json_path.with_name(out_file_name)
    
    actual_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(actual_out_path, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
    log.info("Analysis detail JSON saved at %s", actual_out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3: Detailed Ticker Analyzer for LLM Context")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker code")
    parser.add_argument("--entry-price", type=float, default=None, help="Position entry price (optional)")
    parser.add_argument("--model-dir", type=Path, default=Path("models/task2"), help="Model directory")
    parser.add_argument("--reliability-table", type=Path, default=Path("reliability_table.json"), help="Reliability table JSON path")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"), help="Config file path")
    parser.add_argument("--data-dir", type=Path, default=None, help="Feature data directory (optional)")
    parser.add_argument("--out-json", type=Path, default=Path("reports/ticker_detail.json"), help="Output detailed JSON path")
    
    args = parser.parse_args()
    
    # out-jsonがデフォルト名でかつ ticker が指定されている場合、ファイル名に ticker を付与する
    out_path = args.out_json
    if out_path == Path("reports/ticker_detail.json"):
        out_path = Path(f"reports/{args.ticker}_detail.json")
        
    analyze_positions(
        ticker=args.ticker,
        entry_price=args.entry_price,
        model_dir=args.model_dir,
        reliability_path=args.reliability_table,
        config_path=args.config,
        data_dir=args.data_dir,
        out_json_path=out_path,
    )


# 呼び出し関数名を修正してエイリアス定義
analyze_positions = analyze_ticker

if __name__ == "__main__":
    main()
