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
log = logging.getLogger("check_positions")


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
        
    # 最も近い閾値を見つける
    nearest_threshold = min(thresholds, key=lambda x: abs(x - confidence))
    threshold_str = f"{nearest_threshold:.2f}"
    
    # 万が一文字列キーが一致しない場合のためのフォールバック
    if threshold_str not in label_data:
        # キーの浮動小数点表現で最も近いものを探す
        closest_key = min(label_data.keys(), key=lambda k: abs(float(k) - confidence))
        threshold_str = closest_key
        
    data = label_data[threshold_str]
    return data.get("precision"), data.get("support", 0)


def analyze_positions(
    positions_path: Path,
    model_dir: Path,
    reliability_path: Path,
    config_path: Path,
    data_dir: Path | None,
    report_path: Path,
) -> None:
    # 1. データの読み込み
    if not positions_path.exists():
        raise FileNotFoundError(f"Positions file not found: {positions_path}")
    
    log.info("Loading positions from %s", positions_path)
    positions_df = pl.read_csv(positions_path)
    
    # 必要カラムチェック
    required_cols = ["code", "entry_price"]
    for col in required_cols:
        if col not in positions_df.columns:
            raise ValueError(f"Required column '{col}' is missing in positions file.")
            
    # 2. モデルとメタデータ、校正テーブルのロード
    log.info("Loading models from %s", model_dir)
    models, metadata = load_ensemble_models(model_dir)
    feature_cols = metadata["feature_columns"]
    classes = metadata["label_classes"]
    
    log.info("Loading reliability table from %s", reliability_path)
    if reliability_path.exists():
        with open(reliability_path, "r", encoding="utf-8") as f:
            reliability_table = json.load(f)
    else:
        log.warning("Reliability table not found. Proceeding without calibration data.")
        reliability_table = {}
        
    config = load_config(config_path)
    
    results = []
    
    # 3. 銘柄ごとの予測と判定
    for row in positions_df.iter_rows(named=True):
        ticker = str(row["code"])
        entry_price = float(row["entry_price"])
        
        try:
            # 最新特徴量の取得
            latest_df = get_latest_features(config, ticker, data_dir, feature_cols)
            latest_date = latest_df["date"][0]
            latest_close = float(latest_df["close"][0])
            
            # 推論用の特徴量選択
            X = latest_df.select(feature_cols).to_pandas()
            
            # アンサンブル推論
            probas = [m.predict_proba(X) for m in models]
            avg_proba = np.mean(probas, axis=0)[0]
            
            # 予測ラベルと確信度
            max_idx = np.argmax(avg_proba)
            predicted_label = classes[max_idx]
            confidence = float(avg_proba[max_idx])
            
            # 含み損益率の計算
            unrealized_return = (latest_close - entry_price) / entry_price
            
            # 過去精度情報の取得
            precision, support = get_nearest_reliability(reliability_table, predicted_label, confidence)
            
            # 注意フラグ（flag）の判定
            flags = []
            
            # 予測に基づくアラート
            if predicted_label == "lower" and confidence >= 0.65:
                flags.append("🚨損切り警戒")
            elif predicted_label == "upper" and confidence >= 0.65:
                flags.append("✨利確検討")
                
            # 実損益に基づくアラート
            if unrealized_return >= 0.08:
                flags.append("📈利確目安到達")
            elif unrealized_return <= -0.04:
                flags.append("📉損切り目安到達")
                
            # 母数警告
            support_note = ""
            if support is not None and support < 20:
                support_note = "※母数少"
                
            flag_str = ", ".join(flags) if flags else "正常"
            
            results.append({
                "code": ticker,
                "entry_price": entry_price,
                "latest_close": latest_close,
                "unrealized_return": unrealized_return,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "precision": precision,
                "support": support,
                "support_note": support_note,
                "flag": flag_str,
                "date": latest_date,
                "status": "success",
                "error": ""
            })
            
        except Exception as e:
            log.exception("Error processing ticker %s", ticker)
            results.append({
                "code": ticker,
                "entry_price": entry_price,
                "latest_close": None,
                "unrealized_return": None,
                "predicted_label": "",
                "confidence": None,
                "precision": None,
                "support": 0,
                "support_note": "",
                "flag": "❌エラー",
                "date": "-",
                "status": "error",
                "error": str(e)
            })
            
    # 4. レポート生成
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file_name = f"{report_path.stem}_{timestamp}{report_path.suffix}"
    actual_report_path = report_path.with_name(report_file_name)

    actual_report_path.parent.mkdir(parents=True, exist_ok=True)
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 統計
    total_positions = len(results)
    success_positions = sum(1 for r in results if r["status"] == "success")
    alert_positions = sum(1 for r in results if r["flag"] != "正常" and r["status"] == "success")
    error_positions = sum(1 for r in results if r["status"] == "error")
    
    report_md = []
    report_md.append(f"# ポジション分析一括チェックレポート ({datetime.date.today().isoformat()})")
    report_md.append(f"実行日時: {current_time}  ")
    report_md.append(f"総銘柄数: {total_positions} (正常取得: {success_positions}, 要注意: {alert_positions}, エラー: {error_positions})")
    report_md.append("\n## 要注意銘柄 (アラート発生)")
    
    alerts = [r for r in results if r["flag"] != "正常"]
    if alerts:
        report_md.append("| コード | 取得単価 | 最新終値 | 損益率 | 予測 | 確信度 | 過去精度 | フラグ |")
        report_md.append("|---|---|---|---|---|---|---|---|")
        for r in alerts:
            if r["status"] == "success":
                ret_str = f"{r['unrealized_return']*100:+.2f}%"
                conf_str = f"{r['confidence']*100:.1f}%"
                prec_str = f"{r['precision']*100:.1f}%" if r["precision"] is not None else "-"
                supp_note = f" ({r['support_note']})" if r["support_note"] else ""
                report_md.append(f"| {r['code']} | {r['entry_price']:.1f} | {r['latest_close']:.1f} | {ret_str} | {r['predicted_label']} | {conf_str} | {prec_str}{supp_note} | {r['flag']} |")
            else:
                report_md.append(f"| {r['code']} | {r['entry_price']:.1f} | - | - | - | - | - | {r['flag']} (エラー: {r['error']}) |")
    else:
        report_md.append("現在、アラートが発生している銘柄はありません。")
        
    report_md.append("\n## 保有ポジション一覧")
    report_md.append("| コード | 取得単価 | 最新終値 | 損益率 | 予測 | 確信度 | 過去精度 (母数) | フラグ | 最新データ日 |")
    report_md.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if r["status"] == "success":
            ret_str = f"{r['unrealized_return']*100:+.2f}%"
            conf_str = f"{r['confidence']*100:.1f}%"
            prec_str = f"{r['precision']*100:.1f}%" if r["precision"] is not None else "-"
            supp_note = f" {r['support_note']}" if r["support_note"] else ""
            report_md.append(f"| {r['code']} | {r['entry_price']:.1f} | {r['latest_close']:.1f} | {ret_str} | {r['predicted_label']} | {conf_str} | {prec_str} ({r['support']}){supp_note} | {r['flag']} | {r['date']} |")
        else:
            report_md.append(f"| {r['code']} | {r['entry_price']:.1f} | - | - | - | - | - | {r['flag']} | - |")
            
    with open(actual_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_md))
        
    log.info("Report generated at %s", actual_report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2: Batch Position Alert Checker")
    parser.add_argument("--positions", type=Path, default=Path("positions.csv"), help="Positions CSV path")
    parser.add_argument("--model-dir", type=Path, default=Path("models/task2"), help="Model directory")
    parser.add_argument("--reliability-table", type=Path, default=Path("reliability_table.json"), help="Reliability table JSON path")
    parser.add_argument("--report-out", type=Path, default=Path("reports/daily_check.md"), help="Report output Markdown path")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"), help="Config file path")
    parser.add_argument("--data-dir", type=Path, default=None, help="Feature data directory (optional)")
    
    args = parser.parse_args()
    
    analyze_positions(
        positions_path=args.positions,
        model_dir=args.model_dir,
        reliability_path=args.reliability_table,
        config_path=args.config,
        data_dir=args.data_dir,
        report_path=args.report_out,
    )


if __name__ == "__main__":
    main()
