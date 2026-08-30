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
from pipeline.rules import evaluate_position_rules

# ログ設定
logging.basicConfig(
  level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("check_positions")


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

  has_entry_date = "entry_date" in positions_df.columns

  # 2. モデルとメタデータ、校正テーブルのロード
  log.info("Loading models from %s", model_dir)
  models, metadata = load_ensemble_models(model_dir)
  feature_cols = metadata["feature_columns"]
  classes = metadata["label_classes"]

  reliability_table = load_reliability_table(reliability_path)

  config = load_config(config_path)

  # 銘柄名マップの読み込み
  names_dir = data_dir if data_dir is not None else Path(config.data.raw_dir)
  names_map = {}
  names_path = names_dir / "names.json"
  if names_path.exists():
    with open(names_path, encoding="utf-8") as f:
      names_map = json.load(f)
  else:
    log.warning("names.json not found at %s. 名称列はコードで表示します。", names_path)

  results = []

  # 3. 銘柄ごとの予測と判定
  for row in positions_df.iter_rows(named=True):
    ticker = str(row["code"])
    entry_price = float(row["entry_price"])
    entry_date = (
      str(row["entry_date"])
      if has_entry_date and row["entry_date"] is not None
      else None
    )

    try:
      # 最新特徴量の取得
      latest_df = get_latest_features(config, ticker, data_dir, feature_cols)
      latest_date = latest_df["date"][0]
      latest_close = float(latest_df["close"][0])

      # 取引ルール（RULE.md）に基づく判定
      pos_status = evaluate_position_rules(
        code=ticker,
        entry_price=entry_price,
        current_price=latest_close,
        entry_date=entry_date,
      )

      # 推論用の特徴量選択
      X = latest_df.select(feature_cols).to_pandas()

      # アンサンブル推論
      probas = [m.predict_proba(X) for m in models]
      avg_proba = np.mean(probas, axis=0)[0]

      # 予測ラベルと確信度
      max_idx = np.argmax(avg_proba)
      predicted_label = classes[max_idx]
      confidence = float(avg_proba[max_idx])

      unrealized_return = pos_status.unrealized_return

      precision, support = lookup_reliability(
        reliability_table, predicted_label, confidence
      )

      # 注意フラグ（flag）の判定
      flags = []

      # 1. ルールに基づくアラート（最優先）
      if pos_status.target_hit.stop_loss_hit:
        flags.append("🚨 損切りライン到達(-5%): 即時損切り対象")
      elif pos_status.target_hit.take_profit_hit:
        flags.append("🎯 利確ライン到達(+10%): 下降兆候監視")
      elif pos_status.time_rule_triggered:
        if unrealized_return >= 0:
          flags.append("⏱️ 10日ルール発動: 全清算検討")
        else:
          flags.append("⏱️ 10日ルール発動: 半数損切り検討")

      # 2. 予測に基づくアラート (閾値 0.65)
      if (
        predicted_label == "lower"
        and confidence >= 0.65
        and not pos_status.target_hit.stop_loss_hit
      ):
        flags.append("🚨 損切り警戒(予測下落)")
      elif (
        predicted_label == "upper"
        and confidence >= 0.65
        and not pos_status.target_hit.take_profit_hit
      ):
        flags.append("✨ 利確検討(予測上昇)")

      # 3. 予測と実損益の相反（矛盾）チェック
      if unrealized_return >= 0.08 and predicted_label == "lower":
        flags.append("⚠️ 含み益到達も予測は下落方向（反落リスクに留意）")
      elif unrealized_return <= -0.04 and predicted_label == "upper":
        flags.append("⚠️ 含み損到達も予測は上昇方向（回復の可能性）")

      # 母数警告
      support_note = ""
      if support is not None and support < 20:
        support_note = "※母数少"

      flag_str = "<br>".join(flags) if flags else "正常"

      results.append(
        {
          "code": ticker,
          "name": names_map.get(ticker, ticker),
          "entry_price": entry_price,
          "entry_date": pos_status.entry_date or "-",
          "holding_days": pos_status.holding_days
          if pos_status.holding_days is not None
          else "-",
          "latest_close": latest_close,
          "unrealized_return": unrealized_return,
          "predicted_label": predicted_label,
          "confidence": confidence,
          "precision": precision,
          "support": support,
          "support_note": support_note,
          "pos_status": pos_status,
          "flag": flag_str,
          "date": latest_date,
          "status": "success",
          "error": "",
        }
      )

    except Exception as e:
      log.exception("Error processing ticker %s", ticker)
      results.append(
        {
          "code": ticker,
          "name": names_map.get(ticker, ticker),
          "entry_price": entry_price,
          "entry_date": entry_date or "-",
          "holding_days": "-",
          "latest_close": None,
          "unrealized_return": None,
          "predicted_label": "",
          "confidence": None,
          "precision": None,
          "support": 0,
          "support_note": "",
          "pos_status": None,
          "flag": "❌エラー",
          "date": "-",
          "status": "error",
          "error": str(e),
        }
      )

  # 4. レポート生成
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  report_file_name = f"{report_path.stem}_{timestamp}{report_path.suffix}"
  actual_report_path = report_path.with_name(report_file_name)

  actual_report_path.parent.mkdir(parents=True, exist_ok=True)

  current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  # 統計
  total_positions = len(results)
  success_positions = sum(1 for r in results if r["status"] == "success")
  alert_positions = sum(
    1 for r in results if r["flag"] != "正常" and r["status"] == "success"
  )
  error_positions = sum(1 for r in results if r["status"] == "error")

  report_md = []
  report_md.append(
    f"# ポジション分析一括チェックレポート ({datetime.date.today().isoformat()})"
  )
  report_md.append(f"実行日時: {current_time}  ")
  report_md.append(
    f"総銘柄数: {total_positions} (正常取得: {success_positions}, 要注意: {alert_positions}, エラー: {error_positions})"
  )
  report_md.append("\n## 要注意銘柄 (アラート発生)")

  alerts = [r for r in results if r["flag"] != "正常"]
  if alerts:
    report_md.append(
      "| コード | 名称 | 取得単価 | 保有日数 | 最新終値 | 損益率 | 予測 | 確信度 | 過去精度 | フラグ / 推奨アクション |"
    )
    report_md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in alerts:
      if r["status"] == "success":
        ret_str = f"{r['unrealized_return'] * 100:+.2f}%"
        conf_str = f"{r['confidence'] * 100:.1f}%"
        prec_str = (
          f"{r['precision'] * 100:.1f}%"
          if r["precision"] is not None
          else "校正データなし"
        )
        supp_note = f" ({r['support_note']})" if r["support_note"] else ""
        action_note = (
          f"<br>💡 <b>推奨:</b> {r['pos_status'].recommended_rule_action}"
          if r["pos_status"]
          else ""
        )
        report_md.append(
          f"| {r['code']} | {r['name']} | {r['entry_price']:.1f} | {r['holding_days']}日 | {r['latest_close']:.1f} | {ret_str} | {r['predicted_label']} | {conf_str} | {prec_str}{supp_note} | {r['flag']}{action_note} |"
        )
      else:
        report_md.append(
          f"| {r['code']} | {r['name']} | {r['entry_price']:.1f} | - | - | - | - | - | - | {r['flag']} (エラー: {r['error']}) |"
        )
  else:
    report_md.append("現在、アラートが発生している銘柄はありません。")

  report_md.append("\n## 保有ポジション一覧")
  report_md.append(
    "| コード | 名称 | 取得単価 | 買付日 | 保有日数 | 最新終値 | 損益率 | 予測 | 確信度 | 過去精度 (母数) | フラグ | 最新データ日 |"
  )
  report_md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
  for r in results:
    if r["status"] == "success":
      ret_str = f"{r['unrealized_return'] * 100:+.2f}%"
      conf_str = f"{r['confidence'] * 100:.1f}%"
      prec_str = (
        f"{r['precision'] * 100:.1f}%"
        if r["precision"] is not None
        else "校正データなし"
      )
      supp_note = f" {r['support_note']}" if r["support_note"] else ""
      report_md.append(
        f"| {r['code']} | {r['name']} | {r['entry_price']:.1f} | {r['entry_date']} | {r['holding_days']}日 | {r['latest_close']:.1f} | {ret_str} | {r['predicted_label']} | {conf_str} | {prec_str} ({r['support']}){supp_note} | {r['flag']} | {r['date']} |"
      )
    else:
      report_md.append(
        f"| {r['code']} | {r['name']} | {r['entry_price']:.1f} | {r['entry_date']} | - | - | - | - | - | - | {r['flag']} | - |"
      )

  with open(actual_report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_md))

  log.info("Report generated at %s", actual_report_path)


def _resolve_default_path(filename: str, candidates: list[Path]) -> Path:
  for c in candidates:
    if c.exists():
      return c
  return candidates[0]


def main() -> None:
  parser = argparse.ArgumentParser(description="Task 2: Batch Position Alert Checker")

  # デフォルトパスの自動解決候補
  default_positions = _resolve_default_path(
    "positions.csv",
    [
      Path("train/positions.csv"),
      Path("positions.csv"),
      Path("../train/positions.csv"),
    ],
  )
  default_models = _resolve_default_path(
    "models/task2",
    [
      Path("train/models/task2"),
      Path("models/task2"),
      Path("../models/task2"),
    ],
  )
  default_reliability = _resolve_default_path(
    "reliability_table.json",
    [
      Path("train/reliability_table.json"),
      Path("reliability_table.json"),
      Path("../train/reliability_table.json"),
    ],
  )
  default_config = _resolve_default_path(
    "config.yaml",
    [
      Path("train/config/config.yaml"),
      Path("config/config.yaml"),
      Path("../train/config/config.yaml"),
    ],
  )
  default_report = (
    Path("reports/daily_check.md")
    if Path("reports").exists()
    else Path("train/reports/daily_check.md")
  )

  parser.add_argument(
    "--positions", type=Path, default=default_positions, help="Positions CSV path"
  )
  parser.add_argument(
    "--model-dir", type=Path, default=default_models, help="Model directory"
  )
  parser.add_argument(
    "--reliability-table",
    type=Path,
    default=default_reliability,
    help="Reliability table JSON path",
  )
  parser.add_argument(
    "--report-out",
    type=Path,
    default=default_report,
    help="Report output Markdown path",
  )
  parser.add_argument(
    "--config", type=Path, default=default_config, help="Config file path"
  )
  parser.add_argument(
    "--data-dir", type=Path, default=None, help="Feature data directory (optional)"
  )

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
