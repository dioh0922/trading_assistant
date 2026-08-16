"""
evaluate_holdout.py（v2: フル実装パイプライン対応版）
──────────────────────────────────────────────
実装計画のフェーズ5（dataset_split.py）が出力する「特徴量・ラベル・split_type列を
持つ結合済みデータセット」と、フェーズ6（train.py）が出力する「fold別モデル群 +
metadata.json」を読み込み、銘柄軸ホールドアウト（stock_holdout）に対して

  1. 基本指標（マクロF1・クラス別precision/recall・混同行列・ベースライン比較）
  2. 確信度フィルタ（confidence threshold sweep）

を評価する。学習・特徴量計算は一切行わない（フェーズ5の出力をそのまま使う）。

前提とするデータセットのスキーマ（合わない場合は --label-col 等で調整可能）:
    - ticker, date 列
    - metadata.json の feature_columns に列挙された特徴量列
    - label列（デフォルト列名 "label"）
    - split列（デフォルト列名 "split_type"、値の例: train/valid/time_holdout/stock_holdout）

前提とするモデルディレクトリ:
    - metadata.json（feature_columns, label_classes, barrier_config を含む）
    - fold*.joblib（例: fold0.joblib, fold1.joblib, ...）が1つ以上

使い方:
    python evaluate_holdout.py \
        --model-dir models/task2 \
        --dataset-path data/datasets/task2_dataset.parquet \
        --split-col split_type --split-value stock_holdout \
        --thresholds 0.5,0.6,0.7,0.8,0.9 \
        --report-out reports/holdout_eval.md
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
  accuracy_score,
  classification_report,
  confusion_matrix,
  f1_score,
)

import sys

try:
  sys.stdout.reconfigure(encoding="utf-8")
  sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
  pass

logging.basicConfig(
  level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("evaluate_holdout")


# ──────────────────────────────────────────────────────────────
# モデル・メタデータの読み込み（fold群のアンサンブル）
# ──────────────────────────────────────────────────────────────
def load_metadata(model_dir: Path) -> dict:
  metadata_path = model_dir / "metadata.json"
  if not metadata_path.exists():
    raise FileNotFoundError(f"{metadata_path} が見つかりません。")
  with open(metadata_path, "r", encoding="utf-8") as f:
    return json.load(f)


def load_fold_models(model_dir: Path) -> list:
  fold_paths = sorted(model_dir.glob("fold*.joblib"))
  if not fold_paths:
    # fold*.joblibが無い場合はmodel.joblib（quick_eval.py形式）にフォールバック
    single = model_dir / "model.joblib"
    if single.exists():
      log.info("fold*.joblibが見つからないため、単一モデル %s を使用します", single)
      return [joblib.load(single)]
    raise FileNotFoundError(
      f"{model_dir} に fold*.joblib も model.joblib も見つかりません。"
    )

  log.info("found %d fold models: %s", len(fold_paths), [p.name for p in fold_paths])
  return [joblib.load(p) for p in fold_paths]


def ensemble_predict_proba(
  models: list, X: pd.DataFrame, label_classes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
  """
  全foldモデルのpredict_probaを平均する。クラス順序はfold間で揃っている前提。

  モデルが整数エンコードされたラベル（例: 0, 1, 2）で学習されている場合、
  metadata.jsonのlabel_classesを使って文字列ラベル（'lower','timeout','upper'）に
  変換する（label_classes[i] が整数コード i に対応している前提）。
  """
  raw_classes = models[0].classes_
  for m in models[1:]:
    if list(m.classes_) != list(raw_classes):
      raise ValueError(
        f"foldモデル間でクラス順序が一致しません: {list(raw_classes)} vs {list(m.classes_)}"
      )

  proba_sum = np.zeros((len(X), len(raw_classes)))
  for m in models:
    proba_sum += m.predict_proba(X)
  proba = proba_sum / len(models)

  if np.issubdtype(np.array(raw_classes).dtype, np.number):
    # 整数エンコードされている場合、label_classesを使って文字列に変換する
    try:
      classes = np.array([label_classes[int(c)] for c in raw_classes])
    except (IndexError, ValueError) as e:
      raise ValueError(
        f"model.classes_ ({list(raw_classes)}) を "
        f"metadata.jsonのlabel_classes ({label_classes}) のインデックスとして解釈できませんでした。"
        f" ラベルエンコードの対応関係を確認してください。"
      ) from e
    log.info(
      "整数エンコードされたラベルを文字列に変換しました: %s -> %s",
      list(raw_classes),
      list(classes),
    )
  else:
    classes = np.array(raw_classes)

  return proba, classes


def get_barrier_config(metadata: dict) -> dict:
  """quick_eval.py形式（upper/lower/max_days）と本実装形式（upper_barrier/lower_barrier/
  time_barrier_days）の両方に対応する。"""
  raw = metadata.get("barrier_config", {})
  if "upper" in raw:
    return {"upper": raw["upper"], "lower": raw["lower"], "max_days": raw["max_days"]}
  if "upper_barrier" in raw:
    return {
      "upper": raw["upper_barrier"],
      "lower": raw["lower_barrier"],
      "max_days": raw["time_barrier_days"],
    }
  raise KeyError(
    f"barrier_configの中身が想定外です: {raw}。"
    " upper/lower/max_days か upper_barrier/lower_barrier/time_barrier_days のいずれかを想定しています。"
  )


# ──────────────────────────────────────────────────────────────
# データセットの読み込み・フィルタ
# ──────────────────────────────────────────────────────────────
def load_holdout_dataset(
  dataset_path: Path,
  split_col: str,
  split_value: str,
  feature_columns: list[str],
  label_col: str,
) -> pd.DataFrame:
  if dataset_path.suffix == ".parquet":
    df = pd.read_parquet(dataset_path)
  else:
    df = pd.read_csv(dataset_path)

  if split_col not in df.columns:
    raise KeyError(
      f"'{split_col}' 列がデータセットにありません。実際の列: {list(df.columns)}\n"
      f"--split-col オプションで正しい列名を指定してください。"
    )
  if label_col not in df.columns:
    raise KeyError(
      f"'{label_col}' 列がデータセットにありません。実際の列: {list(df.columns)}\n"
      f"--label-col オプションで正しい列名を指定してください。"
    )

  missing_features = [c for c in feature_columns if c not in df.columns]
  if missing_features:
    raise KeyError(
      f"metadata.jsonのfeature_columnsのうち、データセットに存在しない列があります: {missing_features}\n"
      f"モデル学習時と評価時でデータセットのバージョンが違う可能性があります。"
    )

  holdout = df[df[split_col] == split_value].copy()
  if holdout.empty:
    available = df[split_col].unique().tolist()
    raise ValueError(
      f"'{split_col}' == '{split_value}' の行が0件でした。存在する値: {available}"
    )

  holdout = holdout.dropna(subset=feature_columns + [label_col])
  n_tickers = holdout["ticker"].nunique() if "ticker" in holdout.columns else None
  log.info(
    "holdout rows: %d%s",
    len(holdout),
    f", tickers: {n_tickers}" if n_tickers is not None else "",
  )
  log.info("label distribution:\n%s", holdout[label_col].value_counts())
  return holdout


# ──────────────────────────────────────────────────────────────
# 1. 基本指標
# ──────────────────────────────────────────────────────────────
def report_basic_metrics(
  y_true: pd.Series, y_pred: np.ndarray, lines: list[str]
) -> None:
  lines.append("## 1. 基本指標（銘柄軸ホールドアウト全体）")
  lines.append("")
  lines.append(f"- サンプル数: {len(y_true)}")
  lines.append("")

  lines.append("### モデル（fold アンサンブル）")
  lines.append("```")
  lines.append(classification_report(y_true, y_pred, zero_division=0))
  lines.append("```")

  majority = y_true.value_counts().idxmax()
  majority_pred = np.full(len(y_true), majority)

  lines.append(
    f"### ベースライン：多数派クラス予測（ホールドアウト内の多数派 = {majority}）"
  )
  lines.append("```")
  lines.append(classification_report(y_true, majority_pred, zero_division=0))
  lines.append("```")

  labels_order = sorted(y_true.unique())
  cm = confusion_matrix(y_true, y_pred, labels=labels_order)
  lines.append("### 混同行列（行=正解ラベル, 列=予測ラベル）")
  lines.append("```")
  header = "          " + "".join(f"{lbl:>10s}" for lbl in labels_order)
  lines.append(header)
  for lbl, row in zip(labels_order, cm):
    lines.append(f"{lbl:>10s}" + "".join(f"{v:10d}" for v in row))
  lines.append("```")

  model_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
  majority_f1 = f1_score(y_true, majority_pred, average="macro", zero_division=0)

  lines.append("### マクロF1サマリー")
  lines.append("")
  lines.append(f"- モデル: {model_f1:.3f}")
  lines.append(f"- 多数派ベースライン: {majority_f1:.3f}")
  lines.append("")

  per_class_recall = {}
  for lbl in labels_order:
    mask = (y_true == lbl).to_numpy()
    if mask.sum() > 0:
      per_class_recall[lbl] = float((y_pred[mask] == lbl).mean())
  low_recall = [lbl for lbl, r in per_class_recall.items() if r < 0.05]
  if low_recall:
    lines.append(
      f"⚠ 警告: 以下のクラスはrecallが極端に低く(<5%)、実用に耐えません: {low_recall}"
    )
  else:
    lines.append(
      "クラス別recallは全クラスで5%を上回っています（最低限のチェックはクリア）。"
    )
  lines.append("")


# ──────────────────────────────────────────────────────────────
# 2. 確信度フィルタ
# ──────────────────────────────────────────────────────────────
def report_confidence_sweep(
  y_true: pd.Series,
  proba: np.ndarray,
  classes: np.ndarray,
  thresholds: list[float],
  lines: list[str],
  upper_label: str = "upper",
) -> pd.DataFrame:
  y_true_arr = y_true.to_numpy()
  pred_idx = np.argmax(proba, axis=1)
  pred_label = classes[pred_idx]
  confidence = proba[np.arange(len(proba)), pred_idx]

  label_order = sorted(classes.tolist())

  rows = []
  for t in thresholds:
    mask = confidence >= t
    coverage = float(mask.mean())
    n = int(mask.sum())

    row = {"threshold": t, "coverage": coverage, "n_samples": n}

    if n == 0:
      row["accuracy"] = np.nan
      row["macro_f1"] = np.nan
      for lbl in label_order:
        row[f"n_pred_{lbl}"] = 0
        row[f"precision_{lbl}"] = np.nan
      rows.append(row)
      continue

    acc = accuracy_score(y_true_arr[mask], pred_label[mask])
    macro_f1 = f1_score(
      y_true_arr[mask], pred_label[mask], average="macro", zero_division=0
    )
    row["accuracy"] = acc
    row["macro_f1"] = macro_f1

    # クラスごとに「そのクラスと予測した件数」と「その予測の的中率(precision)」を出す。
    # これがないと、例えば upper_precision=1.000 の母数が10件なのか10万件なのか判断できない。
    for lbl in label_order:
      pred_lbl_mask = mask & (pred_label == lbl)
      n_pred = int(pred_lbl_mask.sum())
      row[f"n_pred_{lbl}"] = n_pred
      row[f"precision_{lbl}"] = (
        float((y_true_arr[pred_lbl_mask] == lbl).mean()) if n_pred > 0 else np.nan
      )

    rows.append(row)

  sweep_df = pd.DataFrame(rows)

  lines.append("## 2. 確信度フィルタ（confidence threshold sweep）")
  lines.append("")
  lines.append("### 2a. 全体サマリー")
  lines.append("")
  lines.append("| threshold | coverage(残存率) | n_samples | accuracy | macro_f1 |")
  lines.append("|---|---|---|---|---|")
  for _, r in sweep_df.iterrows():
    lines.append(
      f"| {r['threshold']:.2f} | {r['coverage'] * 100:.1f}% | {int(r['n_samples'])} | "
      f"{r['accuracy']:.3f} | {r['macro_f1']:.3f} |"
    )
  lines.append("")

  lines.append("### 2b. クラス別の予測件数とprecision（母数の確認用）")
  lines.append("")
  lines.append(
    "**重要**: `precision_upper=1.000`のような数字が出ても、`n_pred_upper`（その閾値でupperと"
  )
  lines.append(
    "予測された件数）が小さければ統計的に意味を持たない可能性が高い。必ず件数とセットで見ること。"
  )
  lines.append("")
  header_cols = (
    ["threshold"]
    + [f"n_pred_{lbl}" for lbl in label_order]
    + [f"precision_{lbl}" for lbl in label_order]
  )
  lines.append("| " + " | ".join(header_cols) + " |")
  lines.append("|" + "---|" * len(header_cols))
  for _, r in sweep_df.iterrows():
    cells = [f"{r['threshold']:.2f}"]
    for lbl in label_order:
      cells.append(str(int(r[f"n_pred_{lbl}"])))
    for lbl in label_order:
      p = r[f"precision_{lbl}"]
      cells.append("-" if pd.isna(p) else f"{p:.3f}")
    lines.append("| " + " | ".join(cells) + " |")
  lines.append("")

  valid = sweep_df.dropna(subset=["accuracy"])
  if len(valid) >= 2:
    if valid["accuracy"].is_monotonic_increasing:
      lines.append(
        "→ 閾値を上げるほどaccuracyが単調に改善しています。確信度は信頼できる指標として機能している可能性が高いです。"
      )
    else:
      lines.append(
        "→ 閾値を上げてもaccuracyが単調には改善していません。キャリブレーションの確認を推奨します。"
      )
  lines.append("")

  # upper（利確）予測の件数が閾値を上げるにつれてどれだけ減るかを明示的に警告する
  if f"n_pred_{upper_label}" in sweep_df.columns:
    max_n_upper = sweep_df[f"n_pred_{upper_label}"].max()
    high_threshold_rows = sweep_df[sweep_df["threshold"] >= 0.8]
    if not high_threshold_rows.empty:
      min_n_upper_high = high_threshold_rows[f"n_pred_{upper_label}"].min()
      if max_n_upper > 0 and min_n_upper_high < max_n_upper * 0.05:
        lines.append(
          f"⚠ 注意: 閾値0.8以上では『{upper_label}』と予測される件数が"
          f"最大{int(max_n_upper)}件から{int(min_n_upper_high)}件まで減っています。"
          f"precisionの高さだけを見て『高確信度の利確シグナルは信頼できる』と判断せず、"
          f"実運用でその件数が意味のある頻度で発生するか（銘柄軸で見ても偏っていないか）を確認すること。"
        )
        lines.append("")

  return sweep_df


# ──────────────────────────────────────────────────────────────
# メイン処理
# ──────────────────────────────────────────────────────────────
def run(
  model_dir: Path,
  dataset_path: Path,
  split_col: str,
  split_value: str,
  label_col: str,
  thresholds: list[float],
  report_out: Path | None,
  table_out: Path | None = None,
) -> None:
  metadata = load_metadata(model_dir)
  feature_columns = metadata["feature_columns"]
  barrier_config = get_barrier_config(metadata)

  holdout = load_holdout_dataset(
    dataset_path, split_col, split_value, feature_columns, label_col
  )

  models = load_fold_models(model_dir)
  X = holdout[feature_columns]
  y_true = holdout[label_col]

  proba, classes = ensemble_predict_proba(models, X, metadata["label_classes"])
  y_pred = classes[np.argmax(proba, axis=1)]

  lines: list[str] = []
  lines.append("# 銘柄軸ホールドアウト評価レポート")
  lines.append("")
  lines.append(f"- モデル: {model_dir}（fold数: {len(models)}）")
  lines.append(f"- モデル学習日時: {metadata.get('trained_at')}")
  lines.append(
    f"- バリア設定: upper={barrier_config['upper']}, lower={barrier_config['lower']}, "
    f"max_days={barrier_config['max_days']}"
  )
  lines.append(f"- データセット: {dataset_path}（{split_col}=={split_value}）")
  lines.append("")

  report_basic_metrics(y_true, y_pred, lines)
  sweep_df = report_confidence_sweep(y_true, proba, classes, thresholds, lines)

  report_text = "\n".join(lines)
  print(report_text)

  if report_out is not None:
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(report_text, encoding="utf-8")
    log.info("report written to %s", report_out)

  if table_out is not None:
    table_out.parent.mkdir(parents=True, exist_ok=True)
    table = {}
    label_order = sorted(classes.tolist())
    for lbl in label_order:
      table[lbl] = {}
      for _, r in sweep_df.iterrows():
        th_str = f"{r['threshold']:.2f}"
        p = r[f"precision_{lbl}"]
        support = int(r[f"n_pred_{lbl}"])
        table[lbl][th_str] = {
          "precision": float(p) if not pd.isna(p) else None,
          "support": support,
        }
    with open(table_out, "w", encoding="utf-8") as f:
      json.dump(table, f, indent=2, ensure_ascii=False)
    log.info("reliability table written to %s", table_out)


def main() -> None:
  parser = argparse.ArgumentParser(
    description="銘柄軸ホールドアウトに対する基本指標・確信度フィルタ評価（フル実装パイプライン対応版）"
  )
  parser.add_argument(
    "--model-dir",
    type=Path,
    required=True,
    help="fold*.joblib + metadata.jsonがあるディレクトリ",
  )
  parser.add_argument(
    "--dataset-path",
    type=Path,
    required=True,
    help="特徴量・ラベル・split列を持つ結合済みデータセット（parquetまたはcsv）",
  )
  parser.add_argument(
    "--split-col", type=str, default="split_type", help="split種別が入っている列名"
  )
  parser.add_argument(
    "--split-value",
    type=str,
    default="stock_holdout",
    help="評価対象とするsplitの値（例: stock_holdout, time_holdout）",
  )
  parser.add_argument("--label-col", type=str, default="label", help="正解ラベルの列名")
  parser.add_argument("--thresholds", type=str, default="0.5,0.6,0.7,0.8,0.9")
  parser.add_argument("--report-out", type=Path, default=None)
  parser.add_argument(
    "--table-out", type=Path, default=None, help="校正テーブルJSONの出力先パス"
  )
  args = parser.parse_args()

  thresholds = [float(t) for t in args.thresholds.split(",")]

  run(
    model_dir=args.model_dir,
    dataset_path=args.dataset_path,
    split_col=args.split_col,
    split_value=args.split_value,
    label_col=args.label_col,
    thresholds=thresholds,
    report_out=args.report_out,
    table_out=args.table_out,
  )


if __name__ == "__main__":
  main()
