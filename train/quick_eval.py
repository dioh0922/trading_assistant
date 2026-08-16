"""
quick_eval.py
─────────────
実装計画（implementation_plan.md）のフェーズ1〜7を、少数銘柄・単純な分割で
最短ルートで一周させるための中間チェックスクリプト。

目的:
  - パイプラインが機能しているか（バグの有無）を早期に確認する
  - タスク2（利確/損切り判定）について、モデルがベースラインより
    優れているかどうかを確認する
  - 学習済みモデルをファイルに保存し、以後は再学習せずに使い回せるようにする

このスクリプトは "本番評価" ではない。Purged K-Fold・銘柄軸ホールドアウト・
クロスセクション特徴量は含まない簡易版。本番評価は implementation_plan.md
のフェーズ5〜7（dataset_split.py / train.py / evaluate.py）で行うこと。

使い方:
    # 1. 学習してモデルをファイルに保存する（一度だけ実行すればよい）
    python quick_eval.py train --data-dir data_raw --model-dir models/task2 \
        --upper 0.10 --lower -0.05 --max-days 60

    # 2. 保存済みモデルを読み込んで、指定銘柄を予測する（再学習しない）
    python quick_eval.py predict --model-dir models/task2 --data-dir data_raw --ticker 285A
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

logging.basicConfig(
  level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("quick_eval")

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"


# ──────────────────────────────────────────────────────────────
# フェーズ1相当: CSV読み込み・フォーマット統一
# ──────────────────────────────────────────────────────────────
def parse_single_csv(csv_path: Path) -> pd.DataFrame:
  """
  yfinance形式のCSV（1行目=列名, 2行目=Ticker, 3行目=Date見出し, 4行目以降=データ）
  を読み込み、[date, ticker, open, high, low, close, volume] に統一する。
  """
  raw = pd.read_csv(csv_path, header=0)
  # 1行目のヘッダーは ["Price", "Close", "High", "Low", "Open", "Volume"] 想定
  # (yfinance特有の仕様で、実質1列目は日付インデックス)
  raw = raw.rename(columns={raw.columns[0]: "date"})
  # 2, 3行目 (Ticker行, Date見出し行) を除去
  raw = raw[~raw["date"].isin(["Ticker", "Date"])].copy()

  df = pd.DataFrame(
    {
      "date": pd.to_datetime(raw["date"]),
      "close": pd.to_numeric(raw["Close"], errors="coerce"),
      "high": pd.to_numeric(raw["High"], errors="coerce"),
      "low": pd.to_numeric(raw["Low"], errors="coerce"),
      "open": pd.to_numeric(raw["Open"], errors="coerce"),
      "volume": pd.to_numeric(raw["Volume"], errors="coerce"),
    }
  )
  df["ticker"] = csv_path.stem
  df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
  return df


def ingest_all(data_dir: Path, sample_tickers: int | None = None) -> pd.DataFrame:
  csv_paths = sorted(data_dir.glob("*.csv"))
  if sample_tickers is not None:
    csv_paths = csv_paths[:sample_tickers]
  if not csv_paths:
    raise FileNotFoundError(f"No CSV files found in {data_dir}")

  frames = [parse_single_csv(p) for p in csv_paths]
  prices = pd.concat(frames, ignore_index=True)
  log.info("ingest: %d tickers, %d rows", prices["ticker"].nunique(), len(prices))
  return prices


# ──────────────────────────────────────────────────────────────
# フェーズ3a相当: 銘柄単体特徴量（簡易版。クロスセクション特徴量は含まない）
# ──────────────────────────────────────────────────────────────
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
  delta = close.diff()
  gain = delta.clip(lower=0)
  loss = -delta.clip(upper=0)
  avg_gain = gain.rolling(period).mean()
  avg_loss = loss.rolling(period).mean()
  rs = avg_gain / avg_loss.replace(0, np.nan)
  return 100 - (100 / (1 + rs))


def compute_atr(
  high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
  prev_close = close.shift(1)
  tr = pd.concat(
    [
      (high - low),
      (high - prev_close).abs(),
      (low - prev_close).abs(),
    ],
    axis=1,
  ).max(axis=1)
  return tr.rolling(period).mean()


def build_features_single_ticker(df: pd.DataFrame) -> pd.DataFrame:
  """1銘柄分の時系列(date昇順)から特徴量を作る。未来の値は一切参照しない。"""
  out = df.copy()
  for w in (5, 20, 60):
    out[f"return_{w}d"] = out["close"].pct_change(w)
    out[f"ma_dev_{w}"] = out["close"] / out["close"].rolling(w).mean() - 1
    out[f"volatility_{w}d"] = (
      np.log(out["close"] / out["close"].shift(1)).rolling(w).std()
    )
  out["rsi_14"] = compute_rsi(out["close"], 14)
  out["atr_14"] = compute_atr(out["high"], out["low"], out["close"], 14)
  out["volume_ratio_20d"] = out["volume"] / out["volume"].rolling(20).mean()
  out["range_position_60d"] = (out["close"] - out["low"].rolling(60).min()) / (
    out["high"].rolling(60).max() - out["low"].rolling(60).min()
  )
  return out


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
  parts = []
  for ticker, group in prices.groupby("ticker"):
    feat = build_features_single_ticker(group)
    feat["ticker"] = ticker
    parts.append(feat)
  return pd.concat(parts, ignore_index=True)


# ──────────────────────────────────────────────────────────────
# フェーズ4相当: トリプルバリアラベル生成（タスク2: 利確/損切り）
# ──────────────────────────────────────────────────────────────
def triple_barrier_label_single_ticker(
  df: pd.DataFrame, upper: float, lower: float, max_days: int
) -> pd.DataFrame:
  """
  各起点日 t について、t+1日目以降で
    - 終値が (1+upper) 倍以上になった  → "upper" (利確)
    - 終値が (1+lower) 倍以下になった  → "lower" (損切り)
    - max_days以内にどちらにも到達しない → "timeout"
  を判定する。単純な終値ベースの実装（高値・安値の日中到達までは見ない簡易版）。
  """
  close = df["close"].to_numpy()
  n = len(close)

  labels = np.full(n, "", dtype=object)
  days_to_hit = np.full(n, np.nan)

  for i in range(n):
    entry_price = close[i]
    upper_price = entry_price * (1 + upper)
    lower_price = entry_price * (1 + lower)
    end = min(i + max_days, n - 1)
    if i + 1 > end:
      labels[i] = "timeout"
      continue

    window = close[i + 1 : end + 1]
    hit_upper = np.where(window >= upper_price)[0]
    hit_lower = np.where(window <= lower_price)[0]

    first_upper = hit_upper[0] if len(hit_upper) else np.inf
    first_lower = hit_lower[0] if len(hit_lower) else np.inf

    if first_upper == np.inf and first_lower == np.inf:
      labels[i] = "timeout"
    elif first_upper <= first_lower:
      labels[i] = "upper"
      days_to_hit[i] = first_upper + 1
    else:
      labels[i] = "lower"
      days_to_hit[i] = first_lower + 1

  result = df[["date"]].copy()
  result["label"] = labels
  result["days_to_hit"] = days_to_hit
  return result


def build_labels(
  prices: pd.DataFrame, upper: float, lower: float, max_days: int
) -> pd.DataFrame:
  parts = []
  for ticker, group in prices.groupby("ticker"):
    lbl = triple_barrier_label_single_ticker(group, upper, lower, max_days)
    lbl["ticker"] = ticker
    parts.append(lbl)
  return pd.concat(parts, ignore_index=True)


# ──────────────────────────────────────────────────────────────
# フェーズ5相当: 簡易時系列分割（purge/embargoなしの最短版）
# ──────────────────────────────────────────────────────────────
def simple_time_split(dataset: pd.DataFrame, train_ratio: float = 0.7) -> pd.DataFrame:
  """
  銘柄ごとに時系列順で train_ratio 分をtrain、残りをtestにする最も簡易な分割。
  本番評価では dataset_split.py の Purged K-Fold + 銘柄軸ホールドアウトを使うこと。
  """
  dataset = dataset.sort_values(["ticker", "date"])
  parts = []
  for ticker, group in dataset.groupby("ticker"):
    cut = int(len(group) * train_ratio)
    group = group.copy()
    group["split"] = ["train"] * cut + ["test"] * (len(group) - cut)
    parts.append(group)
  return pd.concat(parts, ignore_index=True)


# ──────────────────────────────────────────────────────────────
# ベースライン
# ──────────────────────────────────────────────────────────────
def majority_baseline(y_train: pd.Series, y_test: pd.Series) -> np.ndarray:
  majority = y_train.value_counts().idxmax()
  return np.full(len(y_test), majority)


def rule_based_baseline(test_df: pd.DataFrame) -> np.ndarray:
  """
  単純ルール: 直近20日リターンがプラスなら "upper"、マイナスなら "lower" と予測する。
  """
  pred = np.where(test_df["return_20d"] >= 0, "upper", "lower")
  return pred


# ──────────────────────────────────────────────────────────────
# モデルの永続化（保存・読み込み）
# ──────────────────────────────────────────────────────────────
def save_model_artifact(
  model: LGBMClassifier,
  model_dir: Path,
  feature_columns: list[str],
  barrier_config: dict,
  train_info: dict,
) -> None:
  """
  学習済みモデルと、予測時に必要なメタデータ（特徴量の並び順・ラベル設定など）を
  1つのディレクトリにまとめて保存する。predict時はこのディレクトリを指定するだけでよい。
  """
  model_dir.mkdir(parents=True, exist_ok=True)

  joblib.dump(model, model_dir / MODEL_FILENAME)

  metadata = {
    "task": "task2_exit_decision",
    "feature_columns": feature_columns,
    "label_classes": list(model.classes_),
    "barrier_config": barrier_config,
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "train_info": train_info,
  }
  with open(model_dir / METADATA_FILENAME, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

  log.info("model saved to %s (%s, %s)", model_dir, MODEL_FILENAME, METADATA_FILENAME)


def load_model_artifact(model_dir: Path) -> tuple[LGBMClassifier, dict]:
  model_path = model_dir / MODEL_FILENAME
  metadata_path = model_dir / METADATA_FILENAME
  if not model_path.exists() or not metadata_path.exists():
    raise FileNotFoundError(
      f"{model_dir} にモデルが見つかりません。先に `train` サブコマンドを実行してください。"
    )
  model = joblib.load(model_path)
  with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)
  return model, metadata


# ──────────────────────────────────────────────────────────────
# 保存済みモデルを使った推論（再学習しない）
# ──────────────────────────────────────────────────────────────
LABEL_JP = {
  "upper": "利確ライン到達",
  "lower": "損切りライン到達",
  "timeout": "様子見（未到達）",
}


def predict_for_ticker(
  model: LGBMClassifier, metadata: dict, data_dir: Path, ticker: str
) -> None:
  csv_path = data_dir / f"{ticker}.csv"
  if not csv_path.exists():
    raise FileNotFoundError(f"{csv_path} が見つかりません。")

  prices = parse_single_csv(csv_path)
  feats = build_features_single_ticker(prices)
  feature_columns = metadata["feature_columns"]

  latest = feats.dropna(subset=feature_columns).iloc[-1:]
  if latest.empty:
    raise ValueError(
      f"{ticker}: 特徴量計算に必要な期間分の価格データがありません（データ不足）。"
    )

  latest_date = latest["date"].iloc[0]
  latest_close = latest["close"].iloc[0]

  proba = model.predict_proba(latest[feature_columns])[0]
  classes = model.classes_

  print(f"\n銘柄: {ticker}")
  print(f"基準日: {latest_date.date()}（終値 {latest_close:.1f}円を取得価格と仮定）")
  barrier = metadata["barrier_config"]
  print(
    f"設定: 利確 +{barrier['upper'] * 100:.0f}% / 損切り {barrier['lower'] * 100:.0f}% "
    f"/ 判定期間 {barrier['max_days']}日"
  )
  print(f"モデル学習日時: {metadata['trained_at']}")
  print("-" * 50)
  for cls, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
    print(f"  {LABEL_JP.get(cls, cls):14s}: {p * 100:5.1f}%")

  best_idx = int(np.argmax(proba))
  print("-" * 50)
  print(
    f"→ 最も可能性が高いのは「{LABEL_JP.get(classes[best_idx], classes[best_idx])}」"
    f"（確率 {proba[best_idx] * 100:.1f}%）"
  )
  print(
    "\n注意: これは簡易モデルによる参考情報であり、投資判断を保証するものではありません。"
  )


FEATURE_COLUMNS = [
  "return_5d",
  "return_20d",
  "return_60d",
  "ma_dev_5",
  "ma_dev_20",
  "ma_dev_60",
  "volatility_5d",
  "volatility_20d",
  "volatility_60d",
  "rsi_14",
  "volume_ratio_20d",
  "range_position_60d",
  "atr_14",
]


def compute_sample_weights(df: pd.DataFrame, max_days: int = 60) -> pd.Series:
  """
  同一銘柄内で起点日が近い（バリアの時間窓が重なる）サンプルほど重みを下げる。
  重み = 重複サンプル数の逆数。ベクトル化済み（高速版）。
  """
  df_reset = df.reset_index(drop=True)
  weights = np.ones(len(df_reset), dtype=np.float64)
  max_days_td = np.timedelta64(max_days, "D")

  for ticker, group in df_reset.groupby("ticker"):
    if len(group) <= 1:
      continue
    dates = group["date"].to_numpy()
    idx = group.index.to_numpy()

    for j in range(len(dates)):
      entry = dates[j]
      window_end = entry + max_days_td
      overlap = int(np.sum((dates > entry - max_days_td) & (dates < window_end))) - 1
      if overlap > 0:
        weights[idx[j]] = 1.0 / (overlap + 1)

  return pd.Series(weights, index=df.index)


def run(
  data_dir: Path,
  upper: float,
  lower: float,
  max_days: int,
  sample_tickers: int | None,
  model_dir: Path,
  multi_split: bool = False,
) -> None:
  prices = ingest_all(data_dir, sample_tickers=sample_tickers)

  log.info("building features...")
  feats = build_features(prices)

  log.info(
    "building triple-barrier labels (task2: upper=%.2f, lower=%.2f, max_days=%d)...",
    upper,
    lower,
    max_days,
  )
  labels = build_labels(prices, upper=upper, lower=lower, max_days=max_days)

  dataset = feats.merge(labels, on=["ticker", "date"], how="inner")
  dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["label"])
  dataset = dataset[dataset["label"] != ""]

  if dataset["label"].nunique() < 2:
    raise ValueError(
      "ラベルの種類が1種類しかありません。データ量が少なすぎるか、"
      "バリア設定が極端な可能性があります。銘柄数・期間を増やすか、"
      "upper/lowerの値を調整してください。"
    )

  log.info(
    "dataset rows: %d, label distribution:\n%s",
    len(dataset),
    dataset["label"].value_counts(),
  )

  split_ratios = [0.6, 0.7, 0.8] if multi_split else [0.7]

  for train_ratio in split_ratios:
    if len(split_ratios) > 1:
      print(f"\n{'#' * 60}")
      print(f"# train_ratio = {train_ratio:.1f} (test_ratio = {1 - train_ratio:.1f})")
      print(f"{'#' * 60}")

    split = simple_time_split(dataset, train_ratio=train_ratio)
    train_df = split[split["split"] == "train"]
    test_df = split[split["split"] == "test"]

    if train_df.empty or test_df.empty:
      log.warning(
        "train_ratio=%.2f: train/testのどちらかが空。スキップします。", train_ratio
      )
      continue

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label"]

    sample_w = compute_sample_weights(train_df, max_days=max_days)

    log.info(
      "training LightGBM classifier (train_ratio=%.2f, class_weight='balanced', sample_weight=yes)...",
      train_ratio,
    )
    model = LGBMClassifier(
      objective="multiclass",
      n_estimators=300,
      learning_rate=0.05,
      num_leaves=31,
      verbosity=-1,
      class_weight="balanced",
    )
    model.fit(X_train, y_train, sample_weight=sample_w)
    model_pred = model.predict(X_test)

    majority_pred = majority_baseline(y_train, y_test)
    rule_pred = rule_based_baseline(test_df)

    print("\n" + "=" * 60)
    print("【モデル】LightGBM (class_weight='balanced', sample_weight適用)")
    print("=" * 60)
    print(classification_report(y_test, model_pred, zero_division=0))

    print("=" * 60)
    print("【ベースライン1】多数派クラス予測")
    print("=" * 60)
    print(classification_report(y_test, majority_pred, zero_division=0))

    print("=" * 60)
    print("【ベースライン2】ルールベース（直近20日リターンの符号で判定）")
    print("=" * 60)
    print(classification_report(y_test, rule_pred, zero_division=0))

    print("=" * 60)
    print("【混同行列】(行=正解ラベル, 列=予測ラベル)")
    print("=" * 60)
    labels_order = sorted(y_test.unique())
    cm = confusion_matrix(y_test, model_pred, labels=labels_order)
    header = "          " + "".join(f"{lbl:>10s}" for lbl in labels_order)
    print(header)
    for lbl, row in zip(labels_order, cm):
      print(f"{lbl:>10s}" + "".join(f"{v:10d}" for v in row))

    print("\n" + "=" * 60)
    print("【テスト期間の相場地合い】")
    print("=" * 60)
    test_dates = test_df["date"]
    print(f"  test期間: {test_dates.min().date()} ～ {test_dates.max().date()}")
    test_avg_ret = test_df["return_20d"].mean()
    print(
      f"  test期間中の平均20日リターン: {test_avg_ret:+.4f} ({test_avg_ret * 100:+.2f}%)"
    )
    for lbl in labels_order:
      n = (y_test == lbl).sum()
      pct = n / len(y_test) * 100
      print(f"  正解ラベル '{lbl}': {n:,}件 ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("【サマリー: マクロF1スコア比較】")
    print("=" * 60)
    model_f1 = f1_score(y_test, model_pred, average="macro", zero_division=0)
    majority_f1 = f1_score(y_test, majority_pred, average="macro", zero_division=0)
    rule_f1 = f1_score(y_test, rule_pred, average="macro", zero_division=0)
    print(f"  LightGBM       : {model_f1:.3f}")
    print(f"  多数派ベースライン: {majority_f1:.3f}")
    print(f"  ルールベース     : {rule_f1:.3f}")

    if model_f1 > max(majority_f1, rule_f1):
      print(
        "\n→ マクロF1では両ベースラインを上回っています(=パイプライン自体は機能している可能性が高い)。"
      )
    else:
      print(
        "\n→ モデルがベースラインを上回っていません。特徴量・ラベル設計・データ量を見直してください。"
      )

    per_class_recall = {}
    for lbl in labels_order:
      mask = y_test == lbl
      if mask.sum() > 0:
        per_class_recall[lbl] = float((model_pred[mask.to_numpy()] == lbl).mean())
    print("\n【クラス別recall】")
    for lbl, r in sorted(per_class_recall.items()):
      flag = " ← 要注意" if r < 0.05 else ""
      print(f"  {lbl:>10s}: {r:.3f}{flag}")

    low_recall_classes = [lbl for lbl, r in per_class_recall.items() if r < 0.05]
    if low_recall_classes:
      print(
        f"\n⚠ 警告: 以下のクラスはrecallが極端に低く(<5%)、ほぼ予測できていません: {low_recall_classes}"
      )
      print("  マクロF1がベースラインを上回っていても、この状態では実用に耐えません。")
      print("  混同行列を見て、このクラスが何に誤分類されているか確認してください。")
      print("  対処例: test期間を複数の相場地合いで検証する / 特徴量を見直す")

    print("\n注意: これは簡易チェックです（purge/embargo・銘柄軸ホールドアウト・")
    print(
      "クロスセクション特徴量なし）。本番評価は必ずフルパイプラインで行ってください。"
    )

    if not multi_split:
      save_model_artifact(
        model=model,
        model_dir=model_dir,
        feature_columns=FEATURE_COLUMNS,
        barrier_config={"upper": upper, "lower": lower, "max_days": max_days},
        train_info={
          "n_tickers": int(prices["ticker"].nunique()),
          "train_rows": int(len(train_df)),
          "test_rows": int(len(test_df)),
          "test_macro_f1": float(model_f1),
          "class_weight": "balanced",
          "sample_weight": True,
        },
      )
      print(
        f"\nモデルを保存しました: {model_dir}/{MODEL_FILENAME}, {model_dir}/{METADATA_FILENAME}"
      )
      print("次回以降は `predict` サブコマンドでこのモデルを再学習なしに使い回せます。")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="実装途中のパイプラインの精度を簡易チェックする"
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  train_parser = subparsers.add_parser(
    "train", help="学習・評価を行い、モデルをファイルに保存する"
  )
  train_parser.add_argument(
    "--data-dir", type=Path, required=True, help="CSVが入ったディレクトリ"
  )
  train_parser.add_argument(
    "--model-dir",
    type=Path,
    default=Path("models/task2"),
    help="モデルの保存先ディレクトリ（デフォルト: models/task2）",
  )
  train_parser.add_argument(
    "--upper", type=float, default=0.10, help="利確ライン（例: 0.10 = +10%）"
  )
  train_parser.add_argument(
    "--lower", type=float, default=-0.05, help="損切りライン（例: -0.05 = -5%）"
  )
  train_parser.add_argument(
    "--max-days", type=int, default=60, help="時間バリア（日数）"
  )
  train_parser.add_argument(
    "--sample-tickers",
    type=int,
    default=None,
    help="使用する銘柄数を制限する場合に指定",
  )
  train_parser.add_argument(
    "--multi-split",
    action="store_true",
    help="複数のtrain/test比率(0.6,0.7,0.8)で比較評価する",
  )

  predict_parser = subparsers.add_parser(
    "predict", help="保存済みモデルを読み込み、再学習せずに予測する"
  )
  predict_parser.add_argument(
    "--model-dir",
    type=Path,
    required=True,
    help="`train`で保存したモデルのディレクトリ",
  )
  predict_parser.add_argument(
    "--data-dir", type=Path, required=True, help="予測対象銘柄のCSVが入ったディレクトリ"
  )
  predict_parser.add_argument(
    "--ticker", type=str, required=True, help="予測したい銘柄コード"
  )

  args = parser.parse_args()

  if args.command == "train":
    run(
      data_dir=args.data_dir,
      upper=args.upper,
      lower=args.lower,
      max_days=args.max_days,
      sample_tickers=args.sample_tickers,
      model_dir=args.model_dir,
      multi_split=args.multi_split,
    )
  elif args.command == "predict":
    model, metadata = load_model_artifact(args.model_dir)
    predict_for_ticker(model, metadata, args.data_dir, args.ticker)


if __name__ == "__main__":
  main()
