"""データ読み込み・統一フォーマット化モジュール

計画書 §2 に対応。
yfinance で取得した CSV（3行ヘッダー形式）を読み込み、
統一フォーマットの DataFrame に変換する。
"""

import os
import glob
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config


def load_single_csv(filepath: str) -> pd.DataFrame:
  """1つのCSVファイルを読み込み、統一フォーマットに変換する。

  CSV形式（yfinance出力）:
      行1: Price,Close,High,Low,Open,Volume
      行2: Ticker,XXXX.T,XXXX.T,...
      行3: Date,,,,,
      行4以降: 日付,終値,高値,安値,始値,出来高

  Args:
      filepath: CSVファイルのパス

  Returns:
      columns=[date, open, high, low, close, volume, ticker] の DataFrame
  """
  # 最初の3行をスキップしてデータ部分を読み込む
  df = pd.read_csv(
    filepath,
    skiprows=3,
    header=None,
    names=["date", "close", "high", "low", "open", "volume"],
  )

  # 日付をdatetime型に変換
  df["date"] = pd.to_datetime(df["date"], errors="coerce")

  # 数値列を確実にfloat型に変換
  for col in ["open", "high", "low", "close", "volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

  # ファイル名から銘柄コードを抽出（例: '285A.csv' → '285A'）
  ticker = Path(filepath).stem
  df["ticker"] = ticker

  # 日付でソート
  df = df.sort_values("date").reset_index(drop=True)

  # 終値がNaNの行を除去
  df = df.dropna(subset=["close"]).reset_index(drop=True)

  # 列順を統一: date, open, high, low, close, volume, ticker
  df = df[["date", "open", "high", "low", "close", "volume", "ticker"]]

  return df


def load_all_csvs(csv_dir: Optional[str] = None) -> pd.DataFrame:
  """指定ディレクトリ内の全CSVファイルを読み込み、結合する。

  Args:
      csv_dir: CSVファイルが格納されたディレクトリ（デフォルト: config.CSV_DIR）

  Returns:
      全銘柄のデータを結合した DataFrame
  """
  if csv_dir is None:
    csv_dir = config.CSV_DIR

  csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))

  if not csv_files:
    raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_dir}")

  dfs = []
  loaded_count = 0
  skipped_count = 0

  for filepath in csv_files:
    filename = os.path.basename(filepath)
    # error.log 等の非データファイルをスキップ
    if not filename.endswith(".csv") or filename == "error.log":
      continue

    try:
      df = load_single_csv(filepath)
      if len(df) > 0:
        dfs.append(df)
        loaded_count += 1
      else:
        print(f"  スキップ（データなし）: {filename}")
        skipped_count += 1
    except Exception as e:
      print(f"  読み込みエラー: {filename} - {e}")
      skipped_count += 1

  if not dfs:
    raise ValueError("有効なCSVデータが見つかりません")

  # 全銘柄を結合
  combined = pd.concat(dfs, ignore_index=True)

  # 銘柄コード → 日付でソート
  combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

  # サマリー出力
  n_tickers = combined["ticker"].nunique()
  total_rows = len(combined)
  date_min = combined["date"].min().strftime("%Y-%m-%d")
  date_max = combined["date"].max().strftime("%Y-%m-%d")

  print("[データ読み込み完了]")
  print(f"  銘柄数: {n_tickers}")
  print(f"  総行数: {total_rows:,}")
  print(f"  期間: {date_min} ～ {date_max}")
  print(f"  読み込み成功: {loaded_count}件, スキップ: {skipped_count}件")

  return combined
