"""
AIトレードシステム ステップ1: データ収集とマルチタイムフレームの統合
対象データ: 村田製作所(6981) 日足OHLCV (resource/sample.xlsx)

このファイルは関数定義のみを提供するモジュール。
実行エントリーポイントは project直下の main.py。

ディレクトリ構成:
    project/
    ├── main.py                                ← 実行はこれ
    ├── src/
    │   └── step1_feature_engineering.py       ← このファイル（関数群）
    └── resource/
        ├── sample.xlsx                        ← 入力Excel
        └── step1_dataset.csv                  ← 出力CSV（実行後に生成）
"""

from pathlib import Path

import pandas as pd
import numpy as np

# このファイル(src/)の親 = project/ ルート
BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCE_DIR = BASE_DIR / "resource"


# =============================================================================
# 1. データ読み込み
# =============================================================================

def load_daily_data(filepath: str) -> pd.DataFrame:
    """
    sample.xlsx から日足データを読み込んでクリーニングする。
    先頭2行はヘッダー情報なのでスキップ。
    """
    df = pd.read_excel(
        filepath,
        skiprows=2,          # 最初の2行（銘柄コード行、関数行）をスキップ
        names=["銘柄名称", "市場名称", "足種", "日付", "時刻",
               "始値", "高値", "安値", "終値", "出来高"],
    )

    # 日付列: "--------" などのゴミ行を除去してdatetime変換
    df = df[pd.to_datetime(df["日付"], errors="coerce").notna()].copy()
    df["日付"] = pd.to_datetime(df["日付"])
    df = df.set_index("日付").sort_index()

    # 必要なOHLCV列だけ残す
    df = df[["始値", "高値", "安値", "終値", "出来高"]].copy()

    # 英語列名に統一（以降の計算で使いやすくする）
    df.columns = ["open", "high", "low", "close", "volume"]

    # 数値型に変換（Excelの浮動小数点ゆれを丸める）
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    print(f"日足データ読み込み完了: {len(df)}行 ({df.index[0].date()} ～ {df.index[-1].date()})")
    return df


# =============================================================================
# 2. 週足データの取得
#    パターンA: Sheet2に週足データが既に存在する場合 → そのまま読み込む
#    パターンB: 日足しかない場合 → resampleで生成する
# =============================================================================

def load_weekly_data(filepath: str, sheet_name: str = "Sheet2") -> pd.DataFrame:
    """
    【パターンA】Sheet2など別シートに格納済みの週足データを読み込む。
    sample.xlsxの実データで確認したところ、Sheet2は足種='W'の週足データが
    Sheet1（日足）と全く同じレイアウトで入っている。
    """
    df = pd.read_excel(
        filepath,
        sheet_name=sheet_name,
        skiprows=2,
        names=["銘柄名称", "市場名称", "足種", "日付", "時刻",
               "始値", "高値", "安値", "終値", "出来高"],
    )

    # ヘッダー行やゴミ行（"--------"等）を除去
    df = df[pd.to_datetime(df["日付"], errors="coerce").notna()].copy()
    df["日付"] = pd.to_datetime(df["日付"])
    df = df.set_index("日付").sort_index()

    df = df[["始値", "高値", "安値", "終値", "出来高"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    print(f"週足データ読み込み完了(Sheet2): {len(df)}週 "
          f"({df.index[0].date()} ～ {df.index[-1].date()})")
    return df


def make_weekly_data(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    【パターンB】週足シートが無い場合のフォールバック。
    日足を週次リサンプルして週足を生成する。
    W-FRI: 各週の金曜日を週の末日として集計（日本市場の慣習に合わせる）。
    """
    weekly = daily_df.resample("W-FRI").agg({
        "open":   "first",   # 週初の始値
        "high":   "max",     # 週中の最高値
        "low":    "min",     # 週中の最安値
        "close":  "last",    # 週末の終値
        "volume": "sum",     # 週間出来高の合計
    }).dropna()

    print(f"週足データ生成完了(リサンプル): {len(weekly)}週")
    return weekly


# =============================================================================
# 3. 週足の指標計算（環境認識用）
# =============================================================================

def add_weekly_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    週足ベースのトレンド方向判定指標を追加する。
    """
    w = weekly_df.copy()

    # --- 移動平均線（5週・13週） ---
    w["weekly_ma5"]  = w["close"].rolling(5).mean()
    w["weekly_ma13"] = w["close"].rolling(13).mean()

    # --- 週足MAの傾き（トレンド方向の定量化） ---
    # (今週のMA - 1週前のMA) / 1週前のMA  →  正=上昇トレンド, 負=下降トレンド
    w["weekly_ma5_slope"]  = w["weekly_ma5"].pct_change()
    w["weekly_ma13_slope"] = w["weekly_ma13"].pct_change()

    # --- 週足のトレンド状態（MA5がMA13の上か下か） ---
    w["weekly_trend"] = np.where(w["weekly_ma5"] > w["weekly_ma13"], 1, -1)

    return w


# =============================================================================
# 4. 日足の指標計算（ボラティリティ・モメンタム）
# =============================================================================

def add_daily_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    日足ベースのボラティリティ・テクニカル指標を追加する。
    """
    d = daily_df.copy()

    # ---- ATR (Average True Range) ----
    # True Range = max(高安幅, 前日終値との乖離上側, 前日終値との乖離下側)
    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_period = 14
    d["atr14"] = tr.ewm(span=atr_period, adjust=False).mean()  # EMA版ATR（Wilder方式）

    # ---- ボリンジャーバンド %B ----
    bb_period = 20
    bb_mid  = d["close"].rolling(bb_period).mean()
    bb_std  = d["close"].rolling(bb_period).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    # %B = 0未満で下バンド割れ(売られすぎ), 1超で上バンド越え(買われすぎ)
    d["bb_pct_b"] = (d["close"] - bb_lower) / (bb_upper - bb_lower)

    # ---- RSI (14日) ----
    delta = d["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi14"] = 100 - (100 / (1 + rs))

    # ---- 移動平均からの乖離率 ----
    d["ma25"]  = d["close"].rolling(25).mean()
    d["ma75"]  = d["close"].rolling(75).mean()
    d["dev_ma25"] = (d["close"] - d["ma25"]) / d["ma25"]   # 下げすぎ/上げすぎの定量化
    d["dev_ma75"] = (d["close"] - d["ma75"]) / d["ma75"]

    # ---- ATRに対する価格乖離（過熱感） ----
    # 1日の値幅がATRの何倍か → 急騰・急落の異常値検知
    d["range_vs_atr"] = (d["high"] - d["low"]) / d["atr14"]

    # ---- モメンタム（価格変化の加速度） ----
    d["mom5"]  = d["close"].pct_change(5)    # 5日間リターン
    d["mom20"] = d["close"].pct_change(20)   # 20日間リターン
    # 加速度 = 短期モメンタム - 長期モメンタム（急騰の一時性を捉える）
    d["mom_accel"] = d["mom5"] - d["mom20"]

    # ---- 出来高系特徴量 ----
    vol_mean = d["volume"].rolling(20).mean()
    vol_std  = d["volume"].rolling(20).std()
    d["volume_zscore"] = (d["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    d["volume_ratio"] = d["volume"] / vol_mean.replace(0, np.nan)

    # ---- ボリンジャーバンド幅 (bb_width) の追加 ----
    d["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    d["bb_width_zscore"] = (
        (d["bb_width"] - d["bb_width"].rolling(60).mean())
        / d["bb_width"].rolling(60).std().replace(0, np.nan)
    )

    # ---- ATRパーセンタイル (atr_percentile) の追加 ----
    d["atr_percentile"] = d["atr14"].rolling(250).rank(pct=True)

    return d


# =============================================================================
# 5. ヒゲ情報の追加（日中のintraday情報集約）
# =============================================================================

def add_wick_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    日足のローソク足から上ヒゲ・下ヒゲ比率を算出する。
    実体に対するヒゲの長さは、売買圧力のバランスを示す。
    """
    d = daily_df.copy()
    body      = (d["close"] - d["open"]).abs()
    candle_range = d["high"] - d["low"]

    # ゼロ除算防止
    safe_range = candle_range.replace(0, np.nan)
    safe_body  = body.replace(0, np.nan)

    # 上ヒゲ = 高値 - max(始値, 終値)
    upper_wick = d["high"] - d[["open", "close"]].max(axis=1)
    # 下ヒゲ = min(始値, 終値) - 安値
    lower_wick = d[["open", "close"]].min(axis=1) - d["low"]

    # 値幅に対するヒゲ比率（0〜1）
    d["upper_wick_ratio"] = upper_wick / safe_range
    d["lower_wick_ratio"] = lower_wick / safe_range

    # 実体に対するヒゲ比率（長いほど上値/下値を抑えられた圧力が強い）
    d["upper_wick_body_ratio"] = upper_wick / safe_body
    d["lower_wick_body_ratio"] = lower_wick / safe_body

    # 陽線(1) / 陰線(-1) フラグ
    d["candle_dir"] = np.where(d["close"] >= d["open"], 1, -1)

    return d


# =============================================================================
# 6. 週足特徴量を日足に結合（前週末の値を当週の日足に付与）
# =============================================================================

def merge_weekly_to_daily(daily_df: pd.DataFrame,
                          weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    週足の特徴量を日足データに結合する。
    週足の値は「その週の取引日すべて」に同じ値を充てる（前週の値を参照）。
    """
    weekly_cols = [
        "weekly_ma5", "weekly_ma13",
        "weekly_ma5_slope", "weekly_ma13_slope",
        "weekly_trend",
    ]

    # 週足インデックス（金曜日）を日足の日付に対してマージ
    # merge_asof: 日足の日付に対し、それ以前の直近週足日付の値を使う
    daily_reset  = daily_df.reset_index()
    weekly_reset = weekly_df[weekly_cols].reset_index()

    merged = pd.merge_asof(
        daily_reset.sort_values("日付"),
        weekly_reset.sort_values("日付"),
        on="日付",
        direction="backward",  # 日付以前の最新週足を参照（未来参照しない）
    )

    merged = merged.set_index("日付").sort_index()
    return merged


# =============================================================================
# 7. メイン処理
# =============================================================================

def build_step1_dataset(filepath: str, weekly_sheet: str = "Sheet2") -> pd.DataFrame:
    """
    ステップ1の全処理をまとめて実行し、学習用DataFrameを返す。

    週足データの取得方法:
      1. weekly_sheet（例: "Sheet2"）が存在し、週足データが入っていればそれを使う
      2. 無ければ日足からresampleで生成する（フォールバック）
    """
    # --- 日足読み込み ---
    daily = load_daily_data(filepath)

    # --- 日足の特徴量を追加 ---
    daily = add_daily_features(daily)
    daily = add_wick_features(daily)

    # --- 週足の取得（Sheet2優先、無ければリサンプル） ---
    try:
        weekly = load_weekly_data(filepath, sheet_name=weekly_sheet)
    except (ValueError, KeyError):
        # シートが存在しない/読み込めない場合は日足からリサンプル生成
        print(f"'{weekly_sheet}' が見つからないため、日足からリサンプルして週足を生成します。")
        weekly = make_weekly_data(daily[["open", "high", "low", "close", "volume"]])

    weekly = add_weekly_features(weekly)

    # --- 週足特徴量を日足に結合 ---
    dataset = merge_weekly_to_daily(daily, weekly)

    # --- 欠損値処理（指標計算のウォームアップ期間を除去） ---
    dataset = dataset.dropna()

    print(f"\n最終データセット: {len(dataset)}行 × {len(dataset.columns)}列")
    print("\n特徴量一覧:")
    for col in dataset.columns:
        print(f"  {col}")

    return dataset
