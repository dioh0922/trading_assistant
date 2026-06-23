"""
AIトレードシステム ステップ5: アシストロジックの実装（3段階シグナル）

plan.md ステップ5に対応:

    | シグナル | 状態           | 判断の考え方                          |
    |----------|----------------|----------------------------------------|
    | 強気     | 保持・新規買い | RSI反転 ＋ 移動平均線乖離の縮小        |
    | 警戒     | エントリー見送り | ATRに対する価格乖離が異常値（急騰）  |
    | 中立     | 様子見         | 上記条件を満たさない局面               |

ステップ4のMLモデルはブラックボックスになりがちなので、ステップ5では
あえてルールベースの解釈可能なロジックに「翻訳」する。
これにより、なぜそのシグナルが出たのかを人間が説明できる状態にする。

入力: step2_domain_features.build_step2_dataset() 以降の出力DataFrame
    （rsi14, rsi_bullish_divergence, dev_ma25_zscore, is_overbought_heat 等）
出力: assist_signal列を追加したDataFrame

このファイルは関数定義のみを提供するモジュール。
実行エントリーポイントは project直下の main.py。
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCE_DIR = BASE_DIR / "resource"

SIGNAL_BULLISH = "強気"
SIGNAL_CAUTION = "警戒"
SIGNAL_NEUTRAL = "中立"


# =============================================================================
# 1. 強気シグナル: RSI反転 ＋ 移動平均線乖離の縮小
# =============================================================================

def _detect_rsi_reversal(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """
    RSI反転（売られすぎからの切り返し）を判定する。

    条件（いずれかを満たせばRSI反転とみなす）:
        a) 直近lookback日以内にRSIが35以下（売られすぎ圏）を経験し、
           かつ当日のRSIが前日比で上昇している
        b) ステップ2で検出済みのRSI強気ダイバージェンスが当日成立している
    """
    rsi = df["rsi14"]
    recent_min_excl_today = rsi.shift(1).rolling(lookback).min()
    was_oversold = recent_min_excl_today <= 35
    turning_up = rsi > rsi.shift(1)

    condition_a = was_oversold & turning_up
    condition_b = df["rsi_bullish_divergence"] == 1

    return condition_a | condition_b


def _detect_deviation_shrinking(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """
    移動平均線からの乖離（下げすぎ）が縮小しつつあるかを判定する。

    条件: 現在もMAより下（dev_ma25_zscore < 0）だが、
          lookback日前と比べてゼロに近づいている
          （= 下げすぎの状態が解消され始めている）
    """
    z = df["dev_ma25_zscore"]
    still_below_ma = z < 0
    is_recovering = z > z.shift(lookback)
    return still_below_ma & is_recovering


# =============================================================================
# 2. 警戒シグナル: ATRに対する価格乖離が異常値（急騰）
# =============================================================================

def _detect_overheat_warning(df: pd.DataFrame) -> pd.Series:
    """
    ステップ2で計算済みの過熱フラグ（買われすぎ方向のみ）をそのまま使う。
    is_overbought_heat: atr_dev_ma25 >= +2 (ATR正規化乖離が異常値)
    """
    return df["is_overbought_heat"] == 1


# =============================================================================
# 3. 3段階シグナルの統合
# =============================================================================

def generate_assist_signal(df: pd.DataFrame,
                           rsi_lookback: int = 5,
                           dev_lookback: int = 5) -> pd.DataFrame:
    """
    3段階アシストシグナル（強気/警戒/中立）を生成する。

    優先順位: 警戒 > 強気 > 中立
    （急騰による過熱が起きている時は、たとえRSI反転条件を満たしていても
      安全側に倒して「警戒」を優先する）
    """
    d = df.copy()

    for col in ["rsi14", "rsi_bullish_divergence", "dev_ma25_zscore", "is_overbought_heat", "weekly_trend", "volume_ratio"]:
        if col not in d.columns:
            raise KeyError(
                f"'{col}' が見つかりません。"
                f"特徴量エンジニアリング（ステップ1・2）が正しく実行されているか確認してください。"
            )

    rsi_reversal      = _detect_rsi_reversal(d, lookback=rsi_lookback)
    deviation_shrink   = _detect_deviation_shrinking(d, lookback=dev_lookback)
    overheat_warning   = _detect_overheat_warning(d)

    # 順張り型の強気条件を追加
    trending_bullish = (
        (d["weekly_trend"] == 1) &
        (d["rsi14"] >= 50) &
        (d["volume_ratio"] >= 1.2)
    )

    # 既存の逆張り強気条件と順張り強気条件のORを取る
    bullish_cond = (rsi_reversal & deviation_shrink) | trending_bullish

    signal = np.where(
        overheat_warning, SIGNAL_CAUTION,
        np.where(bullish_cond, SIGNAL_BULLISH, SIGNAL_NEUTRAL),
    )

    d["rsi_reversal_flag"]   = rsi_reversal.astype(int)
    d["deviation_shrink_flag"] = deviation_shrink.astype(int)
    d["overheat_warning_flag"] = overheat_warning.astype(int)
    d["trending_bullish_flag"] = trending_bullish.astype(int)
    d["assist_signal"] = signal

    return d


# =============================================================================
# 4. シグナルの妥当性検証（ステップ3のラベルと突き合わせ）
# =============================================================================

def evaluate_signal_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    各シグナルが実際にどれくらい「当たっていたか」を、
    ステップ3で計算済みのトリプルバリア結果（tb_label, tb_return）と
    突き合わせて検証する。

    強気シグナルが出た日のtb_label平均（=勝率に相当）が高いほど、
    このルールベースロジックが機能していると言える。

    ※ tb_label/tb_return が無いDataFrame（ステップ2出力のみ等）の場合は
       シグナルの件数集計のみ返す。
    """
    has_labels = "tb_label" in df.columns and "tb_return" in df.columns

    if has_labels:
        summary = df.groupby("assist_signal").agg(
            件数=("assist_signal", "count"),
            勝率=("tb_label", "mean"),
            平均リターン=("tb_return", "mean"),
        )
    else:
        summary = df.groupby("assist_signal").agg(件数=("assist_signal", "count"))

    # 表示順を固定（強気→警戒→中立）
    order = [s for s in [SIGNAL_BULLISH, SIGNAL_CAUTION, SIGNAL_NEUTRAL] if s in summary.index]
    summary = summary.reindex(order)

    return summary


# =============================================================================
# 5. メイン処理
# =============================================================================

def build_step5_dataset(df: pd.DataFrame,
                        rsi_lookback: int = 5,
                        dev_lookback: int = 5) -> pd.DataFrame:
    """
    ステップ5の全処理をまとめて実行する。

    Parameters
    ----------
    df : step2_domain_features.build_step2_dataset() 以降の出力
        （tb_label/tb_returnがあればステップ3出力でも可。検証精度が上がる）
    """
    d = generate_assist_signal(df, rsi_lookback=rsi_lookback, dev_lookback=dev_lookback)

    print(f"\nステップ5 最終データセット: {len(d)}行 × {len(d.columns)}列")

    print("\nシグナル分布:")
    print(d["assist_signal"].value_counts().reindex(
        [SIGNAL_BULLISH, SIGNAL_CAUTION, SIGNAL_NEUTRAL]
    ))

    summary = evaluate_signal_quality(d)
    print("\nシグナル別の検証結果（勝率・平均リターンはトリプルバリア基準）:")
    print(summary)

    return d
