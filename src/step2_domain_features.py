"""
AIトレードシステム ステップ2: 特徴量エンジニアリング（ドメイン知識の注入）

plan.md ステップ2に対応:
    - 相対指標の生成
        * 移動平均線からの乖離率の正規化（下げすぎ・上げすぎの定量化）
        * RSIの「強気のダイバージェンス」検出
    - 過熱感の数値化
        * ATRに対する乖離幅の算出（直近の急騰が異常かどうかの判定）
        * 価格変化の加速度（モメンタム）による急騰の一時性の予測

入力: step1_feature_engineering.build_step1_dataset() の出力DataFrame
      （close, ma25, ma75, atr14, rsi14, mom5, mom20, mom_accel 等を含む）
出力: ステップ2の特徴量を追加したDataFrame

このファイルは関数定義のみを提供するモジュール。
実行エントリーポイントは project直下の main.py。
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCE_DIR = BASE_DIR / "resource"


# =============================================================================
# 1. 相対指標の正規化（下げすぎ・上げすぎの定量化）
# =============================================================================

def add_relative_indicators(df: pd.DataFrame, zscore_window: int = 60) -> pd.DataFrame:
    """
    ステップ1で計算済みの移動平均乖離率（dev_ma25, dev_ma75）を
    ローリングZスコア化し、「統計的にどれくらい下げすぎ/上げすぎか」を表す。

    生の乖離率（%）は銘柄やボラティリティ水準によってスケールが変わるため、
    Zスコア化することで「過去N日の自分自身の分布の中でどの位置か」という
    比較可能な指標に変換する。
    """
    d = df.copy()

    for col in ["dev_ma25", "dev_ma75"]:
        if col not in d.columns:
            raise KeyError(
                f"'{col}' が見つかりません。"
                f"先に step1_feature_engineering.build_step1_dataset() を実行してください。"
            )
        roll_mean = d[col].rolling(zscore_window).mean()
        roll_std  = d[col].rolling(zscore_window).std()
        d[f"{col}_zscore"] = (d[col] - roll_mean) / roll_std.replace(0, np.nan)

    # 「下げすぎ」「上げすぎ」のしきい値フラグ（|Zスコア| >= 2 を統計的な極値とみなす）
    d["oversold_flag"]   = (d["dev_ma25_zscore"] <= -2).astype(int)
    d["overbought_flag"] = (d["dev_ma25_zscore"] >=  2).astype(int)

    return d


# =============================================================================
# 2. RSIの強気ダイバージェンス検出
# =============================================================================

def detect_rsi_bullish_divergence(df: pd.DataFrame,
                                   lookback: int = 20,
                                   price_col: str = "close",
                                   rsi_col: str = "rsi14") -> pd.DataFrame:
    """
    強気のダイバージェンス（Bullish Divergence）を検出する。

    定義: 過去lookback日間の安値更新局面で、
          価格は当時の安値以下を更新しているのに
          RSIは当時の値より切り上がっている状態。
          → 売り圧力が弱まりつつあるサインとして扱われる。

    出力列:
        rsi_bullish_divergence : 1=検出, 0=未検出
        rsi_divergence_strength: 検出時のRSI切り上がり幅（強さの目安。未検出時は0）
    """
    d = df.copy()
    closes = d[price_col].to_numpy()
    rsis   = d[rsi_col].to_numpy()
    n = len(d)

    flag     = np.zeros(n, dtype=int)
    strength = np.zeros(n, dtype=float)

    for i in range(lookback, n):
        window = closes[i - lookback:i]          # 当日を含まない過去lookback日
        prev_low_rel_idx = int(np.argmin(window))
        prev_low_idx = i - lookback + prev_low_rel_idx

        prev_low_price = closes[prev_low_idx]
        prev_low_rsi   = rsis[prev_low_idx]

        today_price = closes[i]
        today_rsi   = rsis[i]

        is_price_lower_low = today_price <= prev_low_price
        is_rsi_higher_low  = today_rsi > prev_low_rsi

        if is_price_lower_low and is_rsi_higher_low:
            flag[i] = 1
            strength[i] = today_rsi - prev_low_rsi

    d["rsi_bullish_divergence"]  = flag
    d["rsi_divergence_strength"] = strength
    return d


# =============================================================================
# 3. 過熱感の数値化（ATR正規化乖離 + 急騰の一時性）
# =============================================================================

def add_overheat_features(df: pd.DataFrame,
                          overheat_threshold: float = 2.0) -> pd.DataFrame:
    """
    ATRを基準とした過熱感を数値化する。

    生の乖離率（%）ではなく「その時点のボラティリティ（ATR）何個分
    乖離しているか」で測ることで、静かな相場でのわずかな乖離と
    荒れた相場での同じ%乖離を区別できる（ドメイン知識の注入）。
    """
    d = df.copy()

    for col in ["atr14", "ma25", "close", "mom_accel"]:
        if col not in d.columns:
            raise KeyError(
                f"'{col}' が見つかりません。"
                f"先に step1_feature_engineering.build_step1_dataset() を実行してください。"
            )

    # ---- MAからのATR正規化乖離幅 ----
    # 正の値が大きい = 直近の急騰がボラティリティに対して異常
    d["atr_dev_ma25"] = (d["close"] - d["ma25"]) / d["atr14"]

    # ---- 過熱（急騰・急落）の異常値フラグ ----
    d["is_overheated"] = (d["atr_dev_ma25"].abs() >= overheat_threshold).astype(int)
    d["is_overbought_heat"] = (d["atr_dev_ma25"] >=  overheat_threshold).astype(int)
    d["is_oversold_heat"]   = (d["atr_dev_ma25"] <= -overheat_threshold).astype(int)

    # ---- 急騰の一時性判定 ----
    # mom_accel(短期モメンタム - 長期モメンタム)が大きいのに過熱フラグも立っている場合、
    # 「急騰だが続きにくい」可能性が高いとみなす一時性スコアを作る。
    # スコアが高いほど「過熱した上での急騰＝反落リスクが高い」状態。
    d["pump_temporary_score"] = d["atr_dev_ma25"].clip(lower=0) * d["mom_accel"].clip(lower=0)
    d["is_likely_temporary_pump"] = (
        (d["is_overbought_heat"] == 1) & (d["mom_accel"] > 0)
    ).astype(int)

    return d


# =============================================================================
# 4. メイン処理
# =============================================================================

def build_step2_dataset(step1_df: pd.DataFrame,
                        zscore_window: int = 60,
                        divergence_lookback: int = 20,
                        overheat_threshold: float = 2.0) -> pd.DataFrame:
    """
    ステップ2の全処理をまとめて実行する。

    Parameters
    ----------
    step1_df : step1_feature_engineering.build_step1_dataset() の出力
    """
    d = step1_df.copy()

    # --- 相対指標の正規化 ---
    d = add_relative_indicators(d, zscore_window=zscore_window)

    # --- RSI強気ダイバージェンス検出 ---
    d = detect_rsi_bullish_divergence(d, lookback=divergence_lookback)

    # --- 過熱感の数値化 ---
    d = add_overheat_features(d, overheat_threshold=overheat_threshold)

    # --- 欠損値処理（Zスコアのウォームアップ期間を除去） ---
    d = d.dropna()

    print(f"\nステップ2 最終データセット: {len(d)}行 × {len(d.columns)}列")
    print("\nステップ2で追加された特徴量:")
    new_cols = [
        "dev_ma25_zscore", "dev_ma75_zscore", "oversold_flag", "overbought_flag",
        "rsi_bullish_divergence", "rsi_divergence_strength",
        "atr_dev_ma25", "is_overheated", "is_overbought_heat", "is_oversold_heat",
        "pump_temporary_score", "is_likely_temporary_pump",
    ]
    for col in new_cols:
        print(f"  {col}")

    return d
