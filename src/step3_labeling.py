"""
AIトレードシステム ステップ3: ターゲット（ラベル）の設計

plan.md ステップ3に対応:
    - トリプルバリア法の適用
        * 「利確目標」「損切りライン」「期間」を設定し、
          どの境界に先に触れるかをラベル化する。
    - ラベルの分類
        * 買いシグナル用: 損切りにならず、かつ利確目標に到達した場合を「1」
        * 見送りシグナル用: エントリー後に一定割合のドローダウンが
          発生するかどうかを示すフラグ

入力: step2_domain_features.build_step2_dataset() の出力DataFrame
      （close, high, low, atr14 等を含む）
出力: トリプルバリアのラベルを追加したDataFrame

利確/損切りラインは固定%ではなくATRの倍数で設定する。
これにより、その時点のボラティリティに応じてバリア幅が自動的に
広がったり狭まったりする（ステップ2までで注入したドメイン知識と一貫性を持たせる）。

このファイルは関数定義のみを提供するモジュール。
実行エントリーポイントは project直下の main.py。
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCE_DIR = BASE_DIR / "resource"


# =============================================================================
# 1. トリプルバリア法
# =============================================================================

def apply_triple_barrier(df: pd.DataFrame,
                         holding_period: int = 10,
                         tp_atr_mult: float = 2.0,
                         sl_atr_mult: float = 1.5,
                         price_col: str = "close",
                         high_col: str = "high",
                         low_col: str = "low",
                         atr_col: str = "atr14") -> pd.DataFrame:
    """
    トリプルバリア法でラベルを付与する。

    各日をエントリー日とみなし、以下3本のバリアのうち
    どれに最初に触れるかを判定する:
        上バリア（利確目標） = entry_price + tp_atr_mult * ATR
        下バリア（損切りライン） = entry_price - sl_atr_mult * ATR
        垂直バリア（期間切れ） = holding_period 日後

    Parameters
    ----------
    holding_period : 最大保有期間（営業日数）
    tp_atr_mult     : 利確バリアのATR倍率（大きいほど目標が遠い）
    sl_atr_mult     : 損切りバリアのATR倍率（小さいほど損切りが早い）

    出力列
    ----------
    tb_label          : 1=利確到達(買いシグナル成功), 0=それ以外
    tb_barrier        : 'take_profit' / 'stop_loss' / 'time_out' / NaN(計算不可)
    tb_days_to_touch  : バリアに触れるまでの営業日数
    tb_return         : バリア到達時点のリターン（%）
    """
    d = df.copy()
    n = len(d)

    closes = d[price_col].to_numpy()
    highs  = d[high_col].to_numpy()
    lows   = d[low_col].to_numpy()
    atrs   = d[atr_col].to_numpy()

    label         = np.full(n, np.nan)
    barrier       = np.array([None] * n, dtype=object)
    days_to_touch = np.full(n, np.nan)
    ret_at_touch  = np.full(n, np.nan)

    for i in range(n - 1):
        entry_price = closes[i]
        upper = entry_price + tp_atr_mult * atrs[i]
        lower = entry_price - sl_atr_mult * atrs[i]

        end_idx = min(i + holding_period, n - 1)
        if end_idx <= i:
            continue  # 末尾でholding_period分の未来データが無い

        touched = False
        for j in range(i + 1, end_idx + 1):
            hit_tp = highs[j] >= upper
            hit_sl = lows[j] <= lower

            if hit_tp and hit_sl:
                # 同日に両方触れた場合は安全側（損切り）を優先
                label[i] = 0
                barrier[i] = "stop_loss"
                days_to_touch[i] = j - i
                ret_at_touch[i] = (lower - entry_price) / entry_price
                touched = True
                break
            if hit_tp:
                label[i] = 1
                barrier[i] = "take_profit"
                days_to_touch[i] = j - i
                ret_at_touch[i] = (upper - entry_price) / entry_price
                touched = True
                break
            if hit_sl:
                label[i] = 0
                barrier[i] = "stop_loss"
                days_to_touch[i] = j - i
                ret_at_touch[i] = (lower - entry_price) / entry_price
                touched = True
                break

        if not touched:
            # 垂直バリア（期間切れ）: 期間内に到達しなければ
            # 期間終了時点の損益の符号でラベル付け
            final_price = closes[end_idx]
            label[i] = 1 if final_price > entry_price else 0
            barrier[i] = "time_out"
            days_to_touch[i] = end_idx - i
            ret_at_touch[i] = (final_price - entry_price) / entry_price

    d["tb_label"]         = label
    d["tb_barrier"]       = barrier
    d["tb_days_to_touch"] = days_to_touch
    d["tb_return"]        = ret_at_touch

    return d


# =============================================================================
# 2. 見送りシグナル用ラベル（フォワード・ドローダウン）
# =============================================================================

def apply_forward_drawdown(df: pd.DataFrame,
                           holding_period: int = 10,
                           drawdown_threshold: float = 0.03,
                           price_col: str = "close",
                           low_col: str = "low") -> pd.DataFrame:
    """
    エントリー後 holding_period 日以内に、
    一定割合（drawdown_threshold）のドローダウンが発生するかを判定する。

    トリプルバリアの「買い成功/失敗」とは独立に、
    「そもそもエントリーを見送るべきか」を判断するための補助ラベル。

    出力列
    ----------
    forward_max_drawdown : 期間内の最大ドローダウン（負の値、%）
    avoid_entry_flag     : 1=ドローダウン閾値超え(見送り推奨), 0=問題なし
    """
    d = df.copy()
    n = len(d)

    closes = d[price_col].to_numpy()
    lows   = d[low_col].to_numpy()

    max_dd = np.full(n, np.nan)
    flag   = np.full(n, np.nan)

    for i in range(n - 1):
        entry_price = closes[i]
        end_idx = min(i + holding_period, n - 1)
        if end_idx <= i:
            continue

        future_lows = lows[i + 1:end_idx + 1]
        worst_low = future_lows.min()
        dd = (worst_low - entry_price) / entry_price  # 負の値

        max_dd[i] = dd
        flag[i] = 1 if dd <= -drawdown_threshold else 0

    d["forward_max_drawdown"] = max_dd
    d["avoid_entry_flag"] = flag

    return d


# =============================================================================
# 3. メイン処理
# =============================================================================

def build_step3_dataset(step2_df: pd.DataFrame,
                        holding_period: int = 10,
                        tp_atr_mult: float = 2.0,
                        sl_atr_mult: float = 1.5,
                        drawdown_threshold: float = 0.03) -> pd.DataFrame:
    """
    ステップ3の全処理をまとめて実行する。

    Parameters
    ----------
    step2_df : step2_domain_features.build_step2_dataset() の出力
    """
    d = step2_df.copy()

    # --- トリプルバリア法（買いシグナル用ラベル） ---
    d = apply_triple_barrier(
        d,
        holding_period=holding_period,
        tp_atr_mult=tp_atr_mult,
        sl_atr_mult=sl_atr_mult,
    )

    # --- フォワードドローダウン（見送りシグナル用ラベル） ---
    d = apply_forward_drawdown(
        d,
        holding_period=holding_period,
        drawdown_threshold=drawdown_threshold,
    )

    # --- 末尾 holding_period 日分は未来データ不足でラベル計算不可 → 除去 ---
    d = d.dropna(subset=["tb_label", "avoid_entry_flag"])
    d["tb_label"] = d["tb_label"].astype(int)
    d["avoid_entry_flag"] = d["avoid_entry_flag"].astype(int)

    print(f"\nステップ3 最終データセット: {len(d)}行 × {len(d.columns)}列")
    print("\nステップ3で追加された特徴量・ラベル:")
    new_cols = [
        "tb_label", "tb_barrier", "tb_days_to_touch", "tb_return",
        "forward_max_drawdown", "avoid_entry_flag",
    ]
    for col in new_cols:
        print(f"  {col}")

    # --- ラベル分布のサマリ表示 ---
    print("\ntb_label 分布 (1=利確到達 / 0=それ以外):")
    print(d["tb_label"].value_counts().sort_index())
    print("\ntb_barrier 内訳:")
    print(d["tb_barrier"].value_counts())
    print(f"\navoid_entry_flag=1 (ドローダウン{drawdown_threshold:.0%}超え) の割合: "
          f"{d['avoid_entry_flag'].mean():.1%}")

    return d
