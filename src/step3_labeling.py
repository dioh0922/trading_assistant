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

利確/損切りラインは2つのモードに対応する:
    - ATRモード（デフォルト）: ATRの倍数でバリア幅を設定。
    ボラティリティに応じてバリア幅が自動的に広がったり狭まったりする。
    - 固定%モード: tp_pct/sl_pctを指定すると自動的にこちらに切り替わる。
    「購入額の+10%で利確、-5%で損切り」のような実運用ルールをそのまま
    ラベルに反映したい場合に使う。holding_periodのデフォルトは45日
    （固定%は到達までATRモードより時間がかかりやすいため）。

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
                         tp_pct: float = None,
                         sl_pct: float = None,
                         price_col: str = "close",
                         high_col: str = "high",
                         low_col: str = "low",
                         atr_col: str = "atr14") -> pd.DataFrame:
    """
    トリプルバリア法でラベルを付与する。

    各日をエントリー日とみなし、以下3本のバリアのうち
    どれに最初に触れるかを判定する:
        上バリア（利確目標） = entry_price + tp_atr_mult * ATR        ［ATRモード］
                            = entry_price * (1 + tp_pct)             ［固定%モード］
        下バリア（損切りライン） = entry_price - sl_atr_mult * ATR     ［ATRモード］
                                = entry_price * (1 - sl_pct)         ［固定%モード］
        垂直バリア（期間切れ） = holding_period 日後

    モードの自動切り替え:
        tp_pct と sl_pct の両方が指定された場合 → 固定%モード
        （例: tp_pct=0.10, sl_pct=0.05 で「+10%利確 / -5%損切り」）
        どちらも指定しない場合 → ATRモード（tp_atr_mult / sl_atr_mult を使用）

    Parameters
    ----------
    holding_period : 最大保有期間（営業日数）
    tp_atr_mult     : [ATRモード] 利確バリアのATR倍率
    sl_atr_mult     : [ATRモード] 損切りバリアのATR倍率
    tp_pct          : [固定%モード] 利確の割合（例: 0.10 = +10%）
    sl_pct          : [固定%モード] 損切りの割合（例: 0.05 = -5%）

    出力列
    ----------
    tb_label          : 1=利確到達(買いシグナル成功), 0=それ以外
    tb_barrier        : 'take_profit' / 'stop_loss' / 'time_out' / NaN(計算不可)
    tb_days_to_touch  : バリアに触れるまでの営業日数
    tb_return         : バリア到達時点のリターン（%）
    tb_barrier_mode   : 'atr' / 'fixed_pct'（どちらのモードで計算したか）
    """
    use_fixed_pct = (tp_pct is not None) and (sl_pct is not None)
    if (tp_pct is not None) != (sl_pct is not None):
        raise ValueError(
            "固定%モードを使う場合は tp_pct と sl_pct の両方を指定してください"
            "（片方だけの指定はできません）。"
        )

    d = df.copy()
    n = len(d)

    closes = d[price_col].to_numpy()
    highs  = d[high_col].to_numpy()
    lows   = d[low_col].to_numpy()
    atrs   = None if use_fixed_pct else d[atr_col].to_numpy()

    label         = np.full(n, np.nan)
    barrier       = np.array([None] * n, dtype=object)
    days_to_touch = np.full(n, np.nan)
    ret_at_touch  = np.full(n, np.nan)

    for i in range(n - 1):
        entry_price = closes[i]

        if use_fixed_pct:
            upper = entry_price * (1 + tp_pct)
            lower = entry_price * (1 - sl_pct)
        else:
            upper = entry_price + tp_atr_mult * atrs[i]
            lower = entry_price - sl_atr_mult * atrs[i]

        end_idx = i + holding_period
        if end_idx >= n:
            continue  # 末尾でholding_period分の未来データが完全には揃わない → ラベル計算をスキップ(NaNのまま)

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
    d["tb_barrier_mode"]  = "fixed_pct" if use_fixed_pct else "atr"

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
                        holding_period: int = None,
                        tp_atr_mult: float = 2.0,
                        sl_atr_mult: float = 1.5,
                        tp_pct: float = None,
                        sl_pct: float = None,
                        drawdown_threshold: float = 0.03) -> pd.DataFrame:
    """
    ステップ3の全処理をまとめて実行する。

    Parameters
    ----------
    step2_df : step2_domain_features.build_step2_dataset() の出力
    holding_period : 最大保有期間（営業日数）。Noneの場合は
        - tp_pct/sl_pctを指定した場合（固定%モード）→ 45日
        - 指定しない場合（ATRモード）→ 10日
        を自動的に使う。
    tp_pct, sl_pct : 両方指定すると固定%モードで実行される
        （例: tp_pct=0.10, sl_pct=0.05 → 「+10%利確 / -5%損切り」）

    使用例
    ----------
    # ATRモード（デフォルト）
    build_step3_dataset(step2_df)

    # 固定%モード（+10%利確 / -5%損切り、保有期間45日）
    build_step3_dataset(step2_df, tp_pct=0.10, sl_pct=0.05)
    """
    use_fixed_pct = (tp_pct is not None) and (sl_pct is not None)

    if holding_period is None:
        holding_period = 45 if use_fixed_pct else 10

    d = step2_df.copy()

    # --- トリプルバリア法（買いシグナル用ラベル） ---
    d = apply_triple_barrier(
        d,
        holding_period=holding_period,
        tp_atr_mult=tp_atr_mult,
        sl_atr_mult=sl_atr_mult,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
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

    mode_label = "固定% (fixed_pct)" if use_fixed_pct else "ATR倍率"
    print(f"\nバリアモード: {mode_label}")
    if use_fixed_pct:
        print(f"  利確: +{tp_pct:.1%}  損切り: -{sl_pct:.1%}  保有期間: {holding_period}営業日")
    else:
        print(f"  利確: ATR×{tp_atr_mult}  損切り: ATR×{sl_atr_mult}  保有期間: {holding_period}営業日")

    print(f"\nステップ3 最終データセット: {len(d)}行 × {len(d.columns)}列")
    print("\nステップ3で追加された特徴量・ラベル:")
    new_cols = [
        "tb_label", "tb_barrier", "tb_days_to_touch", "tb_return", "tb_barrier_mode",
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
