"""
AIトレードシステム ステップ6: 最終意思決定フィルタ（シグナル ＋ リスク予測）

STEP6_PLAN.md / レビュー結果に基づく修正プランに対応:
    [5-1] OOS外データの適切な除外:
          dd_pred_probaで確率が得られない行はNaNとし、評価対象から除外する。
    [5-2] ドローダウン閾値の最適化:
          デフォルト閾値を0.40→0.60〜0.70に引き上げ、本当にリスクが高い
          タイミングのみを除外するよう調整。
    [5-3] dev_ma25_zscoreによる極端な逆張りフィルタ:
          safe_volatilityに dev_ma25_zscore <= -2.5 条件を追加し、
          filtered_by_zscore フラグとして個別に記録。
    [5-4] 順張りシグナルフラグ（trending_bullish_flag）の動的閾値設計:
          trending_bullish_flag==1 の日は dd_threshold を適用し、
          逆張り型（フラグ0）の日は dd_threshold - 0.15 で厳しく判定する。
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# step4のインフラを流用
from src.step4_model import select_feature_columns, _build_model, _BACKEND, RESOURCE_DIR


def train_drawdown_model(df: pd.DataFrame,
                         feature_cols: list,
                         target_col: str = "avoid_entry_flag",
                         n_splits: int = 5,
                         min_train_size: int = 100) -> dict:
    """
    avoid_entry_flag (ドローダウン警告) を予測するモデルを Walk-Forward Validation で学習する。
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_records = []
    oos_pred_frames = []

    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        if len(train_idx) < min_train_size:
            continue

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = _build_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        y_proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

        # 評価指標
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = np.nan

        fold_records.append({
            "fold": fold_i,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc
        })

        oos_df = pd.DataFrame({
            "fold": fold_i,
            "dd_true": y_test.values,
            "dd_pred": y_pred,
            "dd_proba": y_proba,
        }, index=df.index[test_idx])
        oos_pred_frames.append(oos_df)

    fold_metrics = pd.DataFrame(fold_records)
    oos_predictions = pd.concat(oos_pred_frames).sort_index()

    # 最終モデル（全期間で学習）
    final_model = _build_model()
    final_model.fit(X, y)

    return {
        "fold_metrics": fold_metrics,
        "oos_predictions": oos_predictions,
        "final_model": final_model,
    }


def apply_final_filter(df: pd.DataFrame,
                       dd_pred_proba: pd.Series,
                       dd_threshold: float = 0.60,
                       atr_percentile_limit: float = 0.90,
                       zscore_lower_limit: float = -2.5) -> pd.DataFrame:
    """
    強気シグナルに対し、ドローダウン確率とボラティリティ制限・下方乖離制限を
    適用して最終判断を下す。

    [5-3] ルールベースフィルタ条件:
        - atr_percentile >= atr_percentile_limit (0.90) : ボラティリティ過熱を見送り
        - dev_ma25_zscore <= zscore_lower_limit (-2.5)  : 下げすぎ逆張りリスクを見送り
      それぞれ個別のフラグ（filtered_by_atr / filtered_by_zscore）として記録する。

    [5-4] 動的閾値 (順張り/逆張りの切り替え):
        - 順張り型強気（trending_bullish_flag == 1）： dd_threshold (デフォルト0.60) を適用
        - 逆張り型強気（trending_bullish_flag == 0）： dd_threshold - 0.15 (例: 0.45) で厳しく判定
    """
    d = df.copy()

    # [5-1] 各日に対するドローダウン予測確率をマージ（検証期間外は NaN のままにする）
    d["drawdown_prob"] = dd_pred_proba.reindex(d.index)

    # [5-4] 動的なドローダウン許容閾値の計算
    # trending_bullish_flag が存在すれば順張り/逆張りで閾値を切り替える
    if "trending_bullish_flag" in d.columns:
        dynamic_threshold = np.where(
            d["trending_bullish_flag"] == 1,
            dd_threshold,                                  # 順張り型: 閾値を緩和
            np.clip(dd_threshold - 0.15, 0.0, 1.0)        # 逆張り型: 閾値を厳格化
        )
    else:
        dynamic_threshold = dd_threshold

    # 条件判定
    is_bullish = d["assist_signal"] == "強気"
    safe_drawdown = d["drawdown_prob"] < dynamic_threshold

    # [5-3] ルールベースフィルタを個別条件として分離
    safe_atr = d["atr_percentile"] < atr_percentile_limit          # ATRボラティリティ上限
    safe_zscore = d["dev_ma25_zscore"] > zscore_lower_limit         # 下方乖離（底抜けリスク）上限
    safe_volatility = safe_atr & safe_zscore

    # 最終エントリー判断 (1=エントリー, 0=見送り)
    d["final_decision"] = np.where(is_bullish & safe_drawdown & safe_volatility, 1.0, 0.0)

    # [5-1] drawdown_prob が NaN（OOS検証対象外）の行は判定も NaN とする
    d.loc[d["drawdown_prob"].isna(), "final_decision"] = np.nan

    # 見送り理由の個別フラグ化（分析・デバッグ用）
    # ※ 複数条件が同時に成立する場合は両フラグが立つ
    d["filtered_by_drawdown"] = np.where(is_bullish & (~safe_drawdown), 1, 0)
    d["filtered_by_atr"]      = np.where(is_bullish & (~safe_atr),      1, 0)  # ATR過熱フィルタ
    d["filtered_by_zscore"]   = np.where(is_bullish & (~safe_zscore),   1, 0)  # zscore底抜けフィルタ [5-3]
    # 後方互換のために旧フラグも残す（ATR + zscore の OR）
    d["filtered_by_volatility"] = np.where(is_bullish & (~safe_volatility), 1, 0)

    # OOS対象外の行は理由フラグもリセット
    filter_flag_cols = ["filtered_by_drawdown", "filtered_by_atr", "filtered_by_zscore", "filtered_by_volatility"]
    d.loc[d["drawdown_prob"].isna(), filter_flag_cols] = 0

    return d


def evaluate_final_performance(df: pd.DataFrame) -> None:
    """
    検証対象期間（final_decision が NaN 以外のデータ）において、
    最終決定フラグ (final_decision) を用いた場合の勝率・平均リターン・
    プロフィットファクターなどを計算して出力する。

    [5-1] OOS対象外（drawdown_prob が NaN）の行は dropna で事前除外済みであることを前提とする。
    [5-3] フィルタ理由を ATR過熱 / zscore底抜け に分離して個別件数を表示する。
    """
    # 検証対象期間（OOS）のデータのみを抽出
    valid_df = df.dropna(subset=["final_decision"]).copy()
    valid_df["final_decision"] = valid_df["final_decision"].astype(int)

    total_bullish  = (valid_df["assist_signal"] == "強気").sum()
    final_entries  = (valid_df["final_decision"] == 1).sum()
    filtered_dd    = valid_df["filtered_by_drawdown"].sum()
    filtered_vol   = valid_df["filtered_by_volatility"].sum()  # ATR + zscore の OR
    # [5-3] 個別フィルタ件数（存在する場合のみ表示）
    filtered_atr    = valid_df["filtered_by_atr"].sum()    if "filtered_by_atr"    in valid_df.columns else None
    filtered_zscore = valid_df["filtered_by_zscore"].sum() if "filtered_by_zscore" in valid_df.columns else None

    print("\n" + "=" * 50)
    print("ステップ6 意思決定の適用サマリ (OOS検証対象データのみ)")
    print("=" * 50)
    print(f"元の強気シグナル件数  : {total_bullish:>5} 件")
    print(f"最終エントリー件数    : {final_entries:>5} 件")
    print(f"見送り件数（合計）    : {total_bullish - final_entries:>5} 件")
    print("-" * 50)
    print(f"  うち ドローダウン予測フィルタ    : {filtered_dd:>4} 件  [5-2: dd_prob 閾値超過]")
    if filtered_atr is not None:
        print(f"  うち ATRボラティリティフィルタ   : {filtered_atr:>4} 件  [5-3: atr_percentile >= 0.90]")
    if filtered_zscore is not None:
        print(f"  うち zscore底抜けフィルタ        : {filtered_zscore:>4} 件  [5-3: dev_ma25_zscore <= -2.5]")
    print(f"  うち ボラティリティ系フィルタ計  : {filtered_vol:>4} 件  (ATR | zscore)")
    print("=" * 50)

    if "tb_label" in valid_df.columns and "tb_return" in valid_df.columns:
        # --- フィルタ適用前の強気シグナルの成績 ---
        bullish_df = valid_df[valid_df["assist_signal"] == "強気"]
        if len(bullish_df) > 0:
            raw_win_rate = bullish_df["tb_label"].mean()
            raw_ret = bullish_df["tb_return"].mean()
            # プロフィットファクター
            raw_profit = bullish_df.loc[bullish_df["tb_return"] > 0, "tb_return"].sum()
            raw_loss   = bullish_df.loc[bullish_df["tb_return"] < 0, "tb_return"].abs().sum()
            raw_pf = raw_profit / raw_loss if raw_loss > 0 else float("inf")
            print(f"\n[フィルタ前] 強気シグナル ({len(bullish_df)} 件)")
            print(f"  勝率: {raw_win_rate:.1%}  |  平均リターン: {raw_ret:.2%}  |  PF: {raw_pf:.2f}")
        else:
            print("\nフィルタ適用前の強気シグナルはありません。")

        # --- フィルタ適用後の最終エントリーの成績 ---
        final_df = valid_df[valid_df["final_decision"] == 1]
        if len(final_df) > 0:
            final_win_rate = final_df["tb_label"].mean()
            final_ret = final_df["tb_return"].mean()
            cum_ret = final_df["tb_return"].sum()
            # プロフィットファクター [4-3 PF改善の検証]
            profit = final_df.loc[final_df["tb_return"] > 0, "tb_return"].sum()
            loss   = final_df.loc[final_df["tb_return"] < 0, "tb_return"].abs().sum()
            pf = profit / loss if loss > 0 else float("inf")
            print(f"\n[フィルタ後] 最終エントリー ({len(final_df)} 件)")
            print(f"  勝率: {final_win_rate:.1%}  |  平均リターン: {final_ret:.2%}  |  PF: {pf:.2f}")
            print(f"  累積リターン(単利): {cum_ret:.2%}")

            # 勝率改善の確認 [4-1: 55%以上を目標]
            if "tb_label" in bullish_df.columns:
                improvement = final_win_rate - raw_win_rate
                print(f"\n  >> 勝率改善幅: {improvement:+.1%}  (目標: 55%以上)")
                if final_win_rate >= 0.55:
                    print("  [OK] 目標勝率 55% を達成しています。")
                else:
                    print("  [!!] 目標勝率 55% を未達成です。閾値の調整を検討してください。")
        else:
            print("フィルタ適用後の最終エントリーはありません。")
