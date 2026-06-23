"""
AIトレードシステム ステップ6: 最終意思決定フィルタ（シグナル ＋ リスク予測）

STEP6_PLAN.md に対応:
    - MLによるドローダウン予測:
      avoid_entry_flag (エントリー後10日以内に3%以上のドローダウンが発生するフラグ) を
      予測する第2のLightGBMモデルを構築。
    - ルールベースフィルタの結合:
      強気シグナルが出ている日の中でも、ドローダウン確率が閾値以上、または
      ボラティリティが異常に高い日はエントリーを見送る。
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
                       dd_threshold: float = 0.40,
                       atr_percentile_limit: float = 0.90) -> pd.DataFrame:
    """
    強気シグナルに対し、ドローダウン確率とボラティリティ制限を適用して最終判断を下す。
    """
    d = df.copy()

    # 各日に対するドローダウン予測確率をマージ (検証期間外は NaN になるため 0 埋め)
    # index を揃えるために reindex or 直接代入
    d["drawdown_prob"] = dd_pred_proba.reindex(d.index).fillna(0.0)

    # 条件判定
    is_bullish = d["assist_signal"] == "強気"
    safe_drawdown = d["drawdown_prob"] < dd_threshold
    safe_volatility = d["atr_percentile"] < atr_percentile_limit

    # 最終エントリー判断 (1=エントリー, 0=見送り)
    d["final_decision"] = np.where(is_bullish & safe_drawdown & safe_volatility, 1, 0)

    # 見送り理由のフラグ化 (分析用)
    d["filtered_by_drawdown"] = np.where(is_bullish & (~safe_drawdown), 1, 0)
    d["filtered_by_volatility"] = np.where(is_bullish & (~safe_volatility), 1, 0)

    return d


def evaluate_final_performance(df: pd.DataFrame) -> None:
    """
    最終決定フラグ (final_decision) を用いた場合の勝率、平均リターンなどを計算して出力する。
    """
    total_bullish = (df["assist_signal"] == "強気").sum()
    final_entries = (df["final_decision"] == 1).sum()
    filtered_dd = df["filtered_by_drawdown"].sum()
    filtered_vol = df["filtered_by_volatility"].sum()

    print("\n--- ステップ6 意思決定の適用サマリ ---")
    print(f"元の強気シグナル件数: {total_bullish} 件")
    print(f"最終エントリー件数  : {final_entries} 件 (見送り: {total_bullish - final_entries} 件)")
    print(f"  - ドローダウン予測により見送り : {filtered_dd} 件")
    print(f"  - ボラティリティ過熱で見送り   : {filtered_vol} 件")

    if "tb_label" in df.columns and "tb_return" in df.columns:
        # 元の強気シグナルの勝率
        bullish_df = df[df["assist_signal"] == "強気"]
        if len(bullish_df) > 0:
            raw_win_rate = bullish_df["tb_label"].mean()
            raw_ret = bullish_df["tb_return"].mean()
            print(f"\nフィルタ適用前の強気勝率: {raw_win_rate:.1%} (平均リターン: {raw_ret:.2%})")
        else:
            print("\nフィルタ適用前の強気シグナルはありません。")

        # 最終判定をクリアしたエントリーの勝率
        final_df = df[df["final_decision"] == 1]
        if len(final_df) > 0:
            final_win_rate = final_df["tb_label"].mean()
            final_ret = final_df["tb_return"].mean()
            print(f"フィルタ適用後の最終勝率: {final_win_rate:.1%} (平均リターン: {final_ret:.2%})")

            # 累積リターンシミュレーション (単利加算)
            cum_ret = final_df["tb_return"].sum()
            print(f"エントリー合計の累積リターン(単利): {cum_ret:.2%}")
        else:
            print("フィルタ適用後の最終エントリーはありません。")
