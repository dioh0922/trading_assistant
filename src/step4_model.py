"""
AIトレードシステム ステップ4: モデル構築とバリデーション

plan.md ステップ4に対応:
    - アルゴリズム選定: 構造化データに強い LightGBM（未インストール時は
      scikit-learnのHistGradientBoostingClassifierに自動フォールバック）
    - Walk-Forward Validation: データのシャッフルは行わず、過去のデータで
      学習し、常に未来のデータで検証する時系列バリデーション

入力: step3_labeling.build_step3_dataset() の出力DataFrame
出力:
    - フォールドごとの評価指標
    - 特徴量重要度
    - Out-of-Sample（検証期間のみ）予測結果
    - 学習済みモデル（全期間で再学習した最終モデル）

このファイルは関数定義のみを提供するモジュール。
実行エントリーポイントは project直下の main.py。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCE_DIR = BASE_DIR / "resource"

# --- LightGBMが使えればそちらを優先（plan.md推奨）、無ければsklearnにフォールバック ---
try:
    import lightgbm as lgb
    _BACKEND = "lightgbm"
except ImportError:
    lgb = None
    _BACKEND = "sklearn(HistGradientBoosting)"


# =============================================================================
# 1. 特徴量列の選定
# =============================================================================

# 学習に使ってはいけない列（価格スケール依存 or 未来情報のリーク）
_EXCLUDE_COLS = {
    # 生の価格・出来高（銘柄やスケールに依存し汎化しにくいため除外）
    "open", "high", "low", "close", "volume",
    "ma25", "ma75", "weekly_ma5", "weekly_ma13", "atr14",
    # ターゲット（ステップ3で生成。予測対象そのもの）
    "tb_label", "avoid_entry_flag",
    # ターゲット生成の過程で使った未来情報（特徴量に混ぜると正解漏洩になる）
    "tb_barrier", "tb_days_to_touch", "tb_return", "forward_max_drawdown",
}


def select_feature_columns(df: pd.DataFrame) -> list:
    """
    学習に使う特徴量列を選定する。
    価格スケールに依存する生値や、ラベル生成に使った未来情報は除外し、
    比率・乖離率・フラグなど「市場の状態」を表す正規化済み指標のみを残す。
    """
    cols = [c for c in df.columns if c not in _EXCLUDE_COLS]
    # 数値型のみ残す（万一の文字列列混入対策）
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols


# =============================================================================
# 2. モデルの構築
# =============================================================================

def _build_model(random_state: int = 42):
    """
    LightGBMが利用可能ならLGBMClassifier、無ければ
    HistGradientBoostingClassifier（sklearn）を返す。
    どちらも構造化データ・欠損値混在に強い勾配ブースティング木。
    """
    if _BACKEND == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbosity=-1,
        )
    else:
        return HistGradientBoostingClassifier(
            max_depth=5,
            learning_rate=0.05,
            max_iter=300,
            random_state=random_state,
        )


# =============================================================================
# 3. Walk-Forward Validation
# =============================================================================

def walk_forward_validation(df: pd.DataFrame,
                            feature_cols: list,
                            target_col: str = "tb_label",
                            n_splits: int = 5,
                            min_train_size: int = 100) -> dict:
    """
    時系列順を保ったWalk-Forward Validationを実施する。

    sklearnのTimeSeriesSplitを使用: データはシャッフルせず、
    fold_iの学習データは常にfold_iの検証データより過去のみで構成される
    （未来のデータを学習に混ぜない）。

    Returns
    -------
    dict with keys:
        fold_metrics       : フォールドごとの評価指標DataFrame
        feature_importance : 全フォールド平均の特徴量重要度DataFrame
        oos_predictions     : 検証期間（Out-of-Sample）の予測結果DataFrame
        final_model          : 全データで再学習した最終モデル
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_records = []
    importance_records = []
    oos_pred_frames = []

    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        if len(train_idx) < min_train_size:
            continue  # 最初の方は学習データが少なすぎるのでスキップ

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = _build_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        # predict_probaが2クラス分あることを確認してから正例確率を取得
        proba = model.predict_proba(X_test)
        y_proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

        # --- 評価指標 ---
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = np.nan  # 検証期間内に片方のクラスしか無い場合

        train_start, train_end = df.index[train_idx[0]], df.index[train_idx[-1]]
        test_start, test_end = df.index[test_idx[0]], df.index[test_idx[-1]]

        fold_records.append({
            "fold": fold_i,
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "n_train": len(train_idx), "n_test": len(test_idx),
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "roc_auc": auc,
        })

        # --- 特徴量重要度 ---
        if _BACKEND == "lightgbm":
            importance_records.append(pd.Series(model.feature_importances_, index=feature_cols))
        else:
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(model, X_test, y_test, n_repeats=5,
                                        random_state=42, n_jobs=-1)
            importance_records.append(pd.Series(pi.importances_mean, index=feature_cols))

        # --- Out-of-Sample予測の保存 ---
        oos_df = pd.DataFrame({
            "fold": fold_i,
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }, index=df.index[test_idx])
        oos_pred_frames.append(oos_df)

    fold_metrics = pd.DataFrame(fold_records)
    feature_importance = (
        pd.concat(importance_records, axis=1).mean(axis=1)
        .sort_values(ascending=False)
        .rename("importance")
        .to_frame()
    )
    oos_predictions = pd.concat(oos_pred_frames).sort_index()

    # --- 最終モデル: 全期間のデータで再学習（実運用にはこのモデルを使う） ---
    final_model = _build_model()
    final_model.fit(X, y)

    return {
        "fold_metrics": fold_metrics,
        "feature_importance": feature_importance,
        "oos_predictions": oos_predictions,
        "final_model": final_model,
        "backend": _BACKEND,
    }


# =============================================================================
# 4. メイン処理
# =============================================================================

def build_step4_results(step3_df: pd.DataFrame,
                        target_col: str = "tb_label",
                        n_splits: int = 5) -> dict:
    """
    ステップ4の全処理をまとめて実行する。

    Parameters
    ----------
    step3_df : step3_labeling.build_step3_dataset() の出力
    """
    feature_cols = select_feature_columns(step3_df)

    print(f"使用バックエンド: {_BACKEND}")
    print(f"特徴量数: {len(feature_cols)}")
    print(f"特徴量一覧: {feature_cols}")

    results = walk_forward_validation(
        step3_df, feature_cols, target_col=target_col, n_splits=n_splits,
    )

    fold_metrics = results["fold_metrics"]
    print("\n--- フォールドごとの評価指標 ---")
    print(fold_metrics[["fold", "n_train", "n_test", "accuracy", "precision",
                        "recall", "f1", "roc_auc"]].to_string(index=False))

    print("\n--- 平均スコア（全フォールド） ---")
    print(fold_metrics[["accuracy", "precision", "recall", "f1", "roc_auc"]].mean())

    print("\n--- 特徴量重要度 Top10 ---")
    print(results["feature_importance"].head(10))

    return results


def save_step4_results(results: dict,
                      fold_metrics_path: Path = None,
                      feature_importance_path: Path = None,
                      oos_predictions_path: Path = None,
                      model_path: Path = None) -> None:
    """ステップ4の各種結果をresource/に保存する。"""
    fold_metrics_path = fold_metrics_path or RESOURCE_DIR / "step4_fold_metrics.csv"
    feature_importance_path = feature_importance_path or RESOURCE_DIR / "step4_feature_importance.csv"
    oos_predictions_path = oos_predictions_path or RESOURCE_DIR / "step4_oos_predictions.csv"
    model_path = model_path or RESOURCE_DIR / "step4_model.pkl"

    results["fold_metrics"].to_csv(fold_metrics_path, index=False)
    results["feature_importance"].to_csv(feature_importance_path)
    results["oos_predictions"].to_csv(oos_predictions_path)
    joblib.dump(results["final_model"], model_path)

    print(f"\nCSV保存完了: {fold_metrics_path}")
    print(f"CSV保存完了: {feature_importance_path}")
    print(f"CSV保存完了: {oos_predictions_path}")
    print(f"モデル保存完了: {model_path}")
