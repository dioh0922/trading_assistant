"""
experiment_shap.py
───────────────────
shap_experiment_plan.md の検証手順を実行するための実験スクリプト。
analyze_ticker.py 本体は書き換えず、独立したスクリプトとしてSHAP値の妥当性を確認する。

2つのサブコマンドを持つ:

  selftest  : 既知の関係を持つ合成データで、SHAP値が正しく機能を検出できるかを確認する（4a）
  analyze   : 実際のfoldモデル群 + analyze_ticker.py が出力したJSONを使って、
              global_importanceとSHAP値のランキングを比較する（4b）

使い方:
    # 4a: 合成データでの妥当性確認（実データもモデルも不要）
    python experiment_shap.py selftest

    # 4b: 実データでの比較
    python experiment_shap.py analyze \
        --model-dir models/task2 \
        --input-json 6098_detail_20260728_094338.json \
        --top-k 10 \
        --report-out reports/shap_experiment_6098.md
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("experiment_shap")


# ──────────────────────────────────────────────────────────────
# 共通: SHAP出力形式の差異を吸収するユーティリティ
# ──────────────────────────────────────────────────────────────
def extract_class_shap(shap_values, class_index: int) -> np.ndarray:
    """
    shap.TreeExplainer().shap_values(...) の戻り値から、指定クラスに対する
    1サンプル分のSHAP値（特徴量数,）を取り出す。
    shapのバージョンによって以下の2形式がありうるため、両方に対応する:
      - list形式: 長さ=クラス数のリスト、各要素が (サンプル数, 特徴量数)
      - array形式: (サンプル数, 特徴量数, クラス数) の3次元配列
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_index])[0]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[0, :, class_index]
    if arr.ndim == 2:
        # 2値分類などクラス軸がない場合はそのまま返す
        return arr[0]
    raise ValueError(f"想定外のshap_values形状です: {arr.shape}")


def resolve_raw_label(classes: np.ndarray, predicted_label: str, label_classes: list[str]):
    """
    model.classes_ が文字列ラベル（'upper'等）ならそのまま返す。
    整数エンコード（0,1,2等）の場合は、metadata.jsonのlabel_classesを使って
    predicted_label（文字列）に対応する整数コードに変換する
    （label_classes[i] が整数コード i に対応している前提）。
    """
    if np.issubdtype(np.array(classes).dtype, np.number):
        try:
            code = label_classes.index(predicted_label)
        except ValueError as e:
            raise ValueError(
                f"'{predicted_label}' が label_classes ({label_classes}) に見つかりません。"
            ) from e
        return code
    return predicted_label


def compute_ensemble_shap(
    models: list, X_single: pd.DataFrame, predicted_label: str, label_classes: list[str]
) -> tuple[pd.Series, float]:
    """
    fold群それぞれのSHAP値（予測クラスに対する寄与）を計算し、平均する。
    戻り値: (特徴量名 -> 平均SHAP値のSeries, 計算にかかった秒数)
    """
    t0 = time.time()
    all_shap = []
    for model in models:
        raw_label = resolve_raw_label(model.classes_, predicted_label, label_classes)
        class_index = list(model.classes_).index(raw_label)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_single)
        class_shap = extract_class_shap(sv, class_index)
        all_shap.append(class_shap)
    elapsed = time.time() - t0

    mean_shap = np.mean(all_shap, axis=0)
    return pd.Series(mean_shap, index=X_single.columns), elapsed


# ──────────────────────────────────────────────────────────────
# selftest: 4a. 合成データでの妥当性確認
# ──────────────────────────────────────────────────────────────
def run_selftest(n_folds: int = 3, n_samples: int = 3000, seed: int = 0) -> bool:
    """
    feature_a が大きいほど "upper" になりやすい合成データを作り、
    upper方向に予測されたサンプルにおいて、feature_aのSHAP値が
    他の特徴量よりも明確に大きい正の値になることを確認する。
    """
    from lightgbm import LGBMClassifier

    rng = np.random.RandomState(seed)
    feature_columns = ["feature_a", "feature_b_noise", "feature_c_noise", "feature_d_noise"]

    n = n_samples
    feature_a = rng.randn(n)
    noise_b = rng.randn(n)
    noise_c = rng.randn(n)
    noise_d = rng.randn(n)

    label = np.where(
        feature_a > 1.0,
        "upper",
        rng.choice(["lower", "timeout"], size=n),
    )

    X = pd.DataFrame({
        "feature_a": feature_a,
        "feature_b_noise": noise_b,
        "feature_c_noise": noise_c,
        "feature_d_noise": noise_d,
    })

    # foldアンサンブルを模して、ブートストラップサンプルでn_folds個のモデルを学習する
    models = []
    for i in range(n_folds):
        idx = rng.choice(n, size=n, replace=True)
        m = LGBMClassifier(n_estimators=50, verbosity=-1, random_state=seed + i)
        m.fit(X.iloc[idx], label[idx])
        models.append(m)

    # feature_aが大きい（=upperになりやすいはず）サンプルを1つ選ぶ
    target_idx = int(np.argmax(feature_a))
    X_single = X.iloc[[target_idx]]
    proba = np.mean([m.predict_proba(X_single) for m in models], axis=0)[0]
    classes = models[0].classes_
    predicted_label = classes[int(np.argmax(proba))]

    log.info("selftest: target sample feature_a=%.3f, predicted=%s, proba=%s",
              feature_a[target_idx], predicted_label, dict(zip(classes, proba.round(3))))

    shap_series, elapsed = compute_ensemble_shap(models, X_single, predicted_label, list(classes))
    log.info("selftest: SHAP計算時間 %.3fs", elapsed)

    ranking = shap_series.reindex(shap_series.abs().sort_values(ascending=False).index)
    print("\n【selftest】SHAP値ランキング（予測クラス: %s に対する寄与）" % predicted_label)
    print(ranking.to_string())

    top_feature = ranking.index[0]
    top_value = ranking.iloc[0]

    passed = (
        predicted_label == "upper"
        and top_feature == "feature_a"
        and top_value > 0
        and abs(top_value) > 2 * ranking.iloc[1:].abs().max()
    )

    print("\n判定基準:")
    print(f"  - 予測ラベルが'upper'か: {predicted_label == 'upper'}")
    print(f"  - SHAP最上位がfeature_aか: {top_feature == 'feature_a'}")
    print(f"  - feature_aのSHAP値が正か: {top_value > 0}")
    print(f"  - feature_aが他の特徴量の2倍以上の絶対値を持つか: "
          f"{abs(top_value) > 2 * ranking.iloc[1:].abs().max()}")
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'}: selftest {'成功' if passed else '失敗'}")

    return passed


# ──────────────────────────────────────────────────────────────
# analyze: 4b. 実データでの比較
# ──────────────────────────────────────────────────────────────
def load_fold_models(model_dir: Path) -> list:
    fold_paths = sorted(model_dir.glob("fold*.joblib"))
    if not fold_paths:
        single = model_dir / "model.joblib"
        if single.exists():
            log.info("fold*.joblibが見つからないため、単一モデル %s を使用します", single)
            return [joblib.load(single)]
        raise FileNotFoundError(f"{model_dir} に fold*.joblib も model.joblib も見つかりません。")
    log.info("found %d fold models: %s", len(fold_paths), [p.name for p in fold_paths])
    return [joblib.load(p) for p in fold_paths]


def load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} が見つかりません。")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_single_comparison(
    models: list, metadata: dict, detail: dict
) -> dict:
    """
    1件のanalyze_ticker.py出力JSONに対して、SHAP計算とglobal_importanceとの
    比較指標一式を計算する。単発analyzeコマンドとbatchコマンドの両方から使う共通処理。

    戻り値には、レポート表示用の詳細(df)と、集計用のスカラー指標の両方を含める。
    """
    feature_columns = metadata["feature_columns"]
    predicted_label = detail["prediction"]["predicted_label"]
    confidence = detail["prediction"]["confidence"]

    feature_map = {f["feature"]: f["value"] for f in detail["features"]}
    global_importance_map = {f["feature"]: f["global_importance"] for f in detail["features"]}

    missing = [c for c in feature_columns if c not in feature_map]
    if missing:
        raise KeyError(
            f"入力JSONの features に、metadata.jsonのfeature_columnsに存在する列が不足しています: {missing}"
        )

    X_single = pd.DataFrame([{c: feature_map[c] for c in feature_columns}])
    shap_series, elapsed = compute_ensemble_shap(
        models, X_single, predicted_label, metadata["label_classes"]
    )

    rows = []
    for feature in feature_columns:
        rows.append({
            "feature": feature,
            "value": feature_map[feature],
            "global_importance": global_importance_map.get(feature, np.nan),
            "shap_value": shap_series[feature],
        })
    df = pd.DataFrame(rows)
    df["global_rank"] = df["global_importance"].rank(ascending=False, method="min").astype(int)
    df["shap_rank"] = df["shap_value"].abs().rank(ascending=False, method="min").astype(int)
    df["rank_shift"] = df["global_rank"] - df["shap_rank"]
    df = df.sort_values("shap_rank")

    rank_corr, p_value = spearmanr(df["global_rank"], df["shap_rank"])

    return {
        "ticker": detail.get("ticker", "?"),
        "analyzed_at": detail.get("analyzed_at") or detail.get("latest_date", "?"),
        "predicted_label": predicted_label,
        "confidence": confidence,
        "df": df,
        "spearman_corr": rank_corr,
        "spearman_pvalue": p_value,
        "top_shap_feature": df.sort_values("shap_rank").iloc[0]["feature"],
        "top_global_feature": df.sort_values("global_rank").iloc[0]["feature"],
        "shap_elapsed_sec": elapsed,
    }


def jaccard_top_k(df: pd.DataFrame, top_k: int) -> tuple[float, set]:
    top_shap_features = set(df.sort_values("shap_rank").head(top_k)["feature"])
    top_global_features = set(df.sort_values("global_rank").head(top_k)["feature"])
    overlap = top_shap_features & top_global_features
    jaccard = len(overlap) / len(top_shap_features | top_global_features)
    return jaccard, overlap


def run_analyze(model_dir: Path, input_json: Path, top_k: int, report_out: Path | None) -> None:
    with open(input_json, "r", encoding="utf-8") as f:
        detail = json.load(f)

    metadata = load_metadata(model_dir)
    models = load_fold_models(model_dir)

    result = compute_single_comparison(models, metadata, detail)
    df = result["df"]
    predicted_label = result["predicted_label"]
    confidence = result["confidence"]
    elapsed = result["shap_elapsed_sec"]

    lines = []
    lines.append(f"# SHAP実験レポート: {detail.get('ticker', '?')}")
    lines.append("")
    lines.append(f"- 予測ラベル: {predicted_label} (confidence {confidence*100:.2f}%)")
    lines.append(f"- fold数: {len(models)}")
    lines.append(f"- SHAP計算時間: {elapsed:.3f}秒（{len(models)}foldの合計）")
    lines.append("")
    lines.append(f"## SHAP値ランキング（{predicted_label}クラスへの寄与）上位{top_k}件")
    lines.append("")
    lines.append("| shap_rank | feature | value | shap_value | global_importance | global_rank |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in df.head(top_k).iterrows():
        direction = "押し上げ(+)" if r["shap_value"] > 0 else "押し下げ(-)"
        lines.append(
            f"| {int(r['shap_rank'])} | {r['feature']} | {r['value']:.4f} | "
            f"{r['shap_value']:+.4f} {direction} | {r['global_importance']:.1f} | {int(r['global_rank'])} |"
        )
    lines.append("")

    # global_importanceランキングとの一致度を見る（上位k件のジャッカード類似度）
    jaccard, overlap = jaccard_top_k(df, top_k)

    lines.append(f"## global_importance上位{top_k}件との一致度")
    lines.append("")
    lines.append(f"- 一致した特徴量: {sorted(overlap)}")
    lines.append(f"- Jaccard類似度: {jaccard:.2f}（1.0で完全一致、低いほど「モデル全体の重要度」と")
    lines.append(f"  「この1件への実際の寄与」が異なることを意味する）")
    lines.append("")
    if jaccard < 0.7:
        lines.append("→ global_importanceとSHAPのランキングに無視できない差がある。"
                     "global_importanceだけで『なぜこの予測になったか』を説明するのは"
                     "妥当ではないことが、このデータでも裏付けられた。")
    else:
        lines.append("→ 今回のサンプルでは両者のランキングが比較的近い。"
                     "他の銘柄・日付でも同様か、複数件で確認するとよい。")
    lines.append("")

    # Jaccard類似度は「顔ぶれの重複」しか見ないため、全特徴量に対するSpearman順位相関も出す。
    # 顔ぶれが同じでも、序列が大きく入れ替わっているケースを見逃さないようにする。
    rank_corr = result["spearman_corr"]
    p_value = result["spearman_pvalue"]

    lines.append("## global_importanceとSHAPの順位相関（Spearman）")
    lines.append("")
    lines.append(f"- 順位相関係数: {rank_corr:.3f}（p値: {p_value:.3f}）")
    lines.append("- 1.0に近いほど「全特徴量の序列がほぼ同じ」、0に近い・負であるほど")
    lines.append("  「モデル全体の重要度の序列と、この1件での効き方の序列が別物」であることを意味する。")
    lines.append("  Jaccard類似度（上位k件の顔ぶれの重複）とあわせて見ることで、")
    lines.append("  「同じ顔ぶれでも順位が入れ替わっている」ようなケースも検知できる。")
    lines.append("")
    if rank_corr >= 0.7:
        lines.append("→ 全特徴量で見ても序列は比較的近い。")
    elif rank_corr >= 0.3:
        lines.append("→ 序列に緩やかな相関はあるが、無視できない入れ替わりがある。")
    else:
        lines.append("→ 序列はほぼ無相関、またはこの1件で大きく入れ替わっている。"
                     "「モデル全体で重要」と「この予測で効いた」は別物として扱うべき。")
    lines.append("")

    # 顔ぶれが上位k件で重複していても、順位そのものが大きく動いた特徴量を明示する
    df["abs_rank_shift"] = df["rank_shift"].abs()
    top_movers = df.sort_values("abs_rank_shift", ascending=False).head(top_k)

    lines.append(f"## 顕著な順位変動 上位{top_k}件（global_rank と shap_rank の差が大きい順）")
    lines.append("")
    lines.append("| feature | global_rank | shap_rank | 変動 | 解釈 |")
    lines.append("|---|---|---|---|---|")
    for _, r in top_movers.iterrows():
        shift = int(r["rank_shift"])
        if shift > 0:
            interp = f"全体では{int(r['global_rank'])}位だが、この1件では{int(r['shap_rank'])}位まで浮上"
        elif shift < 0:
            interp = f"全体では{int(r['global_rank'])}位の重要特徴量だが、この1件ではほぼ効いていない（{int(r['shap_rank'])}位）"
        else:
            interp = "順位変動なし"
        lines.append(
            f"| {r['feature']} | {int(r['global_rank'])} | {int(r['shap_rank'])} | "
            f"{shift:+d} | {interp} |"
        )
    lines.append("")

    report_text = "\n".join(lines)
    print(report_text)

    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(report_text, encoding="utf-8")
        log.info("report written to %s", report_out)


# ──────────────────────────────────────────────────────────────
# batch: 複数JSONに対する一致度分布の集計（shap_consistency_batch_plan.md 相当）
# ──────────────────────────────────────────────────────────────
def confidence_bucket(confidence: float) -> str:
    lower = int(confidence * 10) / 10
    return f"{lower:.1f}-{lower+0.1:.1f}"


def run_batch(
    model_dir: Path,
    input_dir: Path,
    top_k: int,
    output_dir: Path,
) -> None:
    json_paths = sorted(input_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"{input_dir} にJSONファイルが見つかりません。")
    log.info("found %d json files in %s", len(json_paths), input_dir)

    # 実行のたびにタイムスタンプ付きサブフォルダを作り、CSV・レポート・画像を1か所にまとめて出力する
    run_dir = output_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_csv = run_dir / "records.csv"
    report_out = run_dir / "summary.md"
    plot_out = run_dir / "histogram.png"
    log.info("batch output directory: %s", run_dir)

    metadata = load_metadata(model_dir)
    models = load_fold_models(model_dir)  # 全ファイルで使い回す（毎回ロードし直さない）

    records = []
    errors = []
    for path in json_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                detail = json.load(f)
            result = compute_single_comparison(models, metadata, detail)
            jaccard, overlap = jaccard_top_k(result["df"], top_k)
            records.append({
                "file": path.name,
                "ticker": result["ticker"],
                "analyzed_at": result["analyzed_at"],
                "predicted_label": result["predicted_label"],
                "confidence": result["confidence"],
                "confidence_bucket": confidence_bucket(result["confidence"]),
                "jaccard": jaccard,
                "spearman_corr": result["spearman_corr"],
                "spearman_pvalue": result["spearman_pvalue"],
                "top_shap_feature": result["top_shap_feature"],
                "top_global_feature": result["top_global_feature"],
                "shap_elapsed_sec": result["shap_elapsed_sec"],
            })
        except Exception as e:
            log.warning("failed to process %s: %s", path.name, e)
            errors.append({"file": path.name, "error": str(e)})

    if not records:
        raise RuntimeError("全ファイルの処理に失敗しました。エラー内容を確認してください。")

    df = pd.DataFrame(records)

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        log.info("records written to %s", output_csv)

    lines = []
    lines.append("# SHAP vs global_importance 一致度分布 集計レポート")
    lines.append("")
    lines.append(f"- 対象ファイル数: {len(json_paths)}（成功: {len(records)}, 失敗: {len(errors)}）")
    lines.append(f"- 入力ディレクトリ: {input_dir}")
    lines.append("")

    if errors:
        lines.append("## 処理に失敗したファイル")
        lines.append("")
        for e in errors:
            lines.append(f"- {e['file']}: {e['error']}")
        lines.append("")

    # 4a. 基本統計
    lines.append("## 4a. 基本統計")
    lines.append("")
    lines.append("| 指標 | 平均 | 標準偏差 | 最小 | 中央値 | 最大 |")
    lines.append("|---|---|---|---|---|---|")
    for col in ["jaccard", "spearman_corr"]:
        s = df[col]
        lines.append(
            f"| {col} | {s.mean():.3f} | {s.std():.3f} | {s.min():.3f} | {s.median():.3f} | {s.max():.3f} |"
        )
    lines.append("")

    # 簡易テキストヒストグラム（spearman_corrを-1〜1で10分割）
    lines.append("### spearman_corrの簡易分布（テキストヒストグラム）")
    lines.append("")
    bins = np.linspace(-1, 1, 11)
    hist, edges = np.histogram(df["spearman_corr"], bins=bins)
    for count, lo, hi in zip(hist, edges[:-1], edges[1:]):
        bar = "█" * count
        lines.append(f"[{lo:+.1f}, {hi:+.1f}): {bar} ({count})")
    lines.append("")

    # 4b. セグメント別分析
    lines.append("## 4b. セグメント別分析")
    lines.append("")
    lines.append("### predicted_label別")
    lines.append("")
    lines.append("| predicted_label | 件数 | jaccard平均 | spearman_corr平均 |")
    lines.append("|---|---|---|---|")
    for label, g in df.groupby("predicted_label"):
        lines.append(f"| {label} | {len(g)} | {g['jaccard'].mean():.3f} | {g['spearman_corr'].mean():.3f} |")
    lines.append("")

    lines.append("### confidence帯別")
    lines.append("")
    lines.append("| confidence帯 | 件数 | jaccard平均 | spearman_corr平均 |")
    lines.append("|---|---|---|---|")
    for bucket, g in sorted(df.groupby("confidence_bucket"), key=lambda x: x[0]):
        lines.append(f"| {bucket} | {len(g)} | {g['jaccard'].mean():.3f} | {g['spearman_corr'].mean():.3f} |")
    lines.append("")
    if df["confidence"].corr(df["spearman_corr"]) is not None and len(df) >= 3:
        conf_corr = df["confidence"].corr(df["spearman_corr"])
        lines.append(f"- confidenceとspearman_corrの相関係数: {conf_corr:.3f}"
                     f"（正なら「確信度が高いほどglobal_importanceとの一致度も高い」傾向を示唆）")
        lines.append("")

    # 4c. 外れ値の一覧
    lines.append("## 4c. 一致度が特に低いケース（下位5件）")
    lines.append("")
    lines.append("| file | ticker | analyzed_at | predicted_label | confidence | jaccard | spearman_corr |")
    lines.append("|---|---|---|---|---|---|---|")
    worst = df.sort_values("spearman_corr").head(5)
    for _, r in worst.iterrows():
        lines.append(
            f"| {r['file']} | {r['ticker']} | {r['analyzed_at']} | {r['predicted_label']} | "
            f"{r['confidence']*100:.1f}% | {r['jaccard']:.2f} | {r['spearman_corr']:+.3f} |"
        )
    lines.append("")

    # 全体としての結論の目安
    lines.append("## まとめの目安")
    lines.append("")
    std = df["spearman_corr"].std()
    if std >= 0.3:
        lines.append(f"→ spearman_corrの標準偏差が{std:.3f}と大きく、一致度自体がケースによって大きく"
                     f"振れている。global_importanceは参考程度に留め、個別分析では毎回SHAPを"
                     f"計算する現行方針を支持する結果。")
    else:
        lines.append(f"→ spearman_corrの標準偏差は{std:.3f}とそれほど大きくない。"
                     f"条件（confidence帯等）によってはglobal_importanceの説明で代用できる"
                     f"余地がないか、4bのセグメント別分析を参考に検討する。")
    lines.append("")

    report_text = "\n".join(lines)
    print(report_text)

    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(report_text, encoding="utf-8")
        log.info("report written to %s", report_out)

    if plot_out is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(df["jaccard"], bins=10, range=(0, 1))
            axes[0].set_title("Jaccard similarity")
            axes[0].set_xlabel("jaccard")
            axes[1].hist(df["spearman_corr"], bins=10, range=(-1, 1))
            axes[1].set_title("Spearman rank correlation")
            axes[1].set_xlabel("spearman_corr")
            fig.tight_layout()

            plot_out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_out, dpi=120)
            log.info("histogram written to %s", plot_out)
        except ImportError:
            log.warning("matplotlibが見つからないため、ヒストグラム画像の出力をスキップしました。")

    log.info("=" * 60)
    log.info("batch出力一式: %s", run_dir)
    log.info("  - %s", output_csv.name)
    log.info("  - %s", report_out.name)
    log.info("  - %s", plot_out.name)
    log.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP値によるper-instance特徴量寄与の実験")
    subparsers = parser.add_subparsers(dest="command", required=True)

    selftest_parser = subparsers.add_parser("selftest", help="合成データでの妥当性確認（4a）")
    selftest_parser.add_argument("--n-folds", type=int, default=3)
    selftest_parser.add_argument("--n-samples", type=int, default=3000)
    selftest_parser.add_argument("--seed", type=int, default=0)

    analyze_parser = subparsers.add_parser("analyze", help="実データでの比較（4b）")
    analyze_parser.add_argument("--model-dir", type=Path, required=True)
    analyze_parser.add_argument("--input-json", type=Path, required=True,
                                 help="analyze_ticker.py が出力した銘柄詳細JSON")
    analyze_parser.add_argument("--top-k", type=int, default=10)
    analyze_parser.add_argument("--report-out", type=Path, default=None)

    batch_parser = subparsers.add_parser("batch", help="複数JSONに対する一致度分布の集計")
    batch_parser.add_argument("--model-dir", type=Path, required=True)
    batch_parser.add_argument("--input-dir", type=Path, required=True,
                               help="analyze_ticker.py が出力したJSON群が入ったディレクトリ")
    batch_parser.add_argument("--top-k", type=int, default=10)
    batch_parser.add_argument("--output-dir", type=Path, default=Path("batch"),
                               help="出力先ディレクトリ（この下に run_YYYYMMDD_HHMMSS/ を作り、"
                                    "records.csv・summary.md・histogram.pngを一括で出力する。デフォルト: batch/）")

    args = parser.parse_args()

    if args.command == "selftest":
        passed = run_selftest(n_folds=args.n_folds, n_samples=args.n_samples, seed=args.seed)
        raise SystemExit(0 if passed else 1)
    elif args.command == "analyze":
        run_analyze(
            model_dir=args.model_dir,
            input_json=args.input_json,
            top_k=args.top_k,
            report_out=args.report_out,
        )
    elif args.command == "batch":
        run_batch(
            model_dir=args.model_dir,
            input_dir=args.input_dir,
            top_k=args.top_k,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()