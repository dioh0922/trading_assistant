"""
aggregate_batch_stats.py
──────────────────────────
`experiment_shap.py batch` を複数回実行して溜まった `batch/run_*/records.csv` を
すべて統合し、以下の2つの傾向を正式に検定する。

  1. predicted_label（upper vs lower）でSHAP一致度（spearman_corr）に差があるか
  2. confidenceとSHAP一致度（spearman_corr）に相関があるか

【重要】単純にrecords.csvを縦に結合して検定すると、同じ銘柄が複数回登場する
（例：6098が7/27, 7/28, 7/31, 8/2...と何度も出てくる）ため、実質的な独立サンプル数を
過大評価してしまう（疑似反復 / pseudo-replication の問題）。これを避けるため、
「生データをそのままプールした素朴な検定」と「銘柄ごとに集約してから検定する方法」の
両方を並べて出力し、結論が変わるかどうかを確認できるようにする。

使い方:
    python aggregate_batch_stats.py --batch-dir batch --report-out reports/aggregate_stats.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

logging.basicConfig(
  level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("aggregate_batch_stats")


# ──────────────────────────────────────────────────────────────
# 1. 複数run分のrecords.csvを集めて統合する
# ──────────────────────────────────────────────────────────────
def load_all_records(batch_dir: Path) -> pd.DataFrame:
  csv_paths = sorted(batch_dir.glob("run_*/records.csv"))
  if not csv_paths:
    raise FileNotFoundError(f"{batch_dir} 配下に run_*/records.csv が見つかりません。")

  log.info("found %d records.csv files under %s", len(csv_paths), batch_dir)

  frames = []
  for path in csv_paths:
    df = pd.read_csv(path)
    df["source_run"] = path.parent.name
    frames.append(df)
  combined = pd.concat(frames, ignore_index=True)

  n_before = len(combined)

  # 同じJSON（同一銘柄・同一analyzed_at・同一file名）が複数のbatch実行に
  # またがって重複して含まれている場合があるため、重複を除去する。
  # 「file」列（元のJSONファイル名）が一意キーとして最も信頼できる。
  dedup_keys = (
    ["ticker", "analyzed_at", "file"]
    if "file" in combined.columns
    else ["ticker", "analyzed_at"]
  )
  combined = combined.drop_duplicates(subset=dedup_keys, keep="first").reset_index(
    drop=True
  )

  n_after = len(combined)
  if n_after < n_before:
    log.info(
      "重複除去: %d件 -> %d件（%d件の重複を削除）",
      n_before,
      n_after,
      n_before - n_after,
    )

  return combined


# ──────────────────────────────────────────────────────────────
# 2a. 素朴な検定（生データをそのままプール。疑似反復のリスクあり）
# ──────────────────────────────────────────────────────────────
def naive_label_comparison(df: pd.DataFrame, lines: list[str]) -> None:
  lines.append("### 素朴な検定（生データそのまま。銘柄の重複を考慮しない）")
  lines.append("")

  groups = {
    label: g["spearman_corr"].to_numpy() for label, g in df.groupby("predicted_label")
  }
  counts = {label: len(v) for label, v in groups.items()}
  lines.append(f"- サンプル内訳: {counts}")

  if (
    "upper" in groups
    and "lower" in groups
    and len(groups["upper"]) >= 2
    and len(groups["lower"]) >= 2
  ):
    stat, p = mannwhitneyu(groups["upper"], groups["lower"], alternative="two-sided")
    lines.append(
      f"- upper vs lower の spearman_corr（Mann-Whitney U検定）: "
      f"U={stat:.1f}, p={p:.4f}"
    )
    lines.append(
      f"  upper平均={groups['upper'].mean():.3f} / lower平均={groups['lower'].mean():.3f}"
    )
    lines.append(
      f"  {'✅ 有意差あり（p<0.05）' if p < 0.05 else '△ 有意差なし（この検定単体では）'}"
    )
  else:
    lines.append("- upper/lowerいずれかのサンプル数が2件未満のため、検定を省略。")
  lines.append("")


def naive_confidence_correlation(df: pd.DataFrame, lines: list[str]) -> None:
  lines.append("### 素朴な相関（生データそのまま）")
  lines.append("")
  if len(df) >= 3:
    r_pearson, p_pearson = pearsonr(df["confidence"], df["spearman_corr"])
    r_spearman, p_spearman = spearmanr(df["confidence"], df["spearman_corr"])
    lines.append(f"- Pearson相関: r={r_pearson:.3f}, p={p_pearson:.4f} (n={len(df)})")
    lines.append(f"- Spearman相関: r={r_spearman:.3f}, p={p_spearman:.4f}")
    lines.append(f"  {'✅ 有意（p<0.05）' if p_pearson < 0.05 else '△ 有意ではない'}")
  else:
    lines.append("- サンプル数が3件未満のため、相関を計算できません。")
  lines.append("")


# ──────────────────────────────────────────────────────────────
# 2b. 銘柄クラスタを考慮した検定
# ──────────────────────────────────────────────────────────────
def clustered_label_comparison(df: pd.DataFrame, lines: list[str]) -> None:
  lines.append(
    "### 銘柄クラスタを考慮した検定（銘柄×predicted_labelごとに平均してから検定）"
  )
  lines.append("")
  lines.append(
    "同一銘柄が複数日にわたって同じ結論を出しやすいこと（自己相関）を考慮し、"
  )
  lines.append(
    "まず「銘柄×predicted_label」単位でspearman_corrを平均し、その平均値どうしを比較する。"
  )
  lines.append("")

  ticker_label_means = (
    df.groupby(["ticker", "predicted_label"])["spearman_corr"]
    .agg(["mean", "count"])
    .reset_index()
  )
  lines.append("| ticker | predicted_label | 平均spearman_corr | 観測日数 |")
  lines.append("|---|---|---|---|")
  for _, r in ticker_label_means.sort_values(["predicted_label", "ticker"]).iterrows():
    lines.append(
      f"| {r['ticker']} | {r['predicted_label']} | {r['mean']:.3f} | {int(r['count'])} |"
    )
  lines.append("")

  groups = {
    label: g["mean"].to_numpy()
    for label, g in ticker_label_means.groupby("predicted_label")
  }
  counts = {label: len(v) for label, v in groups.items()}
  lines.append(f"- 銘柄単位でのサンプル内訳（＝ユニーク銘柄数）: {counts}")

  if (
    "upper" in groups
    and "lower" in groups
    and len(groups["upper"]) >= 2
    and len(groups["lower"]) >= 2
  ):
    stat, p = mannwhitneyu(groups["upper"], groups["lower"], alternative="two-sided")
    lines.append(
      f"- upper vs lower（銘柄単位平均, Mann-Whitney U検定）: U={stat:.1f}, p={p:.4f}"
    )
    lines.append(
      f"  upper平均={groups['upper'].mean():.3f} / lower平均={groups['lower'].mean():.3f}"
    )
    lines.append(
      f"  {'✅ 有意差あり（p<0.05）' if p < 0.05 else '△ 有意差なし（銘柄単位に集約すると弱まる/消える可能性あり）'}"
    )
  else:
    lines.append(
      "- 銘柄単位で見ると、upper/lowerいずれかの銘柄数が2件未満のため、検定を省略。"
      "（＝ばらつきの評価に足るだけの『異なる銘柄』が集まっていない）"
    )
  lines.append("")


def clustered_confidence_correlation(df: pd.DataFrame, lines: list[str]) -> None:
  lines.append("### 銘柄クラスタを考慮した相関（銘柄ごとの相関係数を集計）")
  lines.append("")
  lines.append(
    "観測日数が3日以上ある銘柄についてのみ、銘柄内でconfidenceとspearman_corrの"
  )
  lines.append("相関係数を個別に計算し、その分布（銘柄をまたいだ平均）を見る。")
  lines.append(
    "これにより「特定の1銘柄の観測回数が多いせいで、全体の相関が引きずられる」"
  )
  lines.append("ことを避ける。")
  lines.append("")

  per_ticker_corrs = []
  for ticker, g in df.groupby("ticker"):
    if len(g) >= 3 and g["confidence"].nunique() >= 2:
      r, p = pearsonr(g["confidence"], g["spearman_corr"])
      per_ticker_corrs.append({"ticker": ticker, "n": len(g), "corr": r, "p": p})

  if not per_ticker_corrs:
    lines.append(
      "- 観測日数3日以上の銘柄がないため、銘柄内相関を計算できません。"
      "（データを蓄積してから再実行してください）"
    )
    lines.append("")
    return

  corr_df = pd.DataFrame(per_ticker_corrs)
  lines.append("| ticker | 観測日数 | 銘柄内confidence-spearman_corr相関 |")
  lines.append("|---|---|---|")
  for _, r in corr_df.iterrows():
    lines.append(f"| {r['ticker']} | {int(r['n'])} | {r['corr']:+.3f} |")
  lines.append("")

  mean_corr = corr_df["corr"].mean()
  lines.append(
    f"- 銘柄をまたいだ平均相関係数: {mean_corr:+.3f}（対象銘柄数: {len(corr_df)}）"
  )
  if len(corr_df) >= 2:
    # 銘柄ごとの相関係数を「1サンプル」とみなした簡易的な片側検定（符号検定に近い簡便法）
    positive = (corr_df["corr"] > 0).sum()
    negative = (corr_df["corr"] < 0).sum()
    lines.append(
      f"- 符号の内訳: 正の銘柄 {positive} / 負の銘柄 {negative}"
      f"（全銘柄が同じ符号なら傾向として一貫している可能性が高い）"
    )
  lines.append("")


# ──────────────────────────────────────────────────────────────
# メイン処理
# ──────────────────────────────────────────────────────────────
def run(
  batch_dir: Path, report_out: Path | None, combined_csv_out: Path | None
) -> None:
  df = load_all_records(batch_dir)

  n_rows = len(df)
  n_tickers = df["ticker"].nunique()
  n_runs = df["source_run"].nunique()

  lines = []
  lines.append("# batch実行結果の統合・統計検定レポート")
  lines.append("")
  lines.append(f"- 統合したbatch実行数: {n_runs}")
  lines.append(f"- 重複除去後の総サンプル数（＝JSON件数）: {n_rows}")
  lines.append(f"- ユニーク銘柄数: {n_tickers}")
  lines.append(f"- 1銘柄あたりの平均観測回数: {n_rows / n_tickers:.1f}")
  lines.append("")
  lines.append(
    "**注意**: 以下の「素朴な検定」は同一銘柄の複数日データをすべて独立サンプル"
  )
  lines.append("として扱っており、実際の独立性より過大評価している可能性がある。")
  lines.append("「銘柄クラスタを考慮した検定」の結果とあわせて解釈すること。")
  lines.append("")

  lines.append("## 1. predicted_label（upper vs lower）とSHAP一致度の関係")
  lines.append("")
  naive_label_comparison(df, lines)
  clustered_label_comparison(df, lines)

  lines.append("## 2. confidenceとSHAP一致度の相関")
  lines.append("")
  naive_confidence_correlation(df, lines)
  clustered_confidence_correlation(df, lines)

  report_text = "\n".join(lines)
  print(report_text)

  if combined_csv_out is not None:
    combined_csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(combined_csv_out, index=False, encoding="utf-8-sig")
    log.info("combined records written to %s", combined_csv_out)

  if report_out is not None:
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(report_text, encoding="utf-8")
    log.info("report written to %s", report_out)


def main() -> None:
  parser = argparse.ArgumentParser(description="複数batch実行結果の統合・統計検定")
  parser.add_argument(
    "--batch-dir",
    type=Path,
    default=Path("batch"),
    help="experiment_shap.py batch の出力先ディレクトリ（run_*/records.csv を探す）",
  )
  parser.add_argument("--report-out", type=Path, default=None)
  parser.add_argument(
    "--combined-csv-out",
    type=Path,
    default=None,
    help="重複除去後の統合済みrecordsを保存するCSVパス",
  )
  args = parser.parse_args()

  run(
    batch_dir=args.batch_dir,
    report_out=args.report_out,
    combined_csv_out=args.combined_csv_out,
  )


if __name__ == "__main__":
  main()
