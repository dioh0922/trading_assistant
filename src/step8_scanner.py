"""
AIトレードシステム ステップ8: 一括スクリーニング

resource/ にある全ての4桁コードXLSX（XXXX.xlsx）を処理し、
今日のエントリー判断を一覧化したスクリーニングレポートを出力する。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import re
import sys
import traceback

import pandas as pd

from src.step1_feature_engineering import build_step1_dataset, RESOURCE_DIR
from src.step2_domain_features import build_step2_dataset
from src.step3_labeling import build_step3_dataset
from src.step4_model import build_step4_results, save_step4_results, select_feature_columns
from src.step5_assist_signal import build_step5_dataset
from src.step6_filter import train_drawdown_model, apply_final_filter, evaluate_final_performance
from src.step7_entry_signal import build_entry_report
from src.visualize import plot_step1, plot_step2, plot_step3, plot_step4, plot_step5


def collect_xlsx_codes(resource_dir: Path = RESOURCE_DIR) -> list[str]:
    codes = []
    for f in sorted(resource_dir.glob("*.xlsx")):
        if re.match(r"^\d{4}$", f.stem):
            codes.append(f.stem)
    return codes


def is_cache_valid(csv_path: Path, max_days: int = 7) -> bool:
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if len(df) == 0:
            return False
        last_date = df.index[-1].date()
        delta = (date.today() - last_date).days
        return delta <= max_days
    except Exception:
        return False


def run_pipeline_for_code(
    code: str,
    barrier_mode: str,
    tp: float,
    sl: float,
    holding_period: int | None = None,
    drawdown_threshold: float = 0.03,
    step6_enabled: bool = True,
    dd_prob_limit: float = 1.0,
    skip_charts: bool = True,
) -> pd.DataFrame:
    input_path = RESOURCE_DIR / f"{code}.xlsx"
    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_path}")

    output_dir = RESOURCE_DIR / code / barrier_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    step1_output_path = output_dir / "step1_dataset.csv"
    step2_output_path = output_dir / "step2_dataset.csv"
    step3_output_path = output_dir / "step3_dataset.csv"
    step5_output_path = output_dir / "step5_dataset.csv"

    # --- ステップ1 ---
    step1_df = build_step1_dataset(str(input_path))
    step1_df.to_csv(step1_output_path)
    if not skip_charts:
        plot_step1(step1_df, output_dir / "step1_chart.png")

    # --- ステップ2 ---
    step2_df = build_step2_dataset(step1_df)
    step2_df.to_csv(step2_output_path)
    if not skip_charts:
        plot_step2(step2_df, output_dir / "step2_chart.png")

    # --- ステップ3 ---
    if barrier_mode == "fixed_pct":
        step3_df = build_step3_dataset(
            step2_df,
            holding_period=holding_period,
            tp_pct=tp,
            sl_pct=sl,
            drawdown_threshold=drawdown_threshold,
        )
    else:
        step3_df = build_step3_dataset(
            step2_df,
            holding_period=holding_period,
            tp_atr_mult=tp,
            sl_atr_mult=sl,
            drawdown_threshold=drawdown_threshold,
        )
    step3_df.to_csv(step3_output_path)
    if not skip_charts:
        plot_step3(step3_df, output_dir / "step3_chart.png")

    # --- ステップ4 ---
    step4_results = build_step4_results(step3_df)
    save_step4_results(
        step4_results,
        fold_metrics_path=output_dir / "step4_fold_metrics.csv",
        feature_importance_path=output_dir / "step4_feature_importance.csv",
        oos_predictions_path=output_dir / "step4_oos_predictions.csv",
        model_path=output_dir / "step4_model.pkl",
    )
    if not skip_charts:
        plot_step4(
            step4_results["fold_metrics"],
            step4_results["feature_importance"],
            step4_results["oos_predictions"],
            step3_df,
            output_dir / "step4_chart.png",
        )

    # --- ステップ5 ---
    step5_df = build_step5_dataset(step3_df)
    step5_df.to_csv(step5_output_path)
    if not skip_charts:
        plot_step5(step5_df, output_dir / "step5_chart.png")

    # --- ステップ6 ---
    if step6_enabled:
        ml_enabled = dd_prob_limit < 1.0
        feature_cols = select_feature_columns(step3_df)
        dd_results = train_drawdown_model(step3_df, feature_cols)
        dd_pred_proba = dd_results["oos_predictions"]["dd_proba"]
        step6_df = apply_final_filter(
            step5_df,
            dd_pred_proba=dd_pred_proba,
            dd_threshold=dd_prob_limit,
        )
        step6_output_path = output_dir / "step6_dataset.csv"
        step6_df.to_csv(step6_output_path)
        return step6_df
    else:
        return step5_df


def scan_all(
    barrier_mode: str = "atr",
    tp: float = 2.0,
    sl: float = 1.5,
    holding_period: int | None = None,
    drawdown_threshold: float = 0.03,
    step6_enabled: bool = True,
    dd_prob_limit: float = 1.0,
    force_update: bool = False,
) -> list[dict]:
    codes = collect_xlsx_codes(RESOURCE_DIR)
    if not codes:
        print("[Step8] resource/ に4桁コードのXLSXファイルが見つかりませんでした。")
        return []

    print(f"[Step8] {len(codes)}銘柄を処理します: {codes}")
    results: list[dict] = []

    for code in codes:
        print(f"\n{'─' * 40}")
        print(f"[Step8] 処理中: {code}.xlsx")

        try:
            output_dir = RESOURCE_DIR / code / barrier_mode
            step6_csv = output_dir / "step6_dataset.csv"

            use_cache = not force_update and is_cache_valid(step6_csv)

            if use_cache:
                print(f"  → キャッシュ有効: step6_dataset.csv を使用します。")
                source_df = pd.read_csv(step6_csv, index_col=0, parse_dates=True)
            else:
                print(f"  → パイプラインを実行します（グラフ出力なし）...")
                source_df = run_pipeline_for_code(
                    code=code,
                    barrier_mode=barrier_mode,
                    tp=tp,
                    sl=sl,
                    holding_period=holding_period,
                    drawdown_threshold=drawdown_threshold,
                    step6_enabled=step6_enabled,
                    dd_prob_limit=dd_prob_limit,
                    skip_charts=True,
                )

            report = build_entry_report(source_df, barrier_mode, tp, sl)
            report["code"] = code
            results.append(report)

        except Exception as e:
            print(f"  ⚠️  {code} の処理中にエラー: {e}")
            traceback.print_exc()
            results.append({
                "code": code,
                "date": date.today(),
                "decision": "エラー",
                "signal": "不明",
                "reasons": [str(e)],
                "metrics": {},
                "summary_text": "",
            })

    return results


def build_scan_report(results: list[dict]) -> str:
    today = date.today()
    BORDER = "=" * 70
    THIN = "─" * 70

    entry_candidates = [r for r in results if r["decision"] == "エントリー"]
    skipped = [r for r in results if r["decision"] == "見送り"]
    errors = [r for r in results if r["decision"] == "エラー"]
    unknown = [r for r in results if r["decision"] == "判断不可"]

    lines = [
        BORDER,
        f"  ステップ8: 一括スクリーニングレポート",
        f"  スキャン日: {today}  対象銘柄数: {len(results)}",
        BORDER,
        "",
        f"  ✅ エントリー候補: {len(entry_candidates)}銘柄",
        f"  ⛔ 見送り:        {len(skipped)}銘柄",
        f"  ⚠️  判断不可:       {len(unknown)}銘柄",
        f"  ❌ エラー:         {len(errors)}銘柄",
        "",
    ]

    if entry_candidates:
        lines += [THIN, "  【エントリー候補】", THIN]
        for r in entry_candidates:
            m = r.get("metrics", {})
            close = m.get("終値 (円)", m.get("終値", "-"))
            lines.append(
                f"  ✅ {r['code']}  |  "
                f"シグナル: {r['signal']}  |  "
                f"終値: {close}円  |  "
                f"RSI: {m.get('RSI14', '-')}  |  "
                f"週足: {m.get('週足トレンド', '-')}  |  "
                f"ATRパーセンタイル: {m.get('ATRパーセンタイル', '-')}"
            )
    else:
        lines += [THIN, "  【エントリー候補】 なし", THIN]

    lines += ["", THIN, "  【見送り一覧】", THIN]
    for r in skipped:
        reason_head = r["reasons"][0][:50] + "…" if r.get("reasons") else "-"
        lines.append(f"  ⛔ {r['code']}  |  シグナル: {r['signal']}  |  理由: {reason_head}")

    if unknown or errors:
        lines += ["", THIN, "  【判断不可・エラー】", THIN]
        for r in unknown + errors:
            lines.append(f"  ⚠️  {r['code']}  |  {r['decision']}: {r['reasons'][0][:60] if r.get('reasons') else '-'}")

    lines += [
        "",
        BORDER,
        "  ※ 各銘柄の詳細は today_signal.txt または --code XXXX で確認できます。",
        "  ※ このレポートは参考情報です。投資判断はご自身の責任で行ってください。",
        BORDER,
    ]

    return "\n".join(lines)


def save_scan_report(report_text: str, barrier_mode: str) -> Path:
    output_path = RESOURCE_DIR / f"scan_report_{barrier_mode}.txt"
    output_path.write_text(report_text, encoding="utf-8")
    return output_path
