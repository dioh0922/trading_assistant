"""
AIトレードシステム ステップ7: リアルタイム エントリー判断レポート

step6_dataset.csv（存在しない場合は step5_dataset.csv）の最終行を取り出し、
未来情報（tb_label / tb_return / forward_max_drawdown 等）を除いた上で
「今日エントリーすべきか」を人間が読めるレポートとして出力する。

新たなモデル学習や計算は一切行わない。
ステップ1〜5（またはステップ6）のパイプライン実行後に動作することが前提。

実行エントリーポイントは project直下の main.py。
  python main.py --code 7701                # 通常パイプライン実行 + ステップ7出力
  python main.py --code 7701 --today        # 保存済みCSVからレポートのみ再生成
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCE_DIR = BASE_DIR / "resource"

# ─────────────────────────────────────────────────────────────────
# 未来情報列（エントリー時点では知れない情報）
# ─────────────────────────────────────────────────────────────────
_FUTURE_COLS = {
    "tb_label",
    "tb_return",
    "tb_barrier",
    "tb_days_to_touch",
    "forward_max_drawdown",
    "avoid_entry_flag",
    # step6 が追加するフラグ類も未来情報ではないが、
    # None や NaN になり得る行の安全のためここでは除外せずに扱う
}


# ─────────────────────────────────────────────────────────────────
# 公開 API
# ─────────────────────────────────────────────────────────────────

def load_source_df(output_dir: Path) -> tuple[pd.DataFrame, str]:
    """
    output_dir から最新の判断元 DataFrame と使用ファイル名を返す。

    優先順位: step6_dataset.csv > step5_dataset.csv

    Returns
    -------
    (df, source_label)
    """
    for fname in ("step6_dataset.csv", "step5_dataset.csv"):
        csv_path = output_dir / fname
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            return df, fname
    raise FileNotFoundError(
        f"step6_dataset.csv / step5_dataset.csv のどちらも見つかりません: {output_dir}\n"
        "先に python main.py --code XXXX を実行してください。"
    )


def build_entry_report(source_df: pd.DataFrame,
                       barrier_mode: str,
                       tp: float,
                       sl: float,
                       source_label: str = "") -> dict:
    """
    DataFrame の最終行からエントリー判断レポートを生成する。

    Parameters
    ----------
    source_df    : step5_dataset.csv または step6_dataset.csv を読み込んだ DataFrame
    barrier_mode : "atr" / "fixed_pct"
    tp           : 利確設定値（ATRモード=倍率 / 固定%モード=割合）
    sl           : 損切り設定値
    source_label : 使用したファイル名（ログ表示用）

    Returns
    -------
    dict with keys:
        date         : 判断日付
        decision     : "エントリー" / "見送り" / "判断不可"
        signal       : assist_signal の値
        reasons      : 見送り理由リスト（エントリーの場合は空）
        metrics      : 判断根拠の辞書（表示用）
        summary_text : コンソール出力・ファイル保存用テキスト
        source       : 参照元ファイル名
    """
    today_date = source_df.index[-1].date()
    row = source_df.iloc[-1].copy()

    # 未来情報列をマスク（参照自体は行わない）
    for col in _FUTURE_COLS:
        if col in row.index:
            row = row.drop(col)

    # ─── 判断結果の確定 ───────────────────────────────────────────
    final_decision = source_df.iloc[-1].get("final_decision", None)

    if final_decision is None or pd.isna(final_decision):
        # step6 が未実行 → step5 の assist_signal で代替判断
        signal = row.get("assist_signal", "不明")
        if signal == "強気":
            decision = "エントリー候補"   # step6フィルタ未適用のため「候補」扱い
        else:
            decision = "見送り"
        step6_available = False
    else:
        step6_available = True
        decision = "エントリー" if int(final_decision) == 1 else "見送り"
        signal = row.get("assist_signal", "不明")

    # ─── 見送り理由の構築 ─────────────────────────────────────────
    reasons = _build_rejection_reasons(row, signal, step6_available)
    if decision == "エントリー":
        reasons = []

    # ─── 指標の抽出 ──────────────────────────────────────────────
    metrics = _extract_key_metrics(row, barrier_mode, tp, sl, step6_available)

    # ─── テキスト生成 ─────────────────────────────────────────────
    summary_text = _format_summary(
        today_date, decision, signal, reasons, metrics,
        barrier_mode, step6_available, source_label
    )

    return {
        "date":         today_date,
        "decision":     decision,
        "signal":       signal,
        "reasons":      reasons,
        "metrics":      metrics,
        "summary_text": summary_text,
        "source":       source_label,
    }


def save_report(report: dict, output_dir: Path) -> Path:
    """レポートを output_dir/today_signal.txt として保存する。"""
    output_path = output_dir / "today_signal.txt"
    output_path.write_text(report["summary_text"], encoding="utf-8")
    return output_path


# ─────────────────────────────────────────────────────────────────
# 内部ヘルパー
# ─────────────────────────────────────────────────────────────────

def _val(row: pd.Series, col: str, default=None):
    """列が存在し、かつ NaN でなければ値を返す。"""
    v = row.get(col, default)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return v


def _build_rejection_reasons(row: pd.Series, signal: str,
                              step6_available: bool) -> list[str]:
    """見送り理由を日本語で列挙する。"""
    reasons: list[str] = []

    if signal == "警戒":
        atr_dev = _val(row, "atr_dev_ma25")
        reasons.append(
            f"過熱シグナル（ATR正規化乖離: "
            f"{f'{atr_dev:.2f}' if atr_dev is not None else '不明'}）が発生しています。"
            "急騰局面のため見送り。"
        )
        return reasons

    if signal == "中立":
        reasons.append(
            "強気・警戒どちらの条件も成立しておらず、シグナルなし（中立）です。"
        )
        if not _val(row, "rsi_reversal_flag", 0):
            rsi = _val(row, "rsi14")
            reasons.append(
                f"RSI反転条件未成立: RSI={f'{rsi:.1f}' if rsi is not None else '不明'}。"
                "売られすぎ圏（35以下）からの切り返しか、"
                "強気ダイバージェンスが確認できません。"
            )
        if not _val(row, "deviation_shrink_flag", 0):
            z = _val(row, "dev_ma25_zscore")
            reasons.append(
                f"MA乖離縮小条件未成立: MA乖離Zスコア="
                f"{f'{z:.2f}' if z is not None else '不明'}。"
                "移動平均からの下方乖離が縮小していません。"
            )
        if not _val(row, "trending_bullish_flag", 0):
            wt  = _val(row, "weekly_trend", 0)
            rsi = _val(row, "rsi14")
            vr  = _val(row, "volume_ratio")
            reasons.append(
                f"順張り条件未成立: "
                f"週足={'上昇' if wt == 1 else '下降'}、"
                f"RSI={f'{rsi:.1f}' if rsi is not None else '不明'}（≥50必要）、"
                f"出来高比率={f'{vr:.2f}' if vr is not None else '不明'}（≥1.2必要）。"
                "週足上昇トレンド × RSI50以上 × 出来高増加の全条件を満たしていません。"
            )
        return reasons

    # signal == "強気" だが最終的に見送り → step6 フィルタが除外
    if signal == "強気" and step6_available:
        filtered_vol = _val(row, "filtered_by_volatility", 0)
        filtered_dd  = _val(row, "filtered_by_drawdown",  0)

        if filtered_vol:
            atr_p = _val(row, "atr_percentile")
            z     = _val(row, "dev_ma25_zscore")
            if atr_p is not None and atr_p >= 0.90:
                reasons.append(
                    f"ATR過熱フィルタ: ATRパーセンタイル={atr_p:.1%}（過去250日の上位10%）。"
                    "ボラティリティが異常に高い局面のため見送り。"
                )
            if z is not None and z <= -2.5:
                reasons.append(
                    f"下方乖離フィルタ: MA乖離Zスコア={z:.2f}（≤-2.5）。"
                    "下げすぎ極値での逆張りは底抜けリスクがあるため見送り。"
                )

        if filtered_dd:
            dd_p = _val(row, "drawdown_prob")
            reasons.append(
                f"ドローダウン予測フィルタ: "
                f"DD発生確率={f'{dd_p:.1%}' if dd_p is not None else '不明'}。"
                "閾値を超えているため見送り。"
            )

        if not reasons:
            reasons.append(
                "強気シグナルですがフィルタ条件（ボラティリティ / ドローダウン）"
                "のいずれかに引っかかりました。"
            )

    return reasons


def _extract_key_metrics(row: pd.Series, barrier_mode: str,
                          tp: float, sl: float,
                          step6_available: bool) -> dict:
    """判断根拠として表示する指標を抽出する。"""
    # ATRパーセンタイル
    atr_p = _val(row, "atr_percentile")
    if atr_p is not None:
        atr_p_str = f"{atr_p:.1%}  {'⚠️ 過熱' if atr_p >= 0.90 else '✅ 正常'}"
    else:
        atr_p_str = "（データなし: step1を再実行してください）"

    # MA乖離Zスコア
    z = _val(row, "dev_ma25_zscore")
    z_str = (
        f"{z:.2f}  {'⚠️ 底抜けリスク' if z <= -2.5 else '✅ 正常'}"
        if z is not None else "不明"
    )

    # DD予測確率（step6がある場合のみ意味を持つ）
    dd_p = _val(row, "drawdown_prob")
    if not step6_available:
        dd_p_str = "（step6未実行）"
    elif dd_p is not None:
        dd_p_str = f"{dd_p:.1%}"
    else:
        dd_p_str = "不明"

    # 週足トレンド
    wt = _val(row, "weekly_trend", 0)
    wt_str = "上昇 ▲" if wt == 1 else "下降 ▼"

    # RSI
    rsi = _val(row, "rsi14")
    rsi_str = f"{rsi:.1f}" if rsi is not None else "不明"

    # volume_ratio
    vr = _val(row, "volume_ratio")
    vr_str = f"{vr:.2f}" if vr is not None else "（データなし: step1を再実行してください）"

    # 利確・損切り表示
    if barrier_mode == "fixed_pct":
        tp_str = f"+{tp:.0%}"
        sl_str = f"-{sl:.0%}"
    else:
        tp_str = f"ATR×{tp}"
        sl_str = f"ATR×{sl}"

    return {
        # シグナル
        "シグナル":               str(_val(row, "assist_signal", "不明")),
        "RSI反転フラグ":          "✅" if _val(row, "rsi_reversal_flag", 0) else "❌",
        "MA乖離縮小フラグ":       "✅" if _val(row, "deviation_shrink_flag", 0) else "❌",
        "順張りフラグ":           "✅" if _val(row, "trending_bullish_flag", 0) else "❌",
        "過熱警戒フラグ":         "⚠️" if _val(row, "overheat_warning_flag", 0) else "-",
        # 価格
        "終値":                   f"{_val(row, 'close', 0):.0f} 円",
        "RSI14":                  rsi_str,
        "出来高比率(20日)":       vr_str,
        # フィルタ根拠
        "ATRパーセンタイル":      atr_p_str,
        "MA乖離Zスコア":          z_str,
        "ATR正規化乖離":          (
                                    f"{_val(row, 'atr_dev_ma25'):.2f}"
                                    if _val(row, "atr_dev_ma25") is not None else "不明"
                                ),
        "週足トレンド":           wt_str,
        # step6 関連
        "DD予測確率":             dd_p_str,
        "step6フィルタ":          "適用済み" if step6_available else "未適用（step6未実行）",
        # バリア設定（参考）
        "利確目標":               tp_str,
        "損切りライン":           sl_str,
    }


def _format_summary(today_date: date, decision: str, signal: str,
                    reasons: list[str], metrics: dict,
                    barrier_mode: str, step6_available: bool,
                    source_label: str) -> str:
    """コンソール出力・ファイル保存用のテキストを生成する。"""
    W = 62
    BORDER = "=" * W
    THIN   = "─" * W

    DECISION_ICON = {
        "エントリー":    "✅  エントリー",
        "エントリー候補": "🔶  エントリー候補（step6フィルタ未適用）",
        "見送り":        "⛔  見送り",
        "判断不可":      "⚠️   判断不可",
    }
    decision_label = DECISION_ICON.get(decision, decision)

    lines = [
        BORDER,
        f"  ステップ7: エントリー判断レポート",
        f"  判断日付 : {today_date}",
        f"  参照元   : {source_label}",
        f"  バリア設定: {metrics['利確目標']} 利確 / {metrics['損切りライン']} 損切り",
        BORDER,
        "",
        f"  【最終判断】  {decision_label}",
        "",
        THIN,
        "  ■ 判断根拠",
        THIN,
    ]

    # グループ別に整形
    groups = [
        ("シグナル", ["シグナル", "RSI反転フラグ", "MA乖離縮小フラグ", "順張りフラグ", "過熱警戒フラグ"]),
        ("価格・テクニカル", ["終値", "RSI14", "出来高比率(20日)", "週足トレンド", "ATR正規化乖離"]),
        ("フィルタ評価", ["ATRパーセンタイル", "MA乖離Zスコア", "DD予測確率", "step6フィルタ"]),
        ("バリア設定", ["利確目標", "損切りライン"]),
    ]
    for group_name, keys in groups:
        lines.append(f"  [{group_name}]")
        for k in keys:
            v = metrics.get(k, "-")
            lines.append(f"    {k:<18}: {v}")
        lines.append("")

    # 見送り理由
    if reasons:
        lines += [
            THIN,
            "  ■ 見送り理由",
            THIN,
        ]
        for i, r in enumerate(reasons, start=1):
            # 長い文は折り返し（60文字で改行）
            words = r
            indent = "     "
            first  = f"    {i}. "
            chunk = W - len(first)
            lines.append(first + words[:chunk])
            pos = chunk
            while pos < len(words):
                lines.append(indent + words[pos:pos + W - len(indent)])
                pos += W - len(indent)
        lines.append("")

    lines += [
        BORDER,
        "  ※ このレポートは過去データに基づく参考情報です。",
        "    実際の投資判断はご自身の責任で行ってください。",
        BORDER,
    ]
    return "\n".join(lines)
