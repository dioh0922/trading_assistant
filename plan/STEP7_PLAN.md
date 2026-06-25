# ステップ7: リアルタイム エントリー判断レポート 設計プラン

**前提**: ステップ1〜6のパイプライン実行後に `step6_dataset.csv` が存在している状態で動作する。

---

## 1. 背景と課題

### 1-1. ステップ1〜6が解決していないこと

ステップ1〜6のパイプラインは**バックテスト**として設計されており、
全ての出力は「その日にエントリーした場合、実際にどうなったか（`tb_label`, `tb_return`）」を
**未来情報込みで**記録したものです。

```
step6_dataset.csv の最終行（2026-06-05）:
  assist_signal  : 中立
  final_decision : 0
  tb_label       : 0         ← 未来情報（実運用では知ることができない）
  tb_return      : -3.0%     ← 未来情報
  tb_barrier     : stop_loss ← 未来情報
```

実際のトレード場面で必要なのは「**今日のデータを見て、今日エントリーすべきか**」という
判断と、その理由を人間が読める形で出力する機能です。

### 1-2. ステップ7が解決すること

```
ステップ7 = 「今日のシグナルを取り出し、未来情報を除いて、
              人間が読めるレポートとして出力する」
```

新たなモデルや計算は行わず、**既存のパイプライン出力を最大限活用する**設計とします。

---

## 2. アーキテクチャ方針

### 2-1. 「今日」の定義

`step6_dataset.csv` の最終行 = 最も新しいデータ日付 = 今日のシグナルとして扱います。

```python
df = pd.read_csv("step6_dataset.csv", index_col=0, parse_dates=True)
today_row = df.iloc[-1]
today_date = df.index[-1]
```

Excelに当日データが追加されている状態でパイプラインを実行すれば、
`step6_dataset.csv` に当日行が生成され、ステップ7がそれを参照します。

### 2-2. 「未来情報のマスク」

以下の列は未来情報であるためレポートに表示しません：

| 列名 | 理由 |
|---|---|
| `tb_label` | バリア到達結果（未来） |
| `tb_return` | 実際のリターン（未来） |
| `tb_barrier` | どのバリアに触れたか（未来） |
| `tb_days_to_touch` | バリア到達日数（未来） |
| `forward_max_drawdown` | 保有期間中の最大含み損（未来） |
| `avoid_entry_flag` | ドローダウン発生フラグ（未来） |

### 2-3. 処理の流れ

```
step6_dataset.csv を読み込む
        ↓
最終行（最新日付）を抽出
        ↓
未来情報列をマスク
        ↓
シグナル・フィルタ理由を構造化
        ↓
コンソールに出力 + ファイルに保存
```

---

## 3. 実装設計

### 3-1. 新規ファイル: `src/step7_entry_signal.py`

```python
"""
AIトレードシステム ステップ7: リアルタイム エントリー判断レポート

step6_dataset.csv の最終行（最新日付）を取り出し、
未来情報を除いた上で「今日エントリーすべきか」を
人間が読めるレポートとして出力する。

新たなモデル学習や計算は行わない。
ステップ1〜6のパイプライン実行後に動作することを前提とする。
"""

from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

# 未来情報として表示・判断に使ってはいけない列
_FUTURE_COLS = {
    "tb_label", "tb_return", "tb_barrier", "tb_days_to_touch",
    "forward_max_drawdown", "avoid_entry_flag",
}


def build_entry_report(step6_df: pd.DataFrame,
                       barrier_mode: str,
                       tp: float,
                       sl: float) -> dict:
    """
    step6_dataset.csv の最終行からエントリー判断レポートを生成する。

    Returns
    -------
    dict with keys:
        date          : 判断日付
        decision      : "エントリー" or "見送り"
        signal        : assist_signal の値（強気/警戒/中立）
        reasons       : 見送り理由のリスト（エントリーの場合は空）
        metrics       : 判断根拠となる指標の辞書
        summary_text  : コンソール出力用のテキスト
    """
    # 最終行を取得（未来情報列は除外）
    row = step6_df.iloc[-1].drop(labels=[c for c in _FUTURE_COLS if c in step6_df.columns])
    today_date = step6_df.index[-1].date()

    # --- 判断結果 ---
    final_decision = row.get("final_decision", 0)
    # NaN（OOS対象外）= パイプライン未実行 or データ不足
    if pd.isna(final_decision):
        decision = "判断不可"
        reasons = ["ドローダウン予測モデルのOOS検証対象外です。パイプラインを再実行してください。"]
    elif final_decision == 1:
        decision = "エントリー"
        reasons = []
    else:
        decision = "見送り"
        reasons = _build_rejection_reasons(row, barrier_mode)

    # --- 判断根拠となる指標 ---
    metrics = _extract_key_metrics(row, barrier_mode, tp, sl)

    # --- サマリテキスト生成 ---
    summary_text = _format_summary(
        today_date, decision, reasons, metrics, barrier_mode
    )

    return {
        "date": today_date,
        "decision": decision,
        "signal": row.get("assist_signal", "不明"),
        "reasons": reasons,
        "metrics": metrics,
        "summary_text": summary_text,
    }


def _build_rejection_reasons(row: pd.Series, barrier_mode: str) -> list:
    """見送り理由を日本語で列挙する。"""
    reasons = []
    signal = row.get("assist_signal", "中立")

    if signal == "警戒":
        reasons.append(
            f"過熱シグナル（ATR正規化乖離: {row.get('atr_dev_ma25', float('nan')):.2f}）"
            "が発生しています。急騰局面のため見送り。"
        )
    elif signal == "中立":
        reasons.append("強気・警戒どちらの条件も成立しておらず、シグナルなし（中立）です。")
        # 何が足りないか補足
        if not row.get("rsi_reversal_flag", 0):
            reasons.append("RSI反転条件未成立: RSIが売られすぎ圏（35以下）からの切り返しが確認できません。")
        if not row.get("deviation_shrink_flag", 0):
            reasons.append("MA乖離縮小条件未成立: 移動平均からの下方乖離が縮小していません。")
        if not row.get("trending_bullish_flag", 0):
            reasons.append("順張り条件未成立: 週足上昇トレンド × RSI50以上 × 出来高増加の条件を満たしません。")

    elif signal == "強気":
        # 強気だがフィルタで除外されたケース
        if row.get("filtered_by_volatility", 0):
            atr_p = row.get("atr_percentile", float("nan"))
            z = row.get("dev_ma25_zscore", float("nan"))
            if atr_p >= 0.90:
                reasons.append(
                    f"ボラティリティ過熱フィルタ: ATRパーセンタイルが"
                    f"{atr_p:.1%}（過去250日の上位10%）に達しており、値動きが異常に大きい局面です。"
                )
            if z <= -2.5:
                reasons.append(
                    f"下方乖離フィルタ: MA乖離Zスコアが{z:.2f}（-2.5以下）で"
                    "底抜けリスクが高いと判断されました。"
                )
        if row.get("filtered_by_drawdown", 0):
            dd_p = row.get("drawdown_prob", float("nan"))
            reasons.append(
                f"ドローダウン予測フィルタ: ドローダウン発生確率が{dd_p:.1%}で"
                f"閾値を超えています。"
            )

    return reasons


def _extract_key_metrics(row: pd.Series, barrier_mode: str,
                          tp: float, sl: float) -> dict:
    """判断根拠として表示する指標を抽出する。"""
    wt = row.get("weekly_trend", 0)
    return {
        "シグナル":            row.get("assist_signal", "不明"),
        "最終判断":            "エントリー" if row.get("final_decision", 0) == 1 else "見送り",
        # 価格系
        "終値 (円)":           f"{row.get('close', float('nan')):.0f}",
        # 強気/警戒判定の根拠
        "RSI14":               f"{row.get('rsi14', float('nan')):.1f}",
        "RSI反転フラグ":        "✅" if row.get("rsi_reversal_flag", 0) else "❌",
        "MA乖離縮小フラグ":     "✅" if row.get("deviation_shrink_flag", 0) else "❌",
        "順張りフラグ":         "✅" if row.get("trending_bullish_flag", 0) else "❌",
        "過熱警戒フラグ":       "⚠️" if row.get("overheat_warning_flag", 0) else "-",
        # フィルタ根拠
        "ATRパーセンタイル":    f"{row.get('atr_percentile', float('nan')):.1%}  "
                               f"{'⚠️ 過熱' if row.get('atr_percentile', 0) >= 0.90 else '✅ 正常'}",
        "MA乖離Zスコア":        f"{row.get('dev_ma25_zscore', float('nan')):.2f}  "
                               f"{'⚠️ 底抜けリスク' if row.get('dev_ma25_zscore', 0) <= -2.5 else '✅ 正常'}",
        "DD予測確率":           f"{row.get('drawdown_prob', float('nan')):.1%}",
        # 週足・相場フェーズ
        "週足トレンド":         f"{'上昇 ▲' if wt == 1 else '下降 ▼'}",
        "ATR正規化乖離":        f"{row.get('atr_dev_ma25', float('nan')):.2f}",
        # バリア設定（参考）
        "利確目標":             f"+{tp:.0%}" if barrier_mode == "fixed_pct"
                               else f"ATR×{tp}",
        "損切りライン":         f"-{sl:.0%}" if barrier_mode == "fixed_pct"
                               else f"ATR×{sl}",
    }


def _format_summary(today_date: date, decision: str, reasons: list,
                    metrics: dict, barrier_mode: str) -> str:
    """コンソール出力・ファイル保存用のテキストを生成する。"""
    BORDER = "=" * 60
    THIN   = "-" * 60

    decision_label = {
        "エントリー": "✅  エントリー",
        "見送り":     "⛔  見送り",
        "判断不可":   "⚠️  判断不可",
    }.get(decision, decision)

    lines = [
        BORDER,
        f"  ステップ7: エントリー判断レポート",
        f"  判断日付: {today_date}",
        BORDER,
        f"",
        f"  【最終判断】  {decision_label}",
        f"",
        THIN,
        f"  ■ 判断根拠",
        THIN,
    ]

    for key, val in metrics.items():
        lines.append(f"    {key:<18}: {val}")

    if reasons:
        lines += [
            f"",
            THIN,
            f"  ■ 見送り理由",
            THIN,
        ]
        for i, r in enumerate(reasons, start=1):
            # 長い理由を折り返す（60文字で）
            lines.append(f"    {i}. {r}")

    lines += [
        f"",
        BORDER,
        f"  ※ このレポートは過去データに基づく参考情報です。",
        f"    投資判断はご自身の責任で行ってください。",
        BORDER,
    ]
    return "\n".join(lines)


def save_report(report: dict, output_dir: Path) -> Path:
    """レポートをテキストファイルとして保存する。"""
    output_path = output_dir / "today_signal.txt"
    output_path.write_text(report["summary_text"], encoding="utf-8")
    return output_path
```

---

### 3-2. `main.py` への統合

#### 追加する import

```python
from src.step7_entry_signal import build_entry_report, save_report
```

#### ステップ6の後に追加するブロック

```python
# --- ステップ7: リアルタイム エントリー判断レポート ---
print("\n" + "=" * 60)
print("ステップ7: リアルタイム エントリー判断レポート")
print("=" * 60)

# step6を有効化していない場合は step5_df を使う
source_df = step6_df if args.step6 else step5_df

# tp / sl の値をバリアモードに応じて整理
if args.barrier_mode == "fixed_pct":
    tp_disp, sl_disp = args.tp_pct, args.sl_pct
else:
    tp_disp, sl_disp = args.tp_atr_mult, args.sl_atr_mult

report = build_entry_report(source_df, args.barrier_mode, tp_disp, sl_disp)

print(report["summary_text"])

report_path = save_report(report, output_dir)
print(f"\n判断レポート保存完了: {report_path}")
```

#### `--today` フラグの追加（オプション）

パイプライン全体を再実行せず、保存済みの `step6_dataset.csv` だけから
レポートだけを再出力したい場合に使います。

```python
parser.add_argument(
    "--today", action="store_true", default=False,
    help="パイプラインを再実行せず、保存済みのstep6_dataset.csvから"
         "今日のエントリー判断レポートのみを出力する。"
)
```

```python
# main() の先頭に追加
if args.today:
    # 保存済みCSVからレポートのみ生成して終了
    step6_csv = output_dir / "step6_dataset.csv"
    if not step6_csv.exists():
        print(f"step6_dataset.csv が見つかりません: {step6_csv}")
        print("先に python main.py --code {code} を実行してください。")
        sys.exit(1)
    source_df = pd.read_csv(step6_csv, index_col=0, parse_dates=True)
    tp_disp = args.tp_pct if args.barrier_mode == "fixed_pct" else args.tp_atr_mult
    sl_disp = args.sl_pct if args.barrier_mode == "fixed_pct" else args.sl_atr_mult
    report = build_entry_report(source_df, args.barrier_mode, tp_disp, sl_disp)
    print(report["summary_text"])
    save_report(report, output_dir)
    sys.exit(0)
```

---

## 4. 出力サンプル

```
============================================================
  ステップ7: エントリー判断レポート
  判断日付: 2026-06-05
============================================================

  【最終判断】  ⛔  見送り

------------------------------------------------------------
  ■ 判断根拠
------------------------------------------------------------
    シグナル          : 中立
    最終判断          : 見送り
    終値 (円)         : 3886
    RSI14             : 53.4
    RSI反転フラグ     : ❌
    MA乖離縮小フラグ  : ❌
    順張りフラグ      : ❌
    過熱警戒フラグ    : -
    ATRパーセンタイル : 100.0%  ⚠️ 過熱
    MA乖離Zスコア     : 1.33  ✅ 正常
    DD予測確率        : 2.9%
    週足トレンド      : 下降 ▼
    ATR正規化乖離     : 0.84
    利確目標          : +10%
    損切りライン      : -5%

------------------------------------------------------------
  ■ 見送り理由
------------------------------------------------------------
    1. 強気・警戒どちらの条件も成立しておらず、シグナルなし（中立）です。
    2. RSI反転条件未成立: RSIが売られすぎ圏（35以下）からの切り返しが確認できません。
    3. MA乖離縮小条件未成立: 移動平均からの下方乖離が縮小していません。
    4. 順張り条件未成立: 週足上昇トレンド × RSI50以上 × 出来高増加の条件を満たしません。

============================================================
  ※ このレポートは過去データに基づく参考情報です。
    投資判断はご自身の責任で行ってください。
============================================================
```

---

## 5. 出力ファイル

| ファイル | 内容 |
|---|---|
| `resource/{code}/{mode}/today_signal.txt` | テキスト形式のエントリー判断レポート（毎回上書き） |

---

## 6. 実行方法

```bash
# 通常実行（パイプライン全体 + ステップ7のレポートも出力）
python main.py --code 7701 --barrier-mode fixed_pct

# Excelデータを更新した後、レポートだけ更新したい場合
python main.py --code 7701 --barrier-mode fixed_pct --today
```

---

## 7. 設計上の重要な注意事項

### 7-1. 「今日のシグナル」に未来情報は使わない

`tb_label`・`tb_return`・`forward_max_drawdown` はいずれも
エントリー後の結果であり、実運用ではエントリー時点で知ることができません。
ステップ7はこれらを意思決定に用いず、表示もしません。

### 7-2. ドローダウン予測確率（`drawdown_prob`）について

`step6_dataset.csv` に記録されている `drawdown_prob` は
**Walk-Forward ValidationのOOS（検証）期間**で得られた確率です。
最終行はパイプラインが`final_model`（全期間で再学習したモデル）から
算出した値のため、実運用での参照値として利用できます。

ただし `antigravity_2606240844.md` が示すように、
`drawdown_prob` が逆相関（高いほど実際の勝率が高い）になる銘柄では
この値を**フィルタの判断材料として使うべきではありません**。
`--drawdown-prob-limit 1.0`（ML無効）で実行されている場合、
レポートにも「MLフィルタ無効（ATRルールのみ）」と明示します。

### 7-3. レポートは「補助情報」であること

本システムはバックテストに基づく参考情報を提供するものです。
ステップ7の出力を最終的な投資判断として使用することは想定していません。
レポートには必ずその旨を明記します。
