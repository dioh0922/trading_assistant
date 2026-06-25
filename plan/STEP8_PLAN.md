# ステップ8: 一括スクリーニング 設計プラン

`resource/` に存在する全ての銘柄コードXLSX（`XXXX.xlsx`）を順番に処理し、
「今日エントリーしてよいか」をまとめて判定してスクリーニング結果を出力する。

---

## 1. 背景と目的

### 1-1. ステップ7との違い

| | ステップ7 | ステップ8 |
|---|---|---|
| 対象 | 1銘柄 | `resource/*.xlsx`（全銘柄） |
| 実行 | `python main.py --code 7701` | `python main.py --scan` |
| 出力 | `today_signal.txt`（1銘柄） | `scan_report.txt`（全銘柄まとめ） |
| 用途 | 特定銘柄の詳細確認 | 「今日どの銘柄が買えるか」の一覧把握 |

### 1-2. 想定する使い方

```bash
# 毎朝Excelを更新した後にこれを実行し、今日のエントリー候補を把握する
python main.py --scan
python main.py --scan --barrier-mode fixed_pct   # 固定%モードで全銘柄スキャン
python main.py --scan --barrier-mode fixed_pct --no-step6  # step6フィルタなし
```

---

## 2. 処理フローの設計

### 2-1. キャッシュ戦略（処理速度とのトレードオフ）

全銘柄でstep1〜6を毎回フル実行すると銘柄数×約10〜15秒かかります。
一方でstep7（最終行の読み取りのみ）は一瞬です。
そのため **「既存の`step6_dataset.csv`があればそれを使う」キャッシュ戦略**を採用します。

```
resource/XXXX.xlsx を検出
    ↓
resource/XXXX/{mode}/step6_dataset.csv が存在するか？
    ↓ Yes（キャッシュあり）          ↓ No（初回 or 強制更新）
最終行の日付 == 今日か？            フルパイプライン実行
    ↓ Yes           ↓ No           (step1→2→3→4→5→6)
そのまま使う    フルパイプライン実行
    ↓
step7ロジックでシグナルを取り出す
    ↓
全銘柄の結果をまとめてスクリーニングレポートを出力
```

#### キャッシュの「有効期限」の定義

「最終行の日付」と「当日の日付」を比較します。
週末・祝日の場合は「直近の営業日のデータが入っていれば有効」として扱います。

```python
def is_cache_valid(csv_path: Path) -> bool:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    last_date = df.index[-1].date()
    today = date.today()
    # 当日または直近5営業日以内なら有効（週末をまたいでも使える）
    delta = (today - last_date).days
    return delta <= 7
```

`--force`フラグを指定すると全銘柄を強制フル実行します。

### 2-2. 処理フローの全体像

```
python main.py --scan
    ↓
resource/*.xlsx から4桁コードのファイルを収集
    ↓
各銘柄を順次処理（シングルスレッド: CPU=1のため）
    │
    ├── キャッシュ有効 → step6_dataset.csvを読み込む
    └── キャッシュ無効 → フルパイプライン実行（グラフ出力スキップ）
    │
    ↓
step7ロジックで今日のシグナルを取得
    ↓
全銘柄の結果を集約
    ↓
スクリーニングレポートを出力（コンソール + ファイル）
```

---

## 3. 実装設計

### 3-1. 新規ファイル: `src/step8_scanner.py`

```python
"""
AIトレードシステム ステップ8: 一括スクリーニング

resource/ にある全ての4桁コードXLSX（XXXX.xlsx）を処理し、
今日のエントリー判断を一覧化したスクリーニングレポートを出力する。
"""

from pathlib import Path
from datetime import date
import re
import traceback
import pandas as pd

from src.step7_entry_signal import build_entry_report

RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resource"


# ステップ1〜6のパイプラインを1銘柄分実行する関数（main.pyのロジックを抽出）
def run_pipeline_for_code(code: str, barrier_mode: str,
                          tp, sl, holding_period,
                          drawdown_threshold, step6_enabled,
                          dd_prob_limit) -> Path:
    """
    1銘柄のstep1〜6を実行し、output_dirを返す。
    グラフ出力はスキップしてCSVのみ保存する（バッチ処理の高速化）。
    """
    ...  # main.pyのstep1〜6処理を関数化したもの


def collect_xlsx_codes(resource_dir: Path) -> list[str]:
    """
    resource/ 直下の4桁コードXLSXファイルを収集する。
    sample.xlsx など非コードファイルは除外する。
    """
    codes = []
    for f in sorted(resource_dir.glob("*.xlsx")):
        if re.match(r"^\d{4}$", f.stem):
            codes.append(f.stem)
    return codes


def is_cache_valid(csv_path: Path, max_days: int = 7) -> bool:
    """
    step6_dataset.csv が有効なキャッシュかどうかを確認する。
    最終行の日付が今日から max_days 日以内なら有効とみなす。
    """
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True, nrows=1,
                         skiprows=lambda i: i > 0 and i < sum(1 for _ in open(csv_path)) - 2)
        # 末尾の日付だけ取得（大きいファイルの高速読み取り）
        df_tail = pd.read_csv(csv_path, index_col=0, parse_dates=True).iloc[-1:]
        last_date = df_tail.index[-1].date()
        delta = (date.today() - last_date).days
        return delta <= max_days
    except Exception:
        return False


def scan_all(barrier_mode: str = "atr",
             tp=2.0, sl=1.5, holding_period=None,
             drawdown_threshold=0.03,
             step6_enabled=True, dd_prob_limit=1.0,
             force_update: bool = False) -> list[dict]:
    """
    全銘柄のシグナルを収集して返す。

    Returns
    -------
    list of dict（各銘柄の build_entry_report() 結果 + code を付加）
    """
    codes = collect_xlsx_codes(RESOURCE_DIR)
    if not codes:
        print("[Step8] resource/ に4桁コードのXLSXファイルが見つかりませんでした。")
        return []

    print(f"[Step8] {len(codes)}銘柄を処理します: {codes}")
    results = []

    for code in codes:
        print(f"\n{'─'*40}")
        print(f"[Step8] 処理中: {code}.xlsx")

        try:
            output_dir = RESOURCE_DIR / code / barrier_mode
            step6_csv = output_dir / "step6_dataset.csv"
            step5_csv = output_dir / "step5_dataset.csv"

            # キャッシュ判定
            use_cache = not force_update and is_cache_valid(step6_csv)

            if use_cache:
                print(f"  → キャッシュ有効: {step6_csv.name} を使用します。")
                source_df = pd.read_csv(step6_csv, index_col=0, parse_dates=True)
            else:
                print(f"  → パイプラインを実行します（グラフ出力なし）...")
                source_df = run_pipeline_for_code(
                    code, barrier_mode, tp, sl, holding_period,
                    drawdown_threshold, step6_enabled, dd_prob_limit,
                    skip_charts=True,  # バッチ処理ではグラフ生成をスキップ
                )

            # step7ロジックでシグナル取得
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
    """
    全銘柄のシグナルをまとめたスクリーニングレポートのテキストを生成する。
    エントリー候補銘柄を上部に、見送り・エラーを下部に配置する。
    """
    today = date.today()
    BORDER = "=" * 70
    THIN   = "─" * 70

    entry_candidates = [r for r in results if r["decision"] == "エントリー"]
    skipped          = [r for r in results if r["decision"] == "見送り"]
    errors           = [r for r in results if r["decision"] == "エラー"]
    unknown          = [r for r in results if r["decision"] == "判断不可"]

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

    # --- エントリー候補（詳細表示） ---
    if entry_candidates:
        lines += [THIN, "  【エントリー候補】", THIN]
        for r in entry_candidates:
            m = r.get("metrics", {})
            lines.append(
                f"  ✅ {r['code']}  |  "
                f"シグナル: {r['signal']}  |  "
                f"終値: {m.get('終値 (円)', '-')}円  |  "
                f"RSI: {m.get('RSI14', '-')}  |  "
                f"週足: {m.get('週足トレンド', '-')}  |  "
                f"ATRパーセンタイル: {m.get('ATRパーセンタイル', '-')}"
            )
    else:
        lines += [THIN, "  【エントリー候補】 なし", THIN]

    # --- 見送り一覧（理由の先頭のみ） ---
    lines += ["", THIN, "  【見送り一覧】", THIN]
    for r in skipped:
        reason_head = r["reasons"][0][:50] + "…" if r.get("reasons") else "-"
        lines.append(f"  ⛔ {r['code']}  |  シグナル: {r['signal']}  |  理由: {reason_head}")

    # --- 判断不可・エラー ---
    if unknown or errors:
        lines += ["", THIN, "  【判断不可・エラー】", THIN]
        for r in unknown + errors:
            lines.append(f"  ⚠️  {r['code']}  |  {r['decision']}: {r['reasons'][0][:60]}")

    lines += [
        "",
        BORDER,
        "  ※ 各銘柄の詳細は today_signal.txt または --code XXXX で確認できます。",
        "  ※ このレポートは参考情報です。投資判断はご自身の責任で行ってください。",
        BORDER,
    ]

    return "\n".join(lines)


def save_scan_report(report_text: str, barrier_mode: str) -> Path:
    """スクリーニングレポートを resource/ 直下に保存する。"""
    output_path = RESOURCE_DIR / f"scan_report_{barrier_mode}.txt"
    output_path.write_text(report_text, encoding="utf-8")
    return output_path
```

---

### 3-2. `main.py` への統合

#### 追加する引数

```python
parser.add_argument(
    "--scan", action="store_true", default=False,
    help="resource/ 内の全4桁コードXLSXを対象に一括スクリーニングを実行する。",
)
parser.add_argument(
    "--force", action="store_true", default=False,
    help="[--scanと併用] キャッシュを無視して全銘柄のパイプラインを強制再実行する。",
)
```

#### `main()` の先頭に追加するブロック

```python
# --scan モードの処理（ここで完結して return）
if args.scan:
    from src.step8_scanner import scan_all, build_scan_report, save_scan_report
    tp = args.tp_pct if args.barrier_mode == "fixed_pct" else args.tp_atr_mult
    sl = args.sl_pct if args.barrier_mode == "fixed_pct" else args.sl_atr_mult

    results = scan_all(
        barrier_mode=args.barrier_mode,
        tp=tp, sl=sl,
        holding_period=args.holding_period,
        drawdown_threshold=args.drawdown_threshold,
        step6_enabled=args.step6,
        dd_prob_limit=args.drawdown_prob_limit,
        force_update=args.force,
    )

    report_text = build_scan_report(results)
    print("\n" + report_text)

    path = save_scan_report(report_text, args.barrier_mode)
    print(f"\nレポート保存完了: {path}")
    return   # 通常パイプラインは実行しない
```

---

## 4. 出力サンプル

```
======================================================================
  ステップ8: 一括スクリーニングレポート
  スキャン日: 2026-06-05  対象銘柄数: 3
======================================================================

  ✅ エントリー候補: 1銘柄
  ⛔ 見送り:        1銘柄
  ⚠️  判断不可:       0銘柄
  ❌ エラー:         1銘柄

──────────────────────────────────────────────────────────────────────
  【エントリー候補】
──────────────────────────────────────────────────────────────────────
  ✅ 6981  |  シグナル: 強気  |  終値: 11420円  |  RSI: 32.1  |
            週足: 上昇 ▲  |  ATRパーセンタイル: 72.0%  ✅ 正常

──────────────────────────────────────────────────────────────────────
  【見送り一覧】
──────────────────────────────────────────────────────────────────────
  ⛔ 7701  |  シグナル: 中立  |  理由: 強気・警戒どちらの条件も成立しておらず…

──────────────────────────────────────────────────────────────────────
  【判断不可・エラー】
──────────────────────────────────────────────────────────────────────
  ❌ 9999  |  エラー: No sheet named 'Sheet2'...

======================================================================
  ※ 各銘柄の詳細は today_signal.txt または --code XXXX で確認できます。
  ※ このレポートは参考情報です。投資判断はご自身の責任で行ってください。
======================================================================
```

---

## 5. 出力ファイル

| ファイル | 内容 |
|---|---|
| `resource/scan_report_{mode}.txt` | 全銘柄のスクリーニングレポート（毎回上書き） |
| `resource/{code}/{mode}/step6_dataset.csv` | 各銘柄の詳細データ（キャッシュとして参照） |
| `resource/{code}/{mode}/today_signal.txt` | 各銘柄の個別レポート（step7出力） |

---

## 6. 実行方法

```bash
# 全銘柄を一括スクリーニング（キャッシュ活用）
python main.py --scan

# 固定%モードで全銘柄スクリーニング
python main.py --scan --barrier-mode fixed_pct

# Excelを更新した後、全銘柄を強制再実行
python main.py --scan --force

# step6フィルタなし（ATRルール除く）でスキャン
python main.py --scan --no-step6

# スキャン後、特定銘柄の詳細を確認
python main.py --code 6981 --today
```

---

## 7. 設計上の重要な決定事項

### 7-1. グラフ出力のスキップ

`--scan` モードではmatplotlibによるグラフ生成（PNG保存）をスキップします。
グラフ生成は全処理時間の大半を占めるため、10銘柄では体感差が大きくなります。
グラフが必要な場合は `--code XXXX` で個別実行してください。

```python
# run_pipeline_for_code() の skip_charts=True 時の挙動
if not skip_charts:
    plot_step1(step1_df, output_dir / "step1_chart.png")
    # ...
```

### 7-2. エラー銘柄は処理を止めない

1銘柄でエラーが発生しても処理を継続し、エラー情報をレポートに記録します。

```python
try:
    report = process_one_code(code, ...)
except Exception as e:
    results.append({"code": code, "decision": "エラー", "reasons": [str(e)]})
    continue  # 次の銘柄へ
```

### 7-3. キャッシュの有効期間

7日以内（`max_days=7`）に設定することで、週末をまたいだ月曜実行でも
金曜のキャッシュが有効になります。
毎日更新が必要な場合は `--force` を使います。

### 7-4. ステップ7との関係

`--scan` 実行時、各銘柄について `build_entry_report()`（step7の関数）を
内部的に呼び出します。つまり**ステップ8はステップ7を銘柄数分ループ実行する**構造です。
ステップ7が正しく実装・テストされていることが前提です。
