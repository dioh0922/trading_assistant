# ステップ8: 一括スクリーニング 実装計画

## 概要

`resource/` 内の全 `XXXX.xlsx`（4桁コード）を一括処理し、今日エントリー可能な銘柄一覧を `scan_report_{mode}.txt` に出力する。

---

## 1. 新規作成: `src/step8_scanner.py`

### 1-1. 関数一覧

| 関数 | 責務 |
|---|---|
| `collect_xlsx_codes(resource_dir)` | `resource/*.xlsx` から4桁コードのみ抽出 |
| `is_cache_valid(csv_path, max_days=7)` | step6_dataset.csvの最終行日付が7日以内か確認 |
| `run_pipeline_for_code(code, ...)` | main.pyのstep1〜6処理を抽出・関数化。`skip_charts=True`でグラフスキップ |
| `scan_all(...)` → `list[dict]` | 全銘柄ループ、キャッシュ判定、step7呼び出し、結果集約 |
| `build_scan_report(results)` → `str` | エントリー候補/見送り/エラーをセクション分けしたレポート文字列生成 |
| `save_scan_report(report_text, mode)` → `Path` | `resource/scan_report_{mode}.txt` に保存 |

### 1-2. `run_pipeline_for_code()` の設計

main.py の L191〜L316 を関数化。以下の点に注意:

- **グラフ出力**: `skip_charts=True` 時は `plot_stepN()` を全スキップ
- **step6**: `step6_enabled=False` なら step5 dataset で止める
- **返り値**: step6_dataset.csv（or step5_dataset.csv）を読み込んだ `pd.DataFrame`

```python
def run_pipeline_for_code(
    code: str, barrier_mode: str,
    tp, sl, holding_period, drawdown_threshold,
    step6_enabled: bool, dd_prob_limit: float,
    skip_charts: bool = True,
) -> pd.DataFrame:
    # 1. input_path = RESOURCE_DIR / f"{code}.xlsx"
    # 2. output_dir = RESOURCE_DIR / code / barrier_mode
    # 3. step1〜6逐次実行（main.pyと同一ロジック）
    # 4. skip_charts=True なら plot_* を呼ばない
    # 5. step6_dataset.csv or step5_dataset.csv を pd.DataFrame で返す
```

### 1-3. `is_cache_valid()` の注意点

- ファイルが存在しない → `False`
- 読み取りエラー → `False`（安全側に倒してフル実行）
- 最終行の日付と `date.today()` の差が7日以内 → `True`

### 1-4. `scan_all()` のエラーハンドリング

```python
try:
    report = build_entry_report(source_df, barrier_mode, tp, sl)
    report["code"] = code
    results.append(report)
except Exception as e:
    results.append({
        "code": code, "date": date.today(),
        "decision": "エラー", "signal": "不明",
        "reasons": [str(e)], "metrics": {}, "summary_text": "",
    })
    # 次の銘柄へ継続
```

---

## 2. 修正: `main.py`

### 2-1. 追加する引数（2つ）

```python
parser.add_argument("--scan", action="store_true", default=False,
    help="resource/ 内の全4桁コードXLSXを一括スクリーニング")
parser.add_argument("--force", action="store_true", default=False,
    help="[--scanと併用] キャッシュ無視で強制再実行")
```

### 2-2. `main()` の先頭に追加するブロック

```python
if args.scan:
    from src.step8_scanner import scan_all, build_scan_report, save_scan_report
    tp = args.tp_pct if args.barrier_mode == "fixed_pct" else args.tp_atr_mult
    sl = args.sl_pct if args.barrier_mode == "fixed_pct" else args.sl_atr_mult
    results = scan_all(
        barrier_mode=args.barrier_mode, tp=tp, sl=sl,
        holding_period=args.holding_period,
        drawdown_threshold=args.drawdown_threshold,
        step6_enabled=args.step6, dd_prob_limit=args.drawdown_prob_limit,
        force_update=args.force,
    )
    report_text = build_scan_report(results)
    print("\n" + report_text)
    path = save_scan_report(report_text, args.barrier_mode)
    print(f"\nレポート保存完了: {path}")
    return
```

**重要**: `return` で通常パイプラインを実行せず終了する。

---

## 3. 変更影響範囲

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `src/step8_scanner.py` | **新規作成** | 全ロジック |
| `main.py` | 修正 | `--scan`/`--force`引数追加 + `main()`先頭に分岐 |

**変更しないファイル**: step1〜step7（既存のパイプライン・レポート生成は一切変更しない）

---

## 4. テスト計画

### 4-1. 手動テスト

```bash
# 1. キャッシュあり（既存のstep6_dataset.csvがある銘柄）
python main.py --scan

# 2. 強制フル実行
python main.py --scan --force

# 3. 固定%モード
python main.py --scan --barrier-mode fixed_pct

# 4. step6フィルタなし
python main.py --scan --no-step6

# 5. resource/ にxlsxが1つもない場合（エラーメッセージ確認）
```

### 4-2. 異常系テスト

- resource/ に有効な xlsx が存在しない → 「見つかりません」メッセージ
- 1銘柄でエラー → 他銘柄は継続処理
- キャッシュ読み取りエラー → フル実行にフォールバック

---

## 5. 実装手順（TODO）

1. `src/step8_scanner.py` を作成（全関数を実装）
2. `main.py` に `--scan` / `--force` 引数を追加
3. `main.py` の `main()` 先頭に `--scan` 分岐を追加
4. 手動テスト実行（`--scan`, `--scan --force`, `--scan --no-step6`）

---

## 6. 注意点・確認事項

- [ ] `run_pipeline_for_code` のパラメータは `parse_args()` の後処理（自動デフォルト補完）**後**の値が渡されることを前提とする
- [ ] グラフスキップ時でもCSV保存は通常通り行う（キャッシュとして後日再利用できるように）
- [ ] `--scan` と `--code` は同時指定不可（`--scan` が優先され、codeは無視されるがエラーにはしない）
- [ ] `scan_report_{mode}.txt` は **UTF-8 BOMなし** で保存する（既存のtoday_signal.txtと揃える）
