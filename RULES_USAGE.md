# 取引ルール統合システム (RULES_USAGE.md)

- **関連計画**: [`plan/ai/20260829_rule_llm_modular_impl_plan.md`](plan/ai/20260829_rule_llm_modular_impl_plan.md) (Step 1〜4 実装済み)
- **対象ソースコード**:
  - ルール判定コア: [`train/src/pipeline/rules.py`](train/src/pipeline/rules.py)
  - 日次チェック: [`train/src/pipeline/check_positions.py`](train/src/pipeline/check_positions.py)
  - 銘柄詳細JSON出力: [`train/src/pipeline/analyze_ticker.py`](train/src/pipeline/analyze_ticker.py)
  - ルールローダー/プロンプトビルダー: [`src/rules/`](src/rules/)
- **単体テスト**:
  - [`train/tests/test_rules.py`](train/tests/test_rules.py)
  - [`train/tests/test_check_positions.py`](train/tests/test_check_positions.py)
  - [`train/tests/test_analyze_ticker.py`](train/tests/test_analyze_ticker.py)
  - [`tests/test_src_rules.py`](tests/test_src_rules.py)

---

## 1. 概要

`RULE.md` に定義されたユーザー独自の取引規律（+10%利確 / -5%損切り / 10日横ばいルール / 半数決済等）を、機械的に判定し、日次チェック（Markdownレポート）、銘柄詳細（JSON）、LLMプロンプト生成に一貫して統合するシステムです。

### 判定される主なルール
1. **損切りライン（-5%到達）**: 直ちに全ポジション損切りを推奨 (`STOP_LOSS`)
2. **利確ライン（+10%到達）**: 即時全利確せずホールド推奨、反落・下降兆候（RSI・MA等）監視を指示 (`TAKE_PROFIT`)
3. **10日ルール（保有10日以上でトレンドなし）**:
   - 含み益の場合: 全清算を検討 (`TIME_RULE`)
   - 含み損の場合: 半数損切り（リスク縮小）を検討 (`TIME_RULE`)
4. **通常範囲内**: ルール抵触なし・継続保有 (`NONE`)

---

# 取引ルール統合システム (RULES_USAGE.md)

- **関連計画**: [`plan/ai/20260829_rule_llm_modular_impl_plan.md`](plan/ai/20260829_rule_llm_modular_impl_plan.md) (Step 1〜4 実装済み)
- **対象ソースコード**:
  - ルール判定コア: [`train/src/pipeline/rules.py`](train/src/pipeline/rules.py)
  - 日次チェック: [`train/src/pipeline/check_positions.py`](train/src/pipeline/check_positions.py)
  - 銘柄詳細JSON出力: [`train/src/pipeline/analyze_ticker.py`](train/src/pipeline/analyze_ticker.py)
  - ルールローダー/プロンプトビルダー: [`src/rules/`](src/rules/)
  - 統合パイプライン: [`run_pipeline_all.py`](run_pipeline_all.py)

---

## 1. クイック実行（おすすめの短縮コマンド ⭐）

プロジェクトルートから **短いワンコマンド** で各処理を実行できます（パスやパラメータは自動解決されます）。

```powershell
# 1. 日次ポジションチェック（最新データ取得 ＋ アラート判定・レポート出力）
python run_pipeline_all.py --daily

# 2. ポジションチェック・レポート出力のみ即座に実行（スクレイピング省略）
python run_pipeline_all.py --check

# 3. 保有ポジション全銘柄の詳細JSON（position_status入り）を一括生成
python run_pipeline_all.py --json

# 4. 単一銘柄のみ詳細JSONを生成
python run_pipeline_all.py --json --ticker 6857

# 5. 全フロー一括実行（スクレイピング → 日次チェック → JSON生成 → LLM分析）
python run_pipeline_all.py --all
```

---

## 2. モジュール直接実行（引数なしで実行可能）

各スクリプトはデフォルトパスが自動解決されるため、**引数を指定せずシンプルに実行** できます。

```powershell
# 日次チェックレポートの出力 (reports/daily_check_YYYYMMDD_HHMMSS.md)
python -m train.src.pipeline.check_positions

# 保有ポジション全銘柄の詳細JSONの一括出力 (reports/json/{code}_detail.json)
python -m train.src.pipeline.analyze_ticker
```

> **カスタマイズ指定（必要な場合のみ）**:
> ```powershell
> # 特定の銘柄と買付情報を手動で指定してJSON生成する場合
> python -m train.src.pipeline.analyze_ticker --ticker 6857 --entry-price 1000.0 --entry-date 2026-08-10 --no-timestamp
> ```

---

## 3. Pythonコードからの利用方法 (`src/rules`)

```python
import json
from pathlib import Path
from src.rules import load_rules_md, build_full_system_prompt, build_position_status_prompt_note

# 1. RULE.md の動的読み込み
rules_text = load_rules_md()

# 2. position_status からLLM向け注意喚起プロンプトを生成
detail_json = json.loads(Path("reports/json/4503_detail.json").read_text(encoding="utf-8"))
pos_note = build_position_status_prompt_note(detail_json.get("position_status"))

# 3. 取引ルール・出力構成（見出し1〜6）を含む統合システムプロンプトを生成
system_prompt = build_full_system_prompt()
```

---

## 4. 保有ポジション判定結果のターミナル即時確認

```powershell
python -c "
import sys, io, polars as pl
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from train.src.pipeline.ingest import parse_single_csv
from train.src.pipeline.rules import evaluate_position_rules

pos_df = pl.read_csv('train/positions.csv')
raw_dir = Path('data/raw')

print('=== 保有ポジション ルール判定結果 ===\n')
for row in pos_df.iter_rows(named=True):
    code, entry_price = str(row['code']), float(row['entry_price'])
    entry_date = row.get('entry_date')
    csv_path = raw_dir / f'{code}.csv'
    if not csv_path.exists(): continue
    prices_df = parse_single_csv(csv_path)
    latest_row = prices_df.sort('date').tail(1)
    current_price, latest_date = float(latest_row['close'][0]), str(latest_row['date'][0])
    status = evaluate_position_rules(code, entry_price, current_price, entry_date, latest_date)
    print(f'[{code}] 買付: {entry_price:,.1f}円 ({entry_date}) -> 現在: {current_price:,.1f}円 ({status.unrealized_return*100:+.2f}%, 保有{status.holding_days}日)')
    print(f'  判定: {status.alert_level} ({status.alert_message})')
    print(f'  推奨: {status.recommended_rule_action}\n')
"
```

---

## 5. 単体テストの実行

```powershell
pytest train/tests/test_rules.py train/tests/test_check_positions.py train/tests/test_analyze_ticker.py tests/test_src_rules.py -v
```

