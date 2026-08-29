# ルール評価モジュール (pipeline/rules.py) の使い方

- **関連計画**: [`plan/ai/20260829_rule_llm_modular_impl_plan.md`](plan/ai/20260829_rule_llm_modular_impl_plan.md) (Step 1)
- **対象ソースコード**: [`train/src/pipeline/rules.py`](train/src/pipeline/rules.py)
- **単体テスト**: [`train/tests/test_rules.py`](train/tests/test_rules.py)

---

## 1. 概要

`RULE.md` に定義されたユーザー独自の取引規律（+10%利確 / -5%損切り / 10日横ばいルール / 半数決済等）を、機械的に判定するための共通コアモジュールです。

### 判定される主なルール
1. **損切りライン（-5%到達）**: 直ちに全ポジション損切りを推奨 (`STOP_LOSS`)
2. **利確ライン（+10%到達）**: 即時全利確せずホールド推奨、反落・下降兆候（RSI・MA等）監視を指示 (`TAKE_PROFIT`)
3. **10日ルール（保有10日以上でトレンドなし）**:
   - 含み益の場合: 全清算を検討 (`TIME_RULE`)
   - 含み損の場合: 半数損切り（リスク縮小）を検討 (`TIME_RULE`)
4. **通常範囲内**: ルール抵触なし・継続保有 (`NONE`)

---

## 2. Pythonコードからの利用方法

### 基本的な呼び出し

```python
from pipeline.rules import evaluate_position_rules

# ポジションのルール判定を実行
status = evaluate_position_rules(
    code="7974",
    entry_price=7606.0,
    current_price=8728.0,
    entry_date="2026-08-10",  # エントリー日（省略時は日数計算なし）
    current_date="2026-08-27", # 評価日（省略時は当日）
)

# 結果の参照
print(f"銘柄コード: {status.code}")
print(f"損益率: {status.unrealized_return * 100:+.2f}%")
print(f"保有日数: {status.holding_days} 日")
print(f"アラート区分: {status.alert_level}")          # 'TAKE_PROFIT', 'STOP_LOSS', 'TIME_RULE', 'NONE'
print(f"アラート文言: {status.alert_message}")
print(f"推奨アクション: {status.recommended_rule_action}")

# JSON/辞書形式への変換（detail.json等への埋め込み用）
status_dict = status.to_dict()
```

### 戻り値データ構造 (`PositionStatus`)

```python
@dataclass
class PositionStatus:
    code: str                     # 銘柄コード (例: "7974")
    entry_price: float            # 買付価格
    current_price: float          # 現在株価
    unrealized_return: float      # 含み損益率 (例: 0.1475 -> +14.75%)
    entry_date: str | None        # エントリー日
    holding_days: int | None      # 保有日数
    target_hit: TargetHit         # take_profit_hit / stop_loss_hit フラグ
    time_rule_triggered: bool     # 10日ルール発動フラグ (True / False)
    alert_level: str              # "NONE" | "TAKE_PROFIT" | "STOP_LOSS" | "TIME_RULE"
    alert_message: str            # 表示用アラートメッセージ
    recommended_rule_action: str  # ルールに基づく推奨アクション
```

---

## 3. 実データ（positions.csv）でのワンライナー確認方法

ターミナル（PowerShell / Bash）で実際の保有銘柄データを即座に評価・確認するコマンドです。

```powershell
cd train
python -c "
import sys, polars as pl
from pathlib import Path
from pipeline.ingest import parse_single_csv
from pipeline.rules import evaluate_position_rules

sys.stdout.reconfigure(encoding='utf-8')
pos_df = pl.read_csv('positions.csv')
raw_dir = Path('../data/raw')  # または Path('data/raw')

print('=== 保有ポジション ルール判定結果 ===\n')
for row in pos_df.iter_rows(named=True):
    code, entry_price, entry_date = str(row['code']), float(row['entry_price']), row.get('entry_date')
    csv_path = raw_dir / f'{code}.csv'
    if not csv_path.exists(): continue
    prices_df = parse_single_csv(csv_path)
    latest_row = prices_df.sort('date').tail(1)
    current_price, latest_date = float(latest_row['close'][0]), str(latest_row['date'][0])
    status = evaluate_position_rules(code, entry_price, current_price, entry_date, latest_date)
    print(f'[{code}] 買付: {entry_price:,.1f}円 -> 現在: {current_price:,.1f}円 ({status.unrealized_return*100:+.2f}%)')
    print(f'  判定: {status.alert_level} ({status.alert_message})')
    print(f'  推奨: {status.recommended_rule_action}\n')
"
```

---

## 4. 単体テストの実行

```powershell
cd train
pytest tests/test_rules.py -v
```
