"""
pipeline.rules
──────────────
RULE.md で定義されたユーザー取引ルールの機械的判定を行うモジュール。

ルール概要:
1. リスクリワード: +10% (利確ライン) / -5% (損切りライン)
2. 含み損が -5% に達した場合は即座に損切り
3. 含み益が +10% に達した場合は即座に利確せずホールド、ただし下降の兆候が見られる場合は一旦利確
4. 様子見・判断に迷う局面では半数決済（半数利確/半数損切り）
5. 時間軸ルール（10日ルール）:
   保有から10営業日/日以上経過しても明確なトレンドが出ない場合：
   - 含み益がある場合: 全清算を検討
   - 含み損がある場合: 半分損切りを検討
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any


@dataclass
class TargetHit:
  take_profit_hit: bool
  stop_loss_hit: bool


@dataclass
class PositionStatus:
  code: str
  entry_price: float
  current_price: float
  unrealized_return: float
  entry_date: str | None
  holding_days: int | None
  target_hit: TargetHit
  time_rule_triggered: bool
  alert_level: str  # "NONE" | "TAKE_PROFIT" | "STOP_LOSS" | "TIME_RULE" | "WATCH"
  alert_message: str
  recommended_rule_action: str

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


def calculate_holding_days(
  entry_date_val: str | date | datetime | None,
  current_date_val: str | date | datetime | None = None,
) -> int | None:
  """
  エントリー日と現在日の差分（保有日数）を計算する。
  """
  if entry_date_val is None:
    return None

  def _to_date(v: str | date | datetime) -> date:
    if isinstance(v, datetime):
      return v.date()
    if isinstance(v, date):
      return v
    if isinstance(v, str):
      # YYYY-MM-DD または YYYYMMDD または YYYY/MM/DD 等のパース
      clean_str = v.strip().replace("/", "-")
      if len(clean_str) == 8 and clean_str.isdigit():
        return datetime.strptime(clean_str, "%Y%m%d").date()
      return datetime.strptime(clean_str[:10], "%Y-%m-%d").date()
    raise ValueError(f"Unsupported date format: {v}")

  try:
    d_entry = _to_date(entry_date_val)
    d_current = (
      _to_date(current_date_val) if current_date_val is not None else date.today()
    )
    return max(0, (d_current - d_entry).days)
  except Exception:
    return None


def evaluate_position_rules(
  code: str,
  entry_price: float,
  current_price: float,
  entry_date: str | date | datetime | None = None,
  current_date: str | date | datetime | None = None,
  take_profit_threshold: float = 0.10,
  stop_loss_threshold: float = -0.05,
  time_rule_days: int = 10,
  no_trend_range: float = 0.03,
) -> PositionStatus:
  """
  単一ポジションの取引ルール状態を評価する。

  Parameters:
  -----------
  code : str
      銘柄コード
  entry_price : float
      買付価格
  current_price : float
      現在株価
  entry_date : str | date | datetime | None
      エントリー日
  current_date : str | date | datetime | None
      評価日（未指定時は本日）
  take_profit_threshold : float
      利確判定閾値（デフォルト: +10%）
  stop_loss_threshold : float
      損切り判定閾値（デフォルト: -5%）
  time_rule_days : int
      時間軸ルールの基準日数（デフォルト: 10日）
  no_trend_range : float
      10日ルールで「明確なトレンドなし」とみなす変動幅（デフォルト: ±3%）
  """
  if entry_price <= 0:
    raise ValueError(f"entry_price must be positive, got {entry_price}")

  unrealized_return = (current_price - entry_price) / entry_price
  holding_days = calculate_holding_days(entry_date, current_date)
  entry_date_str = str(entry_date) if entry_date is not None else None

  take_profit_hit = unrealized_return >= take_profit_threshold
  stop_loss_hit = unrealized_return <= stop_loss_threshold

  target_hit = TargetHit(
    take_profit_hit=take_profit_hit,
    stop_loss_hit=stop_loss_hit,
  )

  time_rule_triggered = False
  alert_level = "NONE"
  alert_message = ""
  recommended_action = "現状維持（ルール内の推移）"

  # 優先順位 1: 損切りライン到達 (-5%) -> 即時損切り
  if stop_loss_hit:
    alert_level = "STOP_LOSS"
    alert_message = "🚨 損切りライン到達（-5%到達）"
    recommended_action = (
      "直ちに全ポジション損切りを実行してください（即時損切りルール）。"
    )

  # 優先順位 2: 利確ライン到達 (+10%) -> ホールドしつつ反落監視
  elif take_profit_hit:
    alert_level = "TAKE_PROFIT"
    alert_message = "🎯 利確ライン到達（+10%到達・下降兆候監視）"
    recommended_action = (
      "利確目標（+10%）到達。即座に全利確せずホールド推奨ですが、"
      "反落・下降の兆候（RSI過熱・MA割れ等）が見られる場合は一旦利確（または半数利確）してください。"
    )

  # 優先順位 3: 10日ルール判定 (10日以上経過し、顕著な上昇/下落トレンドがない場合)
  elif holding_days is not None and holding_days >= time_rule_days:
    # 目標ラインに達しておらず、トレンドが限定的な場合
    if abs(unrealized_return) <= no_trend_range or not (
      take_profit_hit or stop_loss_hit
    ):
      time_rule_triggered = True
      alert_level = "TIME_RULE"
      if unrealized_return >= 0:
        alert_message = f"⏱️ 10日横ばいルール（保有{holding_days}日・含み益）"
        recommended_action = (
          f"保有から{holding_days}日経過し明確なトレンドが出ていません。"
          "含み益があるため、ルールに従い全清算を検討してください。"
        )
      else:
        alert_message = f"⏱️ 10日横ばいルール（保有{holding_days}日・含み損）"
        recommended_action = (
          f"保有から{holding_days}日経過し明確なトレンドが出ていません。"
          "含み損があるため、ルールに従い半数損切り（ポジション半減）を検討してください。"
        )

  # 優先順位 4: 様子見・通常推移
  else:
    alert_level = "NONE"
    alert_message = "正常範囲内"
    recommended_action = "ルール抵触なし。継続保有・監視。"

  return PositionStatus(
    code=code,
    entry_price=entry_price,
    current_price=current_price,
    unrealized_return=unrealized_return,
    entry_date=entry_date_str,
    holding_days=holding_days,
    target_hit=target_hit,
    time_rule_triggered=time_rule_triggered,
    alert_level=alert_level,
    alert_message=alert_message,
    recommended_rule_action=recommended_action,
  )
