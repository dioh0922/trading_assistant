from __future__ import annotations

from datetime import date

from pipeline.rules import (
  calculate_holding_days,
  evaluate_position_rules,
)


def test_calculate_holding_days():
  assert calculate_holding_days("2026-08-01", "2026-08-11") == 10
  assert calculate_holding_days("20260801", "20260811") == 10
  assert calculate_holding_days(date(2026, 8, 1), date(2026, 8, 15)) == 14
  assert calculate_holding_days(None) is None
  assert calculate_holding_days("invalid-date") is None


def test_evaluate_position_rules_stop_loss():
  # -5% 以下の損切り
  status = evaluate_position_rules(
    code="9104",
    entry_price=1000.0,
    current_price=940.0,  # -6.0%
    entry_date="2026-08-01",
    current_date="2026-08-05",
  )
  assert status.target_hit.stop_loss_hit is True
  assert status.target_hit.take_profit_hit is False
  assert status.alert_level == "STOP_LOSS"
  assert "損切り" in status.recommended_rule_action
  assert status.holding_days == 4


def test_evaluate_position_rules_take_profit():
  # +10% 以上の利確目標到達
  status = evaluate_position_rules(
    code="6857",
    entry_price=1000.0,
    current_price=1110.0,  # +11.0%
    entry_date="2026-08-01",
    current_date="2026-08-05",
  )
  assert status.target_hit.take_profit_hit is True
  assert status.target_hit.stop_loss_hit is False
  assert status.alert_level == "TAKE_PROFIT"
  assert "利確" in status.recommended_rule_action
  assert "ホールド" in status.recommended_rule_action


def test_evaluate_position_rules_10day_gain():
  # 10日経過で含み益 (+2%)
  status = evaluate_position_rules(
    code="4452",
    entry_price=1000.0,
    current_price=1020.0,  # +2.0%
    entry_date="2026-08-01",
    current_date="2026-08-11",  # 10日
  )
  assert status.time_rule_triggered is True
  assert status.alert_level == "TIME_RULE"
  assert "全清算" in status.recommended_rule_action
  assert status.holding_days == 10


def test_evaluate_position_rules_10day_loss():
  # 10日経過で含み損 (-2%)
  status = evaluate_position_rules(
    code="4503",
    entry_price=1000.0,
    current_price=980.0,  # -2.0%
    entry_date="2026-08-01",
    current_date="2026-08-12",  # 11日
  )
  assert status.time_rule_triggered is True
  assert status.alert_level == "TIME_RULE"
  assert "半数損切り" in status.recommended_rule_action
  assert status.holding_days == 11


def test_evaluate_position_rules_normal():
  # 5日経過で+3%（正常推移）
  status = evaluate_position_rules(
    code="7011",
    entry_price=1000.0,
    current_price=1030.0,  # +3.0%
    entry_date="2026-08-01",
    current_date="2026-08-06",
  )
  assert status.time_rule_triggered is False
  assert status.target_hit.take_profit_hit is False
  assert status.target_hit.stop_loss_hit is False
  assert status.alert_level == "NONE"
  assert "正常範囲内" in status.alert_message
