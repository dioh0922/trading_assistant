from __future__ import annotations

from pathlib import Path

from src.rules.loader import load_rules_md
from src.rules.prompt_builder import (
  build_rules_system_prompt_section,
  build_position_status_prompt_note,
  build_full_system_prompt,
)


def test_load_rules_md_exists(tmp_path: Path):
  rule_file = tmp_path / "RULE.md"
  rule_file.write_text("# 取引ルール\n- リスクリワードは10%:5%\n", encoding="utf-8")

  content = load_rules_md(rule_file)
  assert "リスクリワードは10%:5%" in content


def test_load_rules_md_nonexistent(tmp_path: Path):
  non_existent = tmp_path / "NON_EXISTENT.md"
  content = load_rules_md(non_existent)
  assert content == ""


def test_build_rules_system_prompt_section():
  mock_rules = "- リスクリワードは10%:5%\n- 10日ルール"
  section = build_rules_system_prompt_section(mock_rules)
  assert "【ユーザーの取引規律（RULE.md）】" in section
  assert "- リスクリワードは10%:5%" in section


def test_build_position_status_prompt_note():
  pos_stat_take_profit = {
    "entry_price": 1000.0,
    "entry_date": "2026-08-20",
    "holding_days": 10,
    "unrealized_return": 0.12,
    "alert_level": "TAKE_PROFIT",
    "alert_message": "🎯 利確ライン到達",
    "recommended_rule_action": "利確目標到達",
  }
  note = build_position_status_prompt_note(pos_stat_take_profit)
  assert "利確目標(+10%)に到達しています" in note
  assert "+12.00%" in note
  assert "10日" in note

  pos_stat_stop_loss = {
    "entry_price": 1000.0,
    "unrealized_return": -0.06,
    "alert_level": "STOP_LOSS",
  }
  note_loss = build_position_status_prompt_note(pos_stat_stop_loss)
  assert "損切りライン(-5%)に到達しています" in note_loss


def test_build_full_system_prompt():
  prompt = build_full_system_prompt(rules_content="- テストルール")
  assert "【ユーザーの取引規律（RULE.md）】" in prompt
  assert "## 6. 取引ルール（RULE.md）に照らした推奨アクション" in prompt
  assert "【即座のアクション要否】" in prompt
