"""
src.rules
─────────
ユーザー取引ルール（RULE.md）の動的読み込みとプロンプト用コンテキスト生成を行うモジュール。
"""

from src.rules.loader import load_rules_md, find_rule_md_path
from src.rules.prompt_builder import (
  build_rules_system_prompt_section,
  build_position_status_prompt_note,
  build_full_system_prompt,
)

__all__ = [
  "load_rules_md",
  "find_rule_md_path",
  "build_rules_system_prompt_section",
  "build_position_status_prompt_note",
  "build_full_system_prompt",
]
