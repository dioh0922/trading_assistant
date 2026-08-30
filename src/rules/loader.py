"""
src.rules.loader
────────────────
RULE.md の動的ローダーモジュール。
プロジェクトルートまたは指定パスから RULE.md を検索・読み込みます。
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("rules_loader")

# デフォルトの探索パス候補
DEFAULT_RULE_PATHS = [
  Path("RULE.md"),
  Path("../RULE.md"),
  Path(__file__).resolve().parents[2] / "RULE.md",
]


def find_rule_md_path(custom_path: Path | str | None = None) -> Path | None:
  """
  RULE.md のパスを探索して返す。
  見つからない場合は None を返す。
  """
  if custom_path is not None:
    p = Path(custom_path)
    if p.exists():
      return p.resolve()
    log.warning("指定されたルールファイルが見つかりません: %s", custom_path)
    return None

  for p in DEFAULT_RULE_PATHS:
    try:
      if p.exists():
        return p.resolve()
    except Exception:
      continue

  return None


def load_rules_md(custom_path: Path | str | None = None) -> str:
  """
  RULE.md の内容を文字列として読み込む。
  ファイルが存在しない場合は空文字列を返す。
  """
  rule_path = find_rule_md_path(custom_path)
  if rule_path is None:
    log.warning("RULE.md が見つかりませんでした。取引ルールなしで継続します。")
    return ""

  try:
    content = rule_path.read_text(encoding="utf-8")
    log.info("RULE.md を読み込みました (%s, %d bytes)", rule_path, len(content))
    return content.strip()
  except Exception as e:
    log.warning("RULE.md の読み込みに失敗しました (%s): %s", rule_path, e)
    return ""
