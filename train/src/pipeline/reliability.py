from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def load_reliability_table(path: Path) -> dict:
  if path.exists():
    log.info("Loading reliability table from %s", path)
    with open(path, "r", encoding="utf-8") as f:
      return json.load(f)
  log.warning("Reliability table not found at %s", path)
  return {}


def lookup_reliability(
  table: dict, label: str, confidence: float
) -> tuple[float | None, int]:
  if label not in table:
    return None, 0

  label_data = table[label]
  applicable = []
  for k, info in label_data.items():
    try:
      th = float(k)
      if th <= confidence and info.get("support", 0) > 0:
        applicable.append((th, info))
    except ValueError:
      continue

  if not applicable:
    return None, 0

  best_threshold, best_info = max(applicable, key=lambda x: x[0])
  return best_info.get("precision"), best_info.get("support", 0)
