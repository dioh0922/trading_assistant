from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import polars as pl

from pipeline.analyze_ticker import analyze_ticker


def test_analyze_ticker_with_position_status(tmp_path: Path):
  # 1. 模擬 config
  config_path = tmp_path / "config.yaml"
  config_path.write_text("data:\n  raw_dir: .\n", encoding="utf-8")
  reliability_path = tmp_path / "reliability.json"
  reliability_path.write_text("{}", encoding="utf-8")
  out_json = tmp_path / "6857_detail.json"

  # 2. 特徴量モック
  mock_df = pl.DataFrame(
    {
      "date": ["2026-08-30"],
      "close": [1120.0],
      "feat1": [1.0],
    }
  )

  mock_model = MagicMock()
  mock_model.predict_proba.return_value = [[0.1, 0.2, 0.7]]
  mock_model.feature_importances_ = [1.0]

  with patch("pipeline.analyze_ticker.load_ensemble_models") as mock_load_models, patch(
    "pipeline.analyze_ticker.get_latest_features", return_value=mock_df
  ), patch("pipeline.analyze_ticker.lookup_reliability", return_value=(0.8, 100)):
    mock_load_models.return_value = (
      [mock_model],
      {
        "feature_columns": ["feat1"],
        "label_classes": ["lower", "stay", "upper"],
        "barrier_config": {"take_profit": 0.10, "stop_loss": -0.05},
      },
    )

    analyze_ticker(
      ticker="6857",
      entry_price=1000.0,
      model_dir=tmp_path / "models",
      reliability_path=reliability_path,
      config_path=config_path,
      data_dir=None,
      out_json_path=out_json,
      no_timestamp=True,
      entry_date="2026-08-01",
    )

  assert out_json.exists()
  with open(out_json, encoding="utf-8") as f:
    data = json.load(f)

  assert data["ticker"] == "6857"
  assert data["latest_close"] == 1120.0
  assert data["entry_price"] == 1000.0
  assert "position_status" in data
  pos_stat = data["position_status"]
  assert pos_stat is not None
  assert pos_stat["entry_price"] == 1000.0
  assert pos_stat["entry_date"] == "2026-08-01"
  assert pos_stat["holding_days"] is not None
  assert pos_stat["take_profit_hit"] is True
  assert pos_stat["stop_loss_hit"] is False
  assert pos_stat["alert_level"] == "TAKE_PROFIT"
  assert "利確" in pos_stat["recommended_rule_action"]
