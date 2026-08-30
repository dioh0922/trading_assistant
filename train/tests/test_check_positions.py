from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import polars as pl

from pipeline.check_positions import analyze_positions


def test_analyze_positions_with_rules(tmp_path: Path):
  # 1. 模擬 positions.csv
  pos_csv = tmp_path / "positions.csv"
  pos_csv.write_text(
    "code,entry_price,entry_date\n"
    "4503,1000.0,2026-08-01\n"
    "6857,1000.0,2026-08-01\n"
    "9104,1000.0,2026-08-25\n",
    encoding="utf-8",
  )

  # 2. 模擬 config / report paths
  config_path = tmp_path / "config.yaml"
  config_path.write_text("data:\n  raw_dir: .\n", encoding="utf-8")
  reliability_path = tmp_path / "reliability.json"
  reliability_path.write_text("{}", encoding="utf-8")
  report_path = tmp_path / "report.md"

  # 3. モック設定
  # ticker ごとに異なる株価・特徴量を返す
  def mock_get_latest_features(config, ticker, data_dir, feature_cols):
    price_map = {
      "4503": 980.0,  # -2% (10日超で含み損 -> 10日ルール半数損切り)
      "6857": 1120.0,  # +12% (+10%利確ライン到達)
      "9104": 940.0,  # -6% (-5%損切りライン到達)
    }
    close_price = price_map.get(ticker, 1000.0)
    return pl.DataFrame(
      {
        "date": ["2026-08-30"],
        "close": [close_price],
        "feat1": [1.0],
      }
    )

  mock_model = MagicMock()
  mock_model.predict_proba.return_value = [[0.2, 0.6, 0.2]]

  with patch(
    "pipeline.check_positions.load_ensemble_models"
  ) as mock_load_models, patch(
    "pipeline.check_positions.get_latest_features", side_effect=mock_get_latest_features
  ), patch("pipeline.check_positions.lookup_reliability", return_value=(0.7, 50)):
    mock_load_models.return_value = (
      [mock_model],
      {"feature_columns": ["feat1"], "label_classes": ["lower", "stay", "upper"]},
    )

    analyze_positions(
      positions_path=pos_csv,
      model_dir=tmp_path / "models",
      reliability_path=reliability_path,
      config_path=config_path,
      data_dir=None,
      report_path=report_path,
    )

  # 生成されたレポートの確認
  created_reports = list(tmp_path.glob("report_*.md"))
  assert len(created_reports) == 1
  content = created_reports[0].read_text(encoding="utf-8")

  # 各種アラート文言の確認
  assert "🚨 損切りライン到達(-5%): 即時損切り対象" in content
  assert "🎯 利確ライン到達(+10%): 下降兆候監視" in content
  assert "⏱️ 10日ルール発動: 半数損切り検討" in content
  assert "保有日数" in content
  assert "💡 <b>推奨:</b>" in content
