from __future__ import annotations

import datetime
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import polars as pl
import pytest

from pipeline.config import AppConfig, SplitConfig, DataConfig, ModelConfig, LabelsConfig
from pipeline.train import run_training


@pytest.fixture
def temp_workspace():
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_train_pipeline(temp_workspace):
    # Setup directories
    datasets_dir = temp_workspace / "data" / "datasets"
    datasets_dir.mkdir(parents=True)
    models_dir = temp_workspace / "models"
    models_dir.mkdir(parents=True)
    
    # Create fake dataset for task2
    # must have: ticker, date, label, fold, split_type, sample_weight, feature_1, feature_2
    n_rows = 100
    dates = [datetime.date(2026, 1, 1) + datetime.timedelta(days=i) for i in range(n_rows)]
    
    # We'll use simple dates and generate features
    df = pl.DataFrame({
        "ticker": ["TICKER_A"] * n_rows,
        "date": dates,
        "feature_1": np.random.rand(n_rows),
        "feature_2": np.random.rand(n_rows),
        "label": ["timeout" if i % 3 == 0 else ("upper" if i % 3 == 1 else "lower") for i in range(n_rows)],
        "fold": [i % 3 for i in range(n_rows)],  # 3 folds
        "split_type": [["train", "valid", "time_holdout", "stock_holdout"][i % 4] for i in range(n_rows)],
        "sample_weight": [1.0] * n_rows,
    })
    
    # Save dummy dataset
    dataset_path = datasets_dir / "task2_dataset.parquet"
    df.write_parquet(dataset_path)

    # Create config yaml
    config_data = f"""
data:
  raw_dir: "{(temp_workspace / 'data' / 'raw').as_posix()}"
  processed_dir: "{(temp_workspace / 'data' / 'processed').as_posix()}"
  features_dir: "{(temp_workspace / 'data' / 'features').as_posix()}"
  labels_dir: "{(temp_workspace / 'data' / 'labels').as_posix()}"
  datasets_dir: "{datasets_dir.as_posix()}"
split:
  embargo_days: 2
  purge_days: 5
  time_holdout_months: 2
  stock_holdout_ratio: 0.2
  n_folds: 3
model:
  lightgbm:
    objective: "multiclass"
    learning_rate: 0.05
    num_leaves: 7
    n_estimators: 10
    early_stopping_rounds: 3
labels:
  task2:
    upper_barrier: 0.10
    lower_barrier: -0.05
    time_barrier_days: 60
"""
    config_path = temp_workspace / "config.yaml"
    with open(config_path, "w") as f:
        f.write(config_data)

    # We need to monkeypatch/mock models directory because run_training hardcodes Path("models") / task
    # To avoid writing to local workspace's "models", we can temporarily patch models path in run_training if needed,
    # or just let it write and clean up, but wait, run_training does:
    # `models_dir = Path("models") / task`
    # Let's temporarily change the working directory or patch Path in train.py.
    # Actually, we can just run it, but it will write to `models/task2`.
    # Let's run it and then clean up the `models/task2` directory created in cwd.
    
    # Run training
    # To run this test safely without messing with the workspace `models` directory,
    # we can monkeypatch `Path` or `run_training`'s internal path resolution if we edit it.
    # Let's edit train.py to support a custom output models directory, or resolve it relative to config?
    # Better yet, let's keep it simple and clean up models/task2 after execution in the test.
    import os
    original_cwd = os.getcwd()
    os.chdir(temp_workspace)
    try:
        # Create 'config' subdirectory since config is usually loaded from config/config.yaml or whatever
        (temp_workspace / "config").mkdir()
        shutil.copy(config_path, temp_workspace / "config" / "config.yaml")
        
        run_training(temp_workspace / "config" / "config.yaml", "task2")
        
        # Verify outputs
        out_models_dir = temp_workspace / "models" / "task2"
        assert out_models_dir.exists()
        assert (out_models_dir / "fold0.joblib").exists()
        assert (out_models_dir / "fold1.joblib").exists()
        assert (out_models_dir / "fold2.joblib").exists()
        assert (out_models_dir / "metadata.json").exists()
        assert (out_models_dir / "feature_importance.csv").exists()
        
        with open(out_models_dir / "metadata.json") as f:
            meta = json.load(f)
            assert meta["task"] == "task2"
            assert set(meta["feature_columns"]) == { "feature_1", "feature_2" }

        # Now test run_evaluation
        from pipeline.evaluate import run_evaluation
        # Make sure reports directory exists
        (temp_workspace / "reports").mkdir(exist_ok=True)
        run_evaluation(temp_workspace / "config" / "config.yaml", "task2")
        
        report_path = temp_workspace / "reports" / "task2_evaluation.md"
        assert report_path.exists()

        # Test predict.py
        from pipeline.predict import predict_exit
        features_dir = temp_workspace / "data" / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        feat_df = df.with_columns(pl.lit(100.0).alias("close"))
        feat_df.write_parquet(features_dir / "features_cross.parquet")
        
        predict_exit(temp_workspace / "config" / "config.yaml", "TICKER_A", entry_price=100.0)
    finally:
        os.chdir(original_cwd)
