from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    features_dir: str = "data/features"
    labels_dir: str = "data/labels"
    datasets_dir: str = "data/datasets"


class FeaturesConfig(BaseModel):
    windows: list[int] = Field(default_factory=lambda: [5, 20, 60])
    rsi_period: int = 14
    atr_period: int = 14


class Task2Config(BaseModel):
    upper_barrier: float = 0.10
    lower_barrier: float = -0.05
    time_barrier_days: int = 60


class Task3Config(BaseModel):
    time_barrier_days: int = 60


class Task1Config(BaseModel):
    horizon_days: int = 20
    up_threshold: float = 0.03
    down_threshold: float = -0.03


class LabelsConfig(BaseModel):
    task2: Task2Config = Field(default_factory=Task2Config)
    task3: Task3Config = Field(default_factory=Task3Config)
    task1: Task1Config = Field(default_factory=Task1Config)


class SplitConfig(BaseModel):
    embargo_days: int = 10
    purge_days: int = 60
    time_holdout_months: int = 6
    stock_holdout_ratio: float = 0.125
    n_folds: int = 5


class LightGBMConfig(BaseModel):
    objective: str = "multiclass"
    learning_rate: float = 0.05
    num_leaves: int = 63
    n_estimators: int = 1000
    early_stopping_rounds: int = 50


class ModelConfig(BaseModel):
    lightgbm: LightGBMConfig = Field(default_factory=LightGBMConfig)


class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)


def load_config(config_path: Path) -> AppConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
