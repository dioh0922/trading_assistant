from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.preprocessing import LabelEncoder

from pipeline.config import load_config

log = logging.getLogger("train")


def build_train_eval_data(
    dataset_df: pl.DataFrame,
    task: str,
    label_encoder: LabelEncoder | None = None,
) -> tuple[list[str], pl.DataFrame, LabelEncoder | None]:
    # Determine non-feature columns
    meta_cols = {
        "ticker",
        "date",
        "label",
        "fold",
        "split_type",
        "sample_weight",
        "days_to_hit",
        "target_return",
        "horizon_days",
    }
    feature_cols = [c for c in dataset_df.columns if c not in meta_cols]
    feature_cols.sort()

    log.info("Features (%d): %s", len(feature_cols), feature_cols)

    # Process targets
    le = label_encoder
    if task in ("task1", "task2"):
        if le is None:
            # Encode label strings to integers
            le = LabelEncoder()
            # Clean labels (no empty or null)
            labels = dataset_df["label"].to_list()
            encoded = le.fit_transform(labels)
        else:
            labels = dataset_df["label"].to_list()
            encoded = le.transform(labels)
        dataset_df = dataset_df.with_columns(pl.Series("target", encoded))
    elif task == "task3_hit":
        # Target: 1 if upper, 0 if timeout
        dataset_df = dataset_df.with_columns(
            pl.when(pl.col("label") == "upper")
            .then(1)
            .otherwise(0)
            .alias("target")
        )
    elif task == "task3_days":
        # Target: days_to_hit (regression)
        # Only keep rows where target hit (label is upper)
        dataset_df = dataset_df.filter(pl.col("label") == "upper")
        dataset_df = dataset_df.with_columns(pl.col("days_to_hit").alias("target"))
        # Drop rows with null target
        dataset_df = dataset_df.filter(pl.col("target").is_not_null())
    else:
        raise ValueError(f"Unknown task: {task}")

    return feature_cols, dataset_df, le


def run_training(
    config_path: Path,
    task: str,
) -> None:
    config = load_config(config_path)
    datasets_dir = Path(config.data.datasets_dir)
    models_dir = Path("models") / task
    models_dir.mkdir(parents=True, exist_ok=True)

    # Resolve actual dataset path
    # task3_hit and task3_days use the same task3_dataset.parquet
    dataset_name = "task3" if task in ("task3_hit", "task3_days") else task
    dataset_path = datasets_dir / f"{dataset_name}_dataset.parquet"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    log.info("Loading dataset info from %s", dataset_path)
    if dataset_path.is_file():
        # Fallback for single file (e.g. in tests)
        df = pl.read_parquet(dataset_path)
        feature_cols, df, label_encoder = build_train_eval_data(df, task)
        has_weights = "sample_weight" in df.columns
        is_directory_split = False
    else:
        # Directory split (production memory-efficient mode)
        stock_holdout_path = dataset_path / "stock_holdout.parquet"
        time_holdout_path = dataset_path / "time_holdout.parquet"
        if not stock_holdout_path.exists() or not time_holdout_path.exists():
            raise FileNotFoundError(f"Dataset splits missing in {dataset_path}")

        sh_df = pl.read_parquet(stock_holdout_path)
        th_df = pl.read_parquet(time_holdout_path)

        # Determine feature columns
        feature_cols, _, _ = build_train_eval_data(sh_df, task)
        has_weights = "sample_weight" in sh_df.columns

        # Build consistent label encoder if multiclass
        label_encoder = None
        if task in ("task1", "task2"):
            # Gather all unique labels across holdouts to cover the entire space
            sh_labels = sh_df["label"].unique().to_list()
            th_labels = th_df["label"].unique().to_list()
            unique_labels = sorted(list(set(sh_labels + th_labels)))
            label_encoder = LabelEncoder()
            label_encoder.fit(unique_labels)
            log.info("Fit LabelEncoder with classes: %s", label_encoder.classes_)
        is_directory_split = True
        # メタデータ抽出後はsh_df/th_dfを解放
        del sh_df, th_df

    n_folds = config.split.n_folds
    lgb_config = config.model.lightgbm

    importances = []

    for k in range(n_folds):
        log.info("--- Fold %d / %d ---", k + 1, n_folds)
        if is_directory_split:
            train_path = dataset_path / f"fold_{k}_train.parquet"
            valid_path = dataset_path / f"fold_{k}_valid.parquet"

            if not train_path.exists():
                log.warning("Fold %d train data does not exist. Skipping.", k)
                continue

            train_df = pl.read_parquet(train_path)
            valid_df = pl.read_parquet(valid_path) if valid_path.exists() else pl.DataFrame()

            _, train_df, _ = build_train_eval_data(train_df, task, label_encoder=label_encoder)
            if len(train_df) == 0:
                log.warning("Fold %d train data is empty. Skipping.", k)
                continue

        else:
            # Fallback for single file
            train_df = df.filter((pl.col("fold") == k) & (pl.col("split_type") == "train"))
            valid_df = df.filter((pl.col("fold") == k) & (pl.col("split_type") == "valid"))
            if len(train_df) == 0:
                log.warning("Fold %d train data is empty. Skipping.", k)
                continue

        X_train = train_df.select(feature_cols).to_pandas()
        y_train = train_df["target"].to_numpy()
        w_train = train_df["sample_weight"].to_numpy() if has_weights else None

        # Build model
        if task in ("task1", "task2"):
            model = LGBMClassifier(
                objective="multiclass",
                num_class=len(label_encoder.classes_),
                learning_rate=lgb_config.learning_rate,
                num_leaves=lgb_config.num_leaves,
                n_estimators=lgb_config.n_estimators,
                class_weight="balanced",
                verbosity=-1,
            )
        elif task == "task3_hit":
            model = LGBMClassifier(
                objective="binary",
                learning_rate=lgb_config.learning_rate,
                num_leaves=lgb_config.num_leaves,
                n_estimators=lgb_config.n_estimators,
                class_weight="balanced",
                verbosity=-1,
            )
        else:  # task3_days
            model = LGBMRegressor(
                objective="regression",
                learning_rate=lgb_config.learning_rate,
                num_leaves=lgb_config.num_leaves,
                n_estimators=lgb_config.n_estimators,
                verbosity=-1,
            )

        # Early stopping setup
        callbacks = [
            log_evaluation(period=0),  # Suppress logs
        ]
        
        if len(valid_df) > 0:
            _, valid_df, _ = build_train_eval_data(valid_df, task, label_encoder=label_encoder)
            X_val = valid_df.select(feature_cols).to_pandas()
            y_val = valid_df["target"].to_numpy()
            callbacks.append(early_stopping(stopping_rounds=lgb_config.early_stopping_rounds, verbose=False))
            eval_set = [(X_val, y_val)]
        else:
            eval_set = None

        fit_params = {}
        if w_train is not None:
            fit_params["sample_weight"] = w_train

        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
            **fit_params
        )

        # Save fold model
        fold_model_path = models_dir / f"fold{k}.joblib"
        joblib.dump(model, fold_model_path)
        log.info("Saved model to %s", fold_model_path)

        # Store feature importance
        importances.append(model.feature_importances_)

    # Calculate and save feature importance
    if importances:
        mean_importance = np.mean(importances, axis=0)
        fi_df = pl.DataFrame({
            "feature": feature_cols,
            "importance": mean_importance,
        }).sort("importance", descending=True)
        fi_path = models_dir / "feature_importance.csv"
        fi_df.write_csv(fi_path)
        log.info("Saved feature importance to %s", fi_path)

    # Save metadata
    barrier_config = {}
    if task == "task1":
        barrier_config = config.labels.task1.model_dump()
    elif task == "task2":
        barrier_config = config.labels.task2.model_dump()
    elif task in ("task3_hit", "task3_days"):
        barrier_config = config.labels.task3.model_dump()

    label_classes = None
    if label_encoder is not None:
        label_classes = list(label_encoder.classes_)
    elif task == "task3_hit":
        label_classes = ["timeout", "upper"]

    metadata = {
        "task": task,
        "feature_columns": feature_cols,
        "label_classes": label_classes,
        "barrier_config": barrier_config,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_info": {
            "n_folds": n_folds,
            "has_weights": has_weights,
        }
    }
    
    metadata_path = models_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    log.info("Saved metadata to %s", metadata_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6: Model Training")
    parser.add_argument("--task", choices=["task1", "task2", "task3_hit", "task3_days"], required=True, help="Task to train")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"), help="Config file path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_training(args.config, args.task)
