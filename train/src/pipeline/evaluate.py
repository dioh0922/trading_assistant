from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import (
  accuracy_score,
  classification_report,
  confusion_matrix,
  f1_score,
  mean_absolute_error,
  roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from pipeline.config import load_config
from pipeline.train import build_train_eval_data

log = logging.getLogger("evaluate")


def load_ensemble_models(models_dir: Path) -> tuple[list[joblib.load], dict]:
  metadata_path = models_dir / "metadata.json"
  if not metadata_path.exists():
    raise FileNotFoundError(f"Metadata not found: {metadata_path}")

  with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

  n_folds = metadata["train_info"]["n_folds"]
  models = []
  for k in range(n_folds):
    model_path = models_dir / f"fold{k}.joblib"
    if not model_path.exists():
      raise FileNotFoundError(f"Model file not found: {model_path}")
    models.append(joblib.load(model_path))

  return models, metadata


def generate_report_multiclass(
  task: str,
  y_true_time: np.ndarray,
  y_pred_time: np.ndarray,
  y_true_stock: np.ndarray,
  y_pred_stock: np.ndarray,
  classes: list[str],
  output_path: Path,
) -> None:
  # F1 score macro
  f1_time = f1_score(y_true_time, y_pred_time, average="macro", zero_division=0)
  f1_stock = f1_score(y_true_stock, y_pred_stock, average="macro", zero_division=0)

  # Accuracy
  acc_time = accuracy_score(y_true_time, y_pred_time)
  acc_stock = accuracy_score(y_true_stock, y_pred_stock)

  # Detailed report
  report_time = classification_report(
    y_true_time, y_pred_time, target_names=classes, zero_division=0
  )
  report_stock = classification_report(
    y_true_stock, y_pred_stock, target_names=classes, zero_division=0
  )

  # Confusion Matrix
  cm_time = confusion_matrix(y_true_time, y_pred_time)
  cm_stock = confusion_matrix(y_true_stock, y_pred_stock)

  # Markdown format
  lines = [
    f"# Evaluation Report for {task}",
    "",
    "## Summary Metrics",
    "",
    "| Metric | Time Holdout | Stock Holdout | Difference (Time - Stock) |",
    "|---|---|---|---|",
    f"| Accuracy | {acc_time:.4f} | {acc_stock:.4f} | {acc_time - acc_stock:+.4f} |",
    f"| Macro F1 | {f1_time:.4f} | {f1_stock:.4f} | {f1_time - f1_stock:+.4f} |",
    "",
    "## Time Holdout",
    "### Classification Report",
    "```",
    report_time,
    "```",
    "### Confusion Matrix (Row: Actual, Col: Predicted)",
    "```",
    "          " + "".join(f"{c:>10s}" for c in classes),
  ]
  for lbl, row in zip(classes, cm_time):
    lines.append(f"{lbl:>10s}" + "".join(f"{v:10d}" for v in row))
  lines.append("```")
  lines.append("")

  lines.extend(
    [
      "## Stock Holdout",
      "### Classification Report",
      "```",
      report_stock,
      "```",
      "### Confusion Matrix (Row: Actual, Col: Predicted)",
      "```",
      "          " + "".join(f"{c:>10s}" for c in classes),
    ]
  )
  for lbl, row in zip(classes, cm_stock):
    lines.append(f"{lbl:>10s}" + "".join(f"{v:10d}" for v in row))
  lines.append("```")

  # For task2, explicitly report "upper" predicted as "lower"
  if task == "task2" and "upper" in classes and "lower" in classes:
    upper_idx = classes.index("upper")
    lower_idx = classes.index("lower")
    # Mistake: True upper, Pred lower
    mistakes_time = cm_time[upper_idx, lower_idx]
    mistakes_stock = cm_stock[upper_idx, lower_idx]
    lines.extend(
      [
        "",
        "## Critical Exit Failure Analysis (Task 2)",
        "- **Time Holdout**: 利確すべき(upper)なのに損切り(lower)と誤判定した件数: "
        f"{mistakes_time} 件 / {np.sum(cm_time[upper_idx])} 件 ("
        f"{mistakes_time / max(1, np.sum(cm_time[upper_idx])) * 100:.1f}%)",
        "- **Stock Holdout**: 利確すべき(upper)なのに損切り(lower)と誤判定した件数: "
        f"{mistakes_stock} 件 / {np.sum(cm_stock[upper_idx])} 件 ("
        f"{mistakes_stock / max(1, np.sum(cm_stock[upper_idx])) * 100:.1f}%)",
      ]
    )

  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
  log.info("Wrote evaluation report to %s", output_path)


def generate_report_binary(
  task: str,
  y_true_time: np.ndarray,
  y_prob_time: np.ndarray,
  y_true_stock: np.ndarray,
  y_prob_stock: np.ndarray,
  output_path: Path,
) -> None:
  auc_time = roc_auc_score(y_true_time, y_prob_time)
  auc_stock = roc_auc_score(y_true_stock, y_prob_stock)

  y_pred_time = (y_prob_time >= 0.5).astype(int)
  y_pred_stock = (y_prob_stock >= 0.5).astype(int)

  acc_time = accuracy_score(y_true_time, y_pred_time)
  acc_stock = accuracy_score(y_true_stock, y_pred_stock)

  report_time = classification_report(
    y_true_time, y_pred_time, target_names=["timeout", "upper"], zero_division=0
  )
  report_stock = classification_report(
    y_true_stock, y_pred_stock, target_names=["timeout", "upper"], zero_division=0
  )

  lines = [
    f"# Evaluation Report for {task}",
    "",
    "## Summary Metrics",
    "",
    "| Metric | Time Holdout | Stock Holdout | Difference (Time - Stock) |",
    "|---|---|---|---|",
    f"| Accuracy | {acc_time:.4f} | {acc_stock:.4f} | {acc_time - acc_stock:+.4f} |",
    f"| AUC | {auc_time:.4f} | {auc_stock:.4f} | {auc_time - auc_stock:+.4f} |",
    "",
    "## Time Holdout",
    "### Classification Report",
    "```",
    report_time,
    "```",
    "",
    "## Stock Holdout",
    "### Classification Report",
    "```",
    report_stock,
    "```",
  ]

  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
  log.info("Wrote evaluation report to %s", output_path)


def generate_report_regression(
  task: str,
  y_true_time: np.ndarray,
  y_pred_time: np.ndarray,
  y_true_stock: np.ndarray,
  y_pred_stock: np.ndarray,
  output_path: Path,
) -> None:
  mae_time = mean_absolute_error(y_true_time, y_pred_time)
  mae_stock = mean_absolute_error(y_true_stock, y_pred_stock)

  lines = [
    f"# Evaluation Report for {task}",
    "",
    "## Summary Metrics",
    "",
    "| Metric | Time Holdout | Stock Holdout | Difference (Time - Stock) |",
    "|---|---|---|---|",
    f"| Mean Absolute Error (MAE) | {mae_time:.4f} | {mae_stock:.4f} | {mae_time - mae_stock:+.4f} |",
    "",
    "## Time Holdout Statistics",
    f"- Target mean: {np.mean(y_true_time):.2f} days",
    f"- Pred mean  : {np.mean(y_pred_time):.2f} days",
    f"- MAE        : {mae_time:.4f} days",
    "",
    "## Stock Holdout Statistics",
    f"- Target mean: {np.mean(y_true_stock):.2f} days",
    f"- Pred mean  : {np.mean(y_pred_stock):.2f} days",
    f"- MAE        : {mae_stock:.4f} days",
  ]

  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
  log.info("Wrote evaluation report to %s", output_path)


def run_evaluation(
  config_path: Path,
  task: str,
) -> None:
  config = load_config(config_path)
  models_dir = Path("models") / task

  models, metadata = load_ensemble_models(models_dir)
  feature_cols = metadata["feature_columns"]
  label_classes = metadata.get("label_classes")

  datasets_dir = Path(config.data.datasets_dir)
  dataset_name = "task3" if task in ("task3_hit", "task3_days") else task
  dataset_path = datasets_dir / f"{dataset_name}_dataset.parquet"

  log.info("Loading dataset for evaluation: %s", dataset_path)
  if dataset_path.is_file():
    # Fallback for single parquet file (e.g. in tests)
    df = pl.read_parquet(dataset_path)

    # Build consistent label encoder if multiclass
    label_encoder = None
    if task in ("task1", "task2"):
      unique_labels = sorted(df["label"].unique().to_list())
      label_encoder = LabelEncoder()
      label_encoder.fit(unique_labels)
      log.info(
        "Fit LabelEncoder for evaluation with classes: %s", label_encoder.classes_
      )

    # Re-apply the same pipeline target extraction logic to align labels
    _, df, _ = build_train_eval_data(df, task, label_encoder=label_encoder)

    # Filter into time_holdout and stock_holdout
    time_df = df.filter(pl.col("split_type") == "time_holdout")
    stock_df = df.filter(pl.col("split_type") == "stock_holdout")
  else:
    # Directory split (production memory-efficient mode)
    time_path = dataset_path / "time_holdout.parquet"
    stock_path = dataset_path / "stock_holdout.parquet"

    if not time_path.exists() or not stock_path.exists():
      raise FileNotFoundError(f"Evaluation holdouts missing in {dataset_path}")

    time_df = pl.read_parquet(time_path)
    stock_df = pl.read_parquet(stock_path)

    # Build consistent label encoder if multiclass
    label_encoder = None
    if task in ("task1", "task2"):
      sh_labels = stock_df["label"].unique().to_list()
      th_labels = time_df["label"].unique().to_list()
      unique_labels = sorted(list(set(sh_labels + th_labels)))
      label_encoder = LabelEncoder()
      label_encoder.fit(unique_labels)
      log.info(
        "Fit LabelEncoder for evaluation with classes: %s", label_encoder.classes_
      )

    # Re-apply the same pipeline target extraction logic to align labels
    _, time_df, _ = build_train_eval_data(time_df, task, label_encoder=label_encoder)
    _, stock_df, _ = build_train_eval_data(stock_df, task, label_encoder=label_encoder)

  log.info("Time holdout size: %d, Stock holdout size: %d", len(time_df), len(stock_df))

  if len(time_df) == 0 or len(stock_df) == 0:
    log.error("Time or Stock holdout dataset is empty. Cannot evaluate.")
    return

  # Predictions
  X_time = time_df.select(feature_cols).to_pandas()
  y_time = time_df["target"].to_numpy()

  X_stock = stock_df.select(feature_cols).to_pandas()
  y_stock = stock_df["target"].to_numpy()

  report_path = Path("reports") / f"{task}_evaluation.md"

  if task in ("task1", "task2"):
    # Ensemble predictions (multiclass)
    probas_time = [m.predict_proba(X_time) for m in models]
    avg_proba_time = np.mean(probas_time, axis=0)
    pred_time = np.argmax(avg_proba_time, axis=1)

    probas_stock = [m.predict_proba(X_stock) for m in models]
    avg_proba_stock = np.mean(probas_stock, axis=0)
    pred_stock = np.argmax(avg_proba_stock, axis=1)

    generate_report_multiclass(
      task, y_time, pred_time, y_stock, pred_stock, label_classes, report_path
    )

  elif task == "task3_hit":
    # Ensemble predictions (binary classification)
    # Class 1 is 'upper'
    probas_time = [m.predict_proba(X_time)[:, 1] for m in models]
    avg_proba_time = np.mean(probas_time, axis=0)

    probas_stock = [m.predict_proba(X_stock)[:, 1] for m in models]
    avg_proba_stock = np.mean(probas_stock, axis=0)

    generate_report_binary(
      task, y_time, avg_proba_time, y_stock, avg_proba_stock, report_path
    )

  elif task == "task3_days":
    # Ensemble predictions (regression)
    preds_time = [m.predict(X_time) for m in models]
    avg_pred_time = np.mean(preds_time, axis=0)

    preds_stock = [m.predict(X_stock) for m in models]
    avg_pred_stock = np.mean(preds_stock, axis=0)

    generate_report_regression(
      task, y_time, avg_pred_time, y_stock, avg_pred_stock, report_path
    )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Phase 7: Evaluation")
  parser.add_argument(
    "--task",
    choices=["task1", "task2", "task3_hit", "task3_days"],
    required=True,
    help="Task to evaluate",
  )
  parser.add_argument(
    "--config", type=Path, default=Path("config/config.yaml"), help="Config file path"
  )
  args = parser.parse_args()

  logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )
  run_evaluation(args.config, args.task)
