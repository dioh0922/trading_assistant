from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline.config import load_config
from pipeline.features_cross import build_features_cross
from pipeline.features_single import build_features_single
from pipeline.ingest import ingest_all
from pipeline.labels import build_labels_task1, build_labels_task2, build_labels_task3
from pipeline.quality_check import run_quality_check
from pipeline.dataset_split import build_dataset_split
from pipeline.train import run_training
from pipeline.evaluate import run_evaluation

log = logging.getLogger("run_pipeline")


def run_pipeline(
    config_path: Path,
    sample_tickers: int | None = None,
    skip_existing: bool = False,
    force: bool = False,
) -> None:
    config = load_config(config_path)

    raw_dir = Path(config.data.raw_dir)
    processed_dir = Path(config.data.processed_dir)
    features_dir = Path(config.data.features_dir)
    labels_dir = Path(config.data.labels_dir)

    prices_path = processed_dir / "prices.parquet"
    prices_clean_path = processed_dir / "prices_clean.parquet"
    flags_path = Path("reports") / "quality_check_flags.csv"
    features_single_path = features_dir / "features_single.parquet"
    features_cross_path = features_dir / "features_cross.parquet"
    labels_task1_path = labels_dir / "labels_task1.parquet"
    labels_task2_path = labels_dir / "labels_task2.parquet"
    labels_task3_path = labels_dir / "labels_task3.parquet"

    should_skip = skip_existing and not force

    # Phase 1
    if should_skip and prices_path.exists():
        log.info("Phase 1 [ingest]: skip (exists: %s)", prices_path)
    else:
        log.info("Phase 1 [ingest]: CSV -> %s", prices_path)
        ingest_all(raw_dir, prices_path, sample_tickers=sample_tickers)

    # Phase 2
    if should_skip and prices_clean_path.exists():
        log.info("Phase 2 [quality_check]: skip (exists: %s)", prices_clean_path)
    else:
        log.info("Phase 2 [quality_check]: %s -> %s", prices_path, prices_clean_path)
        run_quality_check(prices_path, prices_clean_path, flags_path)

    # Phase 3a
    if should_skip and features_single_path.exists():
        log.info("Phase 3a [features_single]: skip (exists: %s)", features_single_path)
    else:
        log.info("Phase 3a [features_single]: %s -> %s", prices_clean_path, features_single_path)
        build_features_single(
            prices_clean_path,
            features_single_path,
            windows=config.features.windows,
            rsi_period=config.features.rsi_period,
            atr_period=config.features.atr_period,
        )

    # Phase 3b
    if should_skip and features_cross_path.exists():
        log.info("Phase 3b [features_cross]: skip (exists: %s)", features_cross_path)
    else:
        log.info("Phase 3b [features_cross]: %s + %s -> %s", features_single_path, prices_clean_path, features_cross_path)
        build_features_cross(features_single_path, prices_clean_path, features_cross_path)

    # Phase 4
    if should_skip and labels_task1_path.exists():
        log.info("Phase 4 task1 [labels]: skip (exists: %s)", labels_task1_path)
    else:
        log.info("Phase 4 task1 [labels]: %s -> %s", prices_clean_path, labels_task1_path)
        build_labels_task1(
            prices_clean_path,
            labels_task1_path,
            horizon_days=config.labels.task1.horizon_days,
            up_threshold=config.labels.task1.up_threshold,
            down_threshold=config.labels.task1.down_threshold,
        )

    if should_skip and labels_task2_path.exists():
        log.info("Phase 4 task2 [labels]: skip (exists: %s)", labels_task2_path)
    else:
        log.info("Phase 4 task2 [labels]: %s -> %s", prices_clean_path, labels_task2_path)
        build_labels_task2(
            prices_clean_path,
            labels_task2_path,
            upper=config.labels.task2.upper_barrier,
            lower=config.labels.task2.lower_barrier,
            max_days=config.labels.task2.time_barrier_days,
        )

    if should_skip and labels_task3_path.exists():
        log.info("Phase 4 task3 [labels]: skip (exists: %s)", labels_task3_path)
    else:
        log.info("Phase 4 task3 [labels]: %s -> %s", prices_clean_path, labels_task3_path)
        build_labels_task3(
            prices_clean_path,
            labels_task3_path,
            max_days=config.labels.task3.time_barrier_days,
        )

    # Phase 5: Dataset split
    datasets_dir = Path(config.data.datasets_dir)
    tasks = ["task1", "task2", "task3"]
    for t in tasks:
        dataset_path = datasets_dir / f"{t}_dataset.parquet"
        if should_skip and dataset_path.exists():
            log.info("Phase 5 %s [dataset_split]: skip (exists: %s)", t, dataset_path)
        else:
            log.info("Phase 5 %s [dataset_split]: labels_%s.parquet -> %s", t, t, dataset_path)
            labels_path = labels_dir / f"labels_{t}.parquet"
            features_path = features_cross_path if features_cross_path.exists() else features_single_path
            build_dataset_split(
                features_path=features_path,
                labels_path=labels_path,
                output_path=dataset_path,
                config=config.split,
            )

    # Phase 6: Model training
    train_tasks = ["task1", "task2", "task3_hit", "task3_days"]
    for t in train_tasks:
        metadata_path = Path("models") / t / "metadata.json"
        if should_skip and metadata_path.exists():
            log.info("Phase 6 %s [train]: skip (exists: %s)", t, metadata_path)
        else:
            log.info("Phase 6 %s [train]: training models...", t)
            run_training(config_path, t)

    # Phase 7: Evaluation
    for t in train_tasks:
        report_path = Path("reports") / f"{t}_evaluation.md"
        if should_skip and report_path.exists():
            log.info("Phase 7 %s [evaluate]: skip (exists: %s)", t, report_path)
        else:
            log.info("Phase 7 %s [evaluate]: generating evaluation report...", t)
            run_evaluation(config_path, t)

    log.info("Pipeline complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stock-ml pipeline phases 1-4")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"), help="config file path")
    parser.add_argument("--sample-tickers", type=int, default=None, help="limit to first N tickers")
    parser.add_argument("--skip-existing", action="store_true", default=False, help="skip phases with existing output")
    parser.add_argument("--force", action="store_true", default=False, help="re-run even with --skip-existing")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        run_pipeline(
            config_path=args.config,
            sample_tickers=args.sample_tickers,
            skip_existing=args.skip_existing,
            force=args.force,
        )
    except Exception:
        log.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
