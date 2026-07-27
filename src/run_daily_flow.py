import subprocess
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("daily_flow")

def run_command(cmd: list[str], cwd: Path) -> bool:
    log.info("Running: %s in %s", " ".join(cmd), cwd)
    res = subprocess.run(cmd, cwd=cwd)
    return res.returncode == 0

def main():
    project_root = Path(__file__).resolve().parent.parent
    train_dir = project_root / "train"
    scraping_dir = project_root / "scraping"

    scrape_out_dir = scraping_dir / "csv"
    train_raw_dir = train_dir / "data" / "raw"

    # 1. スクレイピングの実行
    log.info("Step 1: Running scraping script...")
    if not run_command(["python", "scraping.py"], cwd=scraping_dir):
        log.error("Scraping failed.")
        return

    # 2. ディレクトリのズレを吸収 (ファイルのコピー)
    log.info("Step 2: Syncing downloaded CSV files to train raw data directory...")
    train_raw_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(scrape_out_dir.glob("*.csv"))
    if not csv_files:
        log.warning("No CSV files found in scraping output directory.")

    for csv_file in csv_files:
        dest_file = train_raw_dir / csv_file.name
        shutil.copy2(csv_file, dest_file)
        log.debug("Copied: %s -> %s", csv_file.name, dest_file)
    log.info("Successfully synced %d files.", len(csv_files))

    # 3. Task 2 ポジションチェックレポート生成の実行
    log.info("Step 3: Running positions alert checker...")
    check_cmd = [
        "python", "-m", "src.pipeline.check_positions",
        "--positions", "positions.csv",
        "--model-dir", "models/task2",
        "--reliability-table", "reliability_table.json",
        "--report-out", "reports/daily_check.md"
    ]
    if not run_command(check_cmd, cwd=train_dir):
        log.error("Positions checker failed.")
        return

    log.info("Daily workflow completed successfully.")

if __name__ == "__main__":
    main()
