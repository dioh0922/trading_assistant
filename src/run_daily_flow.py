import subprocess
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
    data_raw_dir = project_root / "data" / "raw"
    reports_daily_dir = project_root / "reports" / "daily"
    data_raw_dir.mkdir(parents=True, exist_ok=True)
    reports_daily_dir.mkdir(parents=True, exist_ok=True)

    # 1. スクレイピングの実行 (直接 data/raw に出力)
    log.info("Step 1: Running scraping script...")
    scrape_cmd = ["python", str(project_root / "scraping" / "scraping.py"), "--output-dir", str(data_raw_dir)]
    if not run_command(scrape_cmd, cwd=project_root):
        log.error("Scraping failed.")
        return

    # 2. Task 2 ポジションチェックレポート生成の実行
    log.info("Step 2: Running positions alert checker...")
    check_report_path = project_root / "reports" / "daily_check.md"
    check_cmd = [
        "python", "-m", "src.pipeline.check_positions",
        "--positions", str(project_root / "train" / "positions.csv"),
        "--model-dir", "models/task2",
        "--reliability-table", "reliability_table.json",
        "--report-out", str(check_report_path)
    ]
    train_dir = project_root / "train"
    if not run_command(check_cmd, cwd=train_dir):
        log.error("Positions checker failed.")
        return

    log.info("Daily workflow completed successfully. Output: %s", check_report_path)

if __name__ == "__main__":
    main()

