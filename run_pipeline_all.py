"""
run_pipeline_all.py
───────────────────
Trading Assistant の全プロセス（データ収集 → モデル学習 → 日次確認 → 詳細JSON生成 → LLM分析）
をワンコマンドで一括実行・個別ステップ実行するための統合エントリーポイント。

使用例:
  # 全フローをまとめて自動実行
  python run_pipeline_all.py --all

  # 銘柄6857について、詳細JSON生成からLLM分析まで実行
  python run_pipeline_all.py --steps json,llm --ticker 6857

  # LLM呼び出しをドライラン確認
  python run_pipeline_all.py --all --dry-run
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_pipeline_all")

PROJECT_ROOT = Path(__file__).resolve().parent

def run_step_scrape(output_dir: Path) -> bool:
    log.info("=== [Step 1/5] スクレイピング (データ収集) ===")
    script = PROJECT_ROOT / "scraping" / "scraping.py"
    cmd = [sys.executable, str(script), "--output-dir", str(output_dir)]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return res.returncode == 0

def run_step_train(force: bool = False) -> bool:
    log.info("=== [Step 2/5] モデル学習 & データ加工パイプライン ===")
    script = PROJECT_ROOT / "train" / "run_pipeline.py"
    cmd = [sys.executable, str(script)]
    if force:
        cmd.append("--force")
    res = subprocess.run(cmd, cwd=PROJECT_ROOT / "train")
    return res.returncode == 0

def run_step_daily() -> bool:
    log.info("=== [Step 3/5] 日次チェック (デイリーフロー) ===")
    script = PROJECT_ROOT / "src" / "run_daily_flow.py"
    cmd = [sys.executable, str(script)]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return res.returncode == 0

def run_step_json(ticker: str | None = None) -> bool:
    log.info("=== [Step 4/5] 個別詳細JSON生成 ===")
    json_dir = PROJECT_ROOT / "reports" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    out_json_target = json_dir / (f"{ticker}_detail.json" if ticker else "ticker_detail.json")

    cmd = [
        sys.executable, "-m", "src.pipeline.analyze_ticker",
        "--model-dir", "models/task2",
        "--reliability-table", "reliability_table.json",
        "--out-json", str(out_json_target)
    ]
    if ticker:
        cmd.extend(["--ticker", ticker])
    else:
        positions_csv = PROJECT_ROOT / "train" / "positions.csv"
        if not positions_csv.exists():
            positions_csv = PROJECT_ROOT / "positions.csv"
        cmd.extend(["--positions-csv", str(positions_csv)])

    res = subprocess.run(cmd, cwd=PROJECT_ROOT / "train")
    return res.returncode == 0


def _find_latest_daily_check() -> Path | None:
    """reports/ および train/reports/ から最新の daily_check*.md を探す"""
    candidates = []
    for search_dir in [PROJECT_ROOT / "reports", PROJECT_ROOT / "train" / "reports"]:
        if search_dir.exists():
            candidates.extend(search_dir.glob("daily_check*.md"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def run_step_llm(ticker: str | None = None, dry_run: bool = False) -> bool:
    log.info("=== [Step 5/5] LLM分析 & レポート生成 ===")
    script = PROJECT_ROOT / "src" / "llm_analyze.py"
    if ticker:
        out_path = PROJECT_ROOT / "reports" / "llm" / f"{ticker}_llm_analysis.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(script), "single",
            "--ticker", ticker,
            "--json-dir", str(PROJECT_ROOT / "reports" / "json"),
            "--output", str(out_path)
        ]
    else:
        daily_check_report = _find_latest_daily_check()
        if daily_check_report is None:
            log.error("日次レポート (daily_check*.md) が見つかりません。dailyステップを先に実行してください。")
            return False
        log.info("最新の日次レポートを使用: %s", daily_check_report)
        out_dir = PROJECT_ROOT / "reports" / "llm_batch"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(script), "batch",
            "--daily-check-report", str(daily_check_report),
            "--json-dir", str(PROJECT_ROOT / "reports" / "json"),
            "--output-dir", str(out_dir)
        ]

    if dry_run:
        cmd.append("--dry-run")

    res = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return res.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Trading Assistant 統合実行パイプライン")
    parser.add_argument("--all", action="store_true", help="日次一括フロー (scrape, daily, json, llm) を実行 (※モデル学習は除外)")
    parser.add_argument("--include-train", action="store_true", help="--all 実行時にモデル再学習 (train) も含める")
    parser.add_argument("--steps", type=str, default=None,
                        help="実行するステップをカンマ区切りで指定 (例: scrape,train,daily,json,llm)")
    parser.add_argument("--ticker", type=str, default=None, help="特定の銘柄コードのみ処理する場合に指定")
    parser.add_argument("--force", action="store_true", help="モデル学習を強制再実行する (--force)")
    parser.add_argument("--dry-run", action="store_true", help="LLM分析でAPI呼び出しを行わずプロンプトを表示のみ")

    args = parser.parse_args()

    if not args.all and not args.steps:
        parser.print_help()
        print("\n[エラー] --all または --steps のいずれかを指定してください。")
        sys.exit(1)

    steps_to_run = []
    if args.all:
        if args.include_train:
            steps_to_run = ["scrape", "train", "daily", "json", "llm"]
        else:
            steps_to_run = ["scrape", "daily", "json", "llm"]
    elif args.steps:
        steps_to_run = [s.strip().lower() for s in args.steps.split(",") if s.strip()]

    data_raw_dir = PROJECT_ROOT / "data" / "raw"

    if "scrape" in steps_to_run:
        if not run_step_scrape(data_raw_dir):
            log.error("スクレイピングステップでエラーが発生しました。")
            sys.exit(1)

    if "train" in steps_to_run:
        if not run_step_train(force=args.force):
            log.error("モデル学習ステップでエラーが発生しました。")
            sys.exit(1)

    if "daily" in steps_to_run:
        if not run_step_daily():
            log.error("日次チェックステップでエラーが発生しました。")
            sys.exit(1)

    if "json" in steps_to_run:
        if not run_step_json(ticker=args.ticker):
            log.error("詳細JSON生成ステップでエラーが発生しました。")
            sys.exit(1)

    if "llm" in steps_to_run:
        if not run_step_llm(ticker=args.ticker, dry_run=args.dry_run):
            log.error("LLM分析ステップでエラーが発生しました。")
            sys.exit(1)

    log.info("🎉 選択されたすべてのステップが正常に完了しました！")

if __name__ == "__main__":
    main()

