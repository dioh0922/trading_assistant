import os
import time
import argparse
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

MAX_WORKERS = 8
MAX_RETRIES = 3

lock = threading.Lock()
success_count = 0
failed_count = 0
skipped_count = 0
completed_count = 0
output_dir_path = None


def download_with_retry(yf_ticker):
    for attempt in range(MAX_RETRIES):
        for period in ("10y", "5y"):
            try:
                stock_data = yf.download(tickers=yf_ticker, period=period, interval="1d", progress=False)
                if not stock_data.empty:
                    return stock_data
            except Exception:
                pass
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 * (2 ** attempt))
    return None


def process_code(code):
    global success_count, failed_count, skipped_count, completed_count

    yf_ticker = f"{code}.T"
    output_csv = os.path.join(output_dir_path, f"{code}.csv")

    if os.path.exists(output_csv):
        with lock:
            skipped_count += 1
            completed_count += 1
            if completed_count % 100 == 0:
                print(f"進捗: {completed_count}/{total} (成功: {success_count}, 失敗: {failed_count}, スキップ: {skipped_count})")
        return

    error_msg = ""
    try:
        stock_data = download_with_retry(yf_ticker)
    except Exception as e:
        stock_data = None
        error_msg = f"ダウンロード失敗: {e}"

    if stock_data is not None:
        stock_data.to_csv(output_csv)
        with lock:
            success_count += 1
    else:
        if not error_msg:
            error_msg = "データが取得できませんでした (10y/5y)。"
        with lock:
            error_log_path = os.path.join(output_dir_path, "error.log")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(error_log_path, "a", encoding="utf-8") as log_f:
                log_f.write(f"[{timestamp}] {yf_ticker}: {error_msg}\n")
            failed_count += 1

    with lock:
        completed_count += 1
        if completed_count % 100 == 0:
            print(f"進捗: {completed_count}/{total} (成功: {success_count}, 失敗: {failed_count}, スキップ: {skipped_count})")


def main():
    global total, output_dir_path

    project_root = Path(__file__).resolve().parent.parent
    default_output_dir = project_root / "data" / "raw"

    parser = argparse.ArgumentParser(description="Multi-ticker batch scraper")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="Directory to save output CSVs")
    args = parser.parse_args()

    output_dir_path = str(args.output_dir)
    os.makedirs(output_dir_path, exist_ok=True)

    codes_path = os.path.join(os.path.dirname(__file__), "exist.txt")
    with open(codes_path, encoding="utf-8") as f:
        codes = [line.strip() for line in f if line.strip()]

    total = len(codes)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_code, code) for code in codes]
        for _ in as_completed(futures):
            pass

    print(f"完了 - 成功：{success_count}件, 失敗：{failed_count}件, スキップ：{skipped_count}件")


if __name__ == "__main__":
    main()

