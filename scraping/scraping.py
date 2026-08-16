import os
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path
import yfinance as yf


def fetch_name(yf_ticker, code):
  try:
    info = yf.Ticker(yf_ticker).info
    for key in ("shortName", "longName"):
      name = info.get(key)
      if name:
        return str(name).strip()
  except Exception:
    pass
  return code


def main():
  project_root = Path(__file__).resolve().parent.parent
  default_output_dir = project_root / "data" / "raw"

  parser = argparse.ArgumentParser(description="Scrape stock price data for positions.")
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=default_output_dir,
    help="Directory to save output CSVs and names.json",
  )
  args = parser.parse_args()

  output_dir = args.output_dir
  os.makedirs(output_dir, exist_ok=True)

  # positions.csv のパスを探す (ルート直下, train/, または positions.csv)
  candidates = [
    project_root / "positions.csv",
    project_root / "train" / "positions.csv",
    Path("positions.csv"),
  ]
  positions_file = None
  for cand in candidates:
    if cand.exists():
      positions_file = cand
      break

  if not positions_file:
    print("エラー: positions.csv が見つかりません。")
    return

  tickers = []
  # positions.csv から銘柄コード (code列) を読み込む
  with open(positions_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      code = row.get("code")
      if code:
        tickers.append(code.strip())

  if not tickers:
    print("対象の銘柄コードが positions.csv に記載されていません。")
    return

  error_log_path = os.path.join(output_dir, "error.log")

  # 既存の銘柄名マップを読み込み（キャッシュ）
  names_path = os.path.join(output_dir, "names.json")
  names = {}
  if os.path.exists(names_path):
    with open(names_path, encoding="utf-8") as f:
      names = json.load(f)

  success_count = 0
  failed_count = 0

  for ticker in tickers:
    # 数字のみ、または末尾が 'A' で他が数字の場合は東証銘柄とみなして .T を付与
    is_jpx = ticker.isdigit() or (
      ticker[:-1].isdigit() and ticker.upper().endswith("A")
    )
    yf_ticker = f"{ticker}.T" if is_jpx else ticker

    # 銘柄名の取得（キャッシュ済み以外のみ）
    if ticker not in names:
      names[ticker] = fetch_name(yf_ticker, ticker)

    success = False
    stock_data = None
    error_msg = ""

    # まずは 10年分 (10y) でダウンロードを試みる
    try:
      stock_data = yf.download(
        tickers=yf_ticker, period="10y", interval="1d", progress=False
      )
      if not stock_data.empty:
        success = True
      else:
        error_msg = "データが取得できませんでした (period=10y)。"
    except Exception as e:
      error_msg = f"ダウンロード失敗 (period=10y): {e}"

    # 10y で失敗した場合は 5年分 (5y) で再試行
    if not success:
      try:
        stock_data = yf.download(
          tickers=yf_ticker, period="5y", interval="1d", progress=False
        )
        if not stock_data.empty:
          success = True
        else:
          error_msg = "データが取得できませんでした (period=5y)。"
      except Exception as e:
        error_msg = f"ダウンロード失敗 (period=5y): {e}"

    if success:
      # CSVファイルに書き出し
      output_csv = os.path.join(output_dir, f"{ticker}.csv")
      stock_data.to_csv(output_csv)
      success_count += 1
    else:
      print(f"警告/エラー: {yf_ticker} のダウンロードに失敗しました。詳細: {error_msg}")
      # エラーログに書き出し
      with open(error_log_path, "a", encoding="utf-8") as log_f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_f.write(f"[{timestamp}] {yf_ticker}: {error_msg}\n")
      failed_count += 1

  # 銘柄名マップを保存
  with open(names_path, "w", encoding="utf-8") as f:
    json.dump(names, f, ensure_ascii=False, indent=2)
  print(f"銘柄名: {len(names)}件を {names_path} に保存しました。")

  print(f"成功：{success_count}件, 失敗：{failed_count}件")


if __name__ == "__main__":
  main()
