import os
from datetime import datetime
import yfinance as yf


def main():
  # CSV出力用ディレクトリの作成
  output_dir = "./csv/collect"
  os.makedirs(output_dir, exist_ok=True)
  error_log_path = os.path.join(output_dir, "error.log")

  # exist.txt から銘柄コード一覧を読み込む
  codes_path = os.path.join(os.path.dirname(__file__), "exist.txt")
  with open(codes_path, encoding="utf-8") as f:
    codes = [line.strip() for line in f if line.strip()]

  total = len(codes)
  success_count = 0
  failed_count = 0

  for i, code in enumerate(codes):
    ticker = code
    yf_ticker = f"{ticker}.T"

    success = False
    stock_data = None
    error_msg = ""

    # まずは 10年分 (10y) でダウンロードを試みる
    try:
      stock_data = yf.download(tickers=yf_ticker, period="10y", interval="1d")
      if not stock_data.empty:
        success = True
      else:
        error_msg = "データが取得できませんでした (period=10y)。"
    except Exception as e:
      error_msg = f"ダウンロード失敗 (period=10y): {e}"

    # 10y で失敗した場合は 5年分 (5y) で再試行
    if not success:
      try:
        stock_data = yf.download(tickers=yf_ticker, period="5y", interval="1d")
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
      # エラーログに書き出し
      with open(error_log_path, "a", encoding="utf-8") as log_f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_f.write(f"[{timestamp}] {yf_ticker}: {error_msg}\n")
      failed_count += 1

    # 進捗表示 (100件ごと)
    if (i + 1) % 100 == 0:
      print(f"進捗: {i + 1}/{total} (成功: {success_count}, 失敗: {failed_count})")

  print(f"完了 - 成功：{success_count}件, 失敗：{failed_count}件")


if __name__ == "__main__":
  main()
