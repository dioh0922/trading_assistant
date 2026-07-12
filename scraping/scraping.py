import os
from datetime import datetime
import yfinance as yf

def main():
    target_file = "target.txt"
    if not os.path.exists(target_file):
        print(f"エラー: {target_file} が見つかりません。")
        return

    # target.txt から銘柄コードを読み込む
    with open(target_file, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    if not tickers:
        print("対象の銘柄コードが target.txt に記載されていません。")
        return

    # CSV出力用ディレクトリの作成
    output_dir = "./csv"
    os.makedirs(output_dir, exist_ok=True)
    error_log_path = os.path.join(output_dir, "error.log")

    success_count = 0
    failed_count = 0

    for ticker in tickers:
        # 数字のみ、または末尾が 'A' で他が数字の場合は東証銘柄とみなして .T を付与
        is_jpx = ticker.isdigit() or (ticker[:-1].isdigit() and ticker.upper().endswith('A'))
        yf_ticker = f"{ticker}.T" if is_jpx else ticker
        
        # print(f"{yf_ticker} のデータをダウンロード中...")
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
                # print(f"{yf_ticker} のデータを period=5y で再試行中...")
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
            #print(f"成功: {output_csv} にデータを保存しました。")
            success_count += 1
        else:
            print(f"警告/エラー: {yf_ticker} のダウンロードに失敗しました。詳細: {error_msg}")
            # エラーログに書き出し
            with open(error_log_path, "a", encoding="utf-8") as log_f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_f.write(f"[{timestamp}] {yf_ticker}: {error_msg}\n")
            failed_count += 1

    print(f"成功：{success_count}件, 失敗：{failed_count}件")

if __name__ == "__main__":
    main()
