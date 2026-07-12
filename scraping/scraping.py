import os
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

    for ticker in tickers:
        # 数字のみの場合は東証銘柄とみなして .T を付与
        yf_ticker = f"{ticker}.T" if ticker.isdigit() else ticker
        
        print(f"{yf_ticker} のデータをダウンロード中...")
        try:
            # 10年分の日足データを取得
            stock_data = yf.download(tickers=yf_ticker, period="10y", interval="1d")
            
            if stock_data.empty:
                print(f"警告: {yf_ticker} のデータが取得できませんでした。")
                continue
                
            # CSVファイルに書き出し
            output_csv = os.path.join(output_dir, f"{ticker}.csv")
            stock_data.to_csv(output_csv)
            print(f"成功: {output_csv} にデータを保存しました。")
        except Exception as e:
            print(f"エラー ({yf_ticker} のダウンロード失敗): {e}")

if __name__ == "__main__":
    main()
