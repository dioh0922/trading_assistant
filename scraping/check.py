
import os
from datetime import datetime
import yfinance as yf
from rich import print

def exist(info):
  if "shortName" in info:
    return True
  else:
    return False

def main():
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    exist_path = os.path.join(output_dir, "exist.txt")

    for code in range(0, 10000):
      ticker = f"{code:04d}"
      yf_ticker = f"{ticker}.T"
      STOCK = yf.Ticker(yf_ticker)
      if exist(STOCK.info):
        with open(exist_path, "a", encoding="utf-8") as log_f:
          log_f.write(f"{yf_ticker}\n")

if __name__ == "__main__":
    main()
