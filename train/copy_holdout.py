from pathlib import Path
import shutil
import sys

EXTRACT_PATH = Path("extract.txt")
RAW_DIR = Path("data/raw")
HOLDOUT_DIR = Path("data/holdout")


def main() -> None:
  if not EXTRACT_PATH.exists():
    print(f"Error: {EXTRACT_PATH} not found", file=sys.stderr)
    sys.exit(1)
  if not RAW_DIR.exists():
    print(f"Error: {RAW_DIR} not found", file=sys.stderr)
    sys.exit(1)

  HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

  tickers = [
    line.strip() for line in EXTRACT_PATH.read_text().splitlines() if line.strip()
  ]

  copied = 0
  not_found = []
  for t in tickers:
    src = RAW_DIR / f"{t}.csv"
    if src.exists():
      shutil.copy2(src, HOLDOUT_DIR / f"{t}.csv")
      copied += 1
    else:
      not_found.append(t)

  print(f"Copied {copied} files to {HOLDOUT_DIR}/")
  if not_found:
    print(f"Not found ({len(not_found)}): {', '.join(not_found)}", file=sys.stderr)


if __name__ == "__main__":
  main()
