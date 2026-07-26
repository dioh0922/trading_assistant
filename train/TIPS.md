# ポジション分析ツールの活用TIPS

本ドキュメントでは、学習済みモデルに対して最新の株価CSVデータを適用して推論・分析を行う手順と、その内部仕様および注意点についてまとめています。

---

## 1. 最新のCSVをあてた分析の仕組み

本システムの分析モジュール（`check_positions.py`, `analyze_ticker.py`）は、**「最新価格データからリアルタイムに特徴量を生成して推論する」**仕組みが備わっています。

### 処理フロー
1. **CSVファイルの検出**:
   指定されたディレクトリ（デフォルトは設定ファイルの `raw_dir` または `--data-dir` で指定したパス）から、対象銘柄の `{銘柄コード}.csv` をロードします。
2. **動的特徴量の計算**:
   読み込んだCSVデータを元に、単一銘柄のテクニカル特徴量（RSI、ATR、ボラティリティ、移動平均乖離率、直近リターンなど）をその場で動的に計算します。
3. **最新日の抽出と推論**:
   時系列データの最終行（最新日）を抽出し、保存済みの学習済みモデル（`models/task2` のアンサンブルモデル）に入力して予測を行います。

---

## 2. 実行コマンド

最新のCSVファイルを配置した上で、以下のコマンドを実行します。

### モードA: 保有ポジションの一括チェック
`positions.csv` に記載された全銘柄に対し、指定フォルダの最新のCSVを適用して分析レポートを生成します。

```bash
cd /app/train

python -m src.pipeline.check_positions \
    --positions positions.csv \
    --data-dir data/raw \
    --model-dir models/task2 \
    --reliability-table reliability_table.json \
    --report-out reports/daily_check.md
```

### モードB: 特定銘柄の個別深掘り
特定の銘柄コードを指定し、詳細な予測確率、過去の精度、および特徴量の重要度順リストを含んだJSONファイルを出力します。

```bash
cd /app/train

python -m src.pipeline.analyze_ticker \
    --ticker 1301 \
    --entry-price 3000.0 \
    --data-dir data/raw \
    --model-dir models/task2 \
    --reliability-table reliability_table.json \
    --out-json reports/1301_detail.json
```

---

## 3. 注意点とより正確な分析のためのTips

### ① クロスセクション特徴量の補完（フォールバック）
CSV単体から動的に特徴量を計算する場合、**他銘柄との比較が必要な特徴量**（例: 全銘柄中の出来高パーセンタイルを示す `volume_rank_pct` や、セクター平均に対するリターンを示す `sector_relative_return`）は算出できません。

* **仕様**:
  スクリプトはこれらを自動的に検知し、中央値や基準値となるデフォルト値（例: ランク系は `0.5`、相対リターン系は `0.0`）で補完して推論を実行します。そのため、CSV単体でもエラーにならずに動作します。

### ② より厳密な推論を行いたい場合（キャッシュの活用）
全銘柄を含めた最新のクロスセクション特徴量を正しく反映させた高精度な推論を行いたい場合は、以下の手順を踏んでください。

1. **株価データの更新**:
   `/app/scraping` にて `collect.py` 等を実行し、全銘柄の最新CSVを `data/raw` に同期します。
2. **パイプラインによる特徴量キャッシュの再生成**:
   以下のコマンドを実行し、特徴量の全体キャッシュである `features_cross.parquet`（または `features_single.parquet`）を更新します。
   ```bash
   python run_pipeline.py
   ```
3. **分析コマンドの実行（`--data-dir` を指定しない）**:
   `--data-dir` を省略して実行すると、優先的に生成された Parquet キャッシュから最新行をロードするため、補完値（デフォルト値）ではない、真のセクター相対リターンやパーセンタイルランクを用いた厳密な予測が行われます。
   ```bash
   python -m src.pipeline.check_positions \
       --positions positions.csv \
       --model-dir models/task2 \
       --reliability-table reliability_table.json
   ```
