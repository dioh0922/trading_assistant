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

### モードC: positions.csv の全銘柄から個別詳細JSONを一括生成
`positions.csv` に記載された全銘柄に対し、個別詳細分析データ（JSON）を一括でまとめて `reports/json/{銘柄コード}_detail.json` に出力します。

```bash
cd /app/train

python -m src.pipeline.analyze_ticker \
    --positions-csv positions.csv \
    --model-dir models/task2 \
    --reliability-table reliability_table.json
```

---

## 3. 良く使うコマンドまとめ

日常的な運用や実験で頻繁に使用する主要コマンドの一覧です。

### 1. 一括で銘柄データを更新する (`./scraping/collect_multi`)
スクレイピングモジュールを使用して、全銘柄の最新株価データ（CSV）を一括取得・更新します。
```bash
cd /app/scraping
python collect_multi.py
```
*(※全銘柄の最新データを同期・収集します)*

### 2. モデルに学習させる (`./train`)
収集したデータと特徴量パイプラインに基づき、Task2等の推論モデル（LightGBMアンサンブル等）の学習・評価を実行します。
```bash
cd /app/train
python quick_eval.py train --data-dir data/raw --model-dir models/task2
```
*(※ `--sample-tickers 10` などのオプションを指定して高速テストも可能です)*

### 3. positions.csvにある内容で現在の状況を分析する (`run_daily_flow`)
保有ポジション（`positions.csv`）に対して最新価格をあてはめ、予測結果と分析レポートを生成します。
```bash
cd /app/train
python -m src.pipeline.check_positions \
    --positions positions.csv \
    --model-dir models/task2 \
    --reliability-table reliability_table.json \
    --report-out reports/daily_check.md
```

### 4. positions.csv から全銘柄の個別詳細JSONを一括生成する
`positions.csv` 内の全銘柄の予測詳細データ（JSON）を `reports/json/` 以下に一括生成します。
```bash
cd /app/train
python -m src.pipeline.analyze_ticker \
    --positions-csv positions.csv \
    --model-dir models/task2 \
    --reliability-table reliability_table.json
```

### 5. experiment_shap.py batchで検証する
SHAP値を用いた特徴量の重要度分析や複数銘柄の検証バッチ処理を実行します。
```bash
cd /app/train
python experiment_shap.py batch
```

### 6. 校正テーブル (`reliability_table.json`) をホールドアウト評価で再生成する
最新のデータセットと学習済みモデルを用い、銘柄軸ホールドアウト評価を実行して確信度・精度校正テーブル（`reliability_table.json`）を更新・出力します。
```bash
cd /app/train
python evaluate_holdout.py \
    --model-dir models/task2 \
    --dataset-path data/datasets/task2_dataset.parquet \
    --split-col split_type \
    --split-value stock_holdout \
    --label-col label \
    --thresholds 0.5,0.6,0.7,0.8,0.9 \
    --report-out reports/holdout_eval.md \
    --table-out reliability_table.json
```
