# 実行コマンド一覧

## データ加工パイプライン（run_pipeline.py）

```bash
cd /app/train

# フル実行（Phase1〜4、全タスクのラベル生成）
python run_pipeline.py

# 中断からの再開（既存出力をスキップ）
python run_pipeline.py --skip-existing

# 全フェーズを強制的に再実行
python run_pipeline.py --force

# 先頭N銘柄だけで試運転
python run_pipeline.py --sample-tickers 10

# 設定ファイルを指定
python run_pipeline.py --config config/config.yaml
```

引数:
| 引数 | 説明 |
|------|------|
| `--skip-existing` | 出力が既にあるフェーズをスキップ（再開用） |
| `--force` | `--skip-existing` と併用、全フェーズを再実行 |
| `--sample-tickers N` | 先頭N銘柄に限定 |
| `--config PATH` | configファイルのパス |

## 簡易評価（quick_eval.py）

```bash
cd /app/train

# 学習・評価してモデルを保存
python quick_eval.py train \
    --data-dir data/raw \
    --model-dir models/task2 \
    --upper 0.10 --lower -0.05 --max-days 60

# 銘柄数限定で試運転
python quick_eval.py train \
    --data-dir data/raw \
    --model-dir models/task2 \
    --sample-tickers 10

# 複数のtrain/test分割比率で比較
python quick_eval.py train \
    --data-dir data/raw \
    --model-dir models/task2 \
    --multi-split

# 保存済みモデルで1銘柄を予測
python quick_eval.py predict \
    --model-dir models/task2 \
    --data-dir data/raw \
    --ticker 7203
```

引数（train）:
| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--data-dir` | (必須) | CSVディレクトリ |
| `--model-dir` | `models/task2` | モデル保存先 |
| `--upper` | `0.10` | 利確ライン |
| `--lower` | `-0.05` | 損切りライン |
| `--max-days` | `60` | 時間バリア |
| `--sample-tickers` | 全銘柄 | 使用銘柄数制限 |
| `--multi-split` | off | 複数分割比率で評価 |

## データスクレイピング

```bash
cd /app/scraping

# 全銘柄（exist.txt記載の4126銘柄）一括ダウンロード
python collect.py

# 特定銘柄のみ
python scraping.py
```

## 銘柄軸ホールドアウト評価（evaluate_holdout.py）

```bash
cd /app/train

# 銘柄軸ホールドアウト評価の実行と校正テーブルの生成
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

引数:
| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--model-dir` | (必須) | fold*.joblib + metadata.jsonがあるディレクトリ |
| `--dataset-path` | (必須) | 特徴量・ラベル・split列を持つデータセットパス |
| `--split-col` | `split_type` | split種別が入っている列名 |
| `--split-value` | `stock_holdout` | 評価対象とするsplitの値 |
| `--label-col` | `label` | 正解ラベルの列名 |
| `--thresholds` | `0.5,0.6,0.7,0.8,0.9` | 確信度の閾値リスト（カンマ区切り） |
| `--report-out` | `None` | レポートの出力先パス |
| `--table-out` | `None` | 校正テーブルJSONの出力先パス |

## ポジションの一括チェック (check_positions.py)

```bash
cd /app/train

# 毎日の保有ポジション一括チェックとレポート生成
python -m src.pipeline.check_positions \
    --positions positions.csv \
    --model-dir models/task2 \
    --reliability-table reliability_table.json \
    --report-out reports/daily_check.md
```

引数:
| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--positions` | `positions.csv` | 保有ポジション情報CSVのパス |
| `--model-dir` | `models/task2` | Task 2アンサンブルモデルが格納されているディレクトリ |
| `--reliability-table` | `reliability_table.json` | 確信度・精度の校正テーブルJSONのパス |
| `--report-out` | `reports/daily_check.md` | 一括チェックレポートMarkdownの出力先パス |
| `--config` | `config/config.yaml` | 設定ファイルYAMLのパス |
| `--data-dir` | `None` | 特徴量データディレクトリ (指定がない場合は config の設定に基づく) |

## 個別深掘り用データの作成 (analyze_ticker.py)

```bash
cd /app/train

# 特定銘柄の詳細予測・特徴量分析データの出力 (JSON)
python -m src.pipeline.analyze_ticker \
    --ticker 6981 \
    --entry-price 9066.0 \
    --model-dir models/task2 \
    --reliability-table reliability_table.json \
    --out-json reports/6981_detail.json
```

引数:
| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--ticker` | (必須) | 分析対象の銘柄コード |
| `--entry-price` | `None` | ポジション取得価格 (指定すると含み損益を計算) |
| `--model-dir` | `models/task2` | Task 2アンサンブルモデルが格納されているディレクトリ |
| `--reliability-table` | `reliability_table.json` | 確信度・精度の校正テーブルJSONのパス |
| `--out-json` | `reports/{ticker}_detail.json` | 出力先JSONパス (実際にはファイル名末尾にタイムスタンプが付与されます) |
| `--config` | `config/config.yaml` | 設定ファイルYAMLのパス |
| `--data-dir` | `None` | 特徴量データディレクトリ (指定がない場合は config の設定に基づく) |

## テスト

```bash
cd /app/train

# 全テスト
python -m pytest tests/

# 詳細表示
python -m pytest tests/ -v

# 特定のテストファイル
python -m pytest tests/test_ingest.py
```

## パイプラインの構成

```
Phase 1:  CSV → Parquet           ingest.py           → prices.parquet
Phase 2:  品質チェック            quality_check.py    → prices_clean.parquet
Phase 3a: 単一銘柄特徴量          features_single.py  → features_single.parquet
Phase 3b: クロスセクション特徴量  features_cross.py   → features_cross.parquet
Phase 4:  ラベル生成              labels.py
  Task1:  トレンド予測                                 → labels_task1.parquet
  Task2:  利確/損切り                                  → labels_task2.parquet ❌未生成
  Task3:  目標価格到達                                 → labels_task3.parquet ❌未生成
```
