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
