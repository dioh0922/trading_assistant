# LLM_TIPS.md — LLM連携 運用手順まとめ

このプロジェクトでLLM(Claude)を使った銘柄分析を行うための、環境準備から日次運用・トラブルシューティングまでの手順をまとめる。

---

## 0. 全体の位置づけ

```
[毎日] check_positions.py (モードA・一括チェック、LLM不使用)
           ↓ flagが立った銘柄
       analyze_ticker.py (モードB・個別深掘りJSON生成)
           ↓
       llm_analyze.py single/batch (LLMに解釈させる)

[随時] experiment_shap.py analyze/batch (SHAP検証、モデルの説明力チェック)
           ↓
       aggregate_batch_stats.py (複数run統合・統計検定)
           ↓ combined_records.csv
       llm_analyze.py が explanation_reliability の根拠として参照
```

**モードA(一括チェック)ではLLMを呼ばない**。flagが立った銘柄だけ、モードB→LLMに進む。（`position_analysis_policy.md`参照）

---

## 1. 環境準備

### 1.1 APIキー

環境変数に`ANTHROPIC_API_KEY`を設定する。

```bash
# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Linux/Mac
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 1.2 必要なライブラリ

```bash
pip install anthropic pandas scipy
```

### 1.3 APIキーなしでの動作確認

`llm_analyze.py`は`--dry-run`オプションを持つ。**APIキーを設定していなくても、プロンプトが正しく組み立てられるかはこれで確認できる。** 新しくコードを書き換えたときは、まず`--dry-run`で動作確認してからAPIキーありで実行するのが安全。

```bash
python llm_analyze.py single --ticker 6098 --dry-run
```

---

## 2. 日次運用の手順

### Step 1: 一括チェック（LLM不使用）

```bash
python -m pipeline.check_positions --positions positions.csv --report-out reports/daily_check_YYYYMMDD_HHMMSS.md
```

`positions.csv`（`code,entry_price`の2列）を読み、保有銘柄全件の予測・含み損益・flagを一覧化する。

### Step 2: 個別深掘りJSONの生成（flagが立った銘柄のみ）

```bash
python -m pipeline.analyze_ticker --ticker 6098 --entry-price 12605.0 --out-json reports/json/6098_detail_YYYYMMDD_HHMMSS.json
```

`reports/json/`配下に`{code}_detail_{timestamp}.json`の形式で保存する（この命名規則は`llm_analyze.py`の自動検索が前提にしている）。

### Step 3: LLM分析

**a) daily_checkのレポートから「要注意銘柄」を自動抽出して一括分析**

```bash
python llm_analyze.py batch \
    --daily-check-report reports/daily_check_YYYYMMDD_HHMMSS.md \
    --json-dir reports/json \
    --combined-records reports/combined_records.csv \
    --output-dir reports/llm_batch/YYYYMMDD
```

`reports/llm_batch/YYYYMMDD/`配下に、銘柄ごとの分析結果(`{code}_llm_analysis.md`)と`summary.md`がまとまって出力される。

**b) 気になった銘柄を1件だけ深掘り**

```bash
python llm_analyze.py single --ticker 6098 \
    --combined-records reports/combined_records.csv \
    --output reports/6098_llm_analysis.md
```

`--ticker`だけ指定すればよく、`reports/json/`内で最新タイムスタンプのJSONが自動的に選ばれる。ファイルパスを手打ちする必要はない。

---

## 3. SHAP検証・校正データの更新手順（随時／月次目安）

モデルを再学習したときや、ある程度日数が経ったときに実施する。

### Step 1: 個別JSONに対してSHAP比較を実行

```bash
python experiment_shap.py analyze \
    --model-dir models/task2 \
    --input-json reports/json/6098_detail_YYYYMMDD_HHMMSS.json \
    --top-k 10 \
    --report-out reports/shap_experiment_6098.md
```

### Step 2: 複数JSONをまとめて一致度分布を集計

```bash
python experiment_shap.py batch \
    --model-dir models/task2 \
    --input-dir reports/json \
    --top-k 8
```

`batch/run_YYYYMMDD_HHMMSS/`配下に`records.csv` / `summary.md` / `histogram.png`が出力される（実行のたびに新しいタイムスタンプフォルダが作られ、過去の結果は消えない）。

### Step 3: 複数run分を統合し、正式な統計検定を行う

```bash
python aggregate_batch_stats.py \
    --batch-dir batch \
    --report-out reports/aggregate_stats.md \
    --combined-csv-out reports/combined_records.csv
```

ここで生成される`combined_records.csv`が、`llm_analyze.py`の`--combined-records`に渡すファイルになる。**このステップを踏まないと、`llm_analyze.py`はハードコードされた古いフォールバック値（upper 0.717 / lower 0.587, 2026-08-08時点）を使い続ける**ので、モデル再学習後は忘れずに回す。

---

## 4. ファイルの役割一覧

| ファイル | 役割 | LLMを呼ぶか |
|---|---|---|
| `check_positions.py` | 保有ポジション一括チェック、flag付け | 呼ばない |
| `analyze_ticker.py` | 1銘柄の深掘り用JSON生成（predict.py相当＋feature importance） | 呼ばない |
| `experiment_shap.py` | SHAP値とglobal_importanceの比較検証 | 呼ばない |
| `aggregate_batch_stats.py` | 複数SHAP検証runの統合・統計検定 | 呼ばない |
| `llm_analyze.py` | JSON＋校正データをLLMに渡して解釈させる | **呼ぶ** |
| `reliability_table.json` | 確信度帯ごとの過去精度（predict.py/check_positions.py用） | - |
| `combined_records.csv` | SHAP一致度の蓄積データ（llm_analyze.pyのexplanation_reliability算出に使用） | - |

---

## 5. プロンプト設計の要点（`llm_analyze.py`のSYSTEM_PROMPTに実装済み）

以下は`llm_integration_policy.md`・`shap_reliability_llm_integration_ideas.md`で検討し、実装に反映済みの内容。プロンプトを直接編集する際は、これらのガードレールを削らないよう注意する。

1. 上位2クラスの確率差が小さい（目安10pt未満）場合は「五分五分に近い」と明言させる
2. `historical_precision`が50%に近い・`historical_support`が小さい場合は数値を割り引いて解釈させる
3. `global_importance`は「モデル全体の傾向」であり「この1件の因果的な説明」ではないと明示する
4. **確信度の高さとSHAP説明の信頼性(`explanation_reliability`)は無関係**（統計検証済み、疑似相関だった）
5. 予測ラベルの日次比較ではなく、確率の推移（特にupper/lower差）を見るよう指示する
6. 投資助言ではなく状況整理の参考情報である旨を明記する

---

## 6. 過去に見つかったバグ・詰まりやすいポイント

同じパターンが再発しやすいので、新しい機能を追加・修正する際はチェックリストとして使う。

### 6.1 校正ロジックの重複実装によるバグ再発（2回発生）

`reliability_table`の参照ロジック（しきい値未満なら「該当なし」を返す）が、モードA（`check_positions.py`）とモードB（`analyze_ticker.py`）に**別々に実装されていて、片方だけ直して再発**したことが2回あった。ロジックは1箇所に共通化し、両方から呼び出す構成にすること。

### 6.2 整数エンコードされたラベルとの不一致（2回発生）

foldモデルが整数（0,1,2）でラベルを学習している場合、`model.classes_`に文字列（'upper'等）が存在せず`.index()`が失敗する。`metadata.json`の`label_classes`を使って変換する必要がある（`evaluate_holdout.py`・`experiment_shap.py`の両方で対応済み。新しくモデルを読み込む処理を書くときは同じ変換処理を入れること）。

### 6.3 ラベル計算のO(n^2)実装によるフリーズ

`compute_sample_weights`が銘柄内で総当たりループになっていたため、4000銘柄規模で終わらなかった。時間窓が絡む計算は、まず計算量（O(n^2)になっていないか）を確認する。

### 6.4 モデル再学習後、校正テーブルだけ古いまま

モデルを再学習しても`reliability_table.json`や`combined_records.csv`を更新し忘れると、予測は新しいのに参考精度だけ古い、という不整合が起きる。本ドキュメントの3節を再学習のたびに実施すること。

### 6.5 単純な相関に注意（シンプソンのパラドックス）

「confidenceが高いほどSHAP一致度が下がる」という相関が、生データをプールすると有意に見えたが、銘柄ごとに分解すると符号がバラバラで、疑似相関だった。**銘柄をまたいだ集計をするときは、必ず`aggregate_batch_stats.py`の「銘柄クラスタを考慮した検定」も併用し、素朴な相関だけで結論を出さないこと。**

---

## 7. よくある質問

**Q. `combined_records.csv`が無い状態でも`llm_analyze.py`は動く？**
A. 動く。`--combined-records`を省略すると、ハードコードされたフォールバック値（`POPULATION_RELIABILITY`定数）を使う。ただし値は2026-08-08時点のものなので、可能な限り最新の`combined_records.csv`を渡すことが望ましい。

**Q. `reports/json/`に同じ銘柄のJSONが複数あるとどうなる？**
A. ファイル名に埋め込まれたタイムスタンプ（`YYYYMMDD_HHMMSS`）が最も新しいものが自動的に選ばれる。

**Q. batchで一部の銘柄だけ処理に失敗したら？**
A. 失敗した銘柄はエラー内容とともに`summary.md`に記録され、他の銘柄の処理は継続する。処理全体が止まることはない。