# モデル学習を省略・縮小する場合の設計案

作成日: 2026-06-30  
対象: `trading_assistant` Step1〜9

## 1. 方針

現状のモデル評価では、Walk-ForwardのAUCが安定しておらず、機械学習モデルを売買判断の中心に置くには根拠が弱い。  
そのため、当面はモデル学習を省略または縮小し、以下の構成へ寄せる。

```text
特徴量作成
  ↓
ルールベース判定
  ↓
過去類似局面の統計検証
  ↓
スコアリング
  ↓
人間が読めるレポート
```

この方式では、MLモデルの予測確率ではなく、過去データ上で確認できる勝率・平均リターン・PF・最大損失を判断の中心に置く。

## 2. なぜMLを縮小するか

現在の課題:

- 銘柄単位の学習データが少ない。
- foldごとのAUCが安定していない。
- 分類精度が売買期待値に直結していない。
- モデルの出力理由が説明しづらい。
- ルールとモデルとDDフィルタが分離しており、最終判断の意味がぼやけている。

MLを縮小するメリット:

- 判断理由が説明しやすい。
- 過学習を避けやすい。
- 条件ごとの勝率・期待値を直接確認できる。
- 銘柄スキャン時に軽く動く。
- LLMに渡す説明材料も明確になる。

## 3. 推奨する新構成

```text
Step1: 特徴量作成
Step2: ドメイン特徴量
Step3: ラベル作成
Step4: モデル学習ではなく、過去統計分析に置き換え
Step5: ルール + スコアリング
Step6: 統計テーブルによる期待値フィルタ
Step7: レポート
Step8: スキャン
Step9: 目標到達推定
Step10以降: LLMは説明・監査役
```

`step4_model.py` は完全削除せず、まずは以下のような分析モジュールに置き換えるのがよい。

```text
step4_edge_analyzer.py
```

出力:

- `condition_stats.csv`
- `signal_quality.csv`
- `score_threshold_report.csv`

## 4. 方式A: 完全ルールベース

MLを使わず、Step1〜3で作成した特徴量を使って条件判定する。

### 基本ルール

```text
除外条件:
  - assist_signal == 警戒
  - atr_percentile >= 0.90
  - is_overbought_heat == 1

エントリー候補:
  - RSIが35〜50
  - dev_ma25_zscoreが極端に悪くない
  - ATRパーセンタイルが0.90未満
  - 出来高比率が1.0以上
  - 週足トレンドが上昇、または短期反転条件あり
```

### メリット

- 最も説明しやすい。
- 実装が簡単。
- バグや未来情報混入を見つけやすい。

### デメリット

- 条件が硬くなりやすい。
- 複雑な組み合わせを拾いにくい。
- 銘柄ごとの癖を吸収しにくい。

## 5. 方式B: スコアリング方式

各条件に点数を付け、合計点で判断する。  
MLではないが、条件ごとの強弱を表現できる。

### スコア例

```text
entry_score =
  +2  weekly_trend == 1
  +1  35 <= rsi14 <= 50
  +1  volume_ratio >= 1.0
  +1  dev_ma25_zscore が5日前より改善
  +1  atr_dev_ma25 < 0.5
  -2  atr_percentile >= 0.90
  -2  is_overbought_heat == 1
  -1  weekly_trend == -1
  -1  rsi14 >= 65
```

判断:

```text
entry_score >= 3: エントリー候補
entry_score 1〜2: 監視
entry_score <= 0: 見送り
```

### 追加列

`step5_dataset.csv` または `step6_dataset.csv` に以下を追加する。

```text
entry_score
score_decision
score_reasons
```

### メリット

- ルールより柔軟。
- MLより説明しやすい。
- 閾値を過去データで検証しやすい。

### デメリット

- 点数配分に主観が入る。
- 過去成績を見ながら調整しすぎると過最適化になる。

## 6. 方式C: 統計テーブル方式

特徴量をビン分けし、過去の類似局面の成績を見る。

### ビン分け例

```text
RSI帯:
  - 30未満
  - 30〜40
  - 40〜50
  - 50〜60
  - 60以上

ATR帯:
  - 0.0〜0.3
  - 0.3〜0.7
  - 0.7〜0.9
  - 0.9以上

週足:
  - 上昇
  - 下降

出来高:
  - 1.0未満
  - 1.0〜1.5
  - 1.5以上
```

### 集計例

```text
RSI 30〜40 × ATR 0.7〜0.9 × 週足上昇
  件数: 24
  勝率: 58%
  平均リターン: +1.2%
  PF: 1.35
  最大損失: -4.5%
```

### 判断ルール

```text
entry_candidate = 1 if (
    sample_count >= 20
    and win_rate >= 0.55
    and avg_return > 0
    and profit_factor >= 1.2
) else 0
```

### メリット

- 過去の類似局面で説明できる。
- ブラックボックスになりにくい。
- 銘柄別、全銘柄横断の両方で使える。

### デメリット

- ビンを細かくしすぎるとサンプル不足になる。
- 条件の組み合わせが増える。
- 相場環境が変わると過去統計が効きにくくなる。

## 7. 方式D: MLを補助情報に下げる

MLを完全削除せず、最終判断からは外す方式。

```text
最終判断:
  ルール + 統計テーブル + スコアリングで決定

ML出力:
  レポート上の参考情報として表示のみ
```

表示例:

```text
参考ML確率: 51%
直近fold AUC: 0.50
評価: 予測力が弱いため判断には使用しない
```

この方式なら、将来データが増えたときにMLを再導入しやすい。

## 8. 評価指標

モデルを使わない場合でも、検証は必須である。  
見るべき指標は分類精度ではなく、売買成績に寄せる。

```text
件数
勝率
平均リターン
中央値リターン
累積リターン
最大損失
最大ドローダウン
プロフィットファクター
平均保有日数
利確到達率
損切り到達率
```

最低基準:

```text
エントリー件数: 少なすぎない
勝率: 55%以上
平均リターン: プラス
PF: 1.2以上
最大損失: 許容範囲内
複数銘柄・複数期間で大崩れしない
```

## 9. 実装方針

### 新規モジュール案

```text
src/step4_edge_analyzer.py
```

責務:

- 条件別の過去成績を集計する。
- スコア閾値別の成績を集計する。
- シグナル別の期待値を集計する。

主要関数:

```python
def add_entry_score(df: pd.DataFrame) -> pd.DataFrame:
    """ルールベースのentry_scoreを追加する。"""

def evaluate_score_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """entry_scoreの閾値別成績を集計する。"""

def build_condition_stats(df: pd.DataFrame) -> pd.DataFrame:
    """RSI帯、ATR帯、週足などの条件別成績を集計する。"""

def evaluate_trade_performance(df: pd.DataFrame, entry_col: str) -> dict:
    """指定したentry_colに基づく売買成績を返す。"""
```

### 既存ファイルの扱い

`src/step4_model.py`:

- すぐ削除しない。
- `--use-ml` 指定時だけ使う。
- デフォルトは統計分析モードにする。

`src/step5_assist_signal.py`:

- `entry_score` を追加する。
- `signal_type` を追加する。

`src/step6_filter.py`:

- DDモデル依存を弱める。
- `entry_score` と統計テーブルを使ったフィルタに置き換える。

`src/step7_entry_signal.py`:

- レポートに `entry_score`、統計上の勝率、平均リターン、サンプル数を表示する。

`src/step8_scanner.py`:

- スキャン結果を `entry_score` 順に並べる。
- `ML不使用モード` を既定にする。

## 10. 最初に試すルール

まずは以下の簡易スコアを実装し、7701と6981で比較する。

```text
加点:
  +2 weekly_trend == 1
  +1 35 <= rsi14 <= 50
  +1 volume_ratio >= 1.0
  +1 dev_ma25_zscore > dev_ma25_zscore.shift(5)
  +1 atr_dev_ma25 < 0.5

減点:
  -2 atr_percentile >= 0.90
  -2 is_overbought_heat == 1
  -1 weekly_trend == -1
  -1 rsi14 >= 65

判定:
  entry_score >= 3: エントリー候補
  entry_score 1〜2: 監視
  entry_score <= 0: 見送り
```

検証:

```text
entry_score >= 2
entry_score >= 3
entry_score >= 4
```

それぞれについて、勝率、平均リターン、PF、最大損失、件数を見る。

## 11. レポート例

```text
【最終判断】 監視

entry_score: 2

加点:
  +1 RSI 35〜50
  +1 ATR正規化乖離が過熱していない

減点:
  -1 週足下降

過去類似局面:
  件数: 31
  勝率: 51.6%
  平均リターン: +0.4%
  PF: 1.08

判断:
  期待値はわずかにプラスだが、基準未達のためエントリーではなく監視。
```

## 12. 推奨結論

当面のおすすめは、以下の順番である。

1. MLを一旦デフォルト無効にする。
2. Step4を `edge_analyzer` に置き換える。
3. Step5に `entry_score` と `signal_type` を追加する。
4. Step6はDDモデルではなく、スコア閾値と統計テーブルでフィルタする。
5. MLは `--use-ml` 指定時のみ参考情報として出す。

この方式なら、判断の透明性を保ちながら、過去データ上で有意性がある条件だけを残していける。
