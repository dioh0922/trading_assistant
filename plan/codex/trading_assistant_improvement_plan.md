# 株式分析プロジェクト 改善検討メモ

作成日: 2026-06-30  
対象: `trading_assistant` Step1〜9、および `resource/7701/atr` の出力結果

## 1. 結論

現状のパイプラインは、特徴量作成、ラベル作成、モデル学習、ルールベース判断、レポート出力まで一通り実装されている。しかし、実際の売買判断として有意性を感じにくい主因は、次の4点にある。

1. モデル評価が売買成績ではなく分類指標中心になっている。
2. Step3のラベルが「利確到達」を重視しており、途中ドローダウンや期待値と十分に統合されていない。
3. Step5のルールベースシグナルが硬く、勝ち局面を中立として見送るケースが多い。
4. Step6のドローダウンフィルタが、既定設定では実質的に効きにくい。

まずはLLMや外部ニュースを追加する前に、定量パイプライン自体の評価軸と意思決定ルールを改善するべきである。

## 2. 7701データから見えた現状

### 2.1 Step4 モデル性能

`step4_fold_metrics.csv` では、Walk-ForwardのAUCが以下の通りだった。

| fold | test期間 | accuracy | precision | recall | roc_auc |
|---:|---|---:|---:|---:|---:|
| 2 | 2025-02-25〜2025-06-20 | 0.380 | 0.349 | 0.733 | 0.585 |
| 3 | 2025-06-23〜2025-10-16 | 0.405 | 0.389 | 0.163 | 0.464 |
| 4 | 2025-10-17〜2026-02-13 | 0.392 | 0.212 | 0.241 | 0.406 |
| 5 | 2026-02-16〜2026-06-12 | 0.532 | 0.500 | 0.568 | 0.503 |

直近foldのAUCはほぼ0.5で、方向判別力はランダムに近い。fold 3〜4では0.5を下回っており、学習されたパターンが将来に安定して持ち越されていない可能性が高い。

### 2.2 Step5 シグナル品質

`step5_dataset.csv` のシグナル別成績は以下。

| シグナル | 件数 | 勝率 | 平均リターン |
|---|---:|---:|---:|
| 強気 | 64 | 50.0% | +0.33% |
| 警戒 | 59 | 25.4% | -1.50% |
| 中立 | 354 | 46.9% | +0.32% |

警戒シグナルは機能している可能性がある。一方で、強気シグナルは中立と比べて明確な優位性が出ていない。強気条件が「勝てる局面」を十分に抽出できていない。

### 2.3 Step6 最終判断

`step6_dataset.csv` のOOS対象316件では、最終エントリーは30件だけだった。

| 指標 | 値 |
|---|---:|
| OOS件数 | 316 |
| 強気シグナル件数 | 38 |
| 最終エントリー件数 | 30 |
| 最終エントリー勝率 | 50.0% |
| 最終エントリー平均リターン | +0.08% |

Step6を通したあとも、勝率と平均リターンに明確な改善が出ていない。現状では「取引数を減らしただけで、期待値を改善できていない」状態に近い。

### 2.4 最新行の例

7701の最終行は以下の状態だった。

- 日付: 2026-06-12
- 終値: 3710
- RSI14: 37.3
- ATRパーセンタイル: 0.98
- 週足トレンド: 下降
- assist_signal: 中立
- drawdown_prob: 18.2%
- final_decision: 見送り
- 後付けラベル: `tb_label=1`, `take_profit`, 10日で利確到達

これは、実際には利確到達した局面をルールが中立として見送っている例である。もちろん未来情報なので当日は分からないが、こうした「拾えなかった勝ち局面」を集計し、どの条件を緩めるべきか分析する必要がある。

## 3. 改善優先順位

## 優先度A: 評価指標を売買成績ベースに変える

対象ファイル:

- `src/step4_model.py`
- `src/step6_filter.py`
- `src/step8_scanner.py`

現状はaccuracy、precision、recall、f1、AUCが中心である。しかしトレードでは、分類精度よりも期待値が重要である。

追加すべき指標:

- エントリー件数
- 勝率
- 平均リターン
- 中央リターン
- 累積リターン
- 最大損失
- 最大ドローダウン
- プロフィットファクター
- Precision@上位N%
- 予測確率上位10%、20%、30%だけに絞った成績

特に `step4_oos_predictions.csv` に `tb_return` を結合し、`y_proba` の上位層が本当に良い成績かを確認する。AUCが低くても、上位10%だけ期待値が高ければ実運用余地がある。逆に上位層でも期待値が出なければ、モデルは売買判断に使わない方がよい。

## 優先度A: ラベルを「良いトレード」基準に作り直す

対象ファイル:

- `src/step3_labeling.py`

現状の `tb_label` は、基本的に利確到達を1としている。一方、`forward_max_drawdown` は別ラベルとして扱われている。このため、「最終的には利確したが、途中で大きく含み損になった」ケースも買い成功として扱われやすい。

改善案:

```text
good_trade_label = 1
  条件:
    - take_profitに到達
    - 途中最大DDが許容範囲内
    - 到達日数が長すぎない

bad_trade_label = 1
  条件:
    - stop_loss到達
    - または期間中DDが閾値超え
    - またはtime_outでリターンが小さい/負
```

最初は以下のような単一ラベルに統合する。

```text
trade_quality_label = 1 if (
    tb_label == 1
    and forward_max_drawdown > -0.03
    and tb_days_to_touch <= holding_period
) else 0
```

これにより、モデルが「ただ利確する可能性」ではなく、「実際に持ちやすく期待値がある局面」を学習できる。

## 優先度A: Step6の閾値設計を実運用向けに直す

対象ファイル:

- `src/step6_filter.py`
- `src/step8_scanner.py`

`step8_scanner.py` の `dd_prob_limit` 既定値が `1.0` になっているため、DDフィルタが実質的に効かない。`run_pipeline_for_code()` 内には `ml_enabled = dd_prob_limit < 1.0` という変数もあるが、実際の制御には使われていない。

改善案:

1. `dd_prob_limit` の既定値を `0.60` などに戻す。
2. 閾値を固定せず、OOSでグリッドサーチする。
3. `dd_threshold` と `dd_prob_limit` の名前が紛らわしいため整理する。

候補:

- `drawdown_label_threshold`: 実際のDDラベル作成用。例: 3%
- `drawdown_prob_limit`: モデル予測確率の許容上限。例: 60%

## 優先度B: Step5の強気条件を再設計する

対象ファイル:

- `src/step5_assist_signal.py`

現状の強気条件:

```text
(RSI反転 and MA乖離縮小) or
(週足上昇 and RSI>=50 and volume_ratio>=1.2)
```

この条件では、RSIが40前後から反転しそうな局面や、下降トレンド内の短期リバウンドを拾いにくい。7701の最終行も、RSI37.3、DD予測18.2%、後付け利確到達だったが、中立扱いだった。

改善案:

1. 強気を1種類にせず、複数タイプに分ける。

```text
強気_順張り:
  weekly_trend == 1
  RSI >= 50
  volume_ratio >= 1.2

強気_反転:
  RSIが35〜45で上向き
  dev_ma25_zscoreが改善
  drawdown_probが低い

強気_低リスク押し目:
  RSI < 45
  atr_dev_ma25が過熱していない
  forward DD予測が低い
```

2. `assist_signal` だけでなく `signal_type` を追加する。

```text
assist_signal: 強気 / 警戒 / 中立
signal_type: trend_follow / reversal / pullback / overheat / none
```

3. シグナル別に成績を出す。

これにより、「強気全体では弱いが、反転型だけは期待値がある」といった分析が可能になる。

## 優先度B: モデル予測をStep6で活用する

対象ファイル:

- `src/step4_model.py`
- `src/step6_filter.py`

現状Step6は、強気シグナルに対してDD予測やATRフィルタを適用する構造である。しかしStep4の `y_proba`、つまり利確側の予測確率が最終判断に直接使われていない。

改善案:

```text
entry_score = y_proba * (1 - drawdown_prob)
```

または、

```text
expected_value =
    y_proba * expected_gain
    - drawdown_prob * expected_loss
```

最終判断は以下のようにする。

```text
final_decision = 1 if (
    assist_signal != "警戒"
    and entry_score >= score_threshold
    and atr_percentile < atr_limit
) else 0
```

これにより、Step4の買い成功確率とStep6のDDリスクが一つの判断軸に統合される。

## 優先度B: 特徴量の追加・整理

対象ファイル:

- `src/step1_feature_engineering.py`
- `src/step2_domain_features.py`

現在の重要特徴量では、`bb_width_zscore`、`dev_ma75_zscore`、`weekly_ma5_slope`、`mom_accel`、`atr_percentile` が上位に出ている。方向性そのものより、「ボラティリティ拡大」「中期乖離」「週足傾き」が効いている可能性がある。

追加候補:

- ギャップ率: `(open - prev_close) / prev_close`
- 直近高値からの下落率
- 直近安値からの反発率
- 過去20日高値/安値レンジ内の現在位置
- 陽線/陰線の連続日数
- 出来高増加を伴う陽線フラグ
- ATRの増減率
- 日経平均やTOPIXとの相対リターン
- セクター平均との差分

ただし、特徴量を増やす前に、現在の特徴量で期待値評価ができるようにすることを優先する。

## 優先度C: Step9の目標到達推定をエントリー判断と分離する

対象ファイル:

- `src/step9_target_estimator.py`

Step9は「現在値から目標価格に到達する確率」を推定しているが、これは「今日エントリーすべきか」とは別問題である。到達確率が高くても、途中DDが大きいならエントリーには向かない。

改善案:

Step9レポートに以下を追加する。

- 到達確率
- 到達前最大DDの分布
- 目標到達までの期待DD
- 到達確率 / DDリスク比
- 現在のStep7判断との整合性

レポート文言も「到達可能性」ではなく「今この価格で狙う妥当性」を分けて書く。

## 優先度C: LLMは最後に説明・監査役として使う

対象ファイル:

- `src/step10_prompt_builder.py`
- `src/step11_llm_client.py`

LLMに売買判断を直接任せるのは早い。まずは定量側で、以下の構造化データを作る。

```json
{
  "quant_decision": "見送り",
  "entry_score": 0.42,
  "win_probability": 0.51,
  "drawdown_probability": 0.18,
  "expected_value": 0.003,
  "signal_type": "pullback",
  "main_risks": ["ATR high", "weekly downtrend"],
  "missing_data": []
}
```

LLMの役割は次に限定する。

- 判断根拠の要約
- 矛盾検出
- 人間が確認すべき項目の提示
- レポートの読みやすさ改善

LLM判断そのものをバックテストで検証できるまでは、最終売買判断には使わない。

## 4. 実装ロードマップ

### Phase 1: 評価基盤の修正

1. `step4_oos_predictions.csv` に `tb_return`、`tb_barrier`、`forward_max_drawdown` を結合する。
2. `step4_model.py` に売買評価関数を追加する。
3. `y_proba` 上位10%、20%、30%の成績を出す。
4. `step6_filter.py` の最終エントリー成績も同じ指標で出す。

成果物:

- `step4_trade_metrics.csv`
- `step6_trade_metrics.csv`

### Phase 2: ラベル改善

1. `step3_labeling.py` に `trade_quality_label` を追加する。
2. Step4のtargetを `tb_label` と `trade_quality_label` で比較する。
3. OOS期待値が高い方を採用する。

成果物:

- `step3_dataset.csv` に `trade_quality_label` 追加
- `step4_fold_metrics.csv` にtarget名を明記

### Phase 3: シグナル再設計

1. `step5_assist_signal.py` に `signal_type` を追加する。
2. 強気を順張り、反転、押し目に分ける。
3. signal_type別の勝率、平均リターン、DDを出す。

成果物:

- `step5_signal_quality.csv`

### Phase 4: 統合スコア導入

1. Step4の `y_proba` とStep6の `drawdown_prob` を統合する。
2. `entry_score` または `expected_value` を作る。
3. 閾値をOOSでグリッドサーチする。

成果物:

- `step6_threshold_search.csv`
- `step6_dataset.csv` に `entry_score` 追加

### Phase 5: レポート改善

1. Step7で `entry_score`、`signal_type`、期待値を表示する。
2. 見送り理由に「どの条件を満たせば候補になるか」を追加する。
3. Step9は到達確率とエントリー妥当性を分けて説明する。

## 5. 最初に直すべき具体箇所

### `src/step8_scanner.py`

`dd_prob_limit=1.0` の既定値を見直す。DDフィルタを使うなら `0.60` 前後から検証する。

### `src/step6_filter.py`

`evaluate_final_performance()` はprintだけでなく、DataFrameを返してCSV保存できるようにする。

### `src/step4_model.py`

OOS予測に元データの `tb_return` を結合し、分類評価と売買評価を同時に保存する。

### `src/step3_labeling.py`

`tb_label` とは別に、DD許容込みの `trade_quality_label` を追加する。

### `src/step5_assist_signal.py`

`assist_signal` に加えて、シグナルの種類を表す `signal_type` を追加する。

## 6. 判断基準

改善後、最低限以下を満たさない場合は、売買判断としては弱い。

- OOSで最終エントリー勝率が55%以上
- 最終エントリー平均リターンが明確にプラス
- プロフィットファクターが1.2以上
- 取引件数が少なすぎない
- 直近foldだけでなく複数foldで成績が安定

これらを満たすまでは、「エントリー推奨システム」ではなく「分析補助レポート」として扱うのが妥当である。

## 7. 推奨する次アクション

最初の実装タスクは以下にする。

1. `step4_model.py` に売買評価指標を追加する。
2. `step3_labeling.py` に `trade_quality_label` を追加する。
3. 7701と6981で `tb_label` と `trade_quality_label` のOOS成績を比較する。
4. 期待値が改善するラベルを採用してから、Step5/6のルールを調整する。

この順番なら、感覚的なチューニングではなく、どの変更が実際に期待値を改善したかを確認しながら進められる。
