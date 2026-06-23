# モデル改善プラン

本ドキュメントは、Gemini・ChatGPT・Copilotの3つの改善提案と、  
実データ（`step4_fold_metrics.csv` / `step4_feature_importance.csv` / `step4_oos_predictions.csv`）の  
数値分析を照合し、**実際に起きていること**と**対処の優先順位**を整理したものです。

---

## 現状スコアの確認

| フォールド | 学習件数 | 検証件数 | accuracy | precision | recall | f1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Fold 1 | 110 | 108 | 0.389 | 0.385 | 0.167 | 0.233 | **0.325** |
| Fold 2 | 218 | 108 | 0.324 | 0.286 | 0.765 | 0.416 | 0.409 |
| Fold 3 | 326 | 108 | 0.546 | 0.462 | 0.255 | 0.329 | 0.426 |
| Fold 4 | 434 | 108 | 0.398 | 0.600 | 0.365 | 0.454 | 0.410 |
| Fold 5 | 542 | 108 | 0.463 | 0.667 | 0.294 | 0.408 | 0.436 |
| **平均** | | | **0.424** | **0.480** | **0.369** | **0.368** | **0.401** |

全フォールドでROC-AUCが0.5を下回っており、**コイン投げ以下**の状態です。

---

## 最重要発見：予測が逆方向に学習している

3つのAIすべてがROC-AUCの低さを指摘しましたが、ChatGPTが提起した  
「**ラベル反転**の可能性」を実データで検証した結果、以下が確認されました。

```
現状のAUC（oos_predictions.csv）:   0.391
予測確率を反転した場合のAUC:         0.609
```

**モデルは「利確到達（label=1）」と「損切り（label=0）」を逆に学習しています。**

これはコードのバグではなく、**データの構造的な問題**です。  
各フォールドの検証期間におけるラベル比率を確認すると、原因が明確になります：

| フォールド | 学習期間 | 学習のlabel=1率 | 検証のlabel=1率 |
|---|---|---|---|
| Fold 1 | 〜2024-03 | 41.8% | 55.6% |
| Fold 2 | 〜2024-08 | 48.6% | 31.5% |
| Fold 3 | 〜2025-01 | 42.9% | 43.5% |
| Fold 4 | 〜2025-07 | 43.1% | **68.5%** |
| Fold 5 | 〜2025-12 | 48.2% | **63.0%** |

Fold4・5の検証期間で「利確到達」の割合が急増しています。  
これは急騰相場への移行（**レジームチェンジ**）によるもので、  
過去の「落ち着いた相場」で学習したモデルが急騰相場に追いつけていない状態です。  
Gemini・Copilotが指摘した「**レジームへの適応不足**」はこれが原因です。

---

## 改善プラン（優先度順）

---

### 優先度A：ラベルとバリア設計の見直し
**対象ファイル：`src/step3_labeling.py`、`main.py`**

3つのAIすべてが独立して指摘した最重要項目です。

#### A-1. ラベル比率の確認と再設計

現状（ATRモード, 保有10日）のラベル分布は `{0: 321, 1: 329}` とほぼ均等ですが、  
**フォールドをまたぐと比率が31%〜68%まで振れています**。  
急騰相場では「ATR2.0倍の利確ライン」が相対的に近くなり、利確到達が増えます。

対処として以下のパターンを比較検証することを推奨します：

| パターン | モード | 利確 | 損切り | 保有期間 | 理論ベースライン勝率 |
|---|---|---|---|---|---|
| 現状 | ATR | 2.0倍 | 1.5倍 | 10日 | 42.9% |
| 既存固定%設定 | 固定% | +10% | -5% | 45日 | 33.3% |
| 均等リスクリワード | ATR | 1.5倍 | 1.5倍 | 20日 | 50.0% |
| 小さいバリア | 固定% | +3% | -3% | 10日 | 50.0% |

**実行方法：**
```bash
# パターン別に実行して fold_metrics.csv を比較する
python main.py --barrier-mode atr --tp-atr-mult 1.5 --sl-atr-mult 1.5 --holding-period 20
python main.py --barrier-mode fixed_pct --tp-pct 0.03 --sl-pct 0.03 --holding-period 10
```

#### A-2. フォールド内のラベル比率をモニタリングする

現在の `step4_fold_metrics.csv` にはラベル比率の列がありません。  
各フォールドの `label_1_rate_train` / `label_1_rate_test` を出力するよう  
`src/step4_model.py` の `fold_records` 記録部分に追加することを推奨します。

---

### 優先度B：特徴量の多様化
**対象ファイル：`src/step1_feature_engineering.py`、`src/step2_domain_features.py`**

Gemini・ChatGPT・Copilotの全員が「**現在の特徴量は価格情報に偏っている**」と指摘しています。  
特徴量重要度Top10はほぼ全てが価格・移動平均から派生したものであり、  
「価格がどこにあるか」を異なる表現で重複して持っている状態です。

```
現状のTop10（全て価格系）
  weekly_ma5_slope, upper_wick_body_ratio, rsi14,
  lower_wick_ratio, candle_dir, mom5, upper_wick_ratio,
  lower_wick_body_ratio, dev_ma25, weekly_trend
```

#### B-1. 出来高系特徴量の追加

出来高は価格と独立した情報源であり、現状の特徴量と**相関が低い**点が強みです。

```python
# 推奨追加特徴量（step1_feature_engineering.py に追加）

# 出来高Zスコア（急増の検知）
vol_mean = df["volume"].rolling(20).mean()
vol_std  = df["volume"].rolling(20).std()
df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std

# 出来高比率（20日平均の何倍か）
df["volume_ratio"] = df["volume"] / vol_mean
```

#### B-2. レジーム識別特徴量の追加

急騰相場・レンジ相場・下落相場を識別するための特徴量を追加します。

```python
# ボリンジャーバンド幅（収縮=レンジ相場、拡大=トレンド相場）
bb_width = (bb_upper - bb_lower) / bb_mid
df["bb_width"] = bb_width
df["bb_width_zscore"] = (bb_width - bb_width.rolling(60).mean()) / bb_width.rolling(60).std()

# ATRの相対水準（過去250日の中で今日のATRが何パーセンタイルか）
df["atr_percentile"] = df["atr14"].rolling(250).rank(pct=True)
```

#### B-3. 市場全体との相対強度（Copilot・ChatGPT推奨）

個別銘柄の値動きが市場全体より強いか弱いかを表す指標です。  
日経平均・TOPIXのデータを`resource/`に別途用意する必要があります。

```python
# Relative Strength（市場全体より強い=+、弱い=-）
df["rs_vs_market"] = df["return_5d"] - df["market_return_5d"]
```

---

### 優先度C：モデル評価の強化
**対象ファイル：`src/step4_model.py`**

現状の評価は「全データに対するAUC」のみですが、  
**Copilotが指摘した「fold間の安定性」**という視点が抜けています。

#### C-1. fold間安定性の計測

```python
# fold_metricsに追加する評価
auc_std = fold_metrics["roc_auc"].std()
# 合格ライン（Copilot提案）: auc_std ≤ 0.03
# 現状: {fold_metrics["roc_auc"].std():.3f}  ← 現状値を記録
```

#### C-2. 確信度フィルタの導入（Gemini推奨）

`step4_oos_predictions.csv`の`y_proba`分布を確認すると、  
曖昧ゾーン（0.4〜0.6）の割合は**8.7%**と少なく、  
モデルは二極化した予測をしていることがわかります（25%tile=0.10、75%tile=0.83）。

この二極化は「確信しているが方向が逆」という状態なので、  
確信度フィルタよりも**先にラベル・特徴量の見直し（優先度A・B）**が有効です。  
A・Bを実施後にAUCが改善されてから、確信度フィルタを導入することを推奨します。

```python
# A・B実施後に導入する確信度フィルタの例
def apply_confidence_filter(y_proba, threshold=0.3):
    """
    予測確率が threshold 以下または (1-threshold) 以上の場合のみエントリー対象とする。
    曖昧な予測（0.3〜0.7）は見送る。
    """
    return (y_proba <= threshold) | (y_proba >= 1 - threshold)
```

#### C-3. 確率キャリブレーション（Copilot推奨）

モデルの出力確率が「実際の発生確率」と一致しているかを確認・補正します。  
A・B実施後のモデルに対して適用することを推奨します。

```python
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
calibrated_model.fit(X_calib, y_calib)
```

---

### 優先度D：ステップ5のシグナル設計の見直し
**対象ファイル：`src/step5_assist_signal.py`**

現状の強気・警戒・中立の勝率は以下の通りです：

| シグナル | 件数 | 勝率 | 平均リターン |
|---|---|---|---|
| 強気 | 29件 | 48.3% | 1.39% |
| 警戒 | 153件 | **59.0%** | 1.72% |
| 中立 | 477件 | 48.1% | 0.32% |

「警戒」の勝率が最も高いという逆転が起きています。  
これはFold4・5で確認されたのと同じ**急騰相場の影響**です。  
急騰中（＝警戒が出る局面）の方が結果的に利確に到達しやすかったためです。

**シグナルの見直し方針（ChatGPT推奨）：**

現状の「強気」条件は逆張り型（売られすぎからの反転）です。  
急騰相場に強い「順張り型」の条件を加えることで、  
相場のフェーズに関係なく機能するシグナルに改善できます。

```python
# 追加候補：順張り型の強気条件
# 週足上昇トレンド + RSI 50以上 + 出来高増加
trending_bullish = (
    (df["weekly_trend"] == 1) &       # 週足が上昇トレンド
    (df["rsi14"] >= 50) &             # RSIが中立以上
    (df["volume_ratio"] >= 1.2)       # 出来高が20日平均の1.2倍以上（B-1追加後）
)
```

---

## 実施ロードマップ

### フェーズ1（最優先・1〜2週間）

**ラベル設計の再検討**（優先度A）

1. `fold_metrics.csv`にラベル比率列を追加し、フォールドごとの偏りを可視化する
2. バリアパターンを4パターン（上記表）で比較実行し、フォールド間のラベル比率が安定するものを選ぶ
3. 固定%モード（+10% / -5%、保有45日）のROC-AUCと比較する

```bash
# 比較実行
python main.py --barrier-mode atr --tp-atr-mult 1.5 --sl-atr-mult 1.5 --holding-period 20
python main.py --barrier-mode fixed_pct --tp-pct 0.03 --sl-pct 0.03 --holding-period 10
python main.py --barrier-mode fixed_pct --tp-pct 0.05 --sl-pct 0.03 --holding-period 10
```

### フェーズ2（フェーズ1完了後・2〜3週間）

**特徴量の多様化**（優先度B）

1. 出来高系（`volume_zscore`、`volume_ratio`）を`step1_feature_engineering.py`に追加
2. レジーム識別系（`bb_width_zscore`、`atr_percentile`）を追加
3. 特徴量追加後にフェーズ1で選定したバリア設定で再学習し、AUCの変化を確認

### フェーズ3（フェーズ2完了後）

**評価・シグナルの強化**（優先度C・D）

1. fold間AUC標準偏差を評価指標に追加（目標：auc_std ≤ 0.03）
2. キャリブレーション（Isotonic Regression）を実装
3. ステップ5に順張り型の強気条件を追加し、シグナル別勝率の逆転が改善されるか確認

---

## 3つのAIの提案と本プランの対応表

| 提案内容 | Gemini | ChatGPT | Copilot | 本プランでの対応 |
|---|---|---|---|---|
| レジームチェンジへの対応 | ✅ | ✅ | ✅ | 優先度A（バリア設計）・優先度B-2（レジーム識別特徴量） |
| Walk-Forward Validationの徹底 | ✅ | ✅ | ✅ | **既に実装済み**（`TimeSeriesSplit`、シャッフルなし） |
| ラベル反転の確認 | — | ✅ | — | **実データで確認済み**。根本原因はレジームチェンジ |
| 特徴量の多様化（出来高・市場環境） | ✅ | ✅ | ✅ | 優先度B |
| 確信度フィルタの導入 | ✅ | — | ✅ | 優先度C-2（フェーズ3で実施） |
| 確率キャリブレーション | — | ✅ | ✅ | 優先度C-3（フェーズ3で実施） |
| fold間安定性の評価 | — | — | ✅ | 優先度C-1 |
| シグナル設計の見直し | — | ✅ | — | 優先度D |
| 多重共線性の排除（VIF/PCA） | — | — | ✅ | フェーズ2の特徴量整理時に合わせて実施 |
| LightGBMパラメータ調整 | — | ✅ | — | フェーズ3（A・B改善後に効果を確認） |

---

## 補足：Walk-Forward Validationについて

3つのAIすべてが「Walk-Forward Validationへの切り替え」を推奨していますが、  
**現在のシステムは既にWalk-Forward Validationを実装しています**  
（`sklearn.model_selection.TimeSeriesSplit`、データシャッフルなし、過去のみで学習）。  
追加対応は不要です。

ただし、Copilotが指摘した「**fold間AUCの標準偏差**」の計測は未実装のため、  
優先度C-1として追加することを推奨します。
