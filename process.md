# 各ステップの処理・学習内容 実装ガイド

各ステップで**何を計算しているか**・**どういうロジックで動いているか**を、
ソースコードに基づいて正確にまとめたドキュメントです。
パラメータを変えたい時や、処理の意図を確認したい時の参照用としてご利用ください。

---

## ステップ1: データ収集とマルチタイムフレームの統合
**ファイル**: `src/step1_feature_engineering.py`

### 処理の流れ

```
Excel読み込み（日足 Sheet1 + 週足 Sheet2）
  ↓
日足の指標計算（ATR・ボリンジャー・RSI・移動平均・モメンタム・ヒゲ）
  ↓
週足の指標計算（移動平均・傾き・トレンド方向）
  ↓
週足を日足に結合（merge_asof）
  ↓
欠損値除去（指標計算のウォームアップ期間）
```

### 週足データの取得（自動フォールバック）

```python
try:
    weekly = load_weekly_data(filepath, sheet_name="Sheet2")  # Sheet2を優先
except (ValueError, KeyError):
    weekly = make_weekly_data(daily_df)  # 無ければ日足をリサンプル
```

Sheet2が存在しない場合は`resample("W-FRI")`で日足から自動生成します。
集計ルールは「始値=週初の始値、高値=週中最大、安値=週中最小、終値=週末の終値、出来高=週合計」です。

### 各指標の計算式

#### ATR（Average True Range）
```
True Range = max(高値 - 安値, |高値 - 前日終値|, |安値 - 前日終値|)
ATR14 = True RangeのEMA（span=14, Wilder方式）
```
前日終値とのギャップ（窓開け）も考慮した値動き幅の14日指数移動平均。

#### ボリンジャーバンド %B
```
中心線 = 終値の20日単純移動平均
上バンド = 中心線 + 2σ（20日標準偏差）
下バンド = 中心線 - 2σ
%B = (終値 - 下バンド) / (上バンド - 下バンド)
```

#### RSI（14日）
```
上昇分 = diff().clip(lower=0)
下落分 = (-diff()).clip(lower=0)
RS = 上昇分のEMA(14) / 下落分のEMA(14)
RSI = 100 - 100 / (1 + RS)
```

#### 移動平均乖離率
```
dev_ma25 = (終値 - MA25) / MA25
dev_ma75 = (終値 - MA75) / MA75
```

#### モメンタム加速度
```
mom5  = 終値.pct_change(5)   # 5営業日リターン
mom20 = 終値.pct_change(20)  # 20営業日リターン
mom_accel = mom5 - mom20     # 短期と長期の差（加速度）
```
プラスなら「短期の勢いが長期を上回っている（加速中）」、マイナスなら「勢いが衰えている」。

#### ローソク足のヒゲ比率
```
上ヒゲ = 高値 - max(始値, 終値)
下ヒゲ = min(始値, 終値) - 安値
upper_wick_ratio = 上ヒゲ / (高値 - 安値)
lower_wick_ratio = 下ヒゲ / (高値 - 安値)
```

#### 週足のトレンド判定
```
weekly_ma5_slope  = (今週MA5 - 前週MA5) / 前週MA5   # 傾き
weekly_trend = +1 if weekly_ma5 > weekly_ma13 else -1
```

### 週足を日足に結合する方法

`pandas.merge_asof(direction="backward")`を使用。
日足の各行に対し「その日以前で最も直近の週足の値」を付与します。
これにより未来の週足データが当日の特徴量に混入しません（**未来参照の防止**）。

### 欠損値除去のタイミングと理由

MA75の計算には75日分のデータが必要なため、最初の74行は欠損値になります。
`dropna()`でこのウォームアップ期間を除去してから次のステップに渡します。

---

## ステップ2: 特徴量エンジニアリング（ドメイン知識の注入）
**ファイル**: `src/step2_domain_features.py`

### 処理の流れ

```
ステップ1出力を受け取る
  ↓
MA乖離率のZスコア化（add_relative_indicators）
  ↓
RSI強気ダイバージェンス検出（detect_rsi_bullish_divergence）
  ↓
ATR正規化による過熱感数値化（add_overheat_features）
  ↓
欠損値除去（Zスコアのウォームアップ期間）
```

### MA乖離率のZスコア化

```python
roll_mean = dev_ma25.rolling(60).mean()
roll_std  = dev_ma25.rolling(60).std()
dev_ma25_zscore = (dev_ma25 - roll_mean) / roll_std
```

「過去60日の自分自身の乖離率の分布の中で今日はどの位置か」を表します。
`±2`を超えると統計的な異常値（外れ値）と判定し、`oversold_flag` / `overbought_flag`を立てます。

生の乖離率（%）ではなくZスコア化する理由は、銘柄や時期によって乖離率の「普通の範囲」が変わるため、
絶対値では比較できないからです。Zスコアにすることで「この銘柄にとっての異常値」として評価できます。

### RSI強気ダイバージェンスの検出アルゴリズム

```
過去20日間の最安値のインデックスを特定する
  ↓
「今日の終値 ≤ その安値」（価格は下値更新）
かつ
「今日のRSI > その日のRSI」（RSIは切り上がり）
  ↓
両方成立 → rsi_bullish_divergence = 1
rsi_divergence_strength = 今日のRSI - その日のRSI（切り上がり幅）
```

価格が下値を更新しているのにRSIが切り上がっている状態は、
「売り圧力が弱まっている＝下落の勢いが衰えつつある」サインとして
テクニカル分析では広く使われています。

### ATR正規化乖離と過熱感フラグ

```python
atr_dev_ma25 = (終値 - MA25) / ATR14
is_overbought_heat = (atr_dev_ma25 >= 2.0)  # ATR2個分以上乖離
is_oversold_heat   = (atr_dev_ma25 <= -2.0)
```

単純な%乖離率と違い、「ボラティリティに対してどれくらい飛び出しているか」で測ります。
値動きが荒い時期には同じ%乖離でも異常度が低く、静かな時期には異常度が高くなります。

#### 急騰の一時性スコア
```python
pump_temporary_score = atr_dev_ma25.clip(lower=0) * mom_accel.clip(lower=0)
is_likely_temporary_pump = (is_overbought_heat == 1) & (mom_accel > 0)
```
「ATR基準で上方に飛び出している」かつ「モメンタムが加速中」という組み合わせを、
「急騰しているが続きにくい可能性が高い状態」として数値化します。

---

## ステップ3: ターゲット（ラベル）の設計
**ファイル**: `src/step3_labeling.py`

### 処理の流れ

```
ステップ2出力を受け取る
  ↓
バリアモードの決定（ATR or 固定%）
  ↓
全行に対してトリプルバリア判定をループ処理
  ↓
フォワードドローダウンの計算
  ↓
末尾の未来データ不足行（NaN）を除去
```

### トリプルバリア法

各日を「仮想エントリー日」とみなし、その後の値動きを追って3本のバリアのどれに
最初に触れるかを記録します。

#### ATRモード（デフォルト）
```
上バリア（利確） = entry_price + tp_atr_mult × ATR14
下バリア（損切り） = entry_price - sl_atr_mult × ATR14
垂直バリア（期間切れ） = 10営業日後
```

#### 固定%モード（`tp_pct`・`sl_pct`を指定した場合）
```
上バリア（利確） = entry_price × (1 + tp_pct)   例: +10%
下バリア（損切り） = entry_price × (1 - sl_pct)  例: -5%
垂直バリア（期間切れ） = 45営業日後（デフォルト）
```

#### 同日に両バリアに触れた場合の処理
```python
if hit_tp and hit_sl:
    label = 0  # 損切り優先（安全側）
    barrier = "stop_loss"
```

#### 期間切れ（time_out）の場合のラベル付け
```python
final_price = closes[end_idx]
label = 1 if final_price > entry_price else 0
```
保有期間が終わった時点の損益の符号でラベルを付けます。

### フォワードドローダウン

```python
future_lows = lows[i+1 : i+holding_period+1]
dd = (min(future_lows) - entry_price) / entry_price  # 負の値
avoid_entry_flag = 1 if dd <= -drawdown_threshold else 0
```

トリプルバリアとは**独立した**補助ラベル。「利確に到達したかどうか」ではなく
「保有中に一定割合（デフォルト3%）以上の含み損を抱えるかどうか」のみを判定します。

### 理論的なベースライン勝率（参考）

バリアの幅が非対称なため、予測力ゼロのランダムウォークでも勝率は50%にはなりません。

```
理論ベースライン ≒ 損切り幅 / (利確幅 + 損切り幅)

例: ATRモード (2.0倍 / 1.5倍) → 1.5 / (2.0 + 1.5) ≒ 42.9%
例: 固定%モード (+10% / -5%) → 5 / (10 + 5) ≒ 33.3%
```

ステップ4で評価する際は、この理論値を「ランダムと同等」の基準として使います。

---

## ステップ4: モデル構築とバリデーション
**ファイル**: `src/step4_model.py`

### 処理の流れ

```
ステップ3出力を受け取る
  ↓
学習に使う特徴量列の選定（_EXCLUDE_COLSで除外）
  ↓
Walk-Forward Validationで5フォールド検証
  ↓
全データで最終モデルを再学習
  ↓
結果・モデルを保存
```

### 使用するアルゴリズム

LightGBMがインストールされていれば自動的にそちらを使用し、
未インストールの場合はscikit-learnの`HistGradientBoostingClassifier`に自動フォールバックします。

```python
try:
    import lightgbm as lgb
    _BACKEND = "lightgbm"
except ImportError:
    _BACKEND = "sklearn(HistGradientBoosting)"
```

#### LightGBMのパラメータ
| パラメータ | 値 | 意味 |
|---|---|---|
| `n_estimators` | 300 | 決定木の本数 |
| `max_depth` | 5 | 1本の木の最大深さ |
| `learning_rate` | 0.05 | 学習率（小さいほど慎重に学習） |
| `num_leaves` | 31 | 1本の木の最大葉数 |
| `subsample` | 0.8 | 各ステップで使う行のサンプリング率 |
| `colsample_bytree` | 0.8 | 各ステップで使う列のサンプリング率 |

### 特徴量の選定（リーク防止）

以下の列は**学習から除外**します。

```python
_EXCLUDE_COLS = {
    # 価格スケール依存（銘柄・時期で絶対値が変わるため汎化しない）
    "open", "high", "low", "close", "volume",
    "ma25", "ma75", "weekly_ma5", "weekly_ma13", "atr14",
    # 予測対象そのもの
    "tb_label", "avoid_entry_flag",
    # ラベル生成に使った未来情報（混ぜると正解漏洩になる）
    "tb_barrier", "tb_days_to_touch", "tb_return", "forward_max_drawdown",
}
```

「未来情報のリーク」とは、答え合わせに使った情報を学習に混ぜてしまうことです。
例えば`tb_return`（実際に何%動いたか）を特徴量に含めると、
モデルは「リターンが高い日は利確する」という当然の規則を学習してしまい、
見かけ上の精度は高くなりますが実際には使えません。

### Walk-Forward Validation

```
全データを時系列順に5分割:

fold1: [学習: 期間1] → [検証: 期間2]
fold2: [学習: 期間1+2] → [検証: 期間3]
fold3: [学習: 期間1+2+3] → [検証: 期間4]
...
```

- データをシャッフルしない（時系列の順序を壊さない）
- 学習データは常に検証データより「過去」のみ
- `min_train_size=100`に満たないフォールドはスキップ

#### 評価指標

| 指標 | 計算式 | 意味 |
|---|---|---|
| accuracy | 正解数 / 全件数 | 全体的な正解率 |
| precision | TP / (TP + FP) | 「1と予測した」中で実際に1だった割合 |
| recall | TP / (TP + FN) | 実際の「1」をどれくらい拾えたか |
| f1 | 2 × precision × recall / (precision + recall) | precisionとrecallの調和平均 |
| roc_auc | — | 0.5=コイン投げ、1.0=完璧。理論ベースライン勝率より有用な基準 |

#### 特徴量重要度の算出方法

- LightGBMの場合: `model.feature_importances_`（分割の寄与度ベース）
- sklearnの場合: `permutation_importance`（特徴量をランダムに並び替えた時のスコア低下度合い）

全フォールドの平均値を最終的な重要度として記録します。

### 最終モデルの再学習

```python
final_model = _build_model()
final_model.fit(X, y)  # 全期間のデータで再学習
```

Walk-Forward Validationはあくまで「精度の評価」のためです。
実運用で使う最終モデルは、評価後に全期間のデータを使って再学習したものを保存します（`step4_model.pkl`）。

---

## ステップ5: アシストロジックの実装（3段階シグナル）
**ファイル**: `src/step5_assist_signal.py`

### 処理の流れ

```
ステップ3（またはステップ2）出力を受け取る
  ↓
RSI反転条件の判定（_detect_rsi_reversal）
  ↓
MA乖離縮小条件の判定（_detect_deviation_shrinking）
  ↓
過熱警戒条件の判定（_detect_overheat_warning）
  ↓
3条件を優先順位に従って統合 → assist_signal列を生成
  ↓
tb_label・tb_returnと突き合わせて検証（evaluate_signal_quality）
```

### 各条件の判定ロジック

#### RSI反転フラグ（`rsi_reversal_flag`）

条件AまたはBを満たせば`1`。

```python
# 条件A: 直近5日以内に売られすぎ圏（RSI≤35）を経験し、今日上昇中
recent_min = rsi.shift(1).rolling(5).min()
condition_a = (recent_min <= 35) & (rsi > rsi.shift(1))

# 条件B: ステップ2でRSI強気ダイバージェンスが成立
condition_b = rsi_bullish_divergence == 1

rsi_reversal_flag = condition_a | condition_b
```

`rsi.shift(1).rolling(5)`とすることで「当日を含まない過去5日」を参照し、
未来参照を防いでいます。

#### MA乖離縮小フラグ（`deviation_shrink_flag`）

```python
z = dev_ma25_zscore
still_below_ma = z < 0           # まだMAより下にいる
is_recovering  = z > z.shift(5)  # 5日前よりゼロに近づいている

deviation_shrink_flag = still_below_ma & is_recovering
```

「まだ下にいるが戻り始めている」状態を捉えます。
既にMAを上回っている（`z > 0`）局面は対象外とすることで、
高値掴みを避ける設計になっています。

#### 過熱警戒フラグ（`overheat_warning_flag`）

```python
overheat_warning_flag = is_overbought_heat == 1
# = atr_dev_ma25 >= 2.0（ATR2個分以上の上方乖離）
```

ステップ2で計算済みの値をそのまま参照するため、この処理自体は軽量です。

### 3段階シグナルの統合と優先順位

```python
signal = np.where(
    overheat_warning,               "警戒",   # 最優先
    np.where(
        rsi_reversal & deviation_shrink, "強気",
        "中立"                              # デフォルト
    )
)
```

**優先順位: 警戒 > 強気 > 中立**

過熱（警戒）が成立している時は、RSI反転と乖離縮小が両方成立していても
「警戒」を優先します。急騰中の逆張りエントリーを避ける設計です。

**強気**には2条件の**AND**を要求します（RSI反転だけ、または乖離縮小だけでは強気にならない）。
これにより誤検知を減らし、シグナルの精度を上げています。

### シグナルの検証

```python
df.groupby("assist_signal").agg(
    件数   = ("assist_signal", "count"),
    勝率   = ("tb_label", "mean"),
    平均リターン = ("tb_return", "mean"),
)
```

ステップ3のラベル（`tb_label`・`tb_return`）と突き合わせることで、
「強気シグナルが出た日の実際の勝率」を事後的に検証できます。

**理想的な結果の基準:**
- 強気の勝率 > 中立の勝率 > 警戒の勝率
- 強気の平均リターン > 中立の平均リターン

これが逆転している場合、シグナルのロジックや閾値がそのデータ・期間に合っていない
可能性を示唆しています。

---

## パラメータ一覧

各ステップで変更可能な主なパラメータをまとめます。

| ステップ | パラメータ | デフォルト値 | 変更方法 |
|---|---|---|---|
| 1 | 週足MAの期間 | 5週・13週 | `add_weekly_features()`内を直接編集 |
| 1 | RSIの期間 | 14日 | `add_daily_features()`内を直接編集 |
| 1 | ボリンジャーの期間 | 20日 | `add_daily_features()`内を直接編集 |
| 2 | ZスコアのウィンドウN | 60日 | `build_step2_dataset(zscore_window=60)` |
| 2 | ダイバージェンスのルックバック | 20日 | `build_step2_dataset(divergence_lookback=20)` |
| 2 | 過熱とみなすATR倍率の閾値 | 2.0倍 | `build_step2_dataset(overheat_threshold=2.0)` |
| 3 | バリアモード | atr | `main.py --barrier-mode fixed_pct` |
| 3 | ATR倍率（利確） | 2.0倍 | `main.py --tp-atr-mult 2.0` |
| 3 | ATR倍率（損切り） | 1.5倍 | `main.py --sl-atr-mult 1.5` |
| 3 | 固定%（利確） | — | `main.py --tp-pct 0.10` |
| 3 | 固定%（損切り） | — | `main.py --sl-pct 0.05` |
| 3 | 最大保有期間 | ATR=10日 / 固定%=45日 | `main.py --holding-period 20` |
| 3 | ドローダウン閾値 | 3% | `main.py --drawdown-threshold 0.03` |
| 4 | 決定木の本数 | 300本 | `_build_model()`内を直接編集 |
| 4 | フォールド数 | 5 | `build_step4_results(n_splits=5)` |
| 5 | RSI反転のルックバック | 5日 | `build_step5_dataset(rsi_lookback=5)` |
| 5 | 乖離縮小のルックバック | 5日 | `build_step5_dataset(dev_lookback=5)` |
