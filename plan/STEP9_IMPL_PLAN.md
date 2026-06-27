# ステップ9: 目標額到達推定 実装計画

## 概要

`--target-price` または `--target-pct` を指定すると、現在値から目標価格への到達確率と所要日数を2手法（経験的・モンテカルロGBM）で推定し、`target_report.txt` に出力する。

---

## 1. 新規作成: `src/step9_target_estimator.py`

### 1-1. 関数一覧

| 関数 | 責務 |
|---|---|
| `estimate_empirical(df, target_pct, horizons)` | 過去データの高値が目標に到達した実績を集計 → 到達確率 + 日数分布 |
| `estimate_monte_carlo(df, target_pct, horizons, n_paths, vol_window)` | GBMで10,000パスシミュレーション → 到達確率 + 日数分布 |
| `generate_regime_comment(signal, target_pct, atr_percentile)` | step7シグナル + ボラティリティ状態から補足コメント生成 |
| `build_target_report(current_price, target_price, ..., empirical, mc, signal, ...)` → `str` | 推定結果を整形したレポートテキスト生成 |
| `save_target_report(report_text, output_dir)` → `Path` | `resource/{code}/{mode}/target_report.txt` に保存 |

### 1-2. 定数設計

```python
ESTIMATION_HORIZONS = [10, 20, 30, 45, 60, 90]  # 推定期間（営業日）
N_MC_PATHS = 10_000                               # MCシミュレーションのパス数
```

### 1-3. コア推定ロジック

#### A. `estimate_empirical()`

```
全日付をループし、各日をエントリー起点として:
  1. entry_price = close[i]
  2. target = entry_price * (1 + target_pct)
  3. i+1 〜 i+max_horizon の範囲で high[j] >= target となる最小の j を探す
  4. 各 horizons に対して到達した数をカウント
  5. 到達した場合の日数を touch_days に記録

戻り値:
  reach_prob: {horizon: 確率, ...}
  days_distribution: {median, mean, p25, p75, p90, n_samples}
  n_total: 計算に使った起点の総数
```

- **使用する価格**: 高値（high） — 実際の指値成立を反映
- **未来参照なし**: ルックアヘッドしないループ処理

#### B. `estimate_monte_carlo()`

```python
# パラメータ推定
ret = close.pct_change().dropna()
sigma = ret.tail(60).std()        # 直近60日の実現ボラ
mu = ret.median()                 # 全期間の中央値（急騰バイアス回避）

# GBMシミュレーション（10,000パス × max_horizon日）
daily_shocks = N(mean=(mu - 0.5*sigma^2), scale=sigma, size=(n_paths, max_h))
cum_ret = exp(cumsum(daily_shocks, axis=1))

# 各パスの到達判定（終値ベース）
各 horizon について cum_ret >= (1 + target_pct) となるパスの割合を集計
```

- **μに中央値を使う理由**: 直近急騰の影響で平均μが過大になるのを防止
- **使用する価格**: 終値（close） — 日中高値の確率モデルは複雑なため保守的に終値

### 1-4. レポート生成 (`build_target_report`)

出力フォーマット:

```
====================================================================
  ステップ9: 目標額到達推定レポート
  銘柄コード   : 7701
  分析日       : 2026-06-19
  現在値       : 3,886 円
  目標価格     : 4,500 円  （現在値より +15.8%）
====================================================================

  ┌ 到達確率テーブル
  │  期間       10日    20日    30日    45日    60日    90日
  │ ────────────────────────────────────────────────────────
  │  経験的     8.7%   18.3%   27.1%   35.5%   41.2%   49.8%
  │  MC        11.2%   22.5%   34.0%   47.3%   57.8%   69.1%
  └

  ┌ 到達した場合の所要日数（目標 +15.8%）
  │              経験的          MC
  │  中央値        28日         24日
  │  平均          34日         30日
  │  25%〜75%   14〜52日     12〜44日
  │  サンプル      89件       3,312件
  └

  ■ 現在の相場状態
  ✅ 現在は強気シグナルが出ており、エントリーに適した状態です。
     （シグナル別にコメントが変化）

  ■ 推定の前提と限界
  ・経験的確率: 過去データに含まれる相場環境に依存します。
  ・モンテカルロ(GBM): 直近60日のボラティリティと中央値リターンから計算します。
====================================================================
```

### 1-5. 現在のシグナルによるコメント分岐 (`generate_regime_comment`)

| signal | コメント内容 |
|---|---|
| 強気 | 「強気シグナルが出ており、エントリーに適した状態。良好な状態での実現可能性が若干高い可能性」 |
| 警戒 | 「過熱シグナル（警戒）。高値圏でのエントリー、目標到達前に反落リスクが通常より高い」 |
| 中立 | 「中立シグナル。参考値としてそのまま利用可能」 |

追加で、`atr_percentile >= 0.90`（高ボラティリティ）や`target_pct >= 0.30`（大きな目標）の場合も警告を追記。

### 1-6. 入出力データの流れ

```
step1_dataset.csv（OHLCV） ──→ estimate_empirical()  ──┐
                                                         ├──→ build_target_report() ──→ target_report.txt
step7のシグナル（最終行）  ──→ estimate_monte_carlo() ──┘
                                 + generate_regime_comment()
```

---

## 2. 修正: `main.py`

### 2-1. 追加する引数（2つ）

```python
parser.add_argument("--target-price", type=float, default=None,
    help="[ステップ9] 目標価格を円で指定（例: --target-price 5000）。現在値より高い値を指定。")
parser.add_argument("--target-pct", type=float, default=None,
    help="[ステップ9] 目標上昇率（例: --target-pct 0.15 で +15%%）。--target-price と排他。")
```

### 2-2. `main()` の末尾に追加（step7の後）

```python
# --- ステップ9: 目標額到達推定 ---
if args.target_price is not None or args.target_pct is not None:
    from src.step9_target_estimator import (
        estimate_empirical, estimate_monte_carlo,
        generate_regime_comment, build_target_report, save_target_report,
    )

    print("\n" + "=" * 60)
    print("ステップ9: 目標額到達推定")
    print("=" * 60)

    # step1データ（OHLCV）を読み込む
    step1_csv = output_dir / "step1_dataset.csv"
    df_ohlcv = pd.read_csv(step1_csv, index_col=0, parse_dates=True)
    current_price = df_ohlcv["close"].iloc[-1]
    analysis_date = df_ohlcv.index[-1].date()

    # 目標価格の決定（--target-price と --target-pct は排他チェック）
    if args.target_price is not None and args.target_pct is not None:
        print("エラー: --target-price と --target-pct は同時に指定できません。")
        sys.exit(1)
    elif args.target_price is not None:
        target_price = args.target_price
        target_pct = (target_price - current_price) / current_price
    else:
        target_pct = args.target_pct
        target_price = current_price * (1 + target_pct)

    if target_pct <= 0:
        print(f"エラー: 目標価格 ({target_price:,.0f}円) は現在値より高くなければなりません。")
        sys.exit(1)

    # 現在のシグナルを取得（step7の結果を流用）
    try:
        source_df, _ = load_source_df(output_dir)
        signal = str(source_df.iloc[-1].get("assist_signal", "不明"))
        atr_p = source_df.iloc[-1].get("atr_percentile", None)
    except (FileNotFoundError, IndexError):
        signal = "不明"
        atr_p = None

    # 2手法で推定
    print("経験的確率を計算中...")
    empirical = estimate_empirical(df_ohlcv, target_pct)
    empirical["atr_percentile"] = atr_p

    print("モンテカルロシミュレーション中（10,000パス）...")
    mc = estimate_monte_carlo(df_ohlcv, target_pct)

    # レポート生成・出力
    report_text = build_target_report(
        current_price, target_price, target_pct, analysis_date,
        empirical, mc, signal, code_label, args.barrier_mode,
    )
    print(report_text)

    report_path = save_target_report(report_text, output_dir)
    print(f"\nレポート保存完了: {report_path}")
```

### 2-3. `--today` モードとの組み合わせ

`--today --target-pct 0.15` でも動作する。step9はstep1_csv + step7シグナルだけあれば動作するため、パイプライン再実行不要。

---

## 3. 変更影響範囲

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `src/step9_target_estimator.py` | **新規作成** | 全ロジック（推定・レポート生成・保存） |
| `main.py` | 修正 | `--target-price`/`--target-pct`引数追加 + `main()`末尾にstep9ブロック追加 |

**変更しないファイル**: step1〜step8（既存のパイプラインは一切変更しない）

---

## 4. テスト計画

### 4-1. 手動テスト

```bash
# 1. step1〜7の通常パイプライン + 目標価格指定
python main.py --code 7701 --target-price 4500

# 2. 上昇率指定
python main.py --code 7701 --target-pct 0.15

# 3. 固定%モードと組み合わせ
python main.py --code 7701 --barrier-mode fixed_pct --target-pct 0.10

# 4. --todayと組み合わせ（パイプライン再実行なし）
python main.py --code 7701 --today --target-pct 0.15

# 5. target-pct のみ（バリアモードはatrでもfixed_pctでもOK）
python main.py --code 7701 --target-pct 0.20

# 6. --scan との併用は想定しない（step9は単一銘柄向け）
```

### 4-2. 異常系テスト

- `--target-price` と `--target-pct` を同時指定 → エラーメッセージ
- `--target-price` が現在値以下 → エラーメッセージ
- `step1_dataset.csv` が存在しない → ファイルエラー
- step7のシグナルが取得できない（step5しかない等）→ signal="不明"でフォールバック

---

## 5. 実装手順（TODO）

1. `src/step9_target_estimator.py` を作成
   - `estimate_empirical()` — 経験的確率推定
   - `estimate_monte_carlo()` — GBMモンテカルロ
   - `generate_regime_comment()` — シグナル別コメント
   - `build_target_report()` — レポートテキスト生成
   - `save_target_report()` — ファイル保存
2. `main.py` に `--target-price` / `--target-pct` 引数を追加
3. `main.py` の `main()` 末尾にstep9ブロックを追加
4. 手動テスト実行（`--target-price`, `--target-pct`, `--today` との組み合わせ）

---

## 6. 注意点・確認事項

- [ ] `step1_dataset.csv` の existence が前提。ない場合はエラーハンドリング
- [ ] `--today` + `--target-pct` の組み合わせでパイプライン再実行せず動作することを確認
- [ ] `--scan` との併用は想定しない（step9は単一銘柄の追加推定）
- [ ] モンテカルロは `n_paths=10_000` で数秒かかる可能性。必要なら `random_state` 固定で再現性確保
- [ ] `scipy` のインストールがない場合は `scipy.stats` は使わない（純粋なnumpy実装でOK）
- [ ] MCのμに中央値を使う判断は実データ検証済み（STEP9_PLAN.md参照）。変更時は要検証
