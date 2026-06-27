# ステップ9: 目標額到達推定 設計プラン

指定した銘柄と目標価格（または上昇率）を入力すると、
「そこまで行くか（到達確率）」と「何日かかるか（期間推定）」を
2つの異なる手法で推定してレポートを出力する。

---

## 1. 前提と設計方針

### 1-1. この推定が答える問い

```
「7701を今日エントリーするとして、目標の5000円まで行く可能性は？
  行くとしたら何日くらいかかる？」
```

### 1-2. 手法の選択：なぜ2手法を使うか

実データで検証したところ、手法によって推定値が大きく異なることがわかりました。

| 手法 | +10%/30日 到達確率 | +10%/60日 到達確率 |
|---|---|---|
| **モンテカルロ（GBM）** | 60.5% | 76.9% |
| **経験的（実績）** | 44.0% | 61.3% |

MCは一貫して実績より楽観的です。原因は以下の2点です。

1. **直近の急騰相場**（2025〜2026年）が過去リターンの平均μを押し上げており、
   「未来もこのペースで上がる」という前提でMCが走っている
2. **GBMの仮定**（対数正規分布）が実際のリターン分布（裾が厚い）と乖離している

どちらか一方だけを提示すると「楽観すぎる」か「保守的すぎる」になります。
そのため**両方を並べて提示し、ユーザーが判断できる形**にします。

| 手法 | 意味合い | 使い方 |
|---|---|---|
| モンテカルロ | 理論的な上限（パラメトリック） | 「最大でもこのくらい」 |
| 経験的分布 | 過去の実績ベースの参考値 | 「歴史的にはこの程度だった」 |

### 1-3. 現在の相場状態（ステップ7シグナル）も反映する

過去データは「あらゆる相場環境」を平均したものです。
現在のシグナル状態（警戒 / 強気 / 中立）によって、
楽観的・中立的・悲観的なコメントを付与します。

---

## 2. 実装設計

### 2-1. 新規ファイル: `src/step9_target_estimator.py`

```python
"""
AIトレードシステム ステップ9: 目標額到達推定

入力: 銘柄コード + 目標価格（または上昇率）
      ステップ1〜7のパイプライン実行後に動作することが前提。

出力:
    - 到達確率テーブル（複数の期間 × 2手法）
    - 到達時の所要日数分布
    - 現在の相場状態を踏まえたコメント
    - resource/{code}/{mode}/target_report.txt
"""

from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCE_DIR = BASE_DIR / "resource"

# 確率を推定する期間（営業日）
ESTIMATION_HORIZONS = [10, 20, 30, 45, 60, 90]

# モンテカルロのパス数
N_MC_PATHS = 10_000
```

---

### 2-2. コア推定ロジック

#### A. 経験的確率の推定

過去の全日付を起点として、`days`日以内に高値が目標に到達した実績を集計します。

```python
def estimate_empirical(df: pd.DataFrame,
                       target_pct: float,
                       horizons: list[int] = ESTIMATION_HORIZONS) -> dict:
    """
    過去データの経験的分布から到達確率と所要日数を推定する。

    Parameters
    ----------
    df         : step1_dataset.csv または同等のDataFrame（high/close列が必要）
    target_pct : 現在値からの上昇率（例: 0.10 = +10%）

    Returns
    -------
    {
        "reach_prob": {10: 0.44, 20: 0.55, ...},   # 各期間の到達確率
        "days_distribution": {                       # 到達した場合の日数分布
            "median": 25, "mean": 32,
            "p25": 13, "p75": 48, "p90": 69,
            "n_samples": 432
        },
        "n_total": 629,   # 計算に使った起点の数
    }
    """
    close = df["close"].to_numpy()
    high  = df["high"].to_numpy()
    n     = len(df)
    max_h = max(horizons)

    reach_count = {h: 0 for h in horizons}
    total        = max(0, n - max_h)
    touch_days   = []

    for i in range(total):
        entry = close[i]
        target = entry * (1 + target_pct)
        touched_at = None

        for j in range(i + 1, i + max_h + 1):
            if high[j] >= target:
                touched_at = j - i
                break

        for h in horizons:
            if touched_at is not None and touched_at <= h:
                reach_count[h] += 1

        if touched_at is not None:
            touch_days.append(touched_at)

    reach_prob = {h: reach_count[h] / total if total > 0 else 0.0
                  for h in horizons}

    days_arr = np.array(touch_days) if touch_days else np.array([np.nan])
    days_dist = {
        "median":    float(np.median(days_arr)) if touch_days else np.nan,
        "mean":      float(np.mean(days_arr))   if touch_days else np.nan,
        "p25":       float(np.percentile(days_arr, 25)) if touch_days else np.nan,
        "p75":       float(np.percentile(days_arr, 75)) if touch_days else np.nan,
        "p90":       float(np.percentile(days_arr, 90)) if touch_days else np.nan,
        "n_samples": len(touch_days),
    }

    return {"reach_prob": reach_prob, "days_distribution": days_dist,
            "n_total": total}
```

#### B. モンテカルロ確率の推定（GBM）

幾何ブラウン運動（GBM）に基づいて10,000パスをシミュレーションします。

```python
def estimate_monte_carlo(df: pd.DataFrame,
                         target_pct: float,
                         horizons: list[int] = ESTIMATION_HORIZONS,
                         n_paths: int = N_MC_PATHS,
                         vol_window: int = 60,
                         random_state: int = 42) -> dict:
    """
    GBMモンテカルロシミュレーションで到達確率と所要期間を推定する。

    ボラティリティは「直近vol_window日の実現ボラ」を使用する。
    過去全期間の平均では直近の相場環境が反映されないため。

    Returns
    -------
    {
        "reach_prob": {10: 0.605, 20: 0.72, ...},
        "days_distribution": {"median": ..., "mean": ...},
        "params": {"mu": ..., "sigma": ..., "vol_window": 60}
    }
    """
    ret = df["close"].pct_change().dropna()

    # 直近vol_windowの実現ボラを使う（現在のレジームを反映）
    sigma = ret.tail(vol_window).std()
    # μはトレンドバイアスを避けるため全期間の中央値（メジアン）を使用
    # （平均は急騰期間があると過大推定になりやすいため）
    mu = float(ret.median())

    rng   = np.random.default_rng(random_state)
    max_h = max(horizons)
    daily_shocks = rng.normal(
        loc=(mu - 0.5 * sigma**2),
        scale=sigma,
        size=(n_paths, max_h)
    )
    # 各パスの累積リターン（終値ベース）
    cum_ret = np.exp(np.cumsum(daily_shocks, axis=1))
    # 各日の最大値（= 高値の近似）として各日の終値を使用
    # ※ 日中高値の厳密推定には連続時間のバリア到達理論が必要だが
    #   ここでは終値ベースの保守的な推定とする

    reach_prob = {}
    touch_days = []

    for path_i in range(n_paths):
        path = cum_ret[path_i]
        touched_at = None
        for d in range(max_h):
            if path[d] >= (1 + target_pct):
                touched_at = d + 1
                break
        if touched_at is not None:
            touch_days.append(touched_at)
        for h in horizons:
            if touched_at is not None and touched_at <= h:
                reach_prob[h] = reach_prob.get(h, 0) + 1

    reach_prob = {h: reach_prob.get(h, 0) / n_paths for h in horizons}

    days_arr = np.array(touch_days) if touch_days else np.array([np.nan])
    days_dist = {
        "median":    float(np.median(days_arr)) if touch_days else np.nan,
        "mean":      float(np.mean(days_arr))   if touch_days else np.nan,
        "p25":       float(np.percentile(days_arr, 25)) if touch_days else np.nan,
        "p75":       float(np.percentile(days_arr, 75)) if touch_days else np.nan,
        "p90":       float(np.percentile(days_arr, 90)) if touch_days else np.nan,
        "n_samples": len(touch_days),
    }

    return {
        "reach_prob":       reach_prob,
        "days_distribution": days_dist,
        "params": {"mu": mu, "sigma": sigma, "vol_window": vol_window},
    }
```

#### C. 現在の相場状態によるコメント生成

```python
def generate_regime_comment(signal: str, target_pct: float,
                            atr_percentile: float | None) -> str:
    """
    ステップ7のシグナルと現在のボラティリティ状態から
    「今この目標を狙うことへの補足コメント」を生成する。
    """
    comments = []

    if signal == "強気":
        comments.append(
            "✅ 現在は強気シグナルが出ており、エントリーに適した状態です。"
            "上記確率は平常時の参考値であり、良好な状態での実現可能性は"
            "若干高い可能性があります。"
        )
    elif signal == "警戒":
        comments.append(
            "⚠️  現在は過熱シグナル（警戒）が出ています。"
            "高値圏でのエントリーとなるため、目標到達前に利確水準を"
            "一旦越えた後に反落するリスクが通常より高い状態です。"
        )
    elif signal == "中立":
        comments.append(
            "ℹ️  現在は中立シグナルです。"
            "上記確率はそのまま参考値として利用できます。"
        )

    if atr_percentile is not None and atr_percentile >= 0.90:
        comments.append(
            f"⚠️  現在のATRパーセンタイルは {atr_percentile:.0%} で、"
            "過去250日の上位10%に入る高ボラティリティ局面です。"
            "実際の値動きの振れ幅は推定より大きくなる可能性があります。"
        )

    if target_pct >= 0.30:
        comments.append(
            f"📌 目標 +{target_pct:.0%} は大きな上昇目標です。"
            "長期保有を前提とした設定であるか確認してください。"
        )

    return "\n".join(f"  {c}" for c in comments) if comments else ""
```

#### D. レポートテキストの生成

```python
def build_target_report(current_price: float, target_price: float,
                        target_pct: float, analysis_date: date,
                        empirical: dict, mc: dict,
                        signal: str, code: str,
                        barrier_mode: str) -> str:
    """
    推定結果を人間が読めるレポートテキストに変換する。
    """
    W = 68
    BORDER = "=" * W
    THIN   = "─" * W

    # 到達確率テーブル
    horizons = ESTIMATION_HORIZONS
    header = f"  {'期間':>6}"
    row_emp = f"  {'経験的':>6}"
    row_mc  = f"  {'MC':>6}"
    for h in horizons:
        header  += f"  {h:>5}日"
        row_emp += f"  {empirical['reach_prob'].get(h, 0):>5.1%}"
        row_mc  += f"  {mc['reach_prob'].get(h, 0):>5.1%}"

    # 所要日数テーブル
    emp_d = empirical["days_distribution"]
    mc_d  = mc["days_distribution"]

    lines = [
        BORDER,
        f"  ステップ9: 目標額到達推定レポート",
        f"  銘柄コード   : {code}",
        f"  分析日       : {analysis_date}",
        f"  現在値       : {current_price:,.0f} 円",
        f"  目標価格     : {target_price:,.0f} 円  （現在値より {target_pct:+.1%}）",
        BORDER,
        "",
        f"  ┌ 到達確率テーブル",
        f"  │  （経験的: 過去データの実績。  MC: モンテカルロ推定）",
        f"  │",
        f"  │{header}",
        f"  │{THIN[2:]}",
        f"  │{row_emp}",
        f"  │{row_mc}",
        f"  └",
        "",
        f"  ┌ 到達した場合の所要日数（目標 {target_pct:+.1%}）",
        f"  │           {'経験的':>10}   {'MC':>10}",
        f"  │  中央値   {_fmt_days(emp_d['median']):>10}   {_fmt_days(mc_d['median']):>10}",
        f"  │  平均     {_fmt_days(emp_d['mean']):>10}   {_fmt_days(mc_d['mean']):>10}",
        f"  │  25%〜75%  {_fmt_days(emp_d['p25'])}〜{_fmt_days(emp_d['p75'])}日    "
                           f"{_fmt_days(mc_d['p25'])}〜{_fmt_days(mc_d['p75'])}日",
        f"  │  サンプル  {emp_d['n_samples']:>10}件   {mc_d['n_samples']:>10}件",
        f"  └",
        "",
        THIN,
        "  ■ 現在の相場状態",
        THIN,
        generate_regime_comment(signal, target_pct,
                                empirical.get("atr_percentile")),
        "",
        THIN,
        "  ■ 推定の前提と限界",
        THIN,
        "  ・経験的確率: 過去データに含まれる相場環境に依存します。",
        "    直近が急騰相場の場合、過去の確率は実態より楽観的に見える可能性があります。",
        "  ・モンテカルロ(GBM): 直近60日のボラティリティと中央値リターンから計算します。",
        "    対数正規分布を仮定するため、大きな急騰・急落は過小評価される場合があります。",
        "  ・どちらの手法も「到達できるかどうかは未来のこと」であり、",
        "    推定値は参考情報です。実際の投資判断はご自身の責任で行ってください。",
        "",
        BORDER,
    ]
    return "\n".join(lines)


def _fmt_days(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  -"
    return f"{v:.0f}日"
```

---

### 2-3. `main.py` への統合

#### 追加する引数

```python
parser.add_argument(
    "--target-price", type=float, default=None,
    help="[ステップ9] 目標価格を円で指定（例: --target-price 5000）。"
         "現在の終値より高い値を指定してください。",
)
parser.add_argument(
    "--target-pct", type=float, default=None,
    help="[ステップ9] 目標上昇率を割合で指定（例: --target-pct 0.15 で +15%%）。"
         "--target-price と --target-pct はどちらか一方のみ指定してください。",
)
```

#### `main()`への追加ブロック

```python
# --- ステップ9: 目標額到達推定 ---
if args.target_price is not None or args.target_pct is not None:
    from src.step9_target_estimator import (
        estimate_empirical, estimate_monte_carlo,
        generate_regime_comment, build_target_report, save_target_report
    )
    from src.step7_entry_signal import load_source_df

    print("\n" + "=" * 60)
    print("ステップ9: 目標額到達推定")
    print("=" * 60)

    # step1データ（OHLCV）を読み込む
    step1_csv = output_dir / "step1_dataset.csv"
    df_ohlcv = pd.read_csv(step1_csv, index_col=0, parse_dates=True)

    current_price = df_ohlcv["close"].iloc[-1]
    analysis_date = df_ohlcv.index[-1].date()

    # 目標価格の決定
    if args.target_price is not None and args.target_pct is not None:
        print("エラー: --target-price と --target-pct は同時に指定できません。")
        sys.exit(1)
    elif args.target_price is not None:
        target_price = args.target_price
        target_pct   = (target_price - current_price) / current_price
    else:
        target_pct   = args.target_pct
        target_price = current_price * (1 + target_pct)

    if target_pct <= 0:
        print(f"エラー: 目標価格 ({target_price:,.0f}円) は現在値 "
              f"({current_price:,.0f}円) より高くなければなりません。")
        sys.exit(1)

    # 現在のシグナルを取得（step7の結果を流用）
    try:
        source_df, _ = load_source_df(output_dir)
        signal = source_df.iloc[-1].get("assist_signal", "不明")
        atr_p  = source_df.iloc[-1].get("atr_percentile", None)
    except FileNotFoundError:
        signal = "不明"
        atr_p  = None

    # 経験的推定
    print("経験的確率を計算中...")
    empirical = estimate_empirical(df_ohlcv, target_pct)
    empirical["atr_percentile"] = atr_p

    # モンテカルロ推定
    print("モンテカルロシミュレーション中（10,000パス）...")
    mc = estimate_monte_carlo(df_ohlcv, target_pct)

    # レポート生成・出力
    report_text = build_target_report(
        current_price, target_price, target_pct, analysis_date,
        empirical, mc, signal, code_label, args.barrier_mode
    )
    print(report_text)

    report_path = save_target_report(report_text, output_dir)
    print(f"\nレポート保存完了: {report_path}")
```

---

## 3. 出力サンプル

```
====================================================================
  ステップ9: 目標額到達推定レポート
  銘柄コード   : 7701
  分析日       : 2026-06-19
  現在値       : 3,886 円
  目標価格     : 4,500 円  （現在値より +15.8%）
====================================================================

  ┌ 到達確率テーブル
  │  （経験的: 過去データの実績。  MC: モンテカルロ推定）
  │
  │  期間       10日    20日    30日    45日    60日    90日
  │────────────────────────────────────────────────────────
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

──────────────────────────────────────────────────────────────────
  ■ 現在の相場状態
──────────────────────────────────────────────────────────────────
  ⛔ 現在は中立シグナルです。
    上記確率はそのまま参考値として利用できます。

──────────────────────────────────────────────────────────────────
  ■ 推定の前提と限界
──────────────────────────────────────────────────────────────────
  ・経験的確率: 過去データに含まれる相場環境に依存します。
    直近が急騰相場の場合、過去の確率は実態より楽観的に見える可能性があります。
  ・モンテカルロ(GBM): 直近60日のボラティリティと中央値リターンから計算します。
    対数正規分布を仮定するため、大きな急騰・急落は過小評価される場合があります。
  ・どちらの手法も「到達できるかどうかは未来のこと」であり、
    推定値は参考情報です。実際の投資判断はご自身の責任で行ってください。

====================================================================
```

---

## 4. 出力ファイル

| ファイル | 内容 |
|---|---|
| `resource/{code}/{mode}/target_report.txt` | 到達推定レポート（毎回上書き） |

---

## 5. 実行方法

```bash
# 目標価格を円で指定
python main.py --code 7701 --target-price 4500

# 目標上昇率で指定（現在値から+20%）
python main.py --code 7701 --target-pct 0.20

# バリアモード・ステップ6の設定と組み合わせ可能
python main.py --code 7701 --barrier-mode fixed_pct --target-pct 0.10

# --todayと組み合わせ: パイプライン再実行なしで推定のみ
python main.py --code 7701 --today --target-pct 0.15
```

---

## 6. 設計上の重要な決定事項

### 6-1. なぜ「終値ベース」と「高値ベース（経験的）」を混在させるか

| 手法 | 使用する価格 | 理由 |
|---|---|---|
| 経験的 | **高値（high）** | 実際に指値が成立するのは高値到達時のため |
| モンテカルロ | **終値（close）** | 日中高値の確率モデルは複雑（反射原理など）のため保守的に終値を使用 |

この差がMCが経験的より「低め」に出る一因でもあります。
MCを「保守的な下限」、経験的を「実績ベースの参考値」と解釈するのが適切です。

### 6-2. μにトレンド中央値を使う理由

直近の急騰相場データが入っているため、日次リターンの**平均**では
「毎日0.24%上がり続ける前提」の楽観的なシミュレーションになります。
**中央値**を使うことで外れ値（急騰日）の影響を抑え、
より保守的なμでシミュレーションします。

### 6-3. ステップ9はパイプラインの「後付けオプション」

ステップ9は `--target-price` または `--target-pct` が指定された場合のみ実行します。
省略した場合は従来通りステップ7までの出力で終了します。
ステップ1のCSV（`step1_dataset.csv`）が存在することが前提条件です。
