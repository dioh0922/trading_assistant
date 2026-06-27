"""
AIトレードシステム ステップ9: 目標額到達推定

入力: 銘柄コード + 目標価格（または上昇率）
      ステップ1〜7のパイプライン実行後に動作することが前提。

出力:
    - 到達確率テーブル（複数の期間 × 2手法）
    - 到達時の所要日数分布
    - 現在の相場状態を踏まえたコメント
    - resource/{code}/{mode}/target_report.txt

実行:
    python main.py --code 7701 --target-price 4500
    python main.py --code 7701 --target-pct 0.15
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.step1_feature_engineering import RESOURCE_DIR

# 確率を推定する期間（営業日）
ESTIMATION_HORIZONS = [10, 20, 30, 45, 60, 90]

# モンテカルロのパス数
N_MC_PATHS = 10_000

# ボラティリティ計算に使う直近日数
VOL_WINDOW = 60


# ─────────────────────────────────────────────────────────────────
# A. 経験的確率の推定
# ─────────────────────────────────────────────────────────────────

def estimate_empirical(df: pd.DataFrame,
                       target_pct: float,
                       horizons: list[int] | None = None) -> dict:
    """
    過去データの経験的分布から到達確率と所要日数を推定する。

    Parameters
    ----------
    df         : step1_dataset.csv 相当のDataFrame（high/close列が必要）
    target_pct : 現在値からの上昇率（例: 0.10 = +10%）
    horizons   : 評価する期間リスト（営業日）

    Returns
    -------
    {
        "reach_prob":       {10: 0.44, 20: 0.55, ...},
        "days_distribution": {
            "median": 25, "mean": 32,
            "p25": 13, "p75": 48, "p90": 69,
            "n_samples": 432
        },
        "n_total": 629,
    }
    """
    if horizons is None:
        horizons = ESTIMATION_HORIZONS

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    n = len(df)
    max_h = max(horizons)

    if n <= max_h:
        return _empty_result(horizons)

    total = n - max_h
    reach_count = {h: 0 for h in horizons}
    touch_days: list[int] = []

    for i in range(total):
        entry = close[i]
        target = entry * (1 + target_pct)
        touched_at: int | None = None

        for j in range(i + 1, i + max_h + 1):
            if high[j] >= target:
                touched_at = j - i
                break

        if touched_at is not None:
            touch_days.append(touched_at)
            for h in horizons:
                if touched_at <= h:
                    reach_count[h] += 1

    reach_prob = {
        h: reach_count[h] / total if total > 0 else 0.0
        for h in horizons
    }

    days_dist = _calc_days_distribution(touch_days)

    return {
        "reach_prob": reach_prob,
        "days_distribution": days_dist,
        "n_total": total,
    }


# ─────────────────────────────────────────────────────────────────
# B. モンテカルロ確率の推定（GBM）
# ─────────────────────────────────────────────────────────────────

def estimate_monte_carlo(df: pd.DataFrame,
                         target_pct: float,
                         horizons: list[int] | None = None,
                         n_paths: int = N_MC_PATHS,
                         vol_window: int = VOL_WINDOW,
                         random_state: int = 42) -> dict:
    """
    GBMモンテカルロシミュレーションで到達確率と所要期間を推定する。

    ボラティリティは直近 vol_window 日の実現ボラを使用。
    μ（ドリフト）は全期間の中央値を使用（急騰バイアス回避）。

    Returns
    -------
    {
        "reach_prob":       {10: 0.605, 20: 0.72, ...},
        "days_distribution": {"median": ..., "mean": ..., ...},
        "params":           {"mu": ..., "sigma": ..., "vol_window": 60},
    }
    """
    if horizons is None:
        horizons = ESTIMATION_HORIZONS

    ret = df["close"].pct_change().dropna()
    max_h = max(horizons)

    if len(ret) < vol_window or len(ret) < 2:
        return _empty_result(horizons)

    sigma = ret.tail(vol_window).std()
    mu = float(ret.median())

    rng = np.random.default_rng(random_state)
    daily_shocks = rng.normal(
        loc=(mu - 0.5 * sigma ** 2),
        scale=sigma,
        size=(n_paths, max_h),
    )
    cum_ret = np.exp(np.cumsum(daily_shocks, axis=1))

    reach_prob = {}
    touch_days: list[int] = []
    threshold = 1 + target_pct

    for path_i in range(n_paths):
        path = cum_ret[path_i]
        touched_at: int | None = None
        for d in range(max_h):
            if path[d] >= threshold:
                touched_at = d + 1
                break
        if touched_at is not None:
            touch_days.append(touched_at)
            for h in horizons:
                if touched_at <= h:
                    reach_prob[h] = reach_prob.get(h, 0) + 1

    reach_prob = {h: reach_prob.get(h, 0) / n_paths for h in horizons}
    days_dist = _calc_days_distribution(touch_days)

    return {
        "reach_prob": reach_prob,
        "days_distribution": days_dist,
        "params": {"mu": mu, "sigma": sigma, "vol_window": vol_window},
    }


# ─────────────────────────────────────────────────────────────────
# C. 現在の相場状態によるコメント生成
# ─────────────────────────────────────────────────────────────────

def generate_regime_comment(signal: str,
                            target_pct: float,
                            atr_percentile: float | None = None) -> str:
    """
    ステップ7のシグナルと現在のボラティリティ状態から
    「今この目標を狙うことへの補足コメント」を生成する。
    """
    comments: list[str] = []

    if signal == "強気":
        comments.append(
            "現在は強気シグナルが出ており、エントリーに適した状態です。"
            "上記確率は平常時の参考値であり、良好な状態での実現可能性は"
            "若干高い可能性があります。"
        )
    elif signal == "警戒":
        comments.append(
            "現在は過熱シグナル（警戒）が出ています。"
            "高値圏でのエントリーとなるため、目標到達前に利確水準を"
            "一旦越えた後に反落するリスクが通常より高い状態です。"
        )
    elif signal == "中立":
        comments.append(
            "現在は中立シグナルです。"
            "上記確率はそのまま参考値として利用できます。"
        )

    if atr_percentile is not None and atr_percentile >= 0.90:
        comments.append(
            f"現在のATRパーセンタイルは {atr_percentile:.0%} で、"
            "過去250日の上位10%に入る高ボラティリティ局面です。"
            "実際の値動きの振れ幅は推定より大きくなる可能性があります。"
        )

    if target_pct >= 0.30:
        comments.append(
            f"目標 +{target_pct:.0%} は大きな上昇目標です。"
            "長期保有を前提とした設定であるか確認してください。"
        )

    if not comments:
        return "  特になし"

    return "\n".join(f"  {c}" for c in comments)


# ─────────────────────────────────────────────────────────────────
# D. レポートテキストの生成
# ─────────────────────────────────────────────────────────────────

def build_target_report(current_price: float,
                        target_price: float,
                        target_pct: float,
                        analysis_date: date,
                        empirical: dict,
                        mc: dict,
                        signal: str,
                        code: str,
                        barrier_mode: str) -> str:
    """推定結果を人間が読めるレポートテキストに変換する。"""
    W = 68
    BORDER = "=" * W
    THIN = "─" * W

    horizons = ESTIMATION_HORIZONS
    header_parts = ["  期間"]
    row_emp_parts = ["  経験的"]
    row_mc_parts = ["  MC"]
    for h in horizons:
        header_parts.append(f"  {h:>5}日")
        emp_prob = empirical.get("reach_prob", {}).get(h, 0)
        mc_prob = mc.get("reach_prob", {}).get(h, 0)
        row_emp_parts.append(f"  {emp_prob:>5.1%}")
        row_mc_parts.append(f"  {mc_prob:>5.1%}")
    header = "".join(header_parts)
    row_emp = "".join(row_emp_parts)
    row_mc = "".join(row_mc_parts)

    emp_d = empirical.get("days_distribution", {})
    mc_d = mc.get("days_distribution", {})

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
        f"  │  中央値   {_fmt_days(emp_d.get('median')):>10}   {_fmt_days(mc_d.get('median')):>10}",
        f"  │  平均     {_fmt_days(emp_d.get('mean')):>10}   {_fmt_days(mc_d.get('mean')):>10}",
        f"  │  25%〜75%  {_fmt_days(emp_d.get('p25'))}〜{_fmt_days(emp_d.get('p75'))}    "
        f"{_fmt_days(mc_d.get('p25'))}〜{_fmt_days(mc_d.get('p75'))}",
        f"  │  サンプル  {emp_d.get('n_samples', 0):>10}件   {mc_d.get('n_samples', 0):>10}件",
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


def save_target_report(report_text: str, output_dir: Path) -> Path:
    """レポートを output_dir/target_report.txt として保存する。"""
    output_path = output_dir / "target_report.txt"
    output_path.write_text(report_text, encoding="utf-8")
    return output_path


# ─────────────────────────────────────────────────────────────────
# 内部ヘルパー
# ─────────────────────────────────────────────────────────────────

def _empty_result(horizons: list[int]) -> dict:
    return {
        "reach_prob": {h: 0.0 for h in horizons},
        "days_distribution": {
            "median": None, "mean": None,
            "p25": None, "p75": None, "p90": None,
            "n_samples": 0,
        },
        "n_total": 0,
    }


def _calc_days_distribution(touch_days: list[int]) -> dict:
    if not touch_days:
        return {
            "median": None, "mean": None,
            "p25": None, "p75": None, "p90": None,
            "n_samples": 0,
        }
    arr = np.array(touch_days)
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "n_samples": len(touch_days),
    }


def _fmt_days(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  -"
    return f"{v:.0f}日"
