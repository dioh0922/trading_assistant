import numpy as np
import pandas as pd
from pathlib import Path

def add_entry_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    ルールベースの entry_score を追加する。
    加点/減点ロジックに基づき、各行のスコアを計算する。
    """
    d = df.copy()
    
    score = np.zeros(len(d), dtype=int)
    
    # 1. 週足トレンド (weekly_trend)
    # 上昇なら+2, 下降なら-1
    if "weekly_trend" in d.columns:
        score += np.where(d["weekly_trend"] == 1, 2, 0)
        score += np.where(d["weekly_trend"] == -1, -1, 0)
        
    # 2. RSI14 (rsi14)
    # 35〜50なら+1, 65以上なら-1
    if "rsi14" in d.columns:
        score += np.where((d["rsi14"] >= 35) & (d["rsi14"] <= 50), 1, 0)
        score += np.where(d["rsi14"] >= 65, -1, 0)
        
    # 3. 出来高比率 (volume_ratio)
    # 1.0以上なら+1
    if "volume_ratio" in d.columns:
        score += np.where(d["volume_ratio"] >= 1.0, 1, 0)
        
    # 4. MA乖離改善 (dev_ma25_zscore が5日前より改善)
    if "dev_ma25_zscore" in d.columns:
        dev_diff = d["dev_ma25_zscore"] - d["dev_ma25_zscore"].shift(5)
        score += np.where(dev_diff > 0, 1, 0)
        
    # 5. ATR正規化乖離が過熱していない (atr_dev_ma25 < 0.5)
    if "atr_dev_ma25" in d.columns:
        score += np.where(d["atr_dev_ma25"] < 0.5, 1, 0)
        
    # 6. ATRパーセンタイルが過熱 (atr_percentile >= 0.90) なら-2
    if "atr_percentile" in d.columns:
        score += np.where(d["atr_percentile"] >= 0.90, -2, 0)
        
    # 7. is_overbought_heat == 1 (過熱買い) なら-2
    if "is_overbought_heat" in d.columns:
        score += np.where(d["is_overbought_heat"] == 1, -2, 0)
        
    d["entry_score"] = score
    
    # 決定フラグ
    d["score_decision"] = np.where(score >= 3, "エントリー候補", 
                                   np.where(score >= 1, "監視", "見送り"))
    
    return d

def evaluate_trade_performance(df: pd.DataFrame, entry_col: str) -> dict:
    """
    指定した entry_col (値が1またはTrueである行) に基づく売買成績を計算する。
    """
    # エントリーシグナルが出ている行を抽出
    entries = df[df[entry_col] == 1].copy()
    total_samples = len(df)
    entry_count = len(entries)
    
    if entry_count == 0:
        return {
            "entry_count": 0,
            "entry_rate": 0.0,
            "win_rate": 0.0,
            "mean_return": 0.0,
            "median_return": 0.0,
            "total_return": 0.0,
            "max_loss": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0
        }
        
    # 勝率: tb_return > 0 または tb_label == 1
    # ※ tb_label == 1 を勝利率とする
    win_rate = (entries["tb_label"] == 1).mean() if "tb_label" in entries.columns else 0.0
    
    # リターン統計
    returns = entries["tb_return"] if "tb_return" in entries.columns else pd.Series(dtype=float)
    mean_ret = returns.mean() if not returns.empty else 0.0
    median_ret = returns.median() if not returns.empty else 0.0
    total_ret = returns.sum() if not returns.empty else 0.0
    max_loss = returns.min() if not returns.empty else 0.0
    
    # 最大ドローダウン (forward_max_drawdownの最小値)
    max_dd = entries["forward_max_drawdown"].min() if "forward_max_drawdown" in entries.columns else 0.0
    
    # プロフィットファクター (PF)
    pf = 0.0
    if not returns.empty:
        gains = returns[returns > 0].sum()
        losses = returns[returns < 0].sum()
        if losses == 0:
            pf = np.inf if gains > 0 else 1.0
        else:
            pf = gains / abs(losses)
            
    return {
        "entry_count": entry_count,
        "entry_rate": entry_count / total_samples,
        "win_rate": win_rate,
        "mean_return": mean_ret,
        "median_return": median_ret,
        "total_return": total_ret,
        "max_loss": max_loss,
        "max_drawdown": max_dd,
        "profit_factor": pf
    }

def evaluate_score_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """
    entry_score の閾値別成績を集計する。
    """
    d = df.copy()
    if "entry_score" not in d.columns:
        d = add_entry_score(d)
        
    records = []
    # スコアの範囲を取得 (-5 から +7 など)
    min_score = int(d["entry_score"].min())
    max_score = int(d["entry_score"].max())
    
    for th in range(min_score, max_score + 1):
        # entry_score >= th をシグナルとする
        d[f"tmp_entry_th_{th}"] = np.where(d["entry_score"] >= th, 1, 0)
        perf = evaluate_trade_performance(d, f"tmp_entry_th_{th}")
        perf["threshold"] = th
        records.append(perf)
        
    res_df = pd.DataFrame(records)
    # カラム並び替え
    cols = ["threshold", "entry_count", "entry_rate", "win_rate", "mean_return", 
            "median_return", "total_return", "max_loss", "max_drawdown", "profit_factor"]
    return res_df[cols].set_index("threshold")

def build_condition_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    RSI帯、ATR帯、週足、出来高などの条件別に過去成績を集計し、期待値を可視化する。
    """
    d = df.copy()
    records = []
    
    # 1. RSI帯
    if "rsi14" in d.columns:
        rsi_bins = [0, 30, 40, 50, 60, 100]
        rsi_labels = ["RSI <30", "RSI 30-40", "RSI 40-50", "RSI 50-60", "RSI >=60"]
        d["rsi_group"] = pd.cut(d["rsi14"], bins=rsi_bins, labels=rsi_labels)
        for label in rsi_labels:
            d[f"tmp_rsi_{label}"] = np.where(d["rsi_group"] == label, 1, 0)
            perf = evaluate_trade_performance(d, f"tmp_rsi_{label}")
            perf["condition"] = label
            records.append(perf)
            
    # 2. ATRパーセンタイル帯
    if "atr_percentile" in d.columns:
        atr_bins = [0.0, 0.3, 0.7, 0.9, 1.0]
        atr_labels = ["ATR <0.3 (低ボラ)", "ATR 0.3-0.7 (中ボラ)", "ATR 0.7-0.9 (高ボラ)", "ATR >=0.9 (極大ボラ)"]
        d["atr_group"] = pd.cut(d["atr_percentile"], bins=atr_bins, labels=atr_labels)
        for label in atr_labels:
            d[f"tmp_atr_{label}"] = np.where(d["atr_group"] == label, 1, 0)
            perf = evaluate_trade_performance(d, f"tmp_atr_{label}")
            perf["condition"] = label
            records.append(perf)
            
    # 3. 週足トレンド
    if "weekly_trend" in d.columns:
        for trend, label in [(1, "週足トレンド:上昇"), (-1, "週足トレンド:下降")]:
            d[f"tmp_trend_{trend}"] = np.where(d["weekly_trend"] == trend, 1, 0)
            perf = evaluate_trade_performance(d, f"tmp_trend_{trend}")
            perf["condition"] = label
            records.append(perf)
            
    # 4. 出来高比率
    if "volume_ratio" in d.columns:
        vol_bins = [0.0, 1.0, 1.5, np.inf]
        vol_labels = ["出来高比率 <1.0", "出来高比率 1.0-1.5", "出来高比率 >=1.5 (急増)"]
        d["vol_group"] = pd.cut(d["volume_ratio"], bins=vol_bins, labels=vol_labels)
        for label in vol_labels:
            d[f"tmp_vol_{label}"] = np.where(d["vol_group"] == label, 1, 0)
            perf = evaluate_trade_performance(d, f"tmp_vol_{label}")
            perf["condition"] = label
            records.append(perf)
            
    res_df = pd.DataFrame(records)
    cols = ["condition", "entry_count", "entry_rate", "win_rate", "mean_return", 
            "median_return", "total_return", "max_loss", "max_drawdown", "profit_factor"]
    return res_df[cols].set_index("condition")

def save_edge_analysis(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    エッジ分析を実行し、CSVとして保存する。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. スコアを付与
    scored_df = add_entry_score(df)
    scored_df.to_csv(output_dir / "step4_scored_dataset.csv")
    
    # 2. スコア閾値別の検証
    threshold_stats = evaluate_score_thresholds(scored_df)
    threshold_stats.to_csv(output_dir / "step4_score_threshold_report.csv")
    
    # 3. 条件別の検証
    condition_stats = build_condition_stats(scored_df)
    condition_stats.to_csv(output_dir / "step4_condition_stats.csv")
    
    # 4. 全体パフォーマンス（基準スコア3以上）
    scored_df["default_entry_sig"] = np.where(scored_df["entry_score"] >= 3, 1, 0)
    default_perf = evaluate_trade_performance(scored_df, "default_entry_sig")
    
    return {
        "scored_dataset_path": output_dir / "step4_scored_dataset.csv",
        "threshold_report_path": output_dir / "step4_score_threshold_report.csv",
        "condition_stats_path": output_dir / "step4_condition_stats.csv",
        "default_performance": default_perf
    }
