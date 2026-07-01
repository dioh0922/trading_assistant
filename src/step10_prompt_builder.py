import pandas as pd
from pathlib import Path

_PROMPT_TEMPLATE = """【指示】
あなたはプロのクオンツトレーダーです。以下の定量データを総合的に分析し、
今日「買いエントリー」をすべきか、見送るべきかを判断してください。

【銘柄情報】
コード: {code}

【定量データ（パイプライン出力）】
{quantitative_data}

【出力フォーマット】
以下のJSON形式で必ず出力してください:
{{
  "final_decision": "エントリー / 見送り",
  "confidence_score": 1-5,
  "reason_summary": "判断理由の要約（50字以内）",
  "key_risks": ["リスク1", "リスク2"],
  "key_opportunities": ["好材料1", "好材料2"]
}}
"""

def build_llm_prompt(
    code: str,
    last_row: pd.Series,
    target_report_info: dict | None = None
) -> str:
    """
    全パイプライン出力を1つのプロンプトテキストに統合する。
    
    Parameters
    ----------
    code : str
        銘柄コード
    last_row : pd.Series
        パイプラインの最新行データ（step5_datasetやstep6_datasetの最終行）
    target_report_info : dict, optional
        Step9目標額到達推定の結果
    """
    # 定量データの文字列構築
    lines = []
    
    # 週足トレンド
    weekly_trend = last_row.get("weekly_trend", "不明")
    lines.append(f"  週足トレンド      : {weekly_trend}")
    
    # RSI
    rsi14 = last_row.get("rsi14", last_row.get("rsi", last_row.get("RSI", "不明")))
    if isinstance(rsi14, (int, float)):
        if rsi14 < 30:
            rsi_desc = f"{rsi14:.1f} (売られすぎ)"
        elif rsi14 > 70:
            rsi_desc = f"{rsi14:.1f} (買われすぎ)"
        else:
            rsi_desc = f"{rsi14:.1f} (中立)"
    else:
        rsi_desc = str(rsi14)
    lines.append(f"  RSI(14)           : {rsi_desc}")
    # entry_score and related fields
    entry_score = last_row.get("entry_score", "不明")
    lines.append(f"  entry_score       : {entry_score}")
    # score_decision (same as entry_score >= threshold?)
    score_decision = ">=4" if isinstance(entry_score, (int, float)) and entry_score >= 4 else "<4"
    lines.append(f"  score_decision    : {score_decision}")
    signal_type = last_row.get("signal_type", "不明")
    lines.append(f"  signal_type       : {signal_type}")
    final_decision = last_row.get("final_decision", "不明")
    lines.append(f"  final_decision    : {final_decision}")
    
    # ATRパーセンタイル
    atr_p = last_row.get("atr_percentile", "不明")
    if isinstance(atr_p, (int, float)):
        atr_pct = atr_p * 100 if atr_p <= 1.0 else atr_p
        if atr_pct < 33:
            atr_desc = f"{atr_pct:.1f}% (低ボラティリティ)"
        elif atr_pct > 66:
            atr_desc = f"{atr_pct:.1f}% (高ボラティリティ)"
        else:
            atr_desc = f"{atr_pct:.1f}% (中ボラティリティ)"
    else:
        atr_desc = str(atr_p)
    lines.append(f"  ATRパーセンテイル  : {atr_desc}")
    
    # アシストシグナル
    assist_signal = last_row.get("assist_signal", "不明")
    lines.append(f"  アシストシグナル  : {assist_signal}")
    
    # ML予測確率
    y_proba = last_row.get("y_proba", None)
    if y_proba is not None and isinstance(y_proba, (int, float)):
        lines.append(f"  ML予測確率        : {y_proba:.2f}")
    else:
        lines.append(f"  ML予測確率        : 不明")
        
    # DD（ドローダウン）予測確率
    dd_proba = last_row.get("drawdown_prob", last_row.get("dd_proba", None))
    if dd_proba is not None and isinstance(dd_proba, (int, float)):
        lines.append(f"  DD予測確率        : {dd_proba:.1%}")
    else:
        lines.append(f"  DD予測確率        : 不明")
        
    # 出来高比率 (Volume Ratio / 20日平均比などがあれば)
    volume_ratio = last_row.get("volume_ratio", None)
    if volume_ratio is not None and isinstance(volume_ratio, (int, float)):
        lines.append(f"  出来高比率        : {volume_ratio:.2f}")
    
    # 目標推定 (Step9の結果を付与)
    if target_report_info:
        lines.append("\n【目標推定（Step9）】")
        lines.append(f"  目標価格   : {target_report_info.get('target_price', '不明')}円 (現在比 {target_report_info.get('target_pct', 0):+.1%})")
        lines.append(f"  経験的確率 : {target_report_info.get('emp_prob_90d', 0):.1%} (90日)")
        lines.append(f"  MC確率     : {target_report_info.get('mc_prob_90d', 0):.1%} (90日)")
        
    quantitative_data = "\n".join(lines)
    
    # テンプレート埋め込み
    prompt = _PROMPT_TEMPLATE.format(
        code=code,
        quantitative_data=quantitative_data
    )
    return prompt

def save_prompt(prompt: str, output_dir: Path) -> Path:
    """プロンプトをテキストファイルとして保存する。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "llm_prompt.txt"
    path.write_text(prompt, encoding="utf-8")
    return path
