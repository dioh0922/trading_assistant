import os
import json
import re
from pathlib import Path
from src.gemini import call_gemini

def call_llm(prompt: str, provider: str = "mock") -> dict:
    """
    プロンプトをLLM APIに送信し、判断結果をdictで返す。
    """
    provider_lower = provider.lower()
    
    if provider_lower == "mock":
        return _mock_llm_response(prompt)
    
    if provider_lower == "gemini":
        try:
            # 既存の gemini.py に実装されている call_gemini を呼び出す
            print("[Gemini API] call_geminiを実行中...")
            raw_text = call_gemini(prompt)
            return _parse_json_response(raw_text)
        except Exception as e:
            return _fallback_decision(f"Gemini API呼び出し中にエラーが発生しました: {e}")
            
    # その他のプロバイダ (OpenAIなど)
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    if not api_key:
        return _fallback_decision(f"APIキー ({provider.upper()}_API_KEY) が未設定のためモックで動作します")
    
    print(f"[{provider.upper()} API] 接続をシミュレート中...")
    return _mock_llm_response(prompt)

def _parse_json_response(raw_text: str) -> dict:
    """LLMから返された生テキストからJSON部分を抽出し、辞書型にパースする"""
    try:
        # Markdownのコードブロック等に囲まれているJSON文字列を抽出
        json_pattern = re.compile(r"\{.*\}", re.DOTALL)
        match = json_pattern.search(raw_text)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
        else:
            data = json.loads(raw_text)
            
        # 必要なキーの存在確認と型の標準化
        return {
            "final_decision": data.get("final_decision", "見送り"),
            "confidence_score": int(data.get("confidence_score", 3)),
            "reason_summary": data.get("reason_summary", "")[:50],
            "key_risks": data.get("key_risks", []),
            "key_opportunities": data.get("key_opportunities", []),
            "raw_response": raw_text
        }
    except Exception as e:
        return _fallback_decision(f"JSONパースエラー: {e}. 生出力: {raw_text[:100]}...")

def _mock_llm_response(prompt: str) -> dict:
    """プロンプトを簡易解析し、それらしいモック結果を返す"""
    decision = "見送り"
    confidence = 3
    reasons = ["定量モデルのシグナルが不十分です。"]
    risks = ["地合いの不確実性", "ボラティリティの低下"]
    opportunities = ["長期トレンドのサポート"]
    
    if "強気" in prompt or "上昇" in prompt:
        decision = "エントリー"
        confidence = 4
        reasons = ["週足トレンドが上昇で、定量シグナルが強気を示しています。"]
        opportunities.append("短期的なリバウンド期待")
    
    if "低ボラティリティ" in prompt:
        risks.append("レンジ相場での推移")
    elif "高ボラティリティ" in prompt:
        risks.append("ボラティリティ急拡大による損切りリスク")
        confidence = max(2, confidence - 1)
        
    return {
        "final_decision": decision,
        "confidence_score": confidence,
        "reason_summary": reasons[0][:50],
        "key_risks": risks,
        "key_opportunities": opportunities,
        "raw_response": json.dumps({
            "final_decision": decision,
            "confidence_score": confidence,
            "reason_summary": reasons[0][:50],
            "key_risks": risks,
            "key_opportunities": opportunities
        }, ensure_ascii=False)
    }

def _fallback_decision(reason: str) -> dict:
    """エラー時などのフォールバック"""
    return {
        "final_decision": "見送り (フォールバック)",
        "confidence_score": 1,
        "reason_summary": f"エラー: {reason}",
        "key_risks": ["API呼び出しまたは解析の失敗"],
        "key_opportunities": [],
        "raw_response": ""
    }

def build_llm_report(
    code: str,
    llm_result: dict,
    step7_decision: str = "不明"
) -> str:
    """
    LLM判断結果と既存パイプラインの判断を比較するレポートを生成する。
    """
    report_lines = [
        "============================================================",
        "ステップ11: LLM (定性・定量ハイブリッド) 判断レポート",
        "============================================================",
        f"  銘柄コード      : {code}",
        f"  定量判断 (Step7): {step7_decision}",
        "  ──────────────────────────────────────────────────────────",
        f"  LLM最終判断     : {llm_result.get('final_decision', '不明')}",
        f"  確信度 (1-5)    : {llm_result.get('confidence_score', 0)} / 5",
        f"  判断理由        : {llm_result.get('reason_summary', 'なし')}",
        "  主なリスク要素  :",
    ]
    for risk in llm_result.get("key_risks", []):
        report_lines.append(f"    - {risk}")
    
    report_lines.append("  主な好材料・機会:")
    for opp in llm_result.get("key_opportunities", []):
        report_lines.append(f"    - {opp}")
        
    report_lines.append("============================================================")
    
    return "\n".join(report_lines)

def save_llm_report(report_text: str, output_dir: Path) -> Path:
    """レポートを保存する。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "step11_llm_report.txt"
    path.write_text(report_text, encoding="utf-8")
    return path
