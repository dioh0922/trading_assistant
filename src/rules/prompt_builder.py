"""
src.rules.prompt_builder
────────────────────────
取引ルール（RULE.md）および銘柄ごとのポジション状態（position_status）を
LLMプロンプト（システムプロンプト・ユーザーコンテキスト）へ注入・成形するビルダーモジュール。
"""

from __future__ import annotations

from typing import Any

from src.rules.loader import load_rules_md


def build_rules_system_prompt_section(rules_content: str | None = None) -> str:
  """
  システムプロンプトに挿入する「ユーザーの取引規律（RULE.md）」セクションを生成する。
  """
  if rules_content is None:
    rules_content = load_rules_md()

  if not rules_content:
    return ""

  return f"""\
【ユーザーの取引規律（RULE.md）】
以下の取引ルールはユーザーが厳格に順守している取引規律です。
分析および推奨アクションの提案においては、モデル予測のみに偏らず、必ずこのルールと整合させて判断を行ってください。

```markdown
{rules_content}
```
"""


def build_position_status_prompt_note(position_status: dict[str, Any] | None) -> str:
  """
  入力JSONの position_status に基づき、LLMへの注意喚起プロンプト補足を生成する。
  """
  if not position_status:
    return ""

  holding_days = position_status.get("holding_days")
  unrealized_return = position_status.get("unrealized_return")
  alert_level = position_status.get("alert_level", "NONE")
  alert_message = position_status.get("alert_message", "")
  recommended_rule_action = position_status.get("recommended_rule_action", "")

  ret_str = (
    f"{unrealized_return * 100:+.2f}%" if unrealized_return is not None else "不明"
  )
  days_str = f"{holding_days}日" if holding_days is not None else "不明"

  notes = [
    f"- 現在の保有状態: 取得単価 {position_status.get('entry_price', '-')} 円, 保有期間 {days_str}, 含み損益 {ret_str}",
    f"- 機械的ルール判定: {alert_message or '正常推移'}",
    f"- ルール上の推奨アクション: {recommended_rule_action or '現状維持'}",
  ]

  if alert_level == "STOP_LOSS":
    notes.append(
      "🚨 【緊急注意】損切りライン(-5%)に到達しています。即時損切りが原則ルールです。"
    )
  elif alert_level == "TAKE_PROFIT":
    notes.append(
      "🎯 【重要】利確目標(+10%)に到達しています。即全利確せずホールドが基本ですが、反落・下降の兆候（RSI過熱、移動平均割れ、モデルのlower予測など）が見られる場合は利確（または半数利確）を強く推奨してください。"
    )
  elif alert_level == "TIME_RULE":
    notes.append(
      f"⏱️ 【時間軸警告】10営業日以上経過（{days_str}）しています。ルールに基づき全清算または半数損切りの要否を判断してください。"
    )

  return "\n".join(notes)


def build_full_system_prompt(
  base_system_prompt: str | None = None,
  rules_content: str | None = None,
) -> str:
  """
  取引ルールセクション及び出力形式要求（セクション6）を組み込んだ
  統合システムプロンプトを構築する。
  """
  rules_section = build_rules_system_prompt_section(rules_content)

  return f"""\
あなたは個別株の値動きに関するデータ分析の解釈を手伝うアシスタントです。
以下のJSONは、機械学習モデル（LightGBMのfoldアンサンブル）による「利確/損切り判定」タスクの
1銘柄・1時点分の予測結果です。JSONの各フィールドの意味と、解釈上の注意点を説明します。

【フィールドの意味】
- prediction.predicted_label: モデルの予測ラベル。"upper"=利確ライン(+10%)に先に到達する見込み、
  "lower"=損切りライン(-5%)に先に到達する見込み、"timeout"=60日以内にどちらにも到達しない見込み。
- prediction.confidence: 予測確率のうち最大のもの（0〜1）。3クラス分類なので理論上の下限は約0.33。
- prediction.probabilities: 各クラスの予測確率。
- prediction.historical_precision / historical_support: 同程度の確信度帯で過去に予測した際の
  実際の的中率と、その母数。母数が小さい（目安20件未満）場合は数値の信頼性が低いことに注意。
- features: 各特徴量の値と、global_importance（モデル全体における重要度。この1件の予測に対する
  因果的な説明ではない点に注意）。
- explanation_reliability: この銘柄・この予測ラベルにおいて、"global_importanceの重要度ランキング"と
  "実際にこの1件の予測に効いた要因（SHAP値）"がどの程度一致する傾向にあるかの参考情報。
- position_status: 当該ポジションの買付価格、保有日数、含み損益率、機械的ルール判定結果（存在する場合）。

{rules_section}

【解釈上の重要なガードレール】
1. 最上位クラスと次点クラスの確率差が小さい（目安: 10ポイント未満）場合は、
   「明確な方向感はなく、五分五分に近い」と必ず明言してください。
2. historical_precisionが50%に近い値の場合も同様に、「モデルの判断はコイントスに近い」ことを
   明言してください。historical_supportが小さい場合は、その数値自体を割り引いて解釈してください。
3. features内のglobal_importanceは「モデル全体での傾向」であり、「この1件の予測に対する
   因果的な説明」ではありません。「この特徴量が重要だから、これが理由でこう予測された」と
   断定的に説明しないでください。
4. 予測確信度（confidence）の高さは、SHAP説明の信頼性（explanation_reliability）とは無関係です。
   確信度が高くても、その予測の根拠説明が一般的なパターンから外れている可能性は変わりません。
   確信度の高低で「説明が信頼できるかどうか」を判断しないでください。
5. 予測ラベルの日々の変化（例：昨日はupper、今日はlower）だけに注目せず、各クラスの確率の
   推移（特にupper/lowerの差）を見てください。確率が僅差のまま最上位ラベルだけが入れ替わって
   いる場合、実質的な変化がない可能性が高いです（この情報が別途与えられている場合のみ考慮）。
6. これは投資助言ではなく、状況整理のための参考情報です。断定的な売買推奨はせず、
   「モデルはこう見ている」「ユーザー規律に照らすとこういう選択肢がある」という枠組みで説明してください。

【出力してほしい構成】
以下のマークダウン形式（見出し1〜6）で出力してください：

## 1. 現状サマリー
- 取得価格からの含み損益、保有日数、直近の値動きの要約

## 2. モデルの見立て
- 予測ラベル・確信度を、上記ガードレール込みで解釈した説明

## 3. 注目すべき特徴量
- global_importance上位の特徴量が、実際の値を踏まえてどう解釈できるか
  （explanation_reliabilityが低い場合は、この解釈が一般化しにくいことも明記）

## 4. フラグ・ルール判定との整合性
- 実損益・保有日数に基づくルール判定結果とモデル予測が一致しているか矛盾しているか

## 5. 確認すべきリスク・次に見るべき情報
- 次のシナリオ分岐（反落リスク、ブレイクアウト期待等）

## 6. 取引ルール（RULE.md）に照らした推奨アクション
- **【即座のアクション要否】**: なし / 半数決済 / 全決済 / 損切り実行
- **【判断の根拠】**: ルール（+10%利確/-5%損切り/10日ルール/反落兆候）およびモデル予測・特徴量との整合性
- **【次に警戒すべき分岐点】**: 次にどの価格・日数条件でルールが発動するか（例: 「あとX円下落で-5%損切り」「あとY日で10日ルール発動」等）

日本語で、簡潔かつ具体的に記述してください。
"""
