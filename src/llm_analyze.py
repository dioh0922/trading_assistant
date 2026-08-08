"""
llm_analyze.py
────────────────
analyze_ticker.py（モードB）が出力した銘柄詳細JSONに、
- explanation_reliability（predicted_labelごとの説明信頼度メモ。銘柄固有実績があればそちらを優先）
を付加した上で、Claude(LLM)に渡して解釈させるスクリプト。

llm_integration_policy.md（プロンプト設計）と
shap_reliability_llm_integration_ideas.md（explanation_reliabilityの設計）を実装したもの。

前提:
- 環境変数 ANTHROPIC_API_KEY にAPIキーが設定されていること
- pip install anthropic 済みであること

使い方:
    # 実際にLLMへ問い合わせる
    python llm_analyze.py \
        --input-json 6098_detail_20260802_105439.json \
        --combined-records reports/combined_records.csv \
        --output reports/6098_llm_analysis.md

    # LLMを呼ばず、組み立てたプロンプトだけを確認する（APIキー不要）
    python llm_analyze.py --input-json 6098_detail_20260802_105439.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("llm_analyze")

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_JSON_DIR = Path("reports/json")

# ファイル名パターン: {ticker}_detail_{YYYYMMDD}_{HHMMSS}.json
DETAIL_FILENAME_RE = re.compile(r"^(?P<ticker>.+)_detail_(?P<timestamp>\d{8}_\d{6})\.json$")


def find_latest_detail_json(json_dir: Path, ticker: str) -> Path:
    """
    json_dir配下から f"{ticker}_detail_*.json" にマッチするファイルを探し、
    ファイル名に埋め込まれたタイムスタンプ（YYYYMMDD_HHMMSS）が最も新しいものを返す。
    タイムスタンプの解析に失敗したファイルは、安全側に倒して除外する。
    """
    candidates = sorted(json_dir.glob(f"{ticker}_detail_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"{json_dir} に '{ticker}_detail_*.json' に一致するファイルが見つかりません。"
        )

    parsed = []
    for path in candidates:
        m = DETAIL_FILENAME_RE.match(path.name)
        if m and m.group("ticker") == ticker:
            parsed.append((m.group("timestamp"), path))

    if not parsed:
        raise FileNotFoundError(
            f"{json_dir} に '{ticker}_detail_*.json' 形式（例: {ticker}_detail_20260728_094338.json）"
            f"に一致するファイルが見つかりません。見つかった候補: {[p.name for p in candidates]}"
        )

    parsed.sort(key=lambda x: x[0])  # YYYYMMDD_HHMMSS形式は文字列比較で時系列順になる
    latest_timestamp, latest_path = parsed[-1]
    log.info("最新のJSONを選択: %s（タイムスタンプ: %s, 候補%d件中）",
              latest_path.name, latest_timestamp, len(parsed))
    return latest_path

# aggregate_batch_stats.py による検証結果（2026-08-08時点、57件・14銘柄）のフォールバック値。
# combined_records.csv が渡された場合は、そちらから動的に再計算する（値はこの定数を上書きする）。
POPULATION_RELIABILITY = {
    "upper": {"mean_spearman_corr": 0.717, "n": 9},
    "lower": {"mean_spearman_corr": 0.587, "n": 47},
    "timeout": {"mean_spearman_corr": None, "n": 1},  # 判断できるほどの件数がない
}

TICKER_SPECIFIC_MIN_N = 3  # 銘柄固有の実績を使うために必要な最低観測件数


# ──────────────────────────────────────────────────────────────
# 1. explanation_reliability の算出
#    （shap_reliability_llm_integration_ideas.md 第1節・第3節の実装）
# ──────────────────────────────────────────────────────────────
def compute_population_reliability(combined_records: pd.DataFrame | None) -> dict:
    """combined_records.csvがあれば、そこからlabel別の平均一致度を動的に計算する。
    無ければハードコードされたフォールバック値を使う。"""
    if combined_records is None:
        return POPULATION_RELIABILITY

    result = {}
    for label in ["upper", "lower", "timeout"]:
        g = combined_records[combined_records["predicted_label"] == label]
        if len(g) > 0:
            result[label] = {"mean_spearman_corr": float(g["spearman_corr"].mean()), "n": len(g)}
        else:
            result[label] = POPULATION_RELIABILITY.get(label, {"mean_spearman_corr": None, "n": 0})
    return result


def compute_explanation_reliability(
    ticker: str, predicted_label: str, combined_records: pd.DataFrame | None
) -> dict:
    """
    predicted_label（と可能なら銘柄固有の実績）から、SHAP説明がどの程度
    一般的なパターンで説明できそうかのメモを作る。

    優先順位:
      1. 対象銘柄・対象labelの過去実績が TICKER_SPECIFIC_MIN_N 件以上あれば、それを使う
      2. なければ label 別の母集団平均（combined_recordsがあれば動的計算、無ければ固定値）にフォールバック
    """
    if combined_records is not None:
        ticker_records = combined_records[
            (combined_records["ticker"].astype(str) == str(ticker))
            & (combined_records["predicted_label"] == predicted_label)
        ]
        if len(ticker_records) >= TICKER_SPECIFIC_MIN_N:
            mean_corr = float(ticker_records["spearman_corr"].mean())
            return {
                "source": "ticker_specific",
                "predicted_label": predicted_label,
                "mean_spearman_corr": mean_corr,
                "n": len(ticker_records),
                "note": (
                    f"この銘柄の{predicted_label}予測では、過去{len(ticker_records)}件の実績で"
                    f"SHAP説明とモデル全体の重要度傾向の一致度が平均{mean_corr:.2f}でした。"
                    f"（1.0に近いほど一般的なパターンに沿った予測、低いほど銘柄固有の要因が"
                    f"強く効いている可能性があります）"
                ),
            }

    population = compute_population_reliability(combined_records)
    pop_info = population.get(predicted_label, {"mean_spearman_corr": None, "n": 0})
    mean_corr = pop_info.get("mean_spearman_corr")
    n = pop_info.get("n", 0)

    if mean_corr is None:
        note = (
            f"predicted_label='{predicted_label}'については、まだ検証データが少なく"
            f"（n={n}）、説明の一致度傾向を判断できません。"
        )
    else:
        note = (
            f"この銘柄自身の実績はまだ少ないため、{predicted_label}予測全体の傾向"
            f"（過去検証n={n}件、平均一致度{mean_corr:.2f}）を参考値として使用しています。"
        )

    return {
        "source": "population_level",
        "predicted_label": predicted_label,
        "mean_spearman_corr": mean_corr,
        "n": n,
        "note": note,
    }


# ──────────────────────────────────────────────────────────────
# 2. プロンプト構成（llm_integration_policy.md 2節の実装）
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
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
   「モデルはこう見ている」という枠組みで説明してください。

【出力してほしい構成】
1. 現状サマリー：取得価格からの含み損益、直近の値動きの要約
2. モデルの見立て：予測ラベル・確信度を、上記ガードレール込みで解釈した説明
3. 注目すべき特徴量：global_importance上位の特徴量が、実際の値を踏まえてどう解釈できるか
   （explanation_reliabilityが低い場合は、この解釈が一般化しにくいことも明記）
4. フラグとの整合性：価格ベースのフラグとモデル予測が一致しているか矛盾しているか（情報がある場合）
5. 確認すべきリスク・次に見るべき情報

日本語で、簡潔かつ具体的に記述してください。
"""


def build_user_content(detail: dict, reliability: dict) -> str:
    payload = dict(detail)
    payload["explanation_reliability"] = reliability
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────
# 3. LLM呼び出し
# ──────────────────────────────────────────────────────────────
def call_llm(system_prompt: str, user_content: str, model: str) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "環境変数 ANTHROPIC_API_KEY が設定されていません。"
            "設定するか、--dry-run オプションでプロンプトの確認のみ行ってください。"
        )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    text_parts = [block.text for block in response.content if block.type == "text"]
    return "\n".join(text_parts)


# ──────────────────────────────────────────────────────────────
# メイン処理
# ──────────────────────────────────────────────────────────────
def run(
    ticker: str,
    json_dir: Path,
    combined_records_path: Path | None,
    model: str,
    output: Path | None,
    dry_run: bool,
) -> None:
    input_json = find_latest_detail_json(json_dir, ticker)

    with open(input_json, "r", encoding="utf-8") as f:
        detail = json.load(f)

    combined_records = None
    if combined_records_path is not None:
        if not combined_records_path.exists():
            log.warning("%s が見つからないため、母集団のフォールバック値を使用します。", combined_records_path)
        else:
            combined_records = pd.read_csv(combined_records_path)
            log.info("combined_recordsを読み込みました: %d件", len(combined_records))

    ticker = detail.get("ticker", "?")
    predicted_label = detail["prediction"]["predicted_label"]

    reliability = compute_explanation_reliability(ticker, predicted_label, combined_records)
    log.info("explanation_reliability: source=%s, mean_spearman_corr=%s, n=%s",
              reliability["source"], reliability["mean_spearman_corr"], reliability["n"])

    user_content = build_user_content(detail, reliability)

    if dry_run:
        print("=" * 60)
        print("【system prompt】")
        print("=" * 60)
        print(SYSTEM_PROMPT)
        print("=" * 60)
        print("【user content（LLMに渡すJSON）】")
        print("=" * 60)
        print(user_content)
        print("\n--dry-run のため、実際のLLM呼び出しは行っていません。")
        return

    log.info("Claude(%s)に問い合わせています...", model)
    result_text = call_llm(SYSTEM_PROMPT, user_content, model)

    print("=" * 60)
    print(f"【{ticker} の分析結果】")
    print("=" * 60)
    print(result_text)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            f"# {ticker} 分析結果\n\n"
            f"- 入力: {input_json}\n"
            f"- モデル: {model}\n"
            f"- explanation_reliability: {reliability}\n\n"
            f"---\n\n{result_text}\n",
            encoding="utf-8",
        )
        log.info("output written to %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(description="銘柄詳細JSONをLLMに渡して解釈させる")
    parser.add_argument("--ticker", type=str, required=True,
                         help="銘柄コード（例: 6098）。指定した銘柄の最新タイムスタンプのJSONを自動選択する")
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR,
                         help=f"analyze_ticker.py の出力JSON群があるディレクトリ（デフォルト: {DEFAULT_JSON_DIR}）")
    parser.add_argument("--combined-records", type=Path, default=None,
                         help="aggregate_batch_stats.py が出力した combined_records.csv"
                              "（省略時は固定のフォールバック値を使用）")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=None, help="分析結果を保存するmdファイルパス")
    parser.add_argument("--dry-run", action="store_true",
                         help="LLMを呼ばず、組み立てたプロンプトを表示するだけ（APIキー不要）")
    args = parser.parse_args()

    run(
        ticker=args.ticker,
        json_dir=args.json_dir,
        combined_records_path=args.combined_records,
        model=args.model,
        output=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()