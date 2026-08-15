"""
llm_analyze.py
────────────────
analyze_ticker.py（モードB）が出力した銘柄詳細JSONに、
- explanation_reliability（predicted_labelごとの説明信頼度メモ。銘柄固有実績があればそちらを優先）
を付加した上で、Gemini(LLM)に渡して解釈させるスクリプト。

llm_integration_policy.md（プロンプト設計）と
shap_reliability_llm_integration_ideas.md（explanation_reliabilityの設計）を実装したもの。

前提:
- 環境変数 GEMINI_API_KEY にAPIキーが設定されていること
- pip install google-genai 済みであること

使い方:
    # 単一銘柄：実際にLLMへ問い合わせる
    python llm_analyze.py single --ticker 6098 \
        --combined-records reports/combined_records.csv \
        --output reports/6098_llm_analysis.md

    # 単一銘柄：LLMを呼ばず、組み立てたプロンプトだけを確認する（APIキー不要）
    python llm_analyze.py single --ticker 6098 --dry-run

    # 一括：check_positions.py（モードA）が出力した日次レポートから
    # 「要注意銘柄（アラート発生）」のコードを自動抽出し、まとめてLLM分析にかける
    python llm_analyze.py batch \
        --daily-check-report daily_check_20260809_113205.md \
        --json-dir reports/json \
        --combined-records reports/combined_records.csv \
        --output-dir reports/llm_batch

    # 一括：銘柄コードを直接指定する場合（レポートのパース不要）
    python llm_analyze.py batch --tickers 3405,4452,6857 --output-dir reports/llm_batch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("llm_analyze")

DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_JSON_DIR = Path("reports/json")

# ファイル名パターン: {ticker}_detail_{YYYYMMDD}_{HHMMSS}.json
DETAIL_FILENAME_RE = re.compile(r"^(?P<ticker>.+)_detail_(?P<timestamp>\d{8}_\d{6})\.json$")


def find_latest_detail_json(json_dir: Path, ticker: str) -> Path:
    """
    json_dir配下から f"{ticker}_detail_*.json" または f"{ticker}_detail.json" を探す。
    タイムスタンプ付きがある場合は最新のものを、タイムスタンプなしがある場合はそれを採用。
    """
    candidates = sorted(json_dir.glob(f"{ticker}_detail_*.json"))
    plain_detail = json_dir / f"{ticker}_detail.json"
    if plain_detail.exists():
        candidates.append(plain_detail)

    if not candidates:
        raise FileNotFoundError(
            f"{json_dir} に '{ticker}_detail_*.json' または '{ticker}_detail.json' が見つかりません。"
        )

    parsed = []
    for path in candidates:
        if path.name == f"{ticker}_detail.json":
            parsed.append(("00000000_000000", path))
            continue
        m = DETAIL_FILENAME_RE.match(path.name)
        if m and m.group("ticker") == ticker:
            parsed.append((m.group("timestamp"), path))

    if not parsed:
        if plain_detail.exists():
            return plain_detail
        raise FileNotFoundError(
            f"{json_dir} に '{ticker}_detail_*.json' 形式（例: {ticker}_detail_20260728_094338.json）"
            f"に一致するファイルが見つかりません。見つかった候補: {[p.name for p in candidates]}"
        )

    parsed.sort(key=lambda x: x[0])  # YYYYMMDD_HHMMSS形式は文字列比較で時系列順になる
    latest_timestamp, latest_path = parsed[-1]
    log.info("最新のJSONを選択: %s（タイムスタンプ: %s, 候補%d件中）",
              latest_path.name, latest_timestamp, len(parsed))
    return latest_path



def extract_flagged_tickers(report_path: Path) -> list[str]:
    """
    check_positions.py（モードA）が出力した日次レポート（daily_check_*.md）から、
    「## 要注意銘柄」セクションのMarkdown表を読み取り、コード列（先頭列）を抽出する。
    """
    # 指定パスが存在しない場合、同ディレクトリおよび reports/, train/reports/ から最新を検索
    actual_path = report_path
    if not actual_path.exists():
        candidates = []
        for search_dir in [report_path.parent, Path("reports"), Path("train/reports")]:
            if search_dir.exists():
                candidates.extend(search_dir.glob("daily_check*.md"))
        candidates = list(set(candidates))
        if not candidates:
            raise FileNotFoundError(
                f"日次レポートが見つかりません: {report_path} "
                f"(reports/ および train/reports/ も検索済み)"
            )
        candidates.sort(key=lambda p: p.name)
        actual_path = candidates[-1]
        log.info("指定レポートが存在しないため最新を自動選択: %s", actual_path)

    text = actual_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    section_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("## 要注意銘柄"):
            section_start = i
            break
    if section_start is None:
        raise ValueError(
            f"{report_path} に '## 要注意銘柄' セクションが見つかりません。"
            f" レポートの形式を確認するか、--tickers で直接指定してください。"
        )

    section_end = len(lines)
    for i in range(section_start + 1, len(lines)):
        if lines[i].strip().startswith("## "):
            section_end = i
            break

    tickers = []
    for line in lines[section_start:section_end]:
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells:
            continue
        first_cell = cells[0]
        # ヘッダー行（「コード」）と区切り行（「---」等）を除外する
        if first_cell in ("コード",) or set(first_cell) <= {"-", ":"}:
            continue
        if first_cell:
            tickers.append(first_cell)

    if not tickers:
        log.warning("'## 要注意銘柄' セクションは見つかりましたが、銘柄コードを1件も抽出できませんでした。")
    else:
        log.info("要注意銘柄を%d件抽出しました: %s", len(tickers), tickers)
    return tickers

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
def call_llm_anthropic(system_prompt: str, user_content: str, model: str = "claude-sonnet-4-6") -> str:
    """旧Anthropic (Claude) 実装。退避用。"""
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


def call_llm_gemini(system_prompt: str, user_content: str, model: str) -> str:
    """Gemini API (google-genai SDK) を使用したLLM呼び出し"""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "環境変数 GEMINI_API_KEY が設定されていません。"
            "設定するか、--dry-run オプションでプロンプトの確認のみ行ってください。"
        )

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=4000,
    )
    response = client.models.generate_content(
        model=model,
        contents=user_content,
        config=config,
    )

    if response.candidates:
        finish_reason = response.candidates[0].finish_reason
        if finish_reason and str(finish_reason) not in ("FinishReason.STOP", "STOP"):
            log.warning("Geminiの応答が完了理由 '%s' により途切れた可能性があります。", finish_reason)

    return response.text or ""


def call_llm(system_prompt: str, user_content: str, model: str) -> str:
    return call_llm_gemini(system_prompt, user_content, model)


# ──────────────────────────────────────────────────────────────
# メイン処理
# ──────────────────────────────────────────────────────────────
def analyze_one_ticker(
    ticker: str,
    json_dir: Path,
    combined_records: pd.DataFrame | None,
    model: str,
    dry_run: bool,
) -> dict:
    """
    1銘柄分の「最新JSON取得 → explanation_reliability算出 → プロンプト構築 → (必要なら)LLM呼び出し」
    をまとめて行う共通処理。単発コマンド（single）・一括コマンド（batch）の両方から使う。

    戻り値: {ticker, input_json, reliability, user_content, result_text(dry_runならNone), error(あれば)}
    """
    try:
        input_json = find_latest_detail_json(json_dir, ticker)
        with open(input_json, "r", encoding="utf-8") as f:
            detail = json.load(f)

        actual_ticker = detail.get("ticker", ticker)
        predicted_label = detail["prediction"]["predicted_label"]

        reliability = compute_explanation_reliability(actual_ticker, predicted_label, combined_records)
        log.info("[%s] explanation_reliability: source=%s, mean_spearman_corr=%s, n=%s",
                  ticker, reliability["source"], reliability["mean_spearman_corr"], reliability["n"])

        user_content = build_user_content(detail, reliability)

        result_text = None
        if not dry_run:
            log.info("[%s] Gemini(%s)に問い合わせています...", ticker, model)
            result_text = call_llm(SYSTEM_PROMPT, user_content, model)

        return {
            "ticker": ticker,
            "input_json": input_json,
            "reliability": reliability,
            "user_content": user_content,
            "result_text": result_text,
            "error": None,
        }
    except Exception as e:
        log.warning("[%s] 処理に失敗しました: %s", ticker, e)
        return {
            "ticker": ticker,
            "input_json": None,
            "reliability": None,
            "user_content": None,
            "result_text": None,
            "error": str(e),
        }


def run_single(
    ticker: str,
    json_dir: Path,
    combined_records_path: Path | None,
    model: str,
    output: Path | None,
    dry_run: bool,
) -> None:
    combined_records = load_combined_records(combined_records_path)
    result = analyze_one_ticker(ticker, json_dir, combined_records, model, dry_run)

    if result["error"] is not None:
        raise RuntimeError(result["error"])

    if dry_run:
        print("=" * 60)
        print("【system prompt】")
        print("=" * 60)
        print(SYSTEM_PROMPT)
        print("=" * 60)
        print("【user content（LLMに渡すJSON）】")
        print("=" * 60)
        print(result["user_content"])
        print("\n--dry-run のため、実際のLLM呼び出しは行っていません。")
        return

    print("=" * 60)
    print(f"【{ticker} の分析結果】")
    print("=" * 60)
    print(result["result_text"])

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            f"# {ticker} 分析結果\n\n"
            f"- 入力: {result['input_json']}\n"
            f"- モデル: {model}\n"
            f"- explanation_reliability: {result['reliability']}\n\n"
            f"---\n\n{result['result_text']}\n",
            encoding="utf-8",
        )
        log.info("output written to %s", output)


def run_batch(
    tickers: list[str],
    json_dir: Path,
    combined_records_path: Path | None,
    model: str,
    output_dir: Path,
    dry_run: bool,
) -> None:
    combined_records = load_combined_records(combined_records_path)

    results = []
    for ticker in tickers:
        result = analyze_one_ticker(ticker, json_dir, combined_records, model, dry_run)
        results.append(result)

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    summary_lines.append("# LLM一括分析 サマリー")
    summary_lines.append("")
    summary_lines.append(f"- 対象銘柄数: {len(tickers)}")
    summary_lines.append(f"- 成功: {sum(1 for r in results if r['error'] is None)}"
                          f" / 失敗: {sum(1 for r in results if r['error'] is not None)}")
    summary_lines.append("")

    for result in results:
        ticker = result["ticker"]
        if result["error"] is not None:
            summary_lines.append(f"## {ticker}: 処理失敗")
            summary_lines.append("")
            summary_lines.append(f"- エラー: {result['error']}")
            summary_lines.append("")
            continue

        if dry_run:
            individual_path = output_dir / f"{ticker}_prompt.md"
            individual_path.write_text(
                f"# {ticker} プロンプト（dry-run）\n\n"
                f"- 入力: {result['input_json']}\n"
                f"- explanation_reliability: {result['reliability']}\n\n"
                f"## system prompt\n\n```\n{SYSTEM_PROMPT}\n```\n\n"
                f"## user content\n\n```json\n{result['user_content']}\n```\n",
                encoding="utf-8",
            )
            summary_lines.append(f"## {ticker}: dry-run（プロンプトのみ生成）")
            summary_lines.append("")
            summary_lines.append(f"- 詳細: {individual_path.name}")
            summary_lines.append(f"- explanation_reliability: {result['reliability']['note']}")
            summary_lines.append("")
        else:
            individual_path = output_dir / f"{ticker}_llm_analysis.md"
            individual_path.write_text(
                f"# {ticker} 分析結果\n\n"
                f"- 入力: {result['input_json']}\n"
                f"- モデル: {model}\n"
                f"- explanation_reliability: {result['reliability']}\n\n"
                f"---\n\n{result['result_text']}\n",
                encoding="utf-8",
            )
            log.info("[%s] output written to %s", ticker, individual_path)

            summary_lines.append(f"## {ticker}")
            summary_lines.append("")
            summary_lines.append(f"- 詳細: {individual_path.name}")
            summary_lines.append("")
            summary_lines.append(result["result_text"])
            summary_lines.append("")

    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    log.info("=" * 60)
    log.info("一括分析完了。出力先: %s", output_dir)
    log.info("=" * 60)
    print("\n".join(summary_lines))


def load_combined_records(combined_records_path: Path | None) -> pd.DataFrame | None:
    if combined_records_path is None:
        return None
    if not combined_records_path.exists():
        log.warning("%s が見つからないため、母集団のフォールバック値を使用します。", combined_records_path)
        return None
    combined_records = pd.read_csv(combined_records_path)
    log.info("combined_recordsを読み込みました: %d件", len(combined_records))
    return combined_records


def main() -> None:
    parser = argparse.ArgumentParser(description="銘柄詳細JSONをLLMに渡して解釈させる")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single_parser = subparsers.add_parser("single", help="1銘柄をLLMに分析させる")
    single_parser.add_argument("--ticker", type=str, required=True,
                                help="銘柄コード（例: 6098）。指定した銘柄の最新タイムスタンプのJSONを自動選択する")
    single_parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR,
                                help=f"analyze_ticker.py の出力JSON群があるディレクトリ（デフォルト: {DEFAULT_JSON_DIR}）")
    single_parser.add_argument("--combined-records", type=Path, default=None,
                                help="aggregate_batch_stats.py が出力した combined_records.csv"
                                     "（省略時は固定のフォールバック値を使用）")
    single_parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    single_parser.add_argument("--output", type=Path, default=None, help="分析結果を保存するmdファイルパス")
    single_parser.add_argument("--dry-run", action="store_true",
                                help="LLMを呼ばず、組み立てたプロンプトを表示するだけ（APIキー不要）")

    batch_parser = subparsers.add_parser("batch", help="複数銘柄をまとめてLLMに分析させる")
    batch_parser.add_argument("--daily-check-report", type=Path, default=None,
                               help="check_positions.py が出力した日次レポート。"
                                    "'## 要注意銘柄' セクションから銘柄コードを自動抽出する")
    batch_parser.add_argument("--tickers", type=str, default=None,
                               help="カンマ区切りの銘柄コードを直接指定する場合"
                                    "（--daily-check-report と併用不可、どちらか一方を指定）")
    batch_parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    batch_parser.add_argument("--combined-records", type=Path, default=None)
    batch_parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    batch_parser.add_argument("--output-dir", type=Path, default=Path("reports/llm_batch"),
                               help="銘柄ごとの分析結果とsummary.mdの出力先ディレクトリ")
    batch_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.command == "single":
        run_single(
            ticker=args.ticker,
            json_dir=args.json_dir,
            combined_records_path=args.combined_records,
            model=args.model,
            output=args.output,
            dry_run=args.dry_run,
        )
    elif args.command == "batch":
        if args.daily_check_report is None and args.tickers is None:
            raise SystemExit("--daily-check-report か --tickers のどちらかを指定してください。")
        if args.daily_check_report is not None and args.tickers is not None:
            raise SystemExit("--daily-check-report と --tickers は同時に指定できません。")

        if args.tickers is not None:
            tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        else:
            tickers = extract_flagged_tickers(args.daily_check_report)

        if not tickers:
            raise SystemExit("対象銘柄が0件のため、処理を終了します。")

        run_batch(
            tickers=tickers,
            json_dir=args.json_dir,
            combined_records_path=args.combined_records,
            model=args.model,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()