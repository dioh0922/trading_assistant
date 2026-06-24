"""
プロジェクト実行エントリーポイント

実行方法（プロジェクトルートで）:
    python main.py                                          # ATRモード（デフォルト, resource/sample.xlsxを使用）
    python main.py --code 6981                              # resource/6981.xlsx を使って検証
    python main.py --code 6981 --barrier-mode fixed_pct
    python main.py --barrier-mode atr
    python main.py --barrier-mode atr --tp-atr-mult 2.5 --sl-atr-mult 1.0 --holding-period 15
    python main.py --barrier-mode fixed_pct                 # 固定%モードをデフォルト値(+10%/-5%)で実行
    python main.py --barrier-mode fixed_pct --tp-pct 0.10 --sl-pct 0.05
    python main.py --tp-pct 0.10 --sl-pct 0.05               # --barrier-mode省略時もpct指定で自動的にfixed_pctになる
    python main.py --no-step6                               # ステップ6を無効化してステップ5までで止める
    python main.py --drawdown-prob-limit 0.60               # MLフィルタも有効化（閾値指定時）

コマンドライン引数:
    --code STR                       4桁の銘柄コードを指定すると resource/{code}.xlsx を読み込む。
                                      省略時は resource/sample.xlsx を使用する。
                                      対象ファイルが存在しない場合はエラーにせず処理を終了する。
    --barrier-mode {atr,fixed_pct}  バリアモードを明示指定（省略時はtp-pct/sl-pct指定の有無で自動判定）
    --tp-pct FLOAT                  [固定%モード] 利確の割合（例: 0.10 = +10%）
    --sl-pct FLOAT                  [固定%モード] 損切りの割合（例: 0.05 = -5%）
    --tp-atr-mult FLOAT              [ATRモード] 利確のATR倍率（デフォルト2.0）
    --sl-atr-mult FLOAT              [ATRモード] 損切りのATR倍率（デフォルト1.5）
    --holding-period INT             最大保有営業日数（省略時: ATR=10日 / 固定%=45日 を自動設定）
    --drawdown-threshold FLOAT       見送りシグナル用ドローダウン閾値（デフォルト0.03 = 3%）
    --step6 / --no-step6             ステップ6有効/無効（デフォルト: 有効）
    --drawdown-prob-limit FLOAT      [ステップ6] ドローダウンMLフィルタの確率閾値。
                                      1.0(デフォルト)=ML無効（ATRルールのみ）。
                                      0.60ー0.70返すとMLフィルタを併用。

ディレクトリ構成:
    project/
    ├── main.py                                ← これを実行
    ├── src/
    │   ├── step1_feature_engineering.py       ← マルチタイムフレーム統合
    │   ├── step2_domain_features.py           ← ドメイン知識の注入
    │   ├── step3_labeling.py                  ← トリプルバリア法のラベリング
    │   ├── step4_model.py                     ← モデル構築とバリデーション
    │   ├── step5_assist_signal.py             ← 3段階アシストシグナル
    │   └── visualize.py                       ← 各ステップの可視化
    └── resource/
        ├── sample.xlsx                        ← デフォルト入力Excel
        ├── XXXX.xlsx                          ← --code XXXX 指定時に読み込む銘柄別Excel
        ├── sample/                            ← --code未指定時の結果出力先
        │   ├── step1_dataset.csv / step1_chart.png
        │   ├── step2_dataset.csv / step2_chart.png
        │   ├── step3_dataset.csv / step3_chart.png
        │   ├── step4_fold_metrics.csv / step4_feature_importance.csv
        │   ├── step4_oos_predictions.csv / step4_model.pkl / step4_chart.png
        │   └── step5_dataset.csv / step5_chart.png
        └── XXXX/                              ← --code XXXX 指定時の結果出力先（同じファイル構成）
"""

import argparse
import sys

from src.step1_feature_engineering import build_step1_dataset, RESOURCE_DIR
from src.step2_domain_features import build_step2_dataset
from src.step3_labeling import build_step3_dataset
from src.step4_model import build_step4_results, save_step4_results
from src.step5_assist_signal import build_step5_dataset
from src.step6_filter import train_drawdown_model, apply_final_filter, evaluate_final_performance
from src.visualize import plot_step1, plot_step2, plot_step3, plot_step4, plot_step5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIトレードシステム パイプライン実行",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--code", type=str, default=None,
        help="4桁の銘柄コードを指定すると resource/{code}.xlsx を読み込んで検証する"
             "（例: --code 6981）。省略時は resource/sample.xlsx を使用する。",
    )
    parser.add_argument(
        "--barrier-mode", choices=["atr", "fixed_pct"], default=None,
        help="ステップ3のバリアモード。省略時は --tp-pct/--sl-pct の指定有無で自動判定する。",
    )
    parser.add_argument("--tp-pct", type=float, default=None,
                       help="[固定%%モード] 利確の割合（例: 0.10 で+10%%。--barrier-mode fixed_pct単体指定時のデフォルト: 0.10）")
    parser.add_argument("--sl-pct", type=float, default=None,
                       help="[固定%%モード] 損切りの割合（例: 0.05 で-5%%。--barrier-mode fixed_pct単体指定時のデフォルト: 0.05）")
    parser.add_argument("--tp-atr-mult", type=float, default=2.0,
                       help="[ATRモード] 利確バリアのATR倍率")
    parser.add_argument("--sl-atr-mult", type=float, default=1.5,
                       help="[ATRモード] 損切りバリアのATR倍率")
    parser.add_argument("--holding-period", type=int, default=None,
                       help="最大保有営業日数。省略時はモードに応じて自動設定（ATR=10日 / 固定%%=45日）")
    parser.add_argument("--drawdown-threshold", type=float, default=0.03,
                       help="見送りシグナル用のフォワードドローダウン閾値")
    parser.add_argument("--step6", action=argparse.BooleanOptionalAction, default=True,
                         help="ステップ6の最終意思決定フィルタを有効にする。--no-step6で無効化（デフォルト: 有効）")
    parser.add_argument("--drawdown-prob-limit", type=float, default=1.0,
                         help="[ステップ6] ドローダウンMLフィルタの確率閾値。"
                              "1.0(デフォルト)=ML無効（ATRルールのみ）。"
                              "0.60ー0.70を指定するとMLフィルタも併用")

    args = parser.parse_args()

    # --- 銘柄コードのバリデーション（4桁の数字のみ許可） ---
    if args.code is not None and not (args.code.isdigit() and len(args.code) == 4):
        parser.error(f"--code は4桁の数字で指定してください（例: 6981）。指定値: {args.code!r}")

    # --- 引数の整合性チェック ---
    pct_specified = (args.tp_pct is not None) or (args.sl_pct is not None)
    if pct_specified and (args.tp_pct is None or args.sl_pct is None):
        parser.error("固定%モードを使う場合は --tp-pct と --sl-pct を両方指定してください。")

    if args.barrier_mode == "atr" and pct_specified:
        parser.error(
            "--barrier-mode atr が指定されていますが --tp-pct/--sl-pct も指定されています。"
            "ATRモードでは --tp-atr-mult/--sl-atr-mult を使ってください。"
        )

    # --barrier-mode未指定時は pct指定の有無から自動判定（build_step3_dataset側のロジックと同じ）
    if args.barrier_mode is None:
        args.barrier_mode = "fixed_pct" if pct_specified else "atr"

    # --barrier-mode fixed_pct が単体指定（--tp-pct/--sl-pct省略）の場合は
    # デフォルトの「+10%利確 / -5%損切り」を自動補完する
    if args.barrier_mode == "fixed_pct" and not pct_specified:
        args.tp_pct = 0.10
        args.sl_pct = 0.05

    return args


def main() -> None:
    args = parse_args()

    if args.code is not None:
        code_label = args.code
        input_path = RESOURCE_DIR / f"{args.code}.xlsx"
    else:
        code_label = "sample"
        input_path = RESOURCE_DIR / "sample.xlsx"

    if not input_path.exists():
        print(f"入力ファイルが見つかりません: {input_path}")
        print("処理を終了します。")
        sys.exit(0)

    # --- 結果出力先: resource/{code_label}/{barrier_mode} （無ければ作成） ---
    output_dir = RESOURCE_DIR / code_label / args.barrier_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"結果出力先: {output_dir}\n")

    step1_output_path = output_dir / "step1_dataset.csv"
    step2_output_path = output_dir / "step2_dataset.csv"
    step3_output_path = output_dir / "step3_dataset.csv"
    step5_output_path = output_dir / "step5_dataset.csv"

    # --- ステップ1: マルチタイムフレーム統合 ---
    print("=" * 60)
    print("ステップ1: データ収集とマルチタイムフレームの統合")
    print("=" * 60)
    step1_df = build_step1_dataset(str(input_path))
    step1_df.to_csv(step1_output_path)
    print(f"\nCSV保存完了: {step1_output_path}")
    plot_step1(step1_df, output_dir / "step1_chart.png")

    # --- ステップ2: ドメイン知識の注入 ---
    print("\n" + "=" * 60)
    print("ステップ2: 特徴量エンジニアリング（ドメイン知識の注入）")
    print("=" * 60)
    step2_df = build_step2_dataset(step1_df)
    step2_df.to_csv(step2_output_path)
    print(f"\nCSV保存完了: {step2_output_path}")
    plot_step2(step2_df, output_dir / "step2_chart.png")

    # --- ステップ3: ターゲット（ラベル）の設計 ---
    print("\n" + "=" * 60)
    print(f"ステップ3: ターゲット（ラベル）の設計 [バリアモード: {args.barrier_mode}]")
    print("=" * 60)
    if args.barrier_mode == "fixed_pct":
        step3_df = build_step3_dataset(
            step2_df,
            holding_period=args.holding_period,
            tp_pct=args.tp_pct,
            sl_pct=args.sl_pct,
            drawdown_threshold=args.drawdown_threshold,
        )
    else:
        step3_df = build_step3_dataset(
            step2_df,
            holding_period=args.holding_period,
            tp_atr_mult=args.tp_atr_mult,
            sl_atr_mult=args.sl_atr_mult,
            drawdown_threshold=args.drawdown_threshold,
        )
    step3_df.to_csv(step3_output_path)
    print(f"\nCSV保存完了: {step3_output_path}")
    plot_step3(step3_df, output_dir / "step3_chart.png")

    # --- ステップ4: モデル構築とバリデーション ---
    print("\n" + "=" * 60)
    print("ステップ4: モデル構築とバリデーション")
    print("=" * 60)
    step4_results = build_step4_results(step3_df)
    save_step4_results(
        step4_results,
        fold_metrics_path=output_dir / "step4_fold_metrics.csv",
        feature_importance_path=output_dir / "step4_feature_importance.csv",
        oos_predictions_path=output_dir / "step4_oos_predictions.csv",
        model_path=output_dir / "step4_model.pkl",
    )
    plot_step4(
        step4_results["fold_metrics"],
        step4_results["feature_importance"],
        step4_results["oos_predictions"],
        step3_df,
        output_dir / "step4_chart.png",
    )

    # --- ステップ5: アシストロジックの実装（3段階シグナル） ---
    print("\n" + "=" * 60)
    print("ステップ5: アシストロジックの実装（3段階シグナル）")
    print("=" * 60)
    step5_df = build_step5_dataset(step3_df)
    step5_df.to_csv(step5_output_path)
    print(f"\nCSV保存完了: {step5_output_path}")
    plot_step5(step5_df, output_dir / "step5_chart.png")

    # --- 確認用: 先頭5行を表示 ---
    print("\n--- ステップ5 先頭5行 ---")
    print(step5_df[["close", "assist_signal"]].head())

    # --- ステップ6: 最終意思決定フィルタ ---
    if args.step6:
        # drawdown-prob-limit が 1.0 のときは ML フィルタ実質無効（ATR ルールのみ）
        ml_enabled = args.drawdown_prob_limit < 1.0
        mode_label = f"MLフィルタ有効 (閾値: {args.drawdown_prob_limit:.0%})" if ml_enabled else "ATRルールのみ (ML無効)"
        print("\n" + "=" * 60)
        print(f"ステップ6: 最終意思決定フィルタ [{mode_label}]")
        print("=" * 60)

        from src.step4_model import select_feature_columns
        feature_cols = select_feature_columns(step3_df)

        # 1. ドローダウン予測モデルの学習と予測確率の取得
        print("ドローダウン予測モデルを構築中...")
        dd_results = train_drawdown_model(step3_df, feature_cols)
        dd_pred_proba = dd_results["oos_predictions"]["dd_proba"]

        # 2. 最終フィルタの適用
        step6_df = apply_final_filter(
            step5_df,
            dd_pred_proba=dd_pred_proba,
            dd_threshold=args.drawdown_prob_limit
        )

        # 3. パフォーマンス評価と出力
        evaluate_final_performance(step6_df)

        # 4. CSVの保存
        step6_output_path = output_dir / "step6_dataset.csv"
        step6_df.to_csv(step6_output_path)
        print(f"\nCSV保存完了: {step6_output_path}")

        # ドローダウン予測モデルを保存
        dd_model_path = output_dir / "step6_dd_model.pkl"
        import joblib
        joblib.dump(dd_results["final_model"], dd_model_path)
        print(f"ドローダウン予測モデル保存完了: {dd_model_path}")


if __name__ == "__main__":
    main()
