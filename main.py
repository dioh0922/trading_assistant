"""
プロジェクト実行エントリーポイント

実行方法（プロジェクトルートで）:
    python main.py

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
        ├── sample.xlsx                        ← 入力Excel（ここに配置する）
        ├── step1_dataset.csv / step1_chart.png
        ├── step2_dataset.csv / step2_chart.png
        ├── step3_dataset.csv / step3_chart.png
        ├── step4_fold_metrics.csv / step4_feature_importance.csv
        ├── step4_oos_predictions.csv / step4_model.pkl / step4_chart.png
        └── step5_dataset.csv / step5_chart.png
"""

from src.step1_feature_engineering import build_step1_dataset, RESOURCE_DIR
from src.step2_domain_features import build_step2_dataset
from src.step3_labeling import build_step3_dataset
from src.step4_model import build_step4_results, save_step4_results
from src.step5_assist_signal import build_step5_dataset
from src.visualize import plot_step1, plot_step2, plot_step3, plot_step4, plot_step5


def main() -> None:
    input_path = RESOURCE_DIR / "sample.xlsx"
    step1_output_path = RESOURCE_DIR / "step1_dataset.csv"
    step2_output_path = RESOURCE_DIR / "step2_dataset.csv"
    step3_output_path = RESOURCE_DIR / "step3_dataset.csv"
    step5_output_path = RESOURCE_DIR / "step5_dataset.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"入力ファイルが見つかりません: {input_path}\n"
            f"resource/ に sample.xlsx を配置してください。"
        )

    # --- ステップ1: マルチタイムフレーム統合 ---
    print("=" * 60)
    print("ステップ1: データ収集とマルチタイムフレームの統合")
    print("=" * 60)
    step1_df = build_step1_dataset(str(input_path))
    step1_df.to_csv(step1_output_path)
    print(f"\nCSV保存完了: {step1_output_path}")
    plot_step1(step1_df, RESOURCE_DIR / "step1_chart.png")

    # --- ステップ2: ドメイン知識の注入 ---
    print("\n" + "=" * 60)
    print("ステップ2: 特徴量エンジニアリング（ドメイン知識の注入）")
    print("=" * 60)
    step2_df = build_step2_dataset(step1_df)
    step2_df.to_csv(step2_output_path)
    print(f"\nCSV保存完了: {step2_output_path}")
    plot_step2(step2_df, RESOURCE_DIR / "step2_chart.png")

    # --- ステップ3: ターゲット（ラベル）の設計 ---
    print("\n" + "=" * 60)
    print("ステップ3: ターゲット（ラベル）の設計")
    print("=" * 60)
    step3_df = build_step3_dataset(step2_df)
    step3_df.to_csv(step3_output_path)
    print(f"\nCSV保存完了: {step3_output_path}")
    plot_step3(step3_df, RESOURCE_DIR / "step3_chart.png")

    # --- ステップ4: モデル構築とバリデーション ---
    print("\n" + "=" * 60)
    print("ステップ4: モデル構築とバリデーション")
    print("=" * 60)
    step4_results = build_step4_results(step3_df)
    save_step4_results(step4_results)
    plot_step4(
        step4_results["fold_metrics"],
        step4_results["feature_importance"],
        step4_results["oos_predictions"],
        step3_df,
        RESOURCE_DIR / "step4_chart.png",
    )

    # --- ステップ5: アシストロジックの実装（3段階シグナル） ---
    print("\n" + "=" * 60)
    print("ステップ5: アシストロジックの実装（3段階シグナル）")
    print("=" * 60)
    step5_df = build_step5_dataset(step3_df)
    step5_df.to_csv(step5_output_path)
    print(f"\nCSV保存完了: {step5_output_path}")
    plot_step5(step5_df, RESOURCE_DIR / "step5_chart.png")

    # --- 確認用: 先頭5行を表示 ---
    print("\n--- ステップ5 先頭5行 ---")
    print(step5_df[["close", "assist_signal"]].head())


if __name__ == "__main__":
    main()
