"""
AIトレードシステム 可視化モジュール

各ステップのCSV（step1_dataset.csv, step2_dataset.csv）を読み込み、
価格チャートと各種指標を多段パネルでまとめて可視化する。

出力:
    resource/step1_chart.png
    resource/step2_chart.png

実行エントリーポイントは project直下の main.py。
単独でも実行可能:
    python src/visualize.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # ファイル出力専用バックエンド（画面表示不要）
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCE_DIR = BASE_DIR / "resource"


# =============================================================================
# 日本語フォントの自動検出
#   OSによって入っているフォントが異なるため、候補リストの中から
#   実際にインストールされているものを自動選択する。
#   見つからない場合は警告を出し、デフォルトフォントのまま続行する
#   （文字化けはするが処理自体は止めない）。
# =============================================================================

_JP_FONT_CANDIDATES = [
    "Yu Gothic",            # Windows 10/11 標準
    "Meiryo",                # Windows 標準（やや古い環境）
    "MS Gothic",              # Windows 古い環境
    "Hiragino Sans",         # macOS 標準
    "Hiragino Kaku Gothic Pro",  # macOS 標準（旧）
    "Noto Sans CJK JP",      # Linux でよく入っている
    "Noto Sans JP",
    "IPAexGothic",            # Linux (IPAフォント)
    "TakaoGothic",            # Linux (Takaoフォント)
]


def _set_japanese_font() -> str | None:
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _JP_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            return name

    print(
        "[警告] 日本語フォントが見つかりませんでした。グラフ内の日本語ラベルが"
        "文字化け（豆腐文字）する可能性があります。\n"
        "  Windows: 通常 'Yu Gothic' か 'Meiryo' がプリインストールされています。\n"
        "  見つからない場合は Noto Sans JP 等をインストールしてください: "
        "https://fonts.google.com/noto/specimen/Noto+Sans+JP\n"
        "  インストール後、フォントキャッシュの再構築が必要な場合があります:\n"
        "    python -c \"import matplotlib.font_manager as fm; fm._load_fontmanager(try_read_cache=False)\""
    )
    return None


_USED_FONT = _set_japanese_font()
if _USED_FONT:
    print(f"[フォント] 日本語フォント '{_USED_FONT}' を使用します。")
plt.rcParams["axes.unicode_minus"] = False

COLOR_PRICE   = "#1f77b4"
COLOR_MA25    = "#ff7f0e"
COLOR_MA75    = "#2ca02c"
COLOR_UP      = "#d62728"   # 上昇・買われすぎ系（日本式に赤=上昇）
COLOR_DOWN    = "#1f77b4"   # 下降・売られすぎ系（青=下降）
COLOR_NEUTRAL = "#7f7f7f"


def _load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return df


# =============================================================================
# ステップ1の可視化
#   1. 価格 + MA25/MA75 + 週足トレンド背景
#   2. 出来高
#   3. RSI14
#   4. ボリンジャー%B
#   5. ATR14
#   6. モメンタム加速度
# =============================================================================

def plot_step1(df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(
        6, 1, figsize=(14, 18), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1, 1, 1, 1]},
    )
    ax_price, ax_vol, ax_rsi, ax_bb, ax_atr, ax_mom = axes

    # --- ① 価格 + MA + 週足トレンド背景 ---
    ax_price.plot(df.index, df["close"], color=COLOR_PRICE, linewidth=1.2, label="終値")
    ax_price.plot(df.index, df["ma25"], color=COLOR_MA25, linewidth=1, label="MA25")
    ax_price.plot(df.index, df["ma75"], color=COLOR_MA75, linewidth=1, label="MA75")

    # 週足トレンド（1=上昇 / -1=下降）を背景色で表示
    trend = df["weekly_trend"]
    up_mask = trend == 1
    down_mask = trend == -1
    ax_price.fill_between(df.index, ax_price.get_ylim()[0], ax_price.get_ylim()[1],
                          where=up_mask, color=COLOR_UP, alpha=0.05, step="mid")
    ax_price.fill_between(df.index, ax_price.get_ylim()[0], ax_price.get_ylim()[1],
                          where=down_mask, color=COLOR_DOWN, alpha=0.05, step="mid")

    ax_price.set_ylabel("価格")
    ax_price.set_title("ステップ1: 価格・移動平均線・週足トレンド（背景: 赤=上昇週足 / 青=下降週足）")
    ax_price.legend(loc="upper left", fontsize=9)
    ax_price.grid(alpha=0.3)

    # --- ② 出来高 ---
    ax_vol.bar(df.index, df["volume"], color=COLOR_NEUTRAL, width=1.0, alpha=0.6)
    ax_vol.set_ylabel("出来高")
    ax_vol.grid(alpha=0.3)

    # --- ③ RSI14 ---
    ax_rsi.plot(df.index, df["rsi14"], color="purple", linewidth=1)
    ax_rsi.axhline(70, color=COLOR_UP, linestyle="--", linewidth=0.8)
    ax_rsi.axhline(30, color=COLOR_DOWN, linestyle="--", linewidth=0.8)
    ax_rsi.set_ylabel("RSI14")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.grid(alpha=0.3)

    # --- ④ ボリンジャー%B ---
    ax_bb.plot(df.index, df["bb_pct_b"], color="teal", linewidth=1)
    ax_bb.axhline(1, color=COLOR_UP, linestyle="--", linewidth=0.8)
    ax_bb.axhline(0, color=COLOR_DOWN, linestyle="--", linewidth=0.8)
    ax_bb.set_ylabel("BB %B")
    ax_bb.grid(alpha=0.3)

    # --- ⑤ ATR14 ---
    ax_atr.plot(df.index, df["atr14"], color="brown", linewidth=1)
    ax_atr.set_ylabel("ATR14")
    ax_atr.grid(alpha=0.3)

    # --- ⑥ モメンタム加速度 ---
    ax_mom.plot(df.index, df["mom_accel"], color="darkgreen", linewidth=1)
    ax_mom.axhline(0, color="black", linewidth=0.6)
    ax_mom.set_ylabel("モメンタム加速度")
    ax_mom.grid(alpha=0.3)

    ax_mom.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_mom.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"ステップ1チャート保存完了: {save_path}")


# =============================================================================
# ステップ2の可視化
#   1. 価格 + 強気ダイバージェンス + 過熱ゾーン + 一時的急騰マーカー
#   2. 移動平均乖離率Zスコア（下げすぎ/上げすぎ）
#   3. ATR正規化乖離（過熱感）
#   4. RSI14（ダイバージェンス強度を重ねる）
# =============================================================================

def plot_step2(df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(
        4, 1, figsize=(14, 14), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1, 1]},
    )
    ax_price, ax_zscore, ax_heat, ax_rsi = axes

    # --- ① 価格 + シグナルマーカー ---
    ax_price.plot(df.index, df["close"], color=COLOR_PRICE, linewidth=1.2, label="終値")

    div_points = df[df["rsi_bullish_divergence"] == 1]
    ax_price.scatter(div_points.index, div_points["close"],
                     marker="^", color="green", s=70, zorder=5,
                     label="RSI強気ダイバージェンス")

    pump_points = df[df["is_likely_temporary_pump"] == 1]
    ax_price.scatter(pump_points.index, pump_points["close"],
                     marker="v", color="red", s=70, zorder=5,
                     label="一時的急騰の疑い")

    overheat_buy = df[df["is_overbought_heat"] == 1]
    ax_price.scatter(overheat_buy.index, overheat_buy["close"],
                     marker=".", color="orange", s=20, alpha=0.5,
                     label="過熱(買われすぎ)")

    overheat_sell = df[df["is_oversold_heat"] == 1]
    ax_price.scatter(overheat_sell.index, overheat_sell["close"],
                     marker=".", color="blue", s=20, alpha=0.5,
                     label="過熱(売られすぎ)")

    ax_price.set_ylabel("価格")
    ax_price.set_title("ステップ2: 価格とドメイン知識シグナル")
    ax_price.legend(loc="upper left", fontsize=8, ncol=2)
    ax_price.grid(alpha=0.3)

    # --- ② 移動平均乖離率Zスコア ---
    ax_zscore.plot(df.index, df["dev_ma25_zscore"], color="slateblue", linewidth=1,
                   label="dev_ma25 Zスコア")
    ax_zscore.axhline(2, color=COLOR_UP, linestyle="--", linewidth=0.8)
    ax_zscore.axhline(-2, color=COLOR_DOWN, linestyle="--", linewidth=0.8)
    ax_zscore.axhline(0, color="black", linewidth=0.5)
    ax_zscore.fill_between(df.index, 2, df["dev_ma25_zscore"].clip(lower=2),
                           color=COLOR_UP, alpha=0.3)
    ax_zscore.fill_between(df.index, -2, df["dev_ma25_zscore"].clip(upper=-2),
                           color=COLOR_DOWN, alpha=0.3)
    ax_zscore.set_ylabel("MA乖離\nZスコア")
    ax_zscore.legend(loc="upper left", fontsize=8)
    ax_zscore.grid(alpha=0.3)

    # --- ③ ATR正規化乖離（過熱感） ---
    ax_heat.plot(df.index, df["atr_dev_ma25"], color="darkorange", linewidth=1,
                label="ATR正規化乖離")
    ax_heat.axhline(2, color=COLOR_UP, linestyle="--", linewidth=0.8)
    ax_heat.axhline(-2, color=COLOR_DOWN, linestyle="--", linewidth=0.8)
    ax_heat.axhline(0, color="black", linewidth=0.5)
    ax_heat.set_ylabel("過熱感\n(ATR単位)")
    ax_heat.legend(loc="upper left", fontsize=8)
    ax_heat.grid(alpha=0.3)

    # --- ④ RSI + ダイバージェンス強度 ---
    ax_rsi.plot(df.index, df["rsi14"], color="purple", linewidth=1, label="RSI14")
    ax_rsi.axhline(70, color=COLOR_UP, linestyle="--", linewidth=0.8)
    ax_rsi.axhline(30, color=COLOR_DOWN, linestyle="--", linewidth=0.8)
    if div_points.shape[0] > 0:
        ax_rsi.scatter(div_points.index, div_points["rsi14"],
                       marker="^", color="green", s=60, zorder=5)
    ax_rsi.set_ylabel("RSI14")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend(loc="upper left", fontsize=8)
    ax_rsi.grid(alpha=0.3)

    ax_rsi.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"ステップ2チャート保存完了: {save_path}")


# =============================================================================
# ステップ3の可視化
#   1. 価格 + トリプルバリアのラベル（成功=緑/失敗=赤）+ 見送りフラグ
#   2. tb_return の分布
#   3. tb_barrier の内訳（円グラフ）
#   4. フォワード・ドローダウンの推移
# =============================================================================

def plot_step3(df: pd.DataFrame, save_path: Path) -> None:
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1, 1])

    ax_price = fig.add_subplot(gs[0, :])
    ax_hist  = fig.add_subplot(gs[1, 0])
    ax_pie   = fig.add_subplot(gs[1, 1])
    ax_dd    = fig.add_subplot(gs[2, :])
    ax_avoid = fig.add_subplot(gs[3, :])

    # --- ① 価格 + ラベルマーカー ---
    ax_price.plot(df.index, df["close"], color=COLOR_PRICE, linewidth=1, alpha=0.8, label="終値")

    win_points  = df[df["tb_label"] == 1]
    lose_points = df[df["tb_label"] == 0]
    ax_price.scatter(win_points.index, win_points["close"],
                     marker="^", color="green", s=10, alpha=0.5, label="利確到達 (label=1)")
    ax_price.scatter(lose_points.index, lose_points["close"],
                     marker="v", color="red", s=10, alpha=0.4, label="利確未到達 (label=0)")

    # 見送り推奨期間は点で重ねるとマーカーが多すぎて見づらいため背景帯で表示
    avoid_flag = df["avoid_entry_flag"]
    ax_price.fill_between(df.index, ax_price.get_ylim()[0], ax_price.get_ylim()[1],
                          where=(avoid_flag == 1), color="black", alpha=0.06, step="mid",
                          label="見送り推奨区間(DD閾値超え)")

    ax_price.set_ylabel("価格")
    ax_price.set_title("ステップ3: トリプルバリア法のラベリング結果")
    ax_price.legend(loc="upper left", fontsize=8, ncol=2)
    ax_price.grid(alpha=0.3)

    # --- ② tb_return の分布 ---
    ax_hist.hist(df["tb_return"] * 100, bins=40, color="slateblue", alpha=0.7)
    ax_hist.axvline(0, color="black", linewidth=0.8)
    ax_hist.set_xlabel("バリア到達時リターン (%)")
    ax_hist.set_ylabel("件数")
    ax_hist.set_title("tb_return 分布")
    ax_hist.grid(alpha=0.3)

    # --- ③ tb_barrier 内訳 ---
    barrier_counts = df["tb_barrier"].value_counts()
    labels_jp = {"take_profit": "利確到達", "stop_loss": "損切り", "time_out": "期間切れ"}
    color_map = {"take_profit": "#2ca02c", "stop_loss": "#d62728", "time_out": "#7f7f7f"}
    pie_labels = [labels_jp.get(k, k) for k in barrier_counts.index]
    pie_colors = [color_map.get(k, "#cccccc") for k in barrier_counts.index]
    ax_pie.pie(barrier_counts.values, labels=pie_labels, autopct="%1.0f%%",
              colors=pie_colors)
    ax_pie.set_title("バリア到達内訳")

    # --- ④ フォワード・ドローダウンの推移 ---
    ax_dd.plot(df.index, df["forward_max_drawdown"] * 100, color="firebrick", linewidth=1)
    ax_dd.axhline(0, color="black", linewidth=0.5)
    ax_dd.set_ylabel("最大DD\n(%)")
    ax_dd.grid(alpha=0.3)

    # --- ⑤ 見送りフラグの推移（点) ---
    avoid_flag = df["avoid_entry_flag"]
    ax_avoid.fill_between(df.index, 0, avoid_flag, step="mid", color="black", alpha=0.4)
    ax_avoid.set_ylabel("見送り\nフラグ")
    ax_avoid.set_ylim(-0.1, 1.1)
    ax_avoid.grid(alpha=0.3)

    ax_avoid.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_avoid.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"ステップ3チャート保存完了: {save_path}")


# =============================================================================
# メイン処理
# =============================================================================

def visualize_step1(csv_path: Path = None, save_path: Path = None) -> None:
    csv_path = csv_path or RESOURCE_DIR / "step1_dataset.csv"
    save_path = save_path or RESOURCE_DIR / "step1_chart.png"
    df = _load_dataset(csv_path)
    plot_step1(df, save_path)


def visualize_step2(csv_path: Path = None, save_path: Path = None) -> None:
    csv_path = csv_path or RESOURCE_DIR / "step2_dataset.csv"
    save_path = save_path or RESOURCE_DIR / "step2_chart.png"
    df = _load_dataset(csv_path)
    plot_step2(df, save_path)


def visualize_step3(csv_path: Path = None, save_path: Path = None) -> None:
    csv_path = csv_path or RESOURCE_DIR / "step3_dataset.csv"
    save_path = save_path or RESOURCE_DIR / "step3_chart.png"
    df = _load_dataset(csv_path)
    plot_step3(df, save_path)


if __name__ == "__main__":
    visualize_step1()
    visualize_step2()
    visualize_step3()
