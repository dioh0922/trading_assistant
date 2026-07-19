"""プロジェクト設定定数

全モジュール共通のパラメータ・パス設定を一元管理する。
"""
import os
from pathlib import Path

# ===== パス設定 =====
# プロジェクトルート（src/ の1つ上）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = str(PROJECT_ROOT / "scraping" / "csv")
OUTPUT_DIR = str(PROJECT_ROOT / "output")

# ===== 乱数シード =====
RANDOM_SEED = 42

# ===== タスク1: トレンド予測 =====
TREND_FORWARD_DAYS = 20   # 予測対象の将来日数
TREND_THRESHOLD = 0.03    # 上昇/下降判定の閾値 (±3%)

# ===== タスク2: 利確/損切り判定 =====
PROFIT_TAKE_PCT = 0.10    # 利確ライン (+10%)
STOP_LOSS_PCT = 0.05      # 損切りライン (-5%)
EXIT_MAX_DAYS = 60        # 最大保有日数

# ===== タスク3: 目標到達予測 =====
TARGET_MAX_DAYS = 60      # 目標到達の最大日数

# ===== 特徴量パラメータ =====
MA_WINDOWS = [5, 25, 75]           # 移動平均の窓サイズ
RETURN_WINDOWS = [1, 5, 20, 60]    # リターン計算の窓サイズ
VOLUME_MA_WINDOW = 20              # 出来高移動平均の窓サイズ
ATR_WINDOW = 14                    # ATRの窓サイズ
RSI_WINDOW = 14                    # RSIの窓サイズ
MACD_FAST = 12                     # MACD 短期EMA
MACD_SLOW = 26                     # MACD 長期EMA
MACD_SIGNAL = 9                    # MACD シグナル線
BB_WINDOW = 20                     # ボリンジャーバンド窓サイズ
BB_STD = 2                         # ボリンジャーバンド標準偏差倍数
STOCH_K = 14                       # ストキャスティクス %K
STOCH_D = 3                        # ストキャスティクス %D

# ===== データ分割 =====
TRAIN_RATIO = 0.70   # 学習データ割合
VAL_RATIO = 0.15     # 検証データ割合
TEST_RATIO = 0.15    # テストデータ割合

# ===== その他 =====
MIN_HISTORY_DAYS = 75  # 特徴量計算に必要な最小履歴日数
