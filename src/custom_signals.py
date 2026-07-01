
"""Custom signal detection utilities for trading_assistant.

Implements detection for the `high_atr_bottoming` signal as described
in the implementation plan (priority C). The function returns a boolean Series
that can be used to assign `signal_type = "high_atr_bottoming"`.
"""

import pandas as pd

def detect_high_atr_bottoming(df: pd.DataFrame) -> pd.Series:
    """Detect high ATR bottoming conditions.

    Conditions (based on plan):
    - weekly_trend == -1
    - 35 <= rsi14 <= 45
    - atr_percentile >= 0.90
    - is_overbought_heat == 0
    - atr_dev_ma25 <= 0

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the required columns.

    Returns
    -------
    pd.Series of bool
        True where all conditions are met.
    """
    required_cols = [
        "weekly_trend",
        "rsi14",
        "atr_percentile",
        "is_overbought_heat",
        "atr_dev_ma25",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' required for high_atr_bottoming detection not found.")
    cond = (
        (df["weekly_trend"] == -1)
        & (df["rsi14"].between(35, 45, inclusive="both"))
        & (df["atr_percentile"] >= 0.90)
        & (df["is_overbought_heat"] == 0)
        & (df["atr_dev_ma25"] <= 0)
    )
    return cond
