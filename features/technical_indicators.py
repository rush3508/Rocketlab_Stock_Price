"""
Computes technical indicators from OHLCV price data.
All functions operate on a DataFrame and return it with new columns appended.
No external TA libraries required — computed in pure pandas/numpy.
"""

import pandas as pd
import numpy as np
from config import RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BBAND_PERIOD, ATR_PERIOD


def add_rsi(df: pd.DataFrame, close_col: str = "rklb_close", period: int = RSI_PERIOD) -> pd.DataFrame:
    delta = df[close_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    close_col: str = "rklb_close",
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    ema_fast = df[close_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[close_col].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(
    df: pd.DataFrame, close_col: str = "rklb_close", period: int = BBAND_PERIOD
) -> pd.DataFrame:
    sma = df[close_col].rolling(period).mean()
    std = df[close_col].rolling(period).std()
    df["bb_upper"] = sma + 2 * std
    df["bb_lower"] = sma - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
    df["bb_pct"] = (df[close_col] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


def add_atr(
    df: pd.DataFrame,
    high_col: str = "rklb_high",
    low_col: str = "rklb_low",
    close_col: str = "rklb_close",
    period: int = ATR_PERIOD,
) -> pd.DataFrame:
    prev_close = df[close_col].shift(1)
    tr = pd.concat(
        [
            df[high_col] - df[low_col],
            (df[high_col] - prev_close).abs(),
            (df[low_col] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(com=period - 1, min_periods=period).mean()
    df["atr_pct"] = df["atr"] / df[close_col]
    return df


def add_volume_features(df: pd.DataFrame, vol_col: str = "rklb_volume") -> pd.DataFrame:
    df["volume_change"] = df[vol_col].pct_change()
    df["volume_ma20"] = df[vol_col].rolling(20).mean()
    df["volume_ratio"] = df[vol_col] / df["volume_ma20"]
    return df


def add_price_features(df: pd.DataFrame, close_col: str = "rklb_close") -> pd.DataFrame:
    df["return_1d"] = df[close_col].pct_change()
    df["return_5d"] = df[close_col].pct_change(5)
    df["return_20d"] = df[close_col].pct_change(20)
    df["momentum_10"] = df[close_col] / df[close_col].shift(10) - 1
    df["sma_20"] = df[close_col].rolling(20).mean()
    df["sma_50"] = df[close_col].rolling(50).mean()
    df["price_vs_sma20"] = df[close_col] / df["sma_20"] - 1
    df["price_vs_sma50"] = df[close_col] / df["sma_50"] - 1
    return df


def add_nasdaq_features(df: pd.DataFrame, nasdaq_col: str = "nasdaq_close") -> pd.DataFrame:
    df["nasdaq_return_1d"] = df[nasdaq_col].pct_change()
    df["nasdaq_return_5d"] = df[nasdaq_col].pct_change(5)
    df["rklb_nasdaq_ratio"] = df["rklb_close"] / df[nasdaq_col]
    return df


def build_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all indicator functions in sequence. Returns augmented DataFrame."""
    df = df.copy()
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_volume_features(df)
    df = add_price_features(df)
    df = add_nasdaq_features(df)
    return df
