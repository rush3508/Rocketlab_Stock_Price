"""
Merges all features into one dataset and builds train/val/test splits.
Uses time-aware splitting to prevent data leakage.
Also builds sliding-window sequences for the LSTM.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from typing import Tuple

from config import (
    LOOKBACK_DAYS, FORECAST_HORIZON, START_DATE,
    SCALER_PATH, FEATURE_COLS_PATH,
)
from data.fetch_price import load_price_data
from features.technical_indicators import build_all_indicators
from features.launch_features import add_launch_features


# Columns that are targets (not fed as input features)
TARGET_COL = "rklb_close"
DIRECTION_COL = "direction"  # 1 = up, 0 = down vs today
RETURN_COL = "target_return"

def build_dataset(sentiment_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Full feature matrix. Merges price, indicators, launch features, and
    optional sentiment scores (daily aggregated FinBERT scores).

    Args:
        sentiment_df: DataFrame indexed by date with column 'sentiment_score'.

    Returns:
        DataFrame indexed by date, all NaNs dropped.
    """
    price_df = load_price_data(start=START_DATE)
    df = build_all_indicators(price_df)
    df = add_launch_features(df)

    if sentiment_df is not None:
        sentiment_df = sentiment_df[["sentiment_score"]].copy()
        sentiment_df.index = pd.to_datetime(sentiment_df.index).normalize()
        df = df.join(sentiment_df, how="left")
        # Cap forward-fill at 3 trading days — after that, sentiment is stale
        # WHY: only 14 days of real scores exist across 988 rows. Unlimited ffill
        # creates fake signal that persists for months from a single article.
        df["sentiment_score"] = (
            df["sentiment_score"].ffill(limit=3).fillna(0.0)
        )
    else:
        df["sentiment_score"] = 0.0

   # Target columns — % return from today's close to each future day's close
    # WHY returns not prices: a model trained on $4-$17 prices can't predict $134.
    # Returns (e.g. +0.03 = up 3%) are scale-invariant and stay valid as price rises.
    for h in range(1, FORECAST_HORIZON + 1):
        df[f"target_close_d{h}"] = df[TARGET_COL].shift(-h)
        df[f"target_return_d{h}"] = (
            df[TARGET_COL].shift(-h) / df[TARGET_COL] - 1
        )  # % return — used by LSTM

    df[DIRECTION_COL] = (df[f"target_return_d1"] > 0).astype(int)

    df.dropna(inplace=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Returns input feature column names (excludes targets and raw OHLCV)."""
    exclude = {
        "rklb_open", "rklb_high", "rklb_low", "rklb_volume",
        "nasdaq_volume", DIRECTION_COL,
    } | {f"target_close_d{h}" for h in range(1, FORECAST_HORIZON + 1)} \
      | {f"target_return_d{h}" for h in range(1, FORECAST_HORIZON + 1)}
    return [c for c in df.columns if c not in exclude]


def time_split(
    df: pd.DataFrame, val_frac: float = 0.12, test_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train / val / test split. No shuffling.
    """
    n = len(df)
    test_start = int(n * (1 - test_frac))
    val_start = int(test_start * (1 - val_frac / (1 - test_frac)))
    return df.iloc[:val_start], df.iloc[val_start:test_start], df.iloc[test_start:]


def build_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> RobustScaler:
    """Fits a RobustScaler on training data only and saves it."""
    scaler = RobustScaler()
    scaler.fit(train_df[feature_cols])
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    return scaler


def load_scaler() -> Tuple[RobustScaler, list[str]]:
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return scaler, feature_cols


def make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: RobustScaler,
    lookback: int = LOOKBACK_DAYS,
    horizon: int = FORECAST_HORIZON,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds (X, y_price, y_dir) arrays for LSTM training.

    X      : (N, lookback, n_features)  scaled input sequences
    y_price: (N, horizon)               future close prices (unscaled)
    y_dir  : (N,)                       direction label (0/1) for day+1
    """
    X_scaled = scaler.transform(df[feature_cols])
    # LSTM trains on % returns — scale-invariant across all price levels
    return_cols = [f"target_return_d{h}" for h in range(1, horizon + 1)]
    y_price = df[return_cols].values   # shape (N, horizon) — values like 0.03, -0.01
    y_dir = df[DIRECTION_COL].values

    X_seq, yp_seq, yd_seq = [], [], []
    for i in range(lookback, len(df)):
        X_seq.append(X_scaled[i - lookback: i])
        yp_seq.append(y_price[i])
        yd_seq.append(y_dir[i])

    return np.array(X_seq), np.array(yp_seq), np.array(yd_seq)
