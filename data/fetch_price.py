"""
Fetches OHLCV price data for RKLB and NASDAQ from yfinance.
Returns a merged daily DataFrame indexed by Date.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from config import TICKER, NASDAQ_TICKER, START_DATE


def _normalize_column_name(column) -> str:
    if isinstance(column, tuple):
        for part in column:
            if isinstance(part, str):
                cleaned = part.strip().lower().replace(" ", "_")
                if cleaned in {"open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits", "capital_gains"}:
                    return cleaned

        for part in reversed(column):
            if isinstance(part, str) and part.strip():
                return part.strip().lower().replace(" ", "_")

        return "_".join(str(part) for part in column).lower().replace(" ", "_")

    return str(column).lower().replace(" ", "_")


def fetch_ohlcv(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    end = end or datetime.now().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df.columns = [_normalize_column_name(c) for c in df.columns]
    df.index.name = "date"
    return df.sort_index()


def load_price_data(start: str = START_DATE, end: str | None = None) -> pd.DataFrame:
    """
    Returns a single DataFrame with RKLB OHLCV columns plus NASDAQ close/volume.
    All columns prefixed: rklb_open, rklb_close, ..., nasdaq_close, nasdaq_volume.
    """
    rklb = fetch_ohlcv(TICKER, start, end)
    nasdaq = fetch_ohlcv(NASDAQ_TICKER, start, end)[["close", "volume"]]

    rklb = rklb.rename(columns=lambda c: f"rklb_{c}")
    nasdaq = nasdaq.rename(columns=lambda c: f"nasdaq_{c}")

    merged = rklb.join(nasdaq, how="inner")
    merged.dropna(inplace=True)
    return merged


if __name__ == "__main__":
    df = load_price_data()
    print(df.tail())
    print(f"\nShape: {df.shape}")
