"""
Fetches RKLB-related news headlines from two sources:
  1. NewsAPI (newsapi.org) — broader financial/tech news
  2. yfinance .news — Yahoo Finance news specific to the ticker

Headlines are merged, deduplicated, and returned as a DataFrame
with columns: [date, title, source, url].
"""

import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from config import NEWSAPI_KEY_PATH, TICKER, NEWSAPI_LOOKBACK_DAYS, NEWSAPI_MAX_ARTICLES


def _load_api_key() -> str:
    return NEWSAPI_KEY_PATH.read_text().strip()


def fetch_newsapi_headlines(
    query: str = "Rocket Lab RKLB",
    days_back: int = NEWSAPI_LOOKBACK_DAYS,
) -> pd.DataFrame:
    key = _load_api_key()
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": NEWSAPI_MAX_ARTICLES,
        "apiKey": key,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])

    rows = []
    for a in articles:
        published = a.get("publishedAt", "")[:10]  # YYYY-MM-DD
        title = (a.get("title") or "").strip()
        if title and published:
            rows.append({
                "date": pd.to_datetime(published),
                "title": title,
                "source": a.get("source", {}).get("name", "NewsAPI"),
                "url": a.get("url", ""),
            })
    return pd.DataFrame(rows)


def fetch_yfinance_headlines() -> pd.DataFrame:
    ticker = yf.Ticker(TICKER)
    news = ticker.news or []
    rows = []
    for item in news:
        ts = item.get("providerPublishTime")
        title = (item.get("title") or "").strip()
        if ts and title:
            rows.append({
                "date": pd.to_datetime(ts, unit="s").normalize(),
                "title": title,
                "source": item.get("publisher", "Yahoo Finance"),
                "url": item.get("link", ""),
            })
    return pd.DataFrame(rows)


def load_news(days_back: int = NEWSAPI_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Combines NewsAPI + yfinance headlines, deduplicates by title, sorts by date.
    Returns DataFrame with columns: date, title, source, url.
    """
    frames = []

    try:
        newsapi_df = fetch_newsapi_headlines(days_back=days_back)
        frames.append(newsapi_df)
        print(f"  NewsAPI: {len(newsapi_df)} articles")
    except Exception as e:
        print(f"  NewsAPI unavailable: {e}")

    try:
        yf_df = fetch_yfinance_headlines()
        frames.append(yf_df)
        print(f"  yfinance: {len(yf_df)} articles")
    except Exception as e:
        print(f"  yfinance news unavailable: {e}")

    if not frames:
        return pd.DataFrame(columns=["date", "title", "source", "url"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="title")
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


if __name__ == "__main__":
    df = load_news()
    print(df[["date", "title", "source"]].tail(10).to_string())
    print(f"\nTotal headlines: {len(df)}")
