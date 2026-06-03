"""
FinBERT-based sentiment scoring for RKLB news headlines.
Model: ProsusAI/finbert (~440MB, CPU-friendly).

Scores are cached to disk so the model only runs on new headlines.
Output: daily sentiment score in [-1, 1] (negative → positive).
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline

from config import FINBERT_MODEL, FINBERT_BATCH, SENTIMENT_CACHE


def _load_pipeline():
    return pipeline(
        "text-classification",
        model=FINBERT_MODEL,
        tokenizer=FINBERT_MODEL,
        device=-1,          # CPU
        truncation=True,
        max_length=512,
    )


def _load_cache() -> dict:
    if SENTIMENT_CACHE.exists():
        with open(SENTIMENT_CACHE, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    SENTIMENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(SENTIMENT_CACHE, "wb") as f:
        pickle.dump(cache, f)


def _label_to_score(label: str, score: float) -> float:
    """Converts FinBERT label+confidence to a scalar in [-1, 1]."""
    if label == "positive":
        return score
    elif label == "negative":
        return -score
    return 0.0  # neutral


def score_headlines(headlines: list[str], use_cache: bool = True) -> list[float]:
    """
    Scores a list of headlines. Returns a list of floats in [-1, 1].
    New headlines are added to cache; already-cached ones skip inference.
    """
    cache = _load_cache() if use_cache else {}
    new_texts = [h for h in headlines if h not in cache]

    if new_texts:
        nlp = _load_pipeline()
        for i in tqdm(range(0, len(new_texts), FINBERT_BATCH), desc="FinBERT"):
            batch = new_texts[i: i + FINBERT_BATCH]
            results = nlp(batch)
            for text, res in zip(batch, results):
                cache[text] = _label_to_score(res["label"], res["score"])
        _save_cache(cache)

    return [cache.get(h, 0.0) for h in headlines]


def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a news DataFrame (columns: date, title) and returns a daily
    sentiment DataFrame indexed by date with column 'sentiment_score'.

    Score per day = mean of all headline scores that day.
    """
    if news_df.empty:
        return pd.DataFrame(columns=["sentiment_score"])

    headlines = news_df["title"].tolist()
    scores = score_headlines(headlines)
    news_df = news_df.copy()
    news_df["score"] = scores

    daily = (
        news_df.groupby(news_df["date"].dt.normalize())["score"]
        .mean()
        .rename("sentiment_score")
        .to_frame()
    )
    return daily


if __name__ == "__main__":
    from data.fetch_news import load_news
    news = load_news()
    daily = aggregate_daily_sentiment(news)
    print(daily.tail(10))
