# RKLB Stock Prediction

A weekend project that spiralled into a full ML pipeline. Predicts Rocket Lab's
4-day price direction using an ensemble of LightGBM, a BiLSTM with attention, and
FinBERT sentiment analysis. Built to run on a laptop — no GPU needed.

## What It Does

Takes OHLCV data, Electron launch schedules, and financial news headlines, then
spits out a 4-day price forecast plus a BUY/SELL/HOLD signal. The ensemble
combines three models that think differently about the problem:

- **LightGBM** — gradient-boosted trees on 31 engineered features. Good at
  capturing regime-independent patterns, especially with walk-forward validation.

- **BiLSTM + Multi-Head Attention** — a two-layer bidirectional LSTM that learns
  temporal structure from 20-day sequences, with a self-attention layer that
  decides which timesteps actually matter.

- **FinBERT** — ProsusAI's financial BERT model scoring news sentiment. Runs on
  CPU at about 1 second per headline. Cached to disk so you only pay the cost once.

The base model predictions feed into a Ridge meta-learner that learns how much to
trust each model based on what actually happened on held-out validation data.

## Why % Returns Instead of Raw Prices

This was the single biggest lesson from building this. Both models originally
predicted raw dollar prices. That worked fine when RKLB traded between $4 and $17
for two years. Then it went to $150 and everything broke — the LSTM checkpoint was
still predicting numbers in the $60s.

Both models now predict **percentage returns** and convert back to dollars at
inference time. A 3% move is a 3% move whether the stock is $5 or $150. The
`current_prices` parameter has to be passed everywhere for this to work — there
are five separate places in the pipeline where forgetting it produces garbage
output at a completely different scale. Ask me how I know.

## Current Numbers (v3, June 2026)

Test set: 149 trading days (Oct 2025 – May 2026). This period covers RKLB's run
from ~$20 to $150 — a bull regime that barely existed in training data.

| Model | d+1 MAE | d+4 MAE | Direction Acc |
|---|---|---|---|
| LSTM + Attention | $3.79 | $7.32 | 43.4% |
| LightGBM | $4.04 | $8.34 | 43.6% |
| Meta-Learner Ensemble | $4.31 | $8.78 | 44.2% |

The direction accuracy is below 50% and I'm not going to pretend otherwise. The
models trained on 2022–2025 price action — choppy, mean-reverting, mostly
sideways — and then got tested on a momentum-driven bull run to all-time highs.
The directional head hasn't seen this regime before. The price forecasts are
decent (LSTM hit a $3.79 d+1 MAE on a $150 stock), but direction is basically
noise right now. I stopped using it for signal generation in v3 and switched to
price-derived signals instead. This should sort itself out as post-rally data
accumulates in the training set.

## Project Layout

```
rocketlab-stock-prediction/
├── notebooks/
│   └── demo.ipynb                 # The only thing you need to run
├── data/
│   ├── fetch_price.py             # yfinance OHLCV (RKLB + NASDAQ)
│   ├── fetch_news.py              # NewsAPI + Yahoo Finance headlines
│   ├── fetch_launches.py          # Electron launch history scraper
│   ├── electron_launches.md       # Human-readable launch table
│   └── electron_launches_clean.csv
├── features/
│   ├── technical_indicators.py    # RSI, MACD, Bollinger, ATR, momentum
│   ├── launch_features.py         # Launch event features
│   └── preprocess.py              # Merge, scale, sequence, split
├── models/
│   ├── lgbm_model.py              # LightGBM: 4 regressors + walk-forward CV
│   ├── lstm_attention.py          # PyTorch BiLSTM + MultiheadAttention
│   └── finbert_sentiment.py       # FinBERT NLP with disk cache
├── ensemble/
│   └── ensemble.py                # Weighted average + Ridge meta-learner
├── config.py                      # Every hyperparameter in one place
├── requirements.txt
└── artifacts/                     # Saved models (gitignored, delete before retrain)
```

## Setup & Running

```bash
pip install -r requirements.txt
echo "your-newsapi-key" > newsapi-key.txt
rm -rf artifacts/*              # Clean slate — always before a fresh run
jupyter notebook notebooks/demo.ipynb
```

First run downloads FinBERT (~440MB) and trains everything from scratch. Takes
about 15–20 minutes on a laptop. Subsequent runs load cached artifacts and finish
in seconds.

To force a full retrain after changing model or feature code:

```bash
rm -rf artifacts/*
```

Then re-run `demo.ipynb` top to bottom. The notebook is the only entry point —
all the `.py` files are imported as library modules.

## Design Choices Worth Mentioning

**Strict chronological split.** 75% train, 10% validation, 15% test. No shuffling
anywhere. You can't random-shuffle time series and pretend your test metrics mean
anything.

**Walk-forward CV on train only.** LightGBM cross-validation runs on `train_df`
exclusively. I originally ran CV on train+val combined and got artificially tight
metrics — turns out the model had seen val-era data in some folds.

**20-day lookback, not 60.** The longer window only produced ~680 training
sequences. At 20 days you get ~2,040 — triple the data for the LSTM to learn
from. 20 trading days is roughly a calendar month, which feels about right for
capturing short-term momentum without dragging in ancient history.

**Sentiment forward-fill capped at 3 days.** There are only about 17 days with
actual FinBERT scores across nearly 1,000 trading days. Without a cap, a single
bullish article from March would still be propping up sentiment scores in June.
Beyond 3 days, sentiment defaults to zero.

**RobustScaler over MinMaxScaler.** RKLB has had multiple ±20% single-day moves.
Median + IQR scaling handles those outliers without compressing the rest of the
distribution into a sliver.

**LSTM loss = MSE(price returns) + 0.3 × BCE(direction).** Direction is an
auxiliary task, not the main event. The 0.3 weight keeps it from dominating
optimisation.

## Hardware

Runs fine on a ThinkPad T14s with 8GB RAM. No GPU. The slowest part is FinBERT
at ~1 second per headline, but results are cached so it's a one-time cost.
LSTM trains in 5–10 minutes.

## Bugs I Fixed (So I Don't Make Them Again)

- **Data leakage in feature columns (Bug #10).** `target_return_d1..d4` columns
  were leaking into the feature set. LightGBM direction accuracy was 100% because
  it could trivially split on `target_return_d1 > 0`. Removed from the feature
  column list.

- **Meta-learner trained on mismatched scales (Bug #7).** LSTM predictions in
  return-scale (~0.001) got fed to the Ridge meta-learner alongside LightGBM
  predictions in dollar-scale (~$150). The meta-learner tried to blend 0.1% and
  $150 and produced nonsense. Everything now flows through dollar conversion
  before hitting the meta-learner.

- **LSTM checkpoint stale across price regimes (Bug #1).** The saved `lstm_model.pt`
  was trained on $4–$17 RKLB. Loading those weights when the stock was at $150
  gave predictions in the $60s. Fixed by switching to % return targets — but the
  lesson is that model checkpoints have a shelf life.

- **LightGBM raw price targets (Bug #12).** Tree splits on absolute price levels
  don't extrapolate. A model trained at $5–$25 couldn't handle testing at
  $50–$150. Switched LGBM to % return targets to match the LSTM approach.

- **Sentiment propagating for months (Bug #4).** `ffill()` with no limit meant
  one article's sentiment score survived for weeks. Capped at 3 days.

Full bug tracker with 12 documented fixes in `CLAUDE.md`.

## To Do

- Retrain after accumulating 3–6 more months of post-rally data to see if
  directional accuracy recovers
- Try adding macroeconomic indicators (interest rates, VIX) as features
- Experiment with a proper backtesting framework instead of the current
  train/val/test split
- CI/CD pipeline for automated retraining on a schedule

## Disclaimer

This is a learning project. Nothing here is financial advice. Past performance —
especially on a test set that barely overlaps with current market conditions —
does not guarantee anything about future returns.
