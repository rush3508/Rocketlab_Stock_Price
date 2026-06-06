# RKLB Stock Prediction

A weekend project that spiralled into a full ML pipeline. Predicts Rocket Lab's
next 4 trading days of prices and generates a BUY/SELL/HOLD signal. Built to run
on a laptop — no GPU needed.

## What It Does

Takes price data, Electron rocket launch schedules, and financial news headlines,
then produces a short-term price forecast plus a trading signal. Three models are
combined — each one "sees" the data differently:

- **LightGBM** — a gradient-boosted decision tree model. It learns from 31 hand-crafted
  features like RSI (a momentum measure), MACD (a trend indicator), Bollinger Bands
  (a volatility measure), and launch event timing. Trees are good at spotting
  non-linear patterns in tabular data without much tuning.

- **BiLSTM + Multi-Head Attention** — a neural network designed for sequences. It looks
  at the last 20 trading days (roughly one calendar month) and learns which days matter
  most for predicting what comes next. "Bi" means it reads the sequence both
  forward and backward. "Attention" lets the model focus on specific timesteps rather
  than treating all 20 days equally.

- **FinBERT** — a language model trained on financial text. It reads news headlines
  and scores them from negative (−1) to positive (+1). This is the sentiment signal.
  Runs on CPU at about 1 second per headline and caches results to disk so you only
  pay that cost once.

The three models' outputs are blended two ways. First, a **weighted average** (48%
LightGBM, 48% LSTM, 4% sentiment — sentiment is downweighted because we only have
real scores on ~20 of ~1,000 trading days). Second, a **Ridge meta-learner** — a
simple linear model trained to find the best mix of all three signals by seeing
what worked on a held-out validation set it never saw during training.

## Why % Returns Instead of Raw Prices

Both models originally predicted raw dollar prices. That worked fine when RKLB
traded between $4 and $17 for two years. Then the stock moved to $150 and
everything broke — the LSTM checkpoint was still predicting numbers in the $60s
because its training data only ever showed prices in the single and low double
digits. Tree splits on dollar thresholds don't extrapolate either.

Both models now predict **percentage returns** (e.g. "up 3%") and convert back to
dollars at inference time using the current price. A 3% move is a 3% move whether
the stock is $5 or $150. This is the single biggest structural lesson from building
this project.

## Current Numbers (v4, June 2026)

**Data:** 1,048 trading days (April 2022 → June 2026).  
**Split:** Train 725 rows (Jun 2022 → May 2025) / Val 120 rows / Test 150 rows (Oct 2025 → Jun 2026).  
**Test context:** RKLB's bull run from ~$20–30 upwards — a regime the models had barely seen during training.

| Model | Day +1 MAE | Day +2 MAE | Day +3 MAE | Day +4 MAE | Direction Acc |
| --- | --- | --- | --- | --- | --- |
| LightGBM | $4.04 | $5.35 | $7.45 | $8.72 | 49.3% |
| LSTM + Attention | $4.06 | $5.17 | $7.03 | $7.97 | 43.9% |
| Weighted Ensemble | $5.29 | $6.07 | $7.21 | $7.99 | 48.5% |
| **Meta-Learner Ensemble** | **$4.33** | **$5.69** | **$7.92** | **$9.11** | **50.0%** |

**MAE** (Mean Absolute Error) is the average dollar gap between the predicted price
and the actual price. Lower is better.

**Direction Accuracy** is how often the model correctly predicted whether the stock
went up or down (ignoring the size of the move). 50% is coin-flip random.

Direction accuracy is basically coin-flip across all models. The models trained on
2022–2025 price action — choppy, mostly sideways between $4–$17 — and then got
tested on a momentum bull run to all-time highs. The direction head hasn't seen
this regime before. Price MAE is more useful here: a $4 miss on a $110 stock is
roughly 3.6%, which is reasonable for a 1-day forecast. Direction signal generation
was replaced with price-derived signals in v3 (if d+1 forecast > current price by
more than 2%, signal BUY; below −2%, SELL; otherwise HOLD).

**LightGBM cross-validation (on training data only):**  
d+1: 4.5% ± 1.2% | d+2: 6.2% ± 1.3% | d+3: 7.7% ± 1.8% | d+4: 8.9% ± 2.1%  
Direction: 0.473 ± 0.059

**LSTM training:** Early stopping at epoch 28, best validation loss 0.2140.

## Live Forecast (as of 2026-06-05)

| Date | Predicted Price | Change from Today | Signal |
|---|---|---|---|
| 2026-06-08 | $110.13 | +$0.05 | **HOLD** |
| 2026-06-09 | $110.66 | +$0.58 | — |

Current price: **$110.08**. Current sentiment score: **+0.797** (positive news).

## Project Layout

```
rocketlab-stock-prediction/
├── notebooks/
│   └── demo.ipynb                 # The only thing you need to run
├── data/
│   ├── fetch_price.py             # yfinance OHLCV (RKLB + NASDAQ)
│   ├── fetch_news.py              # NewsAPI + Yahoo Finance headlines
│   ├── fetch_launches.py          # Electron launch history scraper
│   ├── electron_launches.md       # Human-readable launch table (88 launches)
│   └── electron_launches_clean.csv
├── features/
│   ├── technical_indicators.py    # RSI, MACD, Bollinger Bands, ATR, momentum
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
about 15–20 minutes on a laptop. Subsequent runs load cached model files from
`artifacts/` and finish in seconds.

To force a full retrain after changing features or model code:

```bash
rm -rf artifacts/*
```

Then re-run `demo.ipynb` top to bottom. The notebook is the only entry point —
all the `.py` files are imported as modules, not run directly.

## Design Choices Worth Mentioning

**Strict chronological split — no shuffling.** 75% train, 10% validation, 15%
test, in time order. You can't randomly shuffle a time series and claim your test
metrics mean anything — you'd be letting the model implicitly "see the future"
during training.

**Walk-forward cross-validation on training data only.** LightGBM's internal
cross-validation runs on `train_df` exclusively. An earlier version ran CV on
train+val combined and produced unrealistically tight error estimates — the model
had quietly seen validation-era data in some folds.

**20-day lookback window, not 60.** A 60-day lookback only produced ~680 training
sequences from ~1,000 rows of data. At 20 days you get ~2,040 — triple the data
for the LSTM to learn from. 20 trading days is roughly a calendar month, which
captures recent momentum without dragging in stale history.

**Sentiment forward-fill capped at 3 days.** There are only about 20 days with
real FinBERT scores across ~1,000 trading days. Without a cap, a single article's
sentiment score would propagate forward for weeks. Beyond 3 trading days from the
scored headline, sentiment defaults to zero.

**RobustScaler over MinMaxScaler.** RKLB has had multiple ±20% single-day moves.
`RobustScaler` uses the median and interquartile range to scale the data, making
it much less sensitive to extreme outliers than `MinMaxScaler`, which scales
everything relative to the global min and max. A single spike would compress the
rest of the distribution into a narrow band with MinMaxScaler.

**Multi-task LSTM loss.** `loss = MSE(predicted returns) + 0.3 × BCE(direction)`.
The network simultaneously predicts price returns and up/down direction. The 0.3
weight treats direction as an auxiliary task — useful for learning but not
dominant. BCE is binary cross-entropy, the standard loss for yes/no classification.

## Hardware

Runs fine on a ThinkPad T14s with 8GB RAM. No GPU. The slowest part is FinBERT at
~1 second per headline, but results are cached to disk so it's a one-time cost.
LSTM trains in 5–10 minutes; LightGBM in under a minute.

## Bugs Fixed (So I Don't Make Them Again)

- **Data leakage in feature columns.** `target_return_d1..d4` columns (the thing
  we're trying to predict) were accidentally included in the feature set. LightGBM
  scored 100% direction accuracy because it could trivially check whether the
  target return was positive or negative — it was literally in the input.

- **Meta-learner trained on mismatched scales.** LSTM predictions were in return
  scale (~0.001) while LightGBM predictions were in dollar scale (~$150). The
  linear meta-learner saw one input that was 150,000× larger than another and
  learned garbage weights. Everything now converts to dollars before entering the
  meta-learner.

- **Stale LSTM checkpoint.** The saved model was trained on $4–$17 RKLB. Loading
  it when the stock was at $150 gave predictions in the $60s. Switching to % return
  targets means the model is regime-agnostic — a 3% move is a 3% move.

- **LightGBM raw price targets.** Tree splits on absolute price levels don't
  extrapolate beyond the training range. A model that learned splits at "$10 or
  below is regime A" will produce nonsense when the stock is at $110. Switched to
  % return targets to match the LSTM.

- **Sentiment propagating for months.** `ffill()` with no limit meant one article
  from March was still influencing June's sentiment feature. Capped at 3 days.

- **LightGBM validation leakage in cross-validation (introduced twice).** The first
  time: `train()` concatenated train+val before fitting. The second time: the same
  combined dataset fed into `TimeSeriesSplit`, letting later folds include
  validation-era data in training. CV now runs on training data only.

Full technical bug log with 12 documented fixes in `CLAUDE.md`.

## To Do

- Retrain after accumulating 3–6 more months of post-rally data to see if
  directional accuracy improves beyond 50%
- Add macroeconomic indicators (interest rates, VIX) as features
- Experiment with a proper backtesting framework instead of the current static
  train/val/test split
- CI/CD pipeline for automated retraining on a schedule

## Disclaimer

This is a learning project. Nothing here is financial advice. Past performance —
especially on a test set that barely overlaps with current market conditions —
does not predict anything about future returns.
