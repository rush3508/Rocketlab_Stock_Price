# RKLB Stock Prediction — Multi-Signal Ensemble

**Rocket Lab USA (RKLB) | 4-Day Price Forecast + Directional Signal**

A production-structured ML pipeline that combines classical machine learning, deep learning, and financial NLP to forecast RKLB stock prices and generate trading signals.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Data Sources                        │
│  yfinance (OHLCV)  │  NewsAPI + Yahoo News  │  Launches │
└────────┬───────────┴────────────┬───────────┴─────┬─────┘
         │                       │                   │
         ▼                       ▼                   ▼
┌─────────────────┐   ┌──────────────────┐  ┌───────────────┐
│ Technical Indic.│   │ FinBERT Sentiment│  │ Launch Events │
│ RSI, MACD, ATR  │   │ (ProsusAI/finbert│  │ Success/Fail  │
│ BBands, Momentum│   │  CPU-friendly)   │  │ Days since    │
└────────┬────────┘   └────────┬─────────┘  └──────┬────────┘
         │                     │                    │
         └─────────────┬───────┴────────────────────┘
                        │
              ┌──────────▼──────────┐
              │   Feature Matrix    │
              │  (31 features/day)  │
              └──────┬──────┬───────┘
                     │      │
         ┌───────────┘      └────────────┐
         ▼                               ▼
┌─────────────────┐           ┌────────────────────────┐
│   LightGBM      │           │  LSTM + Multi-Head      │
│   (ML Layer)    │           │  Attention (DL Layer)   │
│                 │           │                         │
│ Walk-forward CV │           │  BiLSTM → MH-Attention  │
│ 4 price models  │           │  → LayerNorm + residual │
│ + direction clf │           │  → return + dir heads   │
└────────┬────────┘           └──────────┬──────────────┘
         │                               │
         └──────────────┬────────────────┘
                         │
               ┌──────────▼──────────┐
               │  Ensemble Layer     │
               │                     │
               │ Weighted average    │
               │ + Ridge meta-learner│
               │ + sentiment bias    │
               └──────────┬──────────┘
                           │
               ┌───────────▼──────────┐
               │   Output             │
               │ • 4-day price series │
               │ • BUY / SELL / HOLD  │
               └──────────────────────┘
```

---

## Project Structure

```
rocketlab-stock-prediction/
├── config.py                    # All paths and hyperparameters
├── requirements.txt
├── notebooks/
│   └── demo.ipynb               # End-to-end showcase notebook
├── data/
│   ├── fetch_price.py           # yfinance OHLCV (RKLB + NASDAQ)
│   ├── fetch_news.py            # NewsAPI + yfinance headlines
│   └── fetch_launches.py        # Electron launch event data
├── features/
│   ├── technical_indicators.py  # RSI, MACD, BBands, ATR, momentum
│   ├── launch_features.py       # Launch event feature engineering
│   └── preprocess.py            # Merge, scale, windowing, splits
├── models/
│   ├── finbert_sentiment.py     # FinBERT NLP scoring + cache
│   ├── lgbm_model.py            # LightGBM train/predict/evaluate
│   └── lstm_attention.py        # PyTorch LSTM + MultiheadAttention
├── ensemble/
│   └── ensemble.py              # Weighted + meta-learner ensemble
└── artifacts/                   # Saved model files (gitignored)
    ├── lgbm_model.pkl
    ├── lstm_model.pt
    ├── meta_learner.pkl
    ├── scaler.pkl
    ├── feature_cols.pkl
    └── sentiment_cache.pkl
```

---

## Techniques Used

| Technique | Why |
|-----------|-----|
| **LightGBM** with walk-forward CV | Fast, interpretable, strong on tabular data; walk-forward CV prevents temporal leakage |
| **Bidirectional LSTM** | Captures sequential dependencies in price time-series from both directions |
| **Multi-Head Self-Attention** (`nn.MultiheadAttention`) | Lets the model focus on the most relevant timesteps, not just the last hidden state |
| **FinBERT** (financial BERT) | Pre-trained for financial sentiment — far superior to rule-based VADER |
| **RobustScaler** | Less sensitive to RKLB's high-volatility outliers than MinMaxScaler |
| **Ridge meta-learner** | Learns optimal combination of base models on held-out validation data |
| **% return targets (LSTM)** | Scale-invariant targets that remain valid as price level changes over time |
| **Domain features** | Electron launch events as unique alpha signals not present in price data |

---

## Setup

```bash
pip install -r requirements.txt
```

Add your NewsAPI key to `newsapi-key.txt` in the project root (one line, no quotes).

**First run** — trains and saves all models to `artifacts/`.  
**Subsequent runs** — loads saved artifacts, skips training.

To force a full retrain (required after any code changes to model or feature logic):

```bash
rm artifacts/lstm_model.pt artifacts/lgbm_model.pkl artifacts/meta_learner.pkl
```

Then re-run `demo.ipynb` top to bottom.

```bash
cd notebooks
jupyter notebook demo.ipynb
```

---

## Hardware Requirements

Designed to run on a standard laptop (tested on Lenovo T14s):
- **CPU-only** — no GPU required
- ~8GB RAM sufficient
- FinBERT inference: ~1s/headline on CPU
- LSTM training: ~5–10 min on CPU

---

## Key Design Decisions

### Train / Val / Test Split
Data is split **strictly chronologically** — no shuffling at any stage.

| Split | Fraction | Purpose |
|-------|----------|---------|
| Train | 75% | Model training and walk-forward CV |
| Val | 10% | Early stopping only — never used to measure accuracy |
| Test | 15% | Final honest evaluation, never touched during training |

### LSTM Predicts % Returns, Not Raw Prices
The LSTM outputs **percentage returns** (e.g. `+0.03` = up 3%) rather than raw dollar prices. This is a deliberate design choice for two reasons:

1. **Scale invariance** — a model trained when RKLB traded at $5–$17 would produce nonsensical predictions when the stock later trades at $100+. Returns are always small numbers regardless of price level.
2. **Stationarity** — raw prices drift over time (non-stationary), making them harder to learn from. Returns are much closer to stationary and generalise better.

The LSTM output is converted back to dollar prices at inference time:
```
predicted_price_d+h = current_price × (1 + predicted_return_d+h)
```

LightGBM continues to predict raw prices directly, as tree-based models handle non-stationarity more robustly.

### Sentiment Forward-Fill Cap
FinBERT sentiment scores are capped at **3-day forward-fill** after the scored date. Beyond 3 days, sentiment is set to zero. This prevents stale news from a single article being treated as a signal for weeks or months.

### Sequence Length
The LSTM lookback window is **20 trading days** (~1 calendar month). A longer window of 60 days was found to produce too few training sequences (~680) given the dataset size, starving the LSTM of sufficient training data. 20 days yields ~2,040 sequences.

### Signal Thresholds
BUY/SELL signals are only triggered when the ensemble directional probability is **above 0.65 (BUY) or below 0.35 (SELL)**. The wide HOLD band (0.35–0.65) is intentional — with the inherent uncertainty in short-term price prediction, forcing a signal on marginal probabilities produces more noise than alpha.

---

## Results

| Model | MAE d+1 | RMSE d+1 | Directional Accuracy |
|-------|---------|----------|---------------------|
| LightGBM | — | — | — |
| LSTM + Attention | — | — | — |
| Ensemble (weighted) | — | — | — |
| Ensemble (meta-learner) | — | — | — |

*Results populated after running `demo.ipynb` with a clean retrain.*

---

## Changelog

### v2 — Pipeline fixes (2026-05)

A full audit of the pipeline identified and corrected the following issues:

**Critical fixes**

- **LSTM target changed from raw price to % return** (`features/preprocess.py`, `models/lstm_attention.py`)
  — The original model predicted raw dollar prices, which became invalid as RKLB's price rose
  significantly from the training era. The model now predicts percentage returns, which are
  scale-invariant. `predict()` converts returns back to dollar prices using the current close.

- **LSTM lookback reduced from 60 to 20 days** (`config.py`)
  — A 60-day lookback on a ~1,000-row dataset produced only ~680 training sequences, far too
  few for a deep learning model to learn meaningfully. Reducing to 20 days triples the sequence
  count to ~2,040.

- **Stale LSTM checkpoint invalidated** (`artifacts/lstm_model.pt`)
  — The saved model weights were trained on the original $4–$17 price regime. Loading these
  weights for inference against $100+ prices produced the observed ~50% underestimation in
  live forecasts. The checkpoint must be deleted and the model retrained after applying the
  return-target fix.

**Moderate fixes**

- **LightGBM training data leakage corrected** (`models/lgbm_model.py`)
  — The original `train()` concatenated `train_df` and `val_df` before fitting, meaning the
  final model had seen validation data before evaluation. The fix trains exclusively on
  `train_df` and uses `val_df` only for early stopping. The previously reported $14 MAE was
  artificially optimistic as a result.

- **Sentiment forward-fill capped at 3 days** (`features/preprocess.py`)
  — With only 14 days of real FinBERT scores across 988 trading days, unlimited `ffill()`
  was propagating single article scores for weeks or months, creating spurious signal.
  Forward-fill is now capped at `limit=3`; beyond that, sentiment defaults to zero.

- **BUY/SELL signal thresholds widened** (`ensemble/ensemble.py`)
  — Changed from `>0.60 / <0.40` to `>0.65 / <0.35`. With directional accuracy near the
  coin-flip level (~47%), tight thresholds caused constant signal flipping on noise. The
  wider HOLD band requires higher model confidence before committing to a direction.

---

## Disclaimer

This project is for educational and research purposes only. Nothing in this repository constitutes financial advice. Past model performance does not guarantee future returns.
