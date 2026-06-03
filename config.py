"""
Central configuration for the RKLB prediction pipeline.
Edit paths and hyperparameters here — nothing else needs changing.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
NOTEBOOKS_DIR = ROOT / "notebooks"

NEWSAPI_KEY_PATH = ROOT / "newsapi-key.txt"
LAUNCHES_REFRESH_DAYS = 7          # re-scrape Wikipedia if CSV is older than this

# Pre-downloaded CSVs (from existing GitHub repo assets)
RKLB_CSV = DATA_DIR / "RKLBhistoricalstockprice.csv"
NASDAQ_CSV = DATA_DIR / "NASDAQhistoricaldata.csv"
LAUNCHES_CSV = DATA_DIR / "electron_launches_clean.csv"
SENTIMENT_CACHE = ARTIFACTS_DIR / "sentiment_cache.pkl"

# Saved model artifacts
LGBM_MODEL_PATH = ARTIFACTS_DIR / "lgbm_model.pkl"
LSTM_MODEL_PATH = ARTIFACTS_DIR / "lstm_model.pt"
META_LEARNER_PATH = ARTIFACTS_DIR / "meta_learner.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_COLS_PATH = ARTIFACTS_DIR / "feature_cols.pkl"

# ── Ticker / date ──────────────────────────────────────────────────────────
TICKER = "RKLB"
NASDAQ_TICKER = "^IXIC"
START_DATE = "2022-04-01"          # RKLB IPO was Oct 2021, data reliable from Apr 2022

# ── Feature engineering ────────────────────────────────────────────────────
LOOKBACK_DAYS = 20                 # sequence length for LSTM
FORECAST_HORIZON = 4               # days ahead to predict
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BBAND_PERIOD = 20
ATR_PERIOD = 14

# ── LightGBM ───────────────────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}
WALK_FORWARD_SPLITS = 5            # number of walk-forward CV folds

# ── LSTM ───────────────────────────────────────────────────────────────────
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
ATTENTION_HEADS = 4
DROPOUT = 0.25                     # 0.2 overfit badly, 0.3 underfit (stopped epoch 13) — split diff
LSTM_EPOCHS = 80
LSTM_BATCH = 32
LSTM_LR = 1e-3
LSTM_PATIENCE = 20                 # early stopping patience
LSTM_WEIGHT_DECAY = 5e-4           # AdamW L2 regularization; 1e-3 was too aggressive

# ── Ensemble ───────────────────────────────────────────────────────────────
# Weights for weighted-average ensemble (lgbm, lstm, sentiment_adjustment)
# Sentiment weight reduced: only 17/992 trading days have real scores (1.7% coverage)
ENSEMBLE_WEIGHTS = [0.06, 0.90, 0.04]

# Signal thresholds — derived from predicted d+1 price vs current price (% return)
# Using price-based signal so BUY/SELL/HOLD is always consistent with the price forecast
SIGNAL_BUY_THRESHOLD  =  0.02     # predicted d+1 return > +2% → BUY
SIGNAL_SELL_THRESHOLD = -0.02     # predicted d+1 return < -2% → SELL

# ── News / sentiment ───────────────────────────────────────────────────────
NEWSAPI_MAX_ARTICLES = 100         # per request
NEWSAPI_LOOKBACK_DAYS = 30        # how far back to fetch news on each run
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_BATCH = 16                 # headlines per inference batch
