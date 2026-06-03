# CLAUDE.md — Rocketlab Stock Prediction

> Auto-generated: 2026-06-03. Source: project audit + notebook execution analysis.
> This is the project bible. Consult it before modifying any code.

---

## 1. What This Project Is

**RKLB Stock Prediction** — an ML ensemble forecasting Rocket Lab USA (RKLB) 4-day price series and generating BUY/SELL/HOLD signals.

- **31 features/day**: technical indicators, FinBERT sentiment, Electron launch events, market context
- **3-layer architecture**: LightGBM (ML) + BiLSTM+Attention (DL) → weighted average + Ridge meta-learner
- **Output**: 4-day price forecast + BUY/SELL/HOLD signal (price-derived)
- **Hardware**: CPU-only, designed for Lenovo T14s laptop (~8GB RAM)
- **Full pipeline**: ingest → train → evaluate ≈ 15–20 minutes

---

## 2. Project Structure

```
rocketlab-stock-prediction/
├── config.py                    # Single source of truth: all paths, hyperparams
├── requirements.txt             # Python deps
├── newsapi-key.txt              # NewsAPI key (one line, no quotes)
├── CLAUDE.md                    # This file
├── data/
│   ├── fetch_price.py           # yfinance OHLCV — RKLB + NASDAQ
│   ├── fetch_news.py            # NewsAPI + Yahoo Finance headlines, deduplicated
│   ├── fetch_launches.py        # Wikipedia scrape → local CSV cache (auto-refresh 7d)
│   ├── electron_launches.md     # Human-readable launch table (88 launches)
│   └── electron_launches_clean.csv  # Machine-readable launch events
├── features/
│   ├── technical_indicators.py  # RSI, MACD, BBands, ATR, volume, momentum, SMA ratios
│   ├── launch_features.py       # Launch event features (success/fail, days since, 30d count)
│   └── preprocess.py            # Merge, feature matrix, RobustScaler, chrono split, LSTM sequences
├── models/
│   ├── finbert_sentiment.py     # FinBERT NLP scoring (ProsusAI/finbert, ~440MB), disk-cached
│   ├── lgbm_model.py            # LightGBM: 4 regressors + 1 classifier, walk-forward CV
│   └── lstm_attention.py        # PyTorch BiLSTM + MultiHeadAttention + residual, multi-task
├── ensemble/
│   └── ensemble.py              # Weighted average + Ridge meta-learner, signal generation
├── notebooks/
│   └── demo.ipynb               # End-to-end showcase — the sole entry point
└── artifacts/                   # Saved model files (gitignored, delete all before clean retrain)
    ├── lgbm_model.pkl           # 5 LightGBM models (4 regressors + 1 classifier)
    ├── lstm_model.pt            # PyTorch model weights (BiLSTM + Attention)
    ├── meta_learner.pkl         # 5 Ridge models (4 price + 1 direction)
    ├── scaler.pkl               # RobustScaler fitted on training data
    ├── feature_cols.pkl         # List of feature column names
    └── sentiment_cache.pkl      # FinBERT headline → score mapping (disk cache)
```

---

## 3. Architecture

```
Data Sources: yfinance (OHLCV) + NewsAPI/Yahoo News + Electron launches (Wikipedia)
    ↓
Feature Engineering (31 features/day):
  Technical (14): RSI, MACD, BBands, ATR, momentum, volume, SMAs
  Volume (3): volume_change, volume_ma20, volume_ratio
  Returns (3): return_1d, return_5d, return_20d
  Market context (3): NASDAQ returns, RKLB/NASDAQ ratio
  Launch events (4): is_launch_day, launch_success, days_since_launch, launches_30d
  Sentiment (1): FinBERT daily aggregate, 3-day forward-fill cap
    ↓
Base Models:
  LightGBM                              |  BiLSTM + Multi-Head Attention
  - 4 regression models (d+1..d+4)      |  - 2-layer bidirectional LSTM (hidden=128)
  - 1 binary direction classifier       |  - 4-head self-attention + LayerNorm + residual
  - Walk-forward CV (5 folds, train only)|  - Price head + direction head (multi-task)
  - Targets: % returns (scale-invariant)|  - Targets: % returns (scale-invariant)
    ↓
Ensemble Layer:
  Weighted average (48% LGBM / 48% LSTM / 4% sentiment)
  + Ridge meta-learner (α=1.0, trained on held-out validation)
    ↓
Output:
  - 4-day price series (d+1 through d+4)
  - Direction probability [0..1] (evaluation only)
  - BUY / SELL / HOLD signal (derived from predicted d+1 return vs current price)
```

---

## 4. Key Configuration (config.py)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `START_DATE` | 2022-04-01 | RKLB data reliable from this date |
| `LOOKBACK_DAYS` | 20 | Was 60; 20-day = ~2,040 sequences (3× more training data) |
| `FORECAST_HORIZON` | 4 | 4-day forward forecast |
| `LGBM_PARAMS` | n_estimators=500, lr=0.05, num_leaves=31 | Conservative to avoid overfitting |
| `WALK_FORWARD_SPLITS` | 5 | TimeSeriesSplit on train only — no val leakage |
| `ENSEMBLE_WEIGHTS` | [0.48, 0.48, 0.04] | Sentiment reduced (1.7% coverage); LSTM slightly favoured |
| `LSTM_HIDDEN` | 128 | 2-layer bidirectional → 256-dim after concat |
| `ATTENTION_HEADS` | 4 | Multi-head self-attention on 256-dim embeddings |
| `DROPOUT` | 0.25 | 0.2 overfit, 0.3 underfit — tuned to current dataset size |
| `LSTM_EPOCHS` | 80 max | Early stopping `patience=20` |
| `LSTM_WEIGHT_DECAY` | 5e-4 | AdamW L2; 1e-3 was too aggressive (stopped epoch 13) |
| `LR scheduler patience` | 8 | `ReduceLROnPlateau` — was 5, too aggressive |
| `val_frac` | 0.12 | Was 0.10; +25% more val sequences for meta-learner |
| Signal BUY | predicted d+1 return > +2% | Price-derived — consistent with forecast |
| Signal SELL | predicted d+1 return < −2% | Same — no contradiction between signal and prices |

---

## 5. Key Design Decisions (DO NOT REVERSE)

### 5.1 LSTM Predicts % Returns, Not Raw Prices
The LSTM outputs **percentage returns** (e.g. `+0.03` = up 3%). Returns are scale-invariant — a model trained when RKLB was $5–$17 still produces valid predictions when the stock trades at $150+.

Conversion at inference: `predicted_price = current_price × (1 + predicted_return)`

LightGBM also predicts % returns (changed in v3 — see Bug #12). Both models now share the same scale-invariant target. `predict()` and `evaluate()` accept `current_prices` to convert to dollars.

**⚠️ The `current_prices` argument MUST be passed to `lstm_attention.predict()` and `lstm_attention.evaluate()` whenever dollar-price output is needed.** Without it, the `prices` field contains raw % returns (~0.001 scale), not dollar prices (~$150 scale). This caused the meta-learner to be trained on mismatched scales (see Bug #8 below).

### 5.2 Strict Chronological Split — No Shuffling
Data is split 75/10/15 train/val/test in strict chronological order. No shuffling at any stage. Validation is for early stopping only; test set is never touched during training.

### 5.3 Sentiment Forward-Fill Capped at 3 Days
With only ~17 days of real FinBERT scores across ~990 trading days, unlimited `ffill()` would propagate single-article scores for weeks. The cap at `limit=3` prevents stale sentiment from creating spurious signal. Beyond 3 days, sentiment defaults to zero.

### 5.4 Multi-Task LSTM with Combined Loss
`loss = MSE(price_returns) + 0.3 × BCE(direction)`. The 0.3 weight keeps direction as an auxiliary task rather than dominating optimisation.

### 5.5 RobustScaler Over MinMaxScaler
RKLB has massive volatility events (±20% single-day moves). RobustScaler (median + IQR) is far less sensitive to these outliers.

### 5.6 Dual-Source News Pipeline
NewsAPI provides broad financial/tech coverage; yfinance `.news` provides ticker-specific headlines. Both deduplicated by title. Graceful degradation if NewsAPI unavailable.

### 5.7 Multi-Head Self-Attention with Residual
`nn.MultiheadAttention` (not a weighted-average hack) with residual connection and LayerNorm: `out = LayerNorm(lstm_out + attention(lstm_out))`. The residual preserves the original LSTM signal while attention learns which timesteps matter most.

---

## 6. Bugs Fixed (DO NOT REINTRODUCE)

### Bug #1: Stale LSTM Checkpoint — Predicting $67 When RKLB at $150
- **Symptom**: Live prediction showed $67.45 for next day when current price was $134.28
- **Root cause**: `lstm_model.pt` was trained on old $4–$17 price regime with raw price targets
- **Fix**: Delete stale checkpoint, retrain with % return targets. LSTM now outputs returns → converted to dollars via `current_prices`
- **Files**: `models/lstm_attention.py` (targets), `features/preprocess.py` (make_sequences returns yp as returns), notebook cell #2 (deletes stale checkpoint)

### Bug #2: LSTM Lookback Window Too Long
- **Symptom**: Only ~680 training sequences with 60-day lookback on ~1,000 rows
- **Root cause**: `LOOKBACK_DAYS=60` meant each sequence consumed 60 rows, leaving few sequences
- **Fix**: Reduced to 20 days → ~2,040 sequences, tripling LSTM training data
- **File**: `config.py` (`LOOKBACK_DAYS`)

### Bug #3: LightGBM Data Leakage
- **Symptom**: Artificially optimistic validation metrics
- **Root cause**: `train()` concatenated train+val before fitting, so validation data leaked into training
- **Fix**: Train on train only, use val for early stopping only
- **File**: `models/lgbm_model.py` (`train()`)

### Bug #4: Sentiment Forward-Fill Unlimited
- **Symptom**: Single-article sentiment propagated for weeks/months
- **Root cause**: `ffill()` with no limit
- **Fix**: Capped at `limit=3` trading days; beyond that, sentiment = 0
- **File**: `features/preprocess.py` (`build_dataset()`)

### Bug #5: Signal Thresholds Too Tight
- **Symptom**: Constant BUY/SELL flipping on noise (directional accuracy was ~47%)
- **Root cause**: Thresholds at 0.60/0.40 gave narrow HOLD band
- **Fix**: Widened to 0.65/0.35 — requires higher model confidence before committing
- **File**: `ensemble/ensemble.py` (`weighted_predict()`, `meta_predict()`)

### Bug #6: `weighted_predict()` Return Statement Eaten by Comment
- **Symptom**: `TypeError: 'NoneType' object is not subscriptable` in ensemble evaluation
- **Root cause**: The `return` statement was on the same line as a comment: `# ...direction.    return {...}` — Python treats everything after `#` as comment
- **Fix**: Newline before `return`
- **File**: `ensemble/ensemble.py` line 55

### Bug #7: Meta-Learner Trained on Mismatched Scales
- **Symptom**: Weighted ensemble MAE $62 (25× worse than LSTM alone at $2.50)
- **Root cause**: In the ensemble training cell, `lstm_val_preds` was called WITHOUT `current_prices`, so `lstm_val_preds["prices"]` contained raw % returns (~0.001 scale) while `lgbm_val_preds["prices"]` contained dollar prices (~$150). The Ridge meta-learner was trained on these mixed scales: `Ridge([LGBM_$$$, LSTM_0.1%, sentiment]) → target_returns`
- **Fix**: (a) Pass `current_prices=val_current_prices` to LSTM val predict, (b) Convert `yp_val` (returns) to dollars via `yp_val_dollars = val_current_prices[:, np.newaxis] * (1 + yp_val)`, (c) Pass `yp_val_dollars` to `train_meta_learner()`
- **Files**: Notebook cell #18, `ensemble/ensemble.py` (`train_meta_learner()`)

### Bug #8: Live Inference Missing `current_prices`
- **Symptom**: Live prediction showed $0.02 when RKLB was at $150.23
- **Root cause**: Live cell called `lstm_attention.predict(lstm_model, X_live_seq)` without `current_prices`. LSTM `prices` contained raw returns (~0.001), not dollars. Meta-learner blended return-scale values with dollar-scale LGBM → garbage
- **Fix**: Pass `current_prices=np.array([current_price])` to LSTM live predict
- **File**: Notebook cell #24

### Bug #9: Ensemble Evaluation Compared Dollars to Returns
- **Symptom**: Would show absurd MAE numbers if ensemble output dollars but targets were returns
- **Root cause**: `ensemble_evaluate(weighted_preds, yp_test, yd_test)` — `yp_test` is % returns from `make_sequences()`, but `weighted_preds["prices"]` is now in dollars
- **Fix**: Convert to `yp_test_dollars = test_current_prices[:, np.newaxis] * (1 + yp_test)` before passing to `ensemble_evaluate()`
- **Files**: Notebook cells #18, #27

### Bug #10: Data Leakage — `target_return_d1..d4` in Feature Set
- **Symptom**: LightGBM direction CV accuracy = 1.000 ± 0.000 (impossible without leakage); test direction acc = 100%
- **Root cause**: `get_feature_columns()` excluded `target_close_d1..d4` but NOT `target_return_d1..d4`. Since `direction = sign(target_return_d1)`, LightGBM trivially split on `target_return_d1 > 0` for 100% accuracy. Price models also had access to future return values.
- **Fix**: Add `{f"target_return_d{h}" for h in range(1, FORECAST_HORIZON + 1)}` to the exclude set in `get_feature_columns()`. Feature count: 35 → 31.
- **File**: `features/preprocess.py` (`get_feature_columns()`)
- **Note**: LSTM was not affected — its sliding window only sees past values of these columns (not the current-row future value being predicted).

### Bug #11: LightGBM Bug #3 Re-introduced (CV on train+val)
- **Symptom**: CV RMSE std dev ($7.5) > mean ($6.4) — CV metric unreliable; val data leaked into some folds
- **Root cause**: `train()` used `combined = pd.concat([train_df, val_df])` for `TimeSeriesSplit`, meaning later folds contained val-era data in training
- **Fix**: CV now runs on `train_df` only; final models still trained on `combined` (correct practice)
- **File**: `models/lgbm_model.py` (`train()`)

### Bug #12: LightGBM Raw Price Targets Break Across Regimes
- **Symptom**: LGBM d+1 MAE $14.48 vs LSTM $3.38 (4× gap); flat degradation d+1→d+4 (only $3 difference)
- **Root cause**: LGBM trained on $5–$25 raw prices but tested on $50–$150. Tree splits on absolute price thresholds don't extrapolate.
- **Fix**: LGBM now predicts `target_return_d{h}` (% returns) — same scale-invariant approach as LSTM. `predict()`/`evaluate()` accept `current_prices` param for dollar conversion.
- **Files**: `models/lgbm_model.py` (targets, predict, evaluate); notebook cells #12, #18, #24

---

## 7. Scale Consistency Rules (CRITICAL)

The entire pipeline must maintain consistent dollar scaling. These invariants must hold:

1. **`lstm_attention.predict(model, X, current_prices=...)`** — ALWAYS pass `current_prices` when you need dollar output. Without it, `prices` = raw returns (~0.001 scale).
2. **`lstm_attention.evaluate(model, X, y_return, y_dir, current_prices=...)`** — same rule.
3. **`train_meta_learner(lgbm_preds, lstm_preds, y_price, y_dir)`** — `lgbm_preds["prices"]`, `lstm_preds["prices"]`, and `y_price` must ALL be in the same scale (dollars or returns — pick one and be consistent). The notebook uses dollars.
4. **`ensemble_evaluate(ensemble_preds, y_price_test, y_dir_test)`** — `ensemble_preds["prices"]` and `y_price_test` must be in the same scale.
5. **`make_sequences()` returns % returns for `yp`** — always convert to dollars before using as price targets in ensemble/meta-learner context.

---

## 8. Current Performance (v3 retrain, 2026-06-03)

Data split: Train 723 rows (2022-06-13 → 2025-04-30) / Val 120 rows / Test 149 rows (2025-10-22 → 2026-05-27).
Test period context: RKLB went from ~$20–30 → ~$150 — strong momentum/bull regime not seen in training.

| Model | d+1 MAE | d+2 MAE | d+3 MAE | d+4 MAE | Direction Acc |
|---|---|---|---|---|---|
| **LSTM + Attention** | **$3.79** | $4.79 | $6.50 | $7.32 | 43.4% ⚠️ |
| LightGBM | $4.04 | $5.33 | $7.51 | $8.34 | 43.6% ⚠️ |
| Meta-Learner Ensemble | $4.31 | $5.75 | $8.11 | $8.78 | 44.2% ⚠️ |
| Weighted Ensemble | $5.19 | $6.58 | $7.93 | $8.57 | 42.6% ⚠️ |

**LightGBM CV (train-only):** d+1: 4.51% ± 1.21% | d+2: 6.23% ± 1.37% | d+3: 7.68% ± 1.85% | d+4: 8.85% ± 2.17% | Direction: 0.530 ± 0.027

**LSTM training:** Stopped at epoch 16, best val loss 0.2142 (train 0.2124 — gap closed vs previous 13:1 ratio).

**Live (2026-05-27, $150.23):** $146.13 / $143.36 / $136.75 / $138.99 — Signal: **SELL** (consistent with price forecast ✅)

**Direction accuracy note:** All models score 42–44% — below random (50%). Root cause: models trained on 2022–2025 mean-reverting $4–$17 RKLB, tested on 2025–2026 momentum bull run to $150. Direction head is no longer used for signal generation (fixed v3). This will self-correct as post-rally data accumulates in training.

---

## 9. How to Run

```bash
pip install -r requirements.txt
echo "your-newsapi-key" > newsapi-key.txt
rm -rf artifacts/*           # Clean slate — always before fresh run
jupyter notebook notebooks/demo.ipynb   # Run top to bottom
```

**First run**: Downloads FinBERT (~440MB), scrapes Electron launches, trains all models, saves to `artifacts/`.  
**Subsequent runs**: Loads saved artifacts, skips training.  
**Force retrain**: Delete specific artifacts or `rm -rf artifacts/*`.

The notebook is the **sole entry point** — no standalone `.py` scripts need to be run first. All 15 `.py` files are imported as library modules.

---

## 10. Dependencies

| Package | Purpose |
|---------|---------|
| `yfinance` | OHLCV data for RKLB + NASDAQ |
| `lightgbm` | Gradient-boosted tree models |
| `torch` | BiLSTM + Attention neural network |
| `transformers` | FinBERT sentiment model (ProsusAI/finbert) |
| `scikit-learn` | Ridge meta-learner, RobustScaler, metrics |
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `joblib` | Model serialisation |
| `requests` | NewsAPI HTTP calls |

---

## 11. Notebook Cell Map

| Cell # | Section | Purpose | Status |
|--------|---------|---------|--------|
| #2 | 0. Setup | Imports, deletes **all** stale artifacts | ✅ Run |
| #4 | 1. Data Ingestion | yfinance + NewsAPI + launches | ✅ Run |
| #5 | 1. Plot | Price history with launch overlay | ✅ Run |
| #7 | 2. Feature Engineering | FinBERT + build_dataset (31 features) | ✅ Run |
| #8 | 2. Plot | Technical indicators | ✅ Run |
| #10 | 3. Split | Train 723 / Val 120 / Test 149 | ✅ Run |
| #12 | 4. LightGBM | Train LGBM (% return targets, CV on train) | ✅ Run |
| #13 | 4. Plot | Feature importance | ✅ Run |
| #15 | 5. LSTM | Train LSTM (stopped epoch 16) | ✅ Run |
| #16 | 5. Plot | LSTM vs actual (dollar scale) | ✅ Run |
| #18 | 6. Ensemble | Weighted + meta-learner + eval (dollar targets) | ✅ Run |
| #19 | 6. Plot | 4-day forecast comparison | ✅ Run |
| #21 | 7. Direction | Classification report | ✅ Run |
| #22 | 7. Plot | Signal timeline (dollar scale, price-derived markers) | ✅ Run |
| #24 | 8. Live | Live 4-day forecast + SELL signal | ✅ Run |
| #25 | 8. Plot | Live forecast chart | ✅ Run |
| #27 | 9. Summary | Performance summary table | ✅ Run |

---

## 12. Related Projects

- `../allied-health-nudge/` — Primary portfolio project (MLOps pipeline)
- `../Stock-Management/` — Bursa Malaysia stock analysis
- `../CI-CD-Plan/` — Future: operationalise this pipeline with GitHub Actions

---

## 13. Tags

`#ml` `#stock-prediction` `#lightgbm` `#lstm` `#attention` `#finbert` `#nlp` `#sentiment` `#ensemble` `#rklb` `#time-series` `#launch-events` `#yfinance` `#portfolio-project`
