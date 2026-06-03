"""
Ensemble layer — combines LightGBM and LSTM predictions.
Sentiment score adjusts the directional signal as a soft bias.

Weighted average ensemble (weights tunable in config.py):
  final_price = w_lgbm * lgbm_price + w_lstm * lstm_price
  final_dir   = sigmoid(w_lgbm * lgbm_dir_logit + w_lstm * lstm_dir_logit + w_sent * sentiment)

Also provides a meta-learner variant (Ridge regression on held-out val predictions).
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

from config import ENSEMBLE_WEIGHTS, FORECAST_HORIZON, ARTIFACTS_DIR, SIGNAL_BUY_THRESHOLD, SIGNAL_SELL_THRESHOLD

META_LEARNER_PATH = ARTIFACTS_DIR / "meta_learner.pkl"


def weighted_predict(
    lgbm_preds: dict,
    lstm_preds: dict,
    sentiment_scores: np.ndarray | None = None,
    weights: list[float] = ENSEMBLE_WEIGHTS,
    current_prices: np.ndarray | None = None,
) -> dict:
    """
    Args:
        lgbm_preds       : output of lgbm_model.predict()
        lstm_preds       : output of lstm_attention.predict()
        sentiment_scores : (N,) daily sentiment in [-1, 1]; None → zeros
        weights          : [w_lgbm, w_lstm, w_sentiment]
        current_prices   : (N,) today's close. If provided, signal is derived from
                           predicted d+1 price vs current price (consistent with forecast).
                           If None, falls back to direction-probability thresholds.

    Returns:
        'prices'    : (N, horizon) weighted average price predictions
        'direction' : (N,) blended up-probability (kept for evaluation)
        'signal'    : (N,) str array — 'BUY' / 'SELL' / 'HOLD'
    """
    w_lgbm, w_lstm, w_sent = weights
    if sentiment_scores is None:
        sentiment_scores = np.zeros(len(lgbm_preds["prices"]))

    prices = w_lgbm * lgbm_preds["prices"] + w_lstm * lstm_preds["prices"]

    # Blend directional probabilities; sentiment shifts the midpoint (kept for evaluation)
    dir_blend = w_lgbm * lgbm_preds["direction"] + w_lstm * lstm_preds["direction"]
    sent_adj = w_sent * (sentiment_scores * 0.5 + 0.5)  # map [-1,1] → [0,1]
    direction = np.clip(dir_blend + sent_adj - w_sent * 0.5, 0, 1)

    if current_prices is not None:
        # Derive signal from price forecast so it is always consistent with predictions
        pred_return_d1 = (prices[:, 0] - current_prices) / current_prices
        signal = np.where(pred_return_d1 > SIGNAL_BUY_THRESHOLD, "BUY",
                          np.where(pred_return_d1 < SIGNAL_SELL_THRESHOLD, "SELL", "HOLD"))
    else:
        signal = np.where(direction > 0.65, "BUY", np.where(direction < 0.35, "SELL", "HOLD"))

    return {"prices": prices, "direction": direction, "signal": signal}


def train_meta_learner(
    lgbm_val_preds: dict,
    lstm_val_preds: dict,
    y_price_val: np.ndarray,
    y_dir_val: np.ndarray,
    sentiment_val: np.ndarray | None = None,
) -> Ridge:
    """
    Fits a Ridge meta-learner on validation-set predictions from both base models.
    Trains one Ridge per horizon day for prices, one for direction.
    Saves to disk.
    """
    if sentiment_val is None:
        sentiment_val = np.zeros(len(y_dir_val))

    meta_models = {}
    for h in range(FORECAST_HORIZON):
        X_meta = np.column_stack([
            lgbm_val_preds["prices"][:, h],
            lstm_val_preds["prices"][:, h],
            sentiment_val,
        ])
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_meta, y_price_val[:, h])
        meta_models[f"price_d{h+1}"] = ridge

    X_dir = np.column_stack([
        lgbm_val_preds["direction"],
        lstm_val_preds["direction"],
        sentiment_val,
    ])
    dir_ridge = Ridge(alpha=1.0)
    dir_ridge.fit(X_dir, y_dir_val.astype(float))
    meta_models["direction"] = dir_ridge

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta_models, META_LEARNER_PATH)
    print(f"  Meta-learner saved to {META_LEARNER_PATH}")
    return meta_models


def meta_predict(
    lgbm_preds: dict,
    lstm_preds: dict,
    sentiment_scores: np.ndarray | None = None,
    current_prices: np.ndarray | None = None,
) -> dict:
    """
    current_prices: (N,) today's close. If provided, signal is derived from predicted
                    d+1 price vs current price. If None, falls back to direction thresholds.
    """
    meta_models = joblib.load(META_LEARNER_PATH)
    if sentiment_scores is None:
        sentiment_scores = np.zeros(len(lgbm_preds["prices"]))

    prices = np.column_stack([
        meta_models[f"price_d{h+1}"].predict(
            np.column_stack([lgbm_preds["prices"][:, h], lstm_preds["prices"][:, h], sentiment_scores])
        )
        for h in range(FORECAST_HORIZON)
    ])
    X_dir = np.column_stack([lgbm_preds["direction"], lstm_preds["direction"], sentiment_scores])
    direction = np.clip(meta_models["direction"].predict(X_dir), 0, 1)

    if current_prices is not None:
        pred_return_d1 = (prices[:, 0] - current_prices) / current_prices
        signal = np.where(pred_return_d1 > SIGNAL_BUY_THRESHOLD, "BUY",
                          np.where(pred_return_d1 < SIGNAL_SELL_THRESHOLD, "SELL", "HOLD"))
    else:
        signal = np.where(direction > 0.65, "BUY", np.where(direction < 0.35, "SELL", "HOLD"))

    return {"prices": prices, "direction": direction, "signal": signal}


def evaluate(ensemble_preds: dict, y_price_test: np.ndarray, y_dir_test: np.ndarray) -> pd.DataFrame:
    rows = []
    for h in range(FORECAST_HORIZON):
        mae = mean_absolute_error(y_price_test[:, h], ensemble_preds["prices"][:, h])
        rmse = np.sqrt(mean_squared_error(y_price_test[:, h], ensemble_preds["prices"][:, h]))
        rows.append({"horizon": f"d+{h+1}", "MAE_$": round(mae, 4), "RMSE_$": round(rmse, 4)})
    dir_acc = accuracy_score(y_dir_test, (ensemble_preds["direction"] > 0.5).astype(int))
    rows.append({"horizon": "direction", "MAE_$": None, "RMSE_$": None, "Accuracy": round(dir_acc, 4)})
    return pd.DataFrame(rows)
