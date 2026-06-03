"""
LightGBM layer — tabular ML model for RKLB price prediction.

Models predict % returns (scale-invariant across all price regimes).
Walk-forward CV runs on train_df only; final models train on train+val.
Pass current_prices to predict()/evaluate() to get dollar output.
"""

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

from config import (
    LGBM_PARAMS, WALK_FORWARD_SPLITS, FORECAST_HORIZON,
    LGBM_MODEL_PATH, ARTIFACTS_DIR,
)


def _regression_params():
    p = LGBM_PARAMS.copy()
    p["objective"] = "regression"
    p["metric"] = "rmse"
    return p


def _classification_params():
    p = LGBM_PARAMS.copy()
    p["objective"] = "binary"
    p["metric"] = "binary_logloss"
    return p


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """
    CV on train_df only (no val leakage). Final models trained on train+val.
    Targets are % returns so predictions stay valid across price regimes.
    """
    combined = pd.concat([train_df, val_df])
    tscv = TimeSeriesSplit(n_splits=WALK_FORWARD_SPLITS)
    models = {}

    # ── Price models (one per horizon, predicts % return) ─────────────────
    for h in range(1, FORECAST_HORIZON + 1):
        target = f"target_return_d{h}"
        X_cv = train_df[feature_cols].values
        y_cv = train_df[target].values

        cv_scores = []
        for tr_idx, va_idx in tscv.split(X_cv):
            m = lgb.LGBMRegressor(**_regression_params())
            m.fit(
                X_cv[tr_idx], y_cv[tr_idx],
                eval_set=[(X_cv[va_idx], y_cv[va_idx])],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            cv_scores.append(np.sqrt(mean_squared_error(y_cv[va_idx], m.predict(X_cv[va_idx]))))

        final = lgb.LGBMRegressor(**_regression_params())
        final.fit(combined[feature_cols].values, combined[target].values,
                  callbacks=[lgb.log_evaluation(-1)])
        models[f"price_d{h}"] = final
        print(f"  LightGBM price_d{h} — CV RMSE: {np.mean(cv_scores)*100:.3f}% ± {np.std(cv_scores)*100:.3f}%")

    # ── Direction classifier ───────────────────────────────────────────────
    X_cv = train_df[feature_cols].values
    y_dir_cv = train_df["direction"].values

    dir_scores = []
    for tr_idx, va_idx in tscv.split(X_cv):
        clf = lgb.LGBMClassifier(**_classification_params())
        clf.fit(
            X_cv[tr_idx], y_dir_cv[tr_idx],
            eval_set=[(X_cv[va_idx], y_dir_cv[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        dir_scores.append(accuracy_score(y_dir_cv[va_idx], clf.predict(X_cv[va_idx])))

    final_clf = lgb.LGBMClassifier(**_classification_params())
    final_clf.fit(combined[feature_cols].values, combined["direction"].values,
                  callbacks=[lgb.log_evaluation(-1)])
    models["direction"] = final_clf
    print(f"  LightGBM direction — CV Accuracy: {np.mean(dir_scores):.3f} ± {np.std(dir_scores):.3f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, LGBM_MODEL_PATH)
    print(f"  Saved to {LGBM_MODEL_PATH}")
    return models


def load() -> dict:
    return joblib.load(LGBM_MODEL_PATH)


def predict(
    models: dict,
    X: np.ndarray,
    current_prices: np.ndarray | None = None,
) -> dict:
    """
    Args:
        models        : dict from train() or load()
        X             : (N, n_features) feature matrix
        current_prices: (N,) today's close. If provided, converts predicted
                        % returns to dollar prices. If None, 'prices' = raw returns.

    Returns dict with:
        'prices'    : (N, 4) dollar prices (or raw returns if current_prices is None)
        'returns'   : (N, 4) raw predicted % returns
        'direction' : (N,) up-probability [0..1]
    """
    return_preds = np.column_stack([
        models[f"price_d{h}"].predict(X) for h in range(1, FORECAST_HORIZON + 1)
    ])
    dir_prob = models["direction"].predict_proba(X)[:, 1]

    if current_prices is not None:
        prices = current_prices[:, np.newaxis] * (1 + return_preds)
    else:
        prices = return_preds

    return {"prices": prices, "returns": return_preds, "direction": dir_prob}


def evaluate(
    models: dict,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    current_prices: np.ndarray | None = None,
) -> pd.DataFrame:
    X = test_df[feature_cols].values
    preds = predict(models, X, current_prices=current_prices)

    rows = []
    if current_prices is not None:
        for h in range(1, FORECAST_HORIZON + 1):
            y_true = current_prices * (1 + test_df[f"target_return_d{h}"].values)
            mae = mean_absolute_error(y_true, preds["prices"][:, h - 1])
            rmse = np.sqrt(mean_squared_error(y_true, preds["prices"][:, h - 1]))
            rows.append({"horizon": f"d+{h}", "MAE_$": round(mae, 4), "RMSE_$": round(rmse, 4)})
    else:
        for h in range(1, FORECAST_HORIZON + 1):
            y_true = test_df[f"target_return_d{h}"].values
            mae = mean_absolute_error(y_true, preds["returns"][:, h - 1])
            rmse = np.sqrt(mean_squared_error(y_true, preds["returns"][:, h - 1]))
            rows.append({"horizon": f"d+{h}", "MAE_%": round(mae * 100, 4), "RMSE_%": round(rmse * 100, 4)})

    dir_acc = accuracy_score(test_df["direction"].values, (preds["direction"] > 0.5).astype(int))
    rows.append({"horizon": "direction", "MAE_$": None, "RMSE_$": None, "Accuracy": round(dir_acc, 4)})
    return pd.DataFrame(rows)


def feature_importance(models: dict, feature_cols: list[str]) -> pd.DataFrame:
    """Returns feature importance averaged across all 4 price models."""
    importances = np.column_stack([
        models[f"price_d{h}"].feature_importances_ for h in range(1, FORECAST_HORIZON + 1)
    ]).mean(axis=1)
    return (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
