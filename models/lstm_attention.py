"""
PyTorch LSTM + Multi-Head Attention model for RKLB price prediction.

Architecture:
  Input (batch, seq_len, n_features)
    → 2-layer Bidirectional LSTM
    → Multi-Head Self-Attention (proper nn.MultiheadAttention, not a hack)
    → LayerNorm + residual
    → Linear head → 4-day price sequence + direction logit

Outputs:
  prices    : (batch, FORECAST_HORIZON)  — next 4 days of close prices
  direction : (batch, 1)                 — logit for up/down day+1
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

from config import (
    LSTM_HIDDEN, LSTM_LAYERS, ATTENTION_HEADS, DROPOUT,
    LSTM_EPOCHS, LSTM_BATCH, LSTM_LR, LSTM_PATIENCE, LSTM_WEIGHT_DECAY,
    FORECAST_HORIZON, LSTM_MODEL_PATH, ARTIFACTS_DIR,
)


class LSTMAttentionModel(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_layers: int, heads: int, dropout: float, horizon: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )
        # Bidirectional doubles hidden dim
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden * 2,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(dropout)

        self.price_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon),
        )
        self.dir_head = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, n_features)
        lstm_out, _ = self.lstm(x)                          # (batch, seq, hidden*2)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)  # self-attention
        out = self.norm(lstm_out + attn_out)                 # residual + norm
        out = self.dropout(out)
        ctx = out[:, -1, :]                                  # last timestep context
        prices = self.price_head(ctx)                        # (batch, horizon)
        direction = self.dir_head(ctx)                       # (batch, 1)
        return prices, direction


def _make_tensors(X, y_price, y_dir):
    return (
        torch.FloatTensor(X),
        torch.FloatTensor(y_price),
        torch.FloatTensor(y_dir).unsqueeze(1),
    )


def train(
    X_train: np.ndarray, y_price_train: np.ndarray, y_dir_train: np.ndarray,
    X_val: np.ndarray, y_price_val: np.ndarray, y_dir_val: np.ndarray,
) -> LSTMAttentionModel:
    """
    Trains the model with early stopping on validation loss.
    Saves the best checkpoint to LSTM_MODEL_PATH.
    """
    n_features = X_train.shape[2]
    model = LSTMAttentionModel(
        n_features=n_features,
        hidden=LSTM_HIDDEN,
        n_layers=LSTM_LAYERS,
        heads=ATTENTION_HEADS,
        dropout=DROPOUT,
        horizon=FORECAST_HORIZON,
    )

    train_ds = TensorDataset(*_make_tensors(X_train, y_price_train, y_dir_train))
    val_ds = TensorDataset(*_make_tensors(X_val, y_price_val, y_dir_val))
    train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=LSTM_BATCH)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LSTM_LR, weight_decay=LSTM_WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
    price_loss_fn = nn.MSELoss()
    dir_loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    patience_counter = 0
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for Xb, yp, yd in train_loader:
            optimizer.zero_grad()
            pred_p, pred_d = model(Xb)
            loss = price_loss_fn(pred_p, yp) + 0.3 * dir_loss_fn(pred_d, yd)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yp, yd in val_loader:
                pred_p, pred_d = model(Xb)
                val_loss += (price_loss_fn(pred_p, yp) + 0.3 * dir_loss_fn(pred_d, yd)).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if epoch % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f} lr={lr:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), LSTM_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= LSTM_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(LSTM_MODEL_PATH, weights_only=True))
    print(f"  Best val loss: {best_val:.4f} — saved to {LSTM_MODEL_PATH}")
    return model


def load(n_features: int) -> LSTMAttentionModel:
    model = LSTMAttentionModel(
        n_features=n_features,
        hidden=LSTM_HIDDEN,
        n_layers=LSTM_LAYERS,
        heads=ATTENTION_HEADS,
        dropout=DROPOUT,
        horizon=FORECAST_HORIZON,
    )
    model.load_state_dict(torch.load(LSTM_MODEL_PATH, weights_only=True))
    model.eval()
    return model


def predict(
    model: LSTMAttentionModel,
    X: np.ndarray,
    current_prices: np.ndarray | None = None,
) -> dict:
    """
    Args:
        X             : (N, seq_len, n_features) numpy array
        current_prices: (N,) array of today's close price for each sample.
                        Used to convert predicted % returns back to dollar prices.
                        If None, returns raw % returns in the 'prices' field.

    Returns dict with:
        'prices'    : (N, horizon) predicted dollar prices (or raw returns if no current_prices)
        'returns'   : (N, horizon) raw predicted % returns (always present)
        'direction' : (N,) up-probabilities [0..1]
    """
    model.eval()
    with torch.no_grad():
        Xt = torch.FloatTensor(X)
        pred_r, pred_d = model(Xt)          # pred_r = predicted % returns
        returns = pred_r.numpy()
        direction = torch.sigmoid(pred_d).squeeze(1).numpy()

    if current_prices is not None:
        # Convert: price_d+h = today_price * (1 + return_d+h)
        prices = current_prices[:, np.newaxis] * (1 + returns)
    else:
        prices = returns  # caller will handle conversion

    return {"prices": prices, "returns": returns, "direction": direction}


def evaluate(
    model: LSTMAttentionModel,
    X_test: np.ndarray,
    y_return_test: np.ndarray,   # renamed: these are now % returns, not raw prices
    y_dir_test: np.ndarray,
    current_prices: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    y_return_test : (N, horizon) — the target_return_d{h} values from make_sequences()
    current_prices: (N,) — today's close for each test row, for converting back to $
    """
    preds = predict(model, X_test, current_prices=current_prices)
    rows = []

    if current_prices is not None:
        # Evaluate in dollar terms so numbers are human-readable
        y_price_actual = current_prices[:, np.newaxis] * (1 + y_return_test)
        for h in range(FORECAST_HORIZON):
            mae = mean_absolute_error(y_price_actual[:, h], preds["prices"][:, h])
            rmse = np.sqrt(mean_squared_error(y_price_actual[:, h], preds["prices"][:, h]))
            rows.append({"horizon": f"d+{h+1}", "MAE_$": round(mae, 4), "RMSE_$": round(rmse, 4)})
    else:
        # Evaluate in return terms (MAE of 0.02 = wrong by 2% on average)
        for h in range(FORECAST_HORIZON):
            mae = mean_absolute_error(y_return_test[:, h], preds["returns"][:, h])
            rmse = np.sqrt(mean_squared_error(y_return_test[:, h], preds["returns"][:, h]))
            rows.append({"horizon": f"d+{h+1}", "MAE_%": round(mae * 100, 4), "RMSE_%": round(rmse * 100, 4)})

    dir_acc = accuracy_score(y_dir_test, (preds["direction"] > 0.5).astype(int))
    rows.append({"horizon": "direction", "MAE_$": None, "RMSE_$": None, "Accuracy": round(dir_acc, 4)})
    return pd.DataFrame(rows)
