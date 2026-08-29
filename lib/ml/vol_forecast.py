"""
ml/vol_forecast.py
Machine-learning volatility forecasting.

Models
------
HAR         — Heterogeneous AR model (Corsi 2009); OLS; interpretable baseline
GBM         — Gradient Boosting regressor (sklearn); 50 features; best practical model
LSTM        — PyTorch sequence model; good for non-linear temporal patterns

HAR model (Corsi 2009):
    RV_t = β₀ + β₁ RV_{t-1}  +  β₂ RV̄_{t-5}  +  β₃ RV̄_{t-22}  +  ε_t

Features for ML models:
    - Lagged RV at [1, 2, 3, 5, 10, 22] days
    - Rolling means [5, 10, 22, 66] days
    - Rolling std [5, 22] days
    - Week-of-month, month, weekday dummies (optional)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# ── feature engineering ───────────────────────────────────────────────────────

def build_har_features(rv: np.ndarray, lag_daily: int = 1,
                        lag_weekly: int = 5, lag_monthly: int = 22) -> tuple:
    """
    Build HAR feature matrix.

    Returns
    -------
    X : (N-lag_monthly, 3) — daily/weekly/monthly lagged RVs
    y : (N-lag_monthly,)   — target RV
    """
    rv   = np.asarray(rv, dtype=float)
    T    = len(rv)
    start = lag_monthly

    y    = rv[start:]
    x_d  = rv[start - lag_daily  : T - lag_daily][:len(y)]
    x_w  = np.array([rv[i - lag_weekly  : i].mean() for i in range(start, T)])
    x_m  = np.array([rv[i - lag_monthly : i].mean() for i in range(start, T)])

    X = np.column_stack([x_d, x_w, x_m])
    return X, y


def build_ml_features(rv: np.ndarray, lags=(1, 2, 3, 5, 10, 22),
                       windows=(5, 10, 22, 66)) -> tuple:
    """
    Build a richer ML feature matrix from realized volatility series.

    Returns
    -------
    X : (N - max_lag, n_features)
    y : (N - max_lag,)
    """
    rv      = np.asarray(rv, dtype=float)
    max_lag = max(max(lags), max(windows))
    T       = len(rv)
    n_out   = T - max_lag
    cols    = []

    # Lagged RV
    for lag in lags:
        col = rv[max_lag - lag : T - lag][:n_out]
        cols.append(col)

    # Rolling means
    for w in windows:
        col = np.array([rv[i - w : i].mean() for i in range(max_lag, T)])[:n_out]
        cols.append(col)

    # Rolling std
    for w in [5, 22]:
        col = np.array([rv[i - w : i].std() for i in range(max_lag, T)])[:n_out]
        cols.append(col)

    # Log RV (more Gaussian)
    for lag in [1, 5, 22]:
        col = np.log(rv[max_lag - lag : T - lag][:n_out] + 1e-10)
        cols.append(col)

    X = np.column_stack(cols)
    y = rv[max_lag : max_lag + n_out]
    return X, y


# ── model results ─────────────────────────────────────────────────────────────

@dataclass
class ForecastResult:
    model_name:  str
    predictions: np.ndarray     # out-of-sample predicted RV
    actuals:     np.ndarray     # actual RV in test window
    train_size:  int
    test_size:   int
    rmse:        float
    mae:         float
    qlike:       float          # QLIKE loss: mean(RV/RV_hat - log(RV/RV_hat) - 1)
    feature_names: list[str] = field(default_factory=list)
    feature_importances: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return (
            f"{self.model_name:<18} "
            f"RMSE={self.rmse:.6f}  MAE={self.mae:.6f}  QLIKE={self.qlike:.4f}"
        )

    @property
    def r2(self) -> float:
        ss_res = np.sum((self.actuals - self.predictions)**2)
        ss_tot = np.sum((self.actuals - self.actuals.mean())**2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _qlike(y_true, y_pred):
    """QLIKE loss — standard in vol forecasting literature."""
    eps   = 1e-10
    ratio = y_true / (y_pred + eps)
    return float(np.mean(ratio - np.log(ratio + eps) - 1))


# ── HAR ───────────────────────────────────────────────────────────────────────

@dataclass
class HARModel:
    """Heterogeneous AR model (Corsi 2009)."""
    coefficients_: Optional[np.ndarray] = None
    intercept_: float = 0.0
    lag_daily:  int = 1
    lag_weekly: int = 5
    lag_monthly:int = 22
    _reg: Optional[LinearRegression] = field(default=None, repr=False)

    def fit(self, rv: np.ndarray) -> "HARModel":
        X, y = build_har_features(rv, self.lag_daily, self.lag_weekly, self.lag_monthly)
        self._reg = LinearRegression()
        self._reg.fit(X, y)
        self.coefficients_ = self._reg.coef_
        self.intercept_    = self._reg.intercept_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._reg.predict(X)

    def forecast(self, rv: np.ndarray, h: int = 1) -> float:
        """One-step or h-step ahead forecast from latest observation."""
        n = self.lag_monthly
        x_d = rv[-self.lag_daily]
        x_w = rv[-self.lag_weekly:].mean()
        x_m = rv[-n:].mean()
        return float(self._reg.predict([[x_d, x_w, x_m]])[0])

    def summary(self) -> str:
        if self.coefficients_ is None:
            return "HARModel (not fitted)"
        c = self.coefficients_
        return (
            f"HAR Model\n"
            f"  intercept = {self.intercept_:.6f}\n"
            f"  β_daily   = {c[0]:.4f}\n"
            f"  β_weekly  = {c[1]:.4f}\n"
            f"  β_monthly = {c[2]:.4f}"
        )


# ── training / evaluation ─────────────────────────────────────────────────────

def train_har(rv: np.ndarray, test_fraction: float = 0.2) -> ForecastResult:
    """Train and evaluate HAR model with a fixed train/test split."""
    rv   = np.asarray(rv, dtype=float)
    X, y = build_har_features(rv)
    n    = len(y)
    n_tr = int(n * (1 - test_fraction))

    X_tr, X_te = X[:n_tr], X[n_tr:]
    y_tr, y_te = y[:n_tr], y[n_tr:]

    model = HARModel().fit(rv[:n_tr + 22])   # fit on training RV
    preds = model._reg.predict(X_te)
    preds = np.maximum(preds, 1e-10)

    return ForecastResult(
        model_name="HAR",
        predictions=preds,
        actuals=y_te,
        train_size=n_tr,
        test_size=len(y_te),
        rmse=float(np.sqrt(mean_squared_error(y_te, preds))),
        mae=float(np.mean(np.abs(y_te - preds))),
        qlike=_qlike(y_te, preds),
        feature_names=["RV_daily", "RV_weekly", "RV_monthly"],
        feature_importances=model.coefficients_,
    )


def train_gbm(rv: np.ndarray, test_fraction: float = 0.2,
              n_estimators: int = 200, max_depth: int = 3,
              learning_rate: float = 0.05, seed: int = 42) -> ForecastResult:
    """Train and evaluate Gradient Boosting vol forecaster."""
    rv   = np.asarray(rv, dtype=float)
    X, y = build_ml_features(rv)
    n    = len(y)
    n_tr = int(n * (1 - test_fraction))

    X_tr, X_te = X[:n_tr], X[n_tr:]
    y_tr, y_te = y[:n_tr], y[n_tr:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        subsample=0.8,
    )
    model.fit(X_tr_s, np.log(y_tr + 1e-10))   # fit on log-RV

    preds_log = model.predict(X_te_s)
    preds     = np.exp(preds_log)
    preds     = np.maximum(preds, 1e-10)

    n_lag = max(max([1,2,3,5,10,22]), max([5,10,22,66]))
    feat_names = (
        [f"RV_lag{l}" for l in [1,2,3,5,10,22]]
        + [f"RV_roll{w}" for w in [5,10,22,66]]
        + [f"RV_std{w}" for w in [5,22]]
        + [f"logRV_lag{l}" for l in [1,5,22]]
    )

    return ForecastResult(
        model_name="GradientBoosting",
        predictions=preds,
        actuals=y_te,
        train_size=n_tr,
        test_size=len(y_te),
        rmse=float(np.sqrt(mean_squared_error(y_te, preds))),
        mae=float(np.mean(np.abs(y_te - preds))),
        qlike=_qlike(y_te, preds),
        feature_names=feat_names,
        feature_importances=model.feature_importances_,
    )


def train_lstm(rv: np.ndarray, test_fraction: float = 0.2,
               seq_len: int = 22, hidden: int = 32,
               n_epochs: int = 50, lr: float = 1e-3,
               seed: int = 42) -> ForecastResult:
    """
    LSTM-based vol forecaster (PyTorch).

    Input:  sequence of length seq_len of log-RV values
    Output: next-day log-RV prediction
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    rv  = np.asarray(rv, dtype=float)
    log_rv = np.log(rv + 1e-10)

    # Build sequences
    X, y = [], []
    for i in range(seq_len, len(log_rv)):
        X.append(log_rv[i - seq_len : i])
        y.append(log_rv[i])
    X = np.array(X); y = np.array(y)

    n    = len(X)
    n_tr = int(n * (1 - test_fraction))
    X_tr, X_te = X[:n_tr], X[n_tr:]
    y_tr, y_te = y[:n_tr], y[n_tr:]

    # Scale
    mu_x, std_x = X_tr.mean(), X_tr.std() + 1e-10
    mu_y, std_y = y_tr.mean(), y_tr.std() + 1e-10
    X_tr_s = (X_tr - mu_x) / std_x
    X_te_s = (X_te - mu_x) / std_x
    y_tr_s = (y_tr - mu_y) / std_y

    # Dataset
    Xt = torch.tensor(X_tr_s[:, :, None], dtype=torch.float32)
    yt = torch.tensor(y_tr_s[:, None],    dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True)

    # Model
    class VolLSTM(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, num_layers=2, batch_first=True,
                                dropout=0.1)
            self.fc   = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    net  = VolLSTM(hidden)
    opt  = torch.optim.Adam(net.parameters(), lr=lr)
    crit = nn.MSELoss()

    for epoch in range(n_epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward()
            opt.step()

    # Predict
    net.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(X_te_s[:, :, None], dtype=torch.float32)
        preds_s = net(Xte_t).numpy().flatten()
    preds_log = preds_s * std_y + mu_y
    preds     = np.exp(preds_log)
    actuals   = np.exp(y_te)
    preds     = np.maximum(preds, 1e-10)

    return ForecastResult(
        model_name="LSTM",
        predictions=preds,
        actuals=actuals,
        train_size=n_tr,
        test_size=len(y_te),
        rmse=float(np.sqrt(mean_squared_error(actuals, preds))),
        mae=float(np.mean(np.abs(actuals - preds))),
        qlike=_qlike(actuals, preds),
    )


def compare_models(rv: np.ndarray, test_fraction: float = 0.20,
                   include_lstm: bool = True) -> list[ForecastResult]:
    """Train HAR, GBM, and (optionally) LSTM; return sorted by RMSE."""
    results = [
        train_har(rv, test_fraction),
        train_gbm(rv, test_fraction),
    ]
    if include_lstm:
        try:
            results.append(train_lstm(rv, test_fraction))
        except Exception:
            pass   # PyTorch may not be available in all envs
    results.sort(key=lambda r: r.rmse)
    return results
