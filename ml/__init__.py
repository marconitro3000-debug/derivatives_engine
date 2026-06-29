"""Machine learning for derivatives: LSM American options, SSVI surface, vol forecasting."""

from .longstaff_schwartz import (
    price_american_lsm, price_bermudan_lsm, LSMResult,
)
from .ssvi import (
    SVIParams, SSVIParams,
    calibrate_svi, calibrate_ssvi,
    svi_fit_summary,
)
from .vol_forecast import (
    HARModel, ForecastResult,
    build_har_features, build_ml_features,
    train_har, train_gbm, train_lstm, compare_models,
)

__all__ = [
    # LSM
    "price_american_lsm", "price_bermudan_lsm", "LSMResult",
    # SVI / SSVI
    "SVIParams", "SSVIParams", "calibrate_svi", "calibrate_ssvi", "svi_fit_summary",
    # vol forecasting
    "HARModel", "ForecastResult",
    "build_har_features", "build_ml_features",
    "train_har", "train_gbm", "train_lstm", "compare_models",
]
