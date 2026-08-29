"""
tests/test_ml.py
Unit tests for ml/ — Longstaff-Schwartz, SVI/SSVI, vol forecasting.
"""

import numpy as np
import pytest

from ml import (
    price_american_lsm, price_bermudan_lsm, LSMResult,
    SVIParams, SSVIParams, calibrate_svi, calibrate_ssvi,
    train_har, train_gbm, compare_models,
    build_har_features, build_ml_features,
    HARModel,
)


# ── shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_rv():
    """Synthetic realized variance series (GARCH-like)."""
    rng = np.random.default_rng(42)
    n   = 500
    rv  = np.zeros(n)
    rv[0] = 0.0004   # initial daily variance ≈ (2% daily vol)²
    for t in range(1, n):
        eps    = rng.standard_normal()
        rv[t]  = max(0.00005 + 0.10 * rv[t-1] * eps**2 + 0.85 * rv[t-1],
                     1e-8)
    return rv   # daily realized variance


@pytest.fixture(scope="module")
def svi_slice():
    """Synthetic SVI smile: k ∈ [-0.3, 0.3], T=1y."""
    true = SVIParams(a=0.04, b=0.10, rho=-0.30, m=0.0, sigma=0.15)
    k    = np.linspace(-0.30, 0.30, 15)
    w    = true.total_var(k)
    return k, w, true


# ── Longstaff-Schwartz ────────────────────────────────────────────────────────

class TestLongstaffSchwartz:

    def test_american_put_positive(self):
        r = price_american_lsm(100, 100, 1.0, 0.05, 0.20,
                                n_sims=10_000, n_steps=50)
        assert r.price > 0

    def test_american_put_early_exercise_premium(self):
        """Deep ITM put: American > European (early exercise premium > 0)."""
        from options.black_scholes import price as bs_price
        r  = price_american_lsm(70, 100, 1.0, 0.05, 0.20, option_type="put",
                                 n_sims=20_000, n_steps=100)
        eu = bs_price(70, 100, 1.0, 0.05, 0.20, "put")
        assert r.price > eu - 0.05   # American ≥ European

    def test_american_ge_european(self):
        """American put ≥ European put."""
        from options.black_scholes import price as bs_price
        am = price_american_lsm(100, 100, 1.0, 0.05, 0.20,
                                 n_sims=20_000, n_steps=100)
        eu = bs_price(100, 100, 1.0, 0.05, 0.20, "put")
        assert am.price >= eu - 0.10   # small tolerance for MC noise

    def test_deep_otm_put_near_zero(self):
        """Deep OTM put: American price ≈ 0."""
        r = price_american_lsm(150, 100, 1.0, 0.05, 0.20,
                                n_sims=10_000, n_steps=50)
        assert r.price < 1.0

    def test_put_price_increases_with_vol(self):
        lo = price_american_lsm(100, 100, 1.0, 0.05, 0.10, n_sims=10_000, n_steps=50)
        hi = price_american_lsm(100, 100, 1.0, 0.05, 0.40, n_sims=10_000, n_steps=50)
        assert hi.price > lo.price

    def test_confidence_interval_contains_mean(self):
        r = price_american_lsm(100, 100, 1.0, 0.05, 0.20,
                                n_sims=10_000, n_steps=50)
        assert r.conf_95_lo <= r.price <= r.conf_95_hi

    def test_result_fields(self):
        r = price_american_lsm(100, 100, 1.0, 0.05, 0.20,
                                n_sims=5_000, n_steps=20)
        assert r.n_sims == 5_000
        assert r.n_steps == 20
        assert r.std_error >= 0
        assert len(r.exercise_boundary) == 20

    def test_put_moneyness_monotone(self):
        """ITM put > ATM put > OTM put."""
        itm = price_american_lsm(90,  100, 1.0, 0.05, 0.20, n_sims=10_000, n_steps=50)
        atm = price_american_lsm(100, 100, 1.0, 0.05, 0.20, n_sims=10_000, n_steps=50)
        otm = price_american_lsm(110, 100, 1.0, 0.05, 0.20, n_sims=10_000, n_steps=50)
        assert itm.price > atm.price > otm.price

    def test_bermudan_between_american_european(self):
        """Bermudan with 4 exercise dates ≥ European."""
        from options.black_scholes import price as bs_price
        ex_dates = [0.25, 0.50, 0.75, 1.0]
        berm = price_bermudan_lsm(100, 100, 1.0, 0.05, 0.20,
                                   ex_dates, n_sims=10_000, n_steps=100)
        eu   = bs_price(100, 100, 1.0, 0.05, 0.20, "put")
        assert berm.price >= eu - 0.10

    def test_laguerre_vs_power_basis_close(self):
        r_l = price_american_lsm(100, 100, 1.0, 0.05, 0.20, basis="laguerre",
                                   n_sims=10_000, n_steps=50, seed=42)
        r_p = price_american_lsm(100, 100, 1.0, 0.05, 0.20, basis="power",
                                   n_sims=10_000, n_steps=50, seed=42)
        assert abs(r_l.price - r_p.price) < 0.50   # within 50 cents


# ── SVI / SSVI ────────────────────────────────────────────────────────────────

class TestSVI:

    def test_svi_total_var_positive(self, svi_slice):
        k, w, params = svi_slice
        w_model = params.total_var(k)
        assert (w_model > 0).all()

    def test_svi_implied_vol_positive(self, svi_slice):
        k, w, params = svi_slice
        iv = params.implied_vol(k, T=1.0)
        assert (iv > 0).all()

    def test_calibrate_svi_round_trip(self, svi_slice):
        k, w_mkt, _ = svi_slice
        fitted = calibrate_svi(k, w_mkt)
        w_fit  = fitted.total_var(k)
        rmse   = np.sqrt(np.mean((w_fit - w_mkt)**2))
        assert rmse < 1e-3   # sub-bp IV accuracy

    def test_ssvi_params_phi_positive(self):
        p = SSVIParams(rho=-0.30, eta=0.50, gamma=0.30)
        for theta in [0.01, 0.04, 0.16, 0.36]:
            assert p.phi(theta) > 0

    def test_ssvi_total_var_positive(self):
        p = SSVIParams(rho=-0.30, eta=0.50, gamma=0.30)
        k = np.linspace(-0.5, 0.5, 20)
        w = p.total_var(k, theta=0.04)
        assert (w > 0).all()

    def test_ssvi_no_butterfly(self):
        p = SSVIParams(rho=-0.30, eta=0.50, gamma=0.30)
        assert p.no_butterfly_arbitrage(theta=0.04)

    def test_ssvi_atm_vol_from_theta(self):
        """ATM vol (k=0) = sqrt(theta/T * 0.5*(1 + sqrt(1-rho^2))) roughly."""
        p     = SSVIParams(rho=0.0, eta=0.10, gamma=0.30)
        theta = 0.04   # ATM total var
        T     = 1.0
        w_atm = p.total_var(np.array([0.0]), theta)[0]
        # For rho=0, k=0: w = (θ/2)(1 + √(1-ρ²)) = (θ/2)(1+1) = θ
        assert abs(w_atm - theta) < 0.01

    def test_calibrate_ssvi_converges(self, svi_slice):
        k, w_mkt, _ = svi_slice
        T = 1.0
        theta = w_mkt[len(w_mkt)//2]   # approx ATM variance
        fitted = calibrate_ssvi([k], [w_mkt], [theta])
        w_fit  = fitted.total_var(k, theta)
        rmse   = np.sqrt(np.mean((w_fit - w_mkt)**2))
        assert rmse < 0.01


# ── Vol Forecasting ───────────────────────────────────────────────────────────

class TestVolForecast:

    def test_build_har_features_shape(self, synthetic_rv):
        X, y = build_har_features(synthetic_rv)
        assert X.shape[1] == 3
        assert len(X) == len(y)

    def test_build_ml_features_shape(self, synthetic_rv):
        X, y = build_ml_features(synthetic_rv)
        assert X.shape[0] == len(y)
        assert X.shape[1] > 3   # more features than HAR

    def test_har_rmse_positive(self, synthetic_rv):
        r = train_har(synthetic_rv)
        assert r.rmse > 0

    def test_har_qlike_positive(self, synthetic_rv):
        r = train_har(synthetic_rv)
        assert r.qlike >= 0

    def test_har_predictions_positive(self, synthetic_rv):
        r = train_har(synthetic_rv)
        assert (r.predictions > 0).all()

    def test_har_r2_reasonable(self, synthetic_rv):
        r = train_har(synthetic_rv)
        assert r.r2 > -1.0   # not completely wrong

    def test_gbm_positive_predictions(self, synthetic_rv):
        r = train_gbm(synthetic_rv)
        assert (r.predictions > 0).all()

    def test_gbm_feature_importances(self, synthetic_rv):
        r = train_gbm(synthetic_rv)
        assert r.feature_importances is not None
        assert abs(r.feature_importances.sum() - 1.0) < 1e-5

    def test_har_model_summary(self, synthetic_rv):
        m = HARModel().fit(synthetic_rv)
        s = m.summary()
        assert "HAR Model" in s

    def test_compare_models_returns_list(self, synthetic_rv):
        results = compare_models(synthetic_rv, include_lstm=False)
        assert len(results) >= 2

    def test_compare_sorted_by_rmse(self, synthetic_rv):
        results = compare_models(synthetic_rv, include_lstm=False)
        rmses = [r.rmse for r in results]
        assert rmses == sorted(rmses)

    def test_har_forecast_positive(self, synthetic_rv):
        m = HARModel().fit(synthetic_rv)
        f = m.forecast(synthetic_rv)
        assert f > 0
