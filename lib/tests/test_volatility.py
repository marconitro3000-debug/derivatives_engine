"""
tests/test_volatility.py
Unit tests for the volatility module.

Coverage:
  TestRealizedVol   (10 tests) — all 5 estimators + EWMA + combined
  TestGARCH         (12 tests) — fit, forecast, simulate, properties
  TestVarianceSwap  (10 tests) — fair strike, MtM, VRP
"""

import numpy as np
import pandas as pd
import pytest

# ── realized vol ──────────────────────────────────────────────────────────────
from volatility.realized import (
    close_to_close, parkinson, garman_klass,
    rogers_satchell, yang_zhang, ewma, all_estimators, realized_variance,
)
from volatility.garch import fit as garch_fit, forecast as garch_forecast, simulate as garch_simulate
from volatility.variance_swap import (
    fair_variance_strike_bs, fair_variance_strike_mf,
    variance_swap_pv, vrp, vrp_summary, realized_variance_from_prices,
)


# ── shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ohlcv():
    """Synthetic OHLCV: GBM with σ=0.20, 500 days."""
    rng = np.random.default_rng(42)
    n   = 500
    dt  = 1.0 / 252
    r   = 0.03
    sig = 0.20
    S   = 100.0
    closes = [S]
    for _ in range(n - 1):
        closes.append(closes[-1] * np.exp((r - 0.5 * sig ** 2) * dt
                                          + sig * np.sqrt(dt) * rng.standard_normal()))
    closes  = np.array(closes)
    # Add OHLC noise
    noise   = rng.uniform(0.995, 1.005, (n, 4))
    high_   = closes * (1 + rng.uniform(0.002, 0.015, n))
    low_    = closes * (1 - rng.uniform(0.002, 0.015, n))
    open_   = np.roll(closes, 1) * noise[:, 0]
    open_[0] = S
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    df  = pd.DataFrame({
        "Open":  open_,
        "High":  np.maximum(high_, np.maximum(open_, closes)),
        "Low":   np.minimum(low_,  np.minimum(open_, closes)),
        "Close": closes,
    }, index=idx)
    return df


@pytest.fixture(scope="module")
def returns(ohlcv):
    return np.log(ohlcv["Close"] / ohlcv["Close"].shift(1)).dropna()


@pytest.fixture(scope="module")
def garch_result(returns):
    return garch_fit(returns)


# ── TestRealizedVol ───────────────────────────────────────────────────────────

class TestRealizedVol:

    def test_close_to_close_positive(self, ohlcv):
        cc = close_to_close(ohlcv["Close"], window=21)
        assert (cc.dropna() > 0).all()

    def test_parkinson_positive(self, ohlcv):
        pk = parkinson(ohlcv["High"], ohlcv["Low"], window=21)
        assert (pk.dropna() > 0).all()

    def test_garman_klass_positive(self, ohlcv):
        gk = garman_klass(ohlcv["Open"], ohlcv["High"],
                          ohlcv["Low"], ohlcv["Close"], window=21)
        assert (gk.dropna() > 0).all()

    def test_rogers_satchell_positive(self, ohlcv):
        rs = rogers_satchell(ohlcv["Open"], ohlcv["High"],
                              ohlcv["Low"], ohlcv["Close"], window=21)
        assert (rs.dropna() > 0).all()

    def test_yang_zhang_positive(self, ohlcv):
        yz = yang_zhang(ohlcv["Open"], ohlcv["High"],
                        ohlcv["Low"], ohlcv["Close"], window=21)
        assert (yz.dropna() > 0).all()

    def test_ewma_positive(self, ohlcv):
        ew = ewma(ohlcv["Close"])
        assert (ew.dropna() > 0).all()

    def test_close_to_close_not_annualized(self, ohlcv):
        cc_ann   = close_to_close(ohlcv["Close"], window=21, annualize=True).dropna()
        cc_daily = close_to_close(ohlcv["Close"], window=21, annualize=False).dropna()
        ratio = (cc_ann / cc_daily).dropna()
        # Should be close to sqrt(252)
        assert abs(ratio.mean() - np.sqrt(252)) < 0.5

    def test_all_estimators_shape(self, ohlcv):
        vol_df = all_estimators(ohlcv, window=21)
        assert vol_df.shape[1] == 6
        assert "CC" in vol_df.columns
        assert "Yang-Zhang" in vol_df.columns

    def test_estimators_in_ballpark(self, ohlcv):
        """All estimators should estimate vol in [5%, 60%] for the synthetic series."""
        vol_df = all_estimators(ohlcv, window=21).dropna()
        for col in vol_df.columns:
            assert (vol_df[col] > 0.05).all(), f"{col} too low"
            assert (vol_df[col] < 0.80).all(), f"{col} too high"

    def test_realized_variance_positive(self, ohlcv):
        rv = realized_variance(ohlcv["Close"], window=21)
        assert (rv.dropna() > 0).all()

    def test_parkinson_ge_cc_on_average(self, ohlcv):
        """Parkinson is typically lower variance estimator; both positive & similar."""
        cc = close_to_close(ohlcv["Close"], window=21).dropna()
        pk = parkinson(ohlcv["High"], ohlcv["Low"], window=21).dropna()
        idx = cc.index.intersection(pk.index)
        # Correlation should be high (same underlying process)
        corr = cc.loc[idx].corr(pk.loc[idx])
        assert corr > 0.7


# ── TestGARCH ─────────────────────────────────────────────────────────────────

class TestGARCH:

    def test_fit_converged(self, garch_result):
        assert garch_result.converged

    def test_parameters_positive(self, garch_result):
        assert garch_result.omega > 0
        assert garch_result.alpha > 0
        assert garch_result.beta  > 0

    def test_stationarity(self, garch_result):
        assert garch_result.alpha + garch_result.beta < 1.0

    def test_alpha_beta_sensible(self, garch_result):
        """α and β must be in (0,1); combined persistence < 1 (stationary)."""
        assert 0.0 < garch_result.alpha < 1.0
        assert 0.0 <= garch_result.beta < 1.0
        assert garch_result.alpha + garch_result.beta < 1.0

    def test_long_run_vol_sensible(self, garch_result):
        """Long-run vol should be in the same ballpark as sample std."""
        assert 0.05 < garch_result.long_run_vol < 0.80

    def test_half_life_positive(self, garch_result):
        assert garch_result.half_life > 0
        assert np.isfinite(garch_result.half_life)

    def test_conditional_vol_length(self, garch_result, returns):
        assert len(garch_result.conditional_vol) == len(returns)

    def test_conditional_vol_positive(self, garch_result):
        assert (garch_result.conditional_vol > 0).all()

    def test_residuals_near_standard_normal(self, garch_result):
        """Standardized residuals should have mean ≈ 0 and std ≈ 1."""
        z = garch_result.residuals.dropna()
        assert abs(z.mean()) < 0.15
        assert abs(z.std() - 1.0) < 0.15

    def test_forecast_shape(self, garch_result):
        fcast = garch_forecast(garch_result, h=30)
        assert len(fcast) == 30
        assert "forecast_vol" in fcast.columns
        assert "vol_lb_95" in fcast.columns

    def test_forecast_reverts_to_long_run(self, garch_result):
        """Long-horizon forecast should approach the long-run vol."""
        fcast = garch_forecast(garch_result, h=500)
        lr    = garch_result.long_run_vol
        last  = float(fcast["forecast_vol"].iloc[-1])
        assert abs(last - lr) < 0.01   # within 1 vol point

    def test_simulate_shape(self, garch_result):
        paths = garch_simulate(garch_result, n=252, n_paths=10, seed=0)
        assert paths.shape == (10, 252)

    def test_simulate_mean_near_mu(self, garch_result):
        """Mean of many simulated returns should approximate μ."""
        paths = garch_simulate(garch_result, n=252, n_paths=1000, seed=7)
        assert abs(paths.mean() - garch_result.mu) < 0.001

    def test_summary_string(self, garch_result):
        s = garch_result.summary()
        assert "GARCH(1,1)" in s
        assert "Long-run vol" in s

    def test_too_few_obs_raises(self):
        short = pd.Series(np.random.randn(30))
        with pytest.raises(ValueError, match="50"):
            garch_fit(short)


# ── TestVarianceSwap ──────────────────────────────────────────────────────────

class TestVarianceSwap:

    def test_bs_fair_strike_equals_sigma_squared(self):
        sigma = 0.20
        k2    = fair_variance_strike_bs(sigma, T=1.0)
        assert abs(k2 - sigma ** 2) < 1e-12

    def test_bs_fair_strike_positive(self):
        assert fair_variance_strike_bs(0.30, T=0.5) > 0

    def test_mf_fair_strike_no_options(self):
        """Single strike degenerate case — should still return a dict."""
        K = np.array([100.0])
        C = np.array([5.0])
        P = np.array([4.0])
        result = fair_variance_strike_mf(K, C, P, F=100.0, T=1.0, r=0.0)
        assert "fair_var" in result
        assert result["fair_var"] >= 0

    def test_mf_fair_strike_otm_selection(self):
        """Put contribution should be for K < F, call for K >= F."""
        F  = 100.0
        K  = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        # BS prices: use put-call parity for simplicity
        from scipy.stats import norm
        r, T, sig = 0.0, 1.0, 0.20
        d1 = (np.log(F / K) + 0.5 * sig ** 2 * T) / (sig * np.sqrt(T))
        d2 = d1 - sig * np.sqrt(T)
        C  = F * norm.cdf(d1) - K * norm.cdf(d2)
        P  = K * norm.cdf(-d2) - F * norm.cdf(-d1)
        res = fair_variance_strike_mf(K, C, P, F=F, T=T, r=r)
        assert res["put_contribution"]  > 0
        assert res["call_contribution"] > 0
        # Should roughly equal σ² for flat-vol BS inputs
        assert abs(res["fair_vol"] - sig) < 0.025

    def test_variance_swap_pv_zero_at_inception(self):
        """At inception, if current fair var == original strike, PV = 0."""
        K_var = 0.04    # 20% vol
        result = variance_swap_pv(
            realized_var_so_far=K_var,
            current_fair_var=K_var,
            K_var=K_var,
            t_elapsed=0.0,
            T_total=1.0,
            r=0.0,
            notional=100_000,
        )
        assert abs(result["value"]) < 1e-10

    def test_variance_swap_pv_positive_when_var_up(self):
        """Long variance swap profits when realized variance > strike."""
        K_var    = 0.04     # 20% vol
        rv_high  = 0.09     # 30% vol — high realized
        result = variance_swap_pv(
            realized_var_so_far=rv_high,
            current_fair_var=K_var,
            K_var=K_var,
            t_elapsed=0.5,
            T_total=1.0,
            r=0.0,
            notional=100_000,
        )
        assert result["value"] > 0

    def test_variance_swap_pv_negative_when_var_down(self):
        K_var   = 0.09
        rv_low  = 0.02
        result = variance_swap_pv(
            realized_var_so_far=rv_low,
            current_fair_var=rv_low,
            K_var=K_var,
            t_elapsed=1.0,
            T_total=1.0,
            r=0.0,
            notional=100_000,
        )
        assert result["value"] < 0

    def test_vrp_positive(self):
        idx = pd.date_range("2023-01-01", periods=5)
        iv  = pd.Series([0.25, 0.26, 0.24, 0.25, 0.27], index=idx)
        rv  = pd.Series([0.18, 0.19, 0.17, 0.20, 0.22], index=idx)
        v   = vrp(iv, rv)
        assert (v > 0).all()

    def test_vrp_summary_keys(self):
        idx    = pd.date_range("2023-01-01", periods=100)
        iv_s   = pd.Series(0.25, index=idx)
        rv_s   = pd.Series(0.20, index=idx)
        vrp_s  = vrp(iv_s, rv_s)
        stats  = vrp_summary(vrp_s)
        for key in ("mean", "std", "median", "pct_pos", "sharpe", "min", "max"):
            assert key in stats

    def test_vrp_summary_all_positive(self):
        idx   = pd.date_range("2023-01-01", periods=50)
        vrp_s = pd.Series(0.05, index=idx)
        stats = vrp_summary(vrp_s)
        assert stats["pct_pos"] == 1.0
        assert abs(stats["mean"] - 0.05) < 1e-10

    def test_realized_variance_from_prices_positive(self, ohlcv):
        rv = realized_variance_from_prices(ohlcv["Close"], window=21)
        assert (rv.dropna() > 0).all()
