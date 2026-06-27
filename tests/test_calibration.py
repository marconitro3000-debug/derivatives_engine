"""
tests/test_calibration.py
Tests for the self-updating calibration engine.
Run with: pytest tests/test_calibration.py -v
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pytest
import tempfile, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from options.engine import PricingEngine
from core.calibration.store import CalibrationStore
from core.calibration.calibrator import calibrate, blend_params
from core.data.loader import SyntheticLoader
from core.models import heston as h
from core.models import svi as svi_mod


@pytest.fixture
def tmp_db():
    path = tempfile.mktemp(suffix=".db")
    yield path
    gc.collect()
    for wal in (path, path + "-wal", path + "-shm"):
        try:
            if os.path.exists(wal):
                os.unlink(wal)
        except PermissionError:
            pass


# ── Heston model ──────────────────────────────────────────────────────────────

class TestHestonModel:

    def test_reduces_to_bs_at_low_vol_of_vol(self):
        from options.black_scholes import price as bs
        p  = h.HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.001, rho=0.0)
        hp = h.price(100, 100, 1.0, 0.05, p, "call")
        bp = bs(100, 100, 1.0, 0.05, 0.2, "call")
        assert abs(hp - bp) < 0.01

    def test_put_call_parity(self):
        p = h.HestonParams(v0=0.04, kappa=2.0, theta=0.05, xi=0.3, rho=-0.5)
        c = h.price(100, 100, 1.0, 0.05, p, "call")
        put = h.price(100, 100, 1.0, 0.05, p, "put")
        parity = c - put - (100 - 100 * np.exp(-0.05))
        assert abs(parity) < 0.05

    def test_feller_condition(self):
        ok  = h.HestonParams(v0=0.04, kappa=3.0, theta=0.05, xi=0.3, rho=-0.5)
        bad = h.HestonParams(v0=0.04, kappa=0.5, theta=0.05, xi=2.0, rho=-0.5)
        assert ok.feller_satisfied()
        assert not bad.feller_satisfied()


# ── SVI model ─────────────────────────────────────────────────────────────────

class TestSVIModel:

    def test_total_variance_positive(self):
        p = svi_mod.SVIParams(a=0.02, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        k = np.linspace(-0.3, 0.3, 20)
        assert (svi_mod.total_variance(k, p) > 0).all()

    def test_arb_free_check(self):
        good = svi_mod.SVIParams(a=0.02, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        bad  = svi_mod.SVIParams(a=0.02, b=3.0, rho=-0.9, m=0.0, sigma=0.1)
        assert svi_mod.is_butterfly_arbitrage_free(good)
        assert not svi_mod.is_butterfly_arbitrage_free(bad)


# ── calibration recovery ──────────────────────────────────────────────────────

class TestCalibrationRecovery:

    def test_heston_recovers_true_params(self):
        loader = SyntheticLoader()
        true_p = h.HestonParams(v0=0.04, kappa=3.0, theta=0.05, xi=0.4, rho=-0.6)
        md = loader.load(true_params=true_p, noise=0.0)
        result = calibrate("heston", md)
        for key in ["v0", "kappa", "theta", "xi", "rho"]:
            assert abs(result.params[key] - true_p.to_dict()[key]) < 0.05

    def test_heston_low_rmse(self):
        loader = SyntheticLoader()
        md = loader.load(noise=0.0)
        result = calibrate("heston", md)
        assert result.rmse < 0.01

    def test_svi_low_rmse(self):
        loader = SyntheticLoader()
        md = loader.load(noise=0.0)
        result = calibrate("svi", md)
        assert result.rmse < 0.01
        assert "slices" in result.params

    def test_svi_one_slice_per_maturity(self):
        loader = SyntheticLoader()
        maturities = np.array([0.25, 0.5, 1.0])
        md = loader.load(maturities=maturities, noise=0.0)
        result = calibrate("svi", md)
        assert len(result.params["slices"]) == 3


# ── warm start & EWMA ─────────────────────────────────────────────────────────

class TestWarmStartAndBlending:

    def test_blend_params_alpha_1(self):
        old = {"a": 1.0, "b": 2.0}
        new = {"a": 3.0, "b": 4.0}
        assert blend_params(old, new, 1.0) == new

    def test_blend_params_alpha_0(self):
        old = {"a": 1.0, "b": 2.0}
        new = {"a": 3.0, "b": 4.0}
        assert blend_params(old, new, 0.0) == old

    def test_blend_params_midpoint(self):
        old = {"a": 0.0}
        new = {"a": 10.0}
        assert blend_params(old, new, 0.5)["a"] == 5.0

    def test_warm_start_flag(self):
        loader = SyntheticLoader()
        md = loader.load(noise=0.0)
        cold = calibrate("heston", md)
        warm = calibrate("heston", md, warm_start=cold.params)
        assert not cold.warm_started
        assert warm.warm_started


# ── SQLite store ──────────────────────────────────────────────────────────────

class TestStore:

    def test_save_and_retrieve_latest(self, tmp_db):
        store  = CalibrationStore(tmp_db)
        loader = SyntheticLoader()
        md = loader.load(noise=0.0)
        result = calibrate("heston", md)
        store.save("TEST", result, spot=md.spot, r=md.r)

        latest = store.latest_params("TEST", "heston")
        assert latest is not None
        assert "kappa" in latest

    def test_history_accumulates(self, tmp_db):
        store  = CalibrationStore(tmp_db)
        loader = SyntheticLoader()
        md = loader.load(noise=0.0)
        for _ in range(3):
            store.save("TEST", calibrate("heston", md), spot=md.spot, r=md.r)
        hist = store.history("TEST", "heston")
        assert len(hist) == 3

    def test_param_timeseries(self, tmp_db):
        store  = CalibrationStore(tmp_db)
        loader = SyntheticLoader()
        md = loader.load(noise=0.0)
        for _ in range(3):
            store.save("TEST", calibrate("heston", md), spot=md.spot, r=md.r)
        ts, vals = store.param_timeseries("TEST", "heston", "kappa")
        assert len(ts) == len(vals) == 3

    def test_tickers_list(self, tmp_db):
        store  = CalibrationStore(tmp_db)
        loader = SyntheticLoader()
        md = loader.load(noise=0.0)
        store.save("AAA", calibrate("heston", md), spot=md.spot, r=md.r)
        store.save("BBB", calibrate("heston", md), spot=md.spot, r=md.r)
        assert set(store.tickers()) == {"AAA", "BBB"}

    def test_latest_returns_none_when_empty(self, tmp_db):
        store = CalibrationStore(tmp_db)
        assert store.latest_params("NONE", "heston") is None


# ── full engine ───────────────────────────────────────────────────────────────

class TestPricingEngine:

    def test_update_and_price_heston(self, tmp_db):
        eng = PricingEngine(model="heston", db_path=tmp_db, source="synthetic")
        eng.update("TEST", noise=0.0)
        px = eng.price_option("TEST", K=100, T=0.5, option="call")
        assert px > 0

    def test_update_and_price_svi(self, tmp_db):
        eng = PricingEngine(model="svi", db_path=tmp_db, source="synthetic")
        eng.update("TEST", noise=0.0)
        px = eng.price_option("TEST", K=100, T=0.5, option="call")
        assert px > 0

    def test_second_update_is_warm(self, tmp_db):
        eng = PricingEngine(model="heston", db_path=tmp_db, source="synthetic")
        r1 = eng.update("TEST", noise=0.0)
        r2 = eng.update("TEST", noise=0.0)
        assert not r1.warm_started
        assert r2.warm_started

    def test_price_without_calibration_raises(self, tmp_db):
        eng = PricingEngine(model="heston", db_path=tmp_db, source="synthetic")
        with pytest.raises(RuntimeError):
            eng.price_option("NEVER_CALIBRATED", K=100, T=0.5)

    def test_svi_maturity_interpolation(self, tmp_db):
        eng = PricingEngine(model="svi", db_path=tmp_db, source="synthetic")
        eng.update("TEST", maturities=np.array([0.25, 0.5]), noise=0.0)
        iv_lo  = eng.implied_vol("TEST", K=100, T=0.25)
        iv_hi  = eng.implied_vol("TEST", K=100, T=0.5)
        iv_mid = eng.implied_vol("TEST", K=100, T=0.375)
        assert min(iv_lo, iv_hi) - 0.02 <= iv_mid <= max(iv_lo, iv_hi) + 0.02

    def test_drift_tracking(self, tmp_db):
        eng = PricingEngine(model="heston", db_path=tmp_db, source="synthetic")
        eng.update("TEST", noise=0.0)
        eng.update("TEST", noise=0.005)
        ts, vals = eng.drift("TEST", "kappa")
        assert len(ts) == 2
