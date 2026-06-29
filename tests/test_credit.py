"""
tests/test_credit.py
Unit tests for the credit module.
Run with: pytest tests/test_credit.py -v
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rates import DiscountCurve
from credit.hazard_rate import HazardCurve
from credit.cds  import (protection_leg, risky_pv01, par_spread,
                          cds_value, cs01, ir01)
from credit.bond import risky_bond_price, z_spread, asset_swap_spread
from credit.cva  import cva, cva_option, dva, bilateral_cva


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_dc():
    return DiscountCurve.flat(0.05, max_maturity=30.0)


@pytest.fixture
def flat_hc():
    """100 bps spread, 40% recovery → h = 100bps / 60% ≈ 1.667%"""
    return HazardCurve.from_spread(100.0, recovery=0.40)


@pytest.fixture
def params(flat_dc, flat_hc):
    return dict(hazard_curve=flat_hc, discount_curve=flat_dc,
                maturity=5.0, recovery=0.40)


# ═══════════════════════════════════════════════════════════════
# HAZARD CURVE
# ═══════════════════════════════════════════════════════════════

class TestHazardCurve:

    def test_survival_at_zero_is_one(self, flat_hc):
        assert flat_hc.survival_prob(0.0) == pytest.approx(1.0)

    def test_survival_decreasing(self, flat_hc):
        q1 = flat_hc.survival_prob(1.0)
        q5 = flat_hc.survival_prob(5.0)
        assert q1 > q5

    def test_default_prob_complement(self, flat_hc):
        for T in [1.0, 3.0, 5.0, 10.0]:
            assert flat_hc.survival_prob(T) + flat_hc.default_prob(T) == pytest.approx(1.0)

    def test_flat_curve_exponential(self):
        h  = 0.02
        hc = HazardCurve.from_flat(h, recovery=0.40)
        T  = 3.0
        assert hc.survival_prob(T) == pytest.approx(np.exp(-h * T), rel=1e-8)

    def test_piecewise_survival(self):
        # h=0.01 on (0,1], h=0.02 on (1,5]
        hc = HazardCurve([1.0, 5.0], [0.01, 0.02], recovery=0.40)
        # Q(0.5) = exp(-0.01*0.5)
        assert hc.survival_prob(0.5) == pytest.approx(np.exp(-0.005), rel=1e-8)
        # Q(2.0) = exp(-0.01*1 - 0.02*1)
        assert hc.survival_prob(2.0) == pytest.approx(np.exp(-0.03), rel=1e-8)

    def test_from_spread_approximation(self):
        # For flat hazard, credit_spread ≈ h * (1-R)
        h  = 0.02
        hc = HazardCurve.from_flat(h, recovery=0.40)
        T  = 5.0
        # Λ(T)/T = h, so spread = h * (1-R) only for short T (approx)
        # For large T: spread = -ln(e^{-hT}) / T * (1-R) = h * (1-R)
        expected = h * (1 - 0.40)
        assert abs(hc.credit_spread(T) - expected) < 1e-10

    def test_bootstrap_recovers_input_spreads(self, flat_dc):
        tenors  = [1.0, 3.0, 5.0]
        spreads = [50.0, 100.0, 150.0]
        hc = HazardCurve.bootstrap(tenors, spreads, flat_dc, recovery=0.40)
        for T, s in zip(tenors, spreads):
            s_model = par_spread(hc, flat_dc, T, 0.40) * 10_000
            assert abs(s_model - s) < 0.2  # within 0.2 bps (numerical integration)


# ═══════════════════════════════════════════════════════════════
# CDS PRICING
# ═══════════════════════════════════════════════════════════════

class TestCDS:

    def test_protection_leg_positive(self, params):
        p = protection_leg(**params)
        assert p > 0

    def test_protection_leg_less_than_lgd(self, params):
        # ProtLeg ≤ (1-R) (since PV of 1 paid on default ≤ discount of 1)
        p = protection_leg(**params)
        assert p <= (1 - params["recovery"]) + 1e-10

    def test_rpv01_positive(self, flat_dc, flat_hc):
        rpv = risky_pv01(flat_hc, flat_dc, maturity=5.0)
        assert rpv > 0

    def test_rpv01_less_than_annuity(self, flat_dc, flat_hc):
        # Risky annuity < risk-free annuity (survival weighting reduces it)
        T     = 5.0
        rpv   = risky_pv01(flat_hc, flat_dc, T)
        times = np.arange(0.25, T + 0.001, 0.25)
        rf_ann = sum(flat_dc.discount_factor(t) * 0.25 for t in times)
        assert rpv < rf_ann

    def test_par_spread_round_trip(self, flat_dc):
        """par_spread from a flat hazard is close to input spread."""
        for s_bps in [50.0, 100.0, 200.0, 500.0]:
            hc    = HazardCurve.from_spread(s_bps, recovery=0.40)
            s_mod = par_spread(hc, flat_dc, 5.0, 0.40) * 10_000
            # Quarterly coupons vs continuous integration → ≤2% relative error
            assert abs(s_mod - s_bps) / s_bps < 0.02

    def test_cds_value_at_par_is_zero(self, flat_dc, flat_hc):
        s_par = par_spread(flat_hc, flat_dc, 5.0, 0.40)
        res   = cds_value(flat_hc, flat_dc, 5.0, s_par, 0.40, "buyer", 1.0)
        assert res["value"] == pytest.approx(0.0, abs=1e-6)

    def test_cds_buyer_above_par_negative(self, flat_dc, flat_hc):
        # Contract spread > par → buyer is overpaying → negative MtM
        s_par = par_spread(flat_hc, flat_dc, 5.0, 0.40)
        res   = cds_value(flat_hc, flat_dc, 5.0, s_par * 2, 0.40, "buyer", 1.0)
        assert res["value"] < 0

    def test_buyer_seller_opposite(self, flat_dc, flat_hc):
        s = 0.01
        b = cds_value(flat_hc, flat_dc, 5.0, s, 0.40, "buyer",  1.0)["value"]
        s_ = cds_value(flat_hc, flat_dc, 5.0, s, 0.40, "seller", 1.0)["value"]
        assert b + s_ == pytest.approx(0.0, abs=1e-10)

    def test_cs01_negative_for_buyer(self, flat_dc, flat_hc):
        # Protection buyer loses when credit improves (spread goes up → value down)
        # Wait: buyer gains when spreads widen (market spread > contract spread)
        # CS01 = change per +1bp spread (hazard) bump
        # When hazard rises, protection leg rises more than premium leg → buyer gains
        val = cs01(flat_hc, flat_dc, 5.0, 0.01, 0.40, "buyer", 1.0)
        # Small positive: protection buyer gains when credit deteriorates
        assert val > 0

    def test_ir01_small_magnitude(self, flat_dc, flat_hc):
        val = ir01(flat_hc, flat_dc, 5.0, 0.01, 0.40, "buyer", 1_000_000)
        # IR01 should be small (CDS is credit-sensitive, not rate-sensitive)
        assert abs(val) < 100  # less than $100 per million notional


# ═══════════════════════════════════════════════════════════════
# RISKY BOND
# ═══════════════════════════════════════════════════════════════

class TestBond:

    def test_zero_credit_risk_equals_risk_free(self, flat_dc):
        # With h=0, risky bond should price like risk-free bond
        hc   = HazardCurve.from_flat(1e-10, recovery=0.0)
        face, coupon, T = 1000.0, 0.05, 5.0
        res  = risky_bond_price(face, coupon, T, hc, flat_dc, pay_freq=2)
        # Risk-free bond price (BS discounting)
        c  = face * coupon / 2
        times = np.arange(0.5, T + 0.001, 0.5)
        pv_rf = sum(c * flat_dc.discount_factor(t) for t in times)
        pv_rf += face * flat_dc.discount_factor(T)
        assert abs(res["price"] - pv_rf) < 0.05

    def test_price_below_par_when_credit_spread_positive(self, flat_dc):
        hc  = HazardCurve.from_spread(200.0, recovery=0.40)
        res = risky_bond_price(1000.0, 0.05, 5.0, hc, flat_dc)
        # Spread > 0 → price < par (assuming coupon ≈ risk-free rate)
        assert res["price"] < 1000.0

    def test_price_increases_as_spread_decreases(self, flat_dc):
        prices = []
        for s in [300, 200, 100, 50]:
            hc  = HazardCurve.from_spread(s, recovery=0.40)
            res = risky_bond_price(1000.0, 0.05, 5.0, hc, flat_dc)
            prices.append(res["price"])
        assert prices == sorted(prices)  # strictly increasing

    def test_ytm_above_risk_free(self, flat_dc):
        hc  = HazardCurve.from_spread(150.0, recovery=0.40)
        res = risky_bond_price(1000.0, 0.05, 5.0, hc, flat_dc)
        rf  = flat_dc.zero_rate(5.0)
        assert res["yield_to_maturity"] > rf

    def test_z_spread_round_trip(self, flat_dc):
        hc    = HazardCurve.from_spread(150.0, recovery=0.40)
        res   = risky_bond_price(1000.0, 0.05, 5.0, hc, flat_dc)
        z     = z_spread(res["price"], 1000.0, 0.05, 5.0, flat_dc)
        # Z-spread should be close to credit spread (YTM - risk-free)
        assert abs(z * 10_000 - res["credit_spread_bps"]) < 5.0  # within 5 bps

    def test_z_spread_zero_for_risk_free_bond(self, flat_dc):
        hc  = HazardCurve.from_flat(1e-10, recovery=0.0)
        res = risky_bond_price(1000.0, 0.05, 5.0, hc, flat_dc, pay_freq=2)
        z   = z_spread(res["price"], 1000.0, 0.05, 5.0, flat_dc, pay_freq=2)
        assert abs(z) < 1e-4  # essentially zero

    def test_asset_swap_spread_sign(self, flat_dc):
        # Risky bond below par → ASW spread positive
        hc  = HazardCurve.from_spread(200.0, recovery=0.40)
        res = risky_bond_price(1000.0, 0.05, 5.0, hc, flat_dc)
        asw = asset_swap_spread(res["price"], 1000.0, 0.05, 5.0, flat_dc)
        assert asw > 0


# ═══════════════════════════════════════════════════════════════
# CVA
# ═══════════════════════════════════════════════════════════════

class TestCVA:

    def test_cva_positive(self, flat_dc, flat_hc):
        times = np.linspace(0.1, 1.0, 20)
        EE    = np.full(len(times), 5.0)   # constant exposure
        res   = cva(flat_hc, flat_dc, times, EE, 0.40)
        assert res["cva"] > 0

    def test_cva_zero_when_no_credit_risk(self, flat_dc):
        hc0   = HazardCurve([30.0], [0.0], recovery=0.40)
        times = np.linspace(0.1, 1.0, 20)
        EE    = np.full(len(times), 5.0)
        res   = cva(hc0, flat_dc, times, EE, 0.40)
        assert res["cva"] == pytest.approx(0.0, abs=1e-8)

    def test_cva_increases_with_spread(self, flat_dc):
        times = np.linspace(0.1, 1.0, 50)
        EE    = np.ones(len(times)) * 10.0
        cvas  = []
        for s in [50, 100, 200, 400]:
            hc  = HazardCurve.from_spread(s, recovery=0.40)
            res = cva(hc, flat_dc, times, EE, 0.40)
            cvas.append(res["cva"])
        assert cvas == sorted(cvas)

    def test_cva_option_less_than_vanilla(self, flat_dc, flat_hc):
        res = cva_option(100, 100, 1.0, 0.05, 0.20, flat_hc, flat_dc,
                         option_type="call", n_steps=100)
        assert res["cva_adjusted_price"] < res["vanilla_price"]

    def test_cva_option_positive_cva(self, flat_dc, flat_hc):
        res = cva_option(100, 100, 1.0, 0.05, 0.20, flat_hc, flat_dc,
                         option_type="call", n_steps=100)
        assert res["cva"] > 0

    def test_bilateral_cva_formula(self, flat_dc, flat_hc):
        times = np.linspace(0.1, 1.0, 20)
        EE    = np.ones(len(times)) * 5.0
        ENE   = np.ones(len(times)) * 3.0
        hc_own = HazardCurve.from_spread(80.0, recovery=0.40)

        cva_res = cva(flat_hc, flat_dc, times, EE,  0.40)
        dva_res = dva(hc_own,  flat_dc, times, ENE, 0.40)
        bcva    = bilateral_cva(cva_res, dva_res)

        assert bcva["bcva"] == pytest.approx(cva_res["cva"] - dva_res["dva"])
        assert bcva["cva"]  == pytest.approx(cva_res["cva"])
        assert bcva["dva"]  == pytest.approx(dva_res["dva"])
