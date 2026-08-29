"""
arbitrage/put_call_parity.py
Put-call parity (PCP) violation scanner.

For European options on a non-dividend-paying asset:
    C(K,T) − P(K,T) = S·e^{-qT} − K·e^{-rT}

where q is the continuous dividend yield.  Any deviation larger than the
bid-ask spread is exploitable via a synthetic forward.

The scanner:
  1. Downloads live call and put quotes from Yahoo Finance.
  2. Pairs calls and puts at the same (K, T).
  3. Computes the observed LHS and the theoretical RHS.
  4. Reports violations that exceed the bid-ask tolerance.

Practical note
--------------
Yahoo Finance provides mid implied-vols, not bid/ask prices, so we reconstruct
mid prices via Black-Scholes and estimate bid-ask half-width from the IV spread
when available.  Violations quoted in dollar terms are therefore approximate.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class PCPViolation:
    """A single put-call parity violation."""
    strike: float
    maturity: float          # years
    call_price: float        # mid
    put_price: float         # mid
    forward_price: float     # S·e^{-qT} − K·e^{-rT}
    observed_lhs: float      # C − P
    deviation: float         # observed_lhs − forward_price  (signed)
    deviation_pct: float     # deviation / spot  (%)
    bid_ask_half: float      # estimated bid-ask half-spread ($)
    exploitable: bool        # |deviation| > bid_ask_half + min_profit

    def __str__(self) -> str:
        sign = "+" if self.deviation > 0 else ""
        expl = " ← EXPLOITABLE" if self.exploitable else ""
        return (
            f"K={self.strike:.1f}  T={self.maturity:.3f}Y  "
            f"C-P={self.observed_lhs:.4f}  Fwd={self.forward_price:.4f}  "
            f"dev={sign}{self.deviation:.4f} ({sign}{self.deviation_pct:.3f}%){expl}"
        )


@dataclass
class PCPScanResult:
    """Full put-call parity scan output."""
    ticker: str
    spot: float
    r: float
    q: float
    scan_time: str
    n_pairs: int
    violations: list[PCPViolation]
    all_deviations: np.ndarray   # signed deviation for every matched pair

    @property
    def n_violations(self) -> int:
        return len(self.violations)

    @property
    def max_deviation(self) -> float:
        return float(np.max(np.abs(self.all_deviations))) if len(self.all_deviations) else 0.0

    @property
    def rmse_deviation(self) -> float:
        return float(np.sqrt(np.mean(self.all_deviations ** 2))) if len(self.all_deviations) else 0.0

    def summary(self) -> str:
        lines = [
            "━━━ Put-Call Parity Scan ━━━",
            f"  Ticker : {self.ticker}  (S = {self.spot:.2f})",
            f"  r = {self.r*100:.2f}%  q = {self.q*100:.2f}%",
            f"  Time   : {self.scan_time}",
            f"  Matched pairs : {self.n_pairs}",
            f"  Max deviation : {self.max_deviation:.4f} ({self.max_deviation/self.spot*100:.3f}%)",
            f"  RMSE deviation: {self.rmse_deviation:.4f}",
            f"  Exploitable   : {self.n_violations}",
        ]
        for v in self.violations:
            lines.append(f"    → {v}")
        if not self.violations:
            lines.append("  No exploitable violations found ✓")
        return "\n".join(lines)


# ── Scanner ───────────────────────────────────────────────────────────────────

class PCPScanner:
    """
    Put-call parity scanner.

    Parameters
    ----------
    min_profit : float
        Minimum net profit ($ per share) beyond bid-ask to flag a violation.
    """

    def __init__(self, min_profit: float = 0.05):
        self.min_profit = min_profit

    def scan_live(
        self,
        ticker: str,
        r: float = 0.04,
        q: float = 0.0,
        moneyness_range: tuple = (0.85, 1.15),
        min_volume: int = 10,
        max_expiries: int = 6,
    ) -> PCPScanResult:
        """Download live quotes and scan for PCP violations."""
        spot, pairs = _load_matched_pairs(
            ticker, r, q, moneyness_range, min_volume, max_expiries
        )
        return self._scan(spot, pairs, r, q, ticker)

    def scan_from_data(
        self,
        spot: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        call_ivs: np.ndarray,
        put_ivs: np.ndarray,
        r: float = 0.04,
        q: float = 0.0,
        ticker: str = "UNK",
    ) -> PCPScanResult:
        """Scan pre-loaded aligned (K, T, call_iv, put_iv) arrays."""
        from options.black_scholes import price as bs_price

        pairs = []
        for K, T, c_iv, p_iv in zip(strikes, maturities, call_ivs, put_ivs):
            c_px  = bs_price(spot, K, T, r, c_iv, "call")
            p_px  = bs_price(spot, K, T, r, p_iv, "put")
            ba_c  = c_px * 0.01    # 1% bid-ask estimate
            ba_p  = p_px * 0.01
            pairs.append((float(K), float(T), c_px, p_px, ba_c + ba_p))

        return self._scan(spot, pairs, r, q, ticker)

    # ── internal ──────────────────────────────────────────────────────────────

    def _scan(
        self, spot: float, pairs: list, r: float, q: float, ticker: str
    ) -> PCPScanResult:
        from options.black_scholes import price as bs_price

        violations    = []
        all_devs      = []

        for K, T, c_px, p_px, ba_spread in pairs:
            fwd  = spot * np.exp(-q * T) - K * np.exp(-r * T)
            lhs  = c_px - p_px
            dev  = lhs - fwd

            all_devs.append(dev)

            ba_half  = ba_spread / 2.0
            exploitable = abs(dev) > (ba_half + self.min_profit)

            violations.append(PCPViolation(
                strike        = K,
                maturity      = T,
                call_price    = c_px,
                put_price     = p_px,
                forward_price = fwd,
                observed_lhs  = lhs,
                deviation     = dev,
                deviation_pct = dev / spot * 100,
                bid_ask_half  = ba_half,
                exploitable   = exploitable,
            ))

        # Keep only exploitable ones in the violations list
        all_devs_arr = np.array(all_devs)
        exploitable_violations = [v for v in violations if v.exploitable]
        # Sort by |deviation|
        exploitable_violations.sort(key=lambda v: abs(v.deviation), reverse=True)

        return PCPScanResult(
            ticker         = ticker,
            spot           = spot,
            r              = r,
            q              = q,
            scan_time      = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            n_pairs        = len(pairs),
            violations     = exploitable_violations,
            all_deviations = all_devs_arr,
        )


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_matched_pairs(
    ticker: str, r: float, q: float,
    moneyness_range: tuple, min_volume: int, max_expiries: int,
) -> tuple[float, list]:
    """
    Download calls and puts, reconstruct BS mid-prices, match by (K, T).
    Returns (spot, list of (K, T, call_px, put_px, bid_ask_spread)).
    """
    import yfinance as yf
    from options.black_scholes import price as bs_price

    tk   = yf.Ticker(ticker)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    now  = datetime.now()

    expiries = []
    for exp in tk.options:
        T = (datetime.strptime(exp, "%Y-%m-%d") - now).days / 365.0
        if T > 0.03:
            expiries.append(exp)
        if len(expiries) >= max_expiries:
            break

    lo, hi  = moneyness_range
    pairs   = []

    for exp in expiries:
        T = (datetime.strptime(exp, "%Y-%m-%d") - now).days / 365.0
        if T <= 0:
            continue

        chain  = tk.option_chain(exp)
        calls  = chain.calls
        puts   = chain.puts

        calls = calls[
            (calls["volume"].fillna(0) >= min_volume) &
            (calls["impliedVolatility"] > 0.005) &
            (calls["strike"] >= spot * lo) &
            (calls["strike"] <= spot * hi)
        ].set_index("strike")

        puts = puts[
            (puts["volume"].fillna(0) >= min_volume) &
            (puts["impliedVolatility"] > 0.005) &
            (puts["strike"] >= spot * lo) &
            (puts["strike"] <= spot * hi)
        ].set_index("strike")

        # Match on strike
        common = calls.index.intersection(puts.index)
        for K in common:
            c_iv = float(calls.loc[K, "impliedVolatility"])
            p_iv = float(puts.loc[K,  "impliedVolatility"])

            c_px = bs_price(spot, K, T, r, c_iv, "call")
            p_px = bs_price(spot, K, T, r, p_iv, "put")

            # Bid-ask estimate: use bid/ask columns if available, else 1%
            try:
                c_ba = float(calls.loc[K, "ask"]) - float(calls.loc[K, "bid"])
                p_ba = float(puts.loc[K,  "ask"]) - float(puts.loc[K,  "bid"])
                c_ba = max(c_ba, 0.0)
                p_ba = max(p_ba, 0.0)
            except Exception:
                c_ba = c_px * 0.01
                p_ba = p_px * 0.01

            pairs.append((float(K), T, c_px, p_px, c_ba + p_ba))

    return spot, pairs


# ── Convenience function ──────────────────────────────────────────────────────

def scan_parity(ticker: str, r: float = 0.04, q: float = 0.0, **kwargs) -> PCPScanResult:
    """One-liner: download live quotes and scan put-call parity."""
    return PCPScanner().scan_live(ticker, r=r, q=q, **kwargs)
