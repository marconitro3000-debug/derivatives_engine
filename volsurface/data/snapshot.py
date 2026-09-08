"""
volsurface/data/snapshot.py
`ChainSnapshot` -- the cleaned, model-ready option chain every surface is fitted
to, and the only data structure that crosses from `volsurface.data` into the
rest of the library.

All arrays are aligned, one entry per surviving quote, and `k` is log-moneyness
against the *fitted* forward of that expiry rather than against the spot. That
choice is what makes the container safe to save and reload: a fitted surface is
useless without the forward and discount factor its volatilities were quoted
against, and those were fitted from these quotes (`data.forward`), not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np


@dataclass
class ChainSnapshot:
    """A cleaned, model-ready option chain at a single point in time.

    All arrays are aligned, one entry per surviving quote. ``k`` is
    log-moneyness against the *fitted forward* of that expiry, which is the
    coordinate every model in this repo works in.
    """

    ticker: str
    asof: date
    spot: float

    k: np.ndarray            # log(K / F_T)
    T: np.ndarray            # year fraction to expiry (ACT/365F)
    iv: np.ndarray           # implied vol actually fitted (de-Americanised)
    iv_european: np.ndarray  # the same quotes inverted as if they were European
    weight: np.ndarray       # fitting weight (vega / spread, mean-normalised)

    strike: np.ndarray
    is_call: np.ndarray      # bool: True for calls (OTM above the forward)
    mid: np.ndarray          # quoted mid price (American, as traded)
    mid_european: np.ndarray # the same quote with its early-exercise premium removed
    spread: np.ndarray       # absolute bid-ask spread
    vega: np.ndarray         # BS vega per 1.00 of vol

    forwards: dict[float, float] = field(default_factory=dict)   # T -> F
    discounts: dict[float, float] = field(default_factory=dict)  # T -> DF

    @property
    def total_variance(self) -> np.ndarray:
        """w = iv^2 * T -- the coordinate no-arbitrage conditions live in."""
        return self.iv ** 2 * self.T

    @property
    def maturities(self) -> np.ndarray:
        return np.array(sorted(self.forwards.keys()))

    def slice_at(self, T: float) -> "ChainSnapshot":
        """The single-expiry sub-chain, for per-slice SVI fitting."""
        return self._masked(self.T == T)

    def _masked(self, mask: np.ndarray) -> "ChainSnapshot":
        kept = set(np.unique(self.T[mask]).tolist())
        return ChainSnapshot(
            ticker=self.ticker, asof=self.asof, spot=self.spot,
            k=self.k[mask], T=self.T[mask], iv=self.iv[mask],
            iv_european=self.iv_european[mask], weight=self.weight[mask],
            strike=self.strike[mask], is_call=self.is_call[mask], mid=self.mid[mask],
            mid_european=self.mid_european[mask],
            spread=self.spread[mask], vega=self.vega[mask],
            forwards={t: f for t, f in self.forwards.items() if t in kept},
            discounts={t: d for t, d in self.discounts.items() if t in kept},
        )

    @property
    def early_exercise_bp(self) -> np.ndarray:
        """Per-quote bias, in bp of vol, that a European inversion would have had.

        Zero everywhere if the chain was built with ``de_americanize=False``, in
        which case `iv` and `iv_european` are the same array.
        """
        return (self.iv_european - self.iv) * 10_000.0

    def __len__(self) -> int:
        return len(self.k)

    # -- persistence ----------------------------------------------------------

    def save(self, path) -> None:
        """Write the cleaned chain to a ``.npz`` archive.

        A fitted surface is useless on its own: pricing a strike needs the
        forward and discount factor of its expiry, and those were *fitted* from
        this chain's own quotes. Saving the model without the chain would force
        a re-download to price anything, and the re-download would return a
        different market.
        """
        np.savez_compressed(
            path,
            ticker=self.ticker, asof=str(self.asof), spot=self.spot,
            k=self.k, T=self.T, iv=self.iv, iv_european=self.iv_european,
            weight=self.weight, strike=self.strike, is_call=self.is_call,
            mid=self.mid, mid_european=self.mid_european, spread=self.spread,
            vega=self.vega,
            forward_T=np.array(sorted(self.forwards)),
            forward_F=np.array([self.forwards[t] for t in sorted(self.forwards)]),
            discount_DF=np.array([self.discounts[t] for t in sorted(self.discounts)]),
        )

    @classmethod
    def load(cls, path) -> "ChainSnapshot":
        """Read back a chain written by `save`."""
        z = np.load(path, allow_pickle=False)
        Ts = z["forward_T"]
        return cls(
            ticker=str(z["ticker"]),
            asof=datetime.strptime(str(z["asof"]), "%Y-%m-%d").date(),
            spot=float(z["spot"]),
            k=z["k"], T=z["T"], iv=z["iv"], iv_european=z["iv_european"],
            weight=z["weight"], strike=z["strike"], is_call=z["is_call"],
            mid=z["mid"], mid_european=z["mid_european"], spread=z["spread"],
            vega=z["vega"],
            forwards={float(t): float(f) for t, f in zip(Ts, z["forward_F"])},
            discounts={float(t): float(d) for t, d in zip(Ts, z["discount_DF"])},
        )

    def forward_at(self, T: float) -> tuple[float, float]:
        """``(forward, discount)`` at any maturity, interpolated between expiries.

        Linear in ``T`` rather than in anything cleverer: between two listed
        expiries the forward is pinned at both ends by the market itself, and a
        smoother scheme would only add a shape nothing observed.
        """
        Ts = self.maturities
        F = float(np.interp(T, Ts, [self.forwards[t] for t in Ts]))
        DF = float(np.interp(T, Ts, [self.discounts[t] for t in Ts]))
        return F, DF

    def summary(self) -> str:
        return (
            f"{self.ticker} @ {self.asof}  spot={self.spot:.2f}\n"
            f"  {len(self)} quotes across {len(self.forwards)} expiries "
            f"({self.T.min():.3f}y - {self.T.max():.3f}y)\n"
            f"  k range [{self.k.min():+.3f}, {self.k.max():+.3f}]  "
            f"IV range [{self.iv.min():.1%}, {self.iv.max():.1%}]"
        )

    def early_exercise_summary(self) -> str:
        """How much volatility a European inversion would have invented."""
        bias = self.early_exercise_bp
        if not np.any(bias > 1e-9):
            return "  de-Americanisation off: quotes inverted as European"

        calls, puts = bias[self.is_call], bias[~self.is_call]
        lines = [
            f"  removed {bias.mean():.2f}bp of vol on average "
            f"(median {np.median(bias):.2f}, p95 {np.percentile(bias, 95):.2f}, "
            f"max {bias.max():.2f})",
            f"    calls {calls.mean():6.2f}bp mean   puts {puts.mean():6.2f}bp mean",
        ]
        for lo, hi in ((0.0, 0.15), (0.15, 0.5), (0.5, 1.0), (1.0, 99.0)):
            sel = bias[(self.T >= lo) & (self.T < hi)]
            if sel.size:
                label = f"T in [{lo:.2f}, {hi:.2f})" if hi < 90 else f"T >= {lo:.2f}"
                lines.append(f"    {label:<18} n={sel.size:4d}  "
                             f"mean {sel.mean():6.2f}bp  p95 {np.percentile(sel, 95):6.2f}bp")
        return "\n".join(lines)
