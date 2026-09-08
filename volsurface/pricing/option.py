"""
volsurface/pricer.py
Price one named option off a fitted surface.

Everything else in this library works in ``(k, T)`` -- log-moneyness against the
fitted forward, and a year fraction. That is the right coordinate for *fitting*
a surface and the wrong one for *using* it: an option is named by a strike, a
calendar expiry date and a side, and nobody holds "SPY, k = +0.013, T = 0.282y".

This module is the translation layer, and it is deliberately the only place the
translation happens:

    quote = price_option(surface, chain, strike=780, expiry="2026-12-19", kind="put")
    quote.price          # 21.734
    quote.implied_vol    # 0.1842
    quote.delta          # -0.383

Three things it does that a bare Black-Scholes call cannot:

**It takes the forward and the discount factor from the chain, not from a rate
assumption.** Both were *fitted* from put-call parity on the quotes
(`chain.fit_forward`), so pricing against a spot-plus-assumed-carry would
silently substitute a different forward than the one the surface was calibrated
in, and the vol it returns would be quoted against the wrong thing.

**It says when the answer is extrapolated.** A strike or an expiry outside the
quoted range still gets a number -- the bounded parametrisation guarantees the
surface degrades to the SSVI prior rather than to noise out there -- but that
number is the model's opinion, not the market's, and `is_extrapolated` says so.

**It reports the local no-arbitrage conditions at the point asked about.**
`dw_dT` and `butterfly_g` are the calendar and butterfly conditions evaluated at
exactly this `(k, T)`. The dense scan in `diagnostics` answers "is this surface
clean"; these two answer "is the specific number I am about to trade on clean".

Greeks convention
-----------------
The surface lives in the forward measure, so the internally consistent set is
the sensitivities of the *discounted* price with respect to the *forward*:

    price   = DF * Black76(F, K, T, sigma)
    delta   = d(price)/dF        vega = d(price)/d(sigma), per 1% of vol
    gamma   = d2(price)/dF2      theta = d(price)/dt, per calendar day

All four carry the discount factor. `delta_spot` converts to a spot delta under
``dF/dS = F/S``, which holds when the forward is the spot grown at a
deterministic carry -- true here by construction, since that is how `fit_forward`
builds it -- and it is reported separately rather than silently substituted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from ..data.conventions import year_fraction
from ..data.snapshot import ChainSnapshot
from ..evaluation.diagnostics import butterfly_g
from ..surfaces.base import VolSurface
from .blackscholes import greeks, price as bs_price

__all__ = ["OptionQuote", "price_option", "resolve_maturity"]

#: Longest year fraction accepted as a numeric expiry. Anything past it is a
#: date that lost its dashes, not a maturity -- see `_checked_years`.
MAX_YEARS = 100.0


# ── input parsing ────────────────────────────────────────────────────────────

def resolve_maturity(expiry, asof, convention: str = "act365f") -> tuple[float, date | None]:
    """Turn whatever the caller has into ``(T in years, expiry date or None)``.

    An expiry reaches this function in one of the four forms people actually
    have one in, and guessing between them is unambiguous:

    ``date`` / ``datetime`` / ``"YYYY-MM-DD"``
        A calendar expiry. Converted against `asof` -- the date the chain was
        snapshotted, not today -- because a surface fitted last Tuesday prices
        a December expiry at the maturity it had *last Tuesday*. Using today
        would quietly shorten every maturity by the age of the chain.
    ``"45d"`` / ``"45D"``
        Calendar days from `asof`.
    ``float`` / ``int`` / ``"0.25"``
        A year fraction, already converted -- this is what the rest of the
        library speaks. Bounded to ``(0, MAX_YEARS]``, which is what
        separates a maturity from a mistyped date: ``"2026"`` is not a
        2026-year option, and pricing one would hand back a
        plausible-looking number for a question nobody asked.

    Returns the expiry date as well as `T` when one is known, so the quote can
    print the option's real name rather than a decimal.
    """
    if isinstance(expiry, (int, float)) and not isinstance(expiry, bool):
        return _checked_years(float(expiry), expiry), None

    if isinstance(expiry, (date, datetime)):
        d = expiry.date() if isinstance(expiry, datetime) else expiry
        return year_fraction(asof, d, convention), d

    if isinstance(expiry, str):
        text = expiry.strip()
        if not text:
            raise ValueError("expiry is empty")

        if text[-1] in "dD" and text[:-1].strip().lstrip("+").isdigit():
            days = int(text[:-1].strip())
            return days / 365.0, None

        # A date wins over a number: "2026" is not a maturity of 2026 years.
        try:
            d = datetime.strptime(text, "%Y-%m-%d").date()
        except ValueError:
            pass
        else:
            return year_fraction(asof, d, convention), d

        try:
            years = float(text.replace(",", "."))
        except ValueError as exc:
            raise ValueError(
                f"could not read {expiry!r} as an expiry -- give a date "
                f"(2026-12-19), a year fraction (0.25), or days (45d)"
            ) from exc
        return _checked_years(years, expiry), None

    raise TypeError(f"unsupported expiry type: {type(expiry).__name__}")


def _checked_years(years: float, original) -> float:
    """A numeric expiry has to be a plausible year fraction.

    Without this, ``"2026"`` -- a year someone meant as a date -- parses as
    a maturity of 2026 years and prices without complaint, and a mistyped
    ``"20261219"`` does the same. Both are far more likely to be a date
    that lost its dashes than an intent, so they are refused by name.
    """
    if 0.0 < years <= MAX_YEARS:
        return years
    raise ValueError(
        f"{original!r} is not a plausible maturity ({years:g} years). Give "
        f"a date as YYYY-MM-DD, days as '45d', or a year fraction in "
        f"(0, {MAX_YEARS:g}]"
    )


def _normalise_kind(kind: str) -> str:
    k = str(kind).strip().lower()
    if k in ("c", "call", "calls"):
        return "call"
    if k in ("p", "put", "puts"):
        return "put"
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


# ── the answer ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OptionQuote:
    """One priced option, with the context needed to judge the number.

    The fields divide into three groups: what was asked (`ticker` .. `kind`),
    what the surface said (`implied_vol` .. `theta`), and whether to believe it
    (`in_quoted_strikes`, `in_quoted_maturities`, `dw_dT`, `butterfly_g`).
    """

    ticker: str
    asof: date
    kind: str                    # "call" or "put"
    strike: float
    expiry: date | None          # None when the caller gave a year fraction
    T: float

    forward: float
    discount: float
    log_moneyness: float

    implied_vol: float
    price: float
    delta: float                 # d(price)/d(forward)
    delta_spot: float            # d(price)/d(spot), under dF/dS = F/S
    gamma: float
    vega: float                  # per 1% of vol
    theta: float                 # per calendar day

    in_quoted_strikes: bool
    in_quoted_maturities: bool
    dw_dT: float
    butterfly_g: float

    @property
    def is_extrapolated(self) -> bool:
        """True when the surface had no quote nearby in strike or in maturity."""
        return not (self.in_quoted_strikes and self.in_quoted_maturities)

    @property
    def arbitrage_free(self) -> bool:
        """Both static no-arbitrage conditions hold at this exact ``(k, T)``."""
        return self.dw_dT >= 0.0 and self.butterfly_g >= 0.0

    def summary(self) -> str:
        """A block a human can read, with the caveats attached to the number."""
        when = self.expiry.isoformat() if self.expiry else f"T = {self.T:.4f}y"
        head = f"{self.ticker} {self.strike:g} {self.kind}  exp {when}"
        if self.expiry:
            head += f"  (T = {self.T:.4f}y)"

        lines = [
            head,
            f"  surface fitted {self.asof}",
            "",
            f"  implied vol   {self.implied_vol:>10.2%}",
            f"  price         {self.price:>10.4f}",
            "",
            f"  delta         {self.delta:>10.4f}   (forward)",
            f"  delta_spot    {self.delta_spot:>10.4f}",
            f"  gamma         {self.gamma:>10.6f}",
            f"  vega          {self.vega:>10.4f}   per 1% of vol",
            f"  theta         {self.theta:>10.4f}   per day",
            "",
            f"  k = {self.log_moneyness:+.4f}   forward {self.forward:.4f}   "
            f"discount {self.discount:.6f}",
        ]

        if self.is_extrapolated:
            missing = []
            if not self.in_quoted_strikes:
                missing.append("strike")
            if not self.in_quoted_maturities:
                missing.append("maturity")
            lines.append(f"  ! EXTRAPOLATED in {' and '.join(missing)} -- no quote "
                         f"nearby; this is the model's")
            lines.append("    opinion, bounded to the SSVI prior, not the market's.")
        else:
            lines.append("  within the quoted strike and maturity range")

        flag = "" if self.arbitrage_free else "   <- VIOLATION"
        lines.append(f"  dw/dT {self.dw_dT:+.3e}   g {self.butterfly_g:+.3e}"
                     f"   (both must be >= 0){flag}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ── the function ─────────────────────────────────────────────────────────────

def price_option(
    surface: VolSurface,
    chain: ChainSnapshot,
    strike: float,
    expiry,
    kind: str = "call",
    *,
    convention: str = "act365f",
) -> OptionQuote:
    """Price a single European option off a fitted surface.

    Parameters
    ----------
    surface : VolSurface
        Any fitted surface -- the neural one, per-slice SVI, joint SSVI. They
        all implement the same interface, so they all price here.
    chain : ChainSnapshot
        The chain the surface was fitted to. Not optional and not a
        convenience: it carries the forward and the discount factor that the
        surface's volatilities are quoted against, both fitted from the market.
    strike : float
        Strike, in the underlying's currency.
    expiry : date | datetime | str | float
        A calendar date (``"2026-12-19"``), days (``"45d"``), or a year
        fraction (``0.25``). See `resolve_maturity`.
    kind : str
        ``"call"`` or ``"put"`` (``"c"`` / ``"p"`` accepted).
    convention : str
        Day count used to turn a date into a year fraction. ACT/365F is what
        the chain itself was built with; changing it here and not there would
        price at a maturity the surface was never fitted at.

    Returns
    -------
    OptionQuote

    Raises
    ------
    ValueError
        On a non-positive strike or maturity. A zero or negative maturity has
        no implied vol to look up -- an expired option is an intrinsic-value
        question, not a surface question, and answering it here would return a
        number that looks like a price.
    """
    kind = _normalise_kind(kind)
    T, expiry_date = resolve_maturity(expiry, chain.asof, convention)

    strike = float(strike)
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if T <= 0:
        raise ValueError(
            f"maturity must be positive, got {T:.6f}y -- an expired or "
            f"same-day option has no implied volatility to look up"
        )

    F, DF = chain.forward_at(T)
    k = float(np.log(strike / F))

    kk = np.array([k], dtype=float)
    TT = np.array([T], dtype=float)

    sigma = float(surface.implied_vol(kk, TT)[0])
    dwdT = float(surface.dw_dT(kk, TT)[0])
    g = float(butterfly_g(surface, kk, TT)[0])

    # S = F with r = q = 0 is Black-76 in the forward measure; discount once,
    # and discount the sensitivities with it so the whole row is consistent.
    undiscounted = bs_price(F, strike, T, 0.0, sigma, kind)
    gk = greeks(F, strike, T, 0.0, sigma)

    delta = DF * (gk["delta_call"] if kind == "call" else gk["delta_put"])
    theta = DF * (gk["theta_call"] if kind == "call" else gk["theta_put"])

    return OptionQuote(
        ticker=chain.ticker,
        asof=chain.asof,
        kind=kind,
        strike=strike,
        expiry=expiry_date,
        T=T,
        forward=F,
        discount=DF,
        log_moneyness=k,
        implied_vol=sigma,
        price=DF * undiscounted,
        delta=delta,
        delta_spot=delta * F / chain.spot,
        gamma=DF * gk["gamma"],
        vega=DF * gk["vega"],
        theta=theta,
        in_quoted_strikes=bool(chain.k.min() <= k <= chain.k.max()),
        in_quoted_maturities=bool(chain.T.min() <= T <= chain.T.max()),
        dw_dT=dwdT,
        butterfly_g=g,
    )
