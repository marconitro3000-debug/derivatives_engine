"""
volsurface/chain.py
Raw option chain -> clean implied-volatility point cloud.

The quality of an implied-vol surface is decided here, not in the model. Five
things matter and each is done explicitly rather than delegated to the data
vendor:

1. **Liquidity filtering.** Two-sided quotes only, positive size, and a cap on
   the relative bid-ask spread. Stale or one-sided rows produce IVs that no
   model should be asked to fit.

2. **Forward and discount factor implied from the market.** Instead of assuming
   ``F = S * exp((r - q) T)`` with a guessed dividend yield, the forward is
   recovered from put-call parity across matched strikes:

       C(K) - P(K) = DF * (F - K)

   which is a straight line in ``K`` with slope ``-DF`` and intercept
   ``DF * F``. A least-squares fit on the liquid matched strikes therefore
   yields both the discount factor and the forward the market is actually
   trading. This is what removes the systematic skew tilt you get from a wrong
   dividend assumption.

3. **OTM-only selection.** Calls above the forward, puts below it. Those are the
   liquid instruments; the ITM wing carries the same information with a wider
   spread, and including both would double-count every quote.

4. **In-house IV inversion.** Vendor ``impliedVolatility`` fields are computed
   with the vendor's own rate and dividend assumptions and are frequently wrong
   in the wings. The IVs here come from ``volsurface.impliedvol`` applied to the
   forward-measure price ``mid / DF``, so they are consistent with the forward
   fitted in step 2.

5. **De-Americanisation.** Listed equity and ETF options are American. A
   European inversion has nowhere to put the early-exercise premium and charges
   it to volatility instead -- on SPY that is around 12bp of vol beyond a year,
   and it is not uniform: it sits almost entirely in the puts and grows with
   maturity. `volsurface.american` prices the premium on a lattice and strips it
   out, and `fit_forward` iterates that together with step 2, because put-call
   parity is a theorem about European options and the forward fitted from raw
   American mids is biased too.

Every quote also carries a fitting weight built from Black-Scholes vega and the
quoted spread: an IV error on a 0.05-vega wing option is worth far less than the
same error at the money, and unweighted least squares in vol space gets this
backwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

from volsurface.american import carry_from_forward, de_americanised_iv
from volsurface.conventions import year_fraction
from volsurface.blackscholes import greeks, price as bs_price
from volsurface.impliedvol import implied_vol


# -- snapshot container -------------------------------------------------------

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


# -- forward / discount from put-call parity ----------------------------------

def implied_forward(strikes: np.ndarray, calls: np.ndarray, puts: np.ndarray,
                    weights: np.ndarray | None = None) -> tuple[float, float]:
    """Fit ``C(K) - P(K) = DF * (F - K)`` by weighted least squares.

    Parameters
    ----------
    strikes, calls, puts : matched arrays -- same strike, same expiry.
    weights              : optional per-strike weights (use inverse spread).

    Returns
    -------
    ``(forward, discount_factor)``

    Raises
    ------
    ValueError
        If there are fewer than three matched strikes, or the fit implies a
        discount factor outside a plausible range -- which means the quotes were
        too stale to trust and the caller should fall back to a rate assumption.
    """
    if len(strikes) < 3:
        raise ValueError("need at least 3 matched call/put strikes for a parity fit")

    strikes = np.asarray(strikes, dtype=float)
    y = np.asarray(calls, dtype=float) - np.asarray(puts, dtype=float)
    wts = np.ones_like(strikes) if weights is None else np.asarray(weights, dtype=float)
    wts = wts / wts.sum()

    # Weighted linear regression y = intercept + slope * K.
    kbar = float(np.sum(wts * strikes))
    ybar = float(np.sum(wts * y))
    cov = float(np.sum(wts * (strikes - kbar) * (y - ybar)))
    var = float(np.sum(wts * (strikes - kbar) ** 2))
    if var <= 0:
        raise ValueError("degenerate strike grid in parity fit")

    slope = cov / var
    intercept = ybar - slope * kbar

    df = -slope
    if not (0.80 < df <= 1.02):
        raise ValueError(f"parity fit implied an implausible discount factor {df:.4f}")
    forward = intercept / df
    if forward <= 0:
        raise ValueError("parity fit implied a non-positive forward")
    return float(forward), float(df)


def fit_forward(spot: float, strikes: np.ndarray,
                call_mid: np.ndarray, put_mid: np.ndarray,
                call_spread: np.ndarray, put_spread: np.ndarray,
                T: float, *, de_americanize: bool = True,
                lattice_steps: int = 150, fallback_rate: float = 0.04,
                n_passes: int = 2) -> tuple[float, float]:
    """Forward and discount factor from parity, corrected for American exercise.

    `implied_forward` fits ``C(K) - P(K) = DF * (F - K)``. That identity is a
    theorem about *European* options. Listed equity and ETF options are
    American, their early-exercise premia differ between the call and the put
    and vary with strike, so ``C - P`` is not a straight line in ``K`` and the
    regression returns a biased slope -- a biased discount factor, and through
    it a biased forward. On an eighteen-month SPY expiry the error is close to a
    full percent of the forward, which is a systematic tilt through the entire
    fitted skew.

    The fix is a short fixed point, because the two unknowns are circular: the
    forward is needed to de-Americanise a quote, and de-Americanised quotes are
    needed to fit the forward.

        1. fit (F, DF) from the raw American mids
        2. strip the early-exercise premium off every matched call and put
        3. refit (F, DF) on the European-equivalent prices
        4. repeat once

    Two passes are enough: the premium depends on the forward only weakly, so
    the second correction is already an order of magnitude smaller than the
    first.
    """
    weights = 1.0 / (call_spread + put_spread + 1e-6)
    F, DF = implied_forward(strikes, call_mid, put_mid, weights=weights)
    if not de_americanize:
        return F, DF

    for _ in range(n_passes):
        r, q = carry_from_forward(spot, F, DF, T)
        euro_c, euro_p = [], []
        for i, K in enumerate(strikes):
            euro_c.append(_european_equivalent(spot, float(K), T, r, q, F, DF,
                                               float(call_mid[i]), "call", lattice_steps))
            euro_p.append(_european_equivalent(spot, float(K), T, r, q, F, DF,
                                               float(put_mid[i]), "put", lattice_steps))
        try:
            F, DF = implied_forward(strikes, np.array(euro_c), np.array(euro_p),
                                    weights=weights)
        except ValueError:
            break                      # keep the last plausible fit
    return F, DF


def _european_equivalent(spot, K, T, r, q, F, DF, mid, option, lattice_steps) -> float:
    """Quote minus its early-exercise premium; the raw mid if that is not computable."""
    try:
        sigma_e = implied_vol(F, K, T, 0.0, mid / DF, option)
    except ValueError:
        return mid                     # unfittable quote: leave it alone
    _, euro = de_americanised_iv(spot, K, T, r, q, F, DF, mid, option,
                                 sigma_e, n_steps=lattice_steps)
    return euro


# -- quote cleaning -----------------------------------------------------------

def _clean_side(df, min_open_interest: int, max_rel_spread: float):
    """Keep two-sided, sized, tight quotes; add ``mid`` and ``spread`` columns."""
    import pandas as pd  # local: pandas is only needed on the raw-chain path

    d = df.copy()
    for col in ("bid", "ask", "strike"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["openInterest"] = pd.to_numeric(d.get("openInterest", 0), errors="coerce").fillna(0)
    d["volume"] = pd.to_numeric(d.get("volume", 0), errors="coerce").fillna(0)

    d = d[(d["bid"] > 0) & (d["ask"] > d["bid"]) & (d["strike"] > 0)]
    d = d[(d["openInterest"] >= min_open_interest) | (d["volume"] > 0)]
    if d.empty:
        return d.assign(mid=0.0, spread=0.0)

    d["mid"] = 0.5 * (d["bid"] + d["ask"])
    d["spread"] = d["ask"] - d["bid"]
    d = d[d["spread"] / d["mid"] <= max_rel_spread]
    return d.groupby("strike", as_index=False).first()


# -- main entry point ---------------------------------------------------------

def build_snapshot(
    ticker: str,
    spot: float,
    asof: date,
    expiry_chains: dict[str, tuple],
    *,
    fallback_rate: float = 0.04,
    min_open_interest: int = 10,
    max_rel_spread: float = 0.25,
    moneyness_range: tuple[float, float] = (-1.0, 0.6),
    min_T: float = 0.02,
    max_T: float = 2.0,
    de_americanize: bool = True,
    lattice_steps: int = 150,
) -> ChainSnapshot:
    """Assemble a `ChainSnapshot` from per-expiry ``(calls_df, puts_df)`` frames.

    Split out from `fetch_chain` so the cleaning logic is testable against
    fixture frames without touching the network.
    """
    k_all, T_all, iv_all, ive_all = [], [], [], []
    K_all, call_all, mid_all, mide_all, spr_all, vega_all = [], [], [], [], [], []
    forwards: dict[float, float] = {}
    discounts: dict[float, float] = {}

    for expiry, (calls_raw, puts_raw) in sorted(expiry_chains.items()):
        T = year_fraction(asof, expiry, "act365f")
        if not (min_T <= T <= max_T):
            continue

        calls = _clean_side(calls_raw, min_open_interest, max_rel_spread)
        puts = _clean_side(puts_raw, min_open_interest, max_rel_spread)
        if len(calls) == 0 or len(puts) == 0:
            continue

        # --- forward + discount from parity on matched strikes ---------------
        matched = np.intersect1d(calls["strike"].to_numpy(), puts["strike"].to_numpy())
        # Near-the-money strikes only: far-from-forward parity residuals are
        # dominated by spread noise rather than by the forward.
        matched = matched[(matched > 0.85 * spot) & (matched < 1.15 * spot)]
        try:
            cm = calls.set_index("strike").loc[matched]
            pm = puts.set_index("strike").loc[matched]
            F, DF = fit_forward(
                spot, matched,
                cm["mid"].to_numpy(), pm["mid"].to_numpy(),
                cm["spread"].to_numpy(), pm["spread"].to_numpy(),
                T, de_americanize=de_americanize, lattice_steps=lattice_steps,
            )
        except (ValueError, KeyError):
            DF = float(np.exp(-fallback_rate * T))
            F = spot / DF

        forwards[T] = F
        discounts[T] = DF

        # --- OTM selection ---------------------------------------------------
        for side, is_call in ((calls[calls["strike"] >= F], True),
                              (puts[puts["strike"] < F], False)):
            for _, row in side.iterrows():
                K = float(row["strike"])
                k = float(np.log(K / F))
                if not (moneyness_range[0] <= k <= moneyness_range[1]):
                    continue

                mid = float(row["mid"])
                option = "call" if is_call else "put"
                try:
                    # r = 0 with S = F is exactly the Black-76 forward measure.
                    sigma = implied_vol(F, K, T, 0.0, mid / DF, option)
                except ValueError:
                    continue                    # below intrinsic -> unfittable
                if not (0.01 < sigma < 3.0):
                    continue

                v = greeks(F, K, T, 0.0, sigma)["vega"] * 100.0 * DF
                if v <= 0:
                    continue

                # Listed equity/ETF options are American; a European inversion
                # charges the early-exercise premium to volatility. Price the
                # premium on a lattice and take it back out.
                sigma_european = sigma
                mid_european = mid
                if de_americanize:
                    r_exp, q_exp = carry_from_forward(spot, F, DF, T)
                    sigma, mid_european = de_americanised_iv(
                        spot, K, T, r_exp, q_exp, F, DF, mid, option,
                        sigma_european, n_steps=lattice_steps,
                    )

                k_all.append(k)
                T_all.append(T)
                iv_all.append(sigma)
                ive_all.append(sigma_european)
                K_all.append(K)
                call_all.append(is_call)
                mid_all.append(mid)
                mide_all.append(mid_european)
                spr_all.append(float(row["spread"]))
                vega_all.append(float(v))

    if not k_all:
        raise RuntimeError(
            f"No quotes for {ticker} survived cleaning. Loosen min_open_interest "
            f"({min_open_interest}) or max_rel_spread ({max_rel_spread})."
        )

    vega = np.array(vega_all)
    spread = np.array(spr_all)

    # Weight ~ vega / spread: how much price information one vol point carries,
    # divided by how uncertain that price is. Clipped so a single crossed quote
    # cannot dominate the objective.
    raw_w = vega / np.maximum(spread, 0.01)
    raw_w = np.minimum(raw_w, np.quantile(raw_w, 0.99))

    return ChainSnapshot(
        ticker=ticker, asof=asof, spot=float(spot),
        k=np.array(k_all), T=np.array(T_all), iv=np.array(iv_all),
        iv_european=np.array(ive_all),
        weight=raw_w / raw_w.mean(),
        strike=np.array(K_all), is_call=np.array(call_all, dtype=bool),
        mid=np.array(mid_all), mid_european=np.array(mide_all),
        spread=spread, vega=vega,
        forwards=forwards, discounts=discounts,
    )


def fetch_chain(ticker: str, *, max_expiries: int = 8, **kwargs) -> ChainSnapshot:
    """Download a live chain from Yahoo Finance and clean it.

    Network-bound and rate-limited; `synthetic_snapshot` is the offline
    equivalent the tests run against.
    """
    try:
        import yfinance as yf
    except ImportError as exc:                                  # pragma: no cover
        raise RuntimeError("pip install yfinance to fetch live chains") from exc

    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d", interval="1d")
    if hist.empty:
        raise RuntimeError(f"no price history for {ticker}")
    spot = float(hist["Close"].dropna().iloc[-1])
    asof = datetime.now().date()

    listed = list(tk.options or [])
    if not listed:
        raise RuntimeError(f"no listed expiries for {ticker}")

    chains: dict[str, tuple] = {}
    for exp in _select_expiries(listed, asof, max_expiries,
                                kwargs.get("min_T", 0.02), kwargs.get("max_T", 2.0)):
        ch = tk.option_chain(exp)
        chains[exp] = (ch.calls, ch.puts)

    return build_snapshot(ticker, spot, asof, chains, **kwargs)


def _select_expiries(listed: list[str], asof: date, n: int,
                     min_T: float, max_T: float) -> list[str]:
    """Pick ``n`` expiries spread across the term structure, not the first ``n``.

    An index like SPY lists daily expiries for the next few weeks and then
    monthlies: taking the first eight gives eight expiries inside three months
    and no term structure at all. Since the calendar-arbitrage condition is a
    statement *across* maturities, a chain with no long end cannot test it and
    the fitted surface has nothing to extrapolate from beyond a quarter.

    Selection is evenly spaced in ``sqrt(T)``, which is roughly how vol term
    structure moves, so the front end still gets the denser sampling it
    deserves without crowding out the back.
    """
    eligible = [(year_fraction(asof, e, "act365f"), e) for e in listed]
    eligible = [(T, e) for T, e in eligible if min_T <= T <= max_T]
    if not eligible:
        raise RuntimeError(
            f"no expiries between {min_T:.3f}y and {max_T:.1f}y among {len(listed)} listed"
        )
    if len(eligible) <= n:
        return [e for _, e in eligible]

    Ts = np.array([T for T, _ in eligible])
    targets = np.linspace(np.sqrt(Ts[0]), np.sqrt(Ts[-1]), n) ** 2
    picked = sorted({int(np.argmin(np.abs(Ts - t))) for t in targets})
    return [eligible[i][1] for i in picked]


# -- offline fixture ----------------------------------------------------------

def synthetic_snapshot(
    *,
    ticker: str = "SYN",
    spot: float = 100.0,
    rate: float = 0.03,
    n_expiries: int = 6,
    n_strikes: int = 21,
    rho: float = -0.65,
    eta: float = 1.0,
    gamma: float = 0.35,
    atm_vol: float = 0.20,
    noise_bps: float = 0.0,
    seed: int = 0,
) -> ChainSnapshot:
    """A chain generated from a known arbitrage-free SSVI surface.

    Used by the tests and by ``scripts/fit_surface.py --synthetic``: because the
    ground-truth surface is known and provably arbitrage-free, any violation the
    diagnostics report on a fit of this data is a defect in the *model*, not a
    feature of the market.
    """
    from volsurface.svi import SSVIParams

    rng = np.random.default_rng(seed)
    ssvi = SSVIParams(rho=rho, eta=eta, gamma=gamma)
    asof = date(2025, 1, 2)

    maturities = np.linspace(0.08, 1.5, n_expiries)
    k_grid = np.linspace(-0.45, 0.35, n_strikes)

    k_all, T_all, iv_all, K_all, call_all, mid_all, spr_all, vega_all = ([] for _ in range(8))
    forwards, discounts = {}, {}

    for T in maturities:
        T = float(round(T, 6))
        DF = float(np.exp(-rate * T))
        F = spot / DF
        forwards[T], discounts[T] = F, DF

        theta = atm_vol ** 2 * T
        w = ssvi.total_var(k_grid, theta)
        iv = np.sqrt(np.maximum(w, 1e-12) / T)
        if noise_bps:
            iv = iv + rng.normal(0.0, noise_bps / 10_000.0, size=iv.shape)

        for k, sigma in zip(k_grid, iv):
            K = float(F * np.exp(k))
            is_call = bool(k >= 0)
            option = "call" if is_call else "put"
            fwd_price = bs_price(F, K, T, 0.0, float(sigma), option)
            v = greeks(F, K, T, 0.0, float(sigma))["vega"] * 100.0 * DF

            k_all.append(float(k)); T_all.append(T); iv_all.append(float(sigma))
            K_all.append(K); call_all.append(is_call)
            mid_all.append(fwd_price * DF); spr_all.append(0.02); vega_all.append(float(v))

    vega = np.array(vega_all)
    raw_w = vega / np.maximum(np.array(spr_all), 0.01)
    return ChainSnapshot(
        ticker=ticker, asof=asof, spot=spot,
        k=np.array(k_all), T=np.array(T_all), iv=np.array(iv_all),
        iv_european=np.array(iv_all),      # generated as European by construction
        weight=raw_w / raw_w.mean(),
        strike=np.array(K_all), is_call=np.array(call_all, dtype=bool),
        mid=np.array(mid_all), mid_european=np.array(mid_all),
        spread=np.array(spr_all), vega=vega,
        forwards=forwards, discounts=discounts,
    )
