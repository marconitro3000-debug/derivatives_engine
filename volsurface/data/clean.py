"""
volsurface/data/clean.py
Raw per-expiry quote frames -> a `ChainSnapshot`.

This is where the quality of the surface is decided. `build_snapshot` is
deliberately separate from `data.fetch` so every filtering decision is testable
against fixture frames without touching the network:

1. **Liquidity filtering** (`_clean_side`) -- two-sided quotes, positive size, a
   cap on the relative spread. Stale or one-sided rows produce IVs no model
   should be asked to fit.
2. **Forward and discount from parity** -- delegated to `data.forward`.
3. **OTM-only selection** -- calls above the forward, puts below. The ITM wing
   carries the same information with a wider spread, and taking both would
   double-count every quote.
4. **In-house IV inversion** -- from ``mid / DF`` in the forward measure, so the
   vols are consistent with the forward fitted in step 2 rather than with a
   vendor's rate and dividend assumptions.
5. **De-Americanisation** -- the early-exercise premium is priced on a lattice
   and stripped out, because a European inversion has nowhere to put it and
   charges it to volatility instead.

Every surviving quote also carries a fitting weight built from vega and the
quoted spread: an IV error on a 0.05-vega wing option is worth far less than the
same error at the money, and unweighted least squares in vol space gets that
backwards.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from volsurface.data.conventions import year_fraction
from volsurface.data.forward import fit_forward
from volsurface.data.snapshot import ChainSnapshot
from volsurface.pricing.american import carry_from_forward, de_americanised_iv
from volsurface.pricing.blackscholes import greeks
from volsurface.pricing.impliedvol import implied_vol


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
