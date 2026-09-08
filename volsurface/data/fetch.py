"""
volsurface/data/fetch.py
The live chain: Yahoo Finance -> `build_snapshot`.

Network-bound and rate-limited, and the only module in the package that is.
`data.synthetic` is the offline equivalent the tests run against.

The one non-obvious decision here is `_select_expiries`: taking the first ``n``
listed expiries off an index that quotes dailies gives eight expiries inside
three months and no term structure at all, and the calendar-arbitrage condition
is a statement *across* maturities.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np

from volsurface.data.clean import build_snapshot
from volsurface.data.conventions import year_fraction
from volsurface.data.snapshot import ChainSnapshot


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
