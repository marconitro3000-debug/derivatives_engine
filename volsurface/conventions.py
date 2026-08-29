"""
volsurface/conventions.py
Day-count conventions for converting calendar dates to year fractions.

Replaces the ad-hoc `(end - start).days / 365.0` scattered across the
codebase (data/ingestion.py, arbitrage/vol_surface.py) with explicit,
named conventions.
"""

from __future__ import annotations

from datetime import date, datetime

DateLike = "date | datetime | str"


def _to_date(d) -> date:
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    if isinstance(d, datetime):
        return d.date()
    return d


def year_fraction(start, end, convention: str = "act365f") -> float:
    """Year fraction between `start` and `end` under the given convention.

    Parameters
    ----------
    start, end : date | datetime | "YYYY-MM-DD" string
    convention : "act365f" (Actual/365 Fixed, the common default for equity
                 vol/options), "act360" (money-market instruments), or
                 "thirty360" (30/360, common for bonds/swaps fixed legs).
    """
    d0, d1 = _to_date(start), _to_date(end)

    if convention == "act365f":
        return (d1 - d0).days / 365.0
    if convention == "act360":
        return (d1 - d0).days / 360.0
    if convention == "thirty360":
        d0_day = min(d0.day, 30)
        d1_day = min(d1.day, 30) if d0_day == 30 else d1.day
        days = 360 * (d1.year - d0.year) + 30 * (d1.month - d0.month) + (d1_day - d0_day)
        return days / 360.0
    raise ValueError(f"unknown day-count convention: {convention}")
