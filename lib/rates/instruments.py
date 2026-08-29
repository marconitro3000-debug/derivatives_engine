"""Simple rate-instrument containers used for curve construction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DepositQuote:
    maturity: float
    rate: float

    def __post_init__(self):
        if self.maturity <= 0:
            raise ValueError("maturity must be positive.")


@dataclass(frozen=True)
class SwapQuote:
    maturity: float
    rate: float
    fixed_freq: int = 1

    def __post_init__(self):
        if self.maturity <= 0:
            raise ValueError("maturity must be positive.")
        if self.fixed_freq <= 0:
            raise ValueError("fixed_freq must be positive.")
