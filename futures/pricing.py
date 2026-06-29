"""
Futures helpers.

With deterministic rates, theoretical futures price equals forward price. The
main practical difference is daily mark-to-market settlement.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

from forwards import forward_price


def futures_price(
    spot: float,
    maturity: float,
    rate: float,
    income_yield: float = 0.0,
    storage_cost: float = 0.0,
    convenience_yield: float = 0.0,
) -> float:
    """Theoretical futures price under deterministic rates."""
    return forward_price(
        spot,
        maturity,
        rate,
        income_yield=income_yield,
        storage_cost=storage_cost,
        convenience_yield=convenience_yield,
    )


def mark_to_market_pnl(
    previous_price: float,
    current_price: float,
    contracts: float = 1.0,
    multiplier: float = 1.0,
    position: str = "long",
) -> float:
    """Daily futures PnL from exchange mark-to-market."""
    sign = 1.0 if position == "long" else -1.0 if position == "short" else None
    if sign is None:
        raise ValueError("position must be 'long' or 'short'.")
    return float(sign * (current_price - previous_price) * contracts * multiplier)


def annualized_basis(spot: float, futures: float, maturity: float) -> float:
    """Continuously compounded futures basis: ln(F/S) / T."""
    if spot <= 0 or futures <= 0 or maturity <= 0:
        raise ValueError("spot, futures, and maturity must be positive.")
    return float(log(futures / spot) / maturity)


@dataclass(frozen=True)
class FuturesContract:
    """Listed futures exposure with contract multiplier."""

    underlying: str
    price: float
    maturity: float
    contracts: float = 1.0
    multiplier: float = 1.0
    position: str = "long"

    def fair_price(
        self,
        spot: float,
        rate: float,
        income_yield: float = 0.0,
        storage_cost: float = 0.0,
        convenience_yield: float = 0.0,
    ) -> float:
        return futures_price(
            spot,
            self.maturity,
            rate,
            income_yield=income_yield,
            storage_cost=storage_cost,
            convenience_yield=convenience_yield,
        )

    def mtm_pnl(self, current_price: float) -> float:
        return mark_to_market_pnl(
            self.price,
            current_price,
            contracts=self.contracts,
            multiplier=self.multiplier,
            position=self.position,
        )
