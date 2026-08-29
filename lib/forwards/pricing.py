"""
Forward pricing under continuous cost of carry.

Generic forward price:

    F = S * exp((r + u - q - y) * T)

where r is the risk-free rate, u storage/financing cost, q income yield, and
y convenience yield.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log


def _validate_positive(name: str, value: float):
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def forward_price(
    spot: float,
    maturity: float,
    rate: float,
    income_yield: float = 0.0,
    storage_cost: float = 0.0,
    convenience_yield: float = 0.0,
) -> float:
    """No-arbitrage forward price per unit."""
    _validate_positive("spot", spot)
    _validate_positive("maturity", maturity)
    carry = rate + storage_cost - income_yield - convenience_yield
    return float(spot * exp(carry * maturity))


def forward_value(
    spot: float,
    delivery_price: float,
    maturity: float,
    rate: float,
    income_yield: float = 0.0,
    storage_cost: float = 0.0,
    convenience_yield: float = 0.0,
    notional: float = 1.0,
    position: str = "long",
) -> float:
    """Present value of an existing forward contract."""
    _validate_positive("delivery_price", delivery_price)
    fair = forward_price(
        spot,
        maturity,
        rate,
        income_yield=income_yield,
        storage_cost=storage_cost,
        convenience_yield=convenience_yield,
    )
    sign = 1.0 if position == "long" else -1.0 if position == "short" else None
    if sign is None:
        raise ValueError("position must be 'long' or 'short'.")
    return float(sign * notional * exp(-rate * maturity) * (fair - delivery_price))


def implied_carry_rate(spot: float, forward: float, maturity: float) -> float:
    """Continuously compounded net carry implied by a forward price."""
    _validate_positive("spot", spot)
    _validate_positive("forward", forward)
    _validate_positive("maturity", maturity)
    return float(log(forward / spot) / maturity)


@dataclass(frozen=True)
class ForwardContract:
    """Existing forward contract with a fixed delivery price."""

    underlying: str
    delivery_price: float
    maturity: float
    notional: float = 1.0
    position: str = "long"

    def fair_price(
        self,
        spot: float,
        rate: float,
        income_yield: float = 0.0,
        storage_cost: float = 0.0,
        convenience_yield: float = 0.0,
    ) -> float:
        return forward_price(
            spot,
            self.maturity,
            rate,
            income_yield=income_yield,
            storage_cost=storage_cost,
            convenience_yield=convenience_yield,
        )

    def value(
        self,
        spot: float,
        rate: float,
        income_yield: float = 0.0,
        storage_cost: float = 0.0,
        convenience_yield: float = 0.0,
    ) -> float:
        return forward_value(
            spot,
            self.delivery_price,
            self.maturity,
            rate,
            income_yield=income_yield,
            storage_cost=storage_cost,
            convenience_yield=convenience_yield,
            notional=self.notional,
            position=self.position,
        )
