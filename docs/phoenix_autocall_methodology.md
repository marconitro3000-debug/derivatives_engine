# Phoenix Autocall methodology

A Phoenix Autocall Note is a path-dependent structured product with conditional coupons, memory features and potential early redemption.

## Simplified product supported in v0

- One underlying.
- Quarterly observation dates.
- Coupon paid if the underlying is above the coupon barrier.
- Missed coupons are accrued if memory is enabled.
- Early redemption if the underlying is above the autocall level on a non-final observation date.
- At maturity, full capital is returned if final spot is above the capital barrier.
- If final spot is below the capital barrier, redemption follows the underlying performance.

## Pricing model

The first model is GBM Monte Carlo:

```text
S(t+dt) = S(t) * exp((r - q - 0.5 σ²)dt + σ sqrt(dt) Z)
```

For every simulated path the engine:

1. checks coupon observations;
2. updates memory coupons;
3. checks autocall observations;
4. computes final redemption;
5. discounts all cashflows;
6. averages discounted payoffs across paths.

## Risk analytics

Reported metrics include:

- fair value as % of notional;
- Monte Carlo standard error and 95% confidence interval;
- autocall probability;
- barrier touch probability;
- final capital-loss probability;
- expected coupon;
- expected life;
- terminal distribution quantiles;
- finite-difference Greeks;
- stress scenario prices;
- spot-vol fair-value heatmap.

## Limitations

This model is deliberately simplified. Real desks use calendars, dividends, repo/funding curves, issuer curves, volatility surfaces, model calibration, XVA, market conventions and independent model validation.
