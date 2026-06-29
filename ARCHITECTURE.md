# Architecture

Technical reference for the current `derivatives-engine` layout.

The repo is organised around one rule: product modules live at the top level,
while reusable infrastructure lives in `core/`.

## Data Flow

```text
Yahoo Finance / SyntheticLoader
        |
        v
MarketData(spot, r, strikes, maturities, ivs, weights)
        |
        +--> calibrate("heston") --> HestonParams --> core.models.heston.price()
        |
        +--> calibrate("svi")    --> SVI slices   --> SVI IV --> options.price()
        |
        +--> direct pricing      --> Black-Scholes / Monte Carlo / Binomial

CalibrationStore (SQLite)
        |
        +--> latest_params(), history(), param_timeseries()
```

## Layout

```text
core/
  models/
    heston.py       Heston stochastic volatility model
    svi.py          SVI implied-volatility smile parametrisation
  calibration/
    calibrator.py   Heston/SVI calibration and MarketData container
    store.py        SQLite calibration history
  data/
    loader.py       YFinanceLoader and SyntheticLoader

options/
  black_scholes.py  European option price, Greeks, put-call parity
  implied_vol.py    Newton-Raphson IV with Brent fallback
  monte_carlo.py    GBM simulation and vanilla/exotic payoffs
  binomial_tree.py  CRR European and American options
  vol_surface.py    Spline/RBF non-parametric IV surface
  engine.py         Self-updating calibration/pricing engine
  charts.py         CLI chart generation

forwards/
  pricing.py        Forward price, value, implied carry

futures/
  pricing.py        Futures price, basis, mark-to-market PnL

rates/
  curves.py         DiscountCurve, bootstrap, PV helper
  swaps.py          Vanilla fixed-for-floating swap valuation
  instruments.py    Deposit and swap quote containers
  charts.py         Zero, discount-factor, and forward-rate charts
```

`credit/`, `volatility/`, `exotics/`, and `structured/` are package
placeholders for future product families.

## Options

### Black-Scholes

European call and put prices:

```text
C = S N(d1) - K exp(-rT) N(d2)
P = K exp(-rT) N(-d2) - S N(-d1)

d1 = [ln(S/K) + (r + sigma^2 / 2)T] / [sigma sqrt(T)]
d2 = d1 - sigma sqrt(T)
```

API:

```python
from options import price, greeks, put_call_parity_check

price(S, K, T, r, sigma, option="call")
greeks(S, K, T, r, sigma)
put_call_parity_check(S, K, T, r, call_price, put_price)
```

### Implied Volatility

`options.implied_vol` inverts Black-Scholes using Newton-Raphson first and Brent
as a robust fallback.

```python
from options import implied_vol, iv_surface

iv = implied_vol(S, K, T, r, market_price, option="call")
surface = iv_surface(S, strikes, maturities, r, market_prices)
```

### Monte Carlo

`options.monte_carlo` simulates GBM paths under the risk-neutral measure:

```text
S[t+dt] = S[t] exp((r - 0.5 sigma^2)dt + sigma sqrt(dt) Z)
```

Supported payoff names include `european_call`, `european_put`,
`asian_call`, `asian_put`, `barrier_call`, `barrier_put`, `lookback_call`,
`lookback_put`, `digital_call`, and `digital_put`.

```python
from options import mc_price

res = mc_price(S, K, T, r, sigma, "european_call", n_sims=100_000)
```

### Binomial Tree

`options.binomial_tree` implements the Cox-Ross-Rubinstein tree for European
and American exercise.

```python
from options import binomial_price

res = binomial_price(S, K, T, r, sigma, option="put", style="american", n_steps=500)
```

## Core Models

### Heston

```text
dS = r S dt + sqrt(v) S dW_S
dv = kappa(theta - v)dt + xi sqrt(v) dW_v
corr(dW_S, dW_v) = rho
```

The implementation prices vanilla options by characteristic-function Fourier
inversion and exposes the Feller condition check:

```text
2 kappa theta >= xi^2
```

```python
from core.models.heston import HestonParams, price as heston_price

p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.6)
heston_price(S, K, T, r, p, option="call")
```

### SVI

SVI models one maturity slice in total implied variance:

```text
w(k) = a + b [rho(k - m) + sqrt((k - m)^2 + sigma^2)]
k = ln(K/F)
```

```python
from core.models.svi import SVIParams, implied_vol_svi

p = SVIParams(a=0.02, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
iv = implied_vol_svi(k, T, p)
```

## Calibration

`core.calibration.calibrator` fits either Heston or SVI to a `MarketData`
snapshot. It supports warm starts and optional EWMA smoothing.

```python
from core.calibration.calibrator import calibrate
from core.data.loader import SyntheticLoader

md = SyntheticLoader().load()
result = calibrate("heston", md)
```

`core.calibration.store.CalibrationStore` persists every calibration in SQLite:

```python
from core.calibration.store import CalibrationStore

store = CalibrationStore("cals.db")
store.save("AAPL", result, spot=md.spot, r=md.r)
store.latest_params("AAPL", "heston")
store.history("AAPL", "heston")
```

## Pricing Engine

`options.engine.PricingEngine` ties together loader, calibrator, store, and
pricing.

```python
from options.engine import PricingEngine

eng = PricingEngine(model="heston", db_path="cals.db", source="synthetic")
eng.update("TEST", noise=0.0)
eng.price_option("TEST", K=100, T=0.5, option="call")
```

For SVI, maturities are interpolated in total-variance space.

## Forwards

Forward fair value uses continuous cost of carry:

```text
F = S exp((r + u - q - y)T)
```

where `u` is storage/financing cost, `q` is income yield, and `y` is
convenience yield.

```python
from forwards import ForwardContract, forward_price, forward_value

fair = forward_price(spot=100, maturity=1.0, rate=0.05, income_yield=0.02)
value = forward_value(spot=105, delivery_price=100, maturity=1.0, rate=0.05)
```

## Futures

With deterministic rates, the theoretical futures price equals the forward
price. Listed futures are then marked to market:

```text
PnL = (F_today - F_yesterday) * contracts * multiplier
```

```python
from futures import FuturesContract, mark_to_market_pnl

pnl = mark_to_market_pnl(5000, 5010, contracts=2, multiplier=50)
future = FuturesContract("ES", price=5000, maturity=0.25, contracts=2, multiplier=50)
future.mtm_pnl(current_price=5010)
```

## Rates

`rates/curves.py` builds and queries discount curves. The internal
representation is discount factors; interpolation is linear in log-discount
space so interpolated discount factors remain positive.

```python
from rates import DiscountCurve, bootstrap_deposit_swap_curve

curve = DiscountCurve.flat(0.05, max_maturity=30.0)
df = curve.discount_factor(5.0)
zero = curve.zero_rate(5.0)
fwd = curve.forward_rate(1.0, 2.0)
swap_rate = curve.par_swap_rate(5.0)
```

Bootstrapping converts market quotes into curve pillars:

```python
curve = bootstrap_deposit_swap_curve(
    deposit_quotes={0.25: 0.041, 0.50: 0.042, 1.00: 0.043},
    swap_quotes={2.0: 0.044, 3.0: 0.045, 5.0: 0.047},
)
```

Deposits use:

```text
DF(T) = 1 / (1 + rT)
```

Par swaps solve:

```text
S * sum(alpha_i * DF(T_i)) = 1 - DF(T_n)
```

`rates/swaps.py` values vanilla fixed-for-floating swaps:

```python
from rates import InterestRateSwap

swap = InterestRateSwap(1_000_000, fixed_rate=0.045, maturity=5.0, position="payer")
pv = swap.pv(curve)
par = swap.par_rate(curve)
```

`rates/charts.py` saves:

```text
zero_curve.png
discount_factors.png
forward_rates.png
```

The CLI writes rate outputs to:

```text
output/rates/<curve_name>/<run_id>/
```

## References

1. Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
2. Cox, J., Ross, S. and Rubinstein, M. (1979). Option Pricing: A Simplified Approach.
3. Heston, S. (1993). A Closed-Form Solution for Options with Stochastic Volatility.
4. Gatheral, J. (2004). A Parsimonious Arbitrage-Free Implied Volatility Parametrization.
5. Albrecher, H. et al. (2007). The Little Heston Trap.
6. Brenner, M. and Subrahmanyam, M. (1988). A Simple Formula to Compute the Implied Standard Deviation.
