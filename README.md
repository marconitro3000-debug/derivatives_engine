# Derivatives Engine

Modular Python library for derivatives pricing, implied-volatility calibration,
and quantitative finance modelling. Covers options, rates, credit, volatility,
structured products, portfolio risk, ML pricing, FX, and commodities.

- [COMMANDS.md](COMMANDS.md) — all CLI commands with examples
- [ARCHITECTURE.md](ARCHITECTURE.md) — module breakdown, maths, API notes
- [derivatives_engine.ipynb](derivatives_engine.ipynb) — theory, formulas, runnable examples

---

## Features

| Module | Capability |
|--------|------------|
| `options/` | Black-Scholes pricing, greeks, implied vol, Monte Carlo, CRR binomial, vol surfaces, pricing engine |
| `core/models/` | Heston stochastic volatility, SVI smile parametrisation |
| `core/calibration/` | Heston/SVI calibration, warm starts, EWMA smoothing, SQLite audit trail |
| `core/data/` | Yahoo Finance loader, synthetic market-data loader |
| `forwards/` | Forward fair price, contract value, implied carry |
| `futures/` | Futures pricing, basis, daily mark-to-market PnL |
| `rates/` | Discount curves, zero/forward rates, bootstrap, vanilla swaps; swaptions (Black/Bachelier), caps/floors, SABR smile, Nelson-Siegel/Svensson curve fitting |
| `exotics/` | Barrier (Reiner-Rubinstein), Asian (geometric CF + arithmetic MC), Lookback (Goldman), Digital (cash/asset-or-nothing, one-touch) |
| `credit/` | Hazard curves, CDS pricing/greeks, risky bonds (Duffie-Singleton), CVA/DVA |
| `volatility/` | Realized estimators (CC/Parkinson/GK/RS/YZ/EWMA), GARCH(1,1) MLE, variance swaps, VRP |
| `structured/` | CDO (Gaussian copula / LHP), autocallable notes (MC), MBS/PSA prepayment model |
| `portfolio/` | Position & Portfolio containers, Greeks aggregation, VaR/CVaR (historical/parametric/Cornish-Fisher/MC), Kupiec backtest, stress scenarios, spot×vol P&L grid |
| `ml/` | Longstaff-Schwartz American/Bermudan options, SVI/SSVI smile calibration, vol forecasting (HAR + Gradient Boosting + LSTM) |
| `fx/` | Garman-Kohlhagen FX option pricing, Greeks (vanna/volga), 25/10-delta smile (RR/BF), vanna-volga exotic pricing, delta↔strike mapping |
| `commodities/` | Futures term structure (contango/backwardation), convenience yield, calendar spreads, crack/spark spreads, Schwartz (1997) mean-reversion model, spread options (Margrabe/Kirk) |

---

## Install

```bash
pip install -e ".[dev]"
pip install -e .
```

Python ≥ 3.11. Live data commands require network access (Yahoo Finance).

---

## Options CLI (live market data)

Fetches live spot and ATM IV from Yahoo Finance. Heston and SVI are calibrated
to the real option chain before pricing.

```bash
python scripts/price_option.py AAPL --strike 185 --expiry 0.25
python scripts/price_option.py SPY  --strike 550 --expiry 0.5  --model heston
python scripts/price_option.py TSLA --strike 250 --expiry 1.0  --type put --style american
python scripts/price_option.py NVDA --strike 900 --expiry 0.25 --model black-scholes
python scripts/price_option.py GS   --strike 520 --expiry 0.75 --model svi
python scripts/price_option.py AAPL --strike 185 --expiry 0.25 --model all
```

Output per run: `output/<TICKER>_<timestamp>/` with charts for history, greeks, smile, MC paths, model comparison.

---

## Python API

### Options

```python
from options import price, greeks, implied_vol, mc_price, binomial_price

c   = price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option="call")
g   = greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
iv  = implied_vol(100, 100, 1.0, 0.05, c, "call")
am  = binomial_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20,
                     option="put", style="american", n_steps=500)
mc  = mc_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20,
               option_type="european_call", n_sims=100_000)
```

### Exotic Options

```python
from exotics import price_barrier, price_asian_geo, price_lookback_float
from exotics import price_cash_or_nothing, price_one_touch

# Barrier: down-and-out call — barrier parity holds
res  = price_barrier(S=100, K=100, T=1.0, r=0.05, sigma=0.20,
                     H=85, option_type="call", barrier_type="down-out")

# Asian geometric (closed-form) — always cheaper than vanilla
geo  = price_asian_geo(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")

# Lookback floating — "bought at the minimum"
lb   = price_lookback_float(S=100, T=1.0, r=0.05, sigma=0.20, option_type="call")

# Digital: cash-or-nothing pays $1 if S_T > K
con  = price_cash_or_nothing(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
ot   = price_one_touch(S=100, T=1.0, r=0.05, sigma=0.20, H=85, touch_type="down")
```

### Forwards & Futures

```python
from forwards import forward_price, forward_value
from futures import FuturesContract

fwd  = forward_price(spot=100, maturity=1.0, rate=0.05, income_yield=0.02)
val  = forward_value(spot=105, delivery_price=100, maturity=1.0,
                     rate=0.05, income_yield=0.02, position="long")

es   = FuturesContract("ES", price=5000, maturity=0.25, contracts=2, multiplier=50)
pnl  = es.mtm_pnl(current_price=5010)
```

### Rates

```python
from rates import (
    DiscountCurve, InterestRateSwap, bootstrap_deposit_swap_curve,
    forward_swap_rate, price_swaption_black, price_swaption_bachelier,
    cap, floor, SABRParams, implied_vol_sabr, calibrate_sabr,
    fit_ns, fit_svensson,
)

curve = bootstrap_deposit_swap_curve(
    deposit_quotes={0.25: 0.041, 0.5: 0.042, 1.0: 0.043},
    swap_quotes={2.0: 0.044, 5.0: 0.047, 10.0: 0.049},
)

# Swaption: 1y expiry into 5y tenor, payer, K=5%, σ=20%
sw   = price_swaption_black(curve, T_exp=1.0, tenor=5.0, strike=0.05, sigma=0.20)
# sw: {'price': 17998, 'forward_swap_rate': 0.0506, 'annuity': 4.156, 'vega': 82839}

# 3-year cap at K=5%
cp   = cap(curve, maturity=3.0, strike=0.05, sigma=0.20)

# SABR smile
p    = SABRParams(alpha=0.04, beta=0.5, rho=-0.30, nu=0.40)
fit  = calibrate_sabr(forward=0.05, expiry=1.0, strikes=strikes, market_vols=vols, beta=0.5)

# Nelson-Siegel / Svensson
ns   = fit_ns(maturities, zero_rates)
sv   = fit_svensson(maturities, zero_rates)
```

### Credit

```python
from credit import HazardCurve, cds_value, risky_bond_price, cva_option

hc   = HazardCurve.bootstrap(
    tenors=[1, 3, 5, 7, 10], spreads_bps=[80, 120, 150, 170, 190],
    discount_curve=curve, recovery=0.40,
)

cds  = cds_value(hc, curve, maturity=5.0, spread=150,
                 position="buyer", notional=10_000_000)
# cds: {'value': ..., 'par_spread_bps': 143.2, 'cs01': ..., 'ir01': ...}

bond = risky_bond_price(face=1000, coupon_rate=0.045, maturity=5.0, hc=hc, dc=curve)
# bond: {'price': 972.3, 'yield_to_maturity': 0.052, ...}

cva  = cva_option(S=100, K=100, T=1.0, r_rate=0.05, sigma=0.20, hc=hc, dc=curve)
# cva: {'cva': 0.23, 'vanilla_price': 10.45, 'cva_adjusted_price': 10.22}
```

### Volatility

```python
from volatility import all_estimators, garch_fit, garch_forecast, vrp, vrp_summary
import yfinance as yf

df      = yf.download("AAPL", period="2y", auto_adjust=True)
vol_df  = all_estimators(df, window=21)
# columns: CC, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang, EWMA

import numpy as np
log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
res     = garch_fit(log_ret)
# GARCH(1,1): α=0.08, β=0.90, long-run vol=19.7%, half-life=43d

fcast   = garch_forecast(res, h=30)
# DataFrame: horizon, forecast_vol, vol_lb_95, vol_ub_95
```

### Structured Products

```python
from structured import cdo_structure, price_autocall, mbs_cashflows, mbs_price, oas
from rates import DiscountCurve

dc = DiscountCurve.flat(0.05, 10.0)

# CDO tranching (Gaussian copula)
tranches = cdo_structure(
    attachment_points=[0.0, 0.03, 0.07, 0.12, 0.22, 1.0],
    pd_1y=0.02, rho=0.20, discount_curve=dc, recovery=0.40, maturity=5.0,
)

# Autocallable note
res = price_autocall(
    S=100, r=0.05, sigma=0.20, T=1.0,
    obs_dates=[0.25, 0.50, 0.75, 1.0],
    autocall_level=1.00, ki_barrier=0.70, coupon_rate=0.08, n_sims=50_000,
)
# res.price=1.0143  res.expected_life=0.52y  res.prob_ki_loss=3.2%

# MBS
cf  = mbs_cashflows(face=1_000_000, wac=0.065, wam=360, psa_speed=100)
px  = mbs_price(cf, yield_=0.07)
z   = oas(98.5, cf, dc)
```

### Portfolio & Risk

```python
from portfolio import Portfolio, Position

port = Portfolio()
port.add(Position("call", params={"S":100,"K":100,"T":1.0,"r":0.05,"sigma":0.20}, quantity=10))
port.add(Position("put",  params={"S":100,"K":95, "T":1.0,"r":0.05,"sigma":0.20}, quantity=-5))

greeks = port.greeks()
# {'delta': 4.83, 'gamma': 0.062, 'vega': 318.4, 'theta': -56.1, 'rho': 37.2}

from portfolio.var import var_historical, var_parametric, var_cornish_fisher, var_monte_carlo

import numpy as np
returns = np.random.normal(0, 0.01, 500)

vh  = var_historical(returns, confidence=0.99)
vp  = var_parametric(returns, confidence=0.99)
vcf = var_cornish_fisher(returns, confidence=0.99)   # Cornish-Fisher skew/kurt adj.
vm  = var_monte_carlo(returns, confidence=0.99)
# vh: VaRResult(var=0.0231, cvar=0.0298, method='historical', confidence=0.99)

from portfolio.stress import stress_portfolio, STANDARD_SCENARIOS
results = stress_portfolio(port, STANDARD_SCENARIOS)
# 11 scenarios: Lehman, COVID, Black Monday, dot-com, GFC, ...
```

### ML / Quantitative Finance

```python
from ml import price_american_lsm, calibrate_svi, calibrate_ssvi, compare_models

# Longstaff-Schwartz: American put (GBM, 50k paths, Laguerre basis)
res = price_american_lsm(S=100, K=100, T=1.0, r=0.05, sigma=0.20,
                          option_type="put", n_sims=50_000, n_steps=100)
# res.price=5.08  res.early_exercise_premium=0.27  res.std_error=0.02

# SVI smile calibration
import numpy as np
log_strikes = np.linspace(-0.3, 0.3, 10)
mkt_var     = 0.04 + 0.01*log_strikes**2 - 0.005*log_strikes
svi_params, rmse = calibrate_svi(log_strikes, mkt_var)

# SSVI (surface SVI) — no-arbitrage across strikes AND tenors
ssvi_params, rmse = calibrate_ssvi(log_strikes, mkt_var, theta=0.04)

# Vol forecasting
from ml import compare_models
results = compare_models(realized_variance_series)
# {'HAR': {'rmse': 1.2e-5, 'qlike': 0.031}, 'GBM': {...}, 'LSTM': {...}}
```

### FX Options

```python
from fx import (
    fx_forward, price_gk, implied_vol_gk, put_call_parity_check,
    delta_to_strike, atm_dns_strike,
    FXSmileQuotes, build_smile, vanna_volga_price, price_fx_barrier_vv,
)

S, r_d, r_f, T = 1.08, 0.05, 0.03, 1.0

# Garman-Kohlhagen
F    = fx_forward(S, r_d, r_f, T)              # 1.1018
call = price_gk(S, K=1.10, T=T, r_d=r_d, r_f=r_f, sigma=0.08)
# call: {'price': 30_245, 'delta': 0.451, 'vanna': -0.87, 'volga': 0.28, ...}

# Implied vol
iv   = implied_vol_gk(market_price, S, K=1.10, T=T, r_d=r_d, r_f=r_f)

# FX smile from 25-delta market quotes
q     = FXSmileQuotes(S=S, T=T, r_d=r_d, r_f=r_f,
                       atm=0.080, rr25=0.010, bf25=0.003)
smile = build_smile(q)
# smile.strikes: [K_25P, K_ATM, K_25C]   smile.vols: [7.80%, 8.00%, 8.80%]

# Vanna-Volga smile-adjusted price
res  = price_gk(S, K=1.10, T=T, r_d=r_d, r_f=r_f, sigma=0.08, option_type="call", notional=1.0)
vv   = vanna_volga_price(res["unit_px"], res["vanna"], res["volga"], smile, "call")

# FX barrier with VV adjustment
bar  = price_fx_barrier_vv(S, K=1.10, H=1.02, T=T, r_d=r_d, r_f=r_f,
                            smile=smile, barrier_type="down-out", notional=1_000_000)
# bar: {'price': 28_140, 'price_bs': 27_850, 'vv_adj': 290}
```

### Commodities

```python
from commodities import (
    FuturesCurve, convenience_yield_curve, implied_convenience_yield,
    SchwartzParams, futures_price, calibrate_schwartz,
    simulate_schwartz, price_commodity_option,
    spread_option_margrabe, spread_option_kirk,
    crack_spread, spark_spread,
)
import numpy as np

# Futures curve
mats   = np.array([1/12, 3/12, 6/12, 1.0, 1.5, 2.0])
prices = np.array([80.0, 80.5, 81.2, 82.0, 82.5, 83.0])
curve  = FuturesCurve(maturities=mats, futures_prices=prices, spot=79.5, commodity="WTI")

print(curve.is_contango())               # True
print(curve.calendar_spread(0.5, 1.0))  # +0.80 (contango)
print(curve.roll_yield(0.25, 1.0))      # annualized roll yield

# Implied convenience yield
cy = implied_convenience_yield(F=82.0, spot=79.5, T=1.0, r=0.05, storage_cost=0.02)

# Schwartz 1F calibration
true_p = SchwartzParams(kappa=0.8, mu_star=np.log(80), sigma=0.30)
F_mkt  = futures_price(79.5, mats, true_p)
fitted, rmse = calibrate_schwartz(F_mkt, mats, 79.5)
# fitted.half_life() → 0.87y   fitted.long_run_price() → 82.1

# Monte Carlo commodity option
opt  = price_commodity_option(79.5, K=80, T=1.0, r=0.05, params=fitted,
                               option_type="call", n_sims=50_000)
# opt: {'price': 8.21, 'std_error': 0.04, 'conf_95_lo': 8.13, 'conf_95_hi': 8.29}

# Crack spread option (Margrabe: max(gasoline - crude, 0))
cs  = spread_option_margrabe(F1=102, F2=82, T=1.0,
                               sigma1=0.25, sigma2=0.22, rho=0.75, r=0.05)
# cs: {'price': 13.12, 'spread_vol': 14.5%}

# Kirk approximation (non-zero strike K)
ks  = spread_option_kirk(F1=102, F2=82, K=5.0, T=1.0,
                          sigma1=0.25, sigma2=0.22, rho=0.75, r=0.05)

# Industry spreads
print(crack_spread(crude_price=80, gasoline_price=100, heating_oil_price=95))
print(spark_spread(power_price=50, gas_price=4.0, heat_rate=7.5))
```

---

## Tests

```bash
pytest                        # 381 tests
pytest tests/test_fx.py -v    # 26 FX tests
pytest tests/test_commodities.py -v   # 31 commodities tests
pytest tests/test_portfolio.py -v     # 35 portfolio/VaR tests
pytest tests/test_ml.py -v            # 30 ML tests
pytest tests/test_rates_advanced.py -v # 44 swaption/cap/SABR/NS tests
pytest -k "swaption or sabr"
pytest -k "var or stress"
pytest --tb=short
```

---

## Project Structure

```text
core/              Heston, SVI, calibration, data loaders
options/           BS, greeks, IV, MC, binomial, vol surface, engine, charts
exotics/           Barrier, Asian, Lookback, Digital + charts
forwards/          Forward contracts, fair price, carry
futures/           Futures pricing, basis, MTM PnL
rates/             Discount curves, swaps, swaptions, caps/floors, SABR, NS/Svensson
credit/            Hazard curves, CDS, risky bonds, CVA/DVA
volatility/        Realized estimators, GARCH, variance swaps, VRP
structured/        CDO, autocallables, MBS/PSA
portfolio/         Positions, Greeks, VaR/CVaR, stress, PnL grid
ml/                Longstaff-Schwartz, SSVI, HAR + GBM + LSTM vol forecast
fx/                Garman-Kohlhagen, smile (RR/BF), vanna-volga
commodities/       Futures curves, convenience yield, Schwartz 1F, spread options
scripts/           11 CLI tools
tests/             381 tests across 13 test files
derivatives_engine.ipynb   62-cell notebook
```
