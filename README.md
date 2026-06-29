# Derivatives Engine

Modular Python library for derivatives pricing, implied-volatility calibration,
and stochastic-volatility modelling.

- [COMMANDS.md](COMMANDS.md): terminal commands with examples
- [ARCHITECTURE.md](ARCHITECTURE.md): module breakdown, maths, API notes
- [derivatives_engine.ipynb](derivatives_engine.ipynb): theory, formulas, and runnable examples

## Features

| Module | Capability |
|--------|------------|
| `options/` | Black-Scholes, implied vol, Monte Carlo, CRR binomial, vol surfaces, pricing engine |
| `core/models/` | Heston stochastic volatility and SVI smile parametrisation |
| `core/calibration/` | Heston/SVI calibration, warm starts, EWMA smoothing, SQLite audit trail |
| `core/data/` | Yahoo Finance loader and synthetic market-data loader |
| `forwards/` | Forward fair price, contract value, implied carry |
| `futures/` | Futures fair price, basis, daily mark-to-market PnL |
| `rates/` | Discount curves, zero/forward rates, bootstrap, vanilla swaps; swaptions (Black/Bachelier), caps/floors, SABR smile, Nelson-Siegel/Svensson curve fitting |
| `exotics/` | Barrier (Reiner-Rubinstein), Asian (geo CF + arithmetic MC), Lookback (Goldman), Digital (cash/asset-or-nothing, one-touch) |
| `credit/` | Hazard curves, CDS pricing/greeks, risky bonds, CVA/DVA |
| `volatility/` | Realized estimators (CC/Parkinson/GK/RS/YZ/EWMA), GARCH(1,1) MLE, variance swaps, VRP |
| `structured/` | CDO (Gaussian copula / LHP), autocallable notes (MC), MBS/PSA prepayment model |
| `portfolio/` | Position & Portfolio containers, Greeks aggregation, VaR/CVaR (historical/parametric/Cornish-Fisher/MC), Kupiec backtest, stress scenarios, spot×vol P&L grid |
| `ml/` | Longstaff-Schwartz American/Bermudan options, SVI/SSVI smile calibration, vol forecasting (HAR + Gradient Boosting + LSTM) |
| `fx/` | Garman-Kohlhagen FX option pricing, Greeks (vanna/volga), 25-delta/10-delta smile (RR/BF), vanna-volga exotic pricing, delta↔strike mapping |
| `commodities/` | Futures term structure (contango/backwardation), convenience yield, calendar spreads, crack/spark spreads, Schwartz (1997) mean-reversion model, spread options (Margrabe/Kirk) |

## Install

```bash
pip install -e ".[dev]"
pip install -e .
```

Python >= 3.11. Network access is required for live Yahoo Finance data.

## CLI

Fetches live spot and ATM IV from Yahoo Finance. Heston and SVI are calibrated
to the real option chain before pricing.

```bash
python scripts/price_option.py AAPL --strike 185 --expiry 0.25
python scripts/price_option.py SPY  --strike 550 --expiry 0.5  --model heston
python scripts/price_option.py TSLA --strike 250 --expiry 1.0  --type put  --style american
python scripts/price_option.py NVDA --strike 900 --expiry 0.25 --model black-scholes
python scripts/price_option.py GS   --strike 520 --expiry 0.75 --model svi
python scripts/price_option.py AAPL --strike 185 --expiry 0.25 --model all --style european
```

Each run saves everything to its own timestamped folder:

```text
output/AAPL_20250627_143022/
  history.png          # 52-week price + realized vs implied vol
  option_value.png     # BS value curves at T, 0.6T, 0.3T, 0.1T, payoff
  greeks.png           # 2×2: Δ Γ Θ Vega vs spot (academic style)
  pnl.png              # long option P&L at expiry with break-even
  model_comparison.png # bar chart comparing all model prices
  smile.png            # market IV smile with strike marker
  mc_paths.png         # 60 GBM sample paths + mean
  crr_convergence.png  # CRR price convergence vs tree depth
  heston_smile.png     # Heston calibrated smile vs market quotes
  svi_smile.png        # SVI calibrated smile vs market quotes
  results.txt          # full terminal log
```

| Argument | Default | Description |
|----------|---------|-------------|
| `ticker` | required | Yahoo Finance ticker |
| `--strike` / `-K` | required | Strike price |
| `--expiry` / `-T` | required | Years to expiry (`0.25` = 3 months) |
| `--type` | `call` | `call` or `put` |
| `--style` | `european` | `european` or `american` |
| `--model` | `all` | `all`, `black-scholes`, `heston`, `svi`, `monte-carlo`, `binomial` |
| `--vol` / `-v` | live ATM IV | Override implied vol (e.g. `0.20`) |
| `--spot` / `-S` | live price | Override spot price |
| `--rate` / `-r` | `0.045` | Risk-free rate |
| `--expiries` | `5` | Expiry dates for Heston/SVI calibration |

## Options

```python
from options import price, greeks, implied_vol, mc_price, binomial_price
from options.vol_surface import VolSurface, from_iv_dict

# European call (Black-Scholes)
c = price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option="call")

# All Greeks at once: Δ Γ Vega Θ ρ
g = greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.20)

# Implied vol (Newton-Raphson + Brent fallback)
iv = implied_vol(100, 100, 1.0, 0.05, c, "call")

# American put (CRR binomial tree, 500 steps)
am_put = binomial_price(
    S=100, K=100, T=1.0, r=0.05, sigma=0.20,
    option="put", style="american", n_steps=500,
)
# am_put: {'price': 10.58, 'early_exercise': 0.09}

# Monte Carlo: european, asian, barrier, lookback, digital
mc = mc_price(
    S=100, K=100, T=1.0, r=0.05, sigma=0.20,
    option_type="european_call", n_sims=100_000,
)
# mc: {'price': 10.44, 'std_error': 0.01, 'conf_95_lo': ..., 'conf_95_hi': ...}
```

### Academic charts from code

```python
import yfinance as yf
from options.charts import plot_history, plot_option_value, plot_greeks, plot_pnl

hist = yf.Ticker("AAPL").history(period="1y")

plot_history(hist, S=185, K=190, T=0.25, sigma=0.28,
             option_type="call", ticker="AAPL", out_dir="output/")
plot_option_value(S=185, K=190, T=0.25, r=0.045, sigma=0.28,
                  option_type="call", out_dir="output/",
                  style="european", premium=5.43, ticker="AAPL")
plot_greeks(S=185, K=190, T=0.25, r=0.045, sigma=0.28,
            option_type="call", out_dir="output/", ticker="AAPL")
plot_pnl(S=185, K=190, T=0.25, r=0.045, sigma=0.28,
         option_type="call", premium=5.43, out_dir="output/", ticker="AAPL")
```

## Calibration

```python
from core.calibration.calibrator import calibrate
from core.data.loader import YFinanceLoader
from options.engine import PricingEngine

loader = YFinanceLoader(risk_free_rate=0.045)
md = loader.load("AAPL", max_expiries=5)

heston_result = calibrate("heston", md)
svi_result = calibrate("svi", md)

engine = PricingEngine(model="heston", db_path="cals.db")
engine.update("AAPL")
px = engine.price_option("AAPL", K=190, T=0.5, option="call")
```

## Forwards and Futures

```python
from forwards import forward_price, forward_value
from futures import FuturesContract

fair_forward = forward_price(
    spot=100,
    maturity=1.0,
    rate=0.05,
    income_yield=0.02,
)

contract_value = forward_value(
    spot=105,
    delivery_price=100,
    maturity=1.0,
    rate=0.05,
    income_yield=0.02,
    position="long",
)

future = FuturesContract("ES", price=5000, maturity=0.25, contracts=2, multiplier=50)
daily_pnl = future.mtm_pnl(current_price=5010)
```

## Rates

```python
from rates import DiscountCurve, InterestRateSwap, bootstrap_deposit_swap_curve

curve = bootstrap_deposit_swap_curve(
    deposit_quotes={0.25: 0.041, 0.50: 0.042, 1.00: 0.043},
    swap_quotes={2.0: 0.044, 3.0: 0.045, 5.0: 0.047, 10.0: 0.049},
)

df = curve.discount_factor(5.0)
zero = curve.zero_rate(5.0)
forward = curve.forward_rate(1.0, 2.0)
par = curve.par_swap_rate(5.0)

swap = InterestRateSwap(1_000_000, fixed_rate=0.045, maturity=5.0)
pv = swap.pv(curve)
```

Generate curve charts and a text summary:

```bash
python scripts/analyze_rates.py
python scripts/analyze_rates.py --name usd_demo --deposit 0.25:0.052 --swap 5:0.046
```

Rates outputs are grouped separately:

```text
output/rates/<curve_name>/<run_id>/
```

## Advanced Rates

```python
from rates import (
    forward_swap_rate, price_swaption_black, price_swaption_bachelier,
    implied_black_vol, cap, floor, collar, cap_floor_parity_check,
    SABRParams, implied_vol_sabr, calibrate_sabr,
    fit_ns, fit_svensson, ns_yield, svensson_yield,
)
from rates import DiscountCurve

dc = DiscountCurve.flat(0.05, 20.0)

# Swaption: 1y into 5y, payer, K=5%, σ=20%
F, A = forward_swap_rate(dc, T_exp=1.0, tenor=5.0)
res  = price_swaption_black(dc, T_exp=1.0, tenor=5.0, strike=0.05, sigma=0.20)
# res: {'price': 17998.27, 'forward_swap_rate': 0.0506, 'annuity': 4.156, 'vega': 82839}

# Payer – Receiver parity = A × (F – K)
receiver = price_swaption_black(dc, 1.0, 5.0, 0.05, 0.20, "receiver")

# 3-year cap at K=5%, σ=20%
cap_res = cap(dc, maturity=3.0, strike=0.05, sigma=0.20)
# cap_res: {'price': 14231.0, 'n_caplets': 6, 'caplet_prices': [...], ...}

# SABR smile: calibrate to market vols
params = SABRParams(alpha=0.04, beta=0.5, rho=-0.30, nu=0.40)
import numpy as np
strikes = np.linspace(0.03, 0.07, 8)
mkt_v   = [implied_vol_sabr(0.05, K, 1.0, params) for K in strikes]
fitted  = calibrate_sabr(0.05, 1.0, strikes, mkt_v, beta=0.5)

# Nelson-Siegel / Svensson curve fitting
maturities = np.array([0.5, 1, 2, 3, 5, 7, 10, 20])
zero_rates  = np.array([0.042, 0.045, 0.048, 0.050, 0.053, 0.055, 0.057, 0.059])
ns  = fit_ns(maturities, zero_rates)
sv  = fit_svensson(maturities, zero_rates)
```

### Advanced rates CLI

```bash
# Swaption price + vol surface
python scripts/analyze_rates_advanced.py swaption --expiry 1 --tenor 5 --strike 0.05 --vol 0.20

# Cap/floor pricing with caplet breakdown
python scripts/analyze_rates_advanced.py capfloor --maturity 3 --strike 0.05 --vol 0.20

# SABR smile calibration
python scripts/analyze_rates_advanced.py sabr --forward 0.05 --expiry 1 --beta 0.5

# Nelson-Siegel and Svensson curve fitting
python scripts/analyze_rates_advanced.py ns
```

Output charts: `swaption_surface.png`, `capfloor_schedule.png`, `sabr_smile.png`, `ns_fit.png`

## Exotic options

```python
from exotics import price_barrier, price_asian_geo, price_lookback_float
from exotics import price_cash_or_nothing, price_one_touch

# Barrier: down-and-out call — cheaper than vanilla, knocked out if S drops to H
res = price_barrier(S=100, K=100, T=1.0, r=0.05, sigma=0.20,
                    H=85, option_type="call", barrier_type="down-out")
# res: {'price': 7.23, 'vanilla': 10.45, 'discount': 3.22}
# Parity: price_barrier(..., "down-in") + price_barrier(..., "down-out") = BS

# Asian geometric (exact closed-form) — always cheaper than vanilla
geo = price_asian_geo(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")

# Lookback floating call — "you bought at the lowest price"
lb = price_lookback_float(S=100, T=1.0, r=0.05, sigma=0.20, option_type="call")

# Cash-or-nothing: pays $1 if S_T > K  (= e^{-rT}·N(d2))
con = price_cash_or_nothing(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")

# One-touch: pays $1 if S ever reaches H=85
ot = price_one_touch(S=100, T=1.0, r=0.05, sigma=0.20, H=85, touch_type="down")
```

### Exotic CLI

```bash
# Barrier: down-and-out call
python scripts/price_exotic.py AAPL barrier --strike 185 --expiry 0.25 --barrier 170 --barrier-type down-out

# Asian (arithmetic MC + geometric closed-form + Kemna-Vorst)
python scripts/price_exotic.py SPY asian --strike 500 --expiry 1.0

# Lookback floating put
python scripts/price_exotic.py TSLA lookback --expiry 0.5 --type put

# Digital cash-or-nothing call
python scripts/price_exotic.py NVDA digital --strike 900 --expiry 0.25 --digital-type cash-or-nothing

# One-touch down
python scripts/price_exotic.py GS digital --expiry 0.25 --barrier 480 --digital-type one-touch
```

Output charts: `barrier_analysis.png`, `asian_paths.png`, `lookback_paths.png`,
`digital_payoff.png`, `exotic_comparison.png`

## Credit

```python
from credit import HazardCurve, cds_value, risky_bond_price, cva_option

# Bootstrap hazard curve from CDS quotes
hc = HazardCurve.bootstrap(
    tenors=[1.0, 3.0, 5.0, 7.0, 10.0],
    spreads_bps=[80, 120, 150, 170, 190],
    discount_curve=dc, recovery=0.40,
)

# CDS mark-to-market
result = cds_value(hc, dc, maturity=5.0, spread=150,
                   position="buyer", notional=10_000_000)
# result: {'value': ..., 'par_spread_bps': 143.2, 'cs01': ..., 'ir01': ...}

# Risky corporate bond (Duffie-Singleton)
bond = risky_bond_price(face=1000, coupon_rate=0.045, maturity=5.0, hc=hc, dc=dc)
# bond: {'price': 972.3, 'price_pct': 97.23, 'yield_to_maturity': 0.052, ...}

# CVA on an option
cva_res = cva_option(S=100, K=100, T=1.0, r_rate=0.05, sigma=0.20, hc=hc, dc=dc)
# cva_res: {'cva': 0.23, 'vanilla_price': 10.45, 'cva_adjusted_price': 10.22}
```

### Credit CLI

```bash
# CDS pricing + 3 charts
python scripts/price_credit.py IBM cds --maturity 5 --spread 150

# Bootstrap hazard curve
python scripts/price_credit.py AAPL curve --quotes 1:80 3:120 5:150 7:170 10:190

# Risky bond analysis
python scripts/price_credit.py GS bond --face 1000 --coupon 4.5 --maturity 5

# Option CVA
python scripts/price_credit.py MSFT cva --strike 420 --expiry 1.0
```

Output charts: `survival_curve.png`, `cds_legs.png`, `cva_profile.png`, `hazard_structure.png`

## Volatility

```python
from volatility import all_estimators, garch_fit, garch_forecast
from volatility import fair_variance_strike_mf, vrp, vrp_summary
import yfinance as yf

df = yf.download("AAPL", period="2y", auto_adjust=True)

# All 5 realized vol estimators (annualized, 21-day window)
vol_df = all_estimators(df, window=21)
# columns: CC, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang, EWMA(λ=0.94)

# Fit GARCH(1,1) via MLE
import numpy as np
log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
result = garch_fit(log_ret)
print(result.summary())
# GARCH(1,1) Fit
#   α = 0.0823, β = 0.9021, α+β = 0.9844 (persistence)
#   Long-run vol = 19.7%, Half-life = 43.6 days

# 30-day forecast cone
fcast = garch_forecast(result, h=30)
# DataFrame: horizon, forecast_vol, vol_lb_95, vol_ub_95

# VRP = ATM IV − realized vol
from volatility import vrp
vrp_ts = vrp(implied_vol_series, vol_df["Yang-Zhang"])
stats  = vrp_summary(vrp_ts)
# {'mean': 0.032, 'pct_pos': 0.78, 'sharpe': 1.4, ...}
```

### Volatility CLI

```bash
# Full analysis: all estimators + GARCH + VRP
python scripts/analyze_vol.py AAPL

# Custom window (30-day) and forecast horizon (60 days)
python scripts/analyze_vol.py SPY --window 30 --horizon 60

# Skip GARCH (faster)
python scripts/analyze_vol.py NVDA --no-garch
```

Output charts: `realized_vol.png`, `garch_forecast.png`, `garch_diagnostics.png`, `vrp.png`

## Structured Products

```python
from structured import cdo_structure, loss_distribution
from structured import price_autocall
from structured import mbs_cashflows, weighted_average_life, mbs_price, oas
from rates import DiscountCurve

dc = DiscountCurve.flat(0.05, 10.0)

# ── CDO (Gaussian Copula / LHP) ──────────────────────────────────────────────
tranches = cdo_structure(
    attachment_points=[0.0, 0.03, 0.07, 0.12, 0.22, 1.0],
    pd_1y=0.02, rho=0.20, discount_curve=dc, recovery=0.40, maturity=5.0,
)
for tr in tranches:
    print(f"{tr['name']:<14}: {tr['fair_spread_bps']:6.1f} bps  ETL={tr['etl_at_maturity']:.2%}")
# Equity       : 1243.7 bps  ETL=47.38%
# Mezzanine 1  :   82.3 bps  ETL= 1.52%
# Senior       :    0.1 bps  ETL= 0.00%

# Loss distribution
L, density = loss_distribution(pd_1y=0.02, rho=0.20, recovery=0.40)

# ── Autocallable Note (Monte Carlo) ──────────────────────────────────────────
result = price_autocall(
    S=100, r=0.05, sigma=0.20, T=1.0,
    obs_dates=[0.25, 0.50, 0.75, 1.0],
    autocall_level=1.00,    # autocall if S ≥ 100% of initial
    ki_barrier=0.70,        # knock-in if S < 70% of initial
    coupon_rate=0.08,       # 8% p.a.
    n_sims=50_000,
)
print(f"Price: {result.price*100:.2f}%  E[life]: {result.expected_life:.2f}y  P(KI): {result.prob_ki_loss:.2%}")
# Price: 101.43%  E[life]: 0.52y  P(KI): 3.21%

# ── MBS / PSA Prepayment ─────────────────────────────────────────────────────
cf_df = mbs_cashflows(face=1_000_000, wac=0.065, wam=360, psa_speed=100)
wal   = weighted_average_life(cf_df)      # → ~8.6 years
price = mbs_price(cf_df, yield_=0.07)    # priced at 7% yield
z     = oas(98.5, cf_df, dc)             # OAS in bps
print(f"WAL: {wal:.2f}y  Price@7%: {price:.2f}  OAS: {z:.1f} bps")
```

### Structured CLI

```bash
# CDO: Gaussian copula tranching + loss distribution
python scripts/price_structured.py cdo --pd 0.02 --rho 0.20 --maturity 5

# Autocallable on AAPL: 8% coupon, 70% KI, 100% autocall, quarterly obs
python scripts/price_structured.py autocall AAPL --coupon 0.08 --ki-barrier 0.70

# MBS: 30-year pool at 6.5% WAC, 100 PSA speed
python scripts/price_structured.py mbs --face 1000000 --wac 6.5 --wam 360 --psa 100
```

Output charts: `cdo_structure.png`, `cdo_sensitivity.png`, `autocall.png`, `mbs_cashflows.png`, `mbs_wal_sensitivity.png`

## Tests

```bash
pytest                                   # full suite (215 tests)
pytest tests/test_options.py -v          # 42 tests: BS, IV, MC, Binomial, Surface
pytest tests/test_calibration.py -v      # 19 tests: Heston, SVI, store, engine
pytest tests/test_exotics.py -v          # 32 tests: Barrier, Asian, Lookback, Digital
pytest tests/test_forwards_futures.py    # forwards + futures
pytest tests/test_rates.py               # rates curve
pytest tests/test_credit.py -v          # 30 tests: HazardCurve, CDS, Bond, CVA
pytest tests/test_volatility.py -v       # 37 tests: realized vol, GARCH, variance swap
pytest tests/test_structured.py -v       # 38 tests: CDO, Autocall, MBS
pytest -k "barrier" -v                   # filter by keyword
pytest --tb=short                        # condensed traceback
```

## Structure

```text
core/
  models/                # Heston, SVI
  calibration/           # calibrator, SQLite store
  data/                  # YFinanceLoader, SyntheticLoader
options/                 # option pricing, IV, surfaces, engine, charts
exotics/                 # barrier, asian, lookback, digital + charts
forwards/                # forward contracts, carry, contract value
futures/                 # futures, basis, mark-to-market PnL
rates/                   # curves, bootstrap, swaps, charts
credit/                  # hazard rates, CDS, bonds, CVA
volatility/              # realized vol, GARCH(1,1), variance swaps, VRP
structured/              # placeholder
scripts/price_option.py  # options CLI
scripts/price_exotic.py  # exotic options CLI
scripts/analyze_rates.py # rates curve CLI
scripts/price_credit.py  # credit (CDS, bond, CVA) CLI
scripts/analyze_vol.py   # volatility analysis CLI
tests/                   # unit tests
```
