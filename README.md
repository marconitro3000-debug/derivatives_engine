# options-pricer

> A modular Python library for derivatives pricing, implied-volatility calibration, and stochastic-volatility modelling.

Built as part of a quantitative finance portfolio. See [demo.ipynb](demo.ipynb) for a full paper-style walkthrough with LaTeX math and visualisations.

---

## Features

| Layer | What it does |
|-------|-------------|
| **Black–Scholes** | Analytical European pricing, all Greeks ($\Delta$, $\Gamma$, $\mathcal{V}$, $\Theta$, $\varrho$), put–call parity |
| **Implied volatility** | Newton–Raphson with Brent fallback; IV surface over $(K, T)$ grid |
| **Monte Carlo** | GBM paths, 5 exotic payoffs, antithetic + control-variate variance reduction |
| **Binomial tree** | CRR for European & American options; early-exercise premium |
| **Heston model** | Characteristic-function pricing (Gil-Pelaez), Feller condition check |
| **SVI model** | Per-maturity smile parametrisation, butterfly-arbitrage check |
| **Calibration** | Weighted least-squares, warm start, EWMA smoothing |
| **Persistence** | SQLite versioned history — full audit trail of parameter drift |
| **Data loaders** | `yfinance` (live) / synthetic (offline) behind one unified API |

---

## Quick Start

```bash
pip install -e ".[dev]"
```

```python
from options_pricer import price, greeks, implied_vol, mc_price, binomial_price

# Black-Scholes European call
c = price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option="call")
# → 10.4506

# All Greeks
g = greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
# → {'delta_call': 0.6368, 'gamma': 0.0188, 'vega': 0.3752, ...}

# Implied volatility (Newton-Raphson, < 5 iterations)
iv = implied_vol(S=100, K=100, T=1.0, r=0.05, market_price=10.4506)
# → 0.2000

# Asian option via Monte Carlo (antithetic variates)
res = mc_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20,
               option_type="asian_call", n_sims=200_000)
# → {'price': 7.53, 'std_error': 0.012, 'conf_95_lo': 7.51, 'conf_95_hi': 7.56}

# American put (CRR binomial tree)
res = binomial_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20,
                     option="put", style="american", n_steps=500)
# → {'price': 6.09, 'early_exercise': 0.54, ...}
```

### Heston pricing

```python
from options_pricer.models.heston import HestonParams, price as heston_price

p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.6)
print(p.feller_satisfied())          # True
c = heston_price(S=100, K=100, T=1.0, r=0.05, p=p, option="call")
```

### Self-updating calibration engine

```python
from options_pricer.engine import PricingEngine

eng = PricingEngine(model="heston", db_path="cals.db", source="yfinance")

eng.update("AAPL")                     # cold calibration, persisted to SQLite
eng.update("AAPL", ewma_alpha=0.3)    # warm-start + EWMA smoothing

eng.price_option("AAPL", K=190, T=0.5, option="call")
eng.implied_vol("AAPL", K=190, T=0.5)
eng.drift("AAPL", "kappa")            # parameter time series from SQLite
```

---

## Package Structure

```
options_pricer/
├── __init__.py              # Public API
├── engine.py                # PricingEngine
├── core/
│   ├── black_scholes.py     # Analytical pricing + Greeks
│   ├── implied_vol.py       # Newton-Raphson + Brent IV extraction
│   ├── monte_carlo.py       # GBM paths + variance reduction
│   └── binomial_tree.py     # CRR tree: European & American
├── surface/
│   └── vol_surface.py       # Spline / RBF vol surface interpolation
├── models/
│   ├── heston.py            # Heston model + char-function pricing
│   └── svi.py               # SVI smile + no-arbitrage check
├── calibration/
│   ├── calibrator.py        # Calibrators + EWMA blending
│   └── store.py             # SQLite versioned persistence
└── data/
    └── loader.py            # yfinance (live) / synthetic (offline)

tests/
├── test_pricer.py           # 37+ pricing tests
└── test_calibration.py      # Calibration recovery tests
```

---

## Supported Payoffs (Monte Carlo)

| Payoff | `option_type` |
|--------|--------------|
| European call / put | `european_call`, `european_put` |
| Asian (arithmetic average) | `asian_call`, `asian_put` |
| Down-and-out barrier | `barrier_call`, `barrier_put` |
| Lookback (floating strike) | `lookback_call`, `lookback_put` |
| Digital (cash-or-nothing) | `digital_call`, `digital_put` |

---

## Tests

```bash
pytest tests/ -v
```

37 tests covering: analytical prices, put-call parity, Greeks bounds, IV round-trip ($\sigma \to C_{BS} \to \hat\sigma$, error $< 10^{-10}$), MC convergence to BS, variance-reduction effectiveness, all exotic payoffs, American vs European premium, CRR convergence, surface interpolation, Heston put-call parity, Feller condition, SVI no-arbitrage, and calibration recovery on synthetic chains.

---

## Mathematics

### Black–Scholes

$$C = S_0 N(d_1) - K e^{-rT} N(d_2), \qquad d_1 = \frac{\ln(S/K) + (r+\sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

### Implied Volatility (Newton–Raphson)

$$\hat\sigma_{n+1} = \hat\sigma_n - \frac{C_{BS}(\hat\sigma_n) - C^{mkt}}{\mathcal{V}(\hat\sigma_n)}$$

Converges in 4–5 iterations from a Brenner–Subrahmanyam ATM seed.

### Monte Carlo GBM

$$S_{t+\Delta t} = S_t \exp\!\left[\left(r - \tfrac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\,Z\right], \quad Z \sim \mathcal{N}(0,1)$$

### CRR Binomial Tree

$$u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad p = \frac{e^{r\Delta t} - d}{u - d}$$

### Heston Stochastic Volatility

$$dS_t = r S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S, \qquad dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v$$

### SVI (Gatheral, 2004)

$$w(k) = a + b\!\left[\rho(k-m) + \sqrt{(k-m)^2 + \sigma^2}\right], \qquad k = \ln(K/F)$$

---

## Dependencies

```
numpy >= 1.24
scipy >= 1.10
matplotlib >= 3.7
yfinance >= 0.2        # live data (optional; synthetic loader works offline)
```

---

## References

1. Black & Scholes (1973). *The Pricing of Options and Corporate Liabilities.* **JPE**.
2. Cox, Ross & Rubinstein (1979). *Option Pricing: A Simplified Approach.* **JFE**.
3. Heston (1993). *A Closed-Form Solution for Options with Stochastic Volatility.* **RFS**.
4. Gatheral (2004). *A Parsimonious Arbitrage-Free Implied Volatility Parametrization.*
5. Albrecher et al. (2007). *The Little Heston Trap.* **Wilmott Magazine**.
