# Architecture & Implementation

Technical reference for `options-pricer`. Covers module design, mathematics, public API, data flow, and implementation decisions.

For a narrative walkthrough with runnable code see [paper.ipynb](paper.ipynb).

---

## Table of contents

1. [Data flow](#data-flow)
2. [core/black_scholes.py](#coreblack_scholespy)
3. [core/implied_vol.py](#coreimplied_volpy)
4. [core/monte_carlo.py](#coremonte_carlopy)
5. [core/binomial_tree.py](#corebinomial_treepy)
6. [models/heston.py](#modelshestonpy)
7. [models/svi.py](#modelssvipy)
8. [calibration/calibrator.py](#calibrationcalibratorpy)
9. [calibration/store.py](#calibrationstorepy)
10. [surface/vol_surface.py](#surfacevol_surfacepy)
11. [data/loader.py](#dataloaderpy)
12. [engine.py](#enginepy)
13. [References](#references)

---

## Data flow

```
Yahoo Finance (yfinance)
        │
        ▼
  YFinanceLoader.load(ticker)
        │  filters to liquid options, computes mid IV
        ▼
    MarketData(spot, r, strikes[], maturities[], ivs[], weights[])
        │
        ├──► calibrate("heston", md)  ──► HestonParams  ──► heston.price()
        │
        ├──► calibrate("svi",    md)  ──► SVIParams[]   ──► svi.implied_vol_svi() ──► bs.price()
        │
        └──► (direct)  ──► bs.price() / mc_price() / binomial_price()

CalibrationStore (SQLite)
        │  persists every CalibrationResult with timestamp
        └──► warm_start for next calibration · drift() time series
```

---

## `core/black_scholes.py`

Analytical closed-form pricing for European options and all first-order Greeks.

### Pricing formula

$$C = S_0 N(d_1) - K e^{-rT} N(d_2)$$
$$P = K e^{-rT} N(-d_2) - S_0 N(-d_1)$$

$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)\,T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

### Greeks

| Greek | Formula |
|-------|---------|
| $\Delta_C$ | $N(d_1)$ |
| $\Delta_P$ | $N(d_1) - 1$ |
| $\Gamma$ | $\frac{n(d_1)}{S\,\sigma\sqrt{T}}$ |
| $\mathcal{V}$ | $S\,n(d_1)\sqrt{T}$ |
| $\Theta_C$ | $-\frac{S\,n(d_1)\,\sigma}{2\sqrt{T}} - r K e^{-rT} N(d_2)$ |
| $\rho_C$ | $K T e^{-rT} N(d_2)$ |

### API

```python
from options_pricer import price, greeks, put_call_parity_check

price(S, K, T, r, sigma, option)   # -> float
greeks(S, K, T, r, sigma)          # -> dict with delta_call/put, gamma, vega, theta_call/put, rho_call/put
put_call_parity_check(S, K, T, r, sigma)  # -> bool
```

---

## `core/implied_vol.py`

Extracts the implied volatility $\hat\sigma$ from a market price by inverting Black-Scholes.

### Algorithm

**Primary:** Newton-Raphson on $f(\sigma) = C_{BS}(\sigma) - C^{mkt} = 0$

$$\hat\sigma_{n+1} = \hat\sigma_n - \frac{C_{BS}(\hat\sigma_n) - C^{mkt}}{\mathcal{V}(\hat\sigma_n)}$$

Seed: Brenner-Subrahmanyam ATM approximation $\hat\sigma_0 \approx \sqrt{2\pi/T}\,(C/S)$.

**Fallback:** Brent's method when Newton-Raphson fails (deep ITM/OTM, near-zero vega). Guaranteed to converge within $[\sigma_{lo},\, \sigma_{hi}]$.

Typical convergence: 4-5 Newton iterations. Round-trip error $|\hat\sigma - \sigma^{mkt}| < 10^{-10}$ on all liquid strikes.

### API

```python
from options_pricer import implied_vol, iv_surface

implied_vol(S, K, T, r, market_price, option="call")  # -> float
iv_surface(S, strikes, maturities, r, prices)          # -> 2-D array (n_strikes, n_mats)
```

---

## `core/monte_carlo.py`

Prices European and exotic options by simulating GBM paths under the risk-neutral measure.

### GBM discretisation (Euler-Maruyama)

$$S_{t + \Delta t} = S_t \exp\!\left[\left(r - \tfrac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\,Z\right], \quad Z \sim \mathcal{N}(0,1)$$

### Variance reduction

**Antithetic variates:** for each $Z$ also simulate $-Z$; average the two payoffs. Halves variance with zero extra model evaluations.

**Control variate:** use the known European BS price as control. Reduces MC standard error by 40-70% for near-ATM options.

### Supported payoffs

| `option_type` | Payoff |
|--------------|--------|
| `european_call/put` | $\max(S_T - K, 0)$ |
| `asian_call/put` | $\max(\bar S - K, 0)$ where $\bar S$ is arithmetic average |
| `barrier_call/put` | European payoff, knocked out if $S_t \leq B$ at any step |
| `lookback_call/put` | $\max(S_T - S_{\min}, 0)$ floating-strike lookback |
| `digital_call/put` | $\mathbf{1}[S_T > K]$ cash-or-nothing |

### API

```python
from options_pricer import mc_price

res = mc_price(S, K, T, r, sigma, option_type, n_sims=100_000, seed=None)
# res: {'price': float, 'std_error': float, 'conf_95_lo': float, 'conf_95_hi': float}
```

---

## `core/binomial_tree.py`

Cox-Ross-Rubinstein (CRR) tree for European and American options. Supports early-exercise for American puts (and theoretically calls on dividend-paying underlyings).

### Tree parameters

$$u = e^{\sigma\sqrt{\Delta t}}, \quad d = \frac{1}{u}, \quad p = \frac{e^{r\,\Delta t} - d}{u - d}$$

At each node, the American value is:

$$V_{i,j} = \max\!\left(\text{intrinsic},\; e^{-r\,\Delta t}\left[p\,V_{i+1,j+1} + (1-p)\,V_{i+1,j}\right]\right)$$

Convergence: CRR converges to BS at $O(1/N)$; oscillations occur at even/odd $N$. Use $N \geq 200$ for stable prices.

### API

```python
from options_pricer import binomial_price

res = binomial_price(S, K, T, r, sigma, option, style, n_steps)
# option: 'call' | 'put'
# style:  'european' | 'american'
# res: {'price': float, 'early_exercise': float}
```

---

## `models/heston.py`

Stochastic volatility model (Heston 1993). Prices options analytically via the characteristic function.

### Dynamics

$$dS_t = r\,S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S$$
$$dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v, \qquad \langle dW^S, dW^v \rangle = \rho\,dt$$

| Parameter | Meaning |
|-----------|---------|
| $v_0$ | Initial variance |
| $\kappa$ | Mean-reversion speed |
| $\theta$ | Long-run variance |
| $\xi$ | Vol-of-vol |
| $\rho$ | Spot-vol correlation (typically $< 0$ for equities) |

**Feller condition:** $2\kappa\theta \geq \xi^2$ ensures $v_t > 0$ a.s. Often violated on real data (the process hits zero but remains well-defined).

### Pricing (Gil-Pelaez inversion)

$$C = S_0 P_1 - K e^{-rT} P_2$$

where $P_{1,2}$ are risk-neutral probabilities obtained by Fourier inversion of the characteristic function:

$$P_j = \frac{1}{2} + \frac{1}{\pi}\int_0^\infty \text{Re}\!\left[\frac{e^{-i\phi\ln K}\,\varphi_j(\phi)}{i\phi}\right] d\phi$$

The characteristic function $\varphi$ is known in closed form (see Albrecher et al. 2007 for the numerically stable "little trap" formulation used here).

### API

```python
from options_pricer.models.heston import HestonParams, price as heston_price

p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.6)
p.feller_satisfied()                              # bool
c = heston_price(S, K, T, r, p, option="call")   # float
```

---

## `models/svi.py`

Gatheral's Stochastic Volatility Inspired (SVI) parametrisation. Defined per maturity slice; fits the smile in total-variance space.

### Parametrisation

$$w(k) = a + b\left[\rho(k - m) + \sqrt{(k-m)^2 + \sigma^2}\right], \qquad k = \ln(K/F)$$

$w = \sigma_{IV}^2 T$ is total implied variance. The five parameters $(a, b, \rho, m, \sigma)$ control:

| Parameter | Effect |
|-----------|--------|
| $a$ | Overall variance level |
| $b$ | Smile steepness |
| $\rho$ | Skew (tilt) |
| $m$ | ATM shift |
| $\sigma$ | Smile curvature |

**No-arbitrage (butterfly):** sufficient condition $b(1 + |\rho|) \leq 4$, verified by `is_butterfly_arbitrage_free()`.

### API

```python
from options_pricer.models.svi import SVIParams, implied_vol_svi, is_butterfly_arbitrage_free

p = SVIParams(a=0.04, b=0.4, rho=-0.3, m=0.0, sigma=0.2)
k = np.log(strikes / forward)
iv = implied_vol_svi(k, T, p)          # array of implied vols
is_butterfly_arbitrage_free(p)          # bool
```

---

## `calibration/calibrator.py`

Unified calibration interface for both models. Minimises weighted RMSE between model and market implied vols.

### Objective

$$\min_\theta \sum_i w_i \left(\sigma_i^{model}(\theta) - \sigma_i^{mkt}\right)^2$$

Weights $w_i = \sqrt{\text{volume}_i + 1}$ — more liquid options carry more weight.

**Heston:** calibrates $\theta = (v_0, \kappa, \theta, \xi, \rho)$ jointly across all maturities. Uses price-space residuals internally (more numerically stable than IV space for deep OTM options). Reports in IV space.

**SVI:** calibrates one slice $(a, b, \rho, m, \sigma)$ per maturity independently. Much faster (~0.03s vs ~40s for Heston) and always butterfly-arb-free.

### Warm start and EWMA

```python
result = calibrate("heston", market_data, warm_start=prev_params, ewma_alpha=0.3)
# alpha=0.3: result.params = 0.3 * new + 0.7 * old  (dampens day-to-day noise)
```

### API

```python
from options_pricer.calibration.calibrator import calibrate, MarketData, blend_params

md = MarketData(spot, r, strikes, maturities, ivs, weights)
result = calibrate("heston" | "svi", md, warm_start=None, ewma_alpha=None)

result.params       # dict of fitted parameters
result.rmse         # root-mean-square IV error
result.max_error    # worst single-point error
result.n_points     # number of market points used
result.arb_free     # passed no-arbitrage check
result.elapsed_sec  # wall-clock time
result.summary()    # formatted string
```

---

## `calibration/store.py`

SQLite-backed versioned history. Every calibration is appended as a new row — nothing is ever overwritten.

### Schema

```sql
calibrations(id, ticker, model, timestamp, spot, r, params_json,
             rmse, max_error, n_points, arb_free, warm_started, elapsed_sec)
```

### API

```python
from options_pricer.calibration.store import CalibrationStore

store = CalibrationStore("cals.db")
store.save(ticker, result, spot, r)
store.latest_params(ticker, model)         # dict or None
store.history(ticker, model, limit=100)    # list of dicts
store.param_timeseries(ticker, model, param_name)  # list of (timestamp, value)
store.tickers()                            # list of calibrated tickers
```

---

## `surface/vol_surface.py`

Non-parametric interpolation of the IV surface from a sparse set of market quotes. Alternative to the parametric Heston/SVI approach when you want a model-free surface.

### Methods

**`spline`:** `scipy.interpolate.RectBivariateSpline` on a regular $(K, T)$ grid. Fast; requires the grid to be rectangular (same strikes at each maturity).

**`rbf`:** `scipy.interpolate.RBFInterpolator` with thin-plate spline kernel. Works on scattered $(K_i, T_i)$ observations; slower but more flexible.

NaN cells (missing quotes) are filled by column mean before fitting.

### API

```python
from options_pricer import VolSurface, from_iv_dict

surf = from_iv_dict(S=spot, iv_dict={(K, T): iv, ...}, method="spline")

surf.iv(K, T)              # float or array — interpolated IV
surf.smile(T)              # (strikes, ivs) at fixed maturity
surf.term_structure(K)     # (maturities, ivs) at fixed strike
surf.grid(n_strikes, n_maturities)  # (K_grid, T_grid, IV_grid) for plotting
```

---

## `data/loader.py`

### `YFinanceLoader`

Fetches real option chains from Yahoo Finance via `yfinance`. Applies liquidity filters:

- `volume >= min_volume` (default 10)
- `impliedVolatility > 0.01`
- Strike within `moneyness_range` of spot (default 0.85-1.15)
- Maturity `>= min_maturity` years (default 0.04, ~2 weeks; filters 0DTE)
- Subsamples to `max_per_expiry` strikes per expiry (default 20) for calibration speed

```python
from options_pricer.data.loader import YFinanceLoader

loader = YFinanceLoader(risk_free_rate=0.045)
md = loader.load("AAPL", max_expiries=5)   # MarketData
```

### `SyntheticLoader`

Generates a synthetic option chain from known Heston parameters. Used in tests to verify calibration recovers the true params.

```python
from options_pricer.data.loader import SyntheticLoader
from options_pricer.models.heston import HestonParams

loader = SyntheticLoader()
md = loader.load(spot=100, true_params=HestonParams(...), noise=0.002)
md.true_params  # ground truth for test assertions
```

---

## `engine.py`

High-level orchestrator that ties together loader, calibrator, store, and pricing. The recommended entry point for production use.

```python
from options_pricer.engine import PricingEngine

eng = PricingEngine(model="heston", db_path="cals.db", risk_free_rate=0.045)

# Calibrate and persist
eng.update("AAPL")                       # cold calibration
eng.update("AAPL", ewma_alpha=0.3)       # warm start + EWMA smoothing

# Price from the live calibrated surface
eng.price_option("AAPL", K=190, T=0.5, option="call")   # float
eng.implied_vol("AAPL", K=190, T=0.5)                    # float

# Audit trail
eng.drift("AAPL", "kappa")   # [(timestamp, value), ...]
eng.history("AAPL")          # list of full calibration records
```

SVI maturity interpolation: the engine interpolates in total-variance space between the two nearest calibrated slices, which preserves calendar-spread no-arbitrage.

---

## References

1. Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy, 81(3), 637–654.
2. Cox, J., Ross, S. & Rubinstein, M. (1979). *Option Pricing: A Simplified Approach.* Journal of Financial Economics, 7(3), 229–263.
3. Heston, S. (1993). *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.* Review of Financial Studies, 6(2), 327–343.
4. Gatheral, J. (2004). *A Parsimonious Arbitrage-Free Implied Volatility Parametrization with Application to the Valuation of Volatility Derivatives.* Presentation at Global Derivatives & Risk Management.
5. Albrecher, H., Mayer, P., Schachermayer, W. & Teichmann, J. (2007). *The Little Heston Trap.* Wilmott Magazine, Jan/Feb, 83–92.
6. Brenner, M. & Subrahmanyam, M. (1988). *A Simple Formula to Compute the Implied Standard Deviation.* Financial Analysts Journal, 44(5), 80–83.
