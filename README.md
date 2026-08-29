# Neural Implied-Volatility Surface

A neural network that fits the implied-volatility surface of a live equity option
chain **and cannot price arbitrage into it**.

Fitting a vol surface is a trade-off, and every practitioner runs into it:

- Calibrate **SVI** to each expiry independently and you track the smiles
  beautifully — but nothing connects the slices, so interpolating between them
  produces calendar-spread arbitrage.
- Calibrate a joint **SSVI** surface and calendar arbitrage is impossible by
  construction — but three global shape parameters cannot follow a chain whose
  smile changes character between the front week and the one-year point.

This project closes that gap. A network learns a bounded correction to an SSVI
prior, trained under soft no-arbitrage penalties enforced on collocation points
that deliberately extend *beyond* the quoted strikes. It fits like the flexible
model and behaves like the safe one.

## Result on a live SPY chain

1,401 quotes across 8 expiries from 13 days to 1.8 years, 29 August 2026.
Every surface is scored by identical code:

```
surface                              IV RMSE   max err   px RMSE  in spread  cal viol  bfly viol
------------------------------------------------------------------------------------------------
Neural (SSVI prior + penalties)        32.4bp    737.1bp    0.3358      15.8%     0.00%      0.01%
SVI (per-slice)                        48.1bp    544.0bp    0.5745       5.7%     0.27%      0.05%
SSVI (joint)                          122.7bp   1549.1bp    1.3299       2.4%     0.00%      0.00%
```

The neural surface fits **1.5x closer than per-slice SVI and 3.8x closer than
SSVI**, while being effectively free of static arbitrage — the residual 0.01% is
a single grid point at the far edge of the extrapolation region, not a hole in
the fitted surface.

![smiles](docs/figures/smiles.png)

The shaded bands are where no options trade. That is the region a surface has to
invent, and where an unconstrained model quietly produces negative densities.

![arbitrage map](docs/figures/arbitrage_neural.png)

Durrleman's `g` and `dw/dT` across the whole plane. Red would be arbitrage;
compare against `docs/figures/arbitrage_svi.png`, where the naive per-slice
construction breaks down between listed expiries.

On a *synthetic* chain generated from a smooth SSVI ground truth, per-slice SVI
edges the network out on RMSE (20.8bp vs 24.3bp) and violates nothing. That is
the expected result and worth stating: the network earns its keep on real
quotes, where the smile is noisy and irregular, not on data that a parametric
model already describes exactly.

## How it works

### 1. The chain is cleaned properly (`marketdata/chain.py`)

Surface quality is decided here, not in the model.

- **The forward and discount factor are implied from the market**, not assumed.
  Put-call parity gives `C(K) - P(K) = DF * (F - K)`, a straight line in `K`; a
  weighted regression over the liquid matched strikes returns both `DF` and the
  forward the market is actually trading. This is what removes the systematic
  skew tilt that a wrong dividend assumption produces.
- **Out-of-the-money quotes only** — calls above the forward, puts below. The
  ITM wing carries the same information with a wider spread.
- **Implied vols are computed in-house**, from `mid / DF` in the forward
  measure, so they are consistent with the fitted forward rather than with the
  data vendor's own rate and dividend assumptions.
- **Every quote carries a fitting weight of roughly vega / spread.** Unweighted
  least squares in vol space chases deep wing options whose volatility is barely
  identified by their price.

### 2. The model is a bounded correction, not a free-form fit (`nn/model.py`)

```
w(k, T) = w_SSVI(k, T) * [ 1 + alpha * tanh( net(k, T) ) ]
```

Three properties follow from that single line:

- **Positivity is structural.** `w_SSVI > 0` and the bracket lives in
  `[1-alpha, 1+alpha]`, so total variance is positive everywhere. No clamping,
  no NaN in `sqrt(w)`.
- **Extrapolation degrades to SSVI, not to noise.** Far outside the quoted
  strikes the network saturates and the surface becomes a fixed multiple of an
  arbitrage-free parametric surface.
- **The error is bounded before training starts.** With `alpha = 0.35` the fit is
  never more than ~16% away from SSVI in vol terms — a guarantee you can state
  in advance.

The output layer is zero-initialised, so the model *starts* exactly at the prior
and a failed run degrades to SSVI rather than to garbage.

Inputs are `(k, sqrt(T), k/sqrt(T), k^2, k*sqrt(T))`. Standardised moneyness
`k/sqrt(T)` matters: it is the coordinate in which smiles across maturities look
alike, and handing it to the network saves it from learning the `sqrt(T)`
scaling from a few hundred points.

### 3. No-arbitrage is enforced where there is no data (`nn/arbitrage.py`)

The two static conditions become penalties:

```
calendar    dw/dT   >= 0
butterfly   g(k,T)  >= 0,   g = (1 - k w_k/2w)^2 - (w_k^2/4)(1/4 + 1/w) + w_kk/2
```

Each is `mean(relu(-violation)^2)`: zero when satisfied, quadratic in the depth
of the breach. All derivatives come from `torch.autograd` with
`create_graph=True` — exact, no finite-difference error floor to tune against.

**Where they are evaluated matters more than their weight.** They are imposed on
random collocation points over a region 30% wider in strike than the quotes and
past the last listed expiry, resampled every epoch. Constraining the surface only
where quotes exist is nearly free and nearly useless: the wings and the gaps
between expiries are exactly where an interpolating network invents negative
densities.

This is why the model runs in float64 with a smooth activation: the butterfly
penalty differentiates twice, and a ReLU network has zero second derivative
almost everywhere.

### 4. Everything is scored by the same code (`core/diagnostics.py`)

Neural, SVI and SSVI all implement `core.surface.VolSurface`, so they go through
identical fit reports and arbitrage scans. Results are reported in vol space
(what a paper quotes), in price space, and as the share of quotes re-priced
inside the bid-ask (what a desk asks about).

## Quick start

```bash
pip install -e ".[data,api,dev]"

# offline — no network needed
python -m scripts.fit_surface --synthetic --plot out/

# live chain
python -m scripts.fit_surface --ticker SPY --plot out/ --save out/spy.pt

# ablation: how much does the SSVI prior actually contribute?
python -m scripts.fit_surface --ticker SPY --prior flat
```

```python
from marketdata import fetch_chain
from nn import train_surface, compare, comparison_table

chain  = fetch_chain("SPY")
result = train_surface(chain)

print(comparison_table(compare(chain, result)))
print(result.model.implied_vol([-0.1, 0.0, 0.1], [0.5, 0.5, 0.5]))
```

### Service

```bash
uvicorn api.main:app --reload   # http://127.0.0.1:8000/docs
```

| endpoint | purpose |
|---|---|
| `POST /api/surface/fit` | calibrate a ticker, return a handle and the full scorecard |
| `POST /api/surface/iv` | implied vol at any `(k, T)` — **with `dw/dT` and `g` alongside** |
| `POST /api/surface/price` | price a listed or unlisted strike off the surface |
| `GET /api/surface/arbitrage` | rescan the fitted surface |
| `GET /api/surface/grid` | dense IV grid for plotting |
| `POST /api/price/black-scholes` | closed-form reference price and Greeks |

Calibration runs on demand and is cached under a handle; every other endpoint is
a lookup. That split mirrors a real vol service, where calibration runs on a
schedule and pricing runs on every request.

## Layout

```
marketdata/chain.py     option chain -> clean (k, T, IV) cloud; forward from parity
core/surface.py         the VolSurface interface: everything is total variance
core/diagnostics.py     arbitrage scans and fit reports, model-agnostic
baselines/svi.py        raw SVI (quasi-explicit calibration) and joint SSVI
nn/prior.py             SSVI in torch, differentiable end to end
nn/model.py             the network: prior x bounded correction
nn/arbitrage.py         autodiff no-arbitrage penalties on collocation points
nn/train.py             vega-weighted objective, penalty warm-up, early stopping
nn/evaluate.py          scorecard and plots
options/                Black-Scholes and implied vol (the surface's ground truth),
                        plus binomial and Monte Carlo as independent checks on them
api/                    FastAPI service
```

## Two things the tests pin down

**Implied vol is not always recoverable.** A deep ITM call at low vol is worth
its intrinsic value to the last bit of a float64: at `S=100, K=70, T=0.6`, every
sigma below ~7% produces the *identical* double. Any root finder returns a
number there and that number is meaningless, so `options.implied_vol` refuses
instead — a silently wrong IV would enter the surface fit as a real data point.
(`test_implied_vol_refuses_when_the_price_does_not_identify_a_vol`)

**SVI cannot be calibrated by a naive five-parameter search.** The objective has
long flat valleys where `rho -> -1` trades off against `sigma -> 0`, and a
simplex started at a fixed point walks into them and returns a degenerate slice
with a large negative `a`. It looks converged and prices nonsense. `baselines`
uses the Zeliade quasi-explicit method instead: for fixed `(m, sigma)` the model
is *linear* in the rest, so a constrained linear least squares sits inside a
2-D search that a multi-start simplex can actually cover.

```bash
pytest        # 74 tests, no network required
```

## References

- Gatheral (2004), *A parsimonious arbitrage-free implied volatility parameterization*
- Gatheral & Jacquier (2014), *Arbitrage-Free SVI Volatility Surfaces*
- Roper (2010), *Arbitrage Free Implied Volatility Surfaces*
- Zeliade Systems (2009), *Quasi-Explicit Calibration of Gatheral's SVI Model*
- Ackerer, Tagasovska & Vatter (2020), *Deep Smoothing of the Implied Volatility Surface*

## License

MIT
