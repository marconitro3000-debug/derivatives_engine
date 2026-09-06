# Usage

Everything you can do with this repo: commands, the Jupyter workflow, the full
Python API, every configuration knob, and what to do when something breaks.

> Looking for the command list specifically — every mode, every flag, every
> script, with a one-line description of each? That is
> **[`COMMANDS.md`](COMMANDS.md)**. This file is the longer-form reference.

- [Install](#install)
- [Running the study](#running-the-study)
- [Jupyter](#jupyter)
- [Configuration reference](#configuration-reference)
- [Python API](#python-api)
- [Pricing an option](#pricing-an-option)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)

---

## Install

```bash
git clone <repo> && cd <repo>
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux

pip install -e .                  # numpy scipy pandas torch matplotlib yfinance
pip install -e ".[dev]"           # + pytest
```

Python 3.11+. Everything runs on CPU — there is no GPU path and none is needed:
a full fit is under two minutes on a laptop.

For the notebook:

```bash
pip install jupyter
```

---

## Running the study

```bash
python main.py
```

No server, nothing to configure. It asks what to do:

```
  [1]  Train a surface        fit a chain, score it, plot everything
  [2]  Price off a surface    query the one trained last time
  [3]  Compare trained models which archived run to use, and why
  [4]  Sweep architectures    is the network too small? fit several sizes
                              on one chain and table the answer
```

Every one of those is also a command, and every knob below is also a flag —
generated from `RunConfig` by [`cli.py`](../cli.py), so the two cannot drift:

```bash
python main.py --help                    # the whole list
python main.py train --ticker AAPL       # a mode, no prompts
python main.py sweep --sweep-seeds 3     # the capacity study
```

See [`COMMANDS.md`](COMMANDS.md) for the flag-by-flag reference and recipes.

Then which underlying:

* **Train** accepts any symbol, typed on the spot — it does not need to exist
  in `config.py` at all. Blank keeps whatever `ticker` is set to there.
* **Price** and **Compare** show what has actually been trained (scanned from
  `models/` and `results/`) and let you pick one, rather than asking you to
  remember a ticker you typed three runs ago:

  ```
    available: SPY, AAPL, QQQ
    ticker [SPY]:
  ```

**In an editor**, press Run on `main.py`. A VS Code launch configuration is
committed at `.vscode/launch.json`. Set `mode` in `config.py` to `"train"`,
`"price"` or `"compare"` to skip the menu, and `ticker` to skip that prompt too
— useful for a scheduled run, where nothing is there to answer a prompt anyway
(`main.py` detects a non-interactive process automatically, defaults to
`"train"`, and uses `config.ticker` without asking).

### [1] Train

1. downloads and cleans a live option chain (falling back to a generated one,
   loudly, if the network is unavailable)
2. de-Americanises the quotes on a binomial lattice
3. calibrates per-slice SVI and joint SSVI
4. trains the neural surface, and again against a flat prior (the ablation)
5. scores all four on fit **and** on static arbitrage
6. writes the report and figures to `results/`, and **archives** the run to
   `models/<ticker>_<timestamp>/`

Two output locations, on purpose:

| directory | what it is | overwritten? |
|---|---|---|
| `results/` | the *cursor* — whatever mode [2] reads to price | yes, every train run |
| `models/<ticker>_<timestamp>/` | the *archive* — one folder per run | never |

`results/` files:

| file | contents |
|---|---|
| `report.txt` | the full run, identical to what was printed |
| `training.png` | fit, generalisation gap, constraint and penalty curves — see below |
| `smiles.png` | market smiles per expiry with every surface overlaid |
| `arbitrage_neural_ssvi.png` / `_neural_flat.png` / `_svi.png` / `_ssvi.png` | Durrleman `g` and `dw/dT` maps, one per surface |
| `sweep.txt`, `sweep.png`, `<ticker>_sweep_chain.npz` | written by mode [4], if run |
| `<ticker>_surface.pt`, `<ticker>_chain.npz` | the model and the chain it was fitted to — both needed to price |
| `model_comparison.png` | written by mode [3], if run |

`models/<ticker>_<timestamp>/` holds the same `surface.pt` / `chain.npz` plus
`history.npz` (the full per-epoch training history) and `metrics.json` (the
numbers `runs_table` reads). Both directories are git-ignored — copy out
whatever you want to keep.

#### Reading `training.png`

Four panels, in the order you should read them:

1. **Fit error** — train and validation IV RMSE, with the epoch actually
   selected marked. The held-out strikes span the whole smile, wings included,
   so validation sits a little *above* training; validation *below* training is
   the signature of a split quietly holding out only the easy points, which is
   what this one used to do.
2. **Generalisation gap** — `val − train`, filled. Flat and small says the fit
   will hold up on a chain it has not seen; widening over epochs is the
   network starting to fit the training strikes at the validation strikes'
   expense, and it is exactly why validation error (not training error)
   chooses the marked epoch.
3. **Worst violation on collocation points** — the smallest `dw/dT` and
   Durrleman `g` found that epoch. Should cross into positive territory during
   the penalty warm-up and stay there. Hovering at zero means the constraints
   are actively fighting the fit; deeply positive throughout means they were
   never binding in the first place.
4. **Penalty terms**, log scale — a working run drives both to numerical zero.

### [2] Price

Loads `<ticker>_surface.pt` and `<ticker>_chain.npz` from `results/` and asks
for a maturity and a strike (or several, space-separated; blank strike gives a
ladder around the forward). For each one it prints implied vol, call and put
price, delta/vega/gamma/theta, and the two no-arbitrage quantities `dw/dT` and
Durrleman `g` evaluated at exactly that point — plus, if you're at a terminal
that can show a window, the smile with your strike marked on it so you can see
at a glance whether the answer came from interpolating real quotes or from
extrapolating past them.

Querying past twice the longest fitted expiry prints a warning: the answer
still degrades gracefully to the SSVI prior rather than to noise, but it is not
information the market gave you.

### [3] Compare

Lists every archived run for the configured ticker, ranked by validation error,
with the generalisation gap and the per-slice-SVI / joint-SSVI numbers on that
same chain alongside each one — the question this answers is "which of these
checkpoints should mode [2] actually use", after which you copy that run's
`surface.pt` / `chain.npz` into `results/` under the names mode [2] expects.
Optionally plots a bar-chart comparison (`model_comparison.png`) and, on
request, an overlay of the validation curves for up to six runs.

### [4] Sweep

Fits several architectures to **one** chain — same split, same budget, same
seed, same penalties — and tables validation error against parameter count. It
exists so "is the network too small?" gets answered with a curve instead of an
opinion. Writes `sweep.txt`, `sweep.png` and the exact chain every arm saw, so
any arm can be replayed with `--chain-file`.

```bash
python main.py sweep
python main.py sweep --sweep-architectures 64x3,256x4,1024x4 --sweep-seeds 3
```

The split is pinned across arms and seeds: letting the seed move it too makes
the seeds incomparable, and that scatter is larger than the architecture effect
being measured.

### Typical runtime

| step | time |
|---|---|
| fetch + clean + de-Americanise 1,400 quotes | ~15 s |
| SVI + SSVI calibration | ~60 s |
| neural training, 1,500 epochs | ~90 s |
| figures | ~15 s |
| **total (train)** | **~3 min** |
| price / compare | instant — no network, no training |
| sweep, 6 architectures x 3 seeds | ~40 min (once, on a cached chain) |

To iterate faster, either set `epochs = 400` and `run_baselines = False` in
`config.py`, or say the same thing on the command line and skip the download
entirely:

```bash
python main.py train --chain-file results/spy_chain.npz --epochs 400                      --no-run-baselines --no-run-prior-ablation
```

---

## Jupyter

`notebook.ipynb` at the repo root is the paper: the same study, written as a
document, with every claim backed by a cell you can re-run. It is committed
**with its outputs**, so it renders in full on GitHub without anyone running
anything.

```bash
jupyter lab notebook.ipynb        # or: jupyter notebook notebook.ipynb
```

Set `LIVE = False` in the first code cell to run the whole notebook on a
generated chain with no network access.

### Working from a blank notebook

```python
import volsurface as vs

chain  = vs.fetch_chain("SPY")
result = vs.train_surface(chain)

print(chain.summary())
print(chain.early_exercise_summary())
print(result.summary())
print(vs.comparison_table(vs.compare(chain, result)))
```

Plots return a matplotlib figure when no path is given, so they display inline:

```python
vs.plot_fit(chain, [result.model])
vs.plot_arbitrage_map(result.model, chain)
```

### Re-running the notebook from the command line

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 notebook.ipynb
```

Export it as a standalone document for sharing:

```bash
jupyter nbconvert --to html notebook.ipynb          # notebook.html
jupyter nbconvert --to pdf  notebook.ipynb          # needs a LaTeX install
```

---

## Configuration reference

Every setting lives in [`config.py`](../config.py) as one dataclass. Edit it and
re-run; there is no CLI, on purpose — a run should be reproducible by reading one
file rather than by remembering which flags were passed.

### What to do

| setting | default | meaning |
|---|---|---|
| `mode` | `"ask"` | `"ask"` shows the menu. `"train"` / `"price"` / `"compare"` skip straight to it. Falls back to `"train"` automatically when nothing is listening for input (a scheduled run). |
| `show_plots` | `True` | pop figures up in a window as well as saving them, when there is a human at the keyboard. Always off in a non-interactive run. |

### What to fit

| setting | default | meaning |
|---|---|---|
| `ticker` | `"SPY"` | underlying. Liquid ETFs and large caps work; illiquid names will not survive the quote filters. |
| `use_live_data` | `True` | `False` uses a chain generated from a known arbitrage-free SSVI surface. |
| `chain_file` | `None` | fit an already-downloaded `*_chain.npz` instead of fetching. Overrides `use_live_data`, and is what makes an experiment reproducible: every arm of a sweep has to see the same quotes, and two downloads a minute apart do not. |
| `max_expiries` | `8` | expiries are sampled evenly in `sqrt(T)` across the whole listed term structure, not taken from the front. |
| `min_maturity_years` | `0.02` | ~1 week. Shorter is dominated by tick size. |
| `max_maturity_years` | `2.0` | longer barely trades. |

### Quote filters

| setting | default | meaning |
|---|---|---|
| `min_open_interest` | `10` | a quote is kept if it clears this **or** has traded today. |
| `max_relative_spread` | `0.25` | drop quotes wider than 25% of their own mid — they carry no usable volatility. |
| `de_americanize` | `True` | strip the early-exercise premium before inverting. Turning this off biases SPY vols by ~13bp on average and ~25bp beyond a year. |
| `lattice_steps` | `150` | binomial steps for that correction. Ample, because the premium is a difference of two prices off the same lattice and the discretisation error cancels. |

### The network

| setting | default | meaning |
|---|---|---|
| `epochs` | `1500` | full-batch Adam with a cosine schedule. |
| `hidden_layers` | `(64, 64, 64)` | ~9k parameters, appropriate for a few thousand quotes. |
| `alpha` | `0.35` | maximum relative correction to the SSVI prior — about 16% in vol terms. The model's error bound, fixed before training. |
| `learning_rate` | `3e-3` | |
| `validation_fraction` | `0.2` | held out stratified **within each expiry**. |
| `seed` | `0` | runs are deterministic given the same chain. |

### No-arbitrage penalties

| setting | default | meaning |
|---|---|---|
| `calendar_weight` | `10.0` | weight on `relu(-dw/dT)^2`. |
| `butterfly_weight` | `1.0` | weight on `relu(-g)^2`. |
| `collocation_points` | `2048` | points per epoch, drawn from a region wider than the quotes and resampled every step. |

Defaults put the penalty terms one to two orders of magnitude below the fit term
at convergence: large enough to eliminate violations, small enough not to distort
the fit where no violation is at stake.

### Output

| setting | default | meaning |
|---|---|---|
| `output_dir` | `Path("results")` | the *cursor* — overwritten every train run; what mode [2] prices from. |
| `models_dir` | `Path("models")` | the *archive* — every train run gets its own `<ticker>_<timestamp>/` folder here, never overwritten; what mode [3] compares. |
| `make_plots` | `True` | writes `training.png`, `smiles.png`, one `arbitrage_*.png` per surface. |
| `save_model` | `True` | also controls whether the run is archived to `models_dir`. |
| `run_baselines` | `True` | SVI calibration is the slowest step; it is also the point of the comparison. |
| `run_prior_ablation` | `True` | retrain the same network against a flat constant-vol prior and add it to the scorecard — the experiment that says how much of the result is the prior and how much is the network. Costs a second training run. |
| `verbose` | `True` | print the training log. |

### Capacity sweep (mode `"sweep"`)

| setting | default | meaning |
|---|---|---|
| `sweep_architectures` | `("32x2","64x3","128x3","256x4","512x4")` | architectures to fit, as `WIDTHxDEPTH`, all on the same chain with the same budget. |
| `sweep_seeds` | `1` | seeds per architecture. One shows the shape of the curve; three is what you want before claiming two architectures differ, since run-to-run scatter on a real chain is a few basis points. The split is pinned across seeds, so only the model moves. |

### Pricing (mode `"price"`, non-interactive only)

Only read when `main.py` runs with nothing attached to stdin (a scheduled job).
In an interactive session these are just the first prompt's defaults — you
answer the prompts instead.

| setting | default | meaning |
|---|---|---|
| `price_maturities` | `(0.25, 0.5, 1.0)` | one price table per maturity, in years. |
| `price_strikes` | `()` | empty means a ladder around the forward at each maturity. |

---

## Python API

### Market data

```python
import volsurface as vs

chain = vs.fetch_chain("SPY", max_expiries=8, min_open_interest=10,
                       max_rel_spread=0.25, de_americanize=True)

chain = vs.synthetic_snapshot(ticker="TEST", noise_bps=25.0, seed=0)
```

`ChainSnapshot` holds aligned arrays, one entry per surviving quote:

| attribute | meaning |
|---|---|
| `k` | log-moneyness `log(K / F_T)` against the **fitted** forward |
| `T` | year fraction, ACT/365F |
| `iv` | implied vol actually fitted (de-Americanised) |
| `iv_european` | the same quotes inverted as if they were European |
| `weight` | fitting weight, ~vega/spread, mean-normalised |
| `strike`, `is_call`, `mid`, `mid_european`, `spread`, `vega` | per quote |
| `forwards`, `discounts` | `dict` keyed by `T` |
| `total_variance` | `iv**2 * T` |
| `maturities` | sorted array of `T` |
| `early_exercise_bp` | per-quote bias a European inversion would have had |

```python
chain.summary()
chain.early_exercise_summary()
chain.slice_at(chain.maturities[0])       # single-expiry sub-chain
chain.forward_at(0.75)                    # (F, DF) interpolated between expiries
len(chain)

chain.save("spy_chain.npz")
chain2 = vs.ChainSnapshot.load("spy_chain.npz")
```

Build from your own data instead of Yahoo:

```python
from volsurface.chain import build_snapshot
# expiry_chains: {"YYYY-MM-DD": (calls_df, puts_df)}
# each frame needs: strike, bid, ask, openInterest, volume
chain = build_snapshot("XYZ", spot=100.0, asof=date.today(),
                       expiry_chains=..., de_americanize=True)
```

### Surfaces

All three implement `vs.VolSurface`, so everything below works on any of them.

```python
svi    = vs.SVISliceSurface.fit(chain)
ssvi   = vs.SSVISurface.fit(chain)
result = vs.train_surface(chain, vs.TrainConfig(epochs=1500))
neural = result.model
```

```python
surface.total_variance(k, T)     # w(k, T)
surface.implied_vol(k, T)        # sqrt(w / T)
surface.call_price(k, T, forward, discount)
surface.dw_dT(k, T)              # calendar condition:  >= 0
surface.dw_dk(k, T)
surface.d2w_dk2(k, T)
```

On the neural surface those derivatives are exact (autograd); on the parametric
ones they are central differences.

```python
neural.save("surface.pt")
reloaded = vs.NeuralVolSurface.load("surface.pt")   # includes the SSVI prior
```

### Training

```python
cfg = vs.TrainConfig(
    epochs=1500, lr=3e-3, val_fraction=0.2, seed=0, log_every=250,
    penalties=vs.PenaltyWeights(calendar=10.0, butterfly=1.0, n_points=2048),
    model=vs.ModelConfig(hidden=(64, 64, 64), alpha=0.35),
)
result = vs.train_surface(chain, cfg, prior="ssvi")   # or prior="flat"
```

`prior="flat"` is the ablation: a constant-vol prior instead of SSVI, which
isolates how much of the result the parametric shape is responsible for.

`TrainResult` carries `.model`, `.prior_surface`, `.history` (per-epoch dict),
`.best_epoch`, `.best_val_rmse_bps`, `.elapsed_sec`, `.summary()`.

### Scoring

```python
vs.fit_report(surface, chain)      # RMSE / MAE / max in bp, price RMSE, % inside spread
vs.scan_arbitrage(surface, k_range=(-1.0, 0.6), T_range=(0.02, 2.0))

score  = vs.evaluate_surface(surface, chain)     # both, over the chain's own range
scores = vs.compare(chain, result)               # neural + both baselines
print(vs.comparison_table(scores))
```

### Plots

```python
vs.plot_fit(chain, [neural, svi, ssvi])          # returns a figure
vs.plot_fit(chain, [neural], "smiles.png")       # or writes a file
vs.plot_arbitrage_map(surface, chain)
vs.plot_training(result)                          # fit / gap / violations / penalties
vs.plot_quote(surface, chain, strike=450, T=0.5)  # smile with one strike marked
vs.plot_model_comparison(records)                 # bar chart across archived runs
```

### The model registry

Every `main.py` train run archives itself; this is the same mechanism from
code, useful for a sweep over hyperparameters you want to compare afterwards.

```python
record = vs.save_run("models", cfg, chain, result, vs.compare(chain, result))
# record.val_rmse_bps, record.generalization_gap_bps, record.dir, ...

records = vs.list_runs("models", ticker="SPY")   # newest first
print(vs.runs_table(records))                     # ranked by validation error

best = min(records, key=lambda r: r.val_rmse_bps)
model, chain = vs.load_run(best)
history = vs.load_history(best)                    # the full per-epoch dict
```

`cfg` only needs `.ticker` and `.seed` attributes — any small object with those
works, not just `config.RunConfig`.

### Pricing primitives

```python
vs.price(S, K, T, r, sigma, "call", q)
vs.greeks(S, K, T, r, sigma, q)                  # vega per 1%, theta per day, rho per 1%
vs.implied_vol(S, K, T, r, market_price, "call")
vs.binomial_price(S, K, T, r, sigma, "put", style="american", n_steps=500, q=0.0)
vs.mc_price(S, K, T, r, sigma, "european_call", n_sims=100_000)
vs.de_americanised_iv(spot, K, T, r, q, F, DF, mid, "put", sigma_european)
vs.carry_from_forward(spot, forward, discount, T)   # -> (r, q)
```

Those are the primitives. To price a *named* option off a fitted surface, use
`price_option` rather than assembling the forward and the discount factor by
hand — see the section below.

---

## Pricing an option

`price_option` is the one entry point that speaks in options rather than in
`(k, T)`. Give it a strike, an expiry and a side; it returns an `OptionQuote`.

```python
from volsurface import ChainSnapshot, NeuralVolSurface, price_option

chain   = ChainSnapshot.load("results/spy_chain.npz")
surface = NeuralVolSurface.load("results/spy_surface.pt")

q = price_option(surface, chain, strike=780, expiry="2026-12-19", kind="put")
print(q.summary())
```

```
SPY 780 put  exp 2026-12-19  (T = 0.2849y)
  surface fitted 2026-09-06

  implied vol       13.54%
  price            24.2961

  delta            -0.5110   (forward)
  delta_spot       -0.5151
  gamma           0.007094
  vega              1.6490   per 1% of vol
  theta            -0.1073   per day

  k = +0.0048   forward 776.2690   discount 0.998015
  within the quoted strike and maturity range
  dw/dT +2.298e-02   g +1.113e+00   (both must be >= 0)
```

### Naming the expiry

| You have | You pass | Notes |
|---|---|---|
| A calendar date | `"2026-12-19"`, `date(2026, 12, 19)`, a `datetime` | Measured from `chain.asof` |
| Days out | `"45d"` | Calendar days from `chain.asof` |
| A year fraction | `0.25`, `"0.25"` | Passed through; must be in `(0, 100]` |

A date is converted against the day the **chain** was snapshotted, not against
today. A surface fitted last Tuesday has to keep pricing a December expiry at
the maturity it had last Tuesday; measuring from today would quietly shorten
every maturity by the age of the chain, and would make the same call return a
different number tomorrow with no new data.

Two inputs are refused rather than answered. A maturity at or below zero has no
implied vol to look up — an expired option is an intrinsic-value question, not a
surface question — and a numeric expiry above 100 years is a date that lost its
dashes (`"2026"`, `"20261219"`), not an intent.

### What comes back

```python
q.strike, q.expiry, q.T, q.kind     # what was asked
q.forward, q.discount               # fitted from the chain, not assumed
q.log_moneyness                     # log(K / F_T)

q.implied_vol                       # the surface at exactly this (k, T)
q.price                             # DF * Black76(F, K, T, sigma)
q.delta, q.delta_spot
q.gamma, q.vega, q.theta            # vega per 1% of vol, theta per day

q.in_quoted_strikes                 # was there a quote near this strike
q.in_quoted_maturities              #   ... and near this expiry
q.is_extrapolated                   # not (both of the above)
q.dw_dT, q.butterfly_g              # the no-arbitrage conditions, right here
q.arbitrage_free                    # both >= 0
```

`OptionQuote` is a frozen dataclass, so a priced quote is a record of an answer
rather than a scratchpad — `dataclasses.asdict(q)` if you want to log it.

### The three things it does that a bare Black-Scholes call cannot

**The forward and the discount factor come from the chain.** Both were fitted
from put-call parity on the quotes, iterated to a fixed point against
de-Americanised prices. Pricing against a spot and an assumed carry would quote
the volatility against a different forward than the one the surface was
calibrated in — on an eighteen-month expiry that error is close to 1%.

**It says when the answer is extrapolated.** A strike or an expiry outside the
quoted range still gets a number: the bounded parametrisation guarantees the
surface degrades to the SSVI prior rather than to noise out there. But that
number is the model's opinion, not the market's, and `is_extrapolated` — and the
`! EXTRAPOLATED` line in `summary()` — says which of the two axes left the data.

**It reports the local no-arbitrage conditions.** The dense scan in
`diagnostics` answers "is this surface clean". `q.dw_dT` and `q.butterfly_g`
answer the narrower and more useful question: is the specific number about to be
traded on clean.

### Greeks convention

The surface lives in the forward measure, so the internally consistent set is
the sensitivities of the **discounted** price with respect to the **forward**:

```
price = DF * Black76(F, K, T, sigma)

delta = d(price)/dF          vega  = d(price)/d(sigma), per 1% of vol
gamma = d2(price)/dF2        theta = d(price)/dt, per calendar day
```

All four carry the discount factor — mixing discounted and undiscounted
sensitivities in one row is a percent-level error on delta at a long expiry and
is invisible in the output. `delta_spot` converts to a spot delta under
`dF/dS = F/S`, which holds here by construction, and is reported separately
rather than silently substituted.

### Any surface, not just the neural one

`price_option` takes a `VolSurface`, so the baselines price through the same
code path — which is the point, and the same reason the scorecard is trustworthy:

```python
from volsurface import SSVISurface, SVISliceSurface

for surf in (surface, SSVISurface.fit(chain), SVISliceSurface.fit(chain)):
    q = price_option(surf, chain, 780, "2026-12-19", "put")
    print(f"{surf.name:<34} {q.implied_vol:>7.2%}  {q.price:>9.4f}")
```

```
Neural (SSVI prior + penalties)     13.54%    24.2961
SSVI (joint)                        13.42%    24.1061
SVI (per-slice)                     13.74%    24.6287
```

---

## Tests

```bash
pytest                              # 126 tests, no network required
pytest tests/test_marketdata.py     # one module
pytest -k de_americanis             # one topic
pytest -x -q                        # stop at the first failure
```

Every test runs against synthetic data or a closed-form answer, so the suite is
deterministic and works offline.

---

## Troubleshooting

**`No quotes for X survived cleaning`** — the filters are too tight for that
name. Raise `max_relative_spread` to `0.5` and drop `min_open_interest` to `0`.
Illiquid single names often have no usable chain at all.

**`live chain unavailable`** — Yahoo rate-limits. Wait a minute, or set
`use_live_data = False` to run on generated data.

**`SSVI needs at least two usable expiries`** — fewer than two expiries survived
the filters. Widen `min_maturity_years` / `max_maturity_years`, raise
`max_expiries`, or loosen the quote filters.

**`implied vol is not identifiable at K=..., T=...`** — not a bug. The option's
vega is so small that its price does not distinguish volatilities at all, so the
inversion refuses rather than returning a meaningless number. The quote is
skipped automatically during chain building.

**Unicode errors printing the report on Windows** — `main.py` widens the console
to UTF-8 and keeps its own output ASCII. If you hit this in your own code, run
`chcp 65001` or use `python -X utf8`.

**A run is too slow** — set `run_baselines = False` (SVI calibration dominates),
lower `epochs`, or reduce `collocation_points` to `512`.

**Results differ between runs** — the chain changes intraday. Given the same
chain the fit is deterministic: `seed` controls the split, the initialisation and
the collocation sampling.
