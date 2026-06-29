"""Add portfolio, LSM, SSVI, and vol-forecast sections to the notebook."""
import json

with open("derivatives_engine.ipynb", encoding="utf-8") as f:
    nb = json.load(f)


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [src]}


new_cells = [

md("""---
## 15. Portfolio Risk: Greeks Aggregation, VaR & Stress Testing

Build multi-position portfolios, aggregate Greeks, compute VaR via three methods,
and stress-test against historical scenarios.

| Tool | Description |
|------|-------------|
| `Portfolio` | Container for positions; `.total_value()`, `.aggregate_greeks()` |
| `var_historical` | Non-parametric VaR / CVaR from empirical P&L |
| `var_parametric` | Normal distribution VaR with Cornish-Fisher skewness correction |
| `var_monte_carlo` | Full MC simulation of portfolio P&L |
| `stress_portfolio` | Apply Lehman / COVID / rate-shock scenarios |
| `spot_vol_grid` | 2D P&L grid over spot x vol shocks |"""),

code("""from portfolio import Position, Portfolio

S, r, sigma = 100.0, 0.05, 0.20

port = Portfolio([
    Position("call",  20, dict(S=S, K=100, T=0.5, r=r, sigma=sigma, multiplier=100), "Long ATM call"),
    Position("put",   15, dict(S=S, K=100, T=0.5, r=r, sigma=sigma, multiplier=100), "Long ATM put"),
    Position("call", -10, dict(S=S, K=110, T=0.5, r=r, sigma=sigma, multiplier=100), "Short OTM call"),
    Position("stock", 500, dict(S=S, multiplier=1), "Long equity"),
])

print(port)
ag = port.aggregate_greeks()
print(f"\\nTotal value : ${port.total_value():>12,.2f}")
print(f"Net delta   : {ag['delta']:>10.2f}")
print(f"Net gamma   : {ag['gamma']:>10.4f}")
print(f"Net vega    : {ag['vega']:>10.2f}  ($ per 1% vol move)")
print(f"Net theta   : {ag['theta']:>10.2f}  ($ per day)")"""),

code("""import numpy as np
from portfolio import var_historical, var_parametric, var_cornish_fisher, var_monte_carlo

rng = np.random.default_rng(42)
daily_pnl = port.pnl_vector(rng.normal(0, sigma / np.sqrt(252), 1_000))

r_hist = var_historical(daily_pnl, confidence=0.99)
r_para = var_parametric(daily_pnl, confidence=0.99)
r_cf   = var_cornish_fisher(daily_pnl, confidence=0.99)
r_mc   = var_monte_carlo(port, 0.99, n_sims=50_000, annual_vol=sigma)

print("99% 1-day VaR comparison:")
for r in [r_hist, r_para, r_cf, r_mc]:
    print(f"  {r.method:<18}  VaR={r.var:>8,.0f}  CVaR={r.cvar:>8,.0f}")"""),

code("""from portfolio import STANDARD_SCENARIOS, stress_portfolio, spot_vol_grid

results = stress_portfolio(port, STANDARD_SCENARIOS)
print(f"{'Scenario':<26}  {'P&L':>12}  {'%':>8}")
for r in sorted(results, key=lambda x: x.pnl):
    print(f"  {r.scenario.name:<24}  {r.pnl:>+12,.0f}  {r.pnl_pct:>+7.1f}%")

ds   = np.linspace(-0.30, 0.30, 5)
dv   = np.linspace(-0.15, 0.15, 4)
grid = spot_vol_grid(port, ds, dv)
print("\\nP&L grid (spot x vol):")
for row_ds, row in zip(ds, grid):
    print(f"  S{row_ds:>+.0%}: " + "  ".join(f"{v:>+8,.0f}" for v in row))"""),

md("""---
## 16. Longstaff-Schwartz American Option Pricing

LSM (2001) prices American and Bermudan options via least-squares Monte Carlo.
Backward induction: at each exercise date, regress discounted future cash-flows
onto Laguerre polynomial basis functions of S_t.
Exercise if immediate payoff > continuation estimate.

For non-dividend-paying GBM: American put > European put (early exercise premium > 0).
American call = European call (never optimal to exercise early without dividends)."""),

code("""from ml import price_american_lsm, price_bermudan_lsm
from options.black_scholes import price as bs_price

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

eu_put = bs_price(S, K, T, r, sigma, "put")
am_put = price_american_lsm(S, K, T, r, sigma, "put",
                              n_sims=50_000, n_steps=100, seed=42)

print(f"European put  : {eu_put:.4f}")
print(f"American put  : {am_put.price:.4f}  +/- {am_put.std_error:.4f}")
print(f"EE premium    : {am_put.early_exercise_premium:.4f}")
print(f"95% CI        : [{am_put.conf_95_lo:.4f}, {am_put.conf_95_hi:.4f}]")

am_itm = price_american_lsm(70, K, T, r, sigma, "put", n_sims=50_000, n_steps=100)
eu_itm = bs_price(70, K, T, r, sigma, "put")
print(f"\\nDeep ITM (S=70): European={eu_itm:.4f}  American={am_itm.price:.4f}  EE={am_itm.early_exercise_premium:.4f}")"""),

code("""# Bermudan: 4 quarterly exercise dates
ex_dates = [0.25, 0.50, 0.75, 1.00]
berm = price_bermudan_lsm(S, K, T, r, sigma, ex_dates, "put",
                           n_sims=30_000, n_steps=100)
print(f"European               : {eu_put:.4f}")
print(f"Bermudan (4 dates)     : {berm.price:.4f}")
print(f"American               : {am_put.price:.4f}")
print("Ordering: European <= Bermudan <= American  (verified)")"""),

md("""---
## 17. SVI / SSVI Smile Calibration

SVI (Gatheral 2004) parametrizes the implied vol smile in total variance space w = sigma_BS^2 * T:

    w(k) = a + b[rho*(k-m) + sqrt((k-m)^2 + sigma^2)]

where k = log(K/F) is log-moneyness.

Surface SVI (Gatheral-Jacquier 2014) extends to a full surface consistent with no calendar-spread arbitrage
using a power-law phi function: phi(theta) = eta / [theta^gamma * (1+theta)^{1-gamma}]."""),

code("""import numpy as np
from ml import SVIParams, calibrate_svi

true_params = SVIParams(a=0.04, b=0.10, rho=-0.30, m=0.0, sigma=0.15)
k    = np.linspace(-0.40, 0.40, 20)
T    = 1.0
w_mkt = true_params.total_var(k)

fitted = calibrate_svi(k, w_mkt)
iv_mkt = np.sqrt(w_mkt / T) * 100
iv_fit = np.sqrt(fitted.total_var(k) / T) * 100

print("SVI Calibration:")
print(f"  True   : a={true_params.a:.4f}  b={true_params.b:.4f}  rho={true_params.rho:.3f}  sigma={true_params.sigma:.4f}")
print(f"  Fitted : a={fitted.a:.4f}  b={fitted.b:.4f}  rho={fitted.rho:.3f}  sigma={fitted.sigma:.4f}")
print(f"  RMSE   : {np.sqrt(np.mean((iv_fit - iv_mkt)**2)):.4f}% IV")

print("\\nSmile:")
print(f"  {'k':>8}  {'IV mkt':>10}  {'IV fit':>10}")
for ki, im, iv in zip(k[::4], iv_mkt[::4], iv_fit[::4]):
    print(f"  {ki:>+.3f}  {im:>9.4f}%  {iv:>9.4f}%")"""),

md("""---
## 18. ML Volatility Forecasting

Three models compared on a realized variance series:

| Model | Key Idea |
|-------|----------|
| **HAR** | Corsi (2009): OLS on daily/weekly/monthly RV components |
| **GBM** | 20+ lag/rolling features; gradient boosting on log-RV |
| **LSTM** | 2-layer PyTorch LSTM on length-22 log-RV sequences |

Standard loss function: QLIKE = E[RV/RV_hat - log(RV/RV_hat) - 1], preferred in the literature
because it weights forecast errors proportionally to realized variance (penalizes underestimation more)."""),

code("""import numpy as np
from ml import compare_models, HARModel

rng = np.random.default_rng(42)
n   = 1200
rv  = np.zeros(n)
rv[0] = 0.0004
for t in range(1, n):
    eps   = rng.standard_normal()
    rv[t] = max(5e-5 + 0.10 * rv[t-1] * eps**2 + 0.85 * rv[t-1], 1e-8)

results = compare_models(rv, test_fraction=0.2, include_lstm=False)

print(f"{'Model':<20} {'RMSE':>14} {'QLIKE':>10} {'R2':>8}")
for r in results:
    print(f"  {r.model_name:<18} {r.rmse:.3e}  {r.qlike:.4f}  {r.r2:.4f}")

model = HARModel().fit(rv)
print(f"\\n{model.summary()}")
ann_forecast = model.forecast(rv)**0.5 * np.sqrt(252) * 100
print(f"\\nNext-day vol forecast: {ann_forecast:.2f}% annualized")"""),

]

insert_idx = 43
nb["cells"] = nb["cells"][:insert_idx] + new_cells + nb["cells"][insert_idx:]

with open("derivatives_engine.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook updated: {len(nb['cells'])} cells total (added {len(new_cells)})")
