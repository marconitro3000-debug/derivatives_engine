"""
price_option.py  --  Price a single option from live market data.

Spot price and ATM implied vol are fetched live from Yahoo Finance.
Heston and SVI are calibrated to the real option chain before pricing.
Results and plots are saved to output/<TICKER_YYYYMMDD_HHMMSS>/.

Usage
-----
    python price_option.py AAPL --strike 185 --expiry 0.25
    python price_option.py SPY  --strike 500 --expiry 0.5  --model heston
    python price_option.py TSLA --strike 250 --expiry 1.0  --type put --model all
    python price_option.py NVDA --strike 900 --expiry 0.25 --model black-scholes
    python price_option.py GS   --strike 520 --expiry 0.75 --model svi --type call
"""

import argparse
import io
import os
import sys
from datetime import datetime

import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from options_pricer import price, greeks, mc_price, binomial_price
from options_pricer.core.implied_vol import implied_vol as bs_iv
from options_pricer.calibration.calibrator import calibrate
from options_pricer.data.loader import YFinanceLoader


# -- CLI -----------------------------------------------------------------------

parser = argparse.ArgumentParser(
    prog="price_option.py",
    description="Price a single option using live Yahoo Finance data.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument("ticker",
                    help="Underlying ticker symbol (e.g. AAPL, SPY, TSLA)")
parser.add_argument("--strike",  "-K", type=float, required=True,
                    help="Strike price")
parser.add_argument("--expiry",  "-T", type=float, required=True,
                    help="Time to expiry in years (0.25 = 3 months, 1.0 = 1 year)")
parser.add_argument("--type",    choices=["call", "put"], default="call",
                    help="Option type (default: call)")
parser.add_argument("--model",   choices=["all", "black-scholes", "heston", "svi",
                                          "monte-carlo", "binomial"],
                    default="all",
                    help="Pricing model (default: all). 'all' runs every model.")
parser.add_argument("--vol",     "-v", type=float, default=None,
                    help="Override implied vol, e.g. 0.20 for 20%%. "
                         "Default: fetches live ATM IV from the option chain.")
parser.add_argument("--spot",    "-S", type=float, default=None,
                    help="Override spot price. Default: fetches live from Yahoo Finance.")
parser.add_argument("--rate",    "-r", type=float, default=0.045,
                    help="Risk-free rate (default: 0.045)")
parser.add_argument("--expiries",      type=int,   default=5,
                    help="Number of option-chain expiries for Heston/SVI calibration "
                         "(default: 5). More = richer surface, slower.")
args = parser.parse_args()

TICKER = args.ticker.upper()
K      = args.strike
T      = args.expiry
r      = args.rate
otype  = args.type
MODEL  = args.model


# -- Run code + output directory -----------------------------------------------

RUN_CODE = f"{TICKER}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", RUN_CODE)
os.makedirs(OUT_DIR, exist_ok=True)


# -- Tee stdout to log buffer --------------------------------------------------

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, d):
        for s in self.streams: s.write(d)
    def flush(self):
        for s in self.streams: s.flush()

_log = io.StringIO()
sys.stdout = _Tee(sys.__stdout__, _log)


# -- Matplotlib style ----------------------------------------------------------

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.35,
    'lines.linewidth': 1.8,
})
_C = plt.rcParams['axes.prop_cycle'].by_key()['color']

def _save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return name


# -- Fetch live spot -----------------------------------------------------------

tk = yf.Ticker(TICKER)

if args.spot is not None:
    S = args.spot
    print(f"Spot (manual):   ${S:.2f}")
else:
    hist = tk.history(period="1d")
    if hist.empty:
        print(f"ERROR: cannot fetch price for {TICKER}. "
              "Check the ticker or your network connection.", file=sys.stderr)
        sys.exit(1)
    S = float(hist["Close"].iloc[-1])
    print(f"Live spot {TICKER}: ${S:.2f}")


# -- Fetch live ATM implied vol ------------------------------------------------

if args.vol is not None:
    sig = args.vol
    print(f"Vol (manual):    {sig:.2%}")
else:
    sig = None
    try:
        now = datetime.now()
        for exp in tk.options:
            T_exp = (datetime.strptime(exp, "%Y-%m-%d") - now).days / 365.0
            if T_exp >= 0.04:
                chain = tk.option_chain(exp).calls
                near  = chain[abs(chain["strike"] - S) / S < 0.02]
                if len(near) > 0:
                    sig = float(near["impliedVolatility"].mean())
                    break
    except Exception:
        pass
    if sig is None or sig <= 0:
        sig = 0.20
        print(f"ATM IV unavailable -- using 20% flat")
    else:
        print(f"Live ATM IV:     {sig:.2%}")


# -- Header -------------------------------------------------------------------

moneyness = K / S
m_label   = ("ITM" if moneyness < 0.99 else
             ("ATM" if moneyness <= 1.01 else "OTM"))

print()
print(f"  Ticker  : {TICKER}   Spot=${S:.2f}")
print(f"  Strike  : ${K:.2f}   Moneyness K/S={moneyness:.3f} ({m_label})")
print(f"  Expiry  : {T:.3f}Y ({T*365:.0f} days)")
print(f"  Type    : {otype}   Rate={r:.2%}   Base vol={sig:.2%}")
print(f"  Run ID  : {RUN_CODE}")
print()


# -- Storage for plot data ----------------------------------------------------

_results = {}   # label -> price (for comparison bar chart)
_plot    = {}   # plot-specific data per model


# -- Fetch market smile data for smile plot -----------------------------------

_smile_K, _smile_iv = [], []
try:
    _now = datetime.now()
    for _exp in tk.options:
        _T_exp = (datetime.strptime(_exp, "%Y-%m-%d") - _now).days / 365.0
        if _T_exp >= 0.04:
            _ch = tk.option_chain(_exp).calls
            _ch = _ch[(_ch['volume'].fillna(0) >= 5) & (_ch['impliedVolatility'] > 0.01)]
            if len(_ch) >= 3:
                _smile_K  = list(_ch['strike'].values)
                _smile_iv = list(_ch['impliedVolatility'].values)
                break
except Exception:
    pass


# -- Pricing functions ---------------------------------------------------------

def run_black_scholes():
    p_bs = price(S, K, T, r, sig, otype)
    g    = greeks(S, K, T, r, sig)
    print("-- Black-Scholes ------------------------------------------------- (European)")
    print(f"  Price:             ${p_bs:.4f}")
    print(f"  Delta:              {g['delta_' + otype]:.4f}")
    print(f"  Gamma:              {g['gamma']:.5f}")
    print(f"  Vega  (per 1% vol): ${g['vega'] / 100:.4f}")
    print(f"  Theta (per day):  ${g['theta_' + otype] / 365:.4f}")
    _results['Black-Scholes'] = p_bs


def run_binomial():
    step_counts = [5, 10, 20, 50, 100, 200, 500]
    eu_prices   = [binomial_price(S, K, T, r, sig, otype, 'european', n)['price']
                   for n in step_counts]
    am_prices   = [binomial_price(S, K, T, r, sig, otype, 'american', n)['price']
                   for n in step_counts]
    eu_px = eu_prices[-1]
    am_px = am_prices[-1]
    eep   = am_px - eu_px

    print("-- Binomial CRR (500 steps) --------------------------------------")
    print(f"  European {otype} price:  ${eu_px:.4f}")
    print(f"  American {otype} price:  ${am_px:.4f}")
    print(f"  Early-exercise premium: ${eep:.4f}")
    if otype == 'call' and abs(eep) < 1e-4:
        print("  (American call = European call: early exercise never optimal on non-dividend stock)")
    _results['Binomial (American)'] = am_px
    _plot['crr'] = (step_counts, eu_prices, am_prices)


def run_monte_carlo():
    ptype = f"european_{otype}"
    res   = mc_price(S, K, T, r, sig, ptype, n_sims=100_000)
    print("-- Monte Carlo (100k paths, antithetic variates) ----------------- (European)")
    print(f"  Price:             ${res['price']:.4f}")
    print(f"  Std error:         ${res['std_error']:.4f}")
    print(f"  95% CI:            [${res['conf_95_lo']:.4f}, ${res['conf_95_hi']:.4f}]")
    _results['Monte Carlo'] = res['price']
    # generate sample paths for the plot
    n_show  = 60
    n_steps = max(int(T * 252), 10)
    dt_s    = T / n_steps
    rng     = np.random.default_rng(42)
    Z       = rng.standard_normal((n_show, n_steps))
    lr      = (r - 0.5*sig**2)*dt_s + sig*np.sqrt(dt_s)*Z
    paths   = S * np.exp(np.hstack([np.zeros((n_show, 1)), np.cumsum(lr, axis=1)]))
    _plot['mc_paths'] = paths


def run_heston():
    print("-- Heston (calibrating to live option chain...) ------------------ (European)")
    try:
        from options_pricer.models.heston import HestonParams
        from options_pricer.models.heston import price as h_price

        loader = YFinanceLoader(risk_free_rate=r)
        md     = loader.load(TICKER, max_expiries=args.expiries)
        result = calibrate("heston", md)
        hp     = HestonParams.from_dict(result.params)
        px     = h_price(S, K, T, r, hp, otype)
        try:
            iv_h   = bs_iv(S, K, T, r, h_price(S, K, T, r, hp, "call"), "call")
            iv_str = f"{iv_h:.2%}"
        except Exception:
            iv_str = "n/a"

        p = result.params
        print(f"  Calibration:       RMSE={result.rmse:.3%}  n={result.n_points}"
              f"  ({result.elapsed_sec:.1f}s)")
        print(f"  Params:            v0={p['v0']:.4f}  kappa={p['kappa']:.3f}"
              f"  theta={p['theta']:.4f}  xi={p['xi']:.3f}  rho={p['rho']:.3f}")
        print(f"  Feller condition:  "
              f"{'satisfied' if hp.feller_satisfied() else 'violated (typical for real data)'}")
        print(f"  Heston IV:         {iv_str}")
        print(f"  Price:             ${px:.4f}")
        _results['Heston'] = px

        # pre-compute smile for plot (nearest maturity to T)
        T_avail = np.unique(md.maturities)
        near_T  = T_avail[np.argmin(np.abs(T_avail - T))]
        mask    = md.maturities == near_T
        K_mkt   = md.strikes[mask]
        iv_mkt  = md.ivs[mask]
        K_fine  = np.linspace(K_mkt.min() * 0.96, K_mkt.max() * 1.04, 40)
        iv_fit  = []
        for k_i in K_fine:
            try:
                px_i = h_price(S, k_i, near_T, r, hp, 'call')
                iv_fit.append(bs_iv(S, k_i, near_T, r, px_i, 'call') * 100)
            except Exception:
                iv_fit.append(np.nan)
        _plot['heston'] = {
            'K_mkt': K_mkt, 'iv_mkt': iv_mkt * 100,
            'K_fine': K_fine, 'iv_fit': iv_fit, 'near_T': near_T,
        }
    except Exception as e:
        print(f"  ERROR: {e}")


def run_svi():
    print("-- SVI (calibrating to live option chain...) -------------------- (European)")
    try:
        from options_pricer.models.svi import SVIParams, implied_vol_svi

        loader = YFinanceLoader(risk_free_rate=r)
        md     = loader.load(TICKER, max_expiries=args.expiries)
        result = calibrate("svi", md)
        slices  = result.params.get("slices", {})
        T_avail = sorted(float(k) for k in slices.keys())
        if not T_avail:
            print("  No SVI slices calibrated.")
            return

        F = S * np.exp(r * T)
        k = np.log(K / F)

        def get_iv(T_s):
            sp = SVIParams.from_dict(slices[f"{T_s:.6f}"])
            return float(implied_vol_svi(np.array([k]), T_s, sp)[0])

        if T <= T_avail[0]:
            iv_svi = get_iv(T_avail[0]) * (T / T_avail[0]) ** 0.5
        elif T >= T_avail[-1]:
            iv_svi = get_iv(T_avail[-1]) * (T / T_avail[-1]) ** 0.5
        else:
            lo = max(t for t in T_avail if t <= T)
            hi = min(t for t in T_avail if t >= T)
            w_lo     = get_iv(lo) ** 2 * lo
            w_hi     = get_iv(hi) ** 2 * hi
            frac     = (T - lo) / (hi - lo)
            w_interp = w_lo + frac * (w_hi - w_lo)
            iv_svi   = float(np.sqrt(max(w_interp, 1e-12) / T))

        px_svi  = price(S, K, T, r, iv_svi, otype)
        near_T  = min(T_avail, key=lambda t: abs(t - T))
        print(f"  Calibration:       RMSE={result.rmse:.3%}  n={result.n_points}"
              f"  ({result.elapsed_sec:.1f}s)  {len(T_avail)} slices")
        print(f"  Nearest slice:     T={near_T*365:.0f}d")
        print(f"  SVI IV:            {iv_svi:.2%}")
        print(f"  Price:             ${px_svi:.4f}")
        _results['SVI'] = px_svi

        # pre-compute smile for plot (nearest slice)
        mask   = md.maturities == near_T
        K_mkt  = md.strikes[mask]
        iv_mkt = md.ivs[mask]
        sp     = SVIParams.from_dict(slices[f"{near_T:.6f}"])
        F_near = S * np.exp(r * near_T)
        K_fine = np.linspace(K_mkt.min() * 0.96, K_mkt.max() * 1.04, 80)
        k_fine = np.log(K_fine / F_near)
        iv_fit = implied_vol_svi(k_fine, near_T, sp) * 100
        _plot['svi'] = {
            'K_mkt': K_mkt, 'iv_mkt': iv_mkt * 100,
            'K_fine': K_fine, 'iv_fit': iv_fit, 'near_T': near_T,
        }
    except Exception as e:
        print(f"  ERROR: {e}")


# -- Dispatch -----------------------------------------------------------------

if MODEL in ("all", "black-scholes"):
    run_black_scholes()
    print()

if MODEL in ("all", "binomial"):
    run_binomial()
    print()

if MODEL in ("all", "monte-carlo"):
    run_monte_carlo()
    print()

if MODEL in ("all", "heston"):
    run_heston()
    print()

if MODEL in ("all", "svi"):
    run_svi()
    print()


# -- Generate and save plots --------------------------------------------------

saved_figs = []

# 1. Market IV smile
if _smile_K:
    fig, ax = plt.subplots(figsize=(9, 4))
    mn = [k_i / S for k_i in _smile_K]
    ax.scatter(mn, [iv * 100 for iv in _smile_iv],
               s=28, color=_C[0], zorder=5, label='Market IV')
    ax.axvline(1.0, color='k', ls='--', lw=1, label=f'ATM  S=${S:.0f}')
    ax.axvline(K / S, color='red', ls=':', lw=1.5, label=f'Strike  K=${K:.0f}')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.set(xlabel='Moneyness  K/S', ylabel='Implied Vol',
           title=f'Market IV Smile  [{TICKER}]')
    ax.legend(fontsize=9)
    plt.tight_layout()
    saved_figs.append(_save_fig(fig, 'smile.png'))

# 2. Model price comparison
if len(_results) > 1:
    fig, ax = plt.subplots(figsize=(max(6, len(_results) * 2), 4))
    names       = list(_results.keys())
    prices_list = list(_results.values())
    bars = ax.bar(names, prices_list, color=_C[:len(names)], width=0.5)
    ax.bar_label(bars, fmt='$%.3f', padding=4, fontsize=10)
    ax.set_ylim(0, max(prices_list) * 1.18)
    ax.set(ylabel='Price ($)',
           title=f'Model Comparison  [{TICKER}  {otype}  K=${K:.0f}  T={T:.3f}Y]')
    plt.tight_layout()
    saved_figs.append(_save_fig(fig, 'model_comparison.png'))

# 3. GBM paths (Monte Carlo)
if 'mc_paths' in _plot:
    paths = _plot['mc_paths']
    t_ax  = np.linspace(0, T * 252, paths.shape[1])
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(paths.shape[0]):
        ax.plot(t_ax, paths[i], alpha=0.18, lw=0.7, color=_C[0])
    ax.plot(t_ax, paths.mean(axis=0), color='k', lw=2, label='Mean path')
    ax.axhline(S, color=_C[0], ls='--', lw=1.2, alpha=0.7, label=f'S0=${S:.0f}')
    ax.axhline(K, color='red', ls=':', lw=1.5, label=f'Strike  K=${K:.0f}')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'${y:,.0f}'))
    ax.set(xlabel='Trading days', ylabel=f'{TICKER} Price',
           title=f'GBM Paths (N=60)  [{TICKER}  vol={sig:.1%}  T={T*365:.0f}d]')
    ax.legend(fontsize=9)
    plt.tight_layout()
    saved_figs.append(_save_fig(fig, 'mc_paths.png'))

# 4. CRR convergence (Binomial)
if 'crr' in _plot:
    steps, eu_p, am_p = _plot['crr']
    bs_ref = price(S, K, T, r, sig, otype)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, eu_p, 'o-', color=_C[0], label=f'European {otype}')
    ax.plot(steps, am_p, 's-', color=_C[1], label=f'American {otype}')
    ax.axhline(bs_ref, color='k', ls='--', lw=1.2, label=f'BS ref  ${bs_ref:.4f}')
    ax.set(xlabel='Steps N', ylabel='Price ($)',
           title=f'CRR Convergence  [{TICKER}  K=${K:.0f}  T={T*365:.0f}d]')
    ax.legend(fontsize=9)
    plt.tight_layout()
    saved_figs.append(_save_fig(fig, 'crr_convergence.png'))

# 5. Calibrated smile -- Heston
if 'heston' in _plot:
    d = _plot['heston']
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(d['K_mkt'], d['iv_mkt'], s=35, color=_C[2], zorder=5,
               edgecolors='white', lw=0.3, label='Market')
    ax.plot(d['K_fine'], d['iv_fit'], color=_C[3], lw=2, label='Heston fit')
    ax.axvline(S, color='k', ls=':', lw=0.8, alpha=0.5, label=f'S=${S:.0f}')
    ax.axvline(K, color='red', ls=':', lw=1.5, label=f'K=${K:.0f}')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.set(xlabel='Strike', ylabel='Implied Vol',
           title=f'Heston Smile vs Market  [{TICKER}  T={d["near_T"]*365:.0f}d]')
    ax.legend(fontsize=9)
    plt.tight_layout()
    saved_figs.append(_save_fig(fig, 'heston_smile.png'))

# 6. Calibrated smile -- SVI
if 'svi' in _plot:
    d = _plot['svi']
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(d['K_mkt'], d['iv_mkt'], s=35, color=_C[2], zorder=5,
               edgecolors='white', lw=0.3, label='Market')
    ax.plot(d['K_fine'], d['iv_fit'], color=_C[4], lw=2, label='SVI fit')
    ax.axvline(S, color='k', ls=':', lw=0.8, alpha=0.5, label=f'S=${S:.0f}')
    ax.axvline(K, color='red', ls=':', lw=1.5, label=f'K=${K:.0f}')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.set(xlabel='Strike', ylabel='Implied Vol',
           title=f'SVI Smile vs Market  [{TICKER}  T={d["near_T"]*365:.0f}d]')
    ax.legend(fontsize=9)
    plt.tight_layout()
    saved_figs.append(_save_fig(fig, 'svi_smile.png'))


# -- Save results.txt ---------------------------------------------------------

sys.stdout = sys.__stdout__
with open(os.path.join(OUT_DIR, "results.txt"), "w", encoding="utf-8") as f:
    f.write(_log.getvalue())

print(f"Saved to output/{RUN_CODE}/")
for fn in sorted(saved_figs):
    sz = os.path.getsize(os.path.join(OUT_DIR, fn)) // 1024
    print(f"  {sz:3d} KB  {fn}")
print(f"          results.txt")
