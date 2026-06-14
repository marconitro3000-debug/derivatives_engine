"""
run_demo.py  --  All figures from REAL market data only (Yahoo Finance, free).

Usage
-----
    python run_demo.py              # SPY (default)
    python run_demo.py AAPL
    python run_demo.py NVDA --model svi
    python run_demo.py TSLA --max-expiries 4
"""

import sys, os, warnings, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, PercentFormatter

mpl.rcParams.update({
    'figure.dpi'        : 150,
    'font.family'       : 'serif',
    'font.size'         : 11,
    'axes.titlesize'    : 12,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.linestyle'    : '--',
    'grid.alpha'        : 0.35,
    'legend.framealpha' : 0.9,
    'legend.fontsize'   : 10,
    'lines.linewidth'   : 1.8,
})
C   = plt.rcParams['axes.prop_cycle'].by_key()['color']
pct = FuncFormatter(lambda y, _: f'{y:.0f}%')
usd = FuncFormatter(lambda y, _: f'${y:,.0f}')

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('ticker', nargs='?', default='SPY')
parser.add_argument('--model', choices=['heston', 'svi'], default='heston')
parser.add_argument('--max-expiries', type=int, default=5)
args = parser.parse_args()

TICKER = args.ticker.upper()
MODEL  = args.model

# ── library ───────────────────────────────────────────────────────────────────
from options_pricer import (
    price, greeks, implied_vol, mc_price, binomial_price, from_iv_dict,
)
from options_pricer.models import heston as heston_mod, svi as svi_mod
from options_pricer.core.implied_vol import implied_vol as bs_iv
from options_pricer.data.loader import YFinanceLoader
from options_pricer.calibration.calibrator import calibrate, blend_params

import yfinance as yf

# ── output ────────────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight')
    plt.close(fig)
    print(f'  saved -> output/{name}')

def section(title):
    print(f'\n{"="*62}\n  {title}\n{"="*62}')


# =============================================================================
section(f'STEP 0  Fetching all real market data: {TICKER}')
# =============================================================================

# ── Option chain + spot ───────────────────────────────────────────────────────
loader = YFinanceLoader(risk_free_rate=0.045)
md     = loader.load(TICKER, max_expiries=args.max_expiries)
spot   = md.spot
r      = md.r
T_u    = np.unique(md.maturities)

# ATM implied vol: avg IV of options within 2% of spot
atm_mask  = np.abs(md.strikes / spot - 1) < 0.02
sigma_atm = float(md.ivs[atm_mask].mean()) if atm_mask.any() else float(np.median(md.ivs))

# Near-ATM call (first maturity, K closest to spot)
first_mask = md.maturities == T_u[0]
idx_atm    = np.argmin(np.abs(md.strikes[first_mask] - spot))
demo_K     = md.strikes[first_mask][idx_atm]
demo_T     = T_u[0]
demo_iv    = md.ivs[first_mask][idx_atm]

# Real market mid price for the demo call (re-fetch raw chain)
tk_yf     = yf.Ticker(TICKER)
demo_exp  = tk_yf.options[0]   # first available expiry (matches T_u[0])
raw_calls = tk_yf.option_chain(demo_exp).calls
raw_calls = raw_calls.sort_values('strike').reset_index(drop=True)

# get bid/ask mid for our demo strike
mask_k = raw_calls['strike'] == demo_K
if mask_k.any():
    row       = raw_calls[mask_k].iloc[0]
    demo_bid  = float(row['bid'])
    demo_ask  = float(row['ask'])
    demo_mid  = (demo_bid + demo_ask) / 2 if demo_ask > 0 else float('nan')
else:
    demo_mid  = float('nan')

# Near-ATM put for early-exercise section
raw_puts  = tk_yf.option_chain(demo_exp).puts
raw_puts  = raw_puts[
    (raw_puts['strike'] <= spot * 0.98) &
    (raw_puts['impliedVolatility'] > 0.01) &
    (raw_puts['volume'].fillna(0) >= 5)
].sort_values('strike')
if len(raw_puts) > 0:
    put_row    = raw_puts.iloc[len(raw_puts)//2]
    put_K      = float(put_row['strike'])
    put_iv     = float(put_row['impliedVolatility'])
    put_mid    = (float(put_row['bid']) + float(put_row['ask'])) / 2
else:
    put_K, put_iv, put_mid = demo_K, demo_iv, demo_mid

# ── Historical prices (3 years daily) ────────────────────────────────────────
print('  Fetching historical prices...')
hist   = yf.download(TICKER, period="3y", interval="1d",
                     progress=False, auto_adjust=True)
prices = hist['Close'].squeeze().dropna()
log_r  = np.log(prices / prices.shift(1)).dropna()

# 21-day rolling realized vol, annualized
rvol      = (log_r.rolling(21).std() * np.sqrt(252)).dropna()
rvol_ewma = rvol.ewm(alpha=0.10, adjust=False).mean()

# VIX as market-implied vol proxy
vix_raw = yf.download("^VIX", period="3y", interval="1d",
                      progress=False, auto_adjust=True)
vix = (vix_raw['Close'].squeeze() / 100).dropna()

# Align all series to common dates
common = rvol.index.intersection(vix.index)
rvol_a  = rvol.loc[common]
ewma_a  = rvol_ewma.loc[common]
vix_a   = vix.loc[common]

# ── Summary ───────────────────────────────────────────────────────────────────
print(f'  Spot    = ${spot:.2f}   r = {r:.2%}   ATM IV = {sigma_atm:.2%}')
print(f'  Demo call: K={demo_K:.1f}, T={demo_T:.3f}Y '
      f'(~{demo_T*365:.0f}d), IV={demo_iv:.2%}, mid=${demo_mid:.2f}')
print(f'  Demo put:  K={put_K:.1f},  T={demo_T:.3f}Y, '
      f'IV={put_iv:.2%}, mid=${put_mid:.2f}')
print(f'  Chain: {len(md.strikes)} calls, {len(T_u)} maturities '
      f'({T_u[0]:.3f}Y - {T_u[-1]:.3f}Y)')
print(f'  History: {len(prices)} days '
      f'({prices.index[0].date()} to {prices.index[-1].date()})')


# =============================================================================
section('S1  Black-Scholes: Pricing & Greeks')
print(f'  Parameters: S={spot:.2f}, sigma_ATM={sigma_atm:.2%}, r={r:.2%}')
print(f'  Maturities from real chain: {[f"{t:.3f}Y" for t in T_u]}')
# =============================================================================

g_demo = greeks(spot, demo_K, demo_T, r, demo_iv)
print(f'  delta={g_demo["delta_call"]:.4f}  gamma={g_demo["gamma"]:.5f}  '
      f'vega={g_demo["vega"]:.4f}  theta={g_demo["theta_call"]:.4f}')

# S1: Use ONLY real strikes from the chain, with per-strike real market IV
# (no synthetic K range, no flat ATM vol assumption)
mats = T_u[:min(4, len(T_u))]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for i, T_ in enumerate(mats):
    mask  = md.maturities == T_
    K_s   = md.strikes[mask]
    iv_s  = md.ivs[mask]
    order = np.argsort(K_s)
    K_s, iv_s = K_s[order], iv_s[order]

    pr_s = [price(spot, K, T_, r, iv, 'call') for K, iv in zip(K_s, iv_s)]
    dl_s = [greeks(spot, K, T_, r, iv)['delta_call'] for K, iv in zip(K_s, iv_s)]
    gm_s = [greeks(spot, K, T_, r, iv)['gamma'] for K, iv in zip(K_s, iv_s)]
    lbl  = f'T={T_*365:.0f}d'

    axes[0].plot(K_s, pr_s, marker='o', ms=4, lw=1.4, color=C[i], label=lbl)
    axes[1].plot(K_s, dl_s, marker='o', ms=4, lw=1.4, color=C[i], label=lbl)
    axes[2].plot(K_s, gm_s, marker='o', ms=4, lw=1.4, color=C[i], label=lbl)

for ax in axes:
    ax.axvline(spot,  color='k',    ls=':', lw=1,   label=f'S=${spot:.0f}')
    ax.axvline(demo_K, color='gray', ls='--', lw=0.8, label=f'ATM K=${demo_K:.0f}')
    ax.xaxis.set_major_formatter(usd)
axes[0].set(title='(A) BS Call Price', xlabel='Strike K', ylabel='Price ($)')
axes[1].set(title='(B) Delta',         xlabel='Strike K', ylabel='delta')
axes[2].set(title='(C) Gamma',         xlabel='Strike K', ylabel='gamma')
axes[0].legend(fontsize=8)
fig.suptitle(f'Black-Scholes Greeks at Real Market Strikes  '
             f'[{TICKER}  S=${spot:.2f}  r={r:.2%}  per-strike market IV]', y=1.02)
plt.tight_layout()
save(fig, 'fig_01_bs_greeks.png')


# =============================================================================
section('S2  Implied Volatility: Round-trip on Real Option Chain')
print(f'  Testing IV solver on {len(md.strikes)} real market options')
# =============================================================================

errors, computed_ivs, market_ivs, moneyness = [], [], [], []
for K_, T_, iv_mkt in zip(md.strikes, md.maturities, md.ivs):
    mkt_px = price(spot, K_, T_, r, iv_mkt, 'call')   # BS price from market IV
    try:
        iv_fit = bs_iv(spot, K_, T_, r, mkt_px, 'call')
        errors.append(abs(iv_fit - iv_mkt))
        computed_ivs.append(iv_fit)
        market_ivs.append(iv_mkt)
        moneyness.append(K_ / spot)
    except Exception:
        pass

errors = np.array(errors)
print(f'  Max round-trip error : {errors.max():.2e}')
print(f'  Mean error           : {errors.mean():.2e}')
print(f'  All < 1e-8           : {(errors < 1e-8).all()}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(np.array(market_ivs)*100, np.array(computed_ivs)*100,
                c=moneyness, cmap='RdYlGn', s=25, alpha=0.8)
lims = [min(np.array(market_ivs)*100)*0.95, max(np.array(market_ivs)*100)*1.05]
axes[0].plot(lims, lims, 'k--', lw=1, label='Perfect fit')
axes[0].set(title='(A) Market IV vs Newton-Raphson IV',
            xlabel='Market IV (%)', ylabel='Computed IV (%)')
axes[0].yaxis.set_major_formatter(pct)
axes[0].xaxis.set_major_formatter(pct)
axes[0].legend()

axes[1].hist(errors, bins=40, color=C[0], edgecolor='white', lw=0.4)
axes[1].set(title=f'(B) Round-trip Error Distribution (n={len(errors)})',
            xlabel='|IV_computed - IV_market|', ylabel='Count')
axes[1].axvline(1e-8, color='r', ls='--', lw=1.2, label='1e-8 threshold')
axes[1].legend()

fig.suptitle(f'IV Solver Verification  [{TICKER}  {len(errors)} real options]', y=1.02)
plt.tight_layout()
save(fig, 'fig_02_iv_roundtrip.png')


# =============================================================================
section('S3  Monte Carlo: GBM Paths (Real Parameters)')
print(f'  S0={spot:.2f}, sigma={sigma_atm:.2%}, r={r:.2%}, T={demo_T:.3f}Y')
# =============================================================================

n_show  = 50
n_steps = max(int(demo_T * 252), 5)
dt      = demo_T / n_steps
rng     = np.random.default_rng(42)
Z       = rng.standard_normal((n_show, n_steps))
lr      = (r - 0.5*sigma_atm**2)*dt + sigma_atm*np.sqrt(dt)*Z
S_paths = spot * np.exp(np.hstack([np.zeros((n_show,1)), np.cumsum(lr, axis=1)]))
t_ax    = np.linspace(0, demo_T * 252, n_steps+1)   # in trading days

fig, ax = plt.subplots(figsize=(11, 4.5))
for i in range(n_show):
    ax.plot(t_ax, S_paths[i], alpha=0.25, lw=0.8, color=C[0])
ax.plot(t_ax, S_paths.mean(axis=0), color='k', lw=2.2, label='Path mean')
ax.axhline(spot,  color=C[0], lw=1.2, ls='--', alpha=0.6, label=f'S0=${spot:.2f}')
ax.axhline(demo_K, color=C[3], lw=1.2, ls='--', label=f'ATM strike K=${demo_K:.0f}')
ax.set(title=f'GBM Paths  [{TICKER}  S0=${spot:.2f}  '
             f'sigma_ATM={sigma_atm:.1%}  T={demo_T*365:.0f}d  N={n_show}]',
       xlabel='Trading days', ylabel=f'{TICKER} Price ($)')
ax.yaxis.set_major_formatter(usd)
ax.legend()
plt.tight_layout()
save(fig, 'fig_03a_gbm_paths.png')

# ── MC flat-vol price vs real bid/ask across all strikes in first maturity ────
# This shows WHERE the flat-vol model fails (motivates Heston/SVI calibration)
mask_first = md.maturities == T_u[0]
K_first    = md.strikes[mask_first]
iv_first   = md.ivs[mask_first]
order_f    = np.argsort(K_first)
K_first, iv_first = K_first[order_f], iv_first[order_f]

K_cmp, mc_p, bs_p, mkt_bid_v, mkt_ask_v, mkt_mid_v = [], [], [], [], [], []
print(f'  MC flat-vol vs real market (T={T_u[0]*365:.0f}d, sigma_ATM={sigma_atm:.2%}):')
for K_, iv_ in zip(K_first, iv_first):
    row = raw_calls[abs(raw_calls['strike'] - K_) < 0.5]
    if len(row) == 0:
        continue
    row  = row.iloc[0]
    bid_ = float(row['bid'])
    ask_ = float(row['ask'])
    if bid_ <= 0 or ask_ <= 0:
        continue
    mid_ = (bid_ + ask_) / 2
    mc_r = mc_price(spot, K_, T_u[0], r, sigma_atm, 'european_call',
                    n_sims=60_000, seed=42)
    bs_v = price(spot, K_, T_u[0], r, sigma_atm, 'call')
    K_cmp.append(K_); mc_p.append(mc_r['price']); bs_p.append(bs_v)
    mkt_bid_v.append(bid_); mkt_ask_v.append(ask_); mkt_mid_v.append(mid_)

K_cmp = np.array(K_cmp)
mc_p  = np.array(mc_p)
bs_p  = np.array(bs_p)
mkt_mid_v = np.array(mkt_mid_v)

atm_err = float(np.abs(mc_p - mkt_mid_v).mean())
print(f'  Options compared: {len(K_cmp)}   mean |MC - market mid| = {atm_err:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
ax.fill_between(K_cmp, mkt_bid_v, mkt_ask_v, alpha=0.25, color=C[2],
                label='Real bid-ask spread')
ax.plot(K_cmp, mkt_mid_v, 'o-', ms=5, color=C[2], lw=1.2, label='Real market mid')
ax.plot(K_cmp, bs_p,      's-', ms=4, color=C[0], lw=1.4, label=f'BS flat vol ({sigma_atm:.1%})')
ax.plot(K_cmp, mc_p,      '^',  ms=4, color=C[1], alpha=0.8, label='MC flat vol')
ax.axvline(spot, color='k', ls=':', lw=1, label=f'S=${spot:.0f}')
ax.set(title=f'(A) Flat-vol BS/MC vs Real Market  [T={T_u[0]*365:.0f}d]',
       xlabel='Strike K', ylabel='Call Price ($)')
ax.xaxis.set_major_formatter(usd)
ax.legend(fontsize=8)

ax = axes[1]
model_err = bs_p - mkt_mid_v
moneyness = K_cmp / spot
sc = ax.scatter(moneyness * 100, model_err, c=iv_first[:len(K_cmp)]*100,
                cmap='RdYlGn_r', s=55, zorder=5, edgecolors='white', lw=0.3)
ax.axhline(0, color='k', lw=1)
ax.axvline(100, color='k', ls=':', lw=0.8, label='ATM (K/S=1)')
plt.colorbar(sc, ax=ax, label='Market IV (%)')
ax.set(title='(B) Flat-vol Error = BS - Market Mid\n(negative = market prices more than flat vol)',
       xlabel='Moneyness K/S (%)', ylabel='Pricing error ($)')
ax.xaxis.set_major_formatter(pct)
ax.legend(fontsize=9)

fig.suptitle(f'MC Flat-vol vs Real {TICKER} Market  '
             f'[shows skew model error that motivates Heston/SVI]', y=1.02)
plt.tight_layout()
save(fig, 'fig_03b_model_vs_market.png')


# =============================================================================
section('S4  CRR Binomial Tree: Convergence & Early Exercise')
print(f'  Call: K={demo_K:.1f}, T={demo_T*365:.0f}d, sigma={sigma_atm:.2%}')
print(f'  Put:  K={put_K:.1f},  T={demo_T*365:.0f}d, sigma={put_iv:.2%}')
# =============================================================================

step_counts = [5, 10, 20, 50, 100, 200, 500, 1000]
bs_call = price(spot, demo_K, demo_T, r, sigma_atm, 'call')
bs_put  = price(spot, put_K,  demo_T, r, put_iv,  'put')

eu_call = [binomial_price(spot, demo_K, demo_T, r, sigma_atm, 'call', 'european', n)['price']
           for n in step_counts]
am_call = [binomial_price(spot, demo_K, demo_T, r, sigma_atm, 'call', 'american', n)['price']
           for n in step_counts]
eu_put  = [binomial_price(spot, put_K,  demo_T, r, put_iv,  'put',  'european', n)['price']
           for n in step_counts]
am_put  = [binomial_price(spot, put_K,  demo_T, r, put_iv,  'put',  'american', n)['price']
           for n in step_counts]

print(f'  N=1000: EU call={eu_call[-1]:.4f}  AM call={am_call[-1]:.4f}  BS={bs_call:.4f}  mkt_mid={demo_mid:.4f}')
print(f'  N=1000: EU put ={eu_put[-1]:.4f}  AM put ={am_put[-1]:.4f}  BS={bs_put:.4f}  mkt_mid={put_mid:.4f}')
early_ex_put = am_put[-1] - eu_put[-1]
print(f'  Early-exercise premium (put): {early_ex_put:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(step_counts, eu_call, marker='o', ms=4, color=C[0], label=f'European call (CRR)')
axes[0].plot(step_counts, am_call, marker='s', ms=4, color=C[1], label=f'American call (CRR)')
axes[0].axhline(bs_call,  color=C[0], ls='--', lw=1, label=f'BS call={bs_call:.4f}')
if np.isfinite(demo_mid):
    axes[0].axhline(demo_mid, color='k',  ls=':', lw=1.2, label=f'Market mid={demo_mid:.4f}')
axes[0].set(title=f'(A) CRR Convergence  [Call K={demo_K:.0f}]',
            xlabel='Steps N', ylabel='Call Price ($)')
axes[0].legend(fontsize=8)

axes[1].plot(step_counts, eu_put, marker='o', ms=4, color=C[2], label='European put (CRR)')
axes[1].plot(step_counts, am_put, marker='s', ms=4, color=C[3], label='American put (CRR)')
axes[1].axhline(bs_put, color=C[2], ls='--', lw=1, label=f'BS put={bs_put:.4f}')
if np.isfinite(put_mid):
    axes[1].axhline(put_mid, color='k', ls=':', lw=1.2, label=f'Market mid={put_mid:.4f}')
axes[1].set(title=f'(B) Early-Exercise Premium  [Put K={put_K:.0f}]',
            xlabel='Steps N', ylabel='Put Price ($)')
axes[1].legend(fontsize=8)

fig.suptitle(f'CRR Binomial Tree  [{TICKER}  S=${spot:.2f}  T={demo_T*365:.0f}d]', y=1.02)
plt.tight_layout()
save(fig, 'fig_04_crr_convergence.png')


# =============================================================================
section(f'S5  Implied-Volatility Surface  [{TICKER}]')
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
for i, T_ in enumerate(T_u):
    mask  = md.maturities == T_
    K_s   = md.strikes[mask]
    iv_s  = md.ivs[mask] * 100
    order = np.argsort(K_s)
    ax.plot(K_s[order], iv_s[order], marker='o', ms=3.5,
            label=f'T={T_*365:.0f}d', color=C[i % len(C)])
ax.axvline(spot, color='k', ls=':', lw=1, label=f'S=${spot:.0f}')
ax.set(title=f'(A) Volatility Smile', xlabel='Strike K', ylabel='Implied Vol')
ax.xaxis.set_major_formatter(usd)
ax.yaxis.set_major_formatter(pct)
ax.legend(fontsize=9)

ax = axes[1]
F0 = spot * np.exp(r * np.array(T_u))
for i, (T_, F_) in enumerate(zip(T_u, F0)):
    mask  = md.maturities == T_
    K_s   = md.strikes[mask]
    iv_s  = md.ivs[mask] * 100
    order = np.argsort(K_s)
    lm    = np.log(K_s[order] / F_)
    ax.plot(lm, iv_s[order], marker='o', ms=3.5,
            label=f'T={T_*365:.0f}d', color=C[i % len(C)])
ax.axvline(0, color='k', ls=':', lw=1, label='ATM (k=0)')
ax.set(title='(B) Smile vs Log-Moneyness  ln(K/F)',
       xlabel='Log-moneyness k', ylabel='Implied Vol')
ax.yaxis.set_major_formatter(pct)
ax.legend(fontsize=9)

fig.suptitle(f'IV Surface  [{TICKER}  S=${spot:.2f}  '
             f'{len(md.strikes)} liquid calls]', y=1.02)
plt.tight_layout()
save(fig, 'fig_05_iv_surface.png')


# =============================================================================
section(f'S6  Volatility Surface Interpolation  [{TICKER}]')
# =============================================================================

iv_dict = {(float(K_), float(T_)): float(iv_)
           for K_, T_, iv_ in zip(md.strikes, md.maturities, md.ivs)}

if len(T_u) >= 2 and len(np.unique(md.strikes)) >= 4:
    surf = from_iv_dict(S=spot, iv_dict=iv_dict, method='spline')
    K_g, T_g, IV_g = surf.grid(n_strikes=60, n_maturities=30)
    KK, TT = np.meshgrid(K_g, T_g, indexing='ij')

    fig = plt.figure(figsize=(14, 5))
    ax3d = fig.add_subplot(121, projection='3d')
    s3d  = ax3d.plot_surface(KK, TT*365, IV_g*100, cmap='RdYlGn_r',
                             alpha=0.90, rstride=1, cstride=1, linewidth=0)
    ax3d.set(xlabel='Strike K', ylabel='Days to exp', zlabel='IV (%)')
    ax3d.set_title('(A) 3-D Volatility Surface', pad=8)
    ax3d.zaxis.set_major_formatter(FuncFormatter(lambda z, _: f'{z:.0f}%'))
    fig.colorbar(s3d, ax=ax3d, shrink=0.45, pad=0.1)

    ax2d = fig.add_subplot(122)
    cf   = ax2d.contourf(KK, TT*365, IV_g*100, levels=14, cmap='RdYlGn_r')
    ax2d.scatter(md.strikes, md.maturities*365, c='k', s=20, zorder=5,
                 label=f'{len(iv_dict)} market quotes')
    ax2d.axvline(spot, color='white', ls='--', lw=1.2, label=f'S=${spot:.0f}')
    fig.colorbar(cf, ax=ax2d, label='IV (%)')
    ax2d.set(title='(B) Contour Map', xlabel='Strike K', ylabel='Days to expiry')
    ax2d.xaxis.set_major_formatter(usd)
    ax2d.legend(fontsize=9)

    fig.suptitle(f'Volatility Surface  [{TICKER}  S=${spot:.2f}]', y=1.02)
    plt.tight_layout()
    save(fig, 'fig_06_vol_surface_3d.png')
    print(f'  Interpolated surface from {len(iv_dict)} real quotes')
else:
    print('  Not enough maturities/strikes for 3-D surface — skipping.')


# =============================================================================
section(f'S7  {MODEL.upper()} Calibration  [{TICKER}]')
# =============================================================================

print(f'  Calibrating {MODEL.upper()} to {len(md.strikes)} real quotes...')
result = calibrate(MODEL, md)
print(f'  {result.summary()}')

if MODEL == 'heston':
    print('\n  Fitted Heston parameters:')
    for p, v in result.params.items():
        print(f'    {p:<8} = {v:.6f}')
    hp = heston_mod.HestonParams.from_dict(result.params)
    print(f'  Feller condition (2*kappa*theta >= xi^2): '
          f'{"satisfied" if hp.feller_satisfied() else "violated (typical for real data)"}')
else:
    n_sl = len(result.params.get('slices', {}))
    print(f'\n  Fitted {n_sl} SVI slices (one per maturity)')

# Calibrated smile vs market
n_m   = len(T_u)
fig, axes = plt.subplots(1, n_m, figsize=(max(3.2*n_m, 9), 4.5), sharey=True)
if n_m == 1:
    axes = [axes]

for ax, T_ in zip(axes, T_u):
    mask   = md.maturities == T_
    K_s    = md.strikes[mask]
    iv_mkt = md.ivs[mask] * 100
    order  = np.argsort(K_s)
    ax.scatter(K_s[order], iv_mkt[order], color=C[2], s=35, zorder=5,
               label='Market', edgecolors='white', lw=0.3)

    K_fine = np.linspace(K_s.min()*0.96, K_s.max()*1.04, 80)
    iv_fit = []

    if MODEL == 'heston':
        hp_fit = heston_mod.HestonParams.from_dict(result.params)
        for K_ in K_fine:
            try:
                px = heston_mod.price(spot, K_, T_, r, hp_fit, 'call')
                iv_fit.append(bs_iv(spot, K_, T_, r, px, 'call') * 100)
            except Exception:
                iv_fit.append(np.nan)
    else:
        slices = result.params.get('slices', {})
        T_key  = f'{T_:.6f}'
        if T_key in slices:
            sp  = svi_mod.SVIParams.from_dict(slices[T_key])
            F_  = spot * np.exp(r * T_)
            k_f = np.log(K_fine / F_)
            iv_fit = list(svi_mod.implied_vol_svi(k_f, T_, sp) * 100)
        else:
            iv_fit = [np.nan] * len(K_fine)

    ax.plot(K_fine, iv_fit, color=C[0], lw=2, label=f'{MODEL.upper()} fit')
    ax.axvline(spot, color='k', ls=':', lw=0.8, alpha=0.5)
    ax.set(title=f'T={T_*365:.0f}d', xlabel='Strike K')
    ax.xaxis.set_major_formatter(usd)
    ax.yaxis.set_major_formatter(pct)
    if ax is axes[0]:
        ax.set_ylabel('IV'); ax.legend(fontsize=8)

fig.suptitle(f'{MODEL.upper()} calibration vs {TICKER} market  '
             f'(RMSE={result.rmse:.2%}  n={result.n_points})', y=1.02)
plt.tight_layout()
save(fig, f'fig_07_calibration_{MODEL}.png')


# =============================================================================
section(f'S8  Historical Realized Volatility + VIX  [{TICKER}]')
print(f'  {len(rvol_a)} trading days of real data')
# =============================================================================

avg_rv  = float(rvol_a.mean())
avg_vix = float(vix_a.mean())
corr_rv_vix = float(np.corrcoef(rvol_a.values, vix_a.values)[0, 1])

print(f'  Mean realized vol (21d): {avg_rv:.2%}')
print(f'  Mean VIX:                {avg_vix:.2%}')
print(f'  Corr(RVol, VIX):         {corr_rv_vix:.3f}')
print(f'  Current ATM IV (chain):  {sigma_atm:.2%}')

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

ax = axes[0]
ax.fill_between(rvol_a.index, rvol_a.values*100, alpha=0.20, color=C[0])
ax.plot(rvol_a.index,  rvol_a.values*100,  color=C[0], lw=1.0, alpha=0.8, label='RVol 21d')
ax.plot(ewma_a.index,  ewma_a.values*100,  color=C[1], lw=2.0,             label='RVol EWMA (a=0.10)')
ax.plot(vix_a.index,   vix_a.values*100,   color=C[3], lw=1.4, alpha=0.9,  label='VIX (market IV)')
ax.axhline(sigma_atm*100, color='k', ls='--', lw=1.2,
           label=f'Current ATM IV {sigma_atm:.1%}')
ax.set(title=f'(A) Realized Vol vs VIX  [{TICKER}  3Y]', ylabel='Annualised Vol (%)')
ax.yaxis.set_major_formatter(pct)
ax.legend(fontsize=9)

ax = axes[1]
vol_premium = (vix_a.values - rvol_a.values) * 100
ax.fill_between(vix_a.index, vol_premium, 0,
                where=(vol_premium >= 0), alpha=0.3, color=C[3], label='VIX > RVol')
ax.fill_between(vix_a.index, vol_premium, 0,
                where=(vol_premium < 0),  alpha=0.3, color=C[0], label='RVol > VIX')
ax.plot(vix_a.index, vol_premium, color='gray', lw=0.8)
ax.axhline(0, color='k', lw=0.8)
ax.axhline(np.mean(vol_premium), color='k', ls='--', lw=1,
           label=f'Mean vol premium = {np.mean(vol_premium):.1f}%')
ax.set(title='(B) Variance Risk Premium: VIX - RVol  (vol sellers earn positive premium on avg)',
       xlabel='Date', ylabel='Vol premium (%)')
ax.yaxis.set_major_formatter(pct)
ax.legend(fontsize=9)

fig.suptitle(f'Volatility Dynamics  [{TICKER}  {rvol_a.index[0].year}-{rvol_a.index[-1].year}]',
             y=1.01)
plt.tight_layout()
save(fig, 'fig_08_vol_history_ewma.png')


# =============================================================================
section('Done')
# =============================================================================
figs = sorted(f for f in os.listdir(OUT) if f.endswith('.png'))
print(f'  Ticker : {TICKER}')
print(f'  Model  : {MODEL}')
print(f'  Output : output/  ({len(figs)} figures)')
for f in figs:
    sz = os.path.getsize(os.path.join(OUT, f)) // 1024
    print(f'  {sz:4d} KB   {f}')
