import type {
  OptionGreeks,
  OptionPricingResult,
  PhoenixRequest,
  PricingResult,
  StrategyLegResult,
  StrategyResult,
  VanillaOptionRequest,
} from "../types";
import type { StrategyLegSpec } from "./compileGraph";

/** Offline demo-mode fallback, ported from the legacy apps/workbench/app.js
 * formula. Only used when the backend is unreachable — always flagged
 * `offline: true` so the UI never silently passes this off as real pricing. */
export function mockPrice(spec: PhoenixRequest): PricingResult {
  const p = spec;
  const fair = 100 + p.coupon_pa * 55 - Math.max(0, p.capital_barrier - 0.62) * 35 + (p.volatility - 0.25) * 14;
  const autocallProb = Math.max(0.05, Math.min(0.95, 0.82 - p.volatility * 0.45 + (0.75 - p.capital_barrier) * 0.3));
  const barrierTouchProb = Math.max(0.02, Math.min(0.7, p.volatility * 0.65 + (p.capital_barrier - 0.65) * 0.55));

  const spotMultipliers = [0.7, 0.85, 1.0, 1.15, 1.3];
  const volLevels = [p.volatility * 0.6, p.volatility * 0.8, p.volatility, p.volatility * 1.2, p.volatility * 1.4];
  // matches backend orientation (structured/heatmap.py): rows = vol_levels, cols = spot_multipliers
  const heatmapGrid = volLevels.map((v) =>
    spotMultipliers.map((sm) => 100 + p.coupon_pa * 55 - Math.max(0, p.capital_barrier - 0.62 * sm) * 35 + (v - 0.25) * 14)
  );

  return {
    fair_value_pct: fair,
    fair_value: (fair / 100) * p.notional,
    mc_confidence_95_pct: 0.4,
    issuer_price_pct: fair + 0.75,
    client_price_pct: fair + 1.25,
    autocall_probability: autocallProb,
    barrier_touch_probability: barrierTouchProb,
    prob_final_capital_loss: Math.max(0.01, p.volatility * 0.15),
    expected_coupon_pct: p.coupon_pa * p.maturity_years * 100,
    expected_life_years: p.maturity_years * (1 - autocallProb * 0.4),
    greeks: { delta: 0.38, gamma: 0.012, vega: 0.74, theta_1d: -0.015, rho: -22.3 },
    stress: [
      { scenario: "Spot -20%", fair_value_pct: fair - 8, delta_vs_base_pct: -8 },
      { scenario: "Spot +20%", fair_value_pct: fair + 3, delta_vs_base_pct: 3 },
      { scenario: "Vol +40%", fair_value_pct: fair - 5, delta_vs_base_pct: -5 },
    ],
    heatmap: { spot_multipliers: spotMultipliers, vol_levels: volLevels, fair_value_pct: heatmapGrid },
    validation: { score: 0, status: "offline", checks: [] },
    offline: true,
  };
}

function normCdf(x: number): number {
  // Abramowitz-Stegun erf approximation, good to ~1e-7
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-x * x);
  const erf = x >= 0 ? y : -y;
  return 0.5 * (1 + erf);
}

function blackScholesLeg(
  S: number,
  K: number,
  T: number,
  r: number,
  sigma: number,
  optionType: "call" | "put"
): { price: number; greeks: OptionGreeks } {
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  const disc = Math.exp(-r * T);

  const price =
    optionType === "call"
      ? S * normCdf(d1) - K * disc * normCdf(d2)
      : K * disc * normCdf(-d2) - S * normCdf(-d1);
  const delta = optionType === "call" ? normCdf(d1) : normCdf(d1) - 1;
  const gamma = Math.exp((-d1 * d1) / 2) / (S * sigma * Math.sqrt(2 * Math.PI * T));
  const vega = (S * Math.sqrt(T) * Math.exp((-d1 * d1) / 2)) / Math.sqrt(2 * Math.PI) / 100;

  return { price, greeks: { delta, gamma, vega } };
}

/** Offline demo-mode fallback for vanilla options: a plain Black-Scholes
 * evaluation done client-side (no backend needed), regardless of which model
 * was selected — always flagged `offline: true`. */
export function mockPriceOption(spec: VanillaOptionRequest): OptionPricingResult {
  const { spot: S, strike: K, maturity_years: T, risk_free_rate: r, volatility: sigma, option_type } = spec;
  const { price, greeks } = blackScholesLeg(S, K, T, r, sigma, option_type);

  return {
    kind: "vanilla_option",
    mode: "single",
    model: spec.model,
    price,
    greeks,
    strike: K,
    spot: S,
    option_type: spec.option_type,
    offline: true,
  };
}

/** Offline demo-mode fallback for multi-leg strategies: prices every leg
 * with the same client-side Black-Scholes as `mockPriceOption` and sums them
 * qty-weighted — always flagged `offline: true`. */
export function mockPriceStrategy(
  strategy: string,
  legs: { leg: StrategyLegSpec; spec: VanillaOptionRequest }[]
): StrategyResult {
  const legResults: StrategyLegResult[] = legs.map(({ leg, spec }) => {
    const { price, greeks } = blackScholesLeg(
      spec.spot, spec.strike, spec.maturity_years, spec.risk_free_rate, spec.volatility, spec.option_type
    );
    return { option_type: leg.option_type, qty: leg.qty, strike: spec.strike, price, greeks };
  });

  const netPrice = legResults.reduce((sum, l) => sum + l.qty * l.price, 0);
  const netGreeks: OptionGreeks = legResults.reduce(
    (acc, l) => ({
      delta: (acc.delta ?? 0) + l.qty * (l.greeks.delta ?? 0),
      gamma: (acc.gamma ?? 0) + l.qty * (l.greeks.gamma ?? 0),
      vega: (acc.vega ?? 0) + l.qty * (l.greeks.vega ?? 0),
    }),
    { delta: 0, gamma: 0, vega: 0 } as OptionGreeks
  );

  return {
    kind: "vanilla_option",
    mode: "strategy",
    strategy,
    underlying: legs[0]?.spec.underlying ?? "",
    spot: legs[0]?.spec.spot ?? 0,
    net_price: netPrice,
    net_greeks: netGreeks,
    legs: legResults,
    offline: true,
  };
}
