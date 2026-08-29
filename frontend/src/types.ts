export type FieldType = "text" | "number" | "boolean" | "select";

export interface FieldDef {
  key: string;
  label: string;
  type: FieldType;
  options?: string[];
  step?: number;
  showIf?: (params: Record<string, any>) => boolean;
}

export type LiveKind = "quote" | "rates" | "volatility" | "surface" | "calibration" | null;
export type NodeCategory = "trigger" | "input" | "param" | "compute" | "output";

export type ProductTag = "phoenix" | "vanilla_option" | "portfolio";

export interface NodeDef {
  kind: string;
  label: string;
  description: string;
  color: string;
  category: NodeCategory;
  live: LiveKind;
  fields: FieldDef[];
  defaults: Record<string, any>;
  /** Which product(s) this card is actually used by — drives palette filtering
   * so you can't drag in a card the current product's compiler ignores. */
  products: ProductTag[];
}

export const NODE_DEFS: NodeDef[] = [
  {
    kind: "manual_trigger",
    label: "Manual Trigger",
    description: "Start pricing run",
    color: "var(--orange)",
    category: "trigger",
    live: null,
    fields: [],
    defaults: {},
    products: ["phoenix", "vanilla_option", "portfolio"],
  },
  {
    kind: "underlying",
    label: "Underlying",
    description: "Real spot + market data",
    color: "var(--blue)",
    category: "input",
    live: "quote",
    fields: [
      { key: "ticker", label: "Ticker", type: "text" },
      { key: "dividend_yield", label: "Dividend yield (Autocall)", type: "number", step: 0.001 },
    ],
    defaults: { ticker: "AAPL", dividend_yield: 0 },
    products: ["phoenix", "vanilla_option"],
  },
  {
    kind: "rates_curve",
    label: "Rates Curve",
    description: "Live risk-free curve",
    color: "var(--purple)",
    category: "input",
    live: "rates",
    fields: [{ key: "currency", label: "Currency", type: "text" }],
    defaults: { currency: "USD" },
    products: ["phoenix", "vanilla_option"],
  },
  {
    kind: "volatility_forecast",
    label: "Volatility Forecast",
    description: "Realized vol or manual",
    color: "var(--green)",
    category: "input",
    live: "volatility",
    fields: [
      { key: "mode", label: "Mode", type: "select", options: ["realized", "manual"] },
      {
        key: "manual_volatility",
        label: "Manual vol",
        type: "number",
        step: 0.01,
        showIf: (p) => p.mode === "manual",
      },
    ],
    defaults: { mode: "realized", manual_volatility: 0.28 },
    products: ["phoenix", "vanilla_option"],
  },
  {
    kind: "implied_vol_surface",
    label: "Implied Vol Surface",
    description: "Live option chain + SSVI smile",
    color: "var(--cyan)",
    category: "input",
    live: "surface",
    fields: [],
    defaults: {},
    products: ["phoenix", "vanilla_option"],
  },
  {
    kind: "barrier",
    label: "Barrier",
    description: "Coupon & capital barrier — checked only at maturity (European), not continuously along the path",
    color: "var(--amber)",
    category: "param",
    live: null,
    fields: [
      { key: "capital_barrier", label: "Capital barrier", type: "number", step: 0.01 },
      { key: "separate_coupon_barrier", label: "Coupon barrier != capital", type: "boolean" },
      {
        key: "coupon_barrier",
        label: "Coupon barrier",
        type: "number",
        step: 0.01,
        showIf: (p) => !!p.separate_coupon_barrier,
      },
    ],
    defaults: { capital_barrier: 0.7, separate_coupon_barrier: false, coupon_barrier: 0.7 },
    products: ["phoenix"],
  },
  {
    kind: "coupon",
    label: "Coupon",
    description: "Annualized coupon",
    color: "var(--pink)",
    category: "param",
    live: null,
    fields: [
      { key: "coupon_pa", label: "Coupon p.a.", type: "number", step: 0.005 },
      { key: "coupon_frequency", label: "Frequency / yr", type: "number", step: 1 },
    ],
    defaults: { coupon_pa: 0.08, coupon_frequency: 4 },
    products: ["phoenix"],
  },
  {
    kind: "memory",
    label: "Memory",
    description: "Accrued missed coupons",
    color: "var(--pink)",
    category: "param",
    live: null,
    fields: [{ key: "enabled", label: "Enabled", type: "boolean" }],
    defaults: { enabled: true },
    products: ["phoenix"],
  },
  {
    kind: "autocall",
    label: "Autocall",
    description: "Early redemption trigger",
    color: "var(--orange2)",
    category: "param",
    live: null,
    fields: [
      { key: "autocall_level", label: "Autocall level", type: "number", step: 0.01 },
      { key: "autocall_start", label: "First obs. (years)", type: "number", step: 0.05 },
    ],
    defaults: { autocall_level: 1.0, autocall_start: 0.25 },
    products: ["phoenix"],
  },
  {
    kind: "pricing_output",
    label: "Pricing Output",
    description: "Monte Carlo, Greeks, CI",
    color: "var(--orange)",
    category: "output",
    live: null,
    fields: [
      { key: "notional", label: "Notional", type: "number", step: 10000 },
      { key: "maturity_years", label: "Maturity (yrs)", type: "number", step: 0.25 },
      { key: "n_paths", label: "MC paths", type: "number", step: 1000 },
      { key: "steps_per_year", label: "Steps / yr", type: "number", step: 1 },
    ],
    defaults: { notional: 1_000_000, maturity_years: 1.0, n_paths: 20_000, steps_per_year: 252 },
    products: ["phoenix"],
  },
  {
    kind: "stress_test",
    label: "Stress Test",
    description: "Connect to show the spot/vol/rate stress table in results — omit to hide it",
    color: "var(--red)",
    category: "output",
    live: null,
    fields: [],
    defaults: {},
    products: ["phoenix"],
  },
  {
    kind: "option_contract",
    label: "Option Contract",
    description: "Strike, type & pricing model",
    color: "var(--cyan)",
    category: "param",
    live: "calibration",
    fields: [
      { key: "strike", label: "Strike", type: "number", step: 1 },
      { key: "maturity_years", label: "Maturity (yrs)", type: "number", step: 0.25 },
      { key: "option_type", label: "Call / Put", type: "select", options: ["call", "put"] },
      {
        key: "model",
        label: "Model",
        type: "select",
        options: ["black_scholes", "binomial", "monte_carlo", "merton", "heston", "svi", "local_vol"],
      },
      {
        key: "style",
        label: "Style",
        type: "select",
        options: ["european", "american"],
        showIf: (p) => p.model === "binomial" || p.model === "local_vol",
      },
      { key: "n_steps", label: "Tree steps", type: "number", step: 50, showIf: (p) => p.model === "binomial" },
      {
        key: "n_sims", label: "MC paths", type: "number", step: 1000,
        showIf: (p) => (p.model === "monte_carlo" || p.model === "local_vol") && p.style !== "american",
      },
      { key: "jump_intensity", label: "Jumps / yr (λ)", type: "number", step: 0.5, showIf: (p) => p.model === "merton" },
      { key: "jump_mean", label: "Mean jump size", type: "number", step: 0.01, showIf: (p) => p.model === "merton" },
      { key: "jump_vol", label: "Jump vol", type: "number", step: 0.01, showIf: (p) => p.model === "merton" },
      {
        key: "seed", label: "MC seed", type: "number", step: 1,
        showIf: (p) => p.model === "monte_carlo" || (p.model === "local_vol" && p.style !== "american"),
      },
    ],
    defaults: {
      strike: 175, maturity_years: 1.0, option_type: "call", model: "black_scholes",
      jump_intensity: 2.0, jump_mean: -0.05, jump_vol: 0.10,
      style: "european", n_steps: 500, n_sims: 20_000, seed: 42,
    },
    products: ["vanilla_option"],
  },
  {
    kind: "option_strategy",
    label: "Option Strategy",
    description: "Multi-leg combination (spreads, straddle...)",
    color: "var(--pink)",
    category: "param",
    live: null,
    fields: [
      {
        key: "strategy",
        label: "Strategy",
        type: "select",
        options: ["bull_call_spread", "bear_put_spread", "straddle", "strangle", "risk_reversal"],
      },
      { key: "otm_pct", label: "OTM offset (% of spot)", type: "number", step: 0.01 },
      { key: "maturity_years", label: "Maturity (yrs)", type: "number", step: 0.25 },
      {
        key: "model",
        label: "Model",
        type: "select",
        options: ["black_scholes", "binomial", "monte_carlo", "merton"],
      },
    ],
    defaults: { strategy: "bull_call_spread", otm_pct: 0.05, maturity_years: 1.0, model: "black_scholes" },
    products: ["vanilla_option"],
  },
  {
    kind: "model_comparison",
    label: "Model Comparison",
    description: "Price the same contract across all models",
    color: "var(--muted)",
    category: "output",
    live: null,
    fields: [],
    defaults: {},
    products: ["vanilla_option"],
  },
  {
    kind: "term_sheet",
    label: "Term Sheet",
    description: "Connect to generate an indicative term sheet alongside the pricing result",
    color: "var(--blue)",
    category: "output",
    live: null,
    fields: [
      { key: "issuer", label: "Issuer", type: "text" },
      { key: "currency", label: "Currency", type: "text" },
    ],
    defaults: { issuer: "Demo Issuer SA", currency: "USD" },
    products: ["phoenix"],
  },
  {
    kind: "portfolio_position",
    label: "Position",
    description: "One name in the portfolio (repeatable)",
    color: "var(--blue)",
    category: "input",
    live: null,
    fields: [
      { key: "ticker", label: "Ticker", type: "text" },
      { key: "notional", label: "Notional", type: "number", step: 100 },
    ],
    defaults: { ticker: "AAPL", notional: 333 },
    products: ["portfolio"],
  },
  {
    kind: "risk_objective",
    label: "Risk Objective",
    description: "What risk you're willing to take — the target the optimizer solves against",
    color: "var(--amber)",
    category: "param",
    live: null,
    fields: [
      { key: "max_drawdown_target_pct", label: "Max drawdown target (%)", type: "number", step: 1 },
      { key: "horizon_years", label: "Horizon (yrs)", type: "number", step: 0.25 },
    ],
    defaults: { max_drawdown_target_pct: 25, horizon_years: 1.0 },
    products: ["portfolio"],
  },
  {
    kind: "hedge_instrument",
    label: "Hedge Instrument",
    description:
      "How the gap gets closed — protective puts only, single currency, static (no rebalancing) hedge held to horizon",
    color: "var(--orange)",
    category: "output",
    live: null,
    fields: [
      { key: "hedge_strike_pct", label: "Put strike (% of spot)", type: "number", step: 0.01 },
      { key: "n_paths", label: "MC paths", type: "number", step: 1000 },
    ],
    defaults: { hedge_strike_pct: 0.9, n_paths: 20_000 },
    products: ["portfolio"],
  },
];

export const NODE_DEF_MAP: Record<string, NodeDef> = Object.fromEntries(
  NODE_DEFS.map((d) => [d.kind, d])
);

export interface PhoenixRequest {
  product_type: string;
  underlying: string;
  spot: number;
  notional: number;
  maturity_years: number;
  coupon_pa: number;
  coupon_frequency: number;
  capital_barrier: number;
  coupon_barrier: number;
  autocall_level: number;
  autocall_start: number;
  risk_free_rate: number;
  dividend_yield: number;
  volatility: number;
  n_paths: number;
  steps_per_year: number;
  memory: boolean;
  return_distributions: boolean;
  objectives: Record<string, any>;
}

export interface PricingResult {
  kind?: "phoenix";
  fair_value_pct: number;
  fair_value: number;
  mc_confidence_95_pct: number;
  issuer_price_pct: number;
  client_price_pct: number;
  autocall_probability: number;
  barrier_touch_probability: number;
  prob_final_capital_loss: number;
  expected_coupon_pct: number;
  expected_life_years: number;
  greeks: { delta: number; gamma: number; vega: number; theta_1d: number; rho: number };
  stress: { scenario: string; fair_value_pct: number; delta_vs_base_pct: number }[];
  heatmap: { spot_multipliers: number[]; vol_levels: number[]; fair_value_pct: number[][] };
  validation: { score: number; status: string; checks: any[] };
  offline?: boolean;
  /** Whether a connected Stress Test card is present — gates the stress table in the UI. */
  stressVisible?: boolean;
  /** Present only when a connected Term Sheet card requested it. */
  termSheet?: Record<string, any> | null;
}

// ── Vanilla option pricing ──────────────────────────────────────────────────

export interface VanillaOptionRequest {
  underlying: string;
  spot: number;
  strike: number;
  maturity_years: number;
  risk_free_rate: number;
  dividend_yield: number;
  volatility: number;
  option_type: "call" | "put";
  model: "black_scholes" | "binomial" | "monte_carlo" | "heston" | "svi" | "merton" | "local_vol";
  style: "european" | "american";
  n_steps: number;
  n_sims: number;
  seed: number;
  jump_intensity: number;
  jump_mean: number;
  jump_vol: number;
}

export interface OptionGreeks {
  delta: number;
  gamma: number;
  vega?: number;
  vega_v0?: number;
  rho?: number;
  theta_1d?: number;
}

export interface OptionPricingResult {
  kind: "vanilla_option";
  mode: "single";
  model: string;
  price: number;
  greeks: OptionGreeks;
  strike: number;
  spot: number;
  option_type: "call" | "put";
  std_error?: number;
  conf_95_lo?: number;
  conf_95_hi?: number;
  n_sims?: number;
  early_exercise_premium?: number;
  n_steps?: number;
  style?: string;
  jump_intensity?: number;
  jump_mean?: number;
  jump_vol?: number;
  offline?: boolean;
}

export interface ModelComparisonRow {
  model: string;
  price?: number;
  greeks?: OptionGreeks;
  calibrated?: boolean;
  note?: string;
  error?: string;
  std_error?: number;
  early_exercise_premium?: number;
}

export interface ModelComparisonResult {
  kind: "vanilla_option";
  mode: "compare";
  underlying: string;
  option_type: string;
  strike: number;
  results: ModelComparisonRow[];
  offline?: boolean;
}

export interface CalibrationStatus {
  calibrated: boolean;
  ticker: string;
  model: string;
  rmse?: number;
  timestamp?: number;
  arb_free?: boolean;
}

// ── Multi-leg option strategies ─────────────────────────────────────────────

export interface OptionLeg {
  option_type: "call" | "put";
  qty: number; // +1 long, -1 short
  strike: number;
}

export interface StrategyLegResult extends OptionLeg {
  price: number;
  greeks: OptionGreeks;
}

export interface StrategyResult {
  kind: "vanilla_option";
  mode: "strategy";
  strategy: string;
  underlying: string;
  spot: number;
  net_price: number;
  net_greeks: OptionGreeks;
  legs: StrategyLegResult[];
  offline?: boolean;
}

// ── Portfolio hedge optimizer ───────────────────────────────────────────────

export interface PortfolioPositionSpec {
  ticker: string;
  notional: number;
}

export interface PortfolioHedgeRequest {
  positions: PortfolioPositionSpec[];
  max_drawdown_target_pct: number;
  hedge_strike_pct: number;
  horizon_years: number;
  risk_free_rate: number;
  n_paths: number;
}

export interface PortfolioRiskReport {
  var_95_pct: number;
  cvar_95_pct: number;
  mean_pnl_pct: number;
  drawdown_p50_pct: number;
  drawdown_p95_pct: number;
  terminal_pnl_pct: number[];
}

export interface PortfolioFrontierPoint {
  hedge_ratio: number;
  drawdown_p95_pct: number;
  premium_cost_pct: number;
}

export interface PortfolioHedgeResult {
  kind: "portfolio";
  tickers: string[];
  hedge_ratio: number;
  target_met: boolean;
  premium_cost: number;
  premium_cost_pct: number;
  correlation: number[][];
  volatility: Record<string, number>;
  unhedged: PortfolioRiskReport;
  hedged: PortfolioRiskReport;
  frontier: PortfolioFrontierPoint[];
  lookback_days: number;
  offline?: boolean;
}

export type AnyPricingResult =
  | PricingResult
  | OptionPricingResult
  | ModelComparisonResult
  | StrategyResult
  | PortfolioHedgeResult;
