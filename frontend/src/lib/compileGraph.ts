import type { Edge, Node } from "@xyflow/react";
import type { PhoenixRequest, PortfolioHedgeRequest, VanillaOptionRequest } from "../types";

export interface LiveData {
  quote?: { price: number; currency?: string };
  rates?: { tenors: Record<string, number>; source: string; interpolated_rate?: number };
  volatility?: { realized_vol_63d?: number | null; realized_vol_21d?: number | null };
  calibration?: { calibrated: boolean; rmse?: number; timestamp?: number };
  dividend?: { dividend_yield: number };
}

export type LiveDataStore = Record<string, LiveData>;
export type ProductType = "phoenix" | "vanilla_option" | "portfolio";

function findByKind(nodes: Node[], kind: string): Node | undefined {
  return nodes.find((n) => n.data?.kind === kind);
}

function findAllByKind(nodes: Node[], kind: string): Node[] {
  return nodes.filter((n) => n.data?.kind === kind);
}

/** All node ids reachable from `rootId` by following edges in either direction —
 * this is what makes wiring a card into the canvas actually matter: a node that
 * exists but isn't connected (by any path) to the root is treated as absent. */
export function connectedComponent(edges: Edge[], rootId: string): Set<string> {
  const adjacency = new Map<string, string[]>();
  edges.forEach((e) => {
    adjacency.set(e.source, [...(adjacency.get(e.source) ?? []), e.target]);
    adjacency.set(e.target, [...(adjacency.get(e.target) ?? []), e.source]);
  });
  const seen = new Set<string>([rootId]);
  const queue = [rootId];
  while (queue.length) {
    const current = queue.shift() as string;
    for (const next of adjacency.get(current) ?? []) {
      if (!seen.has(next)) {
        seen.add(next);
        queue.push(next);
      }
    }
  }
  return seen;
}

export interface CompileResult {
  spec: PhoenixRequest | null;
  missing: string[];
}

export interface VanillaCompileResult {
  spec: VanillaOptionRequest | null;
  missing: string[];
}

export interface CompileOptions {
  requireLiveSpot?: boolean;
  productType?: ProductType;
}

/** Dispatches to the right per-product compiler based on `productType`. */
export function compileGraph(
  nodes: Node[],
  edges: Edge[],
  live: LiveDataStore,
  options: CompileOptions = {}
): CompileResult | VanillaCompileResult {
  if (options.productType === "vanilla_option") {
    return compileVanillaOptionGraph(nodes, edges, live, options);
  }
  return compilePhoenixGraph(nodes, edges, live, options);
}

/** Walks the node graph and merges each node's params into a flat PhoenixRequest,
 * using the exact field names the FastAPI backend expects (api/schemas.py).
 *
 * Connectivity matters: a node has to be reachable, by some chain of wires, from
 * the Pricing Output node to be picked up. A card sitting disconnected on the
 * canvas is treated the same as a missing card — that's what makes drawing (or
 * deleting) a connection actually change the result instead of being cosmetic.
 *
 * `requireLiveSpot` is false in offline/demo mode: the mock pricing formula
 * doesn't depend on spot, so we shouldn't block the whole run just because
 * there's no live quote when the backend is unreachable anyway. */
export function compilePhoenixGraph(
  nodes: Node[],
  edges: Edge[],
  live: LiveDataStore,
  options: { requireLiveSpot?: boolean } = {}
): CompileResult {
  const { requireLiveSpot = true } = options;
  const missing: string[] = [];

  const pricingOutput = findByKind(nodes, "pricing_output");
  if (!pricingOutput) missing.push("Pricing Output");
  const connected = pricingOutput ? connectedComponent(edges, pricingOutput.id) : new Set<string>();

  const required = (kind: string, label: string): Node | undefined => {
    const node = findByKind(nodes, kind);
    if (!node) {
      missing.push(label);
      return undefined;
    }
    if (pricingOutput && !connected.has(node.id)) {
      missing.push(`${label} (no conectada a Pricing Output)`);
      return undefined;
    }
    return node;
  };

  // optional nodes: used only if wired in; otherwise silently fall back to defaults
  const optional = (kind: string): Node | undefined => {
    const node = findByKind(nodes, kind);
    return node && pricingOutput && connected.has(node.id) ? node : undefined;
  };

  const underlying = required("underlying", "Underlying");
  const barrier = required("barrier", "Barrier");
  const coupon = required("coupon", "Coupon");
  const autocall = required("autocall", "Autocall");
  const memory = optional("memory");
  const ratesCurve = optional("rates_curve");
  const volForecast = optional("volatility_forecast");

  const ticker = (underlying?.data.params as any)?.ticker ?? "AAPL";
  const liveSpot = live[underlying?.id ?? ""]?.quote?.price;
  if (liveSpot === undefined && requireLiveSpot) missing.push(`live quote for ${ticker}`);
  const spot = liveSpot ?? 100;

  const ratesLive = live[ratesCurve?.id ?? ""]?.rates;
  const riskFreeRate = ratesLive?.interpolated_rate ?? ratesLive?.tenors?.["1Y"] ?? 0.045;

  const volParams = (volForecast?.data.params as any) ?? { mode: "realized", manual_volatility: 0.28 };
  const volLive = live[volForecast?.id ?? ""]?.volatility;
  const volatility =
    volParams.mode === "manual"
      ? volParams.manual_volatility
      : volLive?.realized_vol_63d ?? volLive?.realized_vol_21d ?? 0.28;

  if (missing.length > 0) return { spec: null, missing };

  const barrierParams = (barrier?.data.params as any) ?? {};
  const capitalBarrier = barrierParams.capital_barrier;
  const dividendYield = (underlying?.data.params as any)?.dividend_yield ?? 0;

  const spec: PhoenixRequest = {
    product_type: "phoenix_autocall",
    underlying: ticker,
    spot: spot as number,
    notional: (pricingOutput?.data.params as any).notional,
    maturity_years: (pricingOutput?.data.params as any).maturity_years,
    coupon_pa: (coupon?.data.params as any).coupon_pa,
    coupon_frequency: (coupon?.data.params as any).coupon_frequency,
    capital_barrier: capitalBarrier,
    coupon_barrier: barrierParams.separate_coupon_barrier ? barrierParams.coupon_barrier : capitalBarrier,
    autocall_level: (autocall?.data.params as any).autocall_level,
    autocall_start: (autocall?.data.params as any).autocall_start,
    risk_free_rate: riskFreeRate,
    dividend_yield: dividendYield,
    volatility,
    n_paths: (pricingOutput?.data.params as any).n_paths,
    steps_per_year: (pricingOutput?.data.params as any).steps_per_year,
    memory: (memory?.data.params as any)?.enabled ?? true,
    return_distributions: true,
    objectives: {},
  };

  return { spec, missing: [] };
}

/** Walks the node graph and merges Underlying/Rates Curve/Volatility Forecast/
 * Option Contract nodes into a flat VanillaOptionRequest, matching api/schemas.py.
 * Same connectivity rule as the Phoenix compiler: nodes must be wired to Option
 * Contract (directly or through Underlying) to count. */
export function compileVanillaOptionGraph(
  nodes: Node[],
  edges: Edge[],
  live: LiveDataStore,
  options: { requireLiveSpot?: boolean } = {}
): VanillaCompileResult {
  const { requireLiveSpot = true } = options;
  const missing: string[] = [];

  const optionContract = findByKind(nodes, "option_contract");
  if (!optionContract) missing.push("Option Contract");
  const connected = optionContract ? connectedComponent(edges, optionContract.id) : new Set<string>();

  const required = (kind: string, label: string): Node | undefined => {
    const node = findByKind(nodes, kind);
    if (!node) {
      missing.push(label);
      return undefined;
    }
    if (optionContract && !connected.has(node.id)) {
      missing.push(`${label} (no conectada a Option Contract)`);
      return undefined;
    }
    return node;
  };

  const optional = (kind: string): Node | undefined => {
    const node = findByKind(nodes, kind);
    return node && optionContract && connected.has(node.id) ? node : undefined;
  };

  const underlying = required("underlying", "Underlying");
  const ratesCurve = optional("rates_curve");
  const volForecast = optional("volatility_forecast");

  const ticker = (underlying?.data.params as any)?.ticker ?? "AAPL";
  const liveSpot = live[underlying?.id ?? ""]?.quote?.price;
  if (liveSpot === undefined && requireLiveSpot) missing.push(`live quote for ${ticker}`);
  const spot = liveSpot ?? 100;

  const ratesLive = live[ratesCurve?.id ?? ""]?.rates;
  const riskFreeRate = ratesLive?.interpolated_rate ?? ratesLive?.tenors?.["1Y"] ?? 0.045;

  const volParams = (volForecast?.data.params as any) ?? { mode: "realized", manual_volatility: 0.28 };
  const volLive = live[volForecast?.id ?? ""]?.volatility;
  const volatility =
    volParams.mode === "manual"
      ? volParams.manual_volatility
      : volLive?.realized_vol_63d ?? volLive?.realized_vol_21d ?? 0.28;

  const dividendYield = live[underlying?.id ?? ""]?.dividend?.dividend_yield ?? 0;

  if (missing.length > 0) return { spec: null, missing };

  const p = (optionContract?.data.params as any) ?? {};
  const spec: VanillaOptionRequest = {
    underlying: ticker,
    spot: spot as number,
    strike: p.strike,
    maturity_years: p.maturity_years ?? 1.0,
    risk_free_rate: riskFreeRate,
    dividend_yield: dividendYield,
    volatility,
    option_type: p.option_type,
    model: p.model,
    style: p.style ?? "european",
    n_steps: p.n_steps ?? 500,
    n_sims: p.n_sims ?? 20_000,
    seed: p.seed ?? 42,
    jump_intensity: p.jump_intensity ?? 2.0,
    jump_mean: p.jump_mean ?? -0.05,
    jump_vol: p.jump_vol ?? 0.10,
  };

  return { spec, missing: [] };
}

// ── Multi-leg option strategies ─────────────────────────────────────────────

export interface StrategyLegSpec {
  option_type: "call" | "put";
  qty: number; // +1 long, -1 short
  strike_mult: number; // multiplier applied to spot
}

/** Pure-options 2-leg combinations, strikes expressed as spot multipliers so
 * a single `otm_pct` field shapes every leg. Deliberately excludes
 * stock+option combos (covered call, protective put) — this tool only prices
 * options, no equity position, so it stays limited to combinations built
 * entirely from calls/puts. */
export const STRATEGY_LEGS: Record<string, (otmPct: number) => StrategyLegSpec[]> = {
  bull_call_spread: (o) => [
    { option_type: "call", qty: 1, strike_mult: 1.0 },
    { option_type: "call", qty: -1, strike_mult: 1 + o },
  ],
  bear_put_spread: (o) => [
    { option_type: "put", qty: 1, strike_mult: 1.0 },
    { option_type: "put", qty: -1, strike_mult: 1 - o },
  ],
  straddle: () => [
    { option_type: "call", qty: 1, strike_mult: 1.0 },
    { option_type: "put", qty: 1, strike_mult: 1.0 },
  ],
  strangle: (o) => [
    { option_type: "call", qty: 1, strike_mult: 1 + o },
    { option_type: "put", qty: 1, strike_mult: 1 - o },
  ],
  risk_reversal: (o) => [
    { option_type: "put", qty: 1, strike_mult: 1 - o },
    { option_type: "call", qty: -1, strike_mult: 1 + o },
  ],
};

export interface StrategyCompileResult {
  strategy: string;
  legs: { leg: StrategyLegSpec; spec: VanillaOptionRequest }[] | null;
  missing: string[];
}

/** Same connectivity rule as the single-contract compiler, rooted at the
 * Option Strategy node instead of Option Contract. */
export function compileOptionStrategyGraph(
  nodes: Node[],
  edges: Edge[],
  live: LiveDataStore,
  options: { requireLiveSpot?: boolean } = {}
): StrategyCompileResult {
  const { requireLiveSpot = true } = options;
  const missing: string[] = [];

  const strategyNode = findByKind(nodes, "option_strategy");
  if (!strategyNode) return { strategy: "", legs: null, missing: ["Option Strategy"] };
  const connected = connectedComponent(edges, strategyNode.id);

  const required = (kind: string, label: string): Node | undefined => {
    const node = findByKind(nodes, kind);
    if (!node) {
      missing.push(label);
      return undefined;
    }
    if (!connected.has(node.id)) {
      missing.push(`${label} (no conectada a Option Strategy)`);
      return undefined;
    }
    return node;
  };
  const optional = (kind: string): Node | undefined => {
    const node = findByKind(nodes, kind);
    return node && connected.has(node.id) ? node : undefined;
  };

  const underlying = required("underlying", "Underlying");
  const ratesCurve = optional("rates_curve");
  const volForecast = optional("volatility_forecast");

  const ticker = (underlying?.data.params as any)?.ticker ?? "AAPL";
  const liveSpot = live[underlying?.id ?? ""]?.quote?.price;
  if (liveSpot === undefined && requireLiveSpot) missing.push(`live quote for ${ticker}`);
  const spot = liveSpot ?? 100;

  const ratesLive = live[ratesCurve?.id ?? ""]?.rates;
  const riskFreeRate = ratesLive?.interpolated_rate ?? ratesLive?.tenors?.["1Y"] ?? 0.045;

  const volParams = (volForecast?.data.params as any) ?? { mode: "realized", manual_volatility: 0.28 };
  const volLive = live[volForecast?.id ?? ""]?.volatility;
  const volatility =
    volParams.mode === "manual"
      ? volParams.manual_volatility
      : volLive?.realized_vol_63d ?? volLive?.realized_vol_21d ?? 0.28;

  const dividendYield = live[underlying?.id ?? ""]?.dividend?.dividend_yield ?? 0;

  if (missing.length > 0) return { strategy: "", legs: null, missing };

  const p = (strategyNode.data.params as any) ?? {};
  const strategy: string = p.strategy ?? "bull_call_spread";
  const legSpecs = (STRATEGY_LEGS[strategy] ?? STRATEGY_LEGS.bull_call_spread)(p.otm_pct ?? 0.05);

  const legs = legSpecs.map((leg) => ({
    leg,
    spec: {
      underlying: ticker,
      spot: spot as number,
      strike: (spot as number) * leg.strike_mult,
      maturity_years: p.maturity_years ?? 1.0,
      risk_free_rate: riskFreeRate,
      dividend_yield: dividendYield,
      volatility,
      option_type: leg.option_type,
      model: p.model ?? "black_scholes",
      style: "european" as const,
      n_steps: 500,
      n_sims: 20_000,
      seed: 42,
      jump_intensity: 2.0,
      jump_mean: -0.05,
      jump_vol: 0.10,
    } as VanillaOptionRequest,
  }));

  return { strategy, legs, missing: [] };
}

// ── Portfolio hedge optimizer ───────────────────────────────────────────────

export interface PortfolioCompileResult {
  spec: PortfolioHedgeRequest | null;
  missing: string[];
}

/** Collects every connected `portfolio_position` card (repeatable — unlike
 * every other product's single-instance nodes), the `risk_objective` (what
 * risk you're willing to take), and the `hedge_instrument` (how the gap gets
 * closed — the root/output) into the hedge-optimizer request. No live-data
 * prefetch needed: the backend fetches real historical vol/correlation
 * itself. */
export function compilePortfolioGraph(nodes: Node[], edges: Edge[]): PortfolioCompileResult {
  const missing: string[] = [];

  const hedgeInstrument = findByKind(nodes, "hedge_instrument");
  if (!hedgeInstrument) missing.push("Hedge Instrument");
  const connected = hedgeInstrument ? connectedComponent(edges, hedgeInstrument.id) : new Set<string>();

  const required = (kind: string, label: string): Node | undefined => {
    const node = findByKind(nodes, kind);
    if (!node) {
      missing.push(label);
      return undefined;
    }
    if (hedgeInstrument && !connected.has(node.id)) {
      missing.push(`${label} (no conectada a Hedge Instrument)`);
      return undefined;
    }
    return node;
  };

  const riskObjective = required("risk_objective", "Risk Objective");

  const positionNodes = findAllByKind(nodes, "portfolio_position").filter(
    (n) => hedgeInstrument && connected.has(n.id)
  );
  if (positionNodes.length === 0) missing.push("Position (at least one, connected to Hedge Instrument)");

  if (missing.length > 0) return { spec: null, missing };

  const positions = positionNodes.map((n) => {
    const p = (n.data.params as any) ?? {};
    return { ticker: String(p.ticker ?? "AAPL").toUpperCase(), notional: p.notional ?? 100 };
  });

  const objective = (riskObjective?.data.params as any) ?? {};
  const instrument = (hedgeInstrument?.data.params as any) ?? {};
  const spec: PortfolioHedgeRequest = {
    positions,
    max_drawdown_target_pct: objective.max_drawdown_target_pct ?? 15,
    horizon_years: objective.horizon_years ?? 1.0,
    hedge_strike_pct: instrument.hedge_strike_pct ?? 0.9,
    risk_free_rate: 0.045,
    n_paths: instrument.n_paths ?? 20_000,
  };

  return { spec, missing: [] };
}
