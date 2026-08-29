import {
  Line,
  LineChart,
  Legend,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import type {
  AnyPricingResult,
  ModelComparisonResult,
  OptionPricingResult,
  PortfolioFrontierPoint,
  PortfolioHedgeResult,
  PricingResult,
  StrategyResult,
} from "../types";

// Fixed categorical order, validated for CVD-safe adjacency + contrast
// (see dataviz skill palette validator) — never cycle/reassign per render.
const EXPIRY_COLORS = ["#2f80ed", "#16a34a", "#ec4899", "#8b5cf6", "#06b6d4", "#f59e0b"];

function orangeShade(value: number, min: number, max: number): string {
  if (max === min) return "#fff2e8";
  const t = (value - min) / (max - min);
  // single-hue sequential ramp, light -> dark orange
  const lightness = 92 - t * 55;
  return `hsl(20, 100%, ${lightness}%)`;
}

function Heatmap({ heatmap }: { heatmap: PricingResult["heatmap"] }) {
  // backend shape (structured/heatmap.py): fair_value_pct is indexed [vol_levels][spot_multipliers]
  const flat = heatmap.fair_value_pct.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="mini">
        <thead>
          <tr>
            <th>Vol ×</th>
            {heatmap.spot_multipliers.map((sm, i) => (
              <th key={i}>{sm.toFixed(2)}x spot</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {heatmap.vol_levels.map((v, r) => (
            <tr key={r}>
              <td>
                <b>{(v * 100).toFixed(0)}%</b>
              </td>
              {heatmap.fair_value_pct[r].map((val, c) => (
                <td key={c} style={{ background: orangeShade(val, min, max), textAlign: "center" }}>
                  {val.toFixed(1)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function IVSmileChart({ surface }: { surface: any }) {
  if (!surface?.slices?.length) return null;
  // reshape per-expiry smiles into one array keyed by log-moneyness for recharts
  const kSet = surface.slices[0].smile.map((p: any) => p.log_moneyness);
  const rows = kSet.map((k: number, idx: number) => {
    const row: Record<string, number> = { k: Math.round(k * 1000) / 1000 };
    surface.slices.forEach((s: any) => {
      row[`T=${s.maturity_years.toFixed(2)}y`] = s.smile[idx].implied_vol;
    });
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="var(--line)" strokeDasharray="2 4" />
        <XAxis dataKey="k" tick={{ fontSize: 10 }} label={{ value: "log-moneyness", fontSize: 10, position: "insideBottom", offset: -2 }} />
        <YAxis tick={{ fontSize: 10 }} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} width={38} />
        <Tooltip formatter={(v: number) => `${(v * 100).toFixed(2)}%`} contentStyle={{ fontSize: 11 }} />
        <Legend wrapperStyle={{ fontSize: 10 }} />
        {surface.slices.map((s: any, i: number) => (
          <Line
            key={s.maturity_years}
            type="monotone"
            dataKey={`T=${s.maturity_years.toFixed(2)}y`}
            stroke={EXPIRY_COLORS[i % EXPIRY_COLORS.length]}
            strokeWidth={2}
            dot={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

function PayoffDiagram({
  strike,
  optionType,
  premium,
  spot,
  delta,
  gamma,
}: {
  strike: number;
  optionType: "call" | "put";
  premium: number;
  spot?: number;
  delta?: number;
  gamma?: number;
}) {
  // center the range on both strike and spot — if the option is deep ITM/OTM,
  // a strike-only range would render a flat, uninformative line.
  const center = spot !== undefined ? Math.min(strike, spot) : strike;
  const span = spot !== undefined ? Math.max(strike, spot) : strike;
  const lo = center * 0.6;
  const hi = span * 1.4;
  const points = 41;
  const showToday = spot !== undefined && delta !== undefined && gamma !== undefined;
  const rows = Array.from({ length: points }, (_, i) => {
    const s = lo + ((hi - lo) * i) / (points - 1);
    const intrinsic = optionType === "call" ? Math.max(s - strike, 0) : Math.max(strike - s, 0);
    const row: Record<string, number> = {
      spot: Math.round(s * 100) / 100,
      "P&L expiry": Math.round((intrinsic - premium) * 100) / 100,
    };
    if (showToday) {
      // 2nd-order Taylor approx of today's mark-to-market P&L around spot, from delta/gamma
      const ds = s - (spot as number);
      row["P&L hoy"] = Math.round(((delta as number) * ds + 0.5 * (gamma as number) * ds * ds) * 100) / 100;
    }
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="var(--line)" strokeDasharray="2 4" />
        <XAxis dataKey="spot" tick={{ fontSize: 10 }} label={{ value: "spot at expiry", fontSize: 10, position: "insideBottom", offset: -2 }} />
        <YAxis tick={{ fontSize: 10 }} width={38} />
        <Tooltip contentStyle={{ fontSize: 11 }} />
        {showToday && <Legend wrapperStyle={{ fontSize: 10 }} />}
        <ReferenceLine y={0} stroke="var(--muted)" strokeDasharray="3 3" />
        <ReferenceLine x={strike} stroke="var(--amber)" strokeDasharray="3 3" label={{ value: "strike", fontSize: 9, position: "top" }} />
        <Line type="monotone" dataKey="P&L expiry" stroke="var(--orange)" strokeWidth={2} dot={false} />
        {showToday && <Line type="monotone" dataKey="P&L hoy" stroke="var(--blue)" strokeWidth={2} strokeDasharray="4 3" dot={false} />}
      </LineChart>
    </ResponsiveContainer>
  );
}

function StrategyPayoffDiagram({
  legs,
  netPrice,
  spot,
}: {
  legs: { option_type: "call" | "put"; qty: number; strike: number }[];
  netPrice: number;
  spot: number;
}) {
  const strikes = legs.map((l) => l.strike);
  const lo = Math.min(spot, ...strikes) * 0.6;
  const hi = Math.max(spot, ...strikes) * 1.4;
  const points = 41;
  const rows = Array.from({ length: points }, (_, i) => {
    const s = lo + ((hi - lo) * i) / (points - 1);
    const intrinsic = legs.reduce(
      (sum, l) => sum + l.qty * (l.option_type === "call" ? Math.max(s - l.strike, 0) : Math.max(l.strike - s, 0)),
      0
    );
    return { spot: Math.round(s * 100) / 100, pnl: Math.round((intrinsic - netPrice) * 100) / 100 };
  });

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="var(--line)" strokeDasharray="2 4" />
        <XAxis dataKey="spot" tick={{ fontSize: 10 }} label={{ value: "spot at expiry", fontSize: 10, position: "insideBottom", offset: -2 }} />
        <YAxis tick={{ fontSize: 10 }} width={38} />
        <Tooltip contentStyle={{ fontSize: 11 }} />
        <ReferenceLine y={0} stroke="var(--muted)" strokeDasharray="3 3" />
        {legs.map((l, i) => (
          <ReferenceLine key={i} x={l.strike} stroke="var(--amber)" strokeDasharray="3 3" />
        ))}
        <Line type="monotone" dataKey="pnl" stroke="var(--orange)" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function OptionResult({ result }: { result: OptionPricingResult }) {
  const g = result.greeks;
  return (
    <>
      <div className="kpi-grid">
        <div className="kpi"><span>Price ({result.model})</span><b>{result.price.toFixed(3)}</b></div>
        <div className="kpi"><span>Delta</span><b>{g.delta.toFixed(3)}</b></div>
        <div className="kpi"><span>Gamma</span><b>{g.gamma.toFixed(4)}</b></div>
        {g.vega !== undefined && <div className="kpi"><span>Vega</span><b>{g.vega.toFixed(3)}</b></div>}
        {g.vega_v0 !== undefined && (
          <div className="kpi"><span>Vega (v0-sensitivity)</span><b>{g.vega_v0.toFixed(3)}</b></div>
        )}
        {g.rho !== undefined && <div className="kpi"><span>Rho</span><b>{g.rho.toFixed(3)}</b></div>}
        {g.theta_1d !== undefined && <div className="kpi"><span>Theta (1d)</span><b>{g.theta_1d.toFixed(4)}</b></div>}
        {result.std_error !== undefined && (
          <div className="kpi"><span>MC std error</span><b>{result.std_error.toFixed(4)}</b></div>
        )}
        {result.early_exercise_premium !== undefined && (
          <div className="kpi"><span>Early-exercise premium</span><b>{result.early_exercise_premium.toFixed(3)}</b></div>
        )}
        {result.jump_intensity !== undefined && (
          <div className="kpi">
            <span>Jump params</span>
            <b>λ={result.jump_intensity} μ={result.jump_mean} δ={result.jump_vol}</b>
          </div>
        )}
      </div>
      {g.vega_v0 !== undefined && (
        <div className="offline-banner" style={{ background: "#2f80ed1a", color: "#1d4ed8", border: "1px solid #2f80ed55" }}>
          "Vega (v0-sensitivity)" is the price sensitivity to Heston's calibrated
          instantaneous variance parameter — not the same quantity as Black-Scholes
          vega (sensitivity to a single flat volatility), which doesn't exist for a
          stochastic-vol model.
        </div>
      )}
      <h2>Payoff at expiry</h2>
      <PayoffDiagram
        strike={result.strike}
        optionType={result.option_type}
        premium={result.price}
        spot={result.spot}
        delta={g.delta}
        gamma={g.gamma}
      />
    </>
  );
}

function comparisonNote(r: ModelComparisonResult["results"][number]): string {
  if (r.note) return r.note;
  if (r.error) return r.error;
  if (r.std_error !== undefined) return `std err ${r.std_error.toFixed(4)}`;
  if (r.early_exercise_premium !== undefined) return `early-ex ${r.early_exercise_premium.toFixed(3)}`;
  return "";
}

function ComparisonResult({ result }: { result: ModelComparisonResult }) {
  return (
    <>
      <h2>Model comparison — {result.underlying} {result.strike} {result.option_type}</h2>
      <table className="mini">
        <thead>
          <tr><th>Model</th><th>Price</th><th>Delta</th><th>Notes</th></tr>
        </thead>
        <tbody>
          {result.results.map((r, i) => (
            <tr key={i}>
              <td>{r.model}</td>
              <td>{r.price !== undefined ? r.price.toFixed(3) : "—"}</td>
              <td>{r.greeks?.delta !== undefined ? r.greeks.delta.toFixed(3) : "—"}</td>
              <td>{comparisonNote(r)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function StrategyResultView({ result }: { result: StrategyResult }) {
  const g = result.net_greeks;
  return (
    <>
      <h2>{result.strategy.replace(/_/g, " ")} — {result.underlying}</h2>
      <div className="kpi-grid">
        <div className="kpi"><span>Net premium</span><b>{result.net_price.toFixed(3)}</b></div>
        <div className="kpi"><span>Net delta</span><b>{(g.delta ?? 0).toFixed(3)}</b></div>
        <div className="kpi"><span>Net gamma</span><b>{(g.gamma ?? 0).toFixed(4)}</b></div>
        <div className="kpi"><span>Net vega</span><b>{(g.vega ?? 0).toFixed(3)}</b></div>
      </div>
      <table className="mini">
        <thead>
          <tr><th>Leg</th><th>Strike</th><th>Qty</th><th>Price</th></tr>
        </thead>
        <tbody>
          {result.legs.map((l, i) => (
            <tr key={i}>
              <td>{l.option_type}</td>
              <td>{l.strike.toFixed(2)}</td>
              <td>{l.qty > 0 ? `+${l.qty}` : l.qty}</td>
              <td>{l.price.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h2>Payoff at expiry</h2>
      <StrategyPayoffDiagram legs={result.legs} netPrice={result.net_price} spot={result.spot} />
    </>
  );
}

function CorrelationTable({ tickers, correlation }: { tickers: string[]; correlation: number[][] }) {
  return (
    <table className="mini">
      <thead>
        <tr>
          <th></th>
          {tickers.map((t) => (
            <th key={t}>{t}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {tickers.map((t, i) => (
          <tr key={t}>
            <td><b>{t}</b></td>
            {tickers.map((_, j) => (
              <td key={j} style={{ background: orangeShade(Math.abs(correlation[i][j]), 0, 1), textAlign: "center" }}>
                {correlation[i][j].toFixed(2)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function PnlDistributionChart({ unhedged, hedged }: { unhedged: number[]; hedged: number[] }) {
  const sortedU = [...unhedged].sort((a, b) => a - b);
  const sortedH = [...hedged].sort((a, b) => a - b);
  const n = Math.max(2, Math.min(sortedU.length, sortedH.length));
  const rows = Array.from({ length: n }, (_, i) => {
    const t = i / (n - 1);
    return {
      pct: Math.round(t * 100),
      "Sin cobertura": Math.round(sortedU[Math.round(t * (sortedU.length - 1))] * 100) / 100,
      "Con cobertura": Math.round(sortedH[Math.round(t * (sortedH.length - 1))] * 100) / 100,
    };
  });
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="var(--line)" strokeDasharray="2 4" />
        <XAxis dataKey="pct" tick={{ fontSize: 10 }} label={{ value: "percentil", fontSize: 10, position: "insideBottom", offset: -2 }} />
        <YAxis tick={{ fontSize: 10 }} width={44} tickFormatter={(v) => `${v}%`} />
        <Tooltip contentStyle={{ fontSize: 11 }} formatter={(v: number) => `${(v as number).toFixed(1)}%`} />
        <Legend wrapperStyle={{ fontSize: 10 }} />
        <ReferenceLine y={0} stroke="var(--muted)" strokeDasharray="3 3" />
        <Line type="monotone" dataKey="Sin cobertura" stroke="var(--red)" strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="Con cobertura" stroke="var(--green)" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function FrontierChart({ frontier, chosenRatio }: { frontier: PortfolioFrontierPoint[]; chosenRatio: number }) {
  const chosen = frontier.reduce((best, p) =>
    Math.abs(p.hedge_ratio - chosenRatio) < Math.abs(best.hedge_ratio - chosenRatio) ? p : best, frontier[0]
  );
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={frontier} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="var(--line)" strokeDasharray="2 4" />
        <XAxis
          dataKey="drawdown_p95_pct"
          type="number"
          domain={["dataMin", "dataMax"]}
          reversed
          tick={{ fontSize: 10 }}
          tickFormatter={(v) => `${(v as number).toFixed(0)}%`}
          label={{ value: "drawdown p95% (menor = más protegido)", fontSize: 10, position: "insideBottom", offset: -2 }}
        />
        <YAxis dataKey="premium_cost_pct" tick={{ fontSize: 10 }} width={44} tickFormatter={(v) => `${(v as number).toFixed(1)}%`} />
        <Tooltip
          contentStyle={{ fontSize: 11 }}
          formatter={(v: number) => `${(v as number).toFixed(2)}%`}
          labelFormatter={(v) => `drawdown p95: ${(v as number).toFixed(1)}%`}
        />
        <Line type="monotone" dataKey="premium_cost_pct" name="Costo prima" stroke="var(--orange)" strokeWidth={2} dot={{ r: 3 }} />
        <ReferenceDot
          x={chosen.drawdown_p95_pct}
          y={chosen.premium_cost_pct}
          r={6}
          fill="var(--green)"
          stroke="#fff"
          strokeWidth={2}
          label={{ value: "elegido", position: "top", fontSize: 10 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

function PortfolioHedgeResultView({ result }: { result: PortfolioHedgeResult }) {
  return (
    <>
      <h2>Cobertura de cartera — {result.tickers.join(", ")}</h2>
      {!result.target_met && (
        <div className="run-error">
          El objetivo de drawdown no se alcanza ni con cobertura al 100% con este strike/horizonte. Mostrando el
          mejor esfuerzo posible (hedge ratio 100%).
        </div>
      )}
      <div className="kpi-grid">
        <div className="kpi"><span>Hedge ratio</span><b>{(result.hedge_ratio * 100).toFixed(1)}%</b></div>
        <div className="kpi"><span>Costo prima</span><b>{result.premium_cost_pct.toFixed(2)}%</b></div>
        <div className="kpi"><span>Drawdown p95 sin cobertura</span><b>{result.unhedged.drawdown_p95_pct.toFixed(1)}%</b></div>
        <div className="kpi"><span>Drawdown p95 con cobertura</span><b>{result.hedged.drawdown_p95_pct.toFixed(1)}%</b></div>
        <div className="kpi"><span>VaR 95% sin cobertura</span><b>{result.unhedged.var_95_pct.toFixed(1)}%</b></div>
        <div className="kpi"><span>VaR 95% con cobertura</span><b>{result.hedged.var_95_pct.toFixed(1)}%</b></div>
      </div>

      <h2>Costo vs. protección (frontera)</h2>
      <p style={{ fontSize: 11.5, color: "var(--muted)", marginTop: -6 }}>
        El punto verde es el hedge ratio elegido — el resto de la curva muestra qué hubiera costado más o menos
        protección, para juzgar la elección en vez de confiar en un número suelto.
      </p>
      <FrontierChart frontier={result.frontier} chosenRatio={result.hedge_ratio} />

      <h2>Correlación real (histórica, 1 año)</h2>
      <CorrelationTable tickers={result.tickers} correlation={result.correlation} />

      <h2>Distribución de P&L al horizonte</h2>
      <PnlDistributionChart unhedged={result.unhedged.terminal_pnl_pct} hedged={result.hedged.terminal_pnl_pct} />
    </>
  );
}

function TermSheetView({ termSheet }: { termSheet: Record<string, any> }) {
  const rows: [string, any][] = [
    ["Título", termSheet.title],
    ["Emisor", termSheet.issuer],
    ["Moneda", termSheet.currency],
    ["Subyacente", termSheet.underlying],
    ["Notional", termSheet.notional],
    ["Maturity (yrs)", termSheet.maturity_years],
    ["Cupón p.a.", termSheet.coupon],
    ["Nivel autocall", termSheet.autocall_level],
    ["Barrera de capital", termSheet.capital_barrier],
    ["Cupón memoria", termSheet.memory_coupon ? "sí" : "no"],
  ];
  return (
    <>
      <h2>Term sheet indicativo</h2>
      <table className="mini">
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label}>
              <td><b>{label}</b></td>
              <td>{value === undefined || value === null ? "—" : String(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {termSheet.risk_note && <p style={{ fontSize: 11, color: "var(--muted)" }}>{termSheet.risk_note}</p>}
    </>
  );
}

function PhoenixResult({ result }: { result: PricingResult }) {
  return (
    <>
      <div className="kpi-grid">
        <div className="kpi"><span>Fair value</span><b>{result.fair_value_pct.toFixed(2)}%</b></div>
        <div className="kpi"><span>Autocall prob.</span><b>{(result.autocall_probability * 100).toFixed(1)}%</b></div>
        <div className="kpi"><span>Barrier touch prob.</span><b>{(result.barrier_touch_probability * 100).toFixed(1)}%</b></div>
        <div className="kpi"><span>Expected life</span><b>{result.expected_life_years.toFixed(2)}y</b></div>
        <div className="kpi"><span>Delta</span><b>{result.greeks.delta.toFixed(3)}</b></div>
        <div className="kpi"><span>Vega</span><b>{result.greeks.vega.toFixed(3)}</b></div>
        <div className="kpi"><span>Gamma</span><b>{result.greeks.gamma.toFixed(4)}</b></div>
        <div className="kpi"><span>Theta (1d)</span><b>{result.greeks.theta_1d.toFixed(4)}</b></div>
      </div>

      {result.stressVisible && (
        <>
          <h2>Stress scenarios</h2>
          <table className="mini">
            <thead>
              <tr><th>Scenario</th><th>Fair value %</th><th>Δ vs base</th></tr>
            </thead>
            <tbody>
              {result.stress.map((s, i) => (
                <tr key={i}>
                  <td>{s.scenario}</td>
                  <td>{s.fair_value_pct.toFixed(2)}</td>
                  <td>{s.delta_vs_base_pct.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      <h2>Spot × vol heatmap (fair value %)</h2>
      <Heatmap heatmap={result.heatmap} />

      {result.termSheet && <TermSheetView termSheet={result.termSheet} />}
    </>
  );
}

export default function ResultsPanel({
  result,
  volSurface,
  missing,
  runError,
}: {
  result: AnyPricingResult | null;
  volSurface: any | null;
  missing: string[];
  runError?: string | null;
}) {
  return (
    <div className="results">
      <h2 style={{ marginTop: 0, fontSize: 11, textTransform: "uppercase", color: "var(--muted)" }}>Results</h2>

      {runError && <div className="run-error">{runError}</div>}

      {!result && !runError && (
        <div className="empty-hint">
          {missing.length > 0
            ? `Add/connect: ${missing.join(", ")}, then hit Run.`
            : "Hit Run to price the graph."}
        </div>
      )}

      {result?.offline && (
        <div className="offline-banner">
          Offline demo mode — backend unreachable. These numbers are from a simplified local
          formula, not the real pricing engine.
        </div>
      )}

      {result && result.kind === "vanilla_option" && result.mode === "single" && <OptionResult result={result} />}
      {result && result.kind === "vanilla_option" && result.mode === "compare" && <ComparisonResult result={result} />}
      {result && result.kind === "vanilla_option" && result.mode === "strategy" && <StrategyResultView result={result} />}
      {result && result.kind === "portfolio" && <PortfolioHedgeResultView result={result} />}
      {result && result.kind !== "vanilla_option" && result.kind !== "portfolio" && <PhoenixResult result={result} />}

      {volSurface && (
        <>
          <h2>Live implied vol smile</h2>
          <IVSmileChart surface={volSurface} />
        </>
      )}
    </div>
  );
}
