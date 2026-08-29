const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function getJSON(path: string) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status}`);
  return res.json();
}

async function postJSON(path: string, body: unknown) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? `${path} -> HTTP ${res.status}`);
  }
  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const r = await getJSON("/api/health");
    return r.status === "ok";
  } catch {
    return false;
  }
}

export function getQuote(symbol: string) {
  return getJSON(`/api/market/quote/${encodeURIComponent(symbol)}`);
}

export function getRatesCurve(currency: string, maturityYears?: number) {
  const suffix = maturityYears !== undefined ? `?maturity_years=${maturityYears}` : "";
  return getJSON(`/api/market/rates/${encodeURIComponent(currency)}${suffix}`);
}

export function getDividendYield(symbol: string) {
  return getJSON(`/api/market/dividend-yield/${encodeURIComponent(symbol)}`);
}

export function getCalibration(symbol: string) {
  return getJSON(`/api/market/calibration/${encodeURIComponent(symbol)}`);
}

export function getVolSurface(symbol: string) {
  return getJSON(`/api/market/vol-surface/${encodeURIComponent(symbol)}`);
}

export function pricePhoenix(spec: Record<string, any>) {
  return postJSON(
    "/api/price/phoenix?include_greeks=true&include_stress=true&include_heatmap=true",
    spec
  );
}

export function termSheetPhoenix(spec: Record<string, any>) {
  return postJSON("/api/term-sheet/phoenix", spec);
}

export function priceOption(spec: Record<string, any>) {
  return postJSON("/api/price/option", spec);
}

export function compareOptionModels(spec: Record<string, any>) {
  return postJSON("/api/price/option/compare", spec);
}

export function hedgePortfolio(spec: Record<string, any>) {
  return postJSON("/api/portfolio/hedge", spec);
}

export async function calibrateModel(ticker: string, model: "heston" | "svi" | "local_vol", source: "yfinance" | "synthetic" = "yfinance") {
  const res = await fetch(
    `${API_BASE}/api/calibrate/${encodeURIComponent(ticker)}?model=${model}&source=${source}`,
    { method: "POST" }
  );
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? `calibrate -> HTTP ${res.status}`);
  }
  return res.json();
}

export function getCalibrationStatus(ticker: string, model: "heston" | "svi" | "local_vol") {
  return getJSON(`/api/calibration/${encodeURIComponent(ticker)}?model=${model}`);
}
