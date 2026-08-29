# Visual Workbench

The workbench in `frontend/` is a real node-based canvas (React + React Flow) for constructing a Phoenix Autocall workflow — drag blocks from the palette, connect them, edit each node's parameters inline, and hit Run.

Node types: Manual Trigger, Underlying, Rates Curve, Volatility Forecast, Implied Vol Surface, Barrier, Coupon, Memory, Autocall, Capital Protection, Payoff Aggregator, Pricing Output, Stress Test, Model Comparison, Term Sheet.

Running the graph:

1. Fetches live market data per node (spot + history, FRED rates curve, realized vol, and — for the Implied Vol Surface node — a real option chain calibrated to SSVI).
2. Compiles the graph into a flat product specification (`frontend/src/lib/compileGraph.ts`), using the exact field names the FastAPI backend expects.
3. Calls `POST /api/price/phoenix` and renders fair value, probabilities, Greeks, stress scenarios, and a spot×vol heatmap.

If the backend is unavailable, the UI switches to a clearly labeled offline/demo mode with a simplified local formula, so the workflow stays inspectable without implying real pricing.

## Running it

```bash
python scripts/run_api.py       # backend, http://127.0.0.1:8000
cd frontend && npm install && npm run dev   # frontend, http://localhost:5173
```
