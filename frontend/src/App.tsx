import { useCallback, useEffect, useRef, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  addEdge,
  useEdgesState,
  useNodesState,
  type Connection,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import Palette from "./components/Palette";
import ResultsPanel from "./components/ResultsPanel";
import HelpGuide from "./components/HelpGuide";
import WorkflowNode from "./nodes/WorkflowNode";
import DeletableEdge from "./edges/DeletableEdge";
import { NODE_DEF_MAP } from "./types";
import type { AnyPricingResult } from "./types";
import {
  compileGraph,
  compileOptionStrategyGraph,
  compilePortfolioGraph,
  connectedComponent,
  type CompileResult,
  type LiveDataStore,
  type ProductType,
  type VanillaCompileResult,
} from "./lib/compileGraph";
import { mockPrice, mockPriceOption, mockPriceStrategy } from "./lib/mockPricing";
import {
  calibrateModel,
  checkHealth,
  getCalibration,
  getCalibrationStatus,
  getDividendYield,
  getQuote,
  getRatesCurve,
  getVolSurface,
  hedgePortfolio,
  priceOption,
  pricePhoenix,
  termSheetPhoenix,
} from "./lib/api";

const nodeTypes = { workflow: WorkflowNode };
const edgeTypes = { deletable: DeletableEdge };
const PROJECTS_STORAGE_KEY = "derivatives-workbench-projects";
const MODEL_COMPARISON_ORDER = ["black_scholes", "binomial", "monte_carlo", "merton", "heston", "svi", "local_vol"];

let nodeIdCounter = 1;
const nextId = () => `n${nodeIdCounter++}`;

const PHOENIX_KINDS = [
  "manual_trigger",
  "underlying",
  "rates_curve",
  "volatility_forecast",
  "barrier",
  "coupon",
  "memory",
  "autocall",
  "pricing_output",
];

const VANILLA_KINDS = [
  "manual_trigger",
  "underlying",
  "rates_curve",
  "volatility_forecast",
  "option_contract",
  "model_comparison",
];

const PORTFOLIO_KINDS = [
  "manual_trigger",
  "portfolio_position",
  "portfolio_position",
  "portfolio_position",
  "risk_objective",
  "hedge_instrument",
];
const PORTFOLIO_DEMO_TICKERS = ["AAPL", "MSFT", "GOOGL"];

function makeNode(kind: string, position: { x: number; y: number }): Node {
  const def = NODE_DEF_MAP[kind];
  return {
    id: nextId(),
    type: "workflow",
    position,
    data: { kind, params: { ...def.defaults }, live: undefined },
  };
}

function initialNodesFor(product: ProductType): Node[] {
  const kinds = product === "phoenix" ? PHOENIX_KINDS : product === "vanilla_option" ? VANILLA_KINDS : PORTFOLIO_KINDS;
  const nodes = kinds.map((kind, i) => makeNode(kind, { x: 40 + (i % 3) * 260, y: 40 + Math.floor(i / 3) * 180 }));
  if (product === "portfolio") {
    let posIdx = 0;
    nodes.forEach((n) => {
      if (n.data.kind === "portfolio_position") {
        (n.data.params as any).ticker = PORTFOLIO_DEMO_TICKERS[posIdx % PORTFOLIO_DEMO_TICKERS.length];
        posIdx += 1;
      }
    });
  }
  return nodes;
}

function edgesFor(product: ProductType, nodes: Node[]): Edge[] {
  const byKind = (k: string) => nodes.find((n) => n.data.kind === k)?.id;

  if (product === "portfolio") {
    const triggerId = byKind("manual_trigger");
    const riskObjectiveId = byKind("risk_objective");
    const hedgeInstrumentId = byKind("hedge_instrument");
    const edges: Edge[] = [];
    nodes
      .filter((n) => n.data.kind === "portfolio_position")
      .forEach((n) => {
        if (triggerId) edges.push({ id: `${triggerId}-${n.id}`, source: triggerId, target: n.id, type: "deletable" });
        if (hedgeInstrumentId)
          edges.push({ id: `${n.id}-${hedgeInstrumentId}`, source: n.id, target: hedgeInstrumentId, type: "deletable" });
      });
    if (riskObjectiveId && hedgeInstrumentId) {
      edges.push({
        id: `${riskObjectiveId}-${hedgeInstrumentId}`,
        source: riskObjectiveId,
        target: hedgeInstrumentId,
        type: "deletable",
      });
    }
    return edges;
  }

  const pairs: [string, string][] =
    product === "phoenix"
      ? [
          ["manual_trigger", "underlying"],
          ["underlying", "volatility_forecast"],
          ["underlying", "barrier"],
          ["rates_curve", "pricing_output"],
          ["volatility_forecast", "pricing_output"],
          ["barrier", "coupon"],
          ["coupon", "memory"],
          ["memory", "autocall"],
          ["autocall", "pricing_output"],
        ]
      : [
          ["manual_trigger", "underlying"],
          ["underlying", "volatility_forecast"],
          ["underlying", "option_contract"],
          ["rates_curve", "option_contract"],
          ["volatility_forecast", "option_contract"],
          ["option_contract", "model_comparison"],
        ];
  return pairs
    .map(([a, b]) => {
      const source = byKind(a);
      const target = byKind(b);
      if (!source || !target) return null;
      return { id: `${source}-${target}`, source, target, type: "deletable" } as Edge;
    })
    .filter((e): e is Edge => e !== null);
}

type SavedProject = {
  id: string;
  name: string;
  productType: ProductType;
  nodes: Node[];
  edges: Edge[];
  updatedAt: string;
};

function projectId() {
  return `p${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`;
}

function cleanNodes(nodes: Node[]): Node[] {
  return nodes.map((node) => ({
    ...node,
    selected: false,
    dragging: false,
    data: {
      kind: node.data.kind,
      params: { ...((node.data.params as Record<string, any>) ?? {}) },
      ...(node.data.label ? { label: node.data.label as string } : {}),
      ...(node.data.note ? { note: node.data.note as string } : {}),
    },
  }));
}

function cleanEdges(edges: Edge[]): Edge[] {
  return edges.map((edge) => ({ ...edge, selected: false }));
}

function createSavedProject(name: string, productType: ProductType): SavedProject {
  const nodes = initialNodesFor(productType);
  return {
    id: projectId(),
    name,
    productType,
    nodes,
    edges: edgesFor(productType, nodes),
    updatedAt: new Date().toISOString(),
  };
}

function syncNodeCounter(nodes: Node[]) {
  const maxId = nodes.reduce((max, node) => {
    const match = /^n(\d+)$/.exec(node.id);
    return match ? Math.max(max, Number(match[1])) : max;
  }, 0);
  nodeIdCounter = Math.max(nodeIdCounter, maxId + 1);
}

function loadProjects(): SavedProject[] {
  try {
    const raw = window.localStorage.getItem(PROJECTS_STORAGE_KEY);
    if (!raw) return [createSavedProject("Proyecto 1", "phoenix")];
    const parsed = JSON.parse(raw) as SavedProject[];
    if (!Array.isArray(parsed) || parsed.length === 0) return [createSavedProject("Proyecto 1", "phoenix")];
    parsed.forEach((project) => syncNodeCounter(project.nodes));
    return parsed;
  } catch {
    return [createSavedProject("Proyecto 1", "phoenix")];
  }
}

export default function App() {
  const initialProjects = useRef(loadProjects()).current;
  const initialProject = initialProjects[0];
  const [projects, setProjects] = useState<SavedProject[]>(initialProjects);
  const [activeProjectId, setActiveProjectId] = useState(initialProject.id);
  const [projectName, setProjectName] = useState(initialProject.name);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const [productType, setProductType] = useState<ProductType>(initialProject.productType);
  const startNodes = useRef(initialProject.nodes).current;
  const [nodes, setNodes, onNodesChange] = useNodesState(startNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialProject.edges);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [result, setResult] = useState<AnyPricingResult | null>(null);
  const [volSurface, setVolSurface] = useState<any | null>(null);
  const [missing, setMissing] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [calibrateError, setCalibrateError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [editingTabId, setEditingTabId] = useState<string | null>(null);
  const [showHelp, setShowHelp] = useState(false);
  const liveRef = useRef<LiveDataStore>({});

  useEffect(() => {
    checkHealth().then(setApiOnline);
    const t = setInterval(() => checkHealth().then(setApiOnline), 15000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    window.localStorage.setItem(PROJECTS_STORAGE_KEY, JSON.stringify(projects));
  }, [projects]);

  const resetTransientState = useCallback(() => {
    liveRef.current = {};
    setResult(null);
    setMissing([]);
    setVolSurface(null);
    setCalibrateError(null);
    setRunError(null);
  }, []);

  // fold current canvas state back into the active project entry, without touching localStorage save-status UI
  const commitActiveProject = useCallback(
    (current: SavedProject[]): SavedProject[] =>
      current.map((project) =>
        project.id === activeProjectId
          ? {
              ...project,
              name: projectName.trim() || project.name,
              productType,
              nodes: cleanNodes(nodes),
              edges: cleanEdges(edges),
              updatedAt: new Date().toISOString(),
            }
          : project
      ),
    [activeProjectId, projectName, productType, nodes, edges]
  );

  const openProject = useCallback(
    (project: SavedProject) => {
      syncNodeCounter(project.nodes);
      setActiveProjectId(project.id);
      setProjectName(project.name);
      setProductType(project.productType);
      setNodes(project.nodes);
      setEdges(project.edges);
      resetTransientState();
      setSaveStatus(null);
    },
    [resetTransientState, setNodes, setEdges]
  );

  const loadProject = useCallback(
    (projectId: string) => {
      if (projectId === activeProjectId) return;
      const updated = commitActiveProject(projects);
      setProjects(updated);
      const project = updated.find((p) => p.id === projectId);
      if (project) openProject(project);
    },
    [projects, activeProjectId, commitActiveProject, openProject]
  );

  const saveCurrentProject = useCallback(() => {
    const updated = commitActiveProject(projects);
    setProjects(updated);
    const savedProject = updated.find((p) => p.id === activeProjectId);
    if (savedProject) setProjectName(savedProject.name);
    setSaveStatus(`Guardado ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`);
  }, [projects, activeProjectId, commitActiveProject]);

  const createNewProject = useCallback(() => {
    const updated = commitActiveProject(projects);
    const project = createSavedProject(`Proyecto ${updated.length + 1}`, productType);
    setProjects(updated.concat(project));
    openProject(project);
    setEditingTabId(project.id);
  }, [projects, productType, commitActiveProject, openProject]);

  const renameProject = useCallback(
    (projectId: string, name: string) => {
      const cleanName = name.trim() || "Sin titulo";
      if (projectId === activeProjectId) setProjectName(cleanName);
      setProjects((current) =>
        current.map((project) => (project.id === projectId ? { ...project, name: cleanName } : project))
      );
    },
    [activeProjectId]
  );

  const deleteProject = useCallback(
    (projectId: string) => {
      if (projects.length <= 1) {
        window.alert("No puedes borrar el unico proyecto.");
        return;
      }
      const target = projects.find((p) => p.id === projectId);
      if (!window.confirm(`Borrar el proyecto "${target?.name ?? ""}"? Esta accion no se puede deshacer.`)) return;
      if (projectId === activeProjectId) {
        const remaining = projects.filter((p) => p.id !== projectId);
        setProjects(remaining);
        openProject(remaining[0]);
      } else {
        setProjects((current) => current.filter((p) => p.id !== projectId));
      }
    },
    [projects, activeProjectId, openProject]
  );

  const switchProduct = useCallback(
    (next: ProductType) => {
      if (next === productType) return;
      const freshNodes = initialNodesFor(next);
      setProductType(next);
      setNodes(freshNodes);
      setEdges(edgesFor(next, freshNodes));
      resetTransientState();
      setSaveStatus(null);
    },
    [productType, resetTransientState, setNodes, setEdges]
  );

  const onParamChange = useCallback(
    (nodeId: string, key: string, value: any) => {
      setNodes((nds) =>
        nds.map((n) =>
          n.id === nodeId ? { ...n, data: { ...n.data, params: { ...(n.data.params as any), [key]: value } } } : n
        )
      );
    },
    [setNodes]
  );

  const onDeleteNode = useCallback(
    (nodeId: string) => {
      setNodes((nds) => nds.filter((n) => n.id !== nodeId));
      setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
    },
    [setNodes, setEdges]
  );

  const onDeleteEdge = useCallback(
    (edgeId: string) => setEdges((eds) => eds.filter((e) => e.id !== edgeId)),
    [setEdges]
  );

  const onRenameNode = useCallback(
    (nodeId: string, label: string) => {
      const cleanLabel = label.trim();
      setNodes((nds) =>
        nds.map((n) => (n.id === nodeId ? { ...n, data: { ...n.data, label: cleanLabel || undefined } } : n))
      );
    },
    [setNodes]
  );

  const onNodeNoteChange = useCallback(
    (nodeId: string, note: string) => {
      setNodes((nds) => nds.map((n) => (n.id === nodeId ? { ...n, data: { ...n.data, note } } : n)));
    },
    [setNodes]
  );

  // inject the onParamChange/onDelete/onRename/onNoteChange callbacks + keep node.data.live in sync for display
  const nodesWithHandlers = nodes.map((n) => ({
    ...n,
    data: {
      ...n.data,
      onParamChange,
      onDelete: onDeleteNode,
      onRename: onRenameNode,
      onNoteChange: onNodeNoteChange,
      live: liveRef.current[n.id],
    },
  }));

  const edgesWithHandlers = edges.map((e) => ({
    ...e,
    type: e.type ?? "deletable",
    data: { ...e.data, onDelete: onDeleteEdge },
  }));

  const onConnect = useCallback(
    (c: Connection) => setEdges((eds) => addEdge({ ...c, type: "deletable" }, eds)),
    [setEdges]
  );

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const kind = event.dataTransfer.getData("application/workflow-node");
      if (!kind || !NODE_DEF_MAP[kind]) return;
      const bounds = (event.target as HTMLElement).closest(".react-flow")?.getBoundingClientRect();
      const position = {
        x: event.clientX - (bounds?.left ?? 0) - 40,
        y: event.clientY - (bounds?.top ?? 0) - 20,
      };
      setNodes((nds) => nds.concat(makeNode(kind, position)));
    },
    [setNodes]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const optionContract = nodes.find((n) => n.data.kind === "option_contract");
  const optionModel = (optionContract?.data.params as any)?.model;
  const canCalibrate =
    productType === "vanilla_option" && (optionModel === "heston" || optionModel === "svi" || optionModel === "local_vol");

  const totalPortfolioNotional = nodes
    .filter((n) => n.data.kind === "portfolio_position")
    .reduce((sum, n) => sum + (Number((n.data.params as any)?.notional) || 0), 0);

  const handleCalibrate = useCallback(async () => {
    const underlying = nodes.find((n) => n.data.kind === "underlying");
    if (!underlying || !optionContract || !canCalibrate) return;
    const ticker = ((underlying.data.params as any)?.ticker ?? "AAPL") as string;
    setCalibrating(true);
    setCalibrateError(null);
    try {
      await calibrateModel(ticker, optionModel);
      const status = await getCalibrationStatus(ticker, optionModel);
      liveRef.current = { ...liveRef.current, [optionContract.id]: { calibration: status } };
      setNodes((nds) => nds.map((n) => ({ ...n })));
    } catch (e) {
      setCalibrateError(e instanceof Error ? e.message : String(e));
    } finally {
      setCalibrating(false);
    }
  }, [nodes, optionContract, optionModel, canCalibrate, setNodes]);

  const handleRun = useCallback(async () => {
    setRunning(true);
    setRunError(null);
    try {
      if (productType === "portfolio") {
        const online = await checkHealth();
        setApiOnline(online);
        const compiled = compilePortfolioGraph(nodes, edges);
        setMissing(compiled.missing);
        if (!compiled.spec) {
          setResult(null);
          return;
        }
        if (!online) {
          setResult(null);
          setRunError(
            "El optimizador de cartera necesita la API online: usa correlaciones y vol reales de mercado, no tiene modo offline."
          );
          return;
        }
        try {
          const hedged = await hedgePortfolio(compiled.spec);
          setResult({ ...hedged, kind: "portfolio" });
        } catch (e) {
          setResult(null);
          setRunError(e instanceof Error ? e.message : String(e));
        }
        return;
      }

      const underlying = nodes.find((n) => n.data.kind === "underlying");
      const ratesCurve = nodes.find((n) => n.data.kind === "rates_curve");
      const volForecast = nodes.find((n) => n.data.kind === "volatility_forecast");
      const impliedSurfaceNode = nodes.find((n) => n.data.kind === "implied_vol_surface");
      const modelComparison = nodes.find((n) => n.data.kind === "model_comparison");
      const optionContractNode = nodes.find((n) => n.data.kind === "option_contract");
      const optionStrategyNode = nodes.find((n) => n.data.kind === "option_strategy");
      const pricingOutputNode = nodes.find((n) => n.data.kind === "pricing_output");
      const stressTestNode = nodes.find((n) => n.data.kind === "stress_test");
      const termSheetNode = nodes.find((n) => n.data.kind === "term_sheet");

      // same "disconnected = absent" rule as the compilers, applied to the
      // optional post-pricing add-ons (vol smile, stress table, term sheet)
      // that live outside the request spec itself
      const graphRoot =
        productType === "phoenix" ? pricingOutputNode : optionStrategyNode ?? optionContractNode;
      const graphConnected = graphRoot ? connectedComponent(edges, graphRoot.id) : new Set<string>();
      const isConnected = (n: Node | undefined) => !!n && !!graphRoot && graphConnected.has(n.id);
      const impliedSurface = isConnected(impliedSurfaceNode) ? impliedSurfaceNode : undefined;
      const hasStressTest = isConnected(stressTestNode);
      const hasTermSheet = isConnected(termSheetNode);

      const ticker = ((underlying?.data.params as any)?.ticker ?? "AAPL") as string;
      const currency = ((ratesCurve?.data.params as any)?.currency ?? "USD") as string;
      const maturityForRate =
        productType === "vanilla_option"
          ? (optionContractNode?.data.params as any)?.maturity_years ?? (optionStrategyNode?.data.params as any)?.maturity_years
          : (pricingOutputNode?.data.params as any)?.maturity_years;

      // each source is fetched independently and patched in as soon as it resolves,
      // so a card lights up the moment its own live data arrives instead of the
      // whole canvas waiting on the slowest fetch
      const patchLive = (nodeId: string | undefined, patch: Record<string, any>) => {
        if (!nodeId) return;
        liveRef.current = { ...liveRef.current, [nodeId]: { ...liveRef.current[nodeId], ...patch } };
        setNodes((nds) => nds.map((n) => ({ ...n })));
      };

      const online = await checkHealth();
      setApiOnline(online);

      if (online) {
        const fetches: Promise<any>[] = [
          getQuote(ticker)
            .then((quote) => patchLive(underlying?.id, { quote: { price: quote.price, currency: quote.currency } }))
            .catch(() => null),
        ];
        if (ratesCurve) {
          fetches.push(
            getRatesCurve(currency, maturityForRate)
              .then((rates) =>
                patchLive(ratesCurve.id, {
                  rates: { tenors: rates.tenors, source: rates.source, interpolated_rate: rates.interpolated_rate },
                })
              )
              .catch(() => null)
          );
        }
        if (volForecast) {
          fetches.push(
            getCalibration(ticker)
              .then((calib) =>
                patchLive(volForecast.id, {
                  volatility: { realized_vol_63d: calib.realized_vol_63d, realized_vol_21d: calib.realized_vol_21d },
                })
              )
              .catch(() => null)
          );
        }
        if (productType === "vanilla_option") {
          fetches.push(
            getDividendYield(ticker)
              .then((dividend) => patchLive(underlying?.id, { dividend: { dividend_yield: dividend.dividend_yield } }))
              .catch(() => null)
          );
        }
        if (impliedSurface) {
          fetches.push(
            getVolSurface(ticker)
              .then((surface) => {
                setVolSurface(surface);
                patchLive(impliedSurface.id, { surface });
              })
              .catch(() => null)
          );
        }
        await Promise.all(fetches);
      }

      if (productType === "vanilla_option" && optionStrategyNode) {
        const compiled = compileOptionStrategyGraph(nodes, edges, liveRef.current, { requireLiveSpot: online });
        setMissing(compiled.missing);
        if (!compiled.legs) {
          setResult(null);
          return;
        }
        if (online) {
          try {
            const legResults = await Promise.all(
              compiled.legs.map(async ({ leg, spec }) => {
                const priced = await priceOption(spec);
                return { option_type: leg.option_type, qty: leg.qty, strike: spec.strike, price: priced.price, greeks: priced.greeks };
              })
            );
            const netPrice = legResults.reduce((sum, l) => sum + l.qty * l.price, 0);
            const netGreeks = legResults.reduce(
              (acc: any, l) => ({
                delta: (acc.delta ?? 0) + l.qty * (l.greeks.delta ?? 0),
                gamma: (acc.gamma ?? 0) + l.qty * (l.greeks.gamma ?? 0),
                vega: (acc.vega ?? 0) + l.qty * (l.greeks.vega ?? 0),
              }),
              {}
            );
            setResult({
              kind: "vanilla_option", mode: "strategy", strategy: compiled.strategy,
              underlying: compiled.legs[0].spec.underlying, spot: compiled.legs[0].spec.spot,
              net_price: netPrice, net_greeks: netGreeks, legs: legResults,
            });
          } catch (e) {
            setResult(null);
            setRunError(e instanceof Error ? e.message : String(e));
          }
        } else {
          setResult(mockPriceStrategy(compiled.strategy, compiled.legs));
        }
        return;
      }

      if (productType === "vanilla_option") {
        const compiled = compileGraph(nodes, edges, liveRef.current, {
          requireLiveSpot: online,
          productType: "vanilla_option",
        }) as VanillaCompileResult;
        setMissing(compiled.missing);
        if (!compiled.spec) {
          setResult(null);
          return;
        }
        if (online) {
          if (modelComparison) {
            // price every model in parallel and drop each row in as it resolves,
            // instead of waiting on one blocking /compare call for all 7 models
            setResult({
              kind: "vanilla_option", mode: "compare",
              underlying: compiled.spec.underlying, option_type: compiled.spec.option_type,
              strike: compiled.spec.strike, results: [],
            });
            await Promise.all(
              MODEL_COMPARISON_ORDER.map(async (model) => {
                let row: any;
                try {
                  row = { model, ...(await priceOption({ ...compiled.spec, model })) };
                } catch (e) {
                  row = { model, error: e instanceof Error ? e.message : String(e) };
                }
                setResult((prev) =>
                  prev && prev.kind === "vanilla_option" && prev.mode === "compare"
                    ? {
                        ...prev,
                        results: [...prev.results.filter((r) => r.model !== model), row].sort(
                          (a, b) => MODEL_COMPARISON_ORDER.indexOf(a.model) - MODEL_COMPARISON_ORDER.indexOf(b.model)
                        ),
                      }
                    : prev
                );
              })
            );
          } else {
            try {
              const priced = await priceOption(compiled.spec);
              setResult({
                kind: "vanilla_option", mode: "single",
                strike: compiled.spec.strike, spot: compiled.spec.spot, option_type: compiled.spec.option_type,
                ...priced,
              });
            } catch (e) {
              setResult(null);
              setRunError(e instanceof Error ? e.message : String(e));
            }
          }
        } else {
          setResult(mockPriceOption(compiled.spec));
        }
        return;
      }

      const compiled = compileGraph(nodes, edges, liveRef.current, { requireLiveSpot: online }) as CompileResult;
      setMissing(compiled.missing);
      if (!compiled.spec) {
        setResult(null);
        return;
      }

      if (online) {
        try {
          const priced = await pricePhoenix(compiled.spec);
          const termSheet = hasTermSheet ? await termSheetPhoenix(compiled.spec).catch(() => null) : null;
          setResult({ ...priced, kind: "phoenix", stressVisible: hasStressTest, termSheet });
        } catch (e) {
          setResult(null);
          setRunError(e instanceof Error ? e.message : String(e));
        }
      } else {
        setResult({ ...mockPrice(compiled.spec), stressVisible: hasStressTest });
      }
    } finally {
      setRunning(false);
    }
  }, [nodes, edges, productType, setNodes]);

  return (
    <div className="app">
      <div className="topbar">
        <b>Derivatives Workbench</b>
        <div className="project-tabs">
          {projects.map((project) => {
            const isActive = project.id === activeProjectId;
            const isEditing = editingTabId === project.id;
            return (
              <div
                key={project.id}
                className={`project-tab ${isActive ? "active" : ""}`}
                onClick={() => loadProject(project.id)}
                onDoubleClick={() => setEditingTabId(project.id)}
                title="Doble clic para renombrar"
              >
                {isEditing ? (
                  <input
                    autoFocus
                    className="tab-rename-input"
                    defaultValue={project.name}
                    onClick={(event) => event.stopPropagation()}
                    onBlur={(event) => {
                      renameProject(project.id, event.target.value);
                      setEditingTabId(null);
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") (event.target as HTMLInputElement).blur();
                      if (event.key === "Escape") setEditingTabId(null);
                    }}
                  />
                ) : (
                  <span className="tab-name">{project.name}</span>
                )}
                <button
                  className="tab-close"
                  title="Borrar proyecto"
                  onClick={(event) => {
                    event.stopPropagation();
                    deleteProject(project.id);
                  }}
                >
                  ×
                </button>
              </div>
            );
          })}
          <button className="tab-add" onClick={createNewProject} title="Nuevo proyecto">
            +
          </button>
          {saveStatus && <span className="save-status">{saveStatus}</span>}
          <button className="save-btn" onClick={saveCurrentProject}>
            Guardar
          </button>
        </div>
        <div className="product-toggle">
          <button
            className={productType === "phoenix" ? "active" : ""}
            onClick={() => switchProduct("phoenix")}
          >
            Autocall
          </button>
          <button
            className={productType === "vanilla_option" ? "active" : ""}
            onClick={() => switchProduct("vanilla_option")}
          >
            Vanilla Option
          </button>
          <button
            className={productType === "portfolio" ? "active" : ""}
            onClick={() => switchProduct("portfolio")}
          >
            Portfolio Hedge
          </button>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <button className="help-btn" onClick={() => setShowHelp(true)} title="Cómo funciona">
            ? Guía
          </button>
          <span className={`status-pill ${apiOnline ? "ok" : "offline"}`}>
            {apiOnline === null ? "checking..." : apiOnline ? "API online" : "API offline (demo mode)"}
          </span>
          {canCalibrate && (
            <button className="calibrate-btn" onClick={handleCalibrate} disabled={calibrating}>
              {calibrating ? "Calibrating on live data... (30-100s)" : `Calibrate ${optionModel}`}
            </button>
          )}
          <button className="run-btn" onClick={handleRun} disabled={running}>
            {running ? "Running..." : "Run ▶"}
          </button>
        </div>
      </div>
      {showHelp && <HelpGuide productType={productType} onClose={() => setShowHelp(false)} />}
      <Palette productType={productType} />
      <div style={{ height: "100%" }} onDrop={onDrop} onDragOver={onDragOver}>
        {calibrateError && <div className="calibrate-error">{calibrateError}</div>}
        {productType === "portfolio" && (
          <div className="total-notional-badge">Total portfolio: {totalPortfolioNotional.toLocaleString()}</div>
        )}
        <ReactFlow
          nodes={nodesWithHandlers}
          edges={edgesWithHandlers}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
        >
          <Background gap={20} />
          <Controls />
        </ReactFlow>
      </div>
      <ResultsPanel result={result} volSurface={volSurface} missing={missing} runError={runError} />
    </div>
  );
}
