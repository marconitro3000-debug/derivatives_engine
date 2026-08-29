import { useState } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { NODE_DEF_MAP } from "../types";

export interface WorkflowNodeData {
  kind: string;
  params: Record<string, any>;
  live?: Record<string, any>;
  label?: string;
  note?: string;
  onParamChange: (nodeId: string, key: string, value: any) => void;
  onDelete: (nodeId: string) => void;
  onRename: (nodeId: string, label: string) => void;
  onNoteChange: (nodeId: string, note: string) => void;
}

function LiveSummary({ kind, live }: { kind: string; live?: Record<string, any> }) {
  if (!live) return <div className="wf-live">no live data yet</div>;
  if (kind === "underlying" && live.quote) {
    return (
      <div className="wf-live">
        spot: <b>{live.quote.price?.toFixed(2)}</b> {live.quote.currency ?? ""}
        {live.dividend && <> · div yield: <b>{(live.dividend.dividend_yield * 100).toFixed(2)}%</b></>}
      </div>
    );
  }
  if (kind === "rates_curve" && live.rates) {
    const rate = live.rates.interpolated_rate ?? live.rates.tenors?.["1Y"];
    const label = live.rates.interpolated_rate !== undefined ? "interp." : "1Y";
    return (
      <div className="wf-live">
        {label}: <b>{(rate * 100).toFixed(2)}%</b> ({live.rates.source})
      </div>
    );
  }
  if (kind === "volatility_forecast" && live.volatility) {
    const v = live.volatility.realized_vol_63d ?? live.volatility.realized_vol_21d;
    return <div className="wf-live">realized 63d: <b>{v ? (v * 100).toFixed(1) + "%" : "n/a"}</b></div>;
  }
  if (kind === "implied_vol_surface" && live.surface) {
    return (
      <div className="wf-live">
        {live.surface.slices?.length ?? 0} expiries, arb-free:{" "}
        <b>{live.surface.is_arbitrage_free ? "yes" : "no"}</b>
      </div>
    );
  }
  if (kind === "option_contract" && live.calibration) {
    const c = live.calibration;
    return c.calibrated ? (
      <div className="wf-live">
        calibrated, rmse: <b>{c.rmse?.toExponential(1)}</b>
      </div>
    ) : (
      <div className="wf-live">not calibrated yet</div>
    );
  }
  return null;
}

export default function WorkflowNode({ id, data }: NodeProps) {
  const nodeData = data as unknown as WorkflowNodeData;
  const def = NODE_DEF_MAP[nodeData.kind];
  const [editingLabel, setEditingLabel] = useState(false);
  const [showNote, setShowNote] = useState(!!nodeData.note);
  if (!def) return null;
  const displayLabel = nodeData.label || def.label;

  return (
    <div className="wf-node">
      <Handle type="target" position={Position.Left} />
      <div className="wf-node-head">
        <div className="wf-node-icon" style={{ background: def.color }} />
        <div className="wf-node-title">
          {editingLabel ? (
            <input
              autoFocus
              className="wf-node-rename"
              defaultValue={displayLabel}
              onClick={(e) => e.stopPropagation()}
              onBlur={(e) => {
                nodeData.onRename(id, e.target.value);
                setEditingLabel(false);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                if (e.key === "Escape") setEditingLabel(false);
              }}
            />
          ) : (
            <b onDoubleClick={() => setEditingLabel(true)} title="Doble clic para renombrar">
              {displayLabel}
            </b>
          )}
          <span>{def.description}</span>
        </div>
        <button
          className="wf-node-delete"
          title="Borrar tarjeta"
          onClick={(e) => {
            e.stopPropagation();
            nodeData.onDelete(id);
          }}
        >
          ×
        </button>
      </div>
      <div className="wf-node-body">
        {def.fields
          .filter((f) => !f.showIf || f.showIf(nodeData.params))
          .map((f) => (
            <div className="wf-field" key={f.key}>
              <label>{f.label}</label>
              {f.type === "select" ? (
                <select
                  value={nodeData.params[f.key]}
                  onChange={(e) => nodeData.onParamChange(id, f.key, e.target.value)}
                >
                  {f.options?.map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </select>
              ) : f.type === "boolean" ? (
                <input
                  type="checkbox"
                  checked={!!nodeData.params[f.key]}
                  onChange={(e) => nodeData.onParamChange(id, f.key, e.target.checked)}
                />
              ) : (
                <input
                  type={f.type === "number" ? "number" : "text"}
                  step={f.step ?? 1}
                  value={nodeData.params[f.key] ?? ""}
                  onChange={(e) =>
                    nodeData.onParamChange(
                      id,
                      f.key,
                      f.type === "number" ? parseFloat(e.target.value) : e.target.value
                    )
                  }
                />
              )}
            </div>
          ))}
        <LiveSummary kind={nodeData.kind} live={nodeData.live} />
        {showNote ? (
          <textarea
            className="wf-node-note"
            placeholder="Nota libre..."
            rows={2}
            value={nodeData.note ?? ""}
            onChange={(e) => nodeData.onNoteChange(id, e.target.value)}
          />
        ) : (
          <button className="wf-node-note-toggle" onClick={() => setShowNote(true)}>
            + nota
          </button>
        )}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
