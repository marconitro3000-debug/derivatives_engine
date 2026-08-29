import { NODE_DEFS, type ProductTag } from "../types";

const PRODUCT_LABELS: Record<ProductTag, string> = {
  phoenix: "Autocall",
  vanilla_option: "Vanilla Option",
  portfolio: "Portfolio Hedge",
};

export default function Palette({ productType }: { productType: ProductTag }) {
  const onDragStart = (event: React.DragEvent, kind: string) => {
    event.dataTransfer.setData("application/workflow-node", kind);
    event.dataTransfer.effectAllowed = "move";
  };

  const groups: Record<string, typeof NODE_DEFS> = {
    trigger: [],
    input: [],
    param: [],
    compute: [],
    output: [],
  };
  NODE_DEFS.filter((d) => d.products.includes(productType)).forEach((d) => groups[d.category].push(d));

  const labels: Record<string, string> = {
    trigger: "Trigger",
    input: "Market data",
    param: "Payoff blocks",
    compute: "Compute",
    output: "Outputs",
  };

  return (
    <div className="panel">
      <h2>Node palette</h2>
      <div className="palette-hint">
        Showing cards for <b>{PRODUCT_LABELS[productType]}</b> — switch product above to see the other sets.
      </div>
      {Object.entries(groups).map(([cat, defs]) =>
        defs.length === 0 ? null : (
          <div key={cat}>
            <h2 style={{ marginTop: 14 }}>{labels[cat]}</h2>
            {defs.map((d) => (
              <div
                key={d.kind}
                className="palette-item"
                draggable
                onDragStart={(e) => onDragStart(e, d.kind)}
              >
                <div className="palette-dot" style={{ background: d.color }} />
                <div>
                  <b>{d.label}</b>
                  <span>{d.description}</span>
                </div>
              </div>
            ))}
          </div>
        )
      )}
    </div>
  );
}
