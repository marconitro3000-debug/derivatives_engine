import type { ProductType } from "../lib/compileGraph";

const REQUIRED: Record<ProductType, { label: string; note?: string }[]> = {
  phoenix: [
    { label: "Underlying" },
    { label: "Barrier" },
    { label: "Coupon" },
    { label: "Autocall" },
    { label: "Pricing Output", note: "el destino final — todo lo demás debe encadenarse hasta acá" },
  ],
  vanilla_option: [
    { label: "Underlying" },
    {
      label: "Option Contract u Option Strategy",
      note: "el destino final — Option Contract precia una sola pata, Option Strategy arma spreads/straddles/etc.",
    },
  ],
  portfolio: [
    { label: "Position", note: "una por cada nombre de la cartera — se pueden repetir, todas conectadas a Hedge Instrument" },
    { label: "Risk Objective", note: "qué riesgo estás dispuesto a tomar — drawdown máximo y horizonte" },
    {
      label: "Hedge Instrument",
      note: "el destino final — cómo se cierra la brecha: strike del put y paths de Monte Carlo",
    },
  ],
};

const PRODUCT_LABELS: Record<ProductType, string> = {
  phoenix: "Autocall",
  vanilla_option: "Vanilla Option",
  portfolio: "Portfolio Hedge",
};

export default function HelpGuide({ productType, onClose }: { productType: ProductType; onClose: () => void }) {
  const required = REQUIRED[productType];
  const productLabel = PRODUCT_LABELS[productType];

  return (
    <div className="help-overlay" onClick={onClose}>
      <div className="help-panel" onClick={(e) => e.stopPropagation()}>
        <div className="help-header">
          <h2>Cómo funciona el Workbench</h2>
          <button className="help-close" onClick={onClose}>×</button>
        </div>

        <div className="help-body">
          <section>
            <h3>La idea</h3>
            <p>
              Armás un producto derivado conectando <b>tarjetas</b> en el canvas, como un diagrama de flujo.
              Cada tarjeta aporta un dato o parámetro (spot, tasa, vol, barrera, cupón...). Solo lo que está{" "}
              <b>conectado hasta la tarjeta final</b> (Pricing Output, Option Contract/Strategy, o Hedge Instrument)
              entra en el cálculo — una tarjeta suelta, sin cable, se ignora igual que si no existiera.
            </p>
            {productType === "portfolio" && (
              <p>
                <b>Portfolio Hedge</b> es distinto a los otros dos: no arma un producto único, sino que toma una
                lista de posiciones reales (ticker + notional), baja correlaciones y volatilidad reales de mercado
                para esos nombres, y busca automáticamente el tamaño mínimo de cobertura con puts que cumpla tu
                objetivo de drawdown máximo — no hace falta elegirlo a mano.
              </p>
            )}
          </section>

          <section>
            <h3>Pasos</h3>
            <ol>
              <li>Elegí el producto arriba: <b>Autocall</b>, <b>Vanilla Option</b> o <b>Portfolio Hedge</b>. Esto cambia qué tarjetas ves en la paleta de la izquierda — no se mezclan entre sí.</li>
              <li>Arrastrá tarjetas de la paleta al canvas.</li>
              <li>Conectá arrastrando desde el punto de la <b>derecha</b> de una tarjeta hacia el punto de la <b>izquierda</b> de la siguiente.</li>
              <li>Pulsá <b>Run ▶</b>. Si falta algo, el panel de Resultados te dice exactamente qué tarjeta falta o no está conectada.</li>
            </ol>
          </section>

          <section>
            <h3>Tarjetas obligatorias para {productLabel}</h3>
            <ul className="help-required">
              {required.map((r) => (
                <li key={r.label}>
                  <b>{r.label}</b>
                  {r.note && <span> — {r.note}</span>}
                </li>
              ))}
            </ul>
          </section>

          <section>
            <h3>Atajos</h3>
            <ul>
              <li>Doble clic en el título de una tarjeta → renombrarla.</li>
              <li>"+ nota" dentro de una tarjeta → agregar un comentario libre.</li>
              <li>El <b>×</b> en una tarjeta o en una conexión → la borra.</li>
              <li>Las pestañas de arriba son tus proyectos: <b>+</b> crea uno nuevo, <b>×</b> lo borra, doble clic lo renombra. Se guardan en este navegador.</li>
            </ul>
          </section>
        </div>
      </div>
    </div>
  );
}
