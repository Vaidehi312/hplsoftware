import { colorForHpc, colorForInflammation, colorForNecrosis, rgbToCss } from "../../colors.js";
import {
  categoryCounts,
  colorForMalignant,
  hpcCounts,
  inflammationLabelKey,
  legendCoveragePairsAdjacency,
  legendCoveragePairsHeatmap,
  malignantLabelKey,
  necrosisLabelKey,
} from "./overlayBuilders.js";

// Port of the mode-dependent legend column in show_wsi (app_v28.py
// ~4209-4396) plus build_legend_panel_html (~4894) for the two modes
// (Adjacency, Heatmap) that use a static swatch panel rather than clickable
// rows. HPC clusters / Inflammation / Necrosis / Malignant get clickable
// rows built inline in the Python, same here.
export default function Legend({
  highlightMode,
  tiles,
  selectedHpc,
  onSelectHpc,
  inflF,
  onSelectInfl,
  necF,
  onSelectNec,
  malF,
  onSelectMal,
  heatHpc,
  usedSurvivalHpcs,
  adjTileSets,
  adjHpcA,
  adjHpcB,
}) {
  if (highlightMode === "Survival Risk Heatmap") {
    return (
      <div className="viewer-legend-panel">
        {usedSurvivalHpcs && usedSurvivalHpcs.length > 0 && (
          <>
            <div className="viewer-legend-title">Legend</div>
            <div className="viewer-legend-caption">
              Red = higher risk-associated signal. Blue = protective-associated signal. Transparent = near neutral.
            </div>
            <SwatchRow items={[
              ["rgba(255,0,0,0.75)", "Risk-associated contribution"],
              ["rgba(0,80,255,0.75)", "Protective-associated contribution"],
              ["rgba(255,255,255,0.15)", "Near neutral"],
            ]} />
          </>
        )}
        <div className="viewer-legend-title" style={{ marginTop: 10 }}>
          Survival Risk Legend
        </div>
        <div className="viewer-legend-caption">Red = risk-associated. Blue = protective. Transparent = neutral.</div>
        {usedSurvivalHpcs && usedSurvivalHpcs.length > 0 ? (
          <div className="viewer-legend-caption">Using survival HPCs: [{usedSurvivalHpcs.join(", ")}]</div>
        ) : (
          <div className="viewer-warning">No survival-linked HPCs found.</div>
        )}
      </div>
    );
  }

  if (highlightMode === "HPC clusters") {
    const n = Math.max(tiles.length, 1);
    const rows = hpcCounts(tiles, 14);
    return (
      <div>
        <div className="viewer-legend-title" style={{ color: "#f3f4f6" }}>
          Legend
        </div>
        <div className="viewer-caption">Click an HPC to isolate it.</div>
        <button type="button" className="viewer-btn viewer-btn-full" onClick={() => onSelectHpc(null)}>
          Show all
        </button>
        {rows.map(({ hpcId, pct }) => (
          <div key={hpcId}>
            <div className="viewer-legend-swatch-bar" style={{ background: rgbToCss(colorForHpc(hpcId)) }} />
            <button
              type="button"
              className="viewer-legend-row-btn"
              style={selectedHpc === hpcId ? { fontWeight: 700, borderColor: "#0f172a" } : undefined}
              onClick={() => onSelectHpc(hpcId)}
            >
              HPC {hpcId} · {pct.toFixed(1)}%{selectedHpc === hpcId ? " ✓" : ""}
            </button>
          </div>
        ))}
        <div className="viewer-caption">{n} tiles total</div>
      </div>
    );
  }

  if (highlightMode === "Inflammation") {
    const rows = categoryCounts(tiles, (t) => inflammationLabelKey(t.inflammation));
    return (
      <div>
        <div className="viewer-legend-title" style={{ color: "#f3f4f6" }}>
          Legend
        </div>
        <div className="viewer-caption">Click a category to isolate it.</div>
        <button type="button" className="viewer-btn viewer-btn-full" onClick={() => onSelectInfl(null)}>
          Show all inflammation
        </button>
        {rows.map(({ key, pct }) => (
          <div key={key}>
            <div className="viewer-legend-swatch-bar" style={{ background: rgbToCss(colorForInflammation(key)) }} />
            <button
              type="button"
              className="viewer-legend-row-btn"
              style={inflF === key ? { fontWeight: 700, borderColor: "#0f172a" } : undefined}
              onClick={() => onSelectInfl(key)}
            >
              {titleCase(key)} · {pct.toFixed(1)}%{inflF === key ? " ✓" : ""}
            </button>
          </div>
        ))}
      </div>
    );
  }

  if (highlightMode === "Necrosis") {
    const rows = categoryCounts(tiles, (t) => necrosisLabelKey(t.necrosis));
    return (
      <div>
        <div className="viewer-legend-title" style={{ color: "#f3f4f6" }}>
          Legend
        </div>
        <div className="viewer-caption">Click a category to isolate it.</div>
        <button type="button" className="viewer-btn viewer-btn-full" onClick={() => onSelectNec(null)}>
          Show all necrosis
        </button>
        {rows.map(({ key, pct }) => (
          <div key={key}>
            <div className="viewer-legend-swatch-bar" style={{ background: rgbToCss(colorForNecrosis(key)) }} />
            <button
              type="button"
              className="viewer-legend-row-btn"
              style={necF === key ? { fontWeight: 700, borderColor: "#0f172a" } : undefined}
              onClick={() => onSelectNec(key)}
            >
              {titleCase(key)} · {pct.toFixed(1)}%{necF === key ? " ✓" : ""}
            </button>
          </div>
        ))}
      </div>
    );
  }

  if (highlightMode === "Malignant") {
    const rows = categoryCounts(tiles, (t) => malignantLabelKey(t.malignant));
    const sampleValueFor = (key) => (key === "malignant" ? true : key === "non-malignant" ? false : null);
    return (
      <div>
        <div className="viewer-legend-title" style={{ color: "#f3f4f6" }}>
          Legend
        </div>
        <div className="viewer-caption">Click a category to isolate it.</div>
        <button type="button" className="viewer-btn viewer-btn-full" onClick={() => onSelectMal(null)}>
          Show all malignant
        </button>
        {rows.map(({ key, pct }) => (
          <div key={key}>
            <div className="viewer-legend-swatch-bar" style={{ background: rgbToCss(colorForMalignant(sampleValueFor(key))) }} />
            <button
              type="button"
              className="viewer-legend-row-btn"
              style={malF === key ? { fontWeight: 700, borderColor: "#0f172a" } : undefined}
              onClick={() => onSelectMal(key)}
            >
              {titleCase(key.replace("-", " "))} · {pct.toFixed(1)}%{malF === key ? " ✓" : ""}
            </button>
          </div>
        ))}
      </div>
    );
  }

  if (highlightMode === "Adjacency") {
    const pairs = legendCoveragePairsAdjacency(tiles, adjTileSets, adjHpcA ?? "?", adjHpcB ?? "?");
    return (
      <div className="viewer-legend-panel">
        <div className="viewer-legend-title">Legend</div>
        <SwatchRow items={[
          ["#50C878", `HPC ${adjHpcA ?? "?"} touching HPC ${adjHpcB ?? "?"}`],
          ["#FFA500", `HPC ${adjHpcB ?? "?"} touching HPC ${adjHpcA ?? "?"}`],
        ]} />
        <CoverageBlock pairs={pairs} />
      </div>
    );
  }

  if (highlightMode === "Heatmap") {
    const pairs = legendCoveragePairsHeatmap(tiles, heatHpc);
    const hh = heatHpc !== null && heatHpc !== undefined ? Number(heatHpc) : 0;
    return (
      <div className="viewer-legend-panel">
        <div className="viewer-legend-title">Legend</div>
        <div className="viewer-legend-caption">
          Colormap for <b>P(HPC {hh})</b>: warm = higher probability; alpha = confidence.
        </div>
        <CoverageBlock pairs={pairs} />
      </div>
    );
  }

  return (
    <div className="viewer-legend-panel">
      <div className="viewer-legend-title">Legend</div>
      <div className="viewer-legend-caption">
        Tile outlines use a <b>stable color per HPC id</b>.
      </div>
    </div>
  );
}

function titleCase(s) {
  return String(s)
    .split(" ")
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
}

function SwatchRow({ items }) {
  return (
    <div>
      {items.map(([bg, label]) => (
        <div className="viewer-legend-swatch-row" key={label}>
          <span className="viewer-legend-swatch" style={{ background: bg }} />
          <span>{label}</span>
        </div>
      ))}
    </div>
  );
}

function CoverageBlock({ pairs }) {
  return (
    <>
      <hr className="viewer-legend-divider" />
      <div style={{ fontWeight: 600, color: "#0f172a", marginBottom: 4 }}>Tile coverage (% of tiles on this slide)</div>
      {!pairs || pairs.length === 0 ? (
        <div style={{ color: "#475569", fontSize: 11 }}>No coverage data.</div>
      ) : (
        pairs.map(([label, pct]) => (
          <div className="viewer-legend-coverage-row" key={label}>
            <span>{label}</span>
            <span className="viewer-legend-coverage-pct">{pct.toFixed(1)}%</span>
          </div>
        ))
      )}
    </>
  );
}
