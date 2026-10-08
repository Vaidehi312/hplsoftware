import { useState } from "react";
import { api } from "../../api.js";
import { colorForHpc, rgbToCss } from "../../colors.js";
import HpcAnnotation from "./HpcAnnotation.jsx";
import { tileDetailSections } from "./tileDetails.js";

// The selected tile, shown beside the slide rather than under it: its pixels,
// its cluster and everything the KB holds about it, in one card that sits
// where the pointer already is. It replaces the table-below-the-viewer, which
// meant clicking a tile and then scrolling past 780 px of slide to read it.
//
// Pixels come from the raw slide (/region at level 0), not the packaged .h5 —
// the same source the click inspector always used, and the one that exists
// for every registered tile whether or not its .h5 is still on disk.
export default function TileDetailCard({ slideId, tile, tileSizeNative, heatHpc = null, hpcTitle = "", onClose = null, onZoom = null }) {
  if (!tile) return null;

  const hpc = tile.hpc_id === null || tile.hpc_id === undefined || Number.isNaN(Number(tile.hpc_id)) ? null : Number(tile.hpc_id);
  const swatch = hpc === null ? "#6b7280" : rgbToCss(colorForHpc(hpc));
  const title = String(tile.hpc_title || hpcTitle || "").trim();
  const sections = tileDetailSections(tile, { heatHpc });
  const pitch = Number(tileSizeNative);

  return (
    <aside className="viewer-tile-card" style={{ borderTopColor: swatch }} aria-label="Selected tile">
      <header className="viewer-tile-card-head">
        <div className="viewer-tile-card-name" title={String(tile.slide_tile || "")}>
          {String(tile.tiles || tile.slide_tile || "")}
        </div>
        <div className="viewer-tile-card-actions">
          {onZoom && (
            <button type="button" className="viewer-btn" onClick={onZoom} title="Zoom the viewer to this tile">
              Zoom to
            </button>
          )}
          {onClose && (
            <button type="button" className="viewer-btn viewer-tile-card-close" onClick={onClose} aria-label="Close tile details">
              ×
            </button>
          )}
        </div>
      </header>

      {/* Keyed on the tile, so picking another one starts its load state over. */}
      <TileImage key={`${slideId}|${tile.x_native}_${tile.y_native}`} slideId={slideId} tile={tile} pitch={pitch} />

      <div className="viewer-tile-card-hpc" style={{ borderLeftColor: swatch }}>
        <span className="viewer-tile-card-hpc-id">{hpc === null ? "No HPC label" : `HPC ${hpc}`}</span>
        {title && <span className="viewer-tile-card-hpc-title">{title}</span>}
      </div>

      {sections.map((section) => (
        <section key={section.title} className="viewer-tile-card-section">
          <h5>{section.title}</h5>
          <dl>
            {section.rows.map(([label, value, hint]) => (
              <div key={label} className="viewer-tile-card-row" title={hint || undefined}>
                <dt>{label}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
        </section>
      ))}

      {hpc !== null && (
        <details className="viewer-tile-card-section">
          <summary>HPC {hpc} interpretation</summary>
          <HpcAnnotation hpcId={hpc} />
        </details>
      )}
    </aside>
  );
}

function TileImage({ slideId, tile, pitch }) {
  const [failed, setFailed] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const usable = Number.isFinite(pitch) && pitch > 0;

  return (
    <div className="viewer-tile-card-image">
      {!failed && usable ? (
        <img
          src={api.regionUrl(slideId, tile.x_native, tile.y_native, pitch, pitch, 0, 90)}
          alt={`${tile.slide_tile} at native resolution`}
          onLoad={() => setLoaded(true)}
          onError={() => setFailed(true)}
          style={{ opacity: loaded ? 1 : 0.25 }}
        />
      ) : (
        <div className="viewer-tile-card-noimg">Tile image unavailable</div>
      )}
      {!loaded && !failed && usable && <div className="viewer-tile-card-loading">Loading tile…</div>}
    </div>
  );
}
