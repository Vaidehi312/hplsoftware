import { useMemo, useState } from "react";
import { colorForHpc, riskToRgba, rgbaToCss, rgbToCss } from "../../colors.js";
import { buildTileLookup, computeMinMax, gridColorForTile, heatmapColor, passesGridFilters, tileAtImagePoint } from "./overlayBuilders.js";
import TileDetailCard from "./TileDetailCard.jsx";

// Port of show_wsi's "Click tile inspector" branch (app_v28.py ~4434-4506)
// plus the raster-drawing helpers _draw_heatmap / _draw_survival_risk_heatmap
// / _draw_grid (~4513-4617) and fetch_tile_region_from_svs (~3100). The
// original draws onto a Pillow image with ImageDraw; here the thumbnail is a
// plain <img> and the same coloring math produces SVG rects layered on top
// via an absolutely-positioned <svg> sized to the image's natural pixels, so
// they scale together regardless of on-screen zoom.
export default function ClickInspectorViewer({
  slideId,
  thumbnailUrl,
  tiles,
  gridTiles,
  downsampleHint,
  tileSizeNative,
  showGrid,
  highlightMode,
  inflF,
  necF,
  malF,
  heatHpc,
  heatAlpha,
  adjTileSets,
  selectedTile,
  onSelectTile,
  hpcTitleMap,
}) {
  const [imgSize, setImgSize] = useState(null); // {width, height} = natural pixel size of the thumbnail actually served

  // downsampleHint is w0 / requested_thumb_width (matches Python's
  // `downsample = w0 / base_region.size[0]`) computed by the parent from
  // slide info + the thumbnail request. If the server returned a slightly
  // different width than requested, recompute against the image we actually
  // got so tile boxes still line up.
  const downsample = useMemo(() => {
    if (!imgSize || !downsampleHint) return null;
    return downsampleHint.w0 / imgSize.width;
  }, [imgSize, downsampleHint]);

  const heatMinMax = useMemo(() => {
    if (highlightMode !== "Heatmap" || heatHpc === null || heatHpc === undefined) return null;
    return computeMinMax(tiles, `p_hpc_${Number(heatHpc)}`);
  }, [tiles, highlightMode, heatHpc]);

  const [noMatchWarning, setNoMatchWarning] = useState(null);
  const [hovered, setHovered] = useState(null); // { tile, left, top } in stage CSS px

  // The same lattice lookup the pyramid viewer hovers with. A click still
  // uses the tolerant scan below, as it always has; hover asks the stricter
  // question "whose cell is this pixel in", which is what the highlight shows.
  const lookup = useMemo(() => buildTileLookup(tiles, tileSizeNative), [tiles, tileSizeNative]);

  if (!thumbnailUrl) return null;

  function handleImgLoad(e) {
    setImgSize({ width: e.target.naturalWidth, height: e.target.naturalHeight });
  }

  function nativePoint(e) {
    const img = e.currentTarget;
    const rect = img.getBoundingClientRect();
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX * downsample,
      y: (e.clientY - rect.top) * scaleY * downsample,
      rect,
    };
  }

  function handleMove(e) {
    if (!downsample) return;
    const { x, y, rect } = nativePoint(e);
    const tile = tileAtImagePoint(lookup, x, y, tileSizeNative);
    if (!tile) {
      if (hovered) setHovered(null);
      return;
    }
    if (hovered && hovered.tile === tile) return;
    // Tooltip anchored beside the tile's right edge, flipped left near the
    // image's edge so it never runs off the stage.
    const cssPerNative = rect.width / (imgSize.width * downsample);
    const tileRight = (Number(tile.x_native) + tileSizeNative) * cssPerNative;
    const tileLeft = Number(tile.x_native) * cssPerNative;
    const left = tileRight + 10 + 220 > rect.width ? Math.max(4, tileLeft - 230) : tileRight + 10;
    const top = Math.max(4, Math.min(Number(tile.y_native) * cssPerNative, rect.height - 70));
    setHovered({ tile, left, top });
  }

  function handleClick(e) {
    if (!downsample) return;
    const img = e.currentTarget;
    const rect = img.getBoundingClientRect();
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;
    const cx = (e.clientX - rect.left) * scaleX;
    const cy = (e.clientY - rect.top) * scaleY;

    const nativeX = cx * downsample;
    const nativeY = cy * downsample;
    const tol = tileSizeNative * 0.1;

    const match = tiles.find((t) => {
      const tx = Number(t.x_native);
      const ty = Number(t.y_native);
      return tx - tol <= nativeX && nativeX <= tx + tileSizeNative + tol && ty - tol <= nativeY && nativeY <= ty + tileSizeNative + tol;
    });

    if (!match) {
      // Matches the Python version's behavior: a miss just warns and leaves
      // whatever tile was already selected (if any) in place.
      setNoMatchWarning("Clicked area does not match any tile.");
      return;
    }

    setNoMatchWarning(null);
    onSelectTile({
      slide_tile: String(match.slide_tile || ""),
      x_native: Number(match.x_native),
      y_native: Number(match.y_native),
      tile: match,
    });
  }

  const ts = downsample ? tileSizeNative / downsample : 0;

  const rects = [];
  if (imgSize && downsample) {
    if (highlightMode === "Heatmap" && heatMinMax) {
      const col = `p_hpc_${Number(heatHpc)}`;
      for (const t of tiles) {
        const p = Number(t[col]);
        if (!Number.isFinite(p)) continue;
        const x = Number(t.x_native) / downsample;
        const y = Number(t.y_native) / downsample;
        const { r, g, b, alpha } = heatmapColor(p, heatMinMax.pmin, heatMinMax.pmax, heatAlpha ?? 0.6);
        rects.push(
          <rect key={`heat_${t.slide_tile}`} x={x} y={y} width={ts} height={ts} fill={`rgba(${r},${g},${b},${alpha.toFixed(3)})`} stroke="rgba(255,255,255,0.16)" />
        );
      }
    } else if (highlightMode === "Survival Risk Heatmap") {
      for (const t of tiles) {
        const score = Number(t.survival_risk_norm);
        if (!Number.isFinite(score)) continue;
        const [r, g, b, a] = riskToRgba(score, 20, 140);
        if (a === 0) continue;
        const x = Number(t.x_native) / downsample;
        const y = Number(t.y_native) / downsample;
        rects.push(
          <rect key={`risk_${t.slide_tile}`} x={x} y={y} width={ts} height={ts} fill={rgbaToCss([r, g, b], a)} stroke="rgba(255,255,255,0.14)" />
        );
      }
    } else if (showGrid) {
      for (const t of gridTiles) {
        if (!passesGridFilters(t, inflF, necF, malF)) continue;
        const color = gridColorForTile(highlightMode, t, adjTileSets);
        if (!color) continue;
        const x = Number(t.x_native) / downsample;
        const y = Number(t.y_native) / downsample;
        rects.push(
          <rect key={`grid_${t.slide_tile}`} x={x} y={y} width={ts} height={ts} fill="none" stroke={rgbToCss(color)} strokeWidth={2} />
        );
      }
    }

    if (selectedTile && selectedTile.x_native !== undefined) {
      const sx = Number(selectedTile.x_native) / downsample;
      const sy = Number(selectedTile.y_native) / downsample;
      rects.push(
        <rect key="sel_outer" x={sx - 3} y={sy - 3} width={ts + 6} height={ts + 6} fill="none" stroke="yellow" strokeWidth={6} />
      );
      rects.push(<rect key="sel_inner" x={sx} y={sy} width={ts} height={ts} fill="none" stroke="lime" strokeWidth={3} />);
    }

    if (hovered) {
      const hx = Number(hovered.tile.x_native) / downsample;
      const hy = Number(hovered.tile.y_native) / downsample;
      rects.push(
        <rect key="hover" x={hx} y={hy} width={ts} height={ts} fill="rgba(255,255,255,0.22)" stroke="#ffffff" strokeWidth={2} />
      );
    }
  }

  const tileRow = selectedTile && selectedTile.tile;
  const heatHpcForInfo = highlightMode === "Heatmap" ? heatHpc : null;
  const hoveredHpc = hovered && hovered.tile.hpc_id !== null && hovered.tile.hpc_id !== undefined ? Number(hovered.tile.hpc_id) : null;
  const hoveredSwatch = hoveredHpc === null ? "#6b7280" : rgbToCss(colorForHpc(hoveredHpc));

  return (
    <div>
      <div className="viewer-caption">Hover to identify a tile; click it for its image and details.</div>
      <div className={`viewer-click-layout${tileRow ? " viewer-click-layout-with-panel" : ""}`}>
        <div className="viewer-click-stage">
          <img
            src={thumbnailUrl}
            alt={`${slideId} thumbnail`}
            onLoad={handleImgLoad}
            onClick={handleClick}
            onMouseMove={handleMove}
            onMouseLeave={() => setHovered(null)}
          />
          {imgSize && (
            <svg className="viewer-click-overlay-svg" viewBox={`0 0 ${imgSize.width} ${imgSize.height}`}>
              {rects}
            </svg>
          )}
          {hovered && (
            <div className="viewer-osd-hud" style={{ left: hovered.left, top: hovered.top, borderLeftColor: hoveredSwatch }}>
              <span className="viewer-osd-hud-tile">{hovered.tile.tiles || hovered.tile.slide_tile}</span>
              <span className="viewer-osd-hud-hpc">
                <span className="viewer-osd-hud-swatch" style={{ background: hoveredSwatch }} />
                {hpcLabel(hoveredHpc, hpcTitleMap, 90, "no HPC label")}
              </span>
              <span className="viewer-osd-hud-hint">Click for details</span>
            </div>
          )}
        </div>

        {tileRow && (
          <TileDetailCard
            slideId={slideId}
            tile={tileRow}
            tileSizeNative={tileSizeNative}
            heatHpc={heatHpcForInfo}
            hpcTitle={hpcTitleMap?.get(Number(tileRow.hpc_id)) || ""}
            onClose={() => onSelectTile(null)}
          />
        )}
      </div>

      {noMatchWarning && <div className="viewer-warning">{noMatchWarning}</div>}
    </div>
  );
}

// Port of hpc_label(hpc_id, max_title_chars=160) (app_v28.py ~line 3378),
// using the slide-scoped title map built by hpcReferenceCache instead of the
// bulk load_hpc_title_map().
function hpcLabel(hpcId, hpcTitleMap, maxTitleChars = 160, missing = "") {
  if (hpcId === null || hpcId === undefined || Number.isNaN(Number(hpcId))) return missing;
  const hid = Number(hpcId);
  let title = (hpcTitleMap && hpcTitleMap.get(hid)) || "";
  if (!title) return `HPC ${hid}`;
  if (title.length > maxTitleChars) title = title.slice(0, maxTitleChars - 1) + "…";
  return `HPC ${hid}: ${title}`;
}
