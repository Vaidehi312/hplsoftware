import { useEffect, useRef, useState } from "react";
import OpenSeadragon from "openseadragon";

// Port of render_openseadragon_viewer (app_v28.py ~line 2974). The Streamlit
// version loads OpenSeadragon from a CDN `<script>` tag inside a
// components.html iframe and syncs an SVG overlay to the viewport on
// open/animation/animation-finish/resize/zoom/pan. Here we drive the
// `openseadragon` npm package directly from a useEffect — the sync math
// (imageToViewportRectangle + pixelFromPoint) is ported near-verbatim.
//
// `overlayTiles`: array of records with {x,y,w,h} in native slide pixels
// plus either `color` (grid-outline mode) or `fill`/`stroke` (heatmap/risk
// mode) — the same shape build_osd_overlay_records / build_survival_osd_overlay_records
// produce.
// `selectedTileRect`: optional {x,y,w,h} in native pixels — draws a
// yellow+lime double outline like the click-inspector's selected-tile
// highlight. NOTE: the original Streamlit app has no click-to-select
// interaction in Pyramid zoom mode at all (only Click tile inspector mode
// supports it), so this rect only ever appears here when a tile was
// selected earlier in Click tile inspector mode and the user then switched
// to Pyramid zoom — a small added-value carry-over, not a literal port.
export default function PyramidViewer({ dziUrl, overlayTiles = [], selectedTileRect = null, height = 780 }) {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const viewerRef = useRef(null);
  const drawRef = useRef(null);
  const overlayTilesRef = useRef(overlayTiles);
  const selectedTileRectRef = useRef(selectedTileRect);
  const [failed, setFailed] = useState(false);

  overlayTilesRef.current = overlayTiles;
  selectedTileRectRef.current = selectedTileRect;

  useEffect(() => {
    if (!dziUrl || !containerRef.current) return undefined;
    setFailed(false);

    const viewer = OpenSeadragon({
      element: containerRef.current,
      prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/",
      tileSources: dziUrl,
      showNavigator: true,
      navigatorPosition: "BOTTOM_RIGHT",
      animationTime: 0.4,
      blendTime: 0.1,
      constrainDuringPan: true,
      visibilityRatio: 0.8,
      minZoomImageRatio: 0.8,
      maxZoomPixelRatio: 3.0,
      showRotationControl: false,
      gestureSettingsMouse: {
        clickToZoom: true,
        dblClickToZoom: true,
        dragToPan: true,
        scrollToZoom: true,
      },
    });
    viewerRef.current = viewer;

    function appendRect(svg, t) {
      const rectVp = viewer.viewport.imageToViewportRectangle(Number(t.x), Number(t.y), Number(t.w), Number(t.h));
      const p1 = viewer.viewport.pixelFromPoint(rectVp.getTopLeft(), true);
      const p2 = viewer.viewport.pixelFromPoint(rectVp.getBottomRight(), true);

      const x = Math.min(p1.x, p2.x);
      const y = Math.min(p1.y, p2.y);
      const w = Math.abs(p2.x - p1.x);
      const h = Math.abs(p2.y - p1.y);

      const container = viewer.container.getBoundingClientRect();
      if (x + w < 0 || y + h < 0 || x > container.width || y > container.height) return;

      const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      r.setAttribute("x", x);
      r.setAttribute("y", y);
      r.setAttribute("width", Math.max(w, 1));
      r.setAttribute("height", Math.max(h, 1));
      r.setAttribute("fill", t.fill || "none");
      r.setAttribute("stroke", t.stroke || t.color || "yellow");
      r.setAttribute("stroke-width", t.stroke_width || Math.max(1, Math.min(4, w / 35)));
      r.setAttribute("opacity", t.opacity || "0.95");
      svg.appendChild(r);
    }

    function drawTileOverlay() {
      if (!viewer || !viewer.viewport || !viewer.world || viewer.world.getItemCount() === 0) return;
      const svg = svgRef.current;
      if (!svg) return;

      const container = viewer.container.getBoundingClientRect();
      svg.setAttribute("viewBox", `0 0 ${container.width} ${container.height}`);
      svg.innerHTML = "";

      for (const t of overlayTilesRef.current || []) {
        appendRect(svg, t);
      }

      const sel = selectedTileRectRef.current;
      if (sel) {
        const pad = Number(sel.w || 0) * 0.06;
        appendRect(svg, {
          x: sel.x - pad,
          y: sel.y - pad,
          w: sel.w + 2 * pad,
          h: sel.h + 2 * pad,
          fill: "none",
          stroke: "yellow",
          stroke_width: 6,
          opacity: "1.0",
        });
        appendRect(svg, { x: sel.x, y: sel.y, w: sel.w, h: sel.h, fill: "none", stroke: "lime", stroke_width: 3, opacity: "1.0" });
      }
    }

    drawRef.current = drawTileOverlay;

    viewer.addHandler("open", drawTileOverlay);
    viewer.addHandler("animation", drawTileOverlay);
    viewer.addHandler("animation-finish", drawTileOverlay);
    viewer.addHandler("resize", drawTileOverlay);
    viewer.addHandler("zoom", drawTileOverlay);
    viewer.addHandler("pan", drawTileOverlay);

    viewer.addHandler("open-failed", (event) => {
      console.error("OpenSeadragon open failed", event);
      setFailed(true);
    });
    viewer.addHandler("tile-load-failed", (event) => {
      console.error("OpenSeadragon tile load failed", event);
    });

    return () => {
      drawRef.current = null;
      viewer.destroy();
      if (viewerRef.current === viewer) viewerRef.current = null;
    };
    // Only (re)create the viewer when the DZI source changes — overlay data
    // changes are handled by the effect below via the ref + a direct redraw.
  }, [dziUrl]);

  useEffect(() => {
    if (drawRef.current) drawRef.current();
  }, [overlayTiles, selectedTileRect]);

  return (
    <div>
      <div className="viewer-caption">OpenSeadragon DZI source: {dziUrl}</div>
      <div className="viewer-osd-wrap" style={{ height }}>
        <div ref={containerRef} className="viewer-osd-canvas" style={{ height }} />
        <svg ref={svgRef} className="viewer-osd-overlay-svg" />
        {failed && (
          <div className="viewer-osd-failed">
            OpenSeadragon failed to open the DZI source. Try opening this directly:{" "}
            <a href={dziUrl} target="_blank" rel="noreferrer">
              {dziUrl}
            </a>
          </div>
        )}
      </div>
    </div>
  );
}
