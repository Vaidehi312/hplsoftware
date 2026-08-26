import { useMemo, useState } from "react";

// Port of the Adjacency expander inside show_wsi (app_v28.py ~line 4071-4091):
// top 15 most-cooccurring HPC pairs, a dropdown to pick one, and a
// "Highlight selected top pair" button.
//
// pairEdgeCounts: array of { a, b, count } (from api.getAdjacency, parsed by
//   the parent the same way load_adjacency_for_slide does).
// tileNeighborPairs: Map keyed by `${min(a,b)}_${max(a,b)}` -> { aTouch: Set<string>, bTouch: Set<string> }.
// onHighlight(a, b, tileSets): called when the button is pressed.
export default function AdjacencyControls({ pairEdgeCounts, tileNeighborPairs, onHighlight }) {
  const topPairs = useMemo(() => {
    return [...(pairEdgeCounts || [])].sort((x, y) => y.count - x.count).slice(0, 15);
  }, [pairEdgeCounts]);

  const [selectedIdx, setSelectedIdx] = useState(0);

  if (topPairs.length === 0) {
    return (
      <details className="viewer-adjacency-controls">
        <summary>Adjacency and cooccurrence controls</summary>
        <div className="viewer-adjacency-body">
          <div className="viewer-info">No adjacent cross HPC pairs found on this slide.</div>
        </div>
      </details>
    );
  }

  const idx = Math.min(selectedIdx, topPairs.length - 1);
  const current = topPairs[idx];

  const handleHighlight = () => {
    const a = current.a;
    const b = current.b;
    const key = a < b ? `${a}_${b}` : `${b}_${a}`;
    const tileSets = (tileNeighborPairs && tileNeighborPairs.get(key)) || { aTouch: new Set(), bTouch: new Set() };
    onHighlight(a, b, tileSets);
  };

  return (
    <details className="viewer-adjacency-controls" open>
      <summary>Adjacency and cooccurrence controls</summary>
      <div className="viewer-adjacency-body">
        <label htmlFor="viewer-adj-pair-select" className="viewer-caption">
          Pick a top cooccurring pair
        </label>
        <select
          id="viewer-adj-pair-select"
          className="viewer-select"
          value={idx}
          onChange={(e) => setSelectedIdx(Number(e.target.value))}
        >
          {topPairs.map((p, i) => (
            <option key={`${p.a}_${p.b}`} value={i}>
              HPC {p.a} ↔ HPC {p.b} ({p.count} edges)
            </option>
          ))}
        </select>
        <button type="button" className="viewer-btn" onClick={handleHighlight}>
          Highlight selected top pair
        </button>
      </div>
    </details>
  );
}
