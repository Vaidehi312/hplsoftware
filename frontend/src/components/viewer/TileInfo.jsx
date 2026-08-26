// Port of render_tile_info(tile_row, heat_hpc) (app_v28.py ~line 3836).
// Small info table for a selected tile, plus a metric when in Heatmap mode.
export default function TileInfo({ tile, heatHpc = null }) {
  if (!tile) return null;

  const clean = (v) => {
    if (v === null || v === undefined) return "";
    if (typeof v === "number" && Number.isNaN(v)) return "";
    return String(v);
  };

  const fields = [
    ["slide_tile", tile.slide_tile],
    ["tiles", tile.tiles],
    ["hpc_id", tile.hpc_id],
    ["hpc_title", tile.hpc_title],
    ["inflammation", tile.inflammation],
    ["necrosis", tile.necrosis],
    ["malignant", tile.malignant],
  ];

  let heatMetric = null;
  if (heatHpc !== null && heatHpc !== undefined) {
    const col = `p_hpc_${Number(heatHpc)}`;
    const p = tile[col];
    if (p !== null && p !== undefined && Number.isFinite(Number(p))) {
      heatMetric = { hpc: heatHpc, value: Number(p) };
    }
  }

  return (
    <div>
      <h4>Tile info</h4>
      <table className="viewer-table">
        <thead>
          <tr>
            <th>Field</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {fields.map(([field, value]) => (
            <tr key={field}>
              <td>{field}</td>
              <td>{clean(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {heatMetric && (
        <div className="viewer-metric">
          <span className="viewer-metric-label">Heatmap probability for HPC {heatMetric.hpc}</span>
          <span className="viewer-metric-value">{heatMetric.value.toFixed(4)}</span>
        </div>
      )}
    </div>
  );
}
