// What the tile card says about a tile, as plain data: [{title, rows:
// [[label, value, hint?]]}]. Kept out of the component so it can be tested
// without a DOM, and so "which fields, in what order, formatted how" lives in
// one place.
//
// A field the tile does not carry is left out rather than shown blank — a KB
// that predates migrate_tile_registry_confidence.sql has no vote margin, and
// a row reading "Vote margin: —" would look like a tile the classifier had
// nothing to say about.

function isMissing(v) {
  if (v === null || v === undefined) return true;
  if (typeof v === "number" && Number.isNaN(v)) return true;
  return typeof v === "string" && v.trim() === "";
}

function fmtNumber(v, digits) {
  const n = Number(v);
  if (!Number.isFinite(n)) return null;
  return n.toFixed(digits);
}

function fmtInt(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return null;
  return Math.round(n).toLocaleString("en-GB");
}

// A reference is a path on the cluster; its file name is what tells two apart.
function basename(v) {
  const s = String(v);
  const i = Math.max(s.lastIndexOf("/"), s.lastIndexOf("\\"));
  return i >= 0 ? s.slice(i + 1) : s;
}

function push(rows, label, value, hint) {
  if (isMissing(value)) return;
  rows.push(hint ? [label, String(value), hint] : [label, String(value)]);
}

export function tileDetailSections(tile, { heatHpc = null } = {}) {
  if (!tile) return [];
  const sections = [];

  const classification = [];
  push(classification, "Vote margin", isMissing(tile.hpc_vote_margin) ? null : fmtNumber(tile.hpc_vote_margin, 3),
    "(top votes − runner-up votes) / k. Near 0: the tile sits between two clusters.");
  push(classification, "Neighbour distance", isMissing(tile.hpc_neighbor_distance) ? null : fmtNumber(tile.hpc_neighbor_distance, 3),
    "Distance to the nearest reference tile. Large: unlike anything in the reference.");
  push(classification, "Reference", isMissing(tile.hpc_reference) ? null : basename(tile.hpc_reference), String(tile.hpc_reference || ""));
  push(classification, "Assigned", tile.hpc_assigned_at);
  if (classification.length) sections.push({ title: "Classification", rows: classification });

  const scores = [];
  if (heatHpc !== null && heatHpc !== undefined) {
    const p = tile[`p_hpc_${Number(heatHpc)}`];
    push(scores, `P(HPC ${heatHpc})`, isMissing(p) ? null : fmtNumber(p, 4));
  }
  push(scores, "Survival risk", isMissing(tile.survival_risk_norm) ? null : fmtNumber(tile.survival_risk_norm, 3),
    "Normalised survival-risk score for this tile, from the HPCs' Cox coefficients.");
  if (scores.length) sections.push({ title: "Scores", rows: scores });

  // These come from hpc_dictionary, joined on the tile's hpc_id — they
  // describe the cluster, not a reading of this tile, and the heading says so.
  const morphology = [];
  push(morphology, "Malignant", tile.malignant);
  push(morphology, "Inflammation", tile.inflammation);
  push(morphology, "Necrosis", tile.necrosis);
  if (morphology.length) sections.push({ title: "Cluster morphology", rows: morphology });

  const position = [];
  const col = isMissing(tile.col) ? null : fmtInt(tile.col);
  const row = isMissing(tile.row) ? null : fmtInt(tile.row);
  push(position, "Grid (col, row)", col !== null && row !== null ? `${col}, ${row}` : null);
  const xn = isMissing(tile.x_native) ? null : fmtInt(tile.x_native);
  const yn = isMissing(tile.y_native) ? null : fmtInt(tile.y_native);
  push(position, "Native x, y (px)", xn !== null && yn !== null ? `${xn}, ${yn}` : null);
  push(position, "HDF5 index", isMissing(tile.h5_index) ? null : fmtInt(tile.h5_index));
  push(position, "Slide", tile.slides);
  push(position, "Key", tile.slide_tile);
  if (position.length) sections.push({ title: "Position", rows: position });

  return sections;
}
