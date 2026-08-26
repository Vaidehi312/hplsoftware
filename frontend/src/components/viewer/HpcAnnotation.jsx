import { useEffect, useState } from "react";
import { colorForHpc, rgbToCss } from "../../colors.js";
import { getHpcInfoCached } from "./hpcReferenceCache.js";

// Port of render_hpc_annotation(hpc_id) (app_v28.py ~line 3756). Fetches the
// per-HPC info row (title + malignant/non-malignant detail fields) and shows
// it as a small panel + details table, skipping null/NaN fields exactly like
// the original.
export default function HpcAnnotation({ hpcId }) {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (hpcId === null || hpcId === undefined || Number.isNaN(Number(hpcId))) {
      setInfo(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    getHpcInfoCached(hpcId)
      .then((data) => {
        if (!cancelled) setInfo(data);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message || String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [hpcId]);

  if (hpcId === null || hpcId === undefined || Number.isNaN(Number(hpcId))) {
    return <div className="viewer-warning">Invalid HPC ID: {String(hpcId)}</div>;
  }

  const h = Number(hpcId);

  if (loading && !info) {
    return <div className="viewer-loading">Loading HPC {h} info…</div>;
  }

  if (error) {
    return (
      <div className="viewer-warning">
        Could not fetch HPC {h} info: {error}
      </div>
    );
  }

  if (!info) return null;

  const hpcTitle = String(info.hpc_title || "").trim();
  const titleText = hpcTitle || "No title available";
  const swatch = rgbToCss(colorForHpc(h));

  const summaryFields = [
    ["malignant", "Malignant"],
    ["inflammation", "Inflammation"],
    ["necrosis", "Necrosis"],
    ["cluster_homogeneity", "Cluster homogeneity"],
  ];

  const isMissing = (v) => v === null || v === undefined || (typeof v === "number" && Number.isNaN(v));

  const rows = [];
  for (const [key, label] of summaryFields) {
    const value = info[key];
    if (!isMissing(value)) rows.push({ field: label, value: String(value) });
  }

  const mal = info.malignant_details || {};
  const nonMal = info.non_malignant_details || {};
  const detailSource = Object.keys(mal).length > 0 ? mal : nonMal;

  for (const [key, value] of Object.entries(detailSource)) {
    if (key === "id" || key === "hpc_id") continue;
    if (isMissing(value)) continue;
    const prettyKey = key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    rows.push({ field: prettyKey, value: String(value) });
  }

  return (
    <div>
      <div className="viewer-hpc-annotation" style={{ borderLeftColor: swatch }}>
        <div className="viewer-hpc-annotation-id">HPC {h}</div>
        <div className="viewer-hpc-annotation-title">{titleText}</div>
      </div>

      {rows.length > 0 ? (
        <table className="viewer-table">
          <thead>
            <tr>
              <th>Field</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td>{r.field}</td>
                <td>{r.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="viewer-info">No additional phenotype details found.</div>
      )}
    </div>
  );
}
