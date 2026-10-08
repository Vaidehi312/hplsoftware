// What the tile card says about a tile. The failure worth pinning is a quiet
// one: a field the KB does not have, shown as a blank or "NaN" row, reads as
// "the classifier had nothing to say about this tile" when the truth is that
// this KB predates the column.
import { describe, expect, it } from "vitest";

import { tileDetailSections } from "../components/viewer/tileDetails.js";

const FULL = {
  slide_tile: "TCGA-55-7574-01Z-00-DX1_18_15.JPEG",
  slides: "TCGA-55-7574-01Z-00-DX1",
  tiles: "18_15.jpeg",
  col: 18,
  row: 15,
  x_native: 31212,
  y_native: 26010,
  h5_index: 4031,
  hpc_id: 12,
  hpc_vote_margin: 0.62,
  hpc_neighbor_distance: 3.14159,
  hpc_reference: "/scratch/refs/hpc_reference_leiden_2p5_fold2.npz",
  hpc_assigned_at: "2026-09-01 09:00 UTC",
  inflammation: "high",
  necrosis: "none",
  malignant: "malignant",
  p_hpc_12: 0.87654,
};

function asMap(sections) {
  const out = {};
  for (const s of sections) for (const [label, value] of s.rows) out[`${s.title}/${label}`] = value;
  return out;
}

describe("tileDetailSections", () => {
  it("lays out every field a fully loaded tile carries", () => {
    const m = asMap(tileDetailSections(FULL, { heatHpc: 12 }));
    expect(m["Classification/Vote margin"]).toBe("0.620");
    expect(m["Classification/Neighbour distance"]).toBe("3.142");
    // The file name, not the cluster path — two references differ there.
    expect(m["Classification/Reference"]).toBe("hpc_reference_leiden_2p5_fold2.npz");
    expect(m["Scores/P(HPC 12)"]).toBe("0.8765");
    expect(m["Cluster morphology/Malignant"]).toBe("malignant");
    expect(m["Position/Grid (col, row)"]).toBe("18, 15");
    expect(m["Position/Native x, y (px)"]).toBe("31,212, 26,010");
    expect(m["Position/Key"]).toBe(FULL.slide_tile);
  });

  it("leaves out what the tile does not have rather than showing it blank", () => {
    const bare = { slide_tile: "S_1_2.JPEG", col: 1, row: 2, x_native: 1600, y_native: 3200, hpc_id: null, hpc_vote_margin: NaN, hpc_reference: "  " };
    const sections = tileDetailSections(bare);
    const titles = sections.map((s) => s.title);
    expect(titles).toEqual(["Position"]);
    for (const s of sections) for (const [, value] of s.rows) {
      expect(value).not.toBe("");
      expect(value).not.toMatch(/NaN|undefined|null/);
    }
  });

  it("shows a heatmap probability only in heatmap mode, and for the chosen HPC", () => {
    expect(asMap(tileDetailSections(FULL))["Scores/P(HPC 12)"]).toBeUndefined();
    expect(asMap(tileDetailSections(FULL, { heatHpc: 7 }))["Scores/P(HPC 7)"]).toBeUndefined();
  });

  it("keeps a real zero, which is a value and not a missing one", () => {
    const m = asMap(tileDetailSections({ ...FULL, hpc_vote_margin: 0, col: 0, row: 0 }));
    expect(m["Classification/Vote margin"]).toBe("0.000");
    expect(m["Position/Grid (col, row)"]).toBe("0, 0");
  });
});
