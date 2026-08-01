#!/usr/bin/env python3
"""Diff a dataset-run manifest against what actually has tiles on disk, to
find which slides never got tiled — e.g. after a partially-failed batched
Slurm submission. Ground-truth filesystem check, not a guess based on which
Slurm batch/job succeeded.

Usage:
    python find_missing_slides.py
        --manifest /path/to/slurm_manifests/wsi_manifest_20260718_193754.txt
        --tile-dir /path/to/processed_tiles
        --dataset-name TCGA
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slide_naming import slide_id_from_raw_path
from tile_metadata import read_tile_metadata, tile_metadata_path


def find_missing_slides_detailed(
    manifest_path: Path, tile_dir: Path, dataset_name: str
) -> dict:
    """Classify every slide in a run's manifest by what its tile metadata
    actually says, not merely whether a file is present at the expected path.

    File-existence alone used to be the whole check, which quietly conflated
    three different outcomes. A CSV left truncated or header-less by a tiling
    task that was OOM-killed or hit the array's time limit still *exists*, so
    it counted as done and was never re-tiled — while packaging, which does
    try to parse it, dropped that slide from the .h5. The slide silently
    vanished from the dataset. Corrupt metadata now lands in `missing`
    (re-tiling can genuinely fix it), while a legitimate zero-tile result
    stays out of it (re-tiling cannot).

    Returns raw paths, not slide_ids, for `missing` — that's what a resume
    submission needs to feed back to Slurm.
    """
    raw_paths = [
        line.strip() for line in manifest_path.read_text().splitlines() if line.strip()
    ]

    missing: list[str] = []       # never attempted — no metadata at all
    corrupt: list[str] = []       # attempted but output unusable — worth retrying
    zero_tile: list[str] = []     # attempted, legitimately no tissue — NOT worth retrying
    complete: list[str] = []

    for raw_path in raw_paths:
        slide_id = slide_id_from_raw_path(raw_path)
        meta = read_tile_metadata(tile_metadata_path(tile_dir, dataset_name, slide_id))
        if meta.status == "missing":
            missing.append(raw_path)
        elif meta.status == "corrupt":
            print(f"[{slide_id}] {meta.detail} — will be re-tiled")
            corrupt.append(raw_path)
        elif meta.status == "empty":
            zero_tile.append(raw_path)
        else:
            complete.append(raw_path)

    return {
        "missing": missing + corrupt,
        "never_attempted": missing,
        "corrupt": corrupt,
        "zero_tile": zero_tile,
        "complete": complete,
        "total": len(raw_paths),
    }


def find_missing_slides(manifest_path: Path, tile_dir: Path, dataset_name: str) -> tuple[list[str], int]:
    """Backwards-compatible wrapper: (slides needing re-tiling, total)."""
    result = find_missing_slides_detailed(manifest_path, tile_dir, dataset_name)
    return result["missing"], result["total"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tile-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, required=True, help="e.g. TCGA or Radiogenomics")
    args = parser.parse_args()

    result = find_missing_slides_detailed(args.manifest, args.tile_dir, args.dataset_name)

    print(f"Manifest:  {args.manifest}")
    print(f"Tile dir:  {args.tile_dir}")
    print(f"Total in manifest:            {result['total']}")
    print(f"Complete (usable tiles):      {len(result['complete'])}")
    print(f"Zero-tile (no tissue, final): {len(result['zero_tile'])}")
    print(f"Never attempted:              {len(result['never_attempted'])}")
    print(f"Corrupt metadata:             {len(result['corrupt'])}")
    print(f"=> need re-tiling:            {len(result['missing'])}")
    print()
    for raw_path in result["missing"]:
        print(raw_path)


if __name__ == "__main__":
    main()
