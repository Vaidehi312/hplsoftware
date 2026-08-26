#!/usr/bin/env python3
"""One-time migration for runs tiled before per-dataset folders existed.

Older runs wrote tiles flat: processed_tiles/<slide_id>/. Everything since
the dataset-folder change (submit_mask_tile_slurm.py, find_missing_slides.py,
make_hpl_hdf5.py) now looks for processed_tiles/<dataset_name>/<slide_id>/
instead. Left unmigrated, a flat run looks like it has zero tiles to those
scripts — "Resume missing slides" then treats the entire dataset as never
attempted and resubmits everything, wastefully re-tiling slides that already
succeeded.

Scans raw_dir directly (ground-truth filesystem check, same approach as
find_missing_slides.py) rather than trusting any one manifest file, since a
dataset can have more than one overlapping submission by now. For each slide
whose tiles exist flat but not yet under the dataset folder, moves (renames)
its whole tile directory into place. Safe to re-run: anything already at the
destination is left alone, nothing is ever deleted.

Usage:
    python migrate_flat_tiles_to_dataset_folder.py \\
        --raw-dir /path/to/Radiogenomics \\
        --tile-dir /path/to/processed_tiles \\
        --dataset-name Radiogenomics \\
        --dry-run   # preview only, no changes

    # once the dry-run output looks right, re-run without --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from slide_naming import slide_id_from_raw_path
from submit_mask_tile_slurm import discover_slides


def migrate(raw_dir: Path, tile_dir: Path, dataset_name: str, dry_run: bool) -> dict:
    slides = discover_slides(raw_dir)
    dest_root = tile_dir / dataset_name
    if not dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)

    moved, already_migrated, not_found = [], [], []
    for slide_path in slides:
        slide_id = slide_id_from_raw_path(slide_path)
        old_dir = tile_dir / slide_id
        new_dir = dest_root / slide_id

        if new_dir.exists():
            already_migrated.append(slide_id)
            continue
        if not old_dir.is_dir():
            not_found.append(slide_id)
            continue

        if not dry_run:
            shutil.move(str(old_dir), str(new_dir))
        moved.append(slide_id)

    return {
        "total_discovered": len(slides),
        "moved": moved,
        "already_migrated": len(already_migrated),
        "not_found": not_found,
        "dry_run": dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--tile-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, required=True, help="e.g. Radiogenomics or TCGA")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = migrate(
        args.raw_dir.expanduser().resolve(),
        args.tile_dir.expanduser().resolve(),
        args.dataset_name,
        args.dry_run,
    )

    verb = "Would move" if result["dry_run"] else "Moved"
    print(f"Slides discovered under raw_dir: {result['total_discovered']}")
    print(f"{verb}: {len(result['moved'])}")
    print(f"Already in new location (untouched): {result['already_migrated']}")
    print(f"Not found at either location: {len(result['not_found'])}")
    if result["not_found"]:
        print("(these were never tiled at all, or tiled under a different dataset_name — not migrated)")
        for slide_id in result["not_found"][:20]:
            print(f"  {slide_id}")
        if len(result["not_found"]) > 20:
            print(f"  ... and {len(result['not_found']) - 20} more")


if __name__ == "__main__":
    main()
