#!/usr/bin/env python3
"""Concatenate the per-shard CSVs from a sharded cluster-assignment run.

The HDF5 counterpart of this is backend/merge_projection_shards.py, and the
validation here is deliberately the same shape even though a CSV concatenation
is far simpler than an HDF5 one. The failure mode is identical: parts joined in
the wrong order, across a gap, or over a part that died mid-write produce a file
with the right columns and no missing values, which every downstream consumer
accepts while every cluster ID is attached to the wrong tile.

So the parts must tile [0, total) exactly before a single row is copied, each
part is checked against the row range its own filename claims, and the result is
written to a temporary file and renamed only once whole.

Usage:
    python merge_assignment_shards.py \\
        --output /path/to/DS_hpc_assignments.csv --expected-rows 1234567 --cleanup
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_PART_RE = re.compile(r"^(?P<stem>.+)\.rows(?P<lo>\d+)-(?P<hi>\d+)\.csv$")


def find_parts(final_path: Path) -> list[tuple[int, int, Path]]:
    stem = final_path.stem
    parts = []
    for candidate in final_path.parent.glob(f"{stem}.rows*-*.csv"):
        m = _PART_RE.match(candidate.name)
        if m and m.group("stem") == stem:
            parts.append((int(m.group("lo")), int(m.group("hi")), candidate))
    return sorted(parts)


def check_parts_tile_exactly(parts, expected_rows: int | None) -> int:
    if not parts:
        raise ValueError(
            "No part files found. Expected files named like "
            "'<output stem>.rows0-12345.csv' beside the output path."
        )
    cursor = 0
    for lo, hi, path in parts:
        if hi <= lo:
            raise ValueError(f"{path.name}: empty or reversed range [{lo}, {hi}).")
        if lo < cursor:
            raise ValueError(
                f"{path.name}: overlaps the previous part — starts at row {lo} but "
                f"rows up to {cursor} are already covered."
            )
        if lo > cursor:
            raise ValueError(
                f"Gap in coverage: no part supplies rows [{cursor}, {lo}). That "
                f"shard is missing or failed — check its Slurm log rather than "
                f"merging what is here."
            )
        cursor = hi
    if expected_rows is not None and cursor != expected_rows:
        raise ValueError(
            f"Parts cover {cursor} rows but the input has {expected_rows}. "
            + ("The last shard is missing."
               if cursor < expected_rows else "A part claims rows beyond the input.")
        )
    return cursor


def _read_header_and_count(path: Path) -> tuple[str, int]:
    with path.open() as fh:
        header = fh.readline().rstrip("\n")
        return header, sum(1 for _ in fh)


def merge_assignment_shards(final_path: Path, *, expected_rows: int | None = None,
                            cleanup: bool = False, force: bool = False) -> dict:
    parts = find_parts(final_path)
    total_rows = check_parts_tile_exactly(parts, expected_rows)

    header = None
    for lo, hi, path in parts:
        part_header, rows = _read_header_and_count(path)
        if header is None:
            header = part_header
        elif part_header != header:
            raise ValueError(
                f"{path.name} has different columns from {parts[0][2].name}:\n"
                f"  {part_header}\nvs\n  {header}\n"
                f"These are not parts of one run."
            )
        # The row range is in the filename precisely so a part that died partway
        # can be caught by counting its rows against its own claim.
        if rows != hi - lo:
            raise ValueError(
                f"{path.name}: holds {rows} rows but its name claims rows "
                f"[{lo}, {hi}) — {hi - lo} of them. That shard is incomplete; "
                f"re-run it rather than merging it."
            )

    if final_path.exists() and not force:
        raise FileExistsError(
            f"{final_path} already exists. Delete it, or pass --force, to rebuild "
            f"it from the parts."
        )

    tmp_path = final_path.with_name(final_path.name + ".merging")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        with tmp_path.open("w") as out:
            out.write(header + "\n")
            for lo, hi, path in parts:
                with path.open() as fh:
                    fh.readline()  # skip the part's own header
                    for line in fh:
                        out.write(line)
                print(f"  merged rows {lo:>10,}-{hi:<10,} from {path.name}", flush=True)
        tmp_path.replace(final_path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    removed = []
    if cleanup:
        for _, _, path in parts:
            path.unlink()
            removed.append(path.name)

    return {"output": str(final_path), "rows": total_rows, "parts": len(parts),
            "removed_parts": removed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True,
                        help="Where the merged CSV should end up. Parts are looked "
                             "for beside it.")
    parser.add_argument("--expected-rows", type=int, default=None,
                        help="Tile count of the projections file, so a missing final "
                             "shard is caught. Without it the parts are believed.")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete the parts once the merge has succeeded.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        info = merge_assignment_shards(
            args.output, expected_rows=args.expected_rows,
            cleanup=args.cleanup, force=args.force,
        )
    except (ValueError, FileExistsError, OSError) as e:
        print(f"Merge failed: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Merged {info['parts']} parts -> {info['output']}")
    print(f"  rows      {info['rows']:,}")
    if info["removed_parts"]:
        print(f"  removed   {len(info['removed_parts'])} part files")


if __name__ == "__main__":
    main()
