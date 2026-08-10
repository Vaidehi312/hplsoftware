#!/usr/bin/env python3
"""Concatenate the part files written by a sharded feature-extraction run back
into the single projections .h5 the rest of the pipeline expects.

When feature extraction runs as N Slurm tasks, each encodes a disjoint row
range of the same input .h5 (see --row_start/--row_stop in HPL-LATTICeA's
run_representationspathology_projection.py) and writes
`<name>.rows<lo>-<hi>.h5` beside where the merged file will go. This puts them
back together.

Everything here is about one failure mode. Concatenating in the wrong order,
or with a gap, or over a part that died halfway through, produces a file of
exactly the right shape and dtype with no missing values — it passes every
completeness check the pipeline has, and every downstream cluster assignment
is silently attached to the wrong tile. So the parts are required to tile
[0, total) exactly before a single row is copied, each part is checked against
the range its own filename claims, and the result is written to a temporary
file and renamed only once it is whole.

Usage:
    python merge_projection_shards.py \\
        --output /path/to/results/BarlowTwins_3/DS/h224_w224_n3_zdim128/hdf5_DS_he_train.h5

    # With the input, so the total row count is verified rather than inferred:
    python merge_projection_shards.py --output ... --input-h5 /path/to/hdf5_DS_he_train.h5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np

# Rows copied per read/write. Latents are small per row (~6 KB for h at 1536
# floats), so this is a few hundred MB in flight — large enough that the copy
# is not dominated by per-call overhead, small enough to stay off the heap.
_COPY_CHUNK = 8192

_PART_RE = re.compile(r"^(?P<stem>.+)\.rows(?P<lo>\d+)-(?P<hi>\d+)\.h5$")


def find_parts(final_path: Path) -> list[tuple[int, int, Path]]:
    """Every part file belonging to final_path, as (lo, hi, path), row-sorted."""
    stem = final_path.name[:-len(".h5")] if final_path.name.endswith(".h5") else final_path.name
    parts = []
    for candidate in final_path.parent.glob(f"{stem}.rows*-*.h5"):
        m = _PART_RE.match(candidate.name)
        if m and m.group("stem") == stem:
            parts.append((int(m.group("lo")), int(m.group("hi")), candidate))
    return sorted(parts)


def check_parts_tile_exactly(
    parts: list[tuple[int, int, Path]], expected_rows: int | None
) -> int:
    """Confirm the parts cover [0, total) with no gap, overlap or duplicate.

    Returns the total row count. Raises ValueError naming the specific defect —
    'shards do not line up' is not something anyone can act on at 2am.
    """
    if not parts:
        raise ValueError(
            "No part files found. Expected files named like "
            "'<output stem>.rows0-12345.h5' beside the output path."
        )

    cursor = 0
    for lo, hi, path in parts:
        if hi <= lo:
            raise ValueError(f"{path.name}: empty or reversed range [{lo}, {hi}).")
        if lo < cursor:
            raise ValueError(
                f"{path.name}: overlaps the previous part — starts at row {lo} "
                f"but rows up to {cursor} are already covered."
            )
        if lo > cursor:
            raise ValueError(
                f"Gap in coverage: no part supplies rows [{cursor}, {lo}). "
                f"The job for that range is missing or failed — check the array "
                f"task's Slurm log rather than merging what is here."
            )
        cursor = hi

    if expected_rows is not None and cursor != expected_rows:
        raise ValueError(
            f"Parts cover {cursor} rows but the input has {expected_rows}. "
            f"{'The last shard is missing.' if cursor < expected_rows else 'A part claims rows beyond the input.'}"
        )
    return cursor


def _dataset_plan(parts: list[tuple[int, int, Path]]) -> dict[str, tuple[tuple, np.dtype]]:
    """Datasets to create in the merged file, checked consistent across parts.

    A part written by a different model, z_dim or input would otherwise
    concatenate cleanly into nonsense.
    """
    plan: dict[str, tuple[tuple, np.dtype]] = {}
    reference: Path | None = None

    for lo, hi, path in parts:
        with h5py.File(path, "r") as f:
            names = set(f.keys())
            if reference is None:
                reference = path
                for name in sorted(names):
                    plan[name] = (tuple(f[name].shape[1:]), f[name].dtype)
            elif names != set(plan):
                missing = sorted(set(plan) - names)
                extra = sorted(names - set(plan))
                raise ValueError(
                    f"{path.name} does not hold the same datasets as {reference.name}"
                    + (f"; missing {missing}" if missing else "")
                    + (f"; unexpected {extra}" if extra else "")
                )

            # sorted() so that a part with more than one defect always reports
            # the same one; otherwise the message varies with the hash seed.
            for name in sorted(names):
                tail, dtype = plan[name]
                if tuple(f[name].shape[1:]) != tail:
                    raise ValueError(
                        f"{path.name}: '{name}' has shape {f[name].shape[1:]} per row, "
                        f"but {reference.name} has {tail}. These are not parts of one run."
                    )
                if f[name].dtype != dtype:
                    raise ValueError(
                        f"{path.name}: '{name}' is {f[name].dtype}, "
                        f"but {reference.name} is {dtype}."
                    )
                # A part that died mid-write is the whole reason the range is
                # in the filename: its rows can be counted against its claim.
                if f[name].shape[0] != hi - lo:
                    raise ValueError(
                        f"{path.name}: '{name}' holds {f[name].shape[0]} rows but the "
                        f"filename claims rows [{lo}, {hi}) — {hi - lo} of them. That "
                        f"part is incomplete; re-run its array task rather than merging it."
                    )
    return plan


def merge_projection_shards(
    final_path: Path,
    *,
    expected_rows: int | None = None,
    cleanup: bool = False,
    force: bool = False,
) -> dict:
    parts = find_parts(final_path)
    total_rows = check_parts_tile_exactly(parts, expected_rows)
    plan = _dataset_plan(parts)

    if final_path.exists() and not force:
        raise FileExistsError(
            f"{final_path} already exists. Delete it, or pass --force, if you mean "
            f"to rebuild it from the parts."
        )

    # Written under a temporary name and renamed at the end. A merge killed
    # halfway would otherwise leave a full-sized, part-zeroed file at exactly
    # the path the encoder treats as 'already done'.
    tmp_path = final_path.with_name(final_path.name + ".merging")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        with h5py.File(tmp_path, "w") as out:
            for name, (tail, dtype) in plan.items():
                out.create_dataset(name, shape=(total_rows,) + tail, dtype=dtype)

            for lo, hi, path in parts:
                with h5py.File(path, "r") as f:
                    for name in plan:
                        src, dst = f[name], out[name]
                        for offset in range(0, hi - lo, _COPY_CHUNK):
                            end = min(offset + _COPY_CHUNK, hi - lo)
                            dst[lo + offset:lo + end] = src[offset:end]
                print(f"  merged rows {lo:>10,}-{hi:<10,} from {path.name}", flush=True)

        tmp_path.replace(final_path)
    except BaseException:
        # Never leave the half-built file behind under either name.
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    removed = []
    if cleanup:
        for _, _, path in parts:
            path.unlink()
            removed.append(path.name)

    return {
        "output": str(final_path),
        "rows": total_rows,
        "parts": len(parts),
        "datasets": sorted(plan),
        "removed_parts": removed,
    }


def _input_rows(path: Path) -> int | None:
    try:
        with h5py.File(path, "r") as f:
            for key in f.keys():
                if "image" in key or "img" in key:
                    return int(f[key].shape[0])
    except (OSError, KeyError, ValueError):
        return None
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True,
                        help="Path the merged projections file should end up at. Part "
                             "files are looked for beside it.")
    parser.add_argument("--input-h5", type=Path, default=None,
                        help="The packaged .h5 that was encoded. Used to verify the parts "
                             "cover every tile rather than inferring the total from them.")
    parser.add_argument("--expected-rows", type=int, default=None,
                        help="Alternative to --input-h5 when the input is not to hand.")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete the part files once the merge has succeeded.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing output file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    expected = args.expected_rows
    if expected is None and args.input_h5 is not None:
        expected = _input_rows(args.input_h5)
        if expected is None:
            print(f"Could not read a row count from {args.input_h5}; "
                  f"merging without that check.", file=sys.stderr)

    try:
        info = merge_projection_shards(
            args.output, expected_rows=expected, cleanup=args.cleanup, force=args.force,
        )
    except (ValueError, FileExistsError, OSError) as e:
        print(f"Merge failed: {e}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Merged {info['parts']} parts -> {info['output']}")
    print(f"  rows      {info['rows']:,}")
    print(f"  datasets  {', '.join(info['datasets'])}")
    if info["removed_parts"]:
        print(f"  removed   {len(info['removed_parts'])} part files")


if __name__ == "__main__":
    main()
