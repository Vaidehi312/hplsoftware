#!/usr/bin/env python3
"""Add the ".jpeg" suffix to tile names in artifacts packaged before the fix.

make_hpl_hdf5.py used to store a tile as "24_10" where Kai's reference CSVs,
tile_coordinates and tile_registry all use "24_10.jpeg". That is fixed at
source, but artifacts already on disk still carry the short form, and every one
of them is unjoinable against the Knowledge Bank because of it.

Nothing about the *data* in those artifacts is wrong. The embeddings were
computed from the correct tile images, the k-NN ran against the correct
reference, and the cluster IDs, margins and neighbour distances are all exactly
what a re-run would produce. Only the label is short. And the mapping is total
and lossless: auto_tile_from_mask.py:150 hardcodes

    tile_filename = f"{col}_{row}.jpeg"

so every tile this pipeline has ever written is a .jpeg, "24_10" identifies
exactly one of them, and "24_10" -> "24_10.jpeg" is a bijection. That is what
makes this a migration rather than a repair — there is no guessing involved.

Which is why this exists instead of a normalisation buried in the loader.
Rewriting the label where it is read would hide the defect for good and would
also "fix" a genuinely different tile name; doing it once, explicitly, to named
files, leaves the loader strict and the fix at the source where it belongs.

    python migrate_tile_names.py --csv assignments.csv          # dry run
    python migrate_tile_names.py --csv assignments.csv --commit
    python migrate_tile_names.py --h5 packaged.h5 --commit

CSV output is written beside the input as "<stem>_tilenames.csv" rather than
over it. An .h5 is rewritten in place — these run to hundreds of gigabytes and
copying one to change a string column is not sensible — but only the `tiles`
dataset is touched, and an interrupted write leaves the dataset missing, which
_validate_h5 rejects loudly rather than mistaking for complete.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slide_naming import _as_text, tiles_missing_suffix  # noqa: E402

_SUFFIX = ".jpeg"

# assign_hpc_clusters.py writes these columns, in this order. The cluster
# column sits fourth and is named for the reference's groupby ("leiden_2.5"),
# so it is filled in separately.
_ASSIGNMENT_COLUMNS = ("samples", "slides", "tiles", None,
                       "vote_margin", "neighbor_distance", "hpc_reference")

_COL_ROW_RE = re.compile(r"^\d+_\d+(\.[A-Za-z0-9]{1,5})?$")
# "hpc_reference_leiden_2p5_fold2" -> "leiden_2.5", the convention
# build_hpc_reference.py names its output by.
_REFERENCE_GROUPBY_RE = re.compile(r"leiden[_-](\d+)p(\d+)")


def cluster_column_from_reference(reference: str) -> str | None:
    """The groupby a reference's name implies, or None if it doesn't say."""
    match = _REFERENCE_GROUPBY_RE.search(str(reference))
    return f"leiden_{match.group(1)}.{match.group(2)}" if match else None


def detect_headerless(csv_path: Path, cluster_column: str | None = None) -> list[str] | None:
    """Column names for an assignments CSV written without a header row, or
    None if it has one.

    Recovering names by position is a guess, so it is only made when the row
    could not plausibly be a header: seven fields, the third shaped like a tile
    ("21_21"), the fourth an integer cluster id, the fifth and sixth floats, and
    the last naming a reference. A real header fails every one of those.
    """
    first = pd.read_csv(csv_path, nrows=1, header=None, dtype=str).iloc[0].tolist()
    if len(first) != len(_ASSIGNMENT_COLUMNS):
        return None
    if not _COL_ROW_RE.match(str(first[2]).strip()):
        return None
    try:
        int(str(first[3]).strip())
        float(str(first[4]).strip())
        float(str(first[5]).strip())
    except ValueError:
        return None
    if "reference" not in str(first[6]):
        return None

    groupby = cluster_column or cluster_column_from_reference(first[6])
    if groupby is None:
        raise SystemExit(
            f"{csv_path} has no header row, and its reference {first[6]!r} does "
            f"not say which Leiden resolution the cluster column holds. Pass "
            f"--cluster-column (e.g. --cluster-column leiden_2.5)."
        )
    return [groupby if name is None else name for name in _ASSIGNMENT_COLUMNS]


def classify(tiles) -> tuple[str, list[str]]:
    """(verdict, sample) for a collection of tile names.

    Verdicts: "short" (none carry an extension — migratable), "done" (all do),
    "mixed" (some do, some do not).

    "mixed" is refused rather than half-migrated. It is the signature of a .h5
    written by a resume that straddled the fix, and the rows on each side of
    that boundary cannot be told apart from their names alone — so the file
    needs repackaging, not patching.
    """
    names = [_as_text(t) for t in tiles]
    with_suffix = [n for n in names if n.lower().endswith(_SUFFIX)]
    if not with_suffix:
        return "short", names[:5]
    if len(with_suffix) == len(names):
        return "done", names[:5]
    return "mixed", [n for n in names if not n.lower().endswith(_SUFFIX)][:5]


def _migrated(names: list[str]) -> list[str]:
    return [n if n.lower().endswith(_SUFFIX) else n + _SUFFIX for n in names]


def migrate_csv(csv_path: Path, commit: bool,
                cluster_column: str | None = None) -> dict:
    """Rewrite an assignments CSV's `tiles` column into a new file.

    Also restores a missing header row. A headerless CSV loses its first tile
    to pandas' header inference, so this is not cosmetic — reading one with
    default settings silently drops a row and names every column after that
    row's values.
    """
    recovered = detect_headerless(csv_path, cluster_column)
    if recovered:
        frame = pd.read_csv(csv_path, header=None, names=recovered)
    else:
        frame = pd.read_csv(csv_path)

    if "tiles" not in frame.columns:
        raise SystemExit(
            f"{csv_path} has no 'tiles' column; found {list(frame.columns)}.\n\n"
            f"If those look like data rather than column names, this CSV has no "
            f"header row and does not match the layout assign_hpc_clusters.py "
            f"writes (samples, slides, tiles, <cluster>, vote_margin, "
            f"neighbor_distance, hpc_reference)."
        )

    verdict, sample = classify(frame["tiles"])
    out_path = csv_path.with_name(f"{csv_path.stem}_tilenames.csv")
    # Slide count and tiles-per-slide, because this is the last place anyone
    # looks at the CSV before it becomes Knowledge Bank rows. An assignment
    # covering a fraction of its input is well-formed in every other respect —
    # right columns, right dtypes, no gaps — and produces per-slide proportions
    # computed from a fraction of each slide, which read as perfectly plausible.
    per_slide = frame.groupby(frame["slides"].astype(str)).size()
    report = {"path": str(csv_path), "rows": len(frame), "verdict": verdict,
              "sample": sample, "output": str(out_path), "written": False,
              "recovered_header": recovered,
              "slides": int(per_slide.size),
              "tiles_per_slide": (int(per_slide.min()), int(per_slide.max()))}
    # A missing header is on its own reason enough to rewrite, even if the tile
    # names are already correct.
    if verdict != "short" and not recovered:
        return report

    names = [_as_text(t) for t in frame["tiles"]]
    frame["tiles"] = _migrated(names)
    if commit:
        # Temp name then rename, so an interrupted run cannot leave a
        # half-written CSV sitting at a path something else will read.
        staging = out_path.with_name(out_path.name + ".partial")
        frame.to_csv(staging, index=False)
        check = pd.read_csv(staging)
        if len(check) != len(frame) or tiles_missing_suffix(check["tiles"]):
            staging.unlink(missing_ok=True)
            raise SystemExit(f"Verification of {staging} failed; nothing written.")
        staging.replace(out_path)
        report["written"] = True
    return report


def migrate_h5(h5_path: Path, commit: bool) -> dict:
    """Rewrite a .h5's `tiles` dataset in place.

    Works on both a packaged .h5 and a projections .h5 — the encoder copies
    `tiles` through unchanged, so they carry the same names and the same defect.

    The column is fixed-length bytes, so it cannot simply be assigned into: the
    stored width was sized for the short form and would truncate the suffix
    straight back off. The dataset is deleted and recreated at the wider dtype.
    """
    with h5py.File(h5_path, "r") as f:
        if "tiles" not in f:
            raise SystemExit(f"{h5_path} has no 'tiles' dataset.")
        names = [_as_text(t) for t in f["tiles"][:]]
        rows = len(names)
        other = {k: f[k].shape[0] for k in ("samples", "slides") if k in f}

    verdict, sample = classify(names)
    report = {"path": str(h5_path), "rows": rows, "verdict": verdict,
              "sample": sample, "written": False}
    if verdict != "short":
        return report

    migrated = _migrated(names)
    if commit:
        width = max(len(n.encode("utf-8")) for n in migrated)
        with h5py.File(h5_path, "r+") as f:
            del f["tiles"]
            f.create_dataset(
                "tiles",
                data=np.array([n.encode("utf-8") for n in migrated], dtype=f"S{width}"),
            )
        # Reopen and confirm, rather than trusting the write.
        with h5py.File(h5_path, "r") as f:
            after = f["tiles"][:]
            if len(after) != rows or tiles_missing_suffix(after):
                raise SystemExit(
                    f"{h5_path}: rewrite did not take — the tiles column still "
                    f"reads as unsuffixed. The file needs repackaging."
                )
            for name, count in other.items():
                if f[name].shape[0] != rows:
                    raise SystemExit(
                        f"{h5_path}: '{name}' has {count} rows but tiles now has "
                        f"{rows}. Do not use this file."
                    )
        report["written"] = True
    return report


def report(result: dict, commit: bool) -> None:
    print(f"\n{result['path']}")
    print(f"  rows        {result['rows']:,}")
    if result.get("slides") is not None:
        low, high = result["tiles_per_slide"]
        print(f"  slides      {result['slides']:,} "
              f"({low:,}-{high:,} tiles each)")
        print(f"              Check this against the projections .h5 this came "
              f"from. Stage 4 writes one row per embedding, so a count below "
              f"that means an incomplete assignment, and every per-slide "
              f"proportion built from it will be wrong but plausible.")
    if result.get("recovered_header"):
        print(f"  header      MISSING — recovered by position as "
              f"{result['recovered_header']}")
        print(f"              (without it pandas reads row 1 as the header, so the "
              f"first tile is lost and every column is misnamed)")
    if result["verdict"] == "done" and result.get("recovered_header"):
        print("  tile names  already carry the suffix; rewriting only to add the header")
    elif result["verdict"] == "done":
        print(f"  tile names  already carry the suffix (e.g. {result['sample'][0]!r}) "
              f"— nothing to do")
        return
    if result["verdict"] == "mixed":
        print(f"  tile names  MIXED — some carry the suffix and some do not; "
              f"unsuffixed e.g. {result['sample']}")
        print("\nRefusing to migrate: this is what a resume straddling the "
              "tile-name fix leaves behind, and the two halves cannot be told "
              "apart from their names. Repackage the dataset instead.")
        raise SystemExit(2)

    print(f"  tile names  short (e.g. {result['sample']})")
    print(f"  would become {[n + _SUFFIX for n in result['sample']]}")
    if result.get("output"):
        print(f"  output      {result['output']}")
    if commit and result["written"]:
        print("  written and verified.")
    elif not commit:
        print("\nDry run — nothing written. Re-run with --commit to apply.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="An assignments CSV from Stage 4.")
    source.add_argument("--h5", type=Path,
                        help="A packaged or projections .h5. Rewritten in place.")
    parser.add_argument("--commit", action="store_true",
                        help="Actually write. Without it this only reports.")
    parser.add_argument("--cluster-column", default=None,
                        help="Name of the cluster column, for a CSV with no header "
                             "row whose reference name does not imply the Leiden "
                             "resolution. e.g. leiden_2.5")
    args = parser.parse_args()

    target = args.csv or args.h5
    if not target.is_file():
        raise SystemExit(f"No such file: {target}")

    result = (migrate_csv(args.csv, args.commit, args.cluster_column) if args.csv
              else migrate_h5(args.h5, args.commit))
    report(result, args.commit)


if __name__ == "__main__":
    raise SystemExit(main())
