#!/usr/bin/env python3
"""Load a cluster-assignment CSV into the Knowledge Bank (tile_registry).

Stage 5, and the step that makes Stage 4's output visible: until a tile's
hpc_id is in tile_registry, the slide viewer's join
(tile_coordinates -> tile_registry -> hpc_dictionary) returns nothing and the
CSV is just a file on disk.

    tile_coordinates.slide_tile ──┐
                                  ├── tile_registry.hpc_id ──> hpc_dictionary
    assignment CSV (slides,tiles)─┘                            (pattern, malignant,
                                                                inflammation, ...)

Everything here is written around one fact: this is the only script in the
pipeline that mutates the shared KB, and its failure mode is not a crash. A
naming mismatch between the CSV and tile_registry silently updates nothing; a
partial match silently updates some tiles and leaves others carrying stale
cluster IDs from an earlier reference. Both look like success. So it refuses to
commit unless the match rate clears a threshold, it reports exactly what it
would change before changing it, and --dry-run is the default posture for
anything unfamiliar.

Writes five columns, all added by migrate_tile_registry_confidence.sql:
    hpc_id, hpc_vote_margin, hpc_neighbor_distance, hpc_assigned_at, hpc_reference

Usage:
    python load_hpc_assignments.py --csv DS_hpc_assignments.csv --dry-run
    python load_hpc_assignments.py --csv DS_hpc_assignments.csv --commit
    python load_hpc_assignments.py --csv DS_hpc_assignments.csv --commit --min-margin 0.25
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy import inspect as sqlalchemy_inspect

# Same defaults and env names as tile_server_v2_.py, so a shell configured for
# the server needs no extra setup here. Duplicated rather than imported: that
# module opens HDF5 handles and builds a FastAPI app at import time, which is a
# lot to drag in for a connection string.
DB_USER = os.getenv("DB_USER", "vpandya")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "hpl_kb")

# Columns assign_hpc_clusters.py writes besides the cluster ID itself.
_META = ("samples", "slides", "tiles")
_CONFIDENCE = ("vote_margin", "neighbor_distance")

# Below this share of CSV rows matching tile_registry, refuse to commit. The
# number this guards against is not 0% — a total mismatch is obvious. It is the
# 3% that means the CSV and the registry disagree about slide naming for all but
# a handful of tiles, which reads as "it worked" in any summary line.
_MIN_MATCH_RATE = 0.95

# Tiles per lookup query. Large enough to keep round trips down, small enough
# to stay well inside any driver's parameter limit.
_LOOKUP_CHUNK = 10_000


def make_engine():
    return create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        pool_pre_ping=True,
    )


def read_assignments(csv_path: Path) -> tuple[pd.DataFrame, str]:
    """The CSV plus the name of its cluster column.

    The cluster column is named for the reference's groupby ('leiden_2.5'), so
    it is found by elimination rather than by name — the same rule the server's
    validator uses, and for the same reason: hardcoding it breaks the moment the
    reference changes resolution.
    """
    frame = pd.read_csv(csv_path)
    known = set(_META) | set(_CONFIDENCE) | {"hpc_reference"}
    missing = [c for c in (*_META, *_CONFIDENCE) if c not in frame.columns]
    if missing:
        raise SystemExit(
            f"{csv_path} is missing {missing}. Expected the output of "
            f"assign_hpc_clusters.py; found columns {list(frame.columns)}."
        )
    cluster_columns = [c for c in frame.columns if c not in known]
    if len(cluster_columns) != 1:
        raise SystemExit(
            f"Could not identify the cluster column in {csv_path}: "
            f"{cluster_columns or 'none'} left after the known ones. "
            f"Columns: {list(frame.columns)}."
        )
    if frame.empty:
        raise SystemExit(f"{csv_path} holds no assignments.")

    # tile_coordinates.slide_tile is "<slides>_<tiles>", e.g.
    # TCGA-55-7574-01Z-00-DX1_18_15.jpeg. The server joins case-insensitively,
    # so match its UPPER() rather than hoping the two agree on case.
    frame["slide_tile"] = (
        frame["slides"].astype(str).str.strip()
        + "_"
        + frame["tiles"].astype(str).str.strip()
    ).str.upper()
    return frame, cluster_columns[0]


def inspect(engine, frame: pd.DataFrame, cluster_column: str, min_margin: float = 0.0) -> dict:
    """What loading this CSV would do, computed without changing anything.

    Deliberately a separate pass rather than a count taken during the write:
    the point is to be able to look before committing, and a preview derived
    from the write path would only exist after the write.

    min_margin previews compute_profiles()'s own exclusion: leave-one-out
    validation against the reference put tiles below 0.1 vote_margin at 57%
    correct and 0.1-0.25 at 76%, against 92%+ once margin clears 0.25 — so a
    tile below threshold is disproportionately likely to be wrong, and
    "excluded_from_aggregates" is how many of those exist in this CSV before
    anything is decided.
    """
    tiles = frame["slide_tile"].tolist()
    # Looked up in chunks with an expanding IN rather than one ANY(array): a
    # single parameter carrying 500k tiles is both a portability problem and a
    # planner one, and this has to work for a whole dataset, not just a subset.
    lookup = text("""
        SELECT UPPER(slide_tile) AS slide_tile, hpc_id, hpc_reference
        FROM tile_registry
        WHERE UPPER(slide_tile) IN :tiles
    """).bindparams(bindparam("tiles", expanding=True))

    pieces = []
    with engine.connect() as conn:
        for start in range(0, len(tiles), _LOOKUP_CHUNK):
            pieces.append(pd.read_sql(
                lookup, conn, params={"tiles": tiles[start:start + _LOOKUP_CHUNK]}
            ))
        clusters = pd.read_sql(
            text("SELECT hpc_id FROM hpc_dictionary"), conn
        )["hpc_id"].astype(str).str.strip().tolist()
    present = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["slide_tile", "hpc_id", "hpc_reference"]
    )

    matched = set(present["slide_tile"])
    assigned = frame[cluster_column].astype(str).str.strip()

    already = present[present["hpc_id"].notna()]
    other_reference = already[
        already["hpc_reference"].notna()
        & (already["hpc_reference"] != frame["hpc_reference"].iloc[0])
    ] if "hpc_reference" in present.columns else already.iloc[0:0]

    return {
        "rows": len(frame),
        "matched": len(matched),
        "unmatched": len(frame) - len(matched),
        "unmatched_examples": [t for t in tiles if t not in matched][:5],
        # A cluster ID with no hpc_dictionary row joins to NULL in the viewer:
        # the tile gets a cluster but no pattern, malignancy or inflammation.
        "unknown_clusters": sorted(set(assigned) - set(clusters)),
        "known_clusters": len(clusters),
        "overwriting": int(len(already)),
        "overwriting_other_reference": int(len(other_reference)),
        "distribution": assigned.value_counts().head(5).to_dict(),
        "low_margin": int((frame["vote_margin"] < 0.1).sum()),
        "reference": str(frame["hpc_reference"].iloc[0]),
        "min_margin": min_margin,
        "excluded_from_aggregates": int((frame["vote_margin"] < min_margin).sum()) if min_margin > 0 else 0,
    }


def load(engine, frame: pd.DataFrame, cluster_column: str, *, batch: int = 5000,
         profiles: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> int:
    """Write the assignments. One transaction: a half-loaded registry, with some
    tiles on the new reference and some on the old, is not a state anything
    downstream can interpret."""
    now = datetime.now(timezone.utc)
    # Zipped columns rather than itertuples/getattr: itertuples renames anything
    # that is not a valid Python identifier, and the cluster column is named for
    # the reference's groupby — "leiden_2.5" — so the dot turns it into a
    # positional alias and getattr raises. Which is to say the obvious way to
    # write this loop fails on every real reference and passes on any test that
    # invents a tidier column name.
    records = [
        {
            "slide_tile": slide_tile,
            "hpc_id": str(cluster).strip(),
            "margin": float(margin),
            "distance": None if pd.isna(distance) else float(distance),
            "reference": str(reference),
            "assigned_at": now,
        }
        for slide_tile, cluster, margin, distance, reference in zip(
            frame["slide_tile"],
            frame[cluster_column],
            frame["vote_margin"],
            frame["neighbor_distance"],
            frame["hpc_reference"],
        )
    ]

    statement = text("""
        UPDATE tile_registry
        SET hpc_id = :hpc_id,
            hpc_vote_margin = :margin,
            hpc_neighbor_distance = :distance,
            hpc_reference = :reference,
            hpc_assigned_at = :assigned_at
        WHERE UPPER(slide_tile) = :slide_tile
    """)

    updated = 0
    # One transaction covering the tiles and both aggregates. Splitting them
    # would allow a registry whose per-tile clusters and per-slide proportions
    # came from different runs, which is worse than either being stale: nothing
    # downstream can tell that has happened.
    with engine.begin() as conn:
        for start in range(0, len(records), batch):
            chunk = records[start:start + batch]
            result = conn.execute(statement, chunk)
            updated += result.rowcount if result.rowcount is not None else 0
            print(f"  tiles {min(start + batch, len(records)):,}/{len(records):,}", flush=True)

        if profiles is not None:
            proportions, summary = profiles
            written = replace_profiles(conn, proportions, summary)
            for table, count in written.items():
                print(f"  {table}: {count:,} rows", flush=True)
    return updated


def compute_profiles(frame: pd.DataFrame, cluster_column: str,
                     cancer_type: str | None, min_margin: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two per-slide aggregates, from the same CSV the tiles came from.

    Definitions taken from the scripts that first populated these tables
    (Filling_out_kb.ipynb and csv_files/For_KB/filling_out_kb_hs.py) so the rows
    this writes are the same shape as the rows already there:

      hpl_profile_proportion  per (samples, slides, hpc_id): that cluster's share
                              of the slide's tiles, summing to 1 per slide.
      hpl_profile_summary     per (samples, slides): tile count and modal cluster.

    They matter because they are what the chatbot and the HPC panels read — not
    tile_registry. Loading per-tile assignments without refreshing these leaves
    the UI showing new clusters per tile and old proportions per slide, with
    nothing to indicate the two disagree.

    min_margin drops tiles below that vote_margin before either aggregate is
    computed. tile_registry is untouched either way — every tile keeps its own
    hpc_id and margin regardless — this only decides what counts toward the
    per-slide numbers people actually read. total_tiles shrinks along with it
    rather than staying at the slide's full tile count: the alternative, a
    total_tiles that counts tiles the proportions and dominant_hpc never saw,
    would report a number next to a composition it does not match.
    """
    work = frame[["samples", "slides", cluster_column, "vote_margin"]].copy()
    work.columns = ["samples", "slides", "hpc_id", "vote_margin"]
    if min_margin > 0:
        work = work[work["vote_margin"] >= min_margin]
    work["hpc_id"] = work["hpc_id"].astype(str).str.strip()

    proportions = work.groupby(["samples", "slides", "hpc_id"], as_index=False).size()
    proportions["proportion"] = proportions.groupby(["samples", "slides"])["size"].transform(
        lambda x: x / x.sum()
    )
    proportions = proportions.drop(columns=["size"])

    summary = work.groupby(["samples", "slides"], as_index=False).agg(
        total_tiles=("hpc_id", "count"),
        dominant_hpc=("hpc_id", lambda x: x.value_counts().idxmax()),
    )
    if cancer_type is not None:
        summary["cancer_type"] = cancer_type
    return proportions, summary


def _existing_columns(conn, table: str) -> set[str]:
    """Columns the live table actually has.

    Inspected rather than assumed: these tables predate this script and were
    filled by hand from notebooks, so an insert naming a column that is not
    there fails the whole transaction — including the tile_registry update that
    had nothing to do with it.
    """
    return {c["name"] for c in sqlalchemy_inspect(conn).get_columns(table)}


def replace_profiles(conn, proportions: pd.DataFrame, summary: pd.DataFrame) -> dict:
    """Swap in the aggregates for just the slides being loaded.

    Scoped by slide, not a whole-table rebuild: loading ten slides must not
    delete the proportions for every other slide in the KB. Delete-then-insert
    rather than upsert because a slide's cluster set changes between references —
    a cluster that no longer appears must lose its row, and an upsert would leave
    it behind at its old proportion.
    """
    slides = sorted(set(summary["slides"].astype(str)))
    written = {}
    for table, frame in (("hpl_profile_proportion", proportions),
                         ("hpl_profile_summary", summary)):
        columns = _existing_columns(conn, table)
        usable = [c for c in frame.columns if c in columns]
        skipped = [c for c in frame.columns if c not in columns]
        if skipped:
            print(f"  {table}: no column(s) {skipped}; not writing them", file=sys.stderr)

        conn.execute(
            text(f"DELETE FROM {table} WHERE TRIM(slides) IN :slides").bindparams(
                bindparam("slides", expanding=True)
            ),
            {"slides": slides},
        )
        placeholders = ", ".join(f":{c}" for c in usable)
        conn.execute(
            text(f"INSERT INTO {table} ({', '.join(usable)}) VALUES ({placeholders})"),
            frame[usable].to_dict("records"),
        )
        written[table] = len(frame)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=Path, required=True,
                        help="Output of assign_hpc_clusters.py.")
    parser.add_argument("--commit", action="store_true",
                        help="Actually write. Without it this only reports.")
    parser.add_argument("--min-match-rate", type=float, default=_MIN_MATCH_RATE,
                        help="Refuse to commit below this share of CSV rows matching "
                             "tile_registry.")
    parser.add_argument("--cancer-type", type=str, default=None,
                        help="Value for hpl_profile_summary.cancer_type (e.g. LUAD). "
                             "Omitted leaves it unset rather than guessing.")
    parser.add_argument("--skip-profiles", action="store_true",
                        help="Only update tile_registry. The per-slide aggregates the "
                             "chatbot and HPC panels read will then disagree with it.")
    parser.add_argument("--allow-unknown-clusters", action="store_true",
                        help="Load cluster IDs that have no hpc_dictionary row. They "
                             "will show in the viewer with no annotations.")
    parser.add_argument("--min-margin", type=float, default=0.0,
                        help="Exclude tiles below this vote_margin from "
                             "hpl_profile_proportion/summary. tile_registry keeps every "
                             "tile's hpc_id and margin regardless of this flag — it only "
                             "changes what counts toward the per-slide aggregates. "
                             "0 (default) excludes nothing.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame, cluster_column = read_assignments(args.csv)
    engine = make_engine()
    report = inspect(engine, frame, cluster_column, min_margin=args.min_margin)

    print(f"CSV            : {args.csv}")
    print(f"  rows         {report['rows']:,}   cluster column '{cluster_column}'")
    print(f"  reference    {report['reference']}")
    print(f"  matched      {report['matched']:,} of {report['rows']:,} "
          f"({report['matched'] / report['rows'] * 100:.1f}%) in tile_registry")
    if report["unmatched"]:
        print(f"  unmatched    {report['unmatched']:,}, e.g. {report['unmatched_examples']}")
    print(f"  overwriting  {report['overwriting']:,} tiles that already have a cluster"
          + (f" ({report['overwriting_other_reference']:,} from a different reference)"
             if report["overwriting_other_reference"] else ""))
    print(f"  clusters     {report['known_clusters']} in hpc_dictionary; "
          f"largest here: " + ", ".join(f"{k}={v:,}" for k, v in report["distribution"].items()))
    print(f"  low margin   {report['low_margin']:,} tiles below 0.1")
    if args.min_margin > 0:
        print(f"  min margin   {args.min_margin} — excludes "
              f"{report['excluded_from_aggregates']:,} tile(s) from the aggregates below")

    profiles = None
    if not args.skip_profiles:
        profiles = compute_profiles(frame, cluster_column, args.cancer_type,
                                    min_margin=args.min_margin)
        proportions, summary = profiles
        print(f"  aggregates   {len(proportions):,} proportion rows and "
              f"{len(summary):,} summary rows across "
              f"{summary['slides'].nunique()} slide(s)")
        if args.cancer_type is None:
            print("               (cancer_type not set — pass --cancer-type to fill it)")

    match_rate = report["matched"] / report["rows"]
    problems = []
    if match_rate < args.min_match_rate:
        problems.append(
            f"only {match_rate * 100:.1f}% of rows match tile_registry (need "
            f"{args.min_match_rate * 100:.0f}%). The usual cause is a slide-naming "
            f"difference between the .h5 and the registry, not missing tiles — "
            f"check the unmatched examples above against "
            f"`SELECT slide_tile FROM tile_registry LIMIT 5`."
        )
    if report["unknown_clusters"] and not args.allow_unknown_clusters:
        problems.append(
            f"{len(report['unknown_clusters'])} cluster ID(s) have no hpc_dictionary "
            f"row: {report['unknown_clusters'][:10]}. Those tiles would show a cluster "
            f"with no pattern or malignancy annotation. Pass "
            f"--allow-unknown-clusters if that is intended."
        )

    if problems:
        print("\nRefusing to load:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        raise SystemExit(1)

    if not args.commit:
        print("\nDry run — nothing written. Re-run with --commit to load.")
        return

    print(f"\nLoading into {DB_NAME}.tile_registry ...")
    updated = load(engine, frame, cluster_column, profiles=profiles)
    print(f"Updated {updated:,} rows.")
    if updated != report["matched"]:
        print(
            f"WARNING: updated {updated:,} but expected {report['matched']:,}. The "
            f"registry changed between the preview and the write.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
