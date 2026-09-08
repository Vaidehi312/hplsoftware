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

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy import inspect as sqlalchemy_inspect

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_record import record_run  # noqa: E402
from slide_naming import (  # noqa: E402
    make_slide_tile_series,
    normalize_tile_names,
    tile_name_verdict,
)

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

    # A CSV produced from a .h5 packaged before make_hpl_hdf5.py started storing
    # the ".jpeg" suffix carries the short form, which joins neither
    # tile_registry nor tile_coordinates and merges zero rows against Kai's
    # reference CSV. Nothing about it needs recomputing — the cluster IDs and
    # margins are correct, only the label is short — so the suffix is appended
    # here and the count reported, rather than the load being refused.
    #
    # Mixed still refuses: some names suffixed and some not is what a resume
    # straddling the fix leaves behind, the two sides are indistinguishable by
    # name, and guessing would attach correct cluster IDs to the wrong tiles.
    # A confidence column that is not numeric. One malformed line in a
    # 7.8M-row CSV — a header repeated by a concatenation, a torn write — turns
    # the whole column to object dtype, and every comparison against it
    # afterwards either raises deep in pandas or silently misbehaves. Refused
    # with the offending values rather than coerced: a file with a stray row in
    # it is a file whose row count nobody should trust, and dropping the row
    # quietly would leave a cohort short by an unknown amount.
    for column in _CONFIDENCE:
        if column not in frame.columns:
            continue
        coerced = pd.to_numeric(frame[column], errors="coerce")
        bad = coerced.isna() & frame[column].notna()
        if bad.any():
            examples = frame.loc[bad, column].astype(str).head(5).tolist()
            raise SystemExit(
                f"{csv_path} has {int(bad.sum()):,} row(s) whose {column} is not "
                f"a number, e.g. {examples}. That is usually a header repeated "
                f"mid-file by a concatenation, or a torn write from an "
                f"interrupted job.\n\n"
                f"Check the file's row count against the projections .h5 before "
                f"loading it — if shards were merged by hand, re-merge them with "
                f"merge_assignment_shards.py, which checks the total against the "
                f"source rather than believing the parts."
            )
        frame[column] = coerced

    verdict = tile_name_verdict(frame["tiles"])
    if verdict == "mixed":
        raise SystemExit(
            f"{csv_path} has SOME tile names with a file extension and some "
            f"without (e.g. {frame['tiles'].iloc[0]!r}). That is what an "
            f"assignment built from a half-migrated .h5 looks like, and the two "
            f"forms cannot be told apart by name. Repackage and re-assign the "
            f"dataset rather than loading this."
        )
    frame["tiles"], renamed = normalize_tile_names(frame["tiles"])

    # tile_coordinates.slide_tile is "<slides>_<tiles>" upper-cased, e.g.
    # TCGA-55-7574-01Z-00-DX1_18_15.JPEG. Built by the shared helper so this and
    # the dataset-registration step cannot drift apart on the key they join on.
    frame["slide_tile"] = make_slide_tile_series(frame["slides"], frame["tiles"])
    # On the frame rather than in the return tuple, which every caller and test
    # already unpacks as exactly (frame, cluster_column).
    frame.attrs["tile_names_normalized"] = renamed
    return frame, cluster_columns[0]


#: Leave-one-out accuracy per vote_margin band, measured on the production
#: reference (CLASSIFIER_TUNING_2026-08-13.md §7: 20,000 tiles, k=10,
#: distance-weighted). (upper edge, share correct) — the band is [previous
#: edge, this edge).
#:
#: These are the only numbers that turn a margin into an accuracy, and they were
#: measured *within LATTICeA*, the cohort the 71 HPCs were defined on. Applying
#: them to another cohort assumes a margin of 0.3 means there what it means
#: here, which is exactly the cohort-shift question that cannot be settled
#: without labels. So anything derived from them is an estimate, and says so.
_MARGIN_ACCURACY_BANDS = (
    (0.10, 0.617),
    (0.25, 0.812),
    (0.50, 0.955),
    (0.75, 0.995),
    (1.01, 1.000),
)

_MARGIN_CUTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50)


def margin_tradeoff(frame: pd.DataFrame, cuts=_MARGIN_CUTS) -> list[dict]:
    """For each candidate min_margin: tiles kept, and their expected accuracy.

    The question this answers is the one actually being decided at Stage 6 — how
    many tiles a confidence floor costs, and what it buys — and it can only be
    answered from the cohort's own margin distribution. Everything else about
    picking a threshold is guesswork against a distribution nobody has looked
    at.

    Expected accuracy is each kept tile's band accuracy, averaged. It is an
    estimate under LATTICeA calibration (see _MARGIN_ACCURACY_BANDS), not a
    measurement of this cohort — which has no labels to measure against.
    """
    margins = pd.to_numeric(frame["vote_margin"], errors="coerce").to_numpy(dtype=float)
    edges = np.array([0.0, *(edge for edge, _ in _MARGIN_ACCURACY_BANDS)])
    accuracies = np.array([acc for _, acc in _MARGIN_ACCURACY_BANDS])
    per_tile = accuracies[np.clip(np.digitize(margins, edges) - 1,
                                  0, len(accuracies) - 1)]

    total = len(margins)
    rows = []
    for cut in cuts:
        kept = margins >= cut
        n = int(kept.sum())
        rows.append({
            "min_margin": float(cut),
            "tiles_kept": n,
            "share_kept": (n / total) if total else 0.0,
            "expected_accuracy": float(per_tile[kept].mean()) if n else 0.0,
        })
    return rows


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
        # How many tile names had ".jpeg" appended on the way in. Zero for a CSV
        # written after the packaging fix; non-zero says the match rate below
        # was only reachable because of that correction.
        "tile_names_normalized": int(frame.attrs.get("tile_names_normalized", 0)),
        # What each candidate threshold would cost and buy, from this cohort's
        # own margins. Computed here so the UI and the CLI show the same table.
        "margin_tradeoff": margin_tradeoff(frame),
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

    Order matters, and not in the obvious way. In Postgres
    hpl_profile_proportion carries
    FOREIGN KEY (samples, slides) REFERENCES hpl_profile_summary ON DELETE CASCADE,
    so this cannot be a per-table delete-then-insert loop:

      * inserting a proportion row for a slide with no summary row yet — every
        slide on a cohort's first load — violates the FK and rolls back the whole
        transaction, taking the tile_registry update with it;
      * inserting proportions first for a slide that *does* exist, then deleting
        its summary row, cascades away the proportions just written, leaving a
        summary row with no proportions.

    So: both deletes first (child, then parent), then both inserts (parent, then
    child). The SQLite tests cannot see this — their fixtures create the two
    tables without the foreign key — so it is asserted against a real FK in
    test_kb_load.py rather than left to the schema.
    """
    # UPPER as well as TRIM: migrate_indexes.sql normalised the live columns to
    # UPPER(TRIM(...)), so matching on TRIM alone finds nothing for any slide
    # whose CSV casing differs. That failure is silent in the worst way — the
    # DELETE removes zero rows, the INSERT still runs, and the slide ends up with
    # two sets of aggregates that both look plausible.
    slides = sorted({s.strip().upper() for s in summary["slides"].astype(str)})
    bind = {"slides": slides}

    # dataset_id is NOT NULL on both aggregate tables in the live database, and
    # compute_profiles() cannot know it — the assignment CSV does not carry a
    # cohort. Resolved here from tile_registry, which registration filled, so
    # the aggregates are scoped to exactly the cohort their tiles belong to
    # rather than to whatever the caller believed.
    #
    # This was a latent break, not a new requirement: before it, the INSERT
    # below raised NotNullViolation on any real Postgres, and because load() is
    # one transaction the rollback took the tile_registry update with it. It had
    # never surfaced because no cohort had ever got past the 95% match gate to
    # reach this line, and the SQLite test fixtures declare dataset_id nullable.
    dataset_by_slide = {}
    if "dataset_id" in _existing_columns(conn, "hpl_profile_summary"):
        rows = conn.execute(
            text("SELECT UPPER(TRIM(slides)) AS s, dataset_id FROM tile_registry "
                 "WHERE UPPER(TRIM(slides)) IN :slides AND dataset_id IS NOT NULL "
                 "GROUP BY 1, 2").bindparams(bindparam("slides", expanding=True)),
            bind,
        ).fetchall()
        for slide, dataset_id in rows:
            # A slide already claimed by two cohorts is refused at registration,
            # so this keeps the first and does not invent a resolution.
            dataset_by_slide.setdefault(slide, dataset_id)

        missing = [s for s in slides if s not in dataset_by_slide]
        if missing:
            raise SystemExit(
                f"{len(missing)} slide(s) have no dataset_id in tile_registry "
                f"(e.g. {missing[:3]}). The aggregates cannot be scoped to a "
                f"cohort without one. Register the dataset first — "
                f"register_dataset.py, or the 'Register in the Knowledge Bank' "
                f"step in the UI."
            )

        for frame in (summary, proportions):
            frame["dataset_id"] = (frame["slides"].astype(str).str.strip().str.upper()
                                   .map(dataset_by_slide))

    # Refuse to delete another cohort's aggregates.
    #
    # These deletes are scoped by slide NAME, not by cohort, because that is
    # what the rewrite has to match. But a slide name is not unique across
    # cohorts, so on a shared database this could silently remove the rows a
    # different dataset_id owns — turning "load a new cohort" into "quietly
    # replace an old one", which is the exact failure this codebase is written
    # against.
    #
    # It cannot simply leave them, either: hpl_profile_summary is UNIQUE on
    # (samples, slides) with no dataset_id, so two cohorts cannot both hold a
    # row for the same slide even in principle. Adding would hit the
    # constraint. So the only honest options are delete-and-replace or refuse,
    # and refusing is the one that never destroys data nobody asked to touch.
    if dataset_by_slide:
        ours = set(dataset_by_slide.values())
        conflicting = conn.execute(
            text("SELECT DISTINCT UPPER(TRIM(slides)), dataset_id "
                 "FROM hpl_profile_summary "
                 "WHERE UPPER(TRIM(slides)) IN :slides "
                 "  AND dataset_id IS NOT NULL AND dataset_id NOT IN :ours"
                 ).bindparams(bindparam("slides", expanding=True),
                              bindparam("ours", expanding=True)),
            {**bind, "ours": sorted(ours)},
        ).fetchall()
        if conflicting:
            listed = ", ".join(f"{s} (owned by {d})" for s, d in conflicting[:5])
            raise SystemExit(
                f"{len(conflicting)} slide(s) already have aggregates belonging to a "
                f"different cohort: {listed}. Loading would delete them, and "
                f"hpl_profile_summary's UNIQUE (samples, slides) has no dataset_id, "
                f"so both cannot coexist. Resolve which cohort owns these slides "
                f"before loading — this will not overwrite another dataset's rows."
            )

    def _delete(table: str) -> None:
        conn.execute(
            text(f"DELETE FROM {table} WHERE UPPER(TRIM(slides)) IN :slides").bindparams(
                bindparam("slides", expanding=True)
            ),
            bind,
        )

    def _insert(table: str, frame: pd.DataFrame) -> int:
        columns = _existing_columns(conn, table)
        usable = [c for c in frame.columns if c in columns]
        skipped = [c for c in frame.columns if c not in columns]
        if skipped:
            print(f"  {table}: no column(s) {skipped}; not writing them", file=sys.stderr)
        placeholders = ", ".join(f":{c}" for c in usable)
        conn.execute(
            text(f"INSERT INTO {table} ({', '.join(usable)}) VALUES ({placeholders})"),
            frame[usable].to_dict("records"),
        )
        return len(frame)

    _delete("hpl_profile_proportion")
    _delete("hpl_profile_summary")
    written = {
        "hpl_profile_summary": _insert("hpl_profile_summary", summary),
        "hpl_profile_proportion": _insert("hpl_profile_proportion", proportions),
    }
    membership = replace_slide_membership(conn, proportions)
    if membership is not None:
        written["slide_hpc_membership"] = membership
    return written


def replace_slide_membership(conn, proportions: pd.DataFrame):
    """Refresh slide_hpc_membership for the slides being loaded.

    Which HPCs appear on which slide — derived from the same assignment as the
    proportions, and refreshed with them, because it is read alongside them.
    The reader is not obvious: app/hpc_chat_handlers_v23.py:334 enumerates
    every table in the database with `insp.get_table_names()`, keeps any that
    has an hpc_id or dominant_hpc column, skipping only hpc_dictionary and
    h_latent_vectors, and renders up to five matching rows straight to the
    user. So this table is answered out of the chatbot without ever being named
    in a query — which is why a grep for it finds nothing and why it had been
    left stale, showing 19,493 rows from cohorts nobody was asking about.

    Derived from `proportions` rather than from the raw assignment so that
    min_margin applies here too. A cluster excluded from a slide's proportions
    but still listed as present would let the chatbot report a slide as
    containing an HPC the aggregate table has no row for, and the two are shown
    side by side.

    Returns None when the table is absent — it has no CREATE TABLE in git older
    than migrate_kb_base_tables.sql, so a database predating that has no such
    table and this must not fail the load.

    Scoped by slide_id and not by dataset_id, because the table has no
    dataset_id column: two cohorts holding the same slide id share these rows.
    Recorded as a property of the schema rather than worked around here.
    """
    if not sqlalchemy_inspect(conn).has_table("slide_hpc_membership"):
        print("  slide_hpc_membership: table absent; not writing it", file=sys.stderr)
        return None

    pairs = (proportions[["slides", "hpc_id"]]
             .dropna()
             .drop_duplicates())
    slides = sorted({str(s).strip().upper() for s in pairs["slides"]})
    if not slides:
        return 0

    # compute_profiles() carries hpc_id as a string, because the cluster column
    # is named for the reference's groupby and its values arrive as whatever the
    # CSV held — "3" from an int column, "3.0" from a float one. This table's
    # column is integer, so int("3.0") would raise and take the tile_registry
    # update down with it.
    records = []
    for slide, hpc in zip(pairs["slides"], pairs["hpc_id"]):
        try:
            cluster = int(float(str(hpc).strip()))
        except (TypeError, ValueError):
            # All or nothing: a membership missing the clusters that happen not
            # to be numeric is a table that looks complete and under-reports,
            # which is worse than one that was not refreshed and says so.
            print(f"  slide_hpc_membership: cluster id {hpc!r} is not an integer; "
                  f"not writing this table", file=sys.stderr)
            return None
        records.append({"slide_id": str(slide).strip().upper(), "hpc_id": cluster})

    conn.execute(
        text("DELETE FROM slide_hpc_membership WHERE UPPER(TRIM(slide_id)) IN :slides")
        .bindparams(bindparam("slides", expanding=True)),
        {"slides": slides},
    )
    conn.execute(
        text("INSERT INTO slide_hpc_membership (slide_id, hpc_id) "
             "VALUES (:slide_id, :hpc_id)"),
        records,
    )
    return len(records)


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
    # See the same flags on register_dataset.py: when this runs as the Slurm
    # job the server submits, the job is what knows whether the write committed.
    parser.add_argument("--record-run", default=None, metavar="SUBMISSION_ID",
                        help="Record the outcome against this run in "
                             "slurm_dataset_runs.")
    parser.add_argument("--record-run-db", default=None, metavar="DBNAME",
                        help="Database holding slurm_dataset_runs. Run tracking "
                             "stays in production whichever Knowledge Bank the "
                             "rows go to, so this is separate from DB_NAME.")
    parser.add_argument("--record-kb-target", default=None,
                        help="Recorded as kb_load_kb_target, so the run says "
                             "which Knowledge Bank it filled.")
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
    if report.get("tile_names_normalized"):
        print(f"  tile names   .jpeg appended to "
              f"{report['tile_names_normalized']:,} name(s) so they match the "
              f"Knowledge Bank's '18_15.jpeg' form; the CSV on disk still holds "
              f"the short form")
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

    if report.get("margin_tradeoff"):
        print("\n  what a --min-margin would cost and buy, on this cohort's own "
              "margins:")
        print(f"    {'cut':>5} {'tiles kept':>14} {'%kept':>7} "
              f"{'expected accuracy':>19}")
        for row in report["margin_tradeoff"]:
            print(f"    {row['min_margin']:>5.2f} {row['tiles_kept']:>14,} "
                  f"{row['share_kept'] * 100:>6.1f}% "
                  f"{row['expected_accuracy'] * 100:>18.2f}%")
        print("    (expected accuracy is this cohort's margin distribution "
              "weighted by leave-one-out accuracy measured on the reference — an "
              "estimate, not a measurement of this cohort)")
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
    try:
        updated = load(engine, frame, cluster_column, profiles=profiles)
    except BaseException as e:
        if args.record_run and args.record_run_db:
            record_run(args.record_run_db, args.record_run,
                       kb_load_error=f"{type(e).__name__}: {e}"[:2000])
        raise
    print(f"Updated {updated:,} rows.")
    if args.record_run and args.record_run_db:
        record_run(
            args.record_run_db, args.record_run,
            kb_load_done=True,
            kb_load_at=datetime.now(timezone.utc),
            kb_load_rows=int(updated),
            kb_load_reference=str(frame["hpc_reference"].iloc[0]),
            kb_load_error=None,
            **({"kb_load_kb_target": args.record_kb_target}
               if args.record_kb_target else {}),
        )
    if updated != report["matched"]:
        print(
            f"WARNING: updated {updated:,} but expected {report['matched']:,}. The "
            f"registry changed between the preview and the write.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
