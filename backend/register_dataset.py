#!/usr/bin/env python3
"""Register a new dataset's tiles into the Knowledge Bank, before Stage 5.

Stage 5 (load_hpc_assignments.py) only ever UPDATEs tile_registry.hpc_id — it
never creates rows. For a dataset that has never touched the KB before, that
means Stage 5's match rate is 0% by construction: there is nothing there to
UPDATE. This is that missing step. It creates the identity rows —
tile_coordinates and tile_registry, with no hpc_id yet — from what Stages 1–2
already wrote to disk. Stage 5 runs unchanged after this and fills in hpc_id.

    Stage 1 metadata CSVs  ──┐
                             ├─► register_dataset.py ─► tile_coordinates
    packaged .h5 (Stage 2) ──┘                          tile_registry (no hpc_id)
                                                              │
                                                         Stage 5 UPDATEs hpc_id ─►
                                                         tile_registry (complete)

image_index (tile_registry) / h5_index (tile_coordinates) is the tile's row
position in the packaged .h5 — the actual array index, not anything derived
from the metadata CSV, because packaging order is not guaranteed stable across
a re-package (see make_hpl_hdf5.py). Reading it back from the file is the only
way to get this right.

Everything is scoped by --dataset-id and refuses to touch another cohort's
rows. Reusing this dataset_id to re-register (the planned path once the full
14,044-slide Radiogenomics run replaces this 10-slide one) requires --replace,
which is still restricted to WHERE dataset_id = the one given — a slide_tile
collision with a DIFFERENT dataset_id is refused as an error, not silently
reassigned, since that would mean two cohorts claiming the same tile.

Usage:
    python register_dataset.py --h5 packaged.h5 --tile-dir /path/to/processed_tiles \\
        --tile-dataset-name Radiogenomics --dataset-id RADIOGENOMICS
    python register_dataset.py --h5 ... --tile-dir ... --tile-dataset-name ... \\
        --dataset-id RADIOGENOMICS --commit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy import inspect as sqlalchemy_inspect

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_hpc_assignments import _LOOKUP_CHUNK, make_engine  # noqa: E402
from slide_naming import make_slide_tile_series, tiles_missing_suffix  # noqa: E402
from tile_metadata import read_tile_metadata, tile_metadata_path  # noqa: E402

_TILE_COORDINATES_COLUMNS = (
    "slides", "tiles", "slide_tile", "col", "row",
    "x_5x", "y_5x", "x_native", "y_native", "h5_index", "dataset_id",
)
_TILE_REGISTRY_COLUMNS = (
    "samples", "slides", "tiles", "slide_tile", "image_index",
    "h5_source_path", "dataset_id",
)


def read_h5_identity(h5_path: Path) -> pd.DataFrame:
    """samples/slides/tiles plus each row's actual position in the .h5.

    Position is read from the file, not computed, because it is the one thing
    nothing else can reconstruct: make_hpl_hdf5.py's own docstring notes
    packaging order is not guaranteed stable across a re-package.
    """
    with h5py.File(h5_path, "r") as f:
        for name in ("samples", "slides", "tiles"):
            if name not in f:
                raise SystemExit(f"{h5_path} has no '{name}' dataset — not a "
                                 f"packaged .h5 from make_hpl_hdf5.py.")
        n = f["tiles"].shape[0]
        if n == 0:
            raise SystemExit(f"{h5_path} holds zero tiles.")
        samples = [v.decode("utf-8", "replace") for v in f["samples"][:]]
        slides = [v.decode("utf-8", "replace") for v in f["slides"][:]]
        tiles_raw = f["tiles"][:]

    if tiles_missing_suffix(tiles_raw):
        raise SystemExit(
            f"{h5_path} has tile names without a file extension (e.g. "
            f"{tiles_raw[0]!r}). This .h5 was packaged before the tile-name fix. "
            f"Migrate it first: python migrate_tile_names.py --h5 {h5_path} --commit"
        )
    tiles = [v.decode("utf-8", "replace") for v in tiles_raw]

    frame = pd.DataFrame({
        "samples": samples, "slides": slides, "tiles": tiles,
        "image_index": np.arange(n, dtype=np.int64),
    })
    frame["slide_tile"] = make_slide_tile_series(frame["slides"], frame["tiles"])
    return frame


def read_tile_coordinates(tile_dir: Path, tile_dataset_name: str,
                          slide_ids) -> tuple[pd.DataFrame, list[str]]:
    """Per-tile coordinates from Stage 1's metadata CSVs, for exactly the
    slides present in the .h5.

    Returns (frame, missing_slides) rather than raising on a missing slide:
    partial coverage is reported and left for the caller to decide about,
    since refusing the whole registration for one bad slide out of thousands
    would be worse than the CLAUDE.md-preferred "loud refusal" here — the
    tiles from every OTHER slide are still real and still worth registering.
    """
    frames, missing = [], []
    for slide_id in slide_ids:
        path = tile_metadata_path(tile_dir, tile_dataset_name, slide_id)
        meta = read_tile_metadata(path)
        if not meta.usable:
            missing.append(f"{slide_id} ({meta.status}: {meta.detail})")
            continue
        frames.append(meta.frame)

    if not frames:
        return pd.DataFrame(columns=["slides", "tiles", "col", "row", "x_5x",
                                     "y_5x", "x_native", "y_native", "slide_tile"]), missing

    coords = pd.concat(frames, ignore_index=True)
    coords["slide_tile"] = make_slide_tile_series(coords["slides"], coords["tiles"])
    return coords, missing


def build_registration(h5_path: Path, tile_dir: Path, tile_dataset_name: str,
                       h5_source_path: str, dataset_id: str) -> dict:
    """Everything read_dataset_run needs, computed without touching the DB."""
    identity = read_h5_identity(h5_path)
    slide_ids = sorted(set(identity["slides"]))
    coords, missing_slides = read_tile_coordinates(tile_dir, tile_dataset_name, slide_ids)

    merged = identity.merge(
        coords[["slide_tile", "col", "row", "x_5x", "y_5x", "x_native", "y_native"]],
        on="slide_tile", how="left", validate="one_to_one",
    )
    unmatched = merged[merged["col"].isna()]

    registry = merged[["samples", "slides", "tiles", "slide_tile", "image_index"]].copy()
    registry["h5_source_path"] = h5_source_path
    registry["dataset_id"] = dataset_id

    coordinates = merged.dropna(subset=["col"]).copy()
    coordinates = coordinates[["slides", "tiles", "slide_tile", "col", "row",
                               "x_5x", "y_5x", "x_native", "y_native", "image_index"]]
    coordinates = coordinates.rename(columns={"image_index": "h5_index"})
    coordinates["dataset_id"] = dataset_id
    for col in ("col", "row", "x_5x", "y_5x", "x_native", "y_native", "h5_index"):
        coordinates[col] = coordinates[col].astype(np.int64)

    return {
        "registry": registry,
        "coordinates": coordinates,
        "slides": slide_ids,
        "missing_slides": missing_slides,
        "unmatched_tiles": unmatched["slide_tile"].tolist(),
    }


def _existing_scope(conn, dataset_id: str) -> dict:
    """What already exists in the KB for this dataset_id, across both tables."""
    scope = {}
    for table in ("tile_registry", "tile_coordinates"):
        row = conn.execute(
            text(f"SELECT COUNT(*) AS n, COUNT(DISTINCT slides) AS slides "
                 f"FROM {table} WHERE dataset_id = :d"),
            {"d": dataset_id},
        ).mappings().one()
        scope[table] = {"rows": row["n"], "slides": row["slides"]}
    return scope


def _foreign_scope(conn, table: str, dataset_id: str, slide_tiles) -> int:
    """How many of these slide_tile keys already belong to a DIFFERENT
    dataset_id. Non-zero means two cohorts are claiming the same tile —
    refused as an error, since a collision here means one of the two datasets
    is misidentified, not that the newer one should win."""
    total = 0
    tiles = list(slide_tiles)
    lookup = text(
        f"SELECT COUNT(*) AS n FROM {table} "
        f"WHERE UPPER(slide_tile) IN :tiles AND dataset_id != :d"
    ).bindparams(bindparam("tiles", expanding=True))
    for start in range(0, len(tiles), _LOOKUP_CHUNK):
        total += conn.execute(
            lookup, {"tiles": [t.upper() for t in tiles[start:start + _LOOKUP_CHUNK]],
                     "d": dataset_id},
        ).scalar()
    return total


def preview(engine, plan: dict, dataset_id: str) -> dict:
    """What committing would do, without doing it."""
    with engine.connect() as conn:
        existing = _existing_scope(conn, dataset_id)
        foreign = {
            "tile_registry": _foreign_scope(conn, "tile_registry", dataset_id,
                                            plan["registry"]["slide_tile"]),
            "tile_coordinates": _foreign_scope(conn, "tile_coordinates", dataset_id,
                                               plan["coordinates"]["slide_tile"]),
        }
    return {
        "dataset_id": dataset_id,
        "slides": len(plan["slides"]),
        "tiles_in_h5": len(plan["registry"]),
        "tiles_with_coordinates": len(plan["coordinates"]),
        "missing_slides": plan["missing_slides"],
        "unmatched_tiles": plan["unmatched_tiles"],
        "existing": existing,
        "foreign_collisions": foreign,
    }


def _insert(conn, table: str, frame: pd.DataFrame, columns: tuple[str, ...]) -> int:
    """Reflect-before-insert, same pattern as load_hpc_assignments.py: these
    tables predate this script, so a column this script assumes and the live
    table lacks must not fail the whole transaction silently — it is reported
    and the column is dropped from what's written."""
    existing_columns = {c["name"] for c in sqlalchemy_inspect(conn).get_columns(table)}
    usable = [c for c in columns if c in existing_columns]
    skipped = [c for c in columns if c not in existing_columns]
    if skipped:
        print(f"  {table}: no column(s) {skipped}; not writing them", file=sys.stderr)
    placeholders = ", ".join(f":{c}" for c in usable)
    conn.execute(
        text(f"INSERT INTO {table} ({', '.join(usable)}) VALUES ({placeholders})"),
        frame[usable].to_dict("records"),
    )
    return len(frame)


def commit(engine, plan: dict, dataset_id: str, replace: bool) -> dict:
    """Write tile_coordinates and tile_registry, scoped to dataset_id.

    One transaction, matching Stage 5's own reasoning: a registration that
    wrote tile_registry but not tile_coordinates (or the reverse) would leave
    the viewer able to find a tile's position but not its identity, or the
    other way round, and nothing downstream could tell that had happened.

    Deleting before inserting rather than upserting, and only for this exact
    dataset_id: a --replace scoped by anything looser could touch another
    cohort's rows sharing a coincidentally identical slide_tile.
    """
    if plan["registry"].empty:
        raise SystemExit("Nothing to register — the .h5 held no tiles.")

    with engine.begin() as conn:
        existing = _existing_scope(conn, dataset_id)
        already_present = existing["tile_registry"]["rows"] or existing["tile_coordinates"]["rows"]
        if already_present and not replace:
            raise SystemExit(
                f"{dataset_id} already has {existing['tile_registry']['rows']:,} "
                f"tile_registry row(s) and {existing['tile_coordinates']['rows']:,} "
                f"tile_coordinates row(s) across {existing['tile_registry']['slides']} "
                f"slide(s). Pass --replace to overwrite them — this dataset_id is "
                f"the intended re-registration path once a fuller run supersedes "
                f"this one, but it is never automatic."
            )

        foreign_registry = _foreign_scope(conn, "tile_registry", dataset_id,
                                          plan["registry"]["slide_tile"])
        foreign_coords = _foreign_scope(conn, "tile_coordinates", dataset_id,
                                        plan["coordinates"]["slide_tile"])
        if foreign_registry or foreign_coords:
            raise SystemExit(
                f"{foreign_registry + foreign_coords} of these tiles' slide_tile "
                f"keys already belong to a DIFFERENT dataset_id. Two cohorts "
                f"cannot claim the same tile — this needs investigating, not "
                f"overwriting."
            )

        if replace:
            conn.execute(text("DELETE FROM tile_registry WHERE dataset_id = :d"),
                        {"d": dataset_id})
            conn.execute(text("DELETE FROM tile_coordinates WHERE dataset_id = :d"),
                        {"d": dataset_id})

        written = {
            "tile_registry": _insert(conn, "tile_registry", plan["registry"],
                                     _TILE_REGISTRY_COLUMNS),
            "tile_coordinates": _insert(conn, "tile_coordinates", plan["coordinates"],
                                        _TILE_COORDINATES_COLUMNS),
        }
    return written


def report(result: dict, commit_mode: bool) -> None:
    print(f"\ndataset_id     {result['dataset_id']}")
    print(f"slides         {result['slides']:,}")
    print(f"tiles in .h5   {result['tiles_in_h5']:,}")
    print(f"with coords    {result['tiles_with_coordinates']:,}")

    if result["missing_slides"]:
        print(f"\n{len(result['missing_slides'])} slide(s) have no usable Stage 1 "
              f"metadata — their tiles will be registered with NO tile_coordinates "
              f"row (position on the slide unknown to the viewer):")
        for s in result["missing_slides"][:10]:
            print(f"  {s}")

    if result["unmatched_tiles"]:
        print(f"\n{len(result['unmatched_tiles'])} tile(s) in the .h5 have no "
              f"matching row in Stage 1's metadata (e.g. {result['unmatched_tiles'][:3]}) "
              f"— registered in tile_registry only.")

    existing = result["existing"]
    if existing["tile_registry"]["rows"] or existing["tile_coordinates"]["rows"]:
        print(f"\n{existing['tile_registry']['rows']:,} tile_registry row(s) and "
              f"{existing['tile_coordinates']['rows']:,} tile_coordinates row(s) "
              f"already exist for this dataset_id. --replace is required to "
              f"overwrite them.")

    collisions = sum(result["foreign_collisions"].values())
    if collisions:
        print(f"\nREFUSING: {collisions} tile(s) already belong to a different "
              f"dataset_id — see above.")

    if not commit_mode:
        print("\nDry run — nothing written. Re-run with --commit to apply.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--h5", type=Path, required=True,
                        help="The packaged .h5 from make_hpl_hdf5.py.")
    parser.add_argument("--tile-dir", type=Path, required=True,
                        help="Root directory tiles were written under (the "
                             "tile_dir passed to Stage 1).")
    parser.add_argument("--tile-dataset-name", required=True,
                        help="Folder under --tile-dir tiles live in, e.g. "
                             "Radiogenomics.")
    parser.add_argument("--dataset-id", required=True,
                        help="The KB cohort this dataset belongs to, e.g. "
                             "RADIOGENOMICS. Everything written is scoped to it, "
                             "and a --replace only ever touches its own rows.")
    parser.add_argument("--h5-source-path", default=None,
                        help="Recorded in tile_registry.h5_source_path. Defaults "
                             "to --h5.")
    parser.add_argument("--commit", action="store_true",
                        help="Actually write. Without it this only previews.")
    parser.add_argument("--replace", action="store_true",
                        help="Overwrite this dataset_id's existing rows. Refused "
                             "without this flag if any already exist.")
    args = parser.parse_args()

    if not args.h5.is_file():
        raise SystemExit(f"No such file: {args.h5}")
    if not args.tile_dir.is_dir():
        raise SystemExit(f"No such directory: {args.tile_dir}")

    plan = build_registration(
        args.h5, args.tile_dir, args.tile_dataset_name,
        args.h5_source_path or str(args.h5), args.dataset_id,
    )
    engine = make_engine()

    if not args.commit:
        report(preview(engine, plan, args.dataset_id), commit_mode=False)
        return

    result = preview(engine, plan, args.dataset_id)  # for the report's numbers
    written = commit(engine, plan, args.dataset_id, args.replace)
    report(result, commit_mode=True)
    print(f"\nwritten        tile_registry +{written['tile_registry']:,}, "
          f"tile_coordinates +{written['tile_coordinates']:,}")
    print("\nNext: run load_hpc_assignments.py to fill in hpc_id and the "
          "per-slide aggregates.")


if __name__ == "__main__":
    raise SystemExit(main())
