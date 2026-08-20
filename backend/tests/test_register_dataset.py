"""Registering a new dataset's identity rows into the Knowledge Bank.

This is the step that did not exist: Stage 5 (load_hpc_assignments.py) only
ever UPDATEs tile_registry.hpc_id, so a dataset that has never touched the KB
refuses at 0% match rate by construction — there is nothing there to update.
These tests are mostly about the two things that make a registration wrong in
a way that is hard to notice afterwards: image_index not actually matching the
.h5's row order, and a --replace or collision touching a dataset_id it
shouldn't.
"""

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from sqlalchemy import create_engine, event, text  # noqa: E402

import register_dataset as rd  # noqa: E402

SLIDE = "BB232560 A3-1 - 2023-10-11 16.41.02"


def _write_h5(path: Path, rows):
    """rows: list of (sample, slide, tile) in .h5 row order."""
    samples = [r[0].encode() for r in rows]
    slides = [r[1].encode() for r in rows]
    tiles = [r[2].encode() for r in rows]
    with h5py.File(path, "w") as f:
        f.create_dataset("img", (len(rows), 2, 2, 3), dtype="uint8")
        f.create_dataset("samples", data=np.array(samples, dtype=f"S{max(len(s) for s in samples)}"))
        f.create_dataset("slides", data=np.array(slides, dtype=f"S{max(len(s) for s in slides)}"))
        f.create_dataset("tiles", data=np.array(tiles, dtype=f"S{max(len(t) for t in tiles)}"))


def _write_metadata(tile_dir: Path, tile_dataset_name: str, slide_id: str, tile_rows):
    """tile_rows: list of (col, row) -> writes a real Stage-1-shaped CSV."""
    slide_dir = tile_dir / tile_dataset_name / slide_id
    slide_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for col, row in tile_rows:
        name = f"{col}_{row}.jpeg"
        records.append({
            "slides": slide_id, "tiles": name, "slide_tile": f"{slide_id}_{name}",
            "col": col, "row": row, "x_5x": col * 224, "y_5x": row * 224,
            "x_native": col * 1600, "y_native": row * 1600, "tissue_percent": 80.0,
        })
    pd.DataFrame(records).to_csv(slide_dir / f"{slide_id}_tile_metadata.csv", index=False)


def _kb_engine(tmp_path: Path, with_dataset_id_column=True):
    engine = create_engine(f"sqlite:///{tmp_path / 'kb.sqlite'}")

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    dsid = "dataset_id TEXT" if with_dataset_id_column else ""
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE tile_registry (
                slide_tile TEXT PRIMARY KEY, samples TEXT, slides TEXT, tiles TEXT,
                image_index INTEGER, h5_source_path TEXT, hpc_id TEXT,
                hpc_vote_margin REAL, hpc_neighbor_distance REAL,
                hpc_reference TEXT, hpc_assigned_at TIMESTAMP, {dsid})"""))
        conn.execute(text(f"""
            CREATE TABLE tile_coordinates (
                slide_tile TEXT PRIMARY KEY, slides TEXT, tiles TEXT,
                col INTEGER, row INTEGER, x_5x INTEGER, y_5x INTEGER,
                x_native INTEGER, y_native INTEGER, h5_index INTEGER, {dsid})"""))
    return engine


def test_image_index_is_the_h5_row_position_not_csv_order(tmp_path):
    """The central claim: image_index must match the ACTUAL array position in
    the .h5, which need not be the order Stage 1's metadata lists tiles in."""
    h5_path = tmp_path / "packaged.h5"
    # .h5 row order deliberately shuffled relative to how metadata will list them.
    _write_h5(h5_path, [
        ("S1", SLIDE, "25_10.jpeg"),
        ("S1", SLIDE, "24_10.jpeg"),
        ("S1", SLIDE, "7_3.jpeg"),
    ])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(24, 10), (25, 10), (7, 3)])

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")
    by_tile = plan["registry"].set_index("tiles")["image_index"].to_dict()
    assert by_tile["25_10.jpeg"] == 0
    assert by_tile["24_10.jpeg"] == 1
    assert by_tile["7_3.jpeg"] == 2

    coords_by_tile = plan["coordinates"].set_index("tiles")["h5_index"].to_dict()
    assert coords_by_tile["25_10.jpeg"] == 0
    assert coords_by_tile["24_10.jpeg"] == 1


def test_slide_with_no_metadata_is_reported_not_dropped_silently(tmp_path):
    """A slide missing its Stage 1 CSV still has real tiles in the .h5 — they
    go into tile_registry with no coordinates, and the gap is reported, not
    silently absorbed."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", SLIDE, "1_1.jpeg"), ("S1", "OTHER SLIDE", "2_2.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(1, 1)])
    # "OTHER SLIDE" has no metadata CSV at all.

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")
    assert len(plan["registry"]) == 2, "both tiles still register"
    assert len(plan["coordinates"]) == 1, "only the one with metadata gets coordinates"
    assert any("OTHER SLIDE" in m for m in plan["missing_slides"])


def test_legacy_h5_is_refused_with_the_migration_command(tmp_path):
    h5_path = tmp_path / "legacy.h5"
    _write_h5(h5_path, [("S1", SLIDE, "24_10")])  # no suffix
    try:
        rd.read_h5_identity(h5_path)
    except SystemExit as e:
        assert "migrate_tile_names.py" in str(e), str(e)
    else:
        raise AssertionError("legacy tile names must be refused")


def test_first_registration_writes_both_tables_in_one_transaction(tmp_path):
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", SLIDE, "24_10.jpeg"), ("S1", SLIDE, "25_10.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(24, 10), (25, 10)])

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")
    engine = _kb_engine(tmp_path)
    written = rd.commit(engine, plan, "RADIOGENOMICS", replace=False)
    assert written == {"tile_registry": 2, "tile_coordinates": 2}

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT slide_tile, hpc_id, dataset_id FROM tile_registry"
        )).mappings().all()
        assert len(rows) == 2
        assert all(r["hpc_id"] is None for r in rows), "hpc_id is Stage 5's job, not this one's"
        assert all(r["dataset_id"] == "RADIOGENOMICS" for r in rows)


def test_second_registration_without_replace_is_refused(tmp_path):
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", SLIDE, "24_10.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(24, 10)])
    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")

    engine = _kb_engine(tmp_path)
    rd.commit(engine, plan, "RADIOGENOMICS", replace=False)

    try:
        rd.commit(engine, plan, "RADIOGENOMICS", replace=False)
    except SystemExit as e:
        assert "--replace" in str(e), str(e)
    else:
        raise AssertionError("re-registering the same dataset_id must be refused")

    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM tile_registry")).scalar() == 1


def test_replace_only_touches_its_own_dataset_id(tmp_path):
    """The one thing a --replace must never do: delete or overwrite another
    cohort's rows, even ones that happen to share table space."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", SLIDE, "24_10.jpeg"), ("S1", SLIDE, "25_10.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(24, 10), (25, 10)])
    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")

    engine = _kb_engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO tile_registry (slide_tile, samples, slides, tiles, "
            "image_index, dataset_id) VALUES ('TCGA-1_1_1.JPEG', 'TCGA-1', "
            "'TCGA-1', '1_1.jpeg', 0, 'TCGA')"
        ))
        conn.execute(text(
            "INSERT INTO tile_coordinates (slide_tile, slides, tiles, col, row, "
            "dataset_id) VALUES ('TCGA-1_1_1.JPEG', 'TCGA-1', '1_1.jpeg', 1, 1, 'TCGA')"
        ))

    rd.commit(engine, plan, "RADIOGENOMICS", replace=False)
    # Replace RADIOGENOMICS with itself — must not touch TCGA's row.
    rd.commit(engine, plan, "RADIOGENOMICS", replace=True)

    with engine.connect() as conn:
        tcga_rows = conn.execute(text(
            "SELECT COUNT(*) FROM tile_registry WHERE dataset_id = 'TCGA'"
        )).scalar()
        radio_rows = conn.execute(text(
            "SELECT COUNT(*) FROM tile_registry WHERE dataset_id = 'RADIOGENOMICS'"
        )).scalar()
    assert tcga_rows == 1, "TCGA's row must survive a RADIOGENOMICS --replace"
    assert radio_rows == 2


def test_slide_tile_collision_across_datasets_is_refused_not_reassigned(tmp_path):
    """Two different dataset_ids producing the same slide_tile key is a real
    problem — a naming collision or a misidentified cohort — and must be
    surfaced as an error, never silently resolved by whichever ran last."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", SLIDE, "24_10.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(24, 10)])
    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")
    colliding_key = plan["registry"]["slide_tile"].iloc[0]

    engine = _kb_engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO tile_registry (slide_tile, samples, slides, tiles, "
            "image_index, dataset_id) VALUES (:k, 'X', 'X', 'x.jpeg', 0, 'OTHER_COHORT')"
        ), {"k": colliding_key})

    try:
        rd.commit(engine, plan, "RADIOGENOMICS", replace=False)
    except SystemExit as e:
        assert "different dataset_id" in str(e).lower() or "DIFFERENT dataset_id" in str(e)
    else:
        raise AssertionError("a cross-cohort slide_tile collision must be refused")


def test_dry_run_writes_nothing(tmp_path):
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", SLIDE, "24_10.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(24, 10)])
    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")
    engine = _kb_engine(tmp_path)

    result = rd.preview(engine, plan, "RADIOGENOMICS")
    assert result["tiles_in_h5"] == 1
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM tile_registry")).scalar() == 0


def test_missing_column_in_a_live_table_does_not_abort_the_whole_write(tmp_path):
    """These tables predate this script. A table missing a column this script
    assumes must drop that column from the insert and say so, not fail the
    whole transaction — same reasoning as load_hpc_assignments.py."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", SLIDE, "24_10.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(24, 10)])
    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")

    engine = create_engine(f"sqlite:///{tmp_path / 'narrow.sqlite'}")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE tile_registry (
                slide_tile TEXT PRIMARY KEY, samples TEXT, slides TEXT,
                tiles TEXT, image_index INTEGER, dataset_id TEXT)"""))
        # No h5_source_path column.
        conn.execute(text("""
            CREATE TABLE tile_coordinates (
                slide_tile TEXT PRIMARY KEY, slides TEXT, tiles TEXT,
                col INTEGER, row INTEGER, x_5x INTEGER, y_5x INTEGER,
                x_native INTEGER, y_native INTEGER, h5_index INTEGER, dataset_id TEXT)"""))

    written = rd.commit(engine, plan, "RADIOGENOMICS", replace=False)
    assert written == {"tile_registry": 1, "tile_coordinates": 1}


def test_registration_takes_stage5_from_zero_percent_to_full_match(tmp_path):
    """The whole point, end to end.

    Before registering, Stage 5 matches 0% — not because of a naming bug but
    because tile_registry has no rows for this cohort at all, and Stage 5 only
    ever UPDATEs. After registering, the same CSV matches every tile. This is
    the exact failure the Radiogenomics load hit, so it is asserted rather than
    described.
    """
    import load_hpc_assignments as loader

    h5_path = tmp_path / "packaged.h5"
    tile_rows = [(24, 10), (25, 10), (7, 3)]
    _write_h5(h5_path, [("S1", SLIDE, f"{c}_{r}.jpeg") for c, r in tile_rows])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, tile_rows)

    # Stage 4's CSV, in the form migrate_tile_names.py produces.
    csv_path = tmp_path / "assignments.csv"
    pd.DataFrame({
        "samples": ["S1"] * 3,
        "slides": [SLIDE] * 3,
        "tiles": [f"{c}_{r}.jpeg" for c, r in tile_rows],
        "leiden_2.5": [28, 50, 19],
        "vote_margin": [0.9, 0.4, 0.2],
        "neighbor_distance": [1.0, 2.0, 3.0],
        "hpc_reference": ["hpc_reference_leiden_2p5_fold2"] * 3,
    }).to_csv(csv_path, index=False)

    engine = _kb_engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE hpc_dictionary (hpc_id TEXT, malignant TEXT)"))
        for cluster in ("28", "50", "19"):
            conn.execute(text("INSERT INTO hpc_dictionary VALUES (:h, 'True')"),
                         {"h": cluster})

    frame, cluster_column = loader.read_assignments(csv_path)

    before = loader.inspect(engine, frame, cluster_column)
    assert before["matched"] == 0, "nothing to match before registration"
    assert before["unmatched"] == 3

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")
    rd.commit(engine, plan, "RADIOGENOMICS", replace=False)

    after = loader.inspect(engine, frame, cluster_column)
    assert after["matched"] == 3, after
    assert after["unmatched"] == 0
    assert after["matched"] / after["rows"] >= loader._MIN_MATCH_RATE


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_register_test_"))
        try:
            fn(tmp_path)
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
