"""Loading cluster assignments into the Knowledge Bank.

This is the only script in the pipeline that writes to the shared KB, and its
failure mode is silence rather than a crash. If the CSV and tile_registry
disagree about slide naming, the UPDATE matches nothing and reports success. If
they agree for only some slides, it updates those and leaves the rest carrying
cluster IDs from an older reference — a registry in two states at once, which
nothing downstream can detect because every row still looks valid.

So the tests here are mostly about the refusals, and about the join key, since a
wrong `slide_tile` is exactly how the silent-no-op happens.

Runs against SQLite standing in for Postgres. That is enough because the loader
deliberately uses portable SQL — an expanding IN rather than PostgreSQL's
ANY(array) — and what is under test is the matching logic and the guards, not
the driver.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from sqlalchemy import create_engine, text  # noqa: E402

import load_hpc_assignments as loader  # noqa: E402

SLIDE = "TCGA-55-7574-01Z-00-DX1"
REFERENCE = "hpc_reference_leiden_2p5_fold2"


def _make_kb(tmp_path: Path, registry_tiles, clusters=("0", "1", "2"), existing=None):
    """A KB with just the two tables the loader touches."""
    engine = create_engine(f"sqlite:///{tmp_path / 'kb.sqlite'}")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE tile_registry (
                slide_tile TEXT, image_index INTEGER, hpc_id TEXT,
                hpc_vote_margin REAL, hpc_neighbor_distance REAL,
                hpc_reference TEXT, hpc_assigned_at TIMESTAMP
            )"""))
        conn.execute(text("CREATE TABLE hpc_dictionary (hpc_id TEXT, malignant TEXT)"))
        for i, tile in enumerate(registry_tiles):
            conn.execute(
                text("INSERT INTO tile_registry (slide_tile, image_index, hpc_id, "
                     "hpc_reference) VALUES (:t, :i, :h, :r)"),
                {"t": tile, "i": i,
                 "h": (existing or {}).get(tile),
                 "r": REFERENCE if (existing or {}).get(tile) else None},
            )
        for cluster in clusters:
            conn.execute(text("INSERT INTO hpc_dictionary VALUES (:h, 'True')"),
                         {"h": cluster})
    return engine


def _make_csv(tmp_path: Path, n=20, slide=SLIDE, clusters=("0", "1", "2"), name="a.csv"):
    rows = [{
        "samples": "S1", "slides": slide, "tiles": f"{i}_{i}.jpeg",
        "leiden_2.5": clusters[i % len(clusters)],
        "vote_margin": 0.05 if i % 10 == 0 else 0.8,
        "neighbor_distance": 1.2,
        "hpc_reference": REFERENCE,
    } for i in range(n)]
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _registry_tiles(n=20, slide=SLIDE):
    """slide_tile as tile_coordinates stores it: "<slides>_<tiles>"."""
    return [f"{slide}_{i}_{i}.jpeg" for i in range(n)]


def test_join_key_matches_the_registrys_format(tmp_path):
    """The whole loader hinges on this. tile_coordinates stores
    '<slides>_<tiles>' — TCGA-55-7574-01Z-00-DX1_18_15.jpeg — and getting it
    wrong makes every UPDATE match nothing while reporting success."""
    csv = _make_csv(tmp_path)
    frame, column = loader.read_assignments(csv)
    assert column == "leiden_2.5"
    assert frame["slide_tile"].iloc[0] == f"{SLIDE}_0_0.JPEG"
    # Upper-cased, because the server joins with UPPER() on both sides.
    assert frame["slide_tile"].str.isupper().all()


def test_load_populates_every_column_the_viewer_reads(tmp_path):
    csv = _make_csv(tmp_path)
    engine = _make_kb(tmp_path, _registry_tiles())
    frame, column = loader.read_assignments(csv)

    report = loader.inspect(engine, frame, column)
    assert report["matched"] == 20 and report["unmatched"] == 0
    assert report["unknown_clusters"] == []
    assert report["low_margin"] == 2  # i % 10 == 0

    assert loader.load(engine, frame, column) == 20
    with engine.connect() as conn:
        got = pd.read_sql(text("SELECT * FROM tile_registry ORDER BY image_index"), conn)
    assert got["hpc_id"].notna().all()
    assert got["hpc_vote_margin"].notna().all()
    assert got["hpc_neighbor_distance"].notna().all()
    assert (got["hpc_reference"] == REFERENCE).all()
    assert got["hpc_assigned_at"].notna().all()
    # Row-for-row: tile i must carry the cluster the CSV gave tile i.
    expected = pd.read_csv(csv)["leiden_2.5"].astype(str).tolist()
    assert got["hpc_id"].tolist() == expected


def test_a_naming_mismatch_is_refused_not_reported_as_success(tmp_path):
    """The failure this script exists to prevent: the registry knows the slide
    by another name, so the UPDATE matches nothing at all."""
    csv = _make_csv(tmp_path, slide="TCGA-55-7574")          # registry has the full ID
    engine = _make_kb(tmp_path, _registry_tiles())
    frame, column = loader.read_assignments(csv)
    report = loader.inspect(engine, frame, column)
    assert report["matched"] == 0
    assert report["matched"] / report["rows"] < loader._MIN_MATCH_RATE
    assert report["unmatched_examples"], "must name the tiles that did not match"


def test_a_partial_match_is_refused(tmp_path):
    """More dangerous than a total mismatch, because a summary line still reads
    as plausible. Half the tiles updated leaves the registry in two states."""
    csv = _make_csv(tmp_path, n=20)
    engine = _make_kb(tmp_path, _registry_tiles(n=10))  # only half are registered
    frame, column = loader.read_assignments(csv)
    report = loader.inspect(engine, frame, column)
    assert report["matched"] == 10
    assert report["matched"] / report["rows"] < loader._MIN_MATCH_RATE


def test_clusters_absent_from_the_dictionary_are_flagged(tmp_path):
    """A cluster with no hpc_dictionary row joins to NULL in the viewer: the
    tile shows a cluster with no pattern, malignancy or inflammation."""
    csv = _make_csv(tmp_path, clusters=("0", "1", "99"))
    engine = _make_kb(tmp_path, _registry_tiles(), clusters=("0", "1", "2"))
    frame, column = loader.read_assignments(csv)
    report = loader.inspect(engine, frame, column)
    assert report["unknown_clusters"] == ["99"]


def test_overwriting_an_earlier_reference_is_reported(tmp_path):
    """Reassigning against a different reference is legitimate, but it changes
    the meaning of every ID in the registry, so it cannot happen silently."""
    tiles = _registry_tiles()
    engine = _make_kb(tmp_path, tiles, existing={t: "7" for t in tiles})
    csv = _make_csv(tmp_path)
    frame, column = loader.read_assignments(csv)
    report = loader.inspect(engine, frame, column)
    assert report["overwriting"] == 20


def test_inspect_changes_nothing(tmp_path):
    """--dry-run has to be trustworthy, or nobody will use it."""
    csv = _make_csv(tmp_path)
    engine = _make_kb(tmp_path, _registry_tiles())
    frame, column = loader.read_assignments(csv)

    with engine.connect() as conn:
        before = pd.read_sql(text("SELECT * FROM tile_registry"), conn).to_json()
    loader.inspect(engine, frame, column)
    with engine.connect() as conn:
        after = pd.read_sql(text("SELECT * FROM tile_registry"), conn).to_json()
    assert before == after


def test_loading_twice_is_idempotent(tmp_path):
    """Re-running the same CSV must converge, not accumulate — the obvious
    reaction to a job that looked like it half-finished is to run it again."""
    csv = _make_csv(tmp_path)
    engine = _make_kb(tmp_path, _registry_tiles())
    frame, column = loader.read_assignments(csv)

    loader.load(engine, frame, column)
    with engine.connect() as conn:
        first = pd.read_sql(
            text("SELECT slide_tile, hpc_id, hpc_vote_margin FROM tile_registry "
                 "ORDER BY image_index"), conn).to_json()
    loader.load(engine, frame, column)
    with engine.connect() as conn:
        second = pd.read_sql(
            text("SELECT slide_tile, hpc_id, hpc_vote_margin FROM tile_registry "
                 "ORDER BY image_index"), conn).to_json()
    assert first == second


def test_lookup_chunking_does_not_lose_tiles(tmp_path):
    """inspect() queries in chunks so a whole dataset does not become one
    enormous parameter list. The chunking must not drop rows at the seams."""
    original = loader._LOOKUP_CHUNK
    try:
        loader._LOOKUP_CHUNK = 3  # forces ragged chunks against 20 rows
        csv = _make_csv(tmp_path, n=20)
        engine = _make_kb(tmp_path, _registry_tiles(n=20))
        frame, column = loader.read_assignments(csv)
        assert loader.inspect(engine, frame, column)["matched"] == 20
    finally:
        loader._LOOKUP_CHUNK = original


# --- the per-slide aggregates --------------------------------------------
# hpl_profile_proportion and hpl_profile_summary are what the chatbot and the
# HPC panels actually read — not tile_registry. Loading tiles without refreshing
# these leaves the UI showing new clusters per tile and old proportions per
# slide, with nothing to indicate the two came from different runs.

def _add_profile_tables(engine, rows=()):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE hpl_profile_proportion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                samples TEXT, slides TEXT, hpc_id TEXT, proportion REAL)"""))
        conn.execute(text("""
            CREATE TABLE hpl_profile_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                samples TEXT, slides TEXT, cancer_type TEXT,
                total_tiles INTEGER, dominant_hpc TEXT)"""))
        for slide, hpc, prop in rows:
            conn.execute(
                text("INSERT INTO hpl_profile_proportion (samples, slides, hpc_id, "
                     "proportion) VALUES ('S1', :s, :h, :p)"),
                {"s": slide, "h": hpc, "p": prop},
            )


def _add_profile_tables_with_fk(engine, rows=()):
    """The aggregate tables *with* the foreign key Postgres actually has.

    schema.sql:486-487 gives hpl_profile_proportion
    FOREIGN KEY (samples, slides) REFERENCES hpl_profile_summary ON DELETE CASCADE.
    _add_profile_tables above omits it, which is why the ordering bug in
    replace_profiles survived every existing test: without the constraint, both
    the child-before-parent insert and the cascade-away-what-was-just-written
    case are invisible.

    SQLite enforces foreign keys only when asked, per connection.
    """
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE hpl_profile_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                samples TEXT, slides TEXT, cancer_type TEXT,
                total_tiles INTEGER, dominant_hpc TEXT,
                UNIQUE (samples, slides))"""))
        conn.execute(text("""
            CREATE TABLE hpl_profile_proportion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                samples TEXT, slides TEXT, hpc_id TEXT, proportion REAL,
                FOREIGN KEY (samples, slides)
                    REFERENCES hpl_profile_summary (samples, slides)
                    ON DELETE CASCADE)"""))
        for samples, slides, hpc, prop in rows:
            conn.execute(
                text("INSERT INTO hpl_profile_summary (samples, slides, total_tiles, "
                     "dominant_hpc) VALUES (:sa, :sl, 1, :h)"),
                {"sa": samples, "sl": slides, "h": hpc},
            )
            conn.execute(
                text("INSERT INTO hpl_profile_proportion (samples, slides, hpc_id, "
                     "proportion) VALUES (:sa, :sl, :h, :p)"),
                {"sa": samples, "sl": slides, "h": hpc, "p": prop},
            )


def test_first_load_of_a_new_cohort_does_not_violate_the_foreign_key(tmp_path):
    """The case that fires on every slide of a cohort's first load.

    hpl_profile_proportion references hpl_profile_summary, so inserting a
    proportion row for a slide that has no summary row yet is rejected — and
    because the loader writes everything in one transaction, that rollback would
    take the tile_registry update with it. Loading Radiogenomics for the first
    time is exactly this case for all ten slides.
    """
    csv = _make_csv(tmp_path, n=30, clusters=("0", "1", "2"))
    frame, column = loader.read_assignments(csv)
    proportions, summary = loader.compute_profiles(frame, column, None)

    engine = create_engine(f"sqlite:///{tmp_path / 'fk.sqlite'}")
    _add_profile_tables_with_fk(engine)  # empty: nothing to be a parent yet

    with engine.begin() as conn:
        loader.replace_profiles(conn, proportions, summary)

    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM hpl_profile_summary")).scalar() == 1
        assert conn.execute(
            text("SELECT COUNT(*) FROM hpl_profile_proportion")).scalar() == len(proportions)


def test_reload_does_not_cascade_away_the_rows_it_just_wrote(tmp_path):
    """The second FK failure, and the quieter one.

    Deleting a slide's summary row cascades to its proportions. Insert the
    proportions before that delete and they are silently removed again, leaving
    a summary row with no proportions — a slide whose HPC panel is simply empty,
    with no error anywhere.
    """
    csv = _make_csv(tmp_path, n=30, clusters=("0", "1", "2"))
    frame, column = loader.read_assignments(csv)
    proportions, summary = loader.compute_profiles(frame, column, None)

    engine = create_engine(f"sqlite:///{tmp_path / 'fk2.sqlite'}")
    # Pre-existing aggregates for the same sample/slide, as a reload would meet.
    _add_profile_tables_with_fk(
        engine, rows=[(summary["samples"].iloc[0], summary["slides"].iloc[0], "9", 1.0)]
    )

    with engine.begin() as conn:
        loader.replace_profiles(conn, proportions, summary)

    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM hpl_profile_summary")).scalar() == 1
        assert conn.execute(
            text("SELECT COUNT(*) FROM hpl_profile_proportion")).scalar() == len(proportions)
        # The stale cluster 9 row is gone, not left beside the new ones.
        assert conn.execute(
            text("SELECT COUNT(*) FROM hpl_profile_proportion WHERE hpc_id = '9'")
        ).scalar() == 0


def test_reload_matches_rows_stored_in_a_different_case(tmp_path):
    """migrate_indexes.sql normalised the live columns to UPPER(TRIM(...)).
    Matching on TRIM alone deletes nothing, and the insert still runs — so the
    slide silently ends up with two full sets of aggregates.
    """
    csv = _make_csv(tmp_path, n=30, clusters=("0", "1", "2"))
    frame, column = loader.read_assignments(csv)
    proportions, summary = loader.compute_profiles(frame, column, None)

    engine = create_engine(f"sqlite:///{tmp_path / 'case.sqlite'}")
    _add_profile_tables_with_fk(
        engine,
        rows=[(summary["samples"].iloc[0].upper(),
               summary["slides"].iloc[0].upper(), "9", 1.0)],
    )

    with engine.begin() as conn:
        loader.replace_profiles(conn, proportions, summary)

    with engine.connect() as conn:
        # One summary row, not two. The upper-cased one must have been replaced.
        assert conn.execute(text("SELECT COUNT(*) FROM hpl_profile_summary")).scalar() == 1
        assert conn.execute(
            text("SELECT COUNT(*) FROM hpl_profile_proportion")).scalar() == len(proportions)


def test_proportions_sum_to_one_per_slide(tmp_path):
    csv = _make_csv(tmp_path, n=30, clusters=("0", "1", "2"))
    frame, column = loader.read_assignments(csv)
    proportions, summary = loader.compute_profiles(frame, column, "LUAD")

    totals = proportions.groupby(["samples", "slides"])["proportion"].sum()
    assert all(abs(t - 1.0) < 1e-9 for t in totals), totals.to_dict()
    # 30 tiles cycling through three clusters -> a third each.
    assert sorted(proportions["proportion"].round(6)) == [round(1 / 3, 6)] * 3


def test_summary_counts_and_dominant_cluster(tmp_path):
    # 10 tiles: cluster "0" six times, "1" four times -> dominant is "0".
    rows = [{
        "samples": "S1", "slides": SLIDE, "tiles": f"{i}_{i}.jpeg",
        "leiden_2.5": "0" if i < 6 else "1",
        "vote_margin": 0.9, "neighbor_distance": 1.0, "hpc_reference": REFERENCE,
    } for i in range(10)]
    csv = tmp_path / "dom.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    frame, column = loader.read_assignments(csv)
    _, summary = loader.compute_profiles(frame, column, "LUAD")
    assert len(summary) == 1
    assert summary["total_tiles"].iloc[0] == 10
    assert summary["dominant_hpc"].iloc[0] == "0"
    assert summary["cancer_type"].iloc[0] == "LUAD"


def test_cancer_type_is_not_invented(tmp_path):
    """The original script hardcoded 'LUAD'. Guessing a cancer type into a KB
    that may hold several would be a quiet data error."""
    csv = _make_csv(tmp_path)
    frame, column = loader.read_assignments(csv)
    _, summary = loader.compute_profiles(frame, column, None)
    assert "cancer_type" not in summary.columns


def test_profiles_replace_only_the_loaded_slides(tmp_path):
    """Loading ten slides must not delete the proportions of every other slide."""
    csv = _make_csv(tmp_path, n=9, clusters=("0", "1", "2"))
    engine = _make_kb(tmp_path, _registry_tiles(n=9))
    _add_profile_tables(engine, rows=[
        (SLIDE, "7", 1.0),              # stale row for the slide being loaded
        ("SOME-OTHER-SLIDE", "3", 1.0),  # must survive
    ])

    frame, column = loader.read_assignments(csv)
    loader.load(engine, frame, column,
                profiles=loader.compute_profiles(frame, column, "LUAD"))

    with engine.connect() as conn:
        got = pd.read_sql(text("SELECT * FROM hpl_profile_proportion"), conn)
    assert "SOME-OTHER-SLIDE" in set(got["slides"]), "unrelated slide was deleted"
    mine = got[got["slides"] == SLIDE]
    # The stale cluster-7 row is gone, replaced by the three real clusters.
    assert set(mine["hpc_id"]) == {"0", "1", "2"}, set(mine["hpc_id"])
    assert abs(mine["proportion"].sum() - 1.0) < 1e-9


def test_ids_come_from_the_sequence_not_a_range(tmp_path):
    """The notebooks assigned id = range(1, n+1), which is right exactly once and
    collides with every existing row afterwards."""
    csv = _make_csv(tmp_path, n=9)
    engine = _make_kb(tmp_path, _registry_tiles(n=9))
    _add_profile_tables(engine, rows=[("OTHER", "3", 1.0)])

    frame, column = loader.read_assignments(csv)
    loader.load(engine, frame, column,
                profiles=loader.compute_profiles(frame, column, "LUAD"))
    with engine.connect() as conn:
        ids = pd.read_sql(text("SELECT id FROM hpl_profile_proportion"), conn)["id"]
    assert ids.is_unique, "ids collided with the pre-existing row"


def test_missing_aggregate_column_does_not_abort_the_load(tmp_path):
    """These tables were filled by hand from notebooks, so a column may not be
    there. Inserting a name that does not exist would fail the whole
    transaction, taking the tile_registry update with it."""
    csv = _make_csv(tmp_path, n=9)
    engine = _make_kb(tmp_path, _registry_tiles(n=9))
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE hpl_profile_proportion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                samples TEXT, slides TEXT, hpc_id TEXT, proportion REAL)"""))
        # No cancer_type column here.
        conn.execute(text("""
            CREATE TABLE hpl_profile_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                samples TEXT, slides TEXT, total_tiles INTEGER, dominant_hpc TEXT)"""))

    frame, column = loader.read_assignments(csv)
    assert loader.load(engine, frame, column,
                       profiles=loader.compute_profiles(frame, column, "LUAD")) == 9
    with engine.connect() as conn:
        assert len(pd.read_sql(text("SELECT * FROM hpl_profile_summary"), conn)) == 1


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_kb_test_"))
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
