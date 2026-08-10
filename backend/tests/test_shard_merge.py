"""Sharded feature extraction splits one .h5 across N GPU jobs and glues the
results back together. These tests are about the glue.

The failure this guards against is not a crash. Concatenating parts in the
wrong order, across a gap, or over a part that died mid-write produces a file
of exactly the right shape and dtype with no missing values — it passes every
completeness check the pipeline has, and every downstream cluster assignment
is attached to the wrong tile. So the merge is required to prove the parts
tile the input exactly before it copies a single row, and these tests are
mostly about the proofs rather than the copying.
"""

import re
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from merge_projection_shards import (  # noqa: E402
    _PART_RE,
    find_parts,
    merge_projection_shards,
)
from submit_feature_extraction import shard_output_path, shard_ranges  # noqa: E402

HPL_FEATURES = (
    Path(__file__).resolve().parents[2] / "HPL-LATTICeA" / "models" / "evaluation" / "features.py"
)

H_DIM, Z_DIM = 1536, 128
CARRIED = ("samples", "slides", "tiles")


def write_full(path: Path, rows: int) -> dict:
    """What an unsharded run produces. Values are a function of the row index
    so any reordering during a merge is visible rather than plausible."""
    idx = np.arange(rows, dtype=np.float32)
    data = {
        "img_h_latent": np.outer(idx, np.arange(1, H_DIM + 1, dtype=np.float32)),
        "img_z_latent": np.outer(idx, -np.arange(1, Z_DIM + 1, dtype=np.float32)),
        "samples": np.arange(rows, dtype=np.int64),
        "slides": np.array([f"slide_{i // 100}".encode() for i in range(rows)]),
        "tiles": np.arange(rows, dtype=np.int64),
    }
    with h5py.File(path, "w") as f:
        for name, arr in data.items():
            f.create_dataset(name, data=arr)
    return data


def write_parts(final: Path, rows: int, bounds, full: dict, skip=(), truncate=None):
    """Split `full` into part files exactly as the sharded encoder would."""
    written = []
    for i, (lo, hi) in enumerate(bounds):
        if i in skip:
            continue
        part = shard_output_path(final, lo, hi)
        n = hi - lo
        if truncate is not None and i == truncate[0]:
            n = truncate[1]
        with h5py.File(part, "w") as f:
            for name, arr in full.items():
                f.create_dataset(name, data=arr[lo:lo + n])
        written.append(part)
    return written


def _fails(fn, *, contains: str) -> str:
    try:
        fn()
    except (ValueError, FileExistsError) as e:
        assert contains in str(e), f"expected {contains!r} in: {e}"
        return str(e)
    raise AssertionError(f"expected a failure mentioning {contains!r}")


# --- the split itself ----------------------------------------------------

def test_ranges_tile_the_input_exactly(tmp_path):
    """No gaps, no overlaps, nothing empty, for every split we might use."""
    del tmp_path
    for rows in (1, 7, 100, 1000, 1000003, 999_999):
        for shards in (1, 2, 3, 4, 8, 16):
            if shards > rows:
                continue
            bounds = shard_ranges(rows, shards)
            assert len(bounds) == shards
            assert bounds[0][0] == 0
            assert bounds[-1][1] == rows
            for (lo, hi), (next_lo, _) in zip(bounds, bounds[1:]):
                assert hi > lo, f"empty shard at {rows}/{shards}"
                assert next_lo == hi, f"gap or overlap at {rows}/{shards}"
            # Sizes differ by at most one, so no task is handed double the work.
            sizes = [hi - lo for lo, hi in bounds]
            assert max(sizes) - min(sizes) <= 1


def test_part_naming_agrees_across_the_three_places_it_is_built(tmp_path):
    """The encoder builds the part name, the submitter predicts it to clear
    stale ones, and the merge parses it back. All three must agree or a run
    looks like it produced nothing."""
    del tmp_path
    final = Path("/results/hdf5_DS_he_train.h5")
    produced = shard_output_path(final, 0, 250)
    assert produced.name == "hdf5_DS_he_train.rows0-250.h5"

    m = _PART_RE.match(produced.name)
    assert m and m.group("stem") == "hdf5_DS_he_train"
    assert (int(m.group("lo")), int(m.group("hi"))) == (0, 250)

    if HPL_FEATURES.is_file():
        # The encoder's own construction, kept in step with the above.
        assert ".rows%s-%s.h5' % (row_lo, row_hi)" in HPL_FEATURES.read_text()


# --- the merge -----------------------------------------------------------

def test_sharded_output_is_identical_to_unsharded(tmp_path):
    """The whole point. Four shards, merged, compared bit for bit against the
    file a single job would have written."""
    rows = 1000
    reference = tmp_path / "reference.h5"
    full = write_full(reference, rows)

    final = tmp_path / "hdf5_DS_he_train.h5"
    bounds = shard_ranges(rows, 4)
    write_parts(final, rows, bounds, full)

    info = merge_projection_shards(final, expected_rows=rows, cleanup=True)
    assert info["rows"] == rows and info["parts"] == 4

    with h5py.File(final, "r") as merged, h5py.File(reference, "r") as ref:
        assert set(merged.keys()) == set(ref.keys())
        for name in ref.keys():
            assert np.array_equal(merged[name][:], ref[name][:]), f"{name} differs"
            assert merged[name].dtype == ref[name].dtype

    assert not find_parts(final), "--cleanup should have removed the parts"


def test_merge_is_order_independent(tmp_path):
    """Parts are located by glob, whose order is arbitrary. Row order must come
    from the filenames, not from whatever the filesystem hands back."""
    rows = 500
    full = write_full(tmp_path / "reference.h5", rows)
    final = tmp_path / "hdf5_DS_he_train.h5"
    bounds = shard_ranges(rows, 5)
    write_parts(final, rows, bounds, full)

    merge_projection_shards(final, expected_rows=rows)
    with h5py.File(final, "r") as f:
        # Row i must still hold row i's values.
        assert np.array_equal(f["img_h_latent"][:, 0], np.arange(rows, dtype=np.float32))
        assert np.array_equal(f["samples"][:], np.arange(rows, dtype=np.int64))


def test_a_missing_shard_is_a_gap_not_a_short_file(tmp_path):
    rows = 1000
    full = write_full(tmp_path / "reference.h5", rows)
    final = tmp_path / "hdf5_DS_he_train.h5"
    bounds = shard_ranges(rows, 4)
    write_parts(final, rows, bounds, full, skip=(1,))

    msg = _fails(lambda: merge_projection_shards(final, expected_rows=rows),
                 contains="Gap in coverage")
    assert "250" in msg  # names the rows nobody supplied
    assert not final.exists(), "nothing may be written when coverage is incomplete"


def test_a_missing_final_shard_is_caught_by_the_row_count(tmp_path):
    """This one has no gap — the parts are contiguous from zero. Only the input
    row count reveals that the tail is absent."""
    rows = 1000
    full = write_full(tmp_path / "reference.h5", rows)
    final = tmp_path / "hdf5_DS_he_train.h5"
    bounds = shard_ranges(rows, 4)
    write_parts(final, rows, bounds, full, skip=(3,))

    _fails(lambda: merge_projection_shards(final, expected_rows=rows),
           contains="last shard is missing")
    # And without the input to check against, it merges what it has — which is
    # exactly why the merge job is always passed --input-h5.
    info = merge_projection_shards(final, expected_rows=None)
    assert info["rows"] == 750


def test_a_part_that_died_mid_write_is_rejected(tmp_path):
    rows = 1000
    full = write_full(tmp_path / "reference.h5", rows)
    final = tmp_path / "hdf5_DS_he_train.h5"
    bounds = shard_ranges(rows, 4)
    write_parts(final, rows, bounds, full, truncate=(2, 100))

    _fails(lambda: merge_projection_shards(final, expected_rows=rows),
           contains="incomplete")
    assert not final.exists()


def test_overlapping_parts_are_rejected(tmp_path):
    rows = 1000
    full = write_full(tmp_path / "reference.h5", rows)
    final = tmp_path / "hdf5_DS_he_train.h5"
    write_parts(final, rows, [(0, 600), (500, 1000)], full)

    _fails(lambda: merge_projection_shards(final, expected_rows=rows),
           contains="overlaps")


def test_parts_from_different_runs_are_rejected(tmp_path):
    """A part encoded at a different z_dim concatenates cleanly into nonsense
    unless the trailing shape is checked."""
    rows = 1000
    full = write_full(tmp_path / "reference.h5", rows)
    final = tmp_path / "hdf5_DS_he_train.h5"
    bounds = shard_ranges(rows, 2)
    write_parts(final, rows, bounds, full)

    odd = shard_output_path(final, *bounds[1])
    with h5py.File(odd, "w") as f:
        n = bounds[1][1] - bounds[1][0]
        f.create_dataset("img_h_latent", data=np.zeros((n, H_DIM), np.float32))
        f.create_dataset("img_z_latent", data=np.zeros((n, 64), np.float32))  # wrong z_dim
        # Carried datasets deliberately match the other part exactly, so the
        # z_dim is the only defect and the assertion cannot pass for the wrong
        # reason.
        f.create_dataset("samples", data=full["samples"][bounds[1][0]:bounds[1][1]])
        f.create_dataset("slides", data=full["slides"][bounds[1][0]:bounds[1][1]])
        f.create_dataset("tiles", data=full["tiles"][bounds[1][0]:bounds[1][1]])

    _fails(lambda: merge_projection_shards(final, expected_rows=rows),
           contains="not parts of one run")


def test_partial_merge_leaves_nothing_behind(tmp_path):
    """A merge killed halfway must not leave a full-sized, part-zeroed file at
    the path the encoder treats as 'already done'."""
    rows = 100
    full = write_full(tmp_path / "reference.h5", rows)
    final = tmp_path / "hdf5_DS_he_train.h5"
    write_parts(final, rows, shard_ranges(rows, 2), full)

    # Make the second part unreadable after validation has passed.
    parts = find_parts(final)
    parts[1][2].chmod(0o000)
    try:
        try:
            merge_projection_shards(final, expected_rows=rows)
        except (OSError, PermissionError):
            pass
    finally:
        parts[1][2].chmod(0o644)

    assert not final.exists(), "no output may survive a failed merge"
    assert not final.with_name(final.name + ".merging").exists(), "temp file left behind"


def test_existing_output_is_not_silently_replaced(tmp_path):
    rows = 100
    full = write_full(tmp_path / "reference.h5", rows)
    final = tmp_path / "hdf5_DS_he_train.h5"
    write_parts(final, rows, shard_ranges(rows, 2), full)
    final.write_bytes(b"not really an h5")

    _fails(lambda: merge_projection_shards(final, expected_rows=rows), contains="already exists")
    assert final.read_bytes() == b"not really an h5"

    merge_projection_shards(final, expected_rows=rows, force=True)
    with h5py.File(final, "r") as f:
        assert f["samples"].shape[0] == rows


def test_no_parts_at_all_says_so(tmp_path):
    _fails(lambda: merge_projection_shards(tmp_path / "hdf5_DS_he_train.h5"),
           contains="No part files found")


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_shard_test_"))
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
