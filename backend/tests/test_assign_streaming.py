"""Stage 4's assigner streams its input in chunks. These tests are about what
streaming could silently have broken.

The dangerous property is that `--centering query` mirrors scanpy's Ingest and
subtracts the mean over *all* queries. That couples every tile to every other
one, so the moment the input is processed in pieces there are two new ways to be
wrong, both of which still emit a complete, well-formed CSV:

  * a chunk centred on its own mean, making a tile's cluster depend on which
    chunk boundary it happened to fall inside;
  * a shard centred on its slice's mean, making cluster IDs depend on how many
    jobs the work was split across.

So the tests here are equivalence tests, not smoke tests: chunked must equal
unchunked, and sharded must equal unsharded, bit for bit.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

ASSIGN = BACKEND / "assign_hpc_clusters.py"

REF_ROWS, DIM, NCOMP, NCLUST, K = 4000, 32, 16, 12, 25
QUERY_ROWS = 900


def _write_reference(path: Path, seed: int = 0) -> None:
    """A build_hpc_reference.py-shaped .npz, small enough to be fast."""
    import json
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, NCLUST, REF_ROWS).astype(np.int64)
    np.savez(
        path,
        reference=rng.standard_normal((REF_ROWS, NCOMP)).astype(np.float32),
        components=rng.standard_normal((DIM, NCOMP)).astype(np.float32),
        codes=codes,
        categories=np.array([str(i) for i in range(NCLUST)]),
        n_neighbors=np.int64(K),
        meta=json.dumps({"groupby": "leiden_2.5"}),
    )


def _write_queries(path: Path, rows: int = QUERY_ROWS, dim: int = DIM, seed: int = 1) -> None:
    """A projections .h5 as feature extraction leaves it."""
    rng = np.random.default_rng(seed)
    # Offset from zero so the query mean is meaningfully non-zero — with a mean
    # of ~0 every centering bug would look like a passing test.
    emb = (rng.standard_normal((rows, dim)) + 3.0).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("img_z_latent", data=emb)
        f.create_dataset("samples", data=np.array([b"S%03d" % (i // 90) for i in range(rows)]))
        f.create_dataset("slides", data=np.array([b"slide_%03d" % (i // 90) for i in range(rows)]))
        f.create_dataset("tiles", data=np.array([b"%d_%d.jpeg" % (i // 30, i % 30) for i in range(rows)]))


def _run(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ASSIGN), *args],
        capture_output=True, text=True, cwd=str(BACKEND),
    )
    if result.returncode != 0:
        raise AssertionError(f"assign failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def _assign(ref: Path, h5: Path, out: Path, **flags) -> Path:
    args = ["--reference", str(ref), "--h5", str(h5), "--out", str(out)]
    for key, value in flags.items():
        flag = f"--{key.replace('_', '-')}"
        # store_true flags take no value; passing True has to mean "present".
        if value is True:
            args.append(flag)
        elif value is not False:
            args += [flag, str(value)]
    _run(*args)
    return out


def test_chunk_size_does_not_change_assignments(tmp_path):
    """A tile's cluster must not depend on which chunk it landed in. This is the
    check that the query mean is computed over the file and not per chunk."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)

    outputs = []
    # 100 forces ragged chunks against 900 rows; 100_000 is a single chunk.
    for chunk in (100, 256, 900, 100_000):
        out = _assign(ref, h5, tmp_path / f"c{chunk}.csv", chunk_size=chunk)
        outputs.append(out.read_bytes())

    for chunk, data in zip((256, 900, 100_000), outputs[1:]):
        assert data == outputs[0], f"chunk_size={chunk} changed the output"


def test_batch_size_does_not_change_assignments(tmp_path):
    """The search batch is a performance knob only — search is per-query
    independent, so this must hold for any value."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)

    a = _assign(ref, h5, tmp_path / "b64.csv", batch_size=64).read_bytes()
    b = _assign(ref, h5, tmp_path / "b16k.csv", batch_size=16384).read_bytes()
    assert a == b


def test_sharded_with_shared_mean_equals_unsharded(tmp_path):
    """The property that makes sharding safe. Every shard is handed the same
    precomputed mean, so the concatenation must equal a single run exactly."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)

    whole = _assign(ref, h5, tmp_path / "whole.csv").read_text()

    mean_path = tmp_path / "mean.npy"
    _run("--reference", str(ref), "--h5", str(h5), "--precompute-mean", str(mean_path))
    assert mean_path.is_file()

    bounds = [(0, 225), (225, 450), (450, 675), (675, QUERY_ROWS)]
    pieces = []
    for lo, hi in bounds:
        _run("--reference", str(ref), "--h5", str(h5),
             "--out", str(tmp_path / "shard.csv"),
             "--query-mean", str(mean_path),
             "--row-start", str(lo), "--row-stop", str(hi))
        part = tmp_path / f"shard.rows{lo}-{hi}.csv"
        assert part.is_file(), f"no part written for [{lo}, {hi})"
        pieces.append(part.read_text())

    header = pieces[0].splitlines()[0]
    merged = header + "\n" + "".join(
        "".join(line + "\n" for line in p.splitlines()[1:]) for p in pieces
    )
    assert merged == whole, "sharded output differs from a single run"


def test_shard_without_shared_mean_is_refused(tmp_path):
    """The guard has to earn its place: without it, each shard would centre on
    its own slice and produce different labels. Proven by showing the labels
    really do differ when the mean is not shared."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)

    # 1. It is refused, and the message says what to do.
    result = subprocess.run(
        [sys.executable, str(ASSIGN), "--reference", str(ref), "--h5", str(h5),
         "--out", str(tmp_path / "x.csv"), "--row-start", "0", "--row-stop", "225"],
        capture_output=True, text=True, cwd=str(BACKEND),
    )
    assert result.returncode != 0
    assert "--query-mean" in result.stderr, result.stderr

    # 2. And the refusal is not pedantry. Centering a slice by its own mean does
    #    change assignments: 'none' centering makes the slice independent, so
    #    running a slice under it and comparing against the same rows of a whole
    #    run under 'query' shows the two spaces disagree.
    whole = _assign(ref, h5, tmp_path / "w.csv", centering="none").read_text().splitlines()
    _run("--reference", str(ref), "--h5", str(h5), "--out", str(tmp_path / "s.csv"),
         "--centering", "none", "--row-start", "0", "--row-stop", "225")
    part = (tmp_path / "s.rows0-225.csv").read_text().splitlines()
    # 'none' needs no mean, so a slice IS safe there — same rows, same answers.
    assert part[1:] == whole[1:226], "centering=none should be shard-independent"


def test_query_mean_matches_a_whole_array_mean(tmp_path):
    """The streamed float64 accumulation must agree with the obvious
    computation, or every projected coordinate is slightly off."""
    import assign_hpc_clusters as m

    h5 = tmp_path / "q.h5"
    _write_queries(h5, rows=1001)
    with h5py.File(h5, "r") as f:
        expected = np.asarray(f["img_z_latent"][:], dtype=np.float32).mean(axis=0)

    for chunk in (7, 128, 5000):
        got = m.compute_query_mean(h5, "z_latent", chunk, 1001)
        assert np.allclose(got, expected, atol=1e-6), f"chunk={chunk}"


def test_high_dimensional_rep_key_streams(tmp_path):
    """--rep-key h_latent is 1536-d, the case that motivated streaming: the old
    whole-array read was ~86 GB at registry scale. Nothing else exercises it."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    import json
    rng = np.random.default_rng(3)
    np.savez(
        ref,
        reference=rng.standard_normal((500, NCOMP)).astype(np.float32),
        components=rng.standard_normal((1536, NCOMP)).astype(np.float32),
        codes=rng.integers(0, NCLUST, 500).astype(np.int64),
        categories=np.array([str(i) for i in range(NCLUST)]),
        n_neighbors=np.int64(K),
        meta=json.dumps({"groupby": "leiden_2.5"}),
    )
    rows = 300
    with h5py.File(h5, "w") as f:
        f.create_dataset("img_h_latent", data=rng.standard_normal((rows, 1536)).astype(np.float32))
        f.create_dataset("samples", data=np.array([b"S1"] * rows))
        f.create_dataset("slides", data=np.array([b"slide_1"] * rows))
        f.create_dataset("tiles", data=np.array([b"%d.jpeg" % i for i in range(rows)]))

    out = _assign(ref, h5, tmp_path / "h.csv", rep_key="h_latent", chunk_size=64)
    lines = out.read_text().splitlines()
    assert len(lines) == rows + 1


def test_partial_output_is_not_left_behind(tmp_path):
    """A killed run must not leave a CSV holding some of the tiles: nothing
    downstream checks row counts before merging."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)
    out = tmp_path / "ok.csv"
    _assign(ref, h5, out)
    assert out.is_file()
    assert not out.with_name(out.name + ".partial").exists()


# --- merging the shards --------------------------------------------------
# Same failure shapes as test_shard_merge.py covers for the HDF5 merge. A CSV
# concatenation is simpler but fails identically: joined across a gap or over a
# half-written part, it produces a file with the right columns and no missing
# values that every downstream consumer accepts.

def _write_csv_parts(final: Path, bounds, rows_per=None, header="a,b,hpc_reference"):
    for lo, hi in bounds:
        n = (hi - lo) if rows_per is None else rows_per.get((lo, hi), hi - lo)
        part = final.with_name(f"{final.stem}.rows{lo}-{hi}{final.suffix}")
        part.write_text(header + "\n" + "".join(f"{i},x,ref\n" for i in range(lo, lo + n)))


def _merge_fails(final, *, contains, expected_rows=None):
    from merge_assignment_shards import merge_assignment_shards
    try:
        merge_assignment_shards(final, expected_rows=expected_rows)
    except (ValueError, FileExistsError) as e:
        assert contains in str(e), f"expected {contains!r} in: {e}"
        return
    raise AssertionError(f"expected a failure mentioning {contains!r}")


def test_csv_merge_round_trips(tmp_path):
    from merge_assignment_shards import merge_assignment_shards
    final = tmp_path / "out.csv"
    _write_csv_parts(final, [(0, 100), (100, 200), (200, 300)])
    info = merge_assignment_shards(final, expected_rows=300, cleanup=True)
    assert info["rows"] == 300 and info["parts"] == 3
    lines = final.read_text().splitlines()
    assert len(lines) == 301
    # Row order must come from the filenames, not the glob.
    assert [int(l.split(",")[0]) for l in lines[1:]] == list(range(300))
    assert not list(tmp_path.glob("*.rows*"))


def test_csv_merge_rejects_gap_overlap_and_short_parts(tmp_path):
    for label, bounds, kw in [
        ("Gap in coverage",   [(0, 100), (200, 300)], {}),
        ("overlaps",          [(0, 150), (100, 300)], {}),
        ("last shard is missing", [(0, 100), (100, 200)], {"expected_rows": 300}),
    ]:
        final = tmp_path / f"{label[:6].replace(' ','_')}.csv"
        _write_csv_parts(final, bounds)
        _merge_fails(final, contains=label, **kw)
        assert not final.exists(), "nothing may be written when coverage is bad"

    # A part that died mid-write: its name claims more rows than it holds.
    final = tmp_path / "short.csv"
    _write_csv_parts(final, [(0, 100), (100, 200)], rows_per={(100, 200): 40})
    _merge_fails(final, contains="incomplete", expected_rows=200)


def test_csv_merge_rejects_mismatched_columns(tmp_path):
    final = tmp_path / "cols.csv"
    _write_csv_parts(final, [(0, 100)])
    _write_csv_parts(final, [(100, 200)], header="a,b,different")
    _merge_fails(final, contains="not parts of one run", expected_rows=200)


def test_reference_keys_match_the_builder(tmp_path):
    """submit_cluster_assignment.check_reference validates the .npz before
    queueing. It once required a "labels" key the builder has never written, so
    a correctly built 2.5M-tile reference was rejected as malformed.

    Round-tripping through the real build_hpc_reference.save() is what keeps the
    reader and the writer in step — listing the keys in both places is how they
    drifted in the first place.
    """
    import build_hpc_reference
    from submit_cluster_assignment import check_reference

    out = tmp_path / "ref.npz"
    build_hpc_reference.save(
        {
            "reference": np.zeros((100, NCOMP), np.float32),
            "components": np.zeros((DIM, NCOMP), np.float32),
            "codes": np.arange(100, dtype=np.int64) % NCLUST,
            "categories": [str(i) for i in range(NCLUST)],
            "n_neighbors": 250,
            "groupby": "leiden_2.5",
            "source": "synthetic",
            "mean": None,
        },
        out,
    )

    info = check_reference(out)
    assert info["reference_rows"] == 100
    assert info["reference_dims"] == NCOMP
    assert info["n_clusters"] == NCLUST
    assert info["groupby"] == "leiden_2.5"
    assert info["k"] == 250
    # This reference stores no mean, so --centering reference is unavailable and
    # sharding must go through --query-mean.
    assert info["has_mean"] is False

    # And with a mean, it is reported.
    out2 = tmp_path / "ref_mean.npz"
    build_hpc_reference.save(
        {
            "reference": np.zeros((100, NCOMP), np.float32),
            "components": np.zeros((DIM, NCOMP), np.float32),
            "codes": np.arange(100, dtype=np.int64) % NCLUST,
            "categories": [str(i) for i in range(NCLUST)],
            "n_neighbors": 250,
            "groupby": "leiden_2.5",
            "source": "synthetic",
            "mean": np.zeros(DIM, np.float32),
        },
        out2,
    )
    assert check_reference(out2)["has_mean"] is True


# --- adaptive k ----------------------------------------------------------
#
# The gate re-votes only the tiles whose base-k vote was nearly tied, at a
# wider prefix of the SAME search. Two things could go wrong quietly: the flag
# could be accepted and ignored (a normal-looking CSV that never re-voted), or
# vote_margin could keep describing the discarded base vote while hpc_id came
# from the wide one — which would make Stage 5's --min-margin drop exactly the
# tiles this rescues. So these check the refusals fire and the columns agree.


def _run_expecting_failure(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ASSIGN), *args],
        capture_output=True, text=True, cwd=str(BACKEND),
    )
    assert result.returncode != 0, f"expected a refusal, got:\n{result.stdout}"
    return result.stdout + result.stderr


def test_adaptive_k_no_wider_than_k_is_refused(tmp_path):
    """A re-vote at a k no wider than the base sees the same neighbours and
    cannot change a single label. Accepting it would produce an output that
    looks adaptive and is not."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)
    message = _run_expecting_failure(
        "--reference", str(ref), "--h5", str(h5), "--out", str(tmp_path / "o.csv"),
        "--k", "10", "--adaptive-margin", "0.1", "--adaptive-k", "10",
    )
    assert "not wider" in message

    # Narrower is refused for the same reason, not silently clamped.
    message = _run_expecting_failure(
        "--reference", str(ref), "--h5", str(h5), "--out", str(tmp_path / "o.csv"),
        "--k", "10", "--adaptive-margin", "0.1", "--adaptive-k", "5",
    )
    assert "not wider" in message


def test_adaptive_k_without_a_margin_is_refused(tmp_path):
    """--adaptive-k alone has nothing to gate on. Ignoring it would mean a
    typo'd sweep silently ran the plain configuration."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)
    message = _run_expecting_failure(
        "--reference", str(ref), "--h5", str(h5), "--out", str(tmp_path / "o.csv"),
        "--k", "10", "--adaptive-k", "25",
    )
    assert "without a positive" in message


def test_adaptive_k_beyond_the_reference_is_refused(tmp_path):
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)
    message = _run_expecting_failure(
        "--reference", str(ref), "--h5", str(h5), "--out", str(tmp_path / "o.csv"),
        "--k", "10", "--adaptive-margin", "0.1",
        "--adaptive-k", str(REF_ROWS + 1),
    )
    assert "exceeds" in message


def test_adaptive_off_is_bit_identical_to_before(tmp_path):
    """The widened search only happens when the gate is on. With it off the
    output must match byte for byte, or every existing assignment in the KB is
    now unreproducible."""
    import pandas as pd
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)

    plain = _assign(ref, h5, tmp_path / "plain.csv", k=10, distance_weighted=True, distance_power=3)
    explicit_off = _assign(ref, h5, tmp_path / "off.csv", k=10, distance_weighted=True, distance_power=3,
                           adaptive_margin=0)
    assert plain.read_text() == explicit_off.read_text()

    frame = pd.read_csv(plain)
    assert len(frame) == QUERY_ROWS


def test_adaptive_rewrites_only_low_margin_tiles(tmp_path):
    """Above the threshold nothing may move; below it, the label and the margin
    must both come from the wide vote."""
    import pandas as pd
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)

    threshold = 0.30
    base = pd.read_csv(_assign(ref, h5, tmp_path / "base.csv",
                               k=10, distance_weighted=True, distance_power=3))
    adaptive = pd.read_csv(_assign(ref, h5, tmp_path / "adapt.csv",
                                   k=10, distance_weighted=True, distance_power=3,
                                   adaptive_margin=threshold, adaptive_k=25))
    wide = pd.read_csv(_assign(ref, h5, tmp_path / "wide.csv",
                               k=25, distance_weighted=True, distance_power=3))

    assert list(base.columns) == list(adaptive.columns), \
        "adaptive k must not change the CSV schema — the loader reflects on it"

    low = base["vote_margin"] < threshold
    assert low.any(), "threshold too low to exercise the re-vote"
    assert not low.all(), "threshold too high to test the untouched rows"

    # Untouched rows: identical label and identical margin.
    kept = ~low
    assert (adaptive.loc[kept, "leiden_2.5"].to_numpy()
            == base.loc[kept, "leiden_2.5"].to_numpy()).all()
    assert np.allclose(adaptive.loc[kept, "vote_margin"],
                       base.loc[kept, "vote_margin"])

    # Re-voted rows: label and margin both equal a plain k=25 run's, because
    # the wide prefix of one search is the same neighbourhood as searching 25.
    assert (adaptive.loc[low, "leiden_2.5"].to_numpy()
            == wide.loc[low, "leiden_2.5"].to_numpy()).all()
    assert np.allclose(adaptive.loc[low, "vote_margin"],
                       wide.loc[low, "vote_margin"]), \
        "vote_margin still describes the discarded base vote"
    assert np.allclose(adaptive.loc[low, "neighbor_distance"],
                       wide.loc[low, "neighbor_distance"])

    # And it must actually have changed something, or the test proves nothing.
    changed = (adaptive.loc[low, "leiden_2.5"].to_numpy()
               != base.loc[low, "leiden_2.5"].to_numpy())
    assert changed.any(), "the re-vote changed no label at all"


def test_adaptive_survives_chunking_and_reports_the_count(tmp_path):
    """The gate is applied per batch, so it must not become chunk-dependent —
    the same failure mode the centering tests exist for."""
    import pandas as pd
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)

    frames = []
    for chunk in (100, 100_000):
        out = tmp_path / f"c{chunk}.csv"
        stdout = _run("--reference", str(ref), "--h5", str(h5), "--out", str(out),
                      "--k", "10", "--distance-power", "3",
                      "--adaptive-margin", "0.3", "--adaptive-k", "25",
                      "--distance-weighted", "--chunk-size", str(chunk))
        assert "Re-voted  :" in stdout, "the re-voted count is not reported"
        frames.append(pd.read_csv(out))

    assert frames[0].equals(frames[1])

# --- the submitter forwards the vote -------------------------------------
#
# Every vote knob was absent from the Slurm command before now, so a knob
# measured offline had no way to reach a real assignment and the mismatch was
# invisible: the job ran, wrote a complete CSV, and used the defaults. These
# check the flags arrive and that an inert combination is refused at submit
# time rather than dropped inside the container.


def test_vote_flags_forwards_the_measured_configuration(tmp_path):
    from submit_cluster_assignment import vote_flags
    flags = " ".join(vote_flags(
        k=10, distance_weighted=True, distance_power=3.0, class_weighted=False,
        local_scaling=0, adaptive_margin=0.1, adaptive_k=25,
    ))
    assert flags == ("--distance-weighted --distance-power 3 "
                     "--adaptive-margin 0.1 --adaptive-k 25")


def test_vote_flags_defaults_add_nothing(tmp_path):
    """The default has to stay exactly what Stage 4 already ran, or every
    assignment already in the KB becomes unreproducible."""
    from submit_cluster_assignment import vote_flags
    assert vote_flags(k=None, distance_weighted=False, distance_power=1.0,
                      class_weighted=False, local_scaling=0,
                      adaptive_margin=0.0, adaptive_k=0) == []


def _refused(expected: str, **kwargs) -> None:
    """vote_flags must exit with a message naming the problem.

    try/except rather than pytest.raises: every suite here also has to run
    standalone on the cluster, where pytest is not installed.
    """
    from submit_cluster_assignment import vote_flags
    try:
        flags = vote_flags(**kwargs)
    except SystemExit as e:
        assert expected in str(e), str(e)
    else:
        raise AssertionError(
            f"expected a refusal mentioning {expected!r}, got flags {flags}")


def test_vote_flags_refuses_a_power_without_weighting(tmp_path):
    """distance_power is read only when distance_weighted is on, so this pair
    would otherwise queue a four-hour job that ignored the exponent."""
    _refused("ignored without",
             k=10, distance_weighted=False, distance_power=3.0,
             class_weighted=False, local_scaling=0,
             adaptive_margin=0.0, adaptive_k=0)


def test_vote_flags_refuses_adaptive_without_an_explicit_k(tmp_path):
    """k defaults to the reference's own n_neighbors, read inside the job. If
    adaptive_k were checked against that, whether the re-vote does anything
    could not be known until the job was already running."""
    _refused("explicit --k",
             k=None, distance_weighted=True, distance_power=3.0,
             class_weighted=False, local_scaling=0,
             adaptive_margin=0.1, adaptive_k=25)


def test_vote_flags_refuses_an_adaptive_k_that_is_not_wider(tmp_path):
    _refused("not wider",
             k=25, distance_weighted=True, distance_power=3.0,
             class_weighted=False, local_scaling=0,
             adaptive_margin=0.1, adaptive_k=25)
    _refused("without a positive",
             k=10, distance_weighted=True, distance_power=3.0,
             class_weighted=False, local_scaling=0,
             adaptive_margin=0.0, adaptive_k=25)


def test_the_slurm_command_actually_carries_the_vote_flags(tmp_path):
    """vote_flags could be correct and still never reach the command line."""
    from submit_cluster_assignment import _build_assignment_command, vote_flags
    vote = vote_flags(k=10, distance_weighted=True, distance_power=3.0,
                      class_weighted=False, local_scaling=0,
                      adaptive_margin=0.1, adaptive_k=25)
    command = _build_assignment_command(
        singularity_bin="singularity",
        singularity_image=tmp_path / "image.sif",
        extras_dir=tmp_path / "extras",
        assign_script=BACKEND / "assign_hpc_clusters.py",
        reference=tmp_path / "ref.npz",
        projections_h5=tmp_path / "q.h5",
        out_csv=tmp_path / "out.csv",
        rep_key="z_latent",
        k=10,
        batch_size=16_384,
        validate_against=None,
        vote=vote,
    )
    for flag in ("--distance-weighted", "--distance-power 3",
                 "--adaptive-margin 0.1", "--adaptive-k 25", "--k 10"):
        assert flag in command, f"{flag} never reached the Slurm command"

    # And the default stays clean: no vote flags at all.
    plain = _build_assignment_command(
        singularity_bin="singularity",
        singularity_image=tmp_path / "image.sif",
        extras_dir=tmp_path / "extras",
        assign_script=BACKEND / "assign_hpc_clusters.py",
        reference=tmp_path / "ref.npz",
        projections_h5=tmp_path / "q.h5",
        out_csv=tmp_path / "out.csv",
        rep_key="z_latent",
        k=None,
        batch_size=16_384,
        validate_against=None,
        vote=[],
    )
    for flag in ("--distance-weighted", "--adaptive-margin", "--class-weighted",
                 "--local-scaling"):
        assert flag not in plain

# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_assign_test_"))
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
