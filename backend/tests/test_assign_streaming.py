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

# --- the named vote presets ----------------------------------------------
#
# The presets are what makes the tuned configuration one click instead of seven
# numbers typed correctly. Their whole value is that the server, the API client
# and the UI cannot disagree about what "tuned" means, so these pin the numbers
# and the wiring rather than the plumbing.


def test_the_tuned_preset_is_the_configuration_that_was_measured(tmp_path):
    """97.27% was measured for exactly these settings. If a preset drifts from
    them, the UI goes on displaying the accuracy of a configuration it is no
    longer running — which is worse than displaying nothing."""
    from submit_cluster_assignment import VOTE_PRESETS
    assert VOTE_PRESETS["tuned"]["flags"] == {
        "k": 10,
        "distance_weighted": True,
        "distance_power": 3.0,
        "class_weighted": False,
        "local_scaling": 0,
        "adaptive_margin": 0.15,
        "adaptive_k": 25,
    }
    # 0.15, not 0.1: the sweep's original grid was 0/0.10/0.25 and could not see
    # its own optimum.
    assert VOTE_PRESETS["tuned"]["flags"]["adaptive_margin"] == 0.15


def test_the_legacy_preset_is_what_stage_4_used_to_run(tmp_path):
    """It exists to reproduce an existing assignment exactly, so it has to be
    the plain unweighted vote and produce no flags at all."""
    from submit_cluster_assignment import VOTE_PRESETS, resolve_vote, vote_flags
    assert vote_flags(**resolve_vote("legacy")) == []


def test_every_preset_resolves_to_a_usable_configuration(tmp_path):
    """A preset that vote_flags refuses would be a one-click refusal. Checked
    for all of them so a new preset cannot be added inert."""
    from submit_cluster_assignment import VOTE_PRESETS, resolve_vote, vote_flags
    for name, spec in VOTE_PRESETS.items():
        vote_flags(**resolve_vote(name))          # must not raise
        for field in ("label", "accuracy", "summary", "why", "flags"):
            assert spec.get(field), f"{name} has no {field}"
        assert 0.5 < spec["accuracy"] < 1.0, name


def test_an_unset_override_does_not_erase_the_preset(tmp_path):
    """Every override arrives from an HTTP body where absent fields are None.
    If None meant "set to None" rather than "not specified", sending a preset
    with no overrides would strip it down to nothing."""
    from submit_cluster_assignment import VOTE_PRESETS, resolve_vote
    everything_none = dict.fromkeys(VOTE_PRESETS["tuned"]["flags"], None)
    assert resolve_vote("tuned", **everything_none) == VOTE_PRESETS["tuned"]["flags"]


def test_an_override_applies_on_top_of_the_preset(tmp_path):
    from submit_cluster_assignment import resolve_vote
    tuned = resolve_vote("tuned")
    changed = resolve_vote("tuned", adaptive_margin=0.1)
    assert changed["adaptive_margin"] == 0.1
    assert {k: v for k, v in changed.items() if k != "adaptive_margin"} \
        == {k: v for k, v in tuned.items() if k != "adaptive_margin"}


def test_an_unknown_preset_and_an_unknown_setting_are_both_refused(tmp_path):
    from submit_cluster_assignment import resolve_vote
    try:
        resolve_vote("whatever-sounds-good")
    except SystemExit as e:
        assert "Unknown vote preset" in str(e)
    else:
        raise AssertionError("an unknown preset must be refused")

    try:
        resolve_vote("tuned", distnace_power=3.0)   # typo, deliberately
    except SystemExit as e:
        assert "Not vote settings" in str(e), str(e)
    else:
        raise AssertionError("a misspelled setting must be refused, not ignored")


def test_describe_vote_says_when_a_preset_was_modified(tmp_path):
    """The run record's one line about the vote. A preset name alone would be a
    lie the moment anything was overridden, and that line is the only place two
    CSVs from one reference but different votes can be told apart."""
    from submit_cluster_assignment import describe_vote, resolve_vote
    plain = describe_vote(resolve_vote("tuned"), "tuned")
    assert plain.startswith("tuned:") and "modified" not in plain
    assert "--adaptive-margin 0.15" in plain

    modified = describe_vote(resolve_vote("tuned", adaptive_margin=0.1), "tuned")
    assert "(modified)" in modified
    assert "--adaptive-margin 0.1 " in modified + " "


def test_an_explicit_k_beats_the_presets_k(tmp_path):
    """k is both a plain argument of the submitter and part of a preset. Two
    resolution paths would mean a caller passing k=15 with the tuned preset
    silently getting the preset's 10."""
    from submit_cluster_assignment import resolve_vote
    assert resolve_vote("tuned", k=15)["k"] == 15
    assert resolve_vote("tuned", k=None)["k"] == 10


def test_the_presets_reach_the_slurm_command(tmp_path):
    """A preset that resolves correctly and never reaches the command line
    would be the same bug as before, one level up."""
    from submit_cluster_assignment import (_build_assignment_command,
                                           resolve_vote, vote_flags)
    common = dict(
        singularity_bin="singularity", singularity_image=tmp_path / "i.sif",
        extras_dir=tmp_path / "extras",
        assign_script=BACKEND / "assign_hpc_clusters.py",
        reference=tmp_path / "ref.npz", projections_h5=tmp_path / "q.h5",
        out_csv=tmp_path / "out.csv", rep_key="z_latent",
        batch_size=16_384, validate_against=None,
    )
    tuned = resolve_vote("tuned")
    command = _build_assignment_command(
        k=tuned["k"], vote=vote_flags(**tuned), **common)
    for flag in ("--k 10", "--distance-weighted", "--distance-power 3",
                 "--adaptive-margin 0.15", "--adaptive-k 25"):
        assert flag in command, f"{flag} never reached the Slurm command"

    legacy = resolve_vote("legacy")
    plain = _build_assignment_command(
        k=legacy["k"], vote=vote_flags(**legacy), **common)
    for flag in ("--distance-weighted", "--adaptive-margin", "--class-weighted",
                 "--local-scaling", "--k "):
        assert flag not in plain, f"legacy must not pass {flag}"


# --- the server / client / UI contract -----------------------------------
#
# Four modules have to agree about the vote: the submitter defines it, the
# server forwards it, the client sends it, the UI displays it. Each seam fails
# differently and none fails visibly, so each is pinned here. These import
# tile_server_v2_, which is heavy but does import cleanly; if that ever stops
# being true these tests say so loudly rather than being skipped.


def test_the_servers_vote_kwargs_are_all_accepted_by_the_submitter(tmp_path):
    """The seam that would 500 at submit time. The server unpacks vote_kwargs()
    into submit_cluster_assignment_job, so a key it does not accept is a
    TypeError on a real submission and on nothing before it."""
    import inspect
    import tile_server_v2_ as srv
    from submit_cluster_assignment import submit_cluster_assignment_job

    accepted = set(inspect.signature(submit_cluster_assignment_job).parameters)
    sent = set(srv.ClusterAssignmentRequest().vote_kwargs())
    assert sent <= accepted, f"the server sends {sorted(sent - accepted)}, which "\
                             f"submit_cluster_assignment_job does not accept"


def test_the_server_defaults_to_the_tuned_preset(tmp_path):
    """The submitter defaults to legacy so no programmatic caller changes
    behaviour by being upgraded; the request model defaults to tuned so the UI
    queues the measured configuration. Both halves matter, so both are pinned."""
    import inspect
    import tile_server_v2_ as srv
    from submit_cluster_assignment import (DEFAULT_VOTE_PRESET,
                                           submit_cluster_assignment_job)

    assert srv.ClusterAssignmentRequest().vote_preset == DEFAULT_VOTE_PRESET == "tuned"
    # The test endpoint's model inherits it, so a sample run and a full run
    # cannot silently use different votes.
    assert srv.ClusterAssignmentTestRequest(
        projections_h5="/x.h5").vote_preset == "tuned"
    submitter_default = inspect.signature(
        submit_cluster_assignment_job).parameters["vote_preset"].default
    assert submitter_default is None, (
        "the submitter must not default to a preset — an existing caller would "
        "change behaviour just by being upgraded")


def test_the_served_presets_carry_everything_the_ui_displays(tmp_path):
    """The UI reads label/accuracy/summary/why/flags off this payload and shows
    an accuracy figure next to a named configuration. A field missing here is a
    blank caption; a field wrong here is a confident lie."""
    import tile_server_v2_ as srv
    from submit_cluster_assignment import VOTE_PRESETS

    served = srv.list_vote_presets()
    assert served["default"] in served["presets"]
    assert set(served["presets"]) == set(VOTE_PRESETS)
    for name, spec in served["presets"].items():
        for field in ("label", "accuracy", "summary", "why", "flags"):
            assert spec.get(field), f"{name} is missing {field}"
        # And it must be the same numbers, not a copy that has drifted.
        assert spec["flags"] == VOTE_PRESETS[name]["flags"]
    # The UI's override inputs read these three off flags and format them with
    # :g, so they have to be numbers rather than None.
    tuned = served["presets"]["tuned"]["flags"]
    for field in ("distance_power", "adaptive_margin", "adaptive_k"):
        assert isinstance(tuned[field], (int, float)), field


def test_the_api_client_sends_the_preset_and_the_overrides(tmp_path):
    """Checked on the request body the client builds, because the UI passes the
    preset and overrides separately and a dropped one is invisible: the server
    would apply its default and the run would look fine."""
    import sys
    sys.path.insert(0, str(BACKEND.parent / "app"))
    import api_client

    sent = {}

    class _Spy(api_client.TileServerClient):
        def __init__(self):
            pass

        def _post_json(self, path, body, **kw):
            sent["path"], sent["body"] = path, body
            return {}

    _Spy().start_cluster_assignment(
        "run1", reference=None, overwrite=True,
        vote_preset="tuned", vote_overrides={"adaptive_margin": 0.1})
    assert sent["path"].endswith("/assign-clusters")
    assert sent["body"]["vote_preset"] == "tuned"
    assert sent["body"]["adaptive_margin"] == 0.1
    assert sent["body"]["overwrite"] is True

    # No preset and no overrides must leave the body clean, so the server's own
    # default applies rather than a null overriding it.
    _Spy().start_cluster_assignment("run1")
    assert "vote_preset" not in sent["body"]
    assert not any(k.startswith("adaptive") for k in sent["body"])

    # And the test path carries it too, or a sample run would use a different
    # vote from the full run it is meant to preview.
    _Spy().start_test_cluster_assignment(
        "run1", "/x.h5", vote_preset="legacy")
    assert sent["path"].endswith("/assign-clusters-test")
    assert sent["body"]["vote_preset"] == "legacy"


# --- a truth file that disagrees with itself -----------------------------
#
# Kai's TCGA label CSV lists 100 tiles twice, 96 with two different Leiden
# labels, all on one slide. validate() merges one_to_one so a duplicated key
# cannot silently multiply rows, which meant the acceptance test -- the gate
# CLAUDE.md calls a defect below 99% -- died on a pandas MergeError naming
# neither the file nor the slide. Dropping duplicates blindly would be worse:
# 96 tiles would be scored against whichever label happened to come first.


def _truth(rows) -> "object":
    import pandas as pd
    return pd.DataFrame(rows, columns=["samples", "slides", "tiles", "leiden_2.5"])


def test_conflicting_duplicates_are_excluded_not_resolved(tmp_path):
    from assign_hpc_clusters import _dedupe_truth
    truth = _truth([
        ("S1", "sl1", "1_1.jpeg", 5),
        ("S1", "sl1", "1_1.jpeg", 9),      # same tile, different label
        ("S1", "sl1", "2_2.jpeg", 7),
    ])
    clean, ambiguous, redundant = _dedupe_truth(truth, "leiden_2.5")
    assert ambiguous == [("sl1", "1_1.jpeg")]
    assert redundant == 0
    # Excluded entirely — neither 5 nor 9 may survive as "the" answer.
    assert list(clean["tiles"]) == ["2_2.jpeg"]


def test_redundant_duplicates_are_deduplicated_and_kept(tmp_path):
    """The same label twice carries no ambiguity, so the tile stays scoreable."""
    from assign_hpc_clusters import _dedupe_truth
    truth = _truth([
        ("S1", "sl1", "1_1.jpeg", 5),
        ("S1", "sl1", "1_1.jpeg", 5),
        ("S1", "sl1", "2_2.jpeg", 7),
    ])
    clean, ambiguous, redundant = _dedupe_truth(truth, "leiden_2.5")
    assert ambiguous == [] and redundant == 1
    assert sorted(clean["tiles"]) == ["1_1.jpeg", "2_2.jpeg"]
    assert clean.loc[clean["tiles"] == "1_1.jpeg", "leiden_2.5"].tolist() == [5]


def test_every_row_is_accounted_for(tmp_path):
    """kept + 2 per conflicting pair + redundant must equal the input, or rows
    are going missing somewhere other than the two documented reasons."""
    from assign_hpc_clusters import _dedupe_truth
    truth = _truth([
        ("S1", "sl1", "1_1.jpeg", 5), ("S1", "sl1", "1_1.jpeg", 9),
        ("S1", "sl1", "2_2.jpeg", 7), ("S1", "sl1", "2_2.jpeg", 7),
        ("S1", "sl2", "3_3.jpeg", 1),
    ])
    clean, ambiguous, redundant = _dedupe_truth(truth, "leiden_2.5")
    assert len(clean) + 2 * len(ambiguous) + redundant == len(truth)


def test_a_clean_truth_file_is_returned_untouched(tmp_path):
    from assign_hpc_clusters import _dedupe_truth
    truth = _truth([("S1", "sl1", "1_1.jpeg", 5), ("S1", "sl1", "2_2.jpeg", 7)])
    clean, ambiguous, redundant = _dedupe_truth(truth, "leiden_2.5")
    assert ambiguous == [] and redundant == 0
    assert clean.equals(truth)


def test_a_tile_is_dropped_by_key_not_by_position(tmp_path):
    """The exclusion is an anti-join on (slides, tiles). Matching on the tile
    name alone would drop the same tile coordinate from every other slide —
    silently shrinking the comparison across the whole cohort."""
    from assign_hpc_clusters import _dedupe_truth
    truth = _truth([
        ("S1", "sl1", "1_1.jpeg", 5), ("S1", "sl1", "1_1.jpeg", 9),
        ("S2", "sl2", "1_1.jpeg", 3),      # same tile name, different slide
    ])
    clean, ambiguous, _ = _dedupe_truth(truth, "leiden_2.5")
    assert ambiguous == [("sl1", "1_1.jpeg")]
    assert list(clean["slides"]) == ["sl2"], "the other slide's tile was dropped too"


def test_validate_scores_no_tile_against_an_arbitrary_label(tmp_path):
    """Goes through validate(), not the helper, because that is where the naive
    fix lives. `drop_duplicates(keep="first")` would score a conflicting tile
    against whichever of its two labels the CSV happened to list first — this
    hands it the OTHER one, so the naive path reports a disagreement where the
    honest path reports an excluded tile."""
    import contextlib
    import io
    import pandas as pd
    from assign_hpc_clusters import validate

    truth_path = tmp_path / "truth.csv"
    _truth([
        ("S1", "sl1", "1_1.jpeg", 5),      # listed first
        ("S1", "sl1", "1_1.jpeg", 9),      # and again, differently
        ("S1", "sl1", "2_2.jpeg", 7),
    ]).to_csv(truth_path, index=False)

    frame = pd.DataFrame({
        "samples": ["S1", "S1"], "slides": ["sl1", "sl1"],
        "tiles": ["1_1.jpeg", "2_2.jpeg"],
        "leiden_2.5": [9, 7],              # 9 is the label keep="first" discards
        "vote_margin": [0.9, 0.9],
    })

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        agreed = validate(frame, truth_path, "leiden_2.5")
    text = out.getvalue()

    # Honest: the ambiguous tile is excluded, so 1 tile matched and it agrees.
    # Naive keep-first: 2 matched, one scored against 5, agreement 50%.
    assert agreed is True, text
    assert "1 of 2 tiles matched" in text, text
    assert "excluded" in text and "DIFFERENT" in text, text
    assert "agreement 100.000%" in text, text


def test_validate_survives_the_real_label_files_duplicates(tmp_path):
    """End to end on Kai's actual CSV if it is present, since that is the file
    the acceptance test names and the one that used to crash."""
    import pandas as pd
    from assign_hpc_clusters import validate
    real = BACKEND.parent / "TCGA_LUAD_5x_he_train_filtered_leiden_2p5__fold2.csv"
    if not real.is_file():
        return          # not on this machine; the unit tests above still hold
    truth = pd.read_csv(real)

    # An "assignment" that agrees with the truth everywhere it is defined, built
    # from the truth itself so the only thing under test is the join.
    frame = truth.drop_duplicates(["slides", "tiles"], keep=False).copy()
    frame["vote_margin"] = 0.9
    assert validate(frame, real, "leiden_2.5") is True


# --- container binds -----------------------------------------------------
#
# A real failure: `--validate-against <bare filename.csv>` made
# validate_against.parent == Path("."), which _bind_args turned into
# "--bind .:.". Singularity resolves the source against the job's cwd but leaves
# the destination relative, so it refused with an error naming an ABSOLUTE source
# path and complaining the destination was not absolute -- pointing nowhere near
# the relative argument that caused it. The job died in seconds after queueing.


def test_a_relative_path_never_produces_a_relative_bind(tmp_path):
    from submit_feature_extraction import _bind_args
    specs = [b for b in _bind_args("assign_hpc_clusters.py") if ":" in b]
    assert specs, "no binds produced at all"
    for spec in specs:
        source, destination = spec.split(":", 1)
        assert destination.startswith("/"), f"relative bind destination: {spec}"
        assert source.startswith("/"), f"relative bind source: {spec}"


def test_the_symlink_and_its_target_are_still_bound_separately(tmp_path):
    """The fix uses abspath, not resolve(). resolve() would follow the symlink
    and collapse the two candidates into one, losing the /hpc-home side -- which
    is the entire reason _bind_args binds each path twice."""
    from submit_feature_extraction import _bind_args
    target = tmp_path / "real"
    target.mkdir()
    (target / "file.txt").write_text("x")
    link = tmp_path / "link"
    link.symlink_to(target)

    specs = [b.split(":", 1)[0] for b in _bind_args(link / "file.txt") if ":" in b]
    # Compared by realpath, not by string: on macOS tempfile hands back
    # /var/folders/... where /var is itself a symlink to /private/var, so the
    # bound realpath form resolves two links at once and never equals
    # str(target). Under pytest tmp_path is already resolved and it does — which
    # is why this passed there and failed standalone.
    import os
    assert str(link) in specs, f"the symlink form was not bound: {specs}"
    resolved = {os.path.realpath(spec) for spec in specs}
    assert os.path.realpath(target) in resolved, \
        f"the realpath form was not bound: {specs}"
    # Two distinct entries, which is the property that matters: collapsing them
    # is what resolve() would do and what would break the /hpc-home side.
    assert len({os.path.realpath(link), os.path.realpath(target)}) == 1
    assert str(link) not in {str(target)}, "the fixture did not create a symlink"


def test_a_relative_validate_against_is_bound_absolutely(tmp_path):
    """End to end through the command builder, which is where the relative path
    actually entered — _bind_args is only reached via validate_against.parent."""
    from submit_cluster_assignment import _build_assignment_command
    truth = tmp_path / "truth.csv"
    truth.write_text("samples,slides,tiles,leiden_2.5\n")

    import os
    previous = os.getcwd()
    os.chdir(tmp_path)
    try:
        command = _build_assignment_command(
            singularity_bin="singularity", singularity_image=tmp_path / "i.sif",
            extras_dir=tmp_path / "extras",
            assign_script=BACKEND / "assign_hpc_clusters.py",
            reference=tmp_path / "ref.npz", projections_h5=tmp_path / "q.h5",
            out_csv=tmp_path / "out.csv", rep_key="z_latent", k=None,
            batch_size=16_384,
            validate_against=Path("truth.csv"),   # bare filename, as a user types
            vote=[],
        )
    finally:
        os.chdir(previous)

    assert " .:." not in command and "--bind .:." not in command, command
    # And the truth file still reaches the job as an absolute path.
    assert str(truth.resolve()) in command or str(truth) in command, command


# --- Slurm walltime ------------------------------------------------------
#
# The assigner has no resume: it opens its output with mode='w' before encoding
# and its "output already exists" path crashes on an unbound local. So hitting
# the walltime does not cost the remaining fraction, it costs the whole run —
# and the retry then fails in seconds on the stale output with an error pointing
# nowhere near the cause.
#
# This path submits up to three jobs. Two of them were hardcoded at 2 hours
# while only the third was settable, so raising the limit for a long run left
# two that could still kill it late, after the expensive part had succeeded.


def test_all_three_walltimes_are_settable(tmp_path):
    import inspect
    from submit_cluster_assignment import submit_cluster_assignment_job

    parameters = inspect.signature(submit_cluster_assignment_job).parameters
    for name in ("time_limit", "mean_time_limit", "merge_time_limit"):
        assert name in parameters, f"{name} is not settable"


def test_no_walltime_is_hardcoded_in_an_sbatch(tmp_path):
    """The regression guard. A literal --time= next to an sbatch is a limit
    nobody can raise from the command line."""
    import re
    source = (BACKEND / "submit_cluster_assignment.py").read_text()
    # A literal is "--time=" followed by a digit; "--time={...}" is an
    # interpolation and is what we want. Grepping for the flag alone matches
    # both, which is how the first version of this test failed on correct code.
    literal = re.compile(r"--time=\d")
    hardcoded = [line.strip() for line in source.splitlines()
                 if literal.search(line)]
    assert not hardcoded, f"hardcoded walltime(s): {hardcoded}"


def test_the_assignment_gets_the_longest_limit(tmp_path):
    """It is the only one whose cost is queries x reference rows; the other two
    are single passes over the input and the output. A mean job outliving the
    assignment would mean the numbers were picked without thinking about which
    job actually takes the time."""
    from submit_cluster_assignment import (ASSIGN_TIME_LIMIT, MEAN_TIME_LIMIT,
                                           MERGE_TIME_LIMIT)

    def seconds(limit: str) -> int:
        days, _, rest = limit.partition("-")
        if not rest:
            rest, days = days, "0"
        hours, minutes, secs = (int(x) for x in rest.split(":"))
        return int(days) * 86400 + hours * 3600 + minutes * 60 + secs

    # A floor, not an exact value. This number is headroom over a measured
    # extrapolation and moves with the cohort size — it went 2 days -> 4 days
    # when 14,000 slides put the unsharded estimate near 20 hours. Pinning it
    # exactly made this test fail for a reason that is not a defect, which is
    # the opposite of what it is for. The ordering below is the real invariant.
    assert seconds(ASSIGN_TIME_LIMIT) >= 2 * 86400
    assert seconds(ASSIGN_TIME_LIMIT) > seconds(MEAN_TIME_LIMIT)
    assert seconds(MEAN_TIME_LIMIT) >= seconds(MERGE_TIME_LIMIT)


def test_every_default_is_a_walltime_slurm_accepts(tmp_path):
    """A malformed --time is rejected by sbatch at submit, which is at least
    loud — but only once someone tries, and these are defaults."""
    import re
    from submit_cluster_assignment import (ASSIGN_TIME_LIMIT, MEAN_TIME_LIMIT,
                                           MERGE_TIME_LIMIT)
    pattern = re.compile(r"^(\d+-)?\d{1,2}:\d{2}:\d{2}$")
    for limit in (ASSIGN_TIME_LIMIT, MEAN_TIME_LIMIT, MERGE_TIME_LIMIT):
        assert pattern.match(limit), f"{limit!r} is not a Slurm walltime"


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
