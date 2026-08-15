"""Leave-one-out validation of the HPC reference.

The measurement has one subtle failure mode, and it is the reason these tests
exist: every reference tile is its own nearest neighbour at distance zero. Leave
those self-matches in and the accuracy is near-perfect no matter how meaningless
the labels are — the number looks like a passing validation while measuring
nothing. test_self_matches_are_excluded is the test that matters here.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

SCRIPT = BACKEND / "validate_reference.py"


def _write_reference(path: Path, *, separable: bool, n=6000, ncomp=12, nclust=8,
                     k=25, seed=0, spread=0.5):
    """A reference whose labels either follow the geometry or don't."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, nclust, n).astype(np.int64)
    if separable:
        centres = rng.standard_normal((nclust, ncomp)) * 12
        vectors = (centres[codes] + rng.standard_normal((n, ncomp)) * spread)
    else:
        # Labels assigned independently of position: k-NN cannot beat chance.
        vectors = rng.standard_normal((n, ncomp))
    np.savez(
        path,
        reference=vectors.astype(np.float32),
        components=np.zeros((ncomp, ncomp), np.float32),
        codes=codes,
        categories=np.array([str(i) for i in range(nclust)]),
        n_neighbors=np.int64(k),
        meta=json.dumps({"groupby": "leiden_2.5"}),
    )
    return nclust


def _run(*args, expect_ok=True):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=str(BACKEND),
    )
    if expect_ok and result.returncode != 0:
        raise AssertionError(f"failed:\n{result.stdout}\n{result.stderr}")
    return result


def _accuracy(stdout: str) -> float:
    line = next(l for l in stdout.splitlines() if l.startswith("Accuracy"))
    return float(line.split(":")[1].split("%")[0].strip())


def test_separable_clusters_are_recovered(tmp_path):
    ref = tmp_path / "sep.npz"
    _write_reference(ref, separable=True)
    accuracy = _accuracy(_run("--reference", str(ref), "--sample", "2000").stdout)
    assert accuracy > 95, accuracy


def test_meaningless_labels_score_near_chance(tmp_path):
    """The number has to be able to come out bad, or it is not a measurement."""
    ref = tmp_path / "noise.npz"
    nclust = _write_reference(ref, separable=False)
    accuracy = _accuracy(_run("--reference", str(ref), "--sample", "2000").stdout)
    chance = 100 / nclust
    assert accuracy < chance * 2, f"{accuracy}% vs chance {chance}%"


def test_self_matches_are_excluded(tmp_path):
    """The bug this design exists to avoid. A tile is its own nearest neighbour
    at distance zero, so including self-matches would make even random labels
    look almost perfectly recovered.

    Demonstrated rather than asserted about internals: on a reference whose
    labels are pure noise, accuracy must stay near chance. It could only be high
    if self-votes were counted.
    """
    ref = tmp_path / "noise.npz"
    nclust = _write_reference(ref, separable=False, n=4000, k=9)
    accuracy = _accuracy(_run("--reference", str(ref), "--sample", "2000").stdout)
    # With self-votes at k=9 a single self-vote would frequently decide the
    # majority, pushing this far above chance.
    assert accuracy < 100 / nclust * 2.5, (
        f"{accuracy}% on random labels suggests self-matches are being counted"
    )


def test_vote_margin_stratifies_accuracy(tmp_path):
    """vote_margin is written into every assignment CSV and indexed on
    tile_registry for "show me the least confident tiles". If low-margin tiles
    were not wrong more often, all of that would be decoration.

    Needs genuinely overlapping clusters: well-separated ones put every tile in
    the top margin band, which proves nothing.
    """
    ref = tmp_path / "overlap.npz"
    rng = np.random.default_rng(2)
    n, ncomp, nclust, k = 12000, 6, 6, 25
    codes = rng.integers(0, nclust, n).astype(np.int64)
    centres = rng.standard_normal((nclust, ncomp)) * 0.45
    np.savez(
        ref,
        reference=(centres[codes] + rng.standard_normal((n, ncomp))).astype(np.float32),
        components=np.zeros((ncomp, ncomp), np.float32),
        codes=codes,
        categories=np.array([str(i) for i in range(nclust)]),
        n_neighbors=np.int64(k),
        meta=json.dumps({"groupby": "leiden_2.5"}),
    )

    stdout = _run("--reference", str(ref), "--sample", "6000").stdout
    bands = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("margin ") and "correct" in stripped:
            bands.append(float(stripped.split()[-2].rstrip("%")))

    assert len(bands) >= 3, f"expected several populated margin bands, got {bands}"
    # Rising margin must mean rising accuracy: the lowest band clearly worse
    # than the highest, and the trend broadly monotonic.
    assert bands[-1] > bands[0] + 15, bands
    assert bands == sorted(bands), f"accuracy not monotonic in margin: {bands}"


def test_truth_neighbours_counts_same_label_neighbours(tmp_path):
    """The error-budget split is only meaningful if this count is real.

    Built so the answer is known by construction: a reference where every tile's
    label is assigned by position into well-separated blobs must have nearly all
    k neighbours sharing the true label, while one with random labels must have
    about k/nclust of them.
    """
    from validate_reference import leave_one_out, load_reference

    sep = tmp_path / "sep.npz"
    _write_reference(sep, separable=True, n=4000, nclust=8, spread=0.4)
    result = leave_one_out(load_reference(sep), 1000, 10, 4096, 0)
    assert result["truth_neighbours"].mean() > 9.5, result["truth_neighbours"].mean()
    assert result["truth_neighbours"].max() <= 10

    noise = tmp_path / "noise.npz"
    _write_reference(noise, separable=False, n=4000, nclust=8)
    result = leave_one_out(load_reference(noise), 1000, 10, 4096, 0)
    # ~10/8 = 1.25 expected. Generous bounds, but far from the separable case.
    assert 0.7 < result["truth_neighbours"].mean() < 2.2, result["truth_neighbours"].mean()


def test_unreachable_errors_are_counted_as_unreachable(tmp_path):
    """The distinction the whole diagnostic rests on: an error where the true
    cluster never appeared among the k neighbours must be reported as
    unreachable, because no vote rule can recover it.

    On a random-label reference almost every error is unreachable-or-lucky, so
    assert the count is non-trivial rather than that it is exactly anything.
    """
    from validate_reference import leave_one_out, load_reference

    ref = tmp_path / "noise.npz"
    _write_reference(ref, separable=False, n=4000, nclust=8)
    result = leave_one_out(load_reference(ref), 1500, 10, 4096, 0)

    wrong = ~result["correct"]
    unreachable = (result["truth_neighbours"][wrong] == 0).sum()
    assert wrong.sum() > 0
    # With random labels and k=10 over 8 clusters, a large share of wrong tiles
    # should have had no same-label neighbour at all.
    assert unreachable > wrong.sum() * 0.1, (unreachable, wrong.sum())

    # And a correct tile can never be unreachable: it won the vote, so at least
    # one neighbour carried its label.
    assert (result["truth_neighbours"][result["correct"]] > 0).all()


def test_error_budget_ceiling_is_between_accuracy_and_one(tmp_path):
    """The ceiling is accuracy + the addressable share. It must sit strictly
    above the measured accuracy when there are recoverable errors, and must not
    claim 100% while unreachable errors exist."""
    from validate_reference import leave_one_out, load_reference

    ref = tmp_path / "overlap.npz"
    rng = np.random.default_rng(5)
    n, ncomp, nclust, k = 8000, 6, 6, 25
    codes = rng.integers(0, nclust, n).astype(np.int64)
    centres = rng.standard_normal((nclust, ncomp)) * 0.9
    np.savez(
        ref,
        reference=(centres[codes] + rng.standard_normal((n, ncomp))).astype(np.float32),
        components=np.zeros((ncomp, ncomp), np.float32),
        codes=codes,
        categories=np.array([str(i) for i in range(nclust)]),
        n_neighbors=np.int64(k),
        meta=json.dumps({"groupby": "leiden_2.5"}),
    )
    result = leave_one_out(load_reference(ref), 3000, 10, 4096, 0)

    wrong = ~result["correct"]
    unreachable = int((result["truth_neighbours"][wrong] == 0).sum())
    reachable = int(wrong.sum()) - unreachable
    ceiling = (int(result["correct"].sum()) + reachable) / result["n"]

    assert reachable > 0 and unreachable > 0, (reachable, unreachable)
    assert result["accuracy"] < ceiling < 1.0, (result["accuracy"], ceiling)


def test_confusion_pairs_reported_by_error_count(tmp_path):
    """The weakest-cluster list ranks by rate, which hides where the errors
    actually are. A tiny cluster at a terrible rate contributes fewer errors
    than a large cluster at a good one, and only the pair table shows that.

    Built so the answer is known: one large cluster overlapping another
    produces the largest error pair, while a tiny scattered cluster does not.
    """
    ref = tmp_path / "sized.npz"
    rng = np.random.default_rng(11)
    ncomp, nclust, k = 6, 5, 25
    # Clusters 0 and 1 large and overlapping; 2-4 small and well separated.
    codes = np.concatenate([
        np.zeros(4000, np.int64), np.ones(4000, np.int64),
        np.full(120, 2, np.int64), np.full(120, 3, np.int64), np.full(120, 4, np.int64),
    ])
    centres = np.zeros((nclust, ncomp), np.float32)
    centres[1, 0] = 1.2                       # overlaps cluster 0
    centres[2, 1] = 40.0
    centres[3, 2] = 40.0
    centres[4, 3] = 40.0
    vectors = centres[codes] + rng.standard_normal((len(codes), ncomp)).astype(np.float32)
    np.savez(
        ref, reference=vectors.astype(np.float32),
        components=np.zeros((ncomp, ncomp), np.float32), codes=codes,
        categories=np.array([str(i) for i in range(nclust)]),
        n_neighbors=np.int64(k), meta=json.dumps({"groupby": "leiden_2.5"}),
    )

    stdout = _run("--reference", str(ref), "--sample", "6000", "--k", "10").stdout
    assert "Biggest confusion pairs by error count" in stdout
    section = stdout.split("Biggest confusion pairs by error count")[1]
    first_pair = next(l for l in section.splitlines()
                      if "→" in l and "the other way" in l)
    # The 0<->1 overlap must dominate, not one of the small clusters.
    assert first_pair.split("→")[0].strip() in {"0", "1"}, first_pair
    assert "symmetric" in first_pair, first_pair


def test_min_accuracy_gates(tmp_path):
    """So this can be used as a check in a pipeline, not just read by a human."""
    ref = tmp_path / "noise.npz"
    _write_reference(ref, separable=False, n=3000)
    failed = _run("--reference", str(ref), "--sample", "1000",
                  "--min-accuracy", "0.99", expect_ok=False)
    assert failed.returncode != 0
    assert "below the required" in failed.stderr

    ref2 = tmp_path / "sep.npz"
    _write_reference(ref2, separable=True, n=3000)
    _run("--reference", str(ref2), "--sample", "1000", "--min-accuracy", "0.90")


def test_sampling_is_reproducible(tmp_path):
    """A validation number nobody can reproduce is not evidence."""
    ref = tmp_path / "sep.npz"
    _write_reference(ref, separable=True, spread=6.0)  # imperfect, so not a flat 100%
    first = _accuracy(_run("--reference", str(ref), "--sample", "1500",
                           "--seed", "7").stdout)
    second = _accuracy(_run("--reference", str(ref), "--sample", "1500",
                            "--seed", "7").stdout)
    assert first == second


def test_missing_reference_says_how_to_build_it(tmp_path):
    result = _run("--reference", str(tmp_path / "absent.npz"), expect_ok=False)
    assert result.returncode != 0
    assert "build_hpc_reference" in result.stderr


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_val_test_"))
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
