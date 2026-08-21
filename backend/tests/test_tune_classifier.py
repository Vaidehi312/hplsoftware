"""The sweep tool had no tests, and every tuned number in
CLASSIFIER_TUNING_2026-08-13.md came out of it.

Its whole premise is that one faiss search can be re-voted into a whole grid of
configurations: `k` is a prefix slice, `distance_power` is a re-vote over the
same distances, adaptive k is a re-vote of the low-margin rows at a wider
prefix. That premise is load-bearing and entirely untested, and it has two
sharp edges.

  * The prefix slice is only the k nearest NON-SELF neighbours if the self-match
    was found and dropped. Miss it and every accuracy is inflated by a tile
    voting for itself — a wrong number that looks completely plausible.
  * The neighbour cache keys on the query set. Serve a cached matrix for a
    different --sample or --seed and every figure describes tiles that were not
    the ones asked for.

Plus the bug these were written for: the reported baseline used to be found by
searching the swept grid for it, so narrowing the sweep to a single
--distance-power killed the run with a bare StopIteration after paying for the
entire search.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import tune_classifier as tc  # noqa: E402

REF_ROWS, NCOMP, NCLUST = 1200, 8, 6

# Calibrated so the baseline lands near 89%, not 100%. Well-separated clusters
# make every configuration perfect and every comparison vacuous — the first
# draft of this fixture scored 100% and the adaptive test passed on a no-op.
# Overlapping clusters are also the honest shape: production's remaining error
# is entirely boundary tiles between neighbouring clusters.
_CENTRE_SPREAD, _WITHIN_CLUSTER_NOISE = 1.2, 1.0


def _reference(seed: int = 0) -> dict:
    """A load_reference()-shaped dict. Clustered, not uniform noise: a k-NN
    vote over uniform noise is near chance, which would hide any real change in
    the vote behind sampling wobble."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, NCLUST, REF_ROWS).astype(np.int64)
    centres = (rng.standard_normal((NCLUST, NCOMP)).astype(np.float32)
               * _CENTRE_SPREAD)
    vectors = (centres[codes]
               + rng.standard_normal((REF_ROWS, NCOMP)).astype(np.float32)
               * _WITHIN_CLUSTER_NOISE)
    return {
        "vectors": np.ascontiguousarray(vectors),
        "codes": codes,
        "categories": np.array([str(i) for i in range(NCLUST)]),
        "n_neighbors": 15,
        "groupby": "leiden_2.5",
    }


def _sweep(reference, **over):
    kwargs = dict(sample=300, seed=0, k_max=20, batch=128,
                  ks=[5, 10], powers=[1.0, 2.0], scalings=[0],
                  adaptives=[0.0], adaptive_ks=[15], class_weightings=[False],
                  cache=None)
    kwargs.update(over)
    return tc.sweep(reference,
                    kwargs["sample"], kwargs["seed"], kwargs["k_max"],
                    kwargs["batch"], kwargs["ks"], kwargs["powers"],
                    kwargs["scalings"], kwargs["adaptives"],
                    kwargs["adaptive_ks"], kwargs["class_weightings"],
                    kwargs["cache"])


# --- the baseline is measured, not looked up ----------------------------

def test_the_fixture_is_hard_enough_to_be_worth_measuring(tmp_path):
    """Guards every other test in this file. At 100% accuracy no configuration
    can differ from any other and the comparisons all pass trivially."""
    reference = _reference()
    _, baseline = _sweep(reference, ks=[10], powers=[2.0])
    assert 0.60 < baseline < 0.98, (
        f"baseline {baseline:.3f}: the fixture is too easy (or too hard) for a "
        f"change in the vote to show up")


def test_a_grid_without_the_baseline_still_reports(tmp_path):
    """The bug. BASELINE is distance^2; sweeping only power 3 leaves it out of
    the results, and the baseline used to be recovered by scanning them."""
    reference = _reference()
    results, baseline = _sweep(reference, ks=[10], powers=[3.0])

    assert 0.0 < baseline <= 1.0
    assert not any(r["distance_power"] == tc.BASELINE["distance_power"]
                   for r in results), \
        "this grid is supposed to exclude the baseline, or it proves nothing"
    # And the deltas are still meaningful — report() divides by nothing else.
    tc.report(results, baseline, top=5)


def test_the_baseline_is_the_same_whatever_is_swept(tmp_path):
    """It is scored independently of the grid, so narrowing the grid must not
    move it. If it did, two sweeps of the same sample would disagree about what
    they were improving on."""
    reference = _reference()
    _, wide = _sweep(reference, ks=[5, 10], powers=[1.0, 2.0, 3.0])
    _, narrow = _sweep(reference, ks=[10], powers=[3.0])
    assert wide == narrow


def test_a_config_equal_to_the_baseline_scores_as_zero_delta(tmp_path):
    """When the grid does contain the baseline, the returned baseline and that
    row must agree — otherwise one of the two is measuring something else."""
    reference = _reference()
    results, baseline = _sweep(reference, ks=[10], powers=[2.0])
    row = next(r for r in results
               if r["k"] == 10 and r["distance_power"] == 2.0
               and not r["adaptive"] and not r["class_weighted"])
    assert row["accuracy"] == baseline
    assert row["net"] == 0 and row["discordant"] == 0


# --- the prefix-slice premise -------------------------------------------

def test_the_self_match_is_excluded_from_every_prefix(tmp_path):
    """Leave-one-out means the query's own row must never be among its
    neighbours. Included, it votes for its own true label at distance zero and
    every accuracy is inflated — most of all at small k, exactly where the
    sweep concentrates."""
    reference = _reference()
    query_index = np.arange(0, REF_ROWS, 7)
    idx, dist = tc.gather_neighbours(reference, query_index, k_max=20, batch=64)

    assert idx.shape == (len(query_index), 20)
    assert not (idx == query_index[:, None]).any(), "a tile is its own neighbour"
    # Distances must be sorted, or a prefix is not the k nearest anything.
    assert (np.diff(dist, axis=1) >= -1e-5).all()


def test_a_prefix_slice_equals_a_search_at_that_k(tmp_path):
    """The claim that makes sweeping k free. If a prefix of the k_max search
    were not the k-nearest set, every k in the table would be mislabelled."""
    reference = _reference()
    query_index = np.arange(0, REF_ROWS, 11)
    wide_idx, wide_dist = tc.gather_neighbours(reference, query_index,
                                               k_max=20, batch=64)
    for k in (5, 10):
        narrow_idx, narrow_dist = tc.gather_neighbours(reference, query_index,
                                                       k_max=k, batch=64)
        assert np.array_equal(wide_idx[:, :k], narrow_idx)
        assert np.allclose(wide_dist[:, :k], narrow_dist)


# --- the cache ----------------------------------------------------------

def test_the_cache_is_reused_only_for_the_same_query_set(tmp_path):
    """Serving a cached matrix for a different sample would attach real-looking
    accuracies to tiles nobody asked about."""
    reference = _reference()
    cache = tmp_path / "nbr.npz"

    first = tc.gather_neighbours(reference, np.arange(0, 200), 20, 64, cache)
    assert cache.is_file()
    again = tc.gather_neighbours(reference, np.arange(0, 200), 20, 64, cache)
    assert np.array_equal(first[0], again[0])
    assert np.array_equal(first[1], again[1])

    # A different query set must re-search rather than serve these rows.
    other = tc.gather_neighbours(reference, np.arange(200, 400), 20, 64, cache)
    assert not np.array_equal(first[0], other[0])
    # Same length, different rows: the shape check alone would have passed it.
    assert other[0].shape == first[0].shape


def test_a_cache_narrower_than_k_max_is_not_padded(tmp_path):
    """A k=10 cache cannot answer a k_max=20 sweep. Reusing it would silently
    truncate every wide configuration, including the adaptive re-vote."""
    reference = _reference()
    cache = tmp_path / "nbr.npz"
    tc.gather_neighbours(reference, np.arange(0, 200), 10, 64, cache)
    idx, _ = tc.gather_neighbours(reference, np.arange(0, 200), 20, 64, cache)
    assert idx.shape[1] == 20
    assert (idx >= 0).all(), "the wide columns came back as padding"


def test_a_cached_sweep_matches_an_uncached_one(tmp_path):
    reference = _reference()
    plain, plain_base = _sweep(reference)
    cached, cached_base = _sweep(reference, cache=tmp_path / "nbr.npz")
    reused, reused_base = _sweep(reference, cache=tmp_path / "nbr.npz")
    assert plain == cached == reused
    assert plain_base == cached_base == reused_base


# --- the grid -----------------------------------------------------------

def test_adaptive_off_is_not_repeated_once_per_adaptive_k(tmp_path):
    """adaptive_k means nothing with the gate off, so sweeping three of them
    must not produce three identical rows — which would also make the table's
    top-N a list of duplicates."""
    reference = _reference()
    results, _ = _sweep(reference, ks=[10], powers=[2.0],
                        adaptives=[0.0, 0.5], adaptive_ks=[12, 15, 20])
    off = [r for r in results if r["adaptive"] == 0.0]
    assert len(off) == 1, off


def test_a_re_query_no_wider_than_the_base_is_dropped(tmp_path):
    """Re-voting at a k <= the base sees the same neighbours. Kept, it would
    appear as a distinct configuration with a spurious 0.00 delta."""
    reference = _reference()
    results, _ = _sweep(reference, ks=[10], powers=[2.0],
                        adaptives=[0.5], adaptive_ks=[5, 10, 20])
    assert sorted(r["adaptive_k"] for r in results) == [20]


def test_adaptive_k_can_actually_change_the_verdict(tmp_path):
    """If the gate never fired, every test above would pass on a no-op."""
    reference = _reference()
    results, baseline = _sweep(reference, ks=[10], powers=[2.0],
                               adaptives=[0.0, 0.9], adaptive_ks=[20])
    gated = next(r for r in results if r["adaptive"] == 0.9)
    assert gated["discordant"] > 0, "the re-vote changed no tile's verdict"
    assert gated["accuracy"] != baseline


# --- McNemar ------------------------------------------------------------

def test_mcnemar_drops_concordant_pairs(tmp_path):
    """The whole reason for a paired test: tiles both configurations agree on
    carry no information about which is better."""
    base = np.array([True, True, False, False, True, False])
    same = base.copy()
    assert tc.mcnemar(base, same) == {
        "fixed": 0, "broke": 0, "net": 0, "discordant": 0, "z": 0.0}

    # Two fixed, one broken: net +1 over 3 discordant pairs.
    other = np.array([True, False, True, True, True, False])
    result = tc.mcnemar(base, other)
    assert (result["fixed"], result["broke"], result["net"]) == (2, 1, 1)
    assert result["discordant"] == 3
    assert result["z"] == 1 / np.sqrt(3)


def test_mcnemar_signs_a_regression_negative(tmp_path):
    """A config that breaks more than it fixes must come out negative, or the
    table would rank regressions as wins."""
    base = np.array([True, True, True, False])
    worse = np.array([False, False, True, True])
    result = tc.mcnemar(base, worse)
    assert result["net"] == -1 and result["z"] < 0


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_tune_test_"))
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
