"""Margin-gated adaptive k: leave_one_out()'s query_index override, and the
comparison script built on it.

The experiment's entire validity rests on one thing: the "before" and "after"
numbers have to come from the *exact* same reference rows, or a difference
could just be sampling noise rather than the effect of k. These tests check
that wiring — not any accuracy claim, which only means something on the real
HPC reference.
"""

import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from validate_reference import leave_one_out  # noqa: E402
from adaptive_k_experiment import compare  # noqa: E402


def _make_reference(n=4000, ncomp=10, nclust=5, seed=0, spread=1.0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, nclust, n).astype(np.int64)
    centres = rng.standard_normal((nclust, ncomp)) * 10
    vectors = (centres[codes] + rng.standard_normal((n, ncomp)) * spread).astype(np.float32)
    return {
        "vectors": vectors,
        "codes": codes,
        "categories": np.array([str(i) for i in range(nclust)]),
        "groupby": "test",
    }


def test_query_index_is_used_verbatim():
    reference = _make_reference()
    chosen = np.array([5, 100, 250, 3000], dtype=np.int64)
    result = leave_one_out(reference, sample=999, k=8, batch=64, seed=1,
                           distance_weighted=True, distance_power=2.0,
                           query_index=chosen)
    np.testing.assert_array_equal(result["query_index"], chosen)
    assert result["n"] == len(chosen)


def test_omitting_query_index_keeps_the_old_random_sampling():
    """Regression guard: every existing caller (validate_reference.py's CLI,
    the existing test suite) omits query_index and must see identical
    behaviour to before this parameter existed."""
    reference = _make_reference()
    a = leave_one_out(reference, sample=500, k=10, batch=128, seed=7,
                      distance_weighted=True, distance_power=2.0)
    b = leave_one_out(reference, sample=500, k=10, batch=128, seed=7,
                      distance_weighted=True, distance_power=2.0)
    np.testing.assert_array_equal(a["query_index"], b["query_index"])
    assert a["accuracy"] == b["accuracy"]


def test_query_index_out_of_range_rejected():
    reference = _make_reference(n=100)
    try:
        leave_one_out(reference, sample=10, k=5, batch=32, seed=0,
                      query_index=np.array([0, 99, 100]))
        raise AssertionError("expected a ValueError")
    except ValueError as e:
        assert "outside" in str(e)


def test_rerunning_the_same_subset_reproduces_the_masked_baseline():
    """Same tiles, same k: whether you get there by masking the full sample or
    by an explicit query_index rerun must give identical per-tile answers —
    this is what makes the before/after k comparison fair rather than a
    coincidence of two different samples."""
    reference = _make_reference()
    baseline = leave_one_out(reference, sample=2000, k=10, batch=256, seed=3,
                             distance_weighted=True, distance_power=2.0)
    threshold = np.quantile(baseline["margins"], 0.3)
    subset_mask = baseline["margins"] < threshold
    subset_index = baseline["query_index"][subset_mask]

    rerun = leave_one_out(reference, sample=len(subset_index), k=10, batch=256,
                          seed=3, distance_weighted=True, distance_power=2.0,
                          query_index=subset_index)

    np.testing.assert_array_equal(rerun["predicted"], baseline["predicted"][subset_mask])
    np.testing.assert_allclose(rerun["margins"], baseline["margins"][subset_mask], atol=1e-6)


def test_compare_reports_no_tiles_when_nothing_is_ambiguous():
    """A margin threshold of 0 (or a trivially separable reference) can
    legitimately flag zero tiles — the script must say so, not divide by zero
    computing an accuracy over an empty set."""
    reference = _make_reference(n=3000, nclust=4, spread=0.15)  # near-perfectly separable
    result = compare(reference, sample=1500, seed=0, k_base=10, k_expand=25,
                     distance_power=2.0, margin_threshold=-1.0,  # nothing is ever < -1
                     batch=256)
    assert result["low_mask"].sum() == 0
    assert result["before_accuracy"] is None
    assert result["after_accuracy"] is None
    assert result["overall_after_accuracy"] is None
    assert result["overall_before_accuracy"] == result["baseline"]["accuracy"]


def test_compare_reruns_only_the_low_margin_subset():
    reference = _make_reference(n=5000, nclust=6, spread=1.2)
    result = compare(reference, sample=2500, seed=1, k_base=8, k_expand=20,
                     distance_power=2.0, margin_threshold=0.3, batch=256)

    low_mask = result["low_mask"]
    if low_mask.sum() == 0:
        return  # this synthetic draw happened to be fully unambiguous; not a failure
    assert result["adaptive"]["n"] == low_mask.sum()
    np.testing.assert_array_equal(result["adaptive"]["query_index"],
                                  result["baseline"]["query_index"][low_mask])
    assert 0.0 <= result["before_accuracy"] <= 1.0
    assert 0.0 <= result["after_accuracy"] <= 1.0

    # overall_after_accuracy must be exactly "confident tiles as scored by the
    # baseline pass, plus re-queried tiles as scored by the adaptive pass" —
    # not, say, accidentally re-deriving it from the adaptive pass alone.
    baseline = result["baseline"]
    expected = (
        int(baseline["correct"][~low_mask].sum()) + int(result["adaptive"]["correct"].sum())
    ) / baseline["n"]
    assert result["overall_after_accuracy"] == expected
    assert result["overall_before_accuracy"] == baseline["accuracy"]


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
