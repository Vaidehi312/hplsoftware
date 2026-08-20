"""Density-adaptive distance scaling in the k-NN vote.

The measured error profile on ref_raw128 was: 100% of errors had the true
cluster among the k neighbours, and 90% had it as the *runner-up*. So the
information was retrieved and then lost in scoring. One mechanism that loses it
is density: sum(1/d^p) over the neighbours is an unnormalised density estimate,
so where two clusters differ in how tightly they are packed, the denser one
wins on having more points nearby rather than on being the better answer.

Leiden itself did not have that problem — it cut a UMAP fuzzy-simplicial-set
graph whose weights are exp(-(d - rho_i)/sigma_i), with sigma_i solved per
reference point. local_scale is the cheap stand-in for that sigma.

These tests are built so the answer is known by construction, and so the guard
can come out bad: a case where scaling must change the vote, and a case where
it must not.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from assign_hpc_clusters import compute_local_scale, vote  # noqa: E402


def test_scale_is_the_rth_neighbour_distance_excluding_self(tmp_path):
    """Points on a line at unit spacing: the r-th neighbour of an interior
    point is exactly r away, so the answer is known without approximation."""
    vectors = np.arange(20, dtype=np.float32).reshape(-1, 1)
    scale = compute_local_scale(vectors, r=3)
    # Interior points have neighbours on both sides, so the 3rd nearest is at 2.
    assert np.isclose(scale[10], 2.0), scale[10]
    # The endpoint has them all on one side, so its 3rd nearest is at 3.
    assert np.isclose(scale[0], 3.0), scale[0]
    assert (scale > 0).all()


def test_dense_region_gets_a_smaller_scale_than_a_sparse_one(tmp_path):
    """The whole premise. If this did not hold, dividing by it would be noise."""
    dense = np.linspace(0, 1, 50, dtype=np.float32).reshape(-1, 1)
    sparse = np.linspace(100, 200, 50, dtype=np.float32).reshape(-1, 1)
    vectors = np.vstack([dense, sparse])
    scale = compute_local_scale(vectors, r=5)
    assert scale[:50].mean() < scale[50:].mean() / 10, (
        scale[:50].mean(), scale[50:].mean()
    )


def test_scaling_flips_a_vote_the_dense_cluster_would_otherwise_win(tmp_path):
    """Constructed so plain 1/d^2 and the scaled version must disagree.

    Cluster 0 is tightly packed and its members sit slightly further from the
    query; cluster 1 is spread out and its single member is nearer. Unscaled,
    cluster 0's numerosity carries it. Scaled, cluster 0's own tightness is
    divided out and the genuinely closer point wins.
    """
    codes = np.array([0, 0, 0, 1], dtype=np.int64)
    # Squared distances, as faiss returns.
    neighbour_distances = np.array([[4.0, 4.41, 4.84, 3.24]], dtype=np.float32)
    neighbour_indices = np.array([[0, 1, 2, 3]], dtype=np.int64)

    unscaled = vote(neighbour_indices, neighbour_distances, codes, 2,
                    distance_weighted=True, distance_power=2.0)[0]
    assert unscaled[0] == 0, "three tight neighbours out-vote one closer point"

    # Cluster 0 sits in a dense region (small scale), cluster 1 in a sparse one.
    local_scale = np.array([0.1, 0.1, 0.1, 5.0], dtype=np.float32)
    scaled = vote(neighbour_indices, neighbour_distances, codes, 2,
                  distance_weighted=True, distance_power=2.0,
                  local_scale=local_scale)[0]
    assert scaled[0] == 1, "scaling must divide out cluster 0's own tightness"


def test_uniform_scale_changes_nothing_but_the_weights_magnitude(tmp_path):
    """A constant scale is a global rescale of every distance, which cannot
    change which cluster wins. If it did, the scaling is not doing what it
    says."""
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 4, 40).astype(np.int64)
    idx = rng.integers(0, 40, (25, 10)).astype(np.int64)
    dist = rng.random((25, 10)).astype(np.float32) * 10

    plain = vote(idx, dist, codes, 4, distance_weighted=True, distance_power=2.0)
    uniform = vote(idx, dist, codes, 4, distance_weighted=True, distance_power=2.0,
                   local_scale=np.full(40, 3.0, dtype=np.float32))
    assert (plain[0] == uniform[0]).all()
    # Margin is a share of total weight, so it is scale-invariant too.
    assert np.allclose(plain[1], uniform[1], atol=1e-6)


def test_neighbor_distance_column_stays_on_raw_distances(tmp_path):
    """mean_distance is written to the assignments CSV as neighbor_distance and
    read as a real distance. Rescaling it would silently redefine that column
    for every consumer."""
    codes = np.array([0, 1], dtype=np.int64)
    idx = np.array([[0, 1]], dtype=np.int64)
    dist = np.array([[4.0, 9.0]], dtype=np.float32)  # squared -> 2.0 and 3.0

    plain = vote(idx, dist, codes, 2, distance_weighted=True)
    scaled = vote(idx, dist, codes, 2, distance_weighted=True,
                  local_scale=np.array([0.5, 4.0], dtype=np.float32))
    assert np.isclose(plain[2][0], 2.5)
    assert np.isclose(scaled[2][0], 2.5), "raw mean distance must be unchanged"


def test_zero_scale_does_not_produce_infinities(tmp_path):
    """A reference point with duplicates has a scale of zero. Dividing by it
    would send its distance to infinity and its weight to zero — silently
    removing exactly the nearest neighbours."""
    codes = np.array([0, 1], dtype=np.int64)
    idx = np.array([[0, 1]], dtype=np.int64)
    dist = np.array([[1.0, 4.0]], dtype=np.float32)
    winner, margin, mean_distance = vote(
        idx, dist, codes, 2, distance_weighted=True,
        local_scale=np.array([0.0, 1.0], dtype=np.float32),
    )
    assert np.isfinite(margin).all()
    assert np.isfinite(mean_distance).all()
    assert winner[0] in (0, 1)


def test_invalid_r_is_rejected(tmp_path):
    try:
        compute_local_scale(np.zeros((5, 2), dtype=np.float32), r=0)
    except ValueError as e:
        assert "at least 1" in str(e)
    else:
        raise AssertionError("r=0 has no meaning and must be rejected")


def test_padded_neighbours_do_not_leak_a_scale(tmp_path):
    """faiss pads with -1 when k exceeds the index. Those slots carry weight 0
    already; the scale lookup must not resurrect them."""
    codes = np.array([0, 1], dtype=np.int64)
    idx = np.array([[0, 1, -1, -1]], dtype=np.int64)
    dist = np.array([[1.0, 4.0, 0.0, 0.0]], dtype=np.float32)
    winner, margin, _ = vote(idx, dist, codes, 2, distance_weighted=True,
                             local_scale=np.array([1.0, 1.0], dtype=np.float32))
    assert np.isfinite(margin).all()
    assert winner[0] == 0, "the nearest valid neighbour must still win"


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_localscale_"))
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
