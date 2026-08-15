"""build_hpc_reference.py --embedding-space {pca,raw}.

The raw path exists to let validate_reference.py compare k-NN accuracy in the
127-d PCA space against the encoder's own 128-d z_latent, skipping the PCA
step entirely — without touching project()/Searcher/vote() in
assign_hpc_clusters.py at all. That only holds if the raw .npz comes back with
an identity `components` (so "projecting" a raw embedding is a no-op) and the
same codes/categories as the pca build from the same .h5ad — this is what the
tests check.
"""

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from build_hpc_reference import build, save  # noqa: E402


def _write_fake_h5ad(path: Path, *, n=500, n_pca=12, n_raw=16, nclust=5,
                     n_neighbors=30, sparse_x=False):
    rng = np.random.default_rng(0)
    codes = rng.integers(0, nclust, n).astype(np.int64)
    categories = np.array([f"cluster_{i}" for i in range(nclust)])

    with h5py.File(path, "w") as f:
        f.create_dataset("obsm/X_pca", data=rng.standard_normal((n, n_pca)).astype(np.float32))
        f.create_dataset("varm/PCs", data=rng.standard_normal((n_raw, n_pca)).astype(np.float32))

        if sparse_x:
            f.create_group("X")  # a group, not a dataset — stands in for CSR-sparse X
        else:
            f.create_dataset("X", data=rng.standard_normal((n, n_raw)).astype(np.float32))

        obs = f.create_group("obs")
        cat_group = obs.create_group("leiden_2.5")
        cat_group.create_dataset("codes", data=codes)
        cat_group.create_dataset(
            "categories", data=np.array([c.encode("utf-8") for c in categories])
        )

        f.create_dataset("uns/nn_leiden/params/n_neighbors", data=np.int64(n_neighbors))

    return codes, categories


def test_pca_space_unchanged_by_default(tmp_path):
    """--embedding-space defaults to pca: identical to the pre-existing build()."""
    h5ad = tmp_path / "fake.h5ad"
    codes, categories = _write_fake_h5ad(h5ad)

    artifact = build(h5ad, "leiden_2.5")
    assert artifact["embedding_space"] == "pca"
    assert artifact["reference"].shape == (500, 12)
    assert artifact["components"].shape == (16, 12)
    assert np.array_equal(artifact["codes"], codes)
    assert list(artifact["categories"]) == list(categories)


def test_raw_space_reads_x_with_identity_components(tmp_path):
    h5ad = tmp_path / "fake.h5ad"
    _write_fake_h5ad(h5ad)

    artifact = build(h5ad, "leiden_2.5", embedding_space="raw")
    assert artifact["embedding_space"] == "raw"
    assert artifact["reference"].shape == (500, 16)
    # Identity so assign_hpc_clusters.project()'s `embeddings @ components` is a
    # no-op for a raw embedding — the whole point of this mode.
    np.testing.assert_array_equal(artifact["components"], np.eye(16, dtype=np.float32))
    assert artifact["mean"] is None


def test_raw_and_pca_agree_on_labels_from_the_same_h5ad(tmp_path):
    """The comparison only means anything if both spaces are labelling the same
    tiles the same way — different embeddings, identical ground truth."""
    h5ad = tmp_path / "fake.h5ad"
    _write_fake_h5ad(h5ad)

    pca_artifact = build(h5ad, "leiden_2.5", embedding_space="pca")
    raw_artifact = build(h5ad, "leiden_2.5", embedding_space="raw")

    assert np.array_equal(pca_artifact["codes"], raw_artifact["codes"])
    assert list(pca_artifact["categories"]) == list(raw_artifact["categories"])
    assert pca_artifact["reference"].shape[0] == raw_artifact["reference"].shape[0]


def test_raw_space_npz_is_loadable_by_validate_reference(tmp_path):
    """save()'s output for the raw space has to be readable by
    validate_reference.load_reference() with no special-casing — that's what
    lets leave-one-out run unmodified against either space."""
    from validate_reference import load_reference

    h5ad = tmp_path / "fake.h5ad"
    _write_fake_h5ad(h5ad)
    artifact = build(h5ad, "leiden_2.5", embedding_space="raw")
    out = tmp_path / "raw128.npz"
    save(artifact, out)

    loaded = load_reference(out)
    assert loaded["vectors"].shape == (500, 16)
    assert loaded["n_neighbors"] == 30
    assert loaded["groupby"] == "leiden_2.5"


def test_raw_space_refuses_sparse_x(tmp_path):
    h5ad = tmp_path / "fake.h5ad"
    _write_fake_h5ad(h5ad, sparse_x=True)

    try:
        build(h5ad, "leiden_2.5", embedding_space="raw")
        raise AssertionError("expected a KeyError for sparse X")
    except KeyError as e:
        assert "sparse" in str(e)


def test_raw_space_refuses_missing_x(tmp_path):
    h5ad = tmp_path / "fake.h5ad"
    with h5py.File(h5ad, "w") as f:
        f.create_dataset("obsm/X_pca", data=np.zeros((10, 4), np.float32))

    try:
        build(h5ad, "leiden_2.5", embedding_space="raw")
        raise AssertionError("expected a KeyError for missing X")
    except KeyError as e:
        assert "X missing" in str(e)


def test_unknown_embedding_space_rejected(tmp_path):
    h5ad = tmp_path / "fake.h5ad"
    _write_fake_h5ad(h5ad)
    try:
        build(h5ad, "leiden_2.5", embedding_space="tsne")
        raise AssertionError("expected a ValueError")
    except ValueError as e:
        assert "tsne" in str(e)


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_build_ref_test_"))
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
