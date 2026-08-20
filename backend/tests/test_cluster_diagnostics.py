"""Explaining why a cluster recovers badly.

The diagnosis names an ACTION — merge, split, accept, gather better features —
so a cluster landing in the wrong class sends someone off to do the wrong thing.
Every test here builds a reference where the right answer is known by
construction, and each diagnosis class is tested both for firing when it should
and for NOT firing when it shouldn't.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import cluster_diagnostics as cd  # noqa: E402
from validate_reference import load_reference  # noqa: E402


def _reference(tmp_path: Path, blocks, ncomp=8, name="ref.npz"):
    """blocks: list of (n, centre_scalar, spread). Cluster code = list position."""
    rng = np.random.default_rng(3)
    vectors, codes = [], []
    for code, (n, centre, spread) in enumerate(blocks):
        base = np.zeros(ncomp, dtype=np.float32)
        base[0] = centre
        vectors.append(base + rng.standard_normal((n, ncomp)).astype(np.float32) * spread)
        codes.append(np.full(n, code))
    vectors = np.vstack(vectors).astype(np.float32)
    codes = np.concatenate(codes).astype(np.int64)
    path = tmp_path / name
    np.savez(path, reference=vectors,
             components=np.zeros((ncomp, ncomp), np.float32), codes=codes,
             categories=np.array([str(i) for i in range(len(blocks))]),
             n_neighbors=np.int64(25), meta=json.dumps({"groupby": "leiden_2.5"}))
    return load_reference(path)


def _by_code(rows):
    return {r["code"]: r for r in rows}


def test_well_separated_clusters_are_all_boundary(tmp_path):
    """The negative control. If separated clusters got flagged as duplicates or
    engulfed, every diagnosis below would be meaningless."""
    ref = _reference(tmp_path, [(600, 0.0, 0.4), (600, 30.0, 0.4), (600, 60.0, 0.4)])
    rows = cd.analyse(ref, k=10, sample=1800, seed=0)
    assert {r["diagnosis"] for r in rows} == {"boundary"}, [r["diagnosis"] for r in rows]
    assert all(r["purity"] > 0.98 for r in rows)
    assert all(r["core"] > 0.95 for r in rows)


def test_two_clusters_on_one_blob_are_called_duplicate(tmp_path):
    """Same centre, same spread, same size — one morphology labelled twice. The
    action is a merge, and it is a labelling decision, not a classifier fix."""
    ref = _reference(tmp_path, [(600, 0.0, 1.0), (600, 0.0, 1.0), (600, 40.0, 0.5)])
    rows = _by_code(cd.analyse(ref, k=10, sample=1800, seed=0))
    assert rows[0]["diagnosis"] == "duplicate", rows[0]
    assert rows[1]["diagnosis"] == "duplicate", rows[1]
    assert rows[0]["mutual"] and rows[1]["mutual"]
    assert rows[0]["partner"] == 1 and rows[1]["partner"] == 0
    # The far-away one must not be dragged in.
    assert rows[2]["diagnosis"] == "boundary", rows[2]


def test_small_cluster_inside_a_large_one_is_called_engulfed(tmp_path):
    """A small cluster sitting inside a much larger one at COMPARABLE density,
    so its own 40 members are far apart while 2,000 foreign ones are nearby.

    Density is what makes this class real. A small but *tight* cluster inside a
    diffuse one is not engulfed at all — its own members are still its nearest
    neighbours, and it recovers perfectly. Only when its members are sparser
    than the surrounding cluster does the neighbourhood stop belonging to it.

    Distinguished from 'duplicate' by the overlap being ONE-WAY: the big
    cluster's foreign neighbours mostly come from elsewhere, so it does not
    point back.
    """
    ref = _reference(tmp_path, [
        (2000, 0.0, 1.0),    # 0: large, and where 1 lives
        (40, 0.0, 1.0),      # 1: few members over the same volume -> sparse
        (2000, 1.5, 1.0),    # 2: overlaps 0 heavily, so 0's partner is 2 not 1
    ])
    rows = _by_code(cd.analyse(ref, k=10, sample=4040, seed=0))
    assert rows[1]["diagnosis"] == "engulfed", rows[1]
    assert rows[1]["partner"] == 0, rows[1]
    assert not rows[1]["mutual"], "one-way overlap is what separates this from duplicate"
    assert rows[1]["partner_ratio"] > 2.0
    assert rows[1]["separation"] < 1.0


def test_purity_is_not_a_restatement_of_recovery(tmp_path):
    """purity is computed without predicting anything. If it were just accuracy
    under another name it would explain nothing — so it must be measurably
    different from the leave-one-out recovery of the same clusters."""
    from validate_reference import leave_one_out
    import contextlib, io

    ref = _reference(tmp_path, [(800, 0.0, 1.2), (800, 2.0, 1.2), (800, 30.0, 0.5)])
    rows = _by_code(cd.analyse(ref, k=10, sample=2400, seed=0))
    with contextlib.redirect_stdout(io.StringIO()):
        loo = leave_one_out(ref, 2400, 10, 4096, 0, distance_weighted=True,
                            distance_power=2.0)

    recovery = {}
    for code in (0, 1, 2):
        mask = loo["truth"] == code
        if mask.any():
            recovery[code] = float(loo["correct"][mask].mean())

    # The vote recovers more than raw purity, because a majority only needs to
    # win, not to be unanimous. That gap is the information purity carries.
    for code, rec in recovery.items():
        assert rec >= rows[code]["purity"] - 1e-6, (code, rec, rows[code]["purity"])
    assert any(rec - rows[code]["purity"] > 0.05 for code, rec in recovery.items()), (
        "purity and recovery are indistinguishable on this fixture"
    )


def test_medoid_is_a_real_tile_not_a_mean(tmp_path):
    """A Leiden cluster is a graph community and need not be convex — a mean can
    land where no tile is, or inside another cluster. Every medoid must be an
    actual member."""
    ref = _reference(tmp_path, [(400, 0.0, 1.0), (400, 20.0, 1.0)])
    rng = np.random.default_rng(0)
    centres = cd.medoids(ref["vectors"], ref["codes"], 2, rng)
    for code in (0, 1):
        members = ref["vectors"][ref["codes"] == code]
        assert (np.abs(members - centres[code]).sum(axis=1) < 1e-6).any(), (
            f"medoid of cluster {code} is not one of its tiles"
        )


def test_diagnosis_classes_are_reachable_and_distinct(tmp_path):
    """Each class must be produced by some real geometry — a class nothing can
    trigger is dead code pretending to be a finding."""
    separated = _reference(tmp_path, [(600, 0.0, 0.4), (600, 30.0, 0.4)], name="a.npz")
    overlapping = _reference(tmp_path, [(600, 0.0, 1.0), (600, 0.0, 1.0),
                                        (600, 40.0, 0.5)], name="b.npz")
    engulfing = _reference(tmp_path, [(2000, 0.0, 1.0), (40, 0.0, 1.0),
                                      (2000, 1.5, 1.0)], name="c.npz")
    produced = set()
    for ref, n in ((separated, 1200), (overlapping, 1800), (engulfing, 4040)):
        produced |= {r["diagnosis"] for r in cd.analyse(ref, k=10, sample=n, seed=0)}
    assert {"boundary", "duplicate", "engulfed"} <= produced, produced


def test_empty_cluster_is_skipped_not_crashed(tmp_path):
    """A category with no members is legitimate — a cluster removed upstream
    leaves a gap in the code range."""
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((300, 6)).astype(np.float32)
    codes = np.repeat([0, 2], 150).astype(np.int64)  # no cluster 1
    path = tmp_path / "gap.npz"
    np.savez(path, reference=vectors, components=np.zeros((6, 6), np.float32),
             codes=codes, categories=np.array(["0", "1", "2"]),
             n_neighbors=np.int64(25), meta=json.dumps({"groupby": "leiden_2.5"}))
    rows = cd.analyse(load_reference(path), k=10, sample=300, seed=0)
    assert {r["code"] for r in rows} == {0, 2}


def test_reworded_descriptions_count_as_the_same_morphology(tmp_path):
    """Byte comparison misses the cases that matter. In the real review sheet
    cluster 57 is "stroma with pigment" and 89 is "pigment and stroma" — one
    morphology written two ways, and exactly the pair the classifier confuses.
    """
    assert cd._same_description({0: "stroma with pigment", 1: "pigment and stroma"}, 0, 1)
    assert cd._same_description({0: "solid retraction artefact",
                                 1: "solid retraction artefact"}, 0, 1)
    assert cd._same_description({0: "mostly vessel lumina", 1: "vessel lumina"}, 0, 1)
    # Genuinely different morphologies must not collapse.
    assert not cd._same_description({0: "infiltrative tumour",
                                     1: "infiltrative tumour with collageous stroma"}, 0, 1)
    assert not cd._same_description({0: "background", 1: "edge"}, 0, 1)
    # Missing or empty entries are not a match.
    assert not cd._same_description({0: "", 1: ""}, 0, 1)
    assert not cd._same_description({0: "background"}, 0, 1)
    assert not cd._same_description({0: "background", 1: "edge"}, 0, None)


def test_matching_descriptions_override_the_geometry(tmp_path):
    """Two clusters the review sheet calls the same thing are a duplicate
    whatever the geometry says — the 'errors' between them are not errors."""
    # Close enough that a partner exists at all, far enough that the geometry
    # alone reads as a normal boundary.
    ref = _reference(tmp_path, [(600, 0.0, 0.5), (600, 2.2, 0.5), (600, 40.0, 0.5)])
    plain = _by_code(cd.analyse(ref, k=10, sample=1800, seed=0))
    assert plain[0]["partner"] == 1, plain[0]
    assert plain[0]["diagnosis"] == "boundary", "geometry alone sees nothing wrong"

    described = _by_code(cd.analyse(
        ref, k=10, sample=1800, seed=0,
        descriptions={0: "solid retraction artefact", 1: "solid retraction artefact",
                      2: "background"},
    ))
    assert described[0]["diagnosis"] == "duplicate", described[0]
    assert described[0]["same_description"]


def test_review_sheet_is_parsed_including_its_bom(tmp_path):
    """The real file is UTF-8 with a BOM, which makes the first column name
    '\ufeffcluster' under a plain open() — so every row would be skipped and the
    sheet would silently look empty."""
    path = tmp_path / "review.csv"
    path.write_text("\ufeffcluster,description,remove\n"
                    "0,background,1\n1,acinar,\n2,edge,1\n", encoding="utf-8")
    descriptions, removed = cd.load_descriptions(path)
    assert descriptions == {0: "background", 1: "acinar", 2: "edge"}
    assert removed == {0, 2}


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_clusterdiag_"))
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
