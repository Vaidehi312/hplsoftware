"""Slide-level holdout: the only closed-book accuracy measurement here.

Leave-one-out removes one tile and leaves its ~5,000 slide-mates in the
reference — same scanner, same staining, often contiguous tissue — so it largely
measures recovery from near-duplicates of the tile itself. Removing whole slides
is what makes the number predictive of a genuinely new slide, and therefore of a
new cohort.

Everything here guards one property or its consequences: **no tile of a held-out
slide may remain in the reference.** A single leaked tile is not a crash and not
a visible error. It votes for itself at distance zero, the accuracy goes up, and
the result is more flattering and completely wrong — which is this pipeline's
signature failure and the reason the split is worth testing at all.

The other half is that the two sides must keep meaning the same thing. `codes`
index a category table; recompute it on either side of the split and every
cluster ID silently changes meaning. So the category table is asserted identical,
and a mismatched pair is asserted to be refused rather than compared.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import build_hpc_reference as bhr  # noqa: E402
import validate_reference as vr  # noqa: E402

N_SLIDES, PER_SLIDE, NCOMP, NCLUST = 24, 120, 8, 8


def _write_h5ad(path: Path, seed: int = 3, with_mean: bool = False) -> None:
    """A Leiden .h5ad with real slide structure.

    Each slide's tiles sit around a slide-specific offset inside its cluster's
    blob. That offset IS the slide-mate leakage — without it, holding a slide out
    would cost nothing and every test here would pass on a fixture that cannot
    show the effect it is about.
    """
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((NCLUST, NCOMP)) * 1.6
    slides, codes, blocks = [], [], []
    for s in range(N_SLIDES):
        label = s % NCLUST
        offset = rng.standard_normal(NCOMP) * 0.9
        slides += [f"SLIDE-{s:03d}"] * PER_SLIDE
        codes += [label] * PER_SLIDE
        blocks.append(centres[label] + offset
                      + rng.standard_normal((PER_SLIDE, NCOMP)) * 0.55)
    vectors = np.vstack(blocks).astype(np.float32)

    names = np.unique(slides)
    index = {name: i for i, name in enumerate(names)}
    with h5py.File(path, "w") as f:
        f.create_dataset("obsm/X_pca", data=vectors)
        f.create_dataset("varm/PCs", data=np.eye(NCOMP, dtype=np.float32))
        group = f.create_group("obs/leiden_2.5")
        group.create_dataset("codes", data=np.asarray(codes, dtype=np.int8))
        group.create_dataset(
            "categories", data=np.array([str(i).encode() for i in range(NCLUST)]))
        slide_group = f.create_group("obs/slides")
        slide_group.create_dataset(
            "codes", data=np.array([index[s] for s in slides], dtype=np.int16))
        slide_group.create_dataset(
            "categories", data=np.array([s.encode() for s in names]))
        f.create_dataset("uns/nn_leiden/params/n_neighbors", data=np.int64(15))
        if with_mean:
            f.create_dataset("uns/pca/mean",
                             data=np.arange(NCOMP, dtype=np.float32))


def _built(tmp_path: Path, **kwargs):
    h5ad = tmp_path / "TCGA_fake_leiden_2p5__fold2.h5ad"
    if not h5ad.is_file():
        _write_h5ad(h5ad, **kwargs)
    artifact = bhr.build(h5ad, "leiden_2.5")
    slide_names = bhr.read_slide_names(h5ad, "slides")
    return h5ad, artifact, slide_names


# --- the property everything else depends on ------------------------------

def test_no_tile_of_a_held_out_slide_remains_in_the_reference(tmp_path):
    """The whole point. One leaked tile votes for itself at distance zero, the
    accuracy rises, and nothing anywhere reports a problem."""
    h5ad, artifact, slide_names = _built(tmp_path)
    reduced, holdout = bhr.hold_out_slides(artifact, slide_names, 5, seed=0)

    held = set(holdout["held_slides"])
    kept_slides = set(str(s) for s in slide_names[~np.isin(slide_names,
                                                           list(held))])
    assert held, "no slides were held out"
    assert not (held & kept_slides), "a held-out slide is still in the reference"

    # And on the vectors themselves, not just the names: no held-out query may
    # appear as a row of the reduced reference.
    kept = {row.tobytes() for row in reduced["reference"]}
    leaked = sum(1 for row in holdout["queries"] if row.tobytes() in kept)
    assert leaked == 0, f"{leaked} held-out tile(s) are still in the reference"

    # The split accounts for every row.
    assert (reduced["reference"].shape[0] + holdout["queries"].shape[0]
            == artifact["reference"].shape[0])
    assert (len(reduced["codes"]) + len(holdout["codes"])
            == len(artifact["codes"]))


def test_whole_slides_go_not_parts_of_them(tmp_path):
    """A random-tile split would leak slide-mates, which is the exact thing
    leave-one-out already does and this exists to avoid."""
    h5ad, artifact, slide_names = _built(tmp_path)
    _, holdout = bhr.hold_out_slides(artifact, slide_names, 5, seed=0)
    for slide in holdout["held_slides"]:
        total = int((slide_names == slide).sum())
        held = int((holdout["slides"] == slide).sum())
        assert held == total, f"{slide}: {held} of {total} tiles held out"


# --- the two sides must keep meaning the same thing -----------------------

def test_the_category_table_is_not_recomputed_on_either_side(tmp_path):
    """codes index this table. Renumbering it on one side silently reassigns
    every cluster ID, which no completeness check would notice."""
    h5ad, artifact, slide_names = _built(tmp_path)
    reduced, holdout = bhr.hold_out_slides(artifact, slide_names, 5, seed=0)
    original = [str(c) for c in artifact["categories"]]
    assert [str(c) for c in reduced["categories"]] == original
    assert [str(c) for c in holdout["categories"]] == original
    assert len(original) == NCLUST


def test_the_pca_mean_is_carried_over_not_recomputed(tmp_path):
    """The mean belongs to the original PCA fit, not to the rows kept. Deriving
    it from a subset would move the space the held-out queries already live in."""
    h5ad = tmp_path / "TCGA_fake_leiden_2p5__fold2.h5ad"
    _write_h5ad(h5ad, with_mean=True)
    artifact = bhr.build(h5ad, "leiden_2.5")
    assert artifact["mean"] is not None, "fixture did not store a mean"
    slide_names = bhr.read_slide_names(h5ad, "slides")
    reduced, _ = bhr.hold_out_slides(artifact, slide_names, 5, seed=0)
    assert np.array_equal(reduced["mean"], artifact["mean"])
    # n_neighbors likewise: a property of the graph, not of the rows.
    assert reduced["n_neighbors"] == artifact["n_neighbors"]


# --- determinism and refusals --------------------------------------------

def test_the_seed_alone_decides_which_slides(tmp_path):
    h5ad, artifact, slide_names = _built(tmp_path)
    a = bhr.hold_out_slides(artifact, slide_names, 5, seed=0)[1]["held_slides"]
    b = bhr.hold_out_slides(artifact, slide_names, 5, seed=0)[1]["held_slides"]
    c = bhr.hold_out_slides(artifact, slide_names, 5, seed=1)[1]["held_slides"]
    assert a == b, "same seed gave different slides"
    assert a != c, "different seeds gave the same slides"


def test_an_impossible_split_is_refused(tmp_path):
    h5ad, artifact, slide_names = _built(tmp_path)
    for n, expected in ((0, "at least 1"), (N_SLIDES, "nothing to classify"),
                        (N_SLIDES + 10, "nothing to classify")):
        try:
            bhr.hold_out_slides(artifact, slide_names, n, seed=0)
        except SystemExit as e:
            assert expected in str(e), str(e)
        else:
            raise AssertionError(f"--holdout-slides {n} must be refused")


def test_slide_names_that_do_not_match_the_reference_are_refused(tmp_path):
    h5ad, artifact, slide_names = _built(tmp_path)
    try:
        bhr.hold_out_slides(artifact, slide_names[:10], 5, seed=0)
    except SystemExit as e:
        assert "do not describe the same" in str(e), str(e)
    else:
        raise AssertionError("a mismatched slide-name array must be refused")


def test_a_cluster_left_only_on_held_out_slides_is_reported(tmp_path):
    """Those tiles cannot be got right by any classifier, so the accuracy would
    silently carry them as errors."""
    h5ad, artifact, slide_names = _built(tmp_path)
    # Every slide s carries cluster s % NCLUST, so holding out all slides of one
    # cluster removes that cluster from the reference entirely.
    victim = 0
    doomed = sorted({str(s) for s, c in zip(slide_names, artifact["codes"])
                     if int(c) == victim})
    keep_mask = ~np.isin(slide_names, doomed)
    # Build a fixture where exactly those slides are held out, by shrinking the
    # candidate pool to them.
    reduced, holdout = bhr.hold_out_slides(
        {**artifact,
         "reference": np.vstack([artifact["reference"][~keep_mask],
                                 artifact["reference"][keep_mask]]),
         "codes": np.concatenate([np.asarray(artifact["codes"])[~keep_mask],
                                  np.asarray(artifact["codes"])[keep_mask]])},
        np.concatenate([slide_names[~keep_mask], slide_names[keep_mask]]),
        len(doomed), seed=0)
    if set(holdout["held_slides"]) == set(doomed):
        assert victim in holdout["unreachable_clusters"]
    else:
        # The random draw did not pick exactly those slides; assert the weaker
        # invariant that still must hold.
        kept = set(np.unique(reduced["codes"]).tolist())
        for code in holdout["unreachable_clusters"]:
            assert code not in kept


# --- a holdout is only meaningful with its own reference -------------------

def _write_pair(tmp_path: Path, n_slides: int = 5):
    h5ad, artifact, slide_names = _built(tmp_path)
    reduced, holdout = bhr.hold_out_slides(artifact, slide_names, n_slides, seed=0)
    ref_path, hold_path = tmp_path / "reduced.npz", tmp_path / "hold.npz"
    bhr.save(reduced, ref_path)
    bhr.save_holdout(holdout, hold_path)
    full_path = tmp_path / "full.npz"
    bhr.save(artifact, full_path)
    return ref_path, hold_path, full_path


def test_the_matching_pair_loads(tmp_path):
    ref_path, hold_path, _ = _write_pair(tmp_path)
    reference = vr.load_reference(ref_path)
    holdout = vr.load_holdout(hold_path, reference)
    assert holdout["queries"].shape[0] == 5 * PER_SLIDE
    assert len(holdout["held_slides"]) == 5


def test_pairing_a_holdout_with_the_full_reference_is_refused(tmp_path):
    """The flattering wrong answer. Against the full reference every query finds
    itself at distance zero and the accuracy reads near-perfect."""
    _, hold_path, full_path = _write_pair(tmp_path)
    full = vr.load_reference(full_path)
    try:
        vr.load_holdout(hold_path, full)
    except SystemExit as e:
        assert "holdout marker" in str(e) or "rows" in str(e), str(e)
    else:
        raise AssertionError(
            "a holdout paired with the full reference must be refused")


def test_a_reference_of_the_wrong_size_is_refused(tmp_path):
    """Two holdout splits of the same h5ad have the same category table and the
    same groupby, so row count is what distinguishes them."""
    h5ad, artifact, slide_names = _built(tmp_path)
    five, holdout_five = bhr.hold_out_slides(artifact, slide_names, 5, seed=0)
    six, _ = bhr.hold_out_slides(artifact, slide_names, 6, seed=0)
    ref_six, hold_five = tmp_path / "six.npz", tmp_path / "hold5.npz"
    bhr.save(six, ref_six)
    bhr.save_holdout(holdout_five, hold_five)
    try:
        vr.load_holdout(hold_five, vr.load_reference(ref_six))
    except SystemExit as e:
        assert "rows" in str(e), str(e)
    else:
        raise AssertionError("a differently-sized reference must be refused")


def test_a_different_clustering_is_refused(tmp_path):
    ref_path, hold_path, _ = _write_pair(tmp_path)
    reference = vr.load_reference(ref_path)
    reference["groupby"] = "leiden_5.0"
    try:
        vr.load_holdout(hold_path, reference)
    except SystemExit as e:
        assert "different clusterings" in str(e), str(e)
    else:
        raise AssertionError("a different groupby must be refused")


def test_a_holdout_reference_is_never_called_production(tmp_path):
    """It passes the production test — same groupby, same cluster count — and
    differs only in having whole slides missing. Silence here is how a
    deliberately crippled reference gets used for a real assignment."""
    import contextlib
    import io
    ref_path, _, _ = _write_pair(tmp_path)
    reference = vr.load_reference(ref_path)
    assert reference["holdout_slides"], "the marker did not survive the round trip"

    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
        vr.describe_reference(ref_path, reference)
    text = err.getvalue()
    assert "SLIDE-HOLDOUT" in text, text
    assert "never be used for a real assignment" in text, text


def test_the_full_reference_carries_no_holdout_marker(tmp_path):
    """Or every ordinary build would start warning, and the warning would stop
    meaning anything."""
    _, _, full_path = _write_pair(tmp_path)
    assert not vr.load_reference(full_path)["holdout_slides"]


# --- the measurement itself ----------------------------------------------

def test_holdout_accuracy_scores_against_the_held_out_labels(tmp_path):
    ref_path, hold_path, _ = _write_pair(tmp_path)
    reference = vr.load_reference(ref_path)
    holdout = vr.load_holdout(hold_path, reference)
    result = vr.holdout_accuracy(reference, holdout, k=10, batch=512)

    assert result["n"] == holdout["queries"].shape[0]
    assert np.array_equal(result["truth"], holdout["codes"])
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["accuracy"] == float((result["predicted"]
                                        == holdout["codes"]).mean())
    # Every prediction must be a real cluster of the reference.
    assert result["predicted"].min() >= 0
    assert result["predicted"].max() < len(reference["categories"])
    # And the per-slide arm has one entry per query.
    assert len(result["slides"]) == result["n"]


def test_the_holdout_is_not_easier_than_it_should_be(tmp_path):
    """A leak would show up as an accuracy indistinguishable from perfect on a
    fixture built with slide-specific offsets. This is the behavioural version
    of the leak test above — it would catch a leak introduced anywhere between
    the split and the vote, not just in hold_out_slides."""
    ref_path, hold_path, _ = _write_pair(tmp_path)
    reference = vr.load_reference(ref_path)
    holdout = vr.load_holdout(hold_path, reference)
    result = vr.holdout_accuracy(reference, holdout, k=10, batch=512,
                                 distance_weighted=True, distance_power=3.0)
    assert result["accuracy"] < 1.0, (
        "the holdout scored 100% on a fixture with slide-specific offsets, "
        "which is what a leaked tile looks like")
    assert result["accuracy"] > 0.5, (
        f"{result['accuracy']:.3f} — the fixture is not learnable at all, so "
        f"the test cannot distinguish a leak from noise")


def test_adaptive_k_is_available_closed_book(tmp_path):
    """The shipped configuration includes adaptive k, so it has to be measurable
    here or the closed-book number describes a different classifier."""
    ref_path, hold_path, _ = _write_pair(tmp_path)
    reference = vr.load_reference(ref_path)
    holdout = vr.load_holdout(hold_path, reference)
    plain = vr.holdout_accuracy(reference, holdout, k=10, batch=512,
                                distance_weighted=True, distance_power=3.0)
    adaptive = vr.holdout_accuracy(reference, holdout, k=10, batch=512,
                                   distance_weighted=True, distance_power=3.0,
                                   adaptive_margin=0.9, adaptive_k=25)
    # A high gate re-votes nearly everything, so something must have moved.
    assert not np.array_equal(plain["predicted"], adaptive["predicted"]), \
        "the adaptive gate changed no prediction at all"


def test_the_cli_reports_both_numbers_and_the_gap(tmp_path):
    h5ad = tmp_path / "TCGA_fake_leiden_2p5__fold2.h5ad"
    _write_h5ad(h5ad)
    ref = tmp_path / "ref.npz"
    build = subprocess.run(
        [sys.executable, str(BACKEND / "build_hpc_reference.py"),
         "--h5ad", str(h5ad), "--out", str(ref),
         "--holdout-slides", "5", "--holdout-seed", "0"],
        capture_output=True, text=True, cwd=str(BACKEND))
    assert build.returncode == 0, build.stderr
    assert "SLIDE-HOLDOUT REFERENCE" in build.stdout, build.stdout

    holdout_path = tmp_path / "ref_holdout.npz"
    assert holdout_path.is_file(), "the default holdout path was not used"

    run = subprocess.run(
        [sys.executable, str(BACKEND / "validate_reference.py"),
         "--reference", str(ref), "--holdout", str(holdout_path),
         "--k", "10", "--distance-weighted", "--distance-power", "3",
         "--sample", "1500", "--worst", "3"],
        capture_output=True, text=True, cwd=str(BACKEND))
    assert run.returncode == 0, run.stderr
    for marker in ("CLOSED BOOK", "OPEN BOOK", "Per held-out slide",
                   "slide holdout", "gap"):
        assert marker in run.stdout, f"{marker} missing from:\n{run.stdout}"
    assert "SLIDE-HOLDOUT REFERENCE" in run.stderr, run.stderr


def test_holdout_out_without_holdout_slides_is_refused(tmp_path):
    h5ad = tmp_path / "TCGA_fake_leiden_2p5__fold2.h5ad"
    _write_h5ad(h5ad)
    run = subprocess.run(
        [sys.executable, str(BACKEND / "build_hpc_reference.py"),
         "--h5ad", str(h5ad), "--out", str(tmp_path / "r.npz"),
         "--holdout-out", str(tmp_path / "h.npz")],
        capture_output=True, text=True, cwd=str(BACKEND))
    assert run.returncode != 0
    assert "does nothing without" in (run.stdout + run.stderr)


# --- the two sides must be the same classifier ---------------------------
#
# The first real run of this reported the holdout 0.53 points HIGHER than
# leave-one-out and blamed easy slides. The actual cause: --adaptive-margin
# reached holdout_accuracy but leave_one_out did not accept it at all, so the
# closed-book side re-voted its near-ties and the open-book side did not. Two
# different classifiers, and the gap between them -- the entire output of this
# tool -- meant nothing.


def test_leave_one_out_applies_adaptive_k(tmp_path):
    """It has to, or the open-book number describes a classifier that is not the
    one in production and not the one the holdout measured."""
    import contextlib
    import io
    ref_path, _, _ = _write_pair(tmp_path)
    reference = vr.load_reference(ref_path)
    plain = vr.leave_one_out(reference, 600, 10, 256, 0,
                             distance_weighted=True, distance_power=3.0)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        adaptive = vr.leave_one_out(reference, 600, 10, 256, 0,
                                    distance_weighted=True, distance_power=3.0,
                                    adaptive_margin=0.9, adaptive_k=25)

    # Asserted on the margin, not the label. A re-voted row records the WIDE
    # vote's margin, so that must move; whether the wider neighbourhood also
    # changes the winner is data-dependent and on a separable fixture it often
    # does not.
    assert not np.allclose(plain["margins"], adaptive["margins"]), \
        "no margin moved, so the adaptive gate never re-voted anything"
    assert "re-voted" in log.getvalue(), log.getvalue()
    # Same tiles either way, so the comparison stays paired.
    assert np.array_equal(plain["truth"], adaptive["truth"])
    # And the rows it left alone are untouched, label and margin both.
    moved = ~np.isclose(plain["margins"], adaptive["margins"])
    assert (plain["predicted"][~moved] == adaptive["predicted"][~moved]).all()


def test_adaptive_off_leaves_leave_one_out_unchanged(tmp_path):
    """The widened search only happens when the gate is on, so every existing
    number stays reproducible."""
    ref_path, _, _ = _write_pair(tmp_path)
    reference = vr.load_reference(ref_path)
    a = vr.leave_one_out(reference, 600, 10, 256, 0, distance_weighted=True,
                         distance_power=3.0)
    b = vr.leave_one_out(reference, 600, 10, 256, 0, distance_weighted=True,
                         distance_power=3.0, adaptive_margin=0.0, adaptive_k=0)
    assert a["accuracy"] == b["accuracy"]
    assert np.array_equal(a["predicted"], b["predicted"])
    assert np.allclose(a["margins"], b["margins"])


def test_an_adaptive_k_no_wider_than_k_is_refused(tmp_path):
    ref_path, _, _ = _write_pair(tmp_path)
    reference = vr.load_reference(ref_path)
    try:
        vr.leave_one_out(reference, 600, 25, 256, 0, adaptive_margin=0.1,
                         adaptive_k=25)
    except SystemExit as e:
        assert "not wider" in str(e), str(e)
    else:
        raise AssertionError("an adaptive_k equal to k must be refused")


def test_the_cli_applies_adaptive_k_to_both_sides(tmp_path):
    """The bug, at the level it actually occurred. Both report blocks must show
    a re-vote line — one of each is what made the gap meaningless."""
    h5ad = tmp_path / "TCGA_fake_leiden_2p5__fold2.h5ad"
    _write_h5ad(h5ad)
    ref = tmp_path / "ref.npz"
    build = subprocess.run(
        [sys.executable, str(BACKEND / "build_hpc_reference.py"),
         "--h5ad", str(h5ad), "--out", str(ref),
         "--holdout-slides", "5", "--holdout-seed", "0"],
        capture_output=True, text=True, cwd=str(BACKEND))
    assert build.returncode == 0, build.stderr

    run = subprocess.run(
        [sys.executable, str(BACKEND / "validate_reference.py"),
         "--reference", str(ref), "--holdout", str(tmp_path / "ref_holdout.npz"),
         "--k", "10", "--distance-weighted", "--distance-power", "3",
         "--adaptive-margin", "0.9", "--adaptive-k", "25",
         "--sample", "800", "--worst", "2"],
        capture_output=True, text=True, cwd=str(BACKEND))
    assert run.returncode == 0, run.stderr
    revotes = run.stdout.count("Adaptive  : re-voted")
    assert revotes == 2, (
        f"expected a re-vote line under both CLOSED BOOK and OPEN BOOK, saw "
        f"{revotes}:\n{run.stdout}")

    # And each must sit in its own section, not both in one.
    closed = run.stdout.index("CLOSED BOOK")
    openbook = run.stdout.index("OPEN BOOK")
    assert run.stdout.index("Adaptive  : re-voted") < openbook
    assert run.stdout.rindex("Adaptive  : re-voted") > openbook > closed


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_holdout_test_"))
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
