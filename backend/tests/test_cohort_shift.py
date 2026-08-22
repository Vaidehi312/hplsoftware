"""Cohort shift: does the reference actually contain tissue like this cohort's?

The failure this exists to catch produces no error at all. k-NN assigns every
tile to its nearest cluster however far away that cluster is, so a cohort from a
different scanner still yields a complete assignments CSV, per-slide proportions
that differ from the reference cohort's, and a survival analysis that reads those
differences as biology.

So the tests are about the two things that could make the check itself lie:

  * saying "consistent" when the cohort has moved. Tested by shifting a synthetic
    cohort and requiring the verdict to change.
  * saying "alarm" for a cohort that is fine, or for a handful of bad slides that
    need dropping rather than the cohort being rejected. Those are different
    actions, so the check has to distinguish them.

The verdict is deliberately driven by two differently-scaled signals, and the
tests pin why: the tail ratio explodes (a one-SD shift sends it past 5x, three
bad slides in ten send it past 20x) while the median percentile stays bounded and
separates "mostly fine with a tail" from "the whole cohort has moved".
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import cohort_shift as cs  # noqa: E402

N_REF = 50_000


def _profile(seed: int = 5, vote: str = "k=10 --distance-power 3") -> dict:
    rng = np.random.default_rng(seed)
    return cs.build_profile(
        np.abs(rng.normal(0.5, 0.12, N_REF)).astype(np.float32),
        np.clip(rng.beta(2.0, 2.0, N_REF), 0, 1).astype(np.float32),
        rng.integers(0, 8, N_REF), [str(i) for i in range(8)],
        reference="ref.npz", reference_rows=2_500_000,
        groupby="leiden_2.5", vote=vote, seed=0)


def _cohort(shift: float = 0.0, n: int = 20_000, slides: int = 10,
            bad: int = 0, margin_shift: float = 0.0,
            seed: int = 11) -> pd.DataFrame:
    """`bad` slides carry the shift; bad=0 means every slide carries it."""
    rng = np.random.default_rng(seed)
    per = n // slides
    distance, margin, names = [], [], []
    for s in range(slides):
        this = shift if (bad == 0 or s < bad) else 0.0
        distance.append(np.abs(rng.normal(0.5 + this, 0.12, per)))
        a, b = (2.0, 2.0 + margin_shift * 6)
        margin.append(np.clip(rng.beta(a, b, per), 0, 1))
        names += [f"S{s:02d}"] * per
    return pd.DataFrame({"neighbor_distance": np.concatenate(distance),
                         "vote_margin": np.concatenate(margin),
                         "slides": names})


# --- it must be able to say "nothing wrong" ------------------------------

def test_an_unshifted_cohort_reads_consistent(tmp_path):
    """The most important negative. A check that always finds something is worth
    nothing, and would train everyone to ignore it."""
    result = cs.compare(_cohort(shift=0.0), _profile())
    level, _ = cs.verdict(result)
    assert level == "consistent", (level, result["novelty_ratio"],
                                  result["median_percentile"])
    # And the numbers behind it should be near their null values.
    assert 0.5 < result["novelty_ratio"] < 2.0, result["novelty_ratio"]
    assert 40 < result["median_percentile"] < 60, result["median_percentile"]


def test_the_envelope_is_the_references_own_99th_percentile(tmp_path):
    """Expressed as a ratio against 1% so it means the same thing whatever the
    reference's absolute distances are."""
    profile = _profile()
    result = cs.compare(_cohort(shift=0.0), profile)
    expected = float(np.interp(cs.ENVELOPE_LEVEL, profile["levels"],
                               profile["neighbor_distance"]))
    assert result["envelope"] == expected
    assert result["expected_beyond"] == 1.0 - cs.ENVELOPE_LEVEL
    assert np.isclose(result["novelty_ratio"],
                      result["beyond_envelope"] / result["expected_beyond"])


# --- and it must detect a real shift -------------------------------------

def test_a_shifted_cohort_is_detected(tmp_path):
    result = cs.compare(_cohort(shift=0.30), _profile())
    level, sentence = cs.verdict(result)
    assert level == "alarm", (level, result["median_percentile"])
    assert result["novelty_ratio"] > 5
    assert result["median_percentile"] > cs._ALARM_PERCENTILE
    assert "extrapolation" in sentence


def test_the_verdict_escalates_monotonically_with_the_shift(tmp_path):
    """A bigger shift must never read as less of a problem."""
    profile = _profile()
    order = {"consistent": 0, "notice": 1, "alarm": 2}
    seen = []
    for shift in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
        result = cs.compare(_cohort(shift=shift), profile)
        seen.append((shift, order[cs.verdict(result)[0]],
                     result["median_percentile"]))
    levels = [entry[1] for entry in seen]
    assert levels == sorted(levels), seen
    assert levels[0] == 0 and levels[-1] == 2, seen
    # The median percentile is the signal the verdict keys on, so it too must
    # rise monotonically or the escalation above is coincidence.
    medians = [entry[2] for entry in seen]
    assert medians == sorted(medians), seen


# --- some bad slides is a different problem from a shifted cohort --------

def test_a_few_bad_slides_are_localised_not_blamed_on_the_cohort(tmp_path):
    """The action differs: drop or re-scan a few slides, versus extend the
    reference. A single verdict driven by the tail ratio alone would call both
    'alarm' — three bad slides in ten push that past 20x while 70% of the cohort
    is untouched."""
    profile = _profile()
    result = cs.compare(_cohort(shift=0.35, bad=3), profile)
    level, sentence = cs.verdict(result)

    assert result["novelty_ratio"] > 5, "the fixture is not shifted enough"
    assert level == "notice", (level, result["median_percentile"])
    assert "some slides rather than the cohort" in sentence

    spread = cs.slide_concentration(result)
    assert spread is not None and spread > cs._SLIDE_CONCENTRATION, spread
    # The bad slides must be the ones at the top of the table.
    worst = [row["slide"] for row in result["per_slide"][:3]]
    assert sorted(worst) == ["S00", "S01", "S02"], result["per_slide"][:4]


def test_an_evenly_shifted_cohort_is_not_blamed_on_slides(tmp_path):
    result = cs.compare(_cohort(shift=0.30), _profile())
    spread = cs.slide_concentration(result)
    assert spread is not None and spread < cs._SLIDE_CONCENTRATION, spread
    assert "property of the cohort" in cs.verdict(result)[1]


def test_slide_concentration_is_none_without_slides(tmp_path):
    frame = _cohort(shift=0.0).drop(columns=["slides"])
    result = cs.compare(frame, _profile())
    assert "per_slide" not in result
    assert cs.slide_concentration(result) is None
    # And the verdict still works, just without the localisation clause.
    level, sentence = cs.verdict(result)
    assert level == "consistent"
    assert "slides" not in sentence.lower()


# --- novelty and ambiguity are different problems ------------------------

def test_ambiguity_is_reported_separately_from_novelty(tmp_path):
    """A cohort whose tissue IS represented but sits on cluster boundaries has
    normal distances and a high low-margin share. Conflating the two would point
    at the wrong lever: --min-margin helps ambiguity and does nothing for
    novelty."""
    profile = _profile()
    result = cs.compare(_cohort(shift=0.0, margin_shift=1.0), profile)

    # Distances unchanged...
    assert cs.verdict(result)[0] == "consistent"
    # ...but the margin distribution has moved.
    assert (result["cohort_low_margin"]["0.10"]
            > result["reference_low_margin"]["0.10"]), (
        result["cohort_low_margin"], result["reference_low_margin"])


# --- the profile ---------------------------------------------------------

def test_a_profile_round_trips(tmp_path):
    path = tmp_path / "profile.json"
    original = _profile()
    cs.save_profile(original, path)
    loaded = cs.load_profile(path)
    assert loaded["vote"] == original["vote"]
    assert loaded["neighbor_distance"] == original["neighbor_distance"]
    assert loaded["levels"] == original["levels"]
    # Cluster proportions are keyed by category name, not by code, so they stay
    # readable and cannot be misindexed.
    assert set(loaded["cluster_proportion"]) == {str(i) for i in range(8)}
    assert np.isclose(sum(loaded["cluster_proportion"].values()), 1.0)


def test_a_profile_records_the_vote_it_describes(tmp_path):
    """Distances depend on k and margins on every knob, so a profile without its
    vote is a baseline nobody can safely compare against."""
    profile = _profile(vote="k=10 --distance-weighted --distance-power 3")
    assert "k=10" in profile["vote"] and "distance-power 3" in profile["vote"]


def test_a_missing_profile_and_a_malformed_one_are_both_refused(tmp_path):
    try:
        cs.load_profile(tmp_path / "nope.json")
    except SystemExit as e:
        assert "No profile" in str(e)
    else:
        raise AssertionError("a missing profile must be refused")

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"levels": [0.5]}))
    try:
        cs.load_profile(bad)
    except SystemExit as e:
        assert "missing" in str(e), str(e)
    else:
        raise AssertionError("a profile without its quantiles must be refused")


def test_a_csv_without_the_confidence_columns_is_refused(tmp_path):
    """Assignments from before migrate_tile_registry_confidence.sql, or any other
    CSV, must be refused rather than compared as zeros."""
    for missing in ("neighbor_distance", "vote_margin"):
        frame = _cohort().drop(columns=[missing])
        try:
            cs.compare(frame, _profile())
        except SystemExit as e:
            assert missing in str(e), str(e)
        else:
            raise AssertionError(f"a CSV without {missing} must be refused")


def test_percentiles_are_clamped_not_extrapolated(tmp_path):
    """A tile beyond the top stored quantile is honestly 'at least 99.9th', not a
    fabricated 99.9997th."""
    profile = _profile()
    top = profile["levels"][-1]
    huge = cs._interp_percentile(profile["neighbor_distance"],
                                 profile["levels"], 1e6)
    tiny = cs._interp_percentile(profile["neighbor_distance"],
                                 profile["levels"], -1e6)
    assert huge == top
    assert tiny == profile["levels"][0]


def test_non_finite_distances_do_not_poison_the_comparison(tmp_path):
    """vote() returns NaN for a tile with no valid neighbours. Counting those as
    beyond the envelope would invent a shift."""
    frame = _cohort(shift=0.0)
    frame.loc[frame.index[:50], "neighbor_distance"] = np.nan
    result = cs.compare(frame, _profile())
    assert result["n"] == len(frame)
    assert result["n_finite_distance"] == len(frame) - 50
    assert cs.verdict(result)[0] == "consistent"


# --- the CLI ------------------------------------------------------------

def test_the_cli_runs_and_prints_a_verdict(tmp_path):
    profile_path = tmp_path / "profile.json"
    cs.save_profile(_profile(), profile_path)
    csv_path = tmp_path / "assignments.csv"
    _cohort(shift=0.30).to_csv(csv_path, index=False)

    run = subprocess.run(
        [sys.executable, str(BACKEND / "cohort_shift.py"),
         "--assignments", str(csv_path), "--profile", str(profile_path)],
        capture_output=True, text=True, cwd=str(BACKEND))
    assert run.returncode == 0, run.stderr
    for marker in ("Novelty", "Distance by quantile", "Ambiguity",
                   "Verdict: ALARM", "Worst"):
        assert marker in run.stdout, f"{marker} missing from:\n{run.stdout}"


# --- the profile must match the vote that produced the CSV ---------------


def test_the_profile_path_is_keyed_on_the_vote_preset(tmp_path):
    """A legacy-vote profile is not a valid baseline for a tuned assignment: the
    distances come from a different k and the margins from a different weighting,
    so comparing across them would manufacture a shift out of a settings
    difference. Separate files make picking the wrong one deliberate."""
    reference = Path("/refs/hpc_reference_leiden_2p5_fold2.npz")
    tuned = cs.default_profile_path(reference, "tuned")
    legacy = cs.default_profile_path(reference, "legacy")
    assert tuned != legacy
    assert tuned.name == "hpc_reference_leiden_2p5_fold2_profile_tuned.json"
    assert tuned.parent == reference.parent


def test_the_server_picks_the_profile_by_the_recorded_vote(tmp_path):
    """The run record is the only place the vote is written down — the CSV cannot
    carry it, since load_hpc_assignments.py finds its cluster column by
    elimination. So the endpoint parses the recorded vote to choose a baseline,
    and that parse has to work for every preset."""
    import tile_server_v2_ as srv
    from submit_cluster_assignment import (VOTE_PRESETS, describe_vote,
                                           resolve_vote)

    for name in VOTE_PRESETS:
        recorded = describe_vote(resolve_vote(name), name)
        matched = next((p for p in VOTE_PRESETS if recorded.startswith(p)), None)
        assert matched == name, (name, recorded, matched)

    # A modified preset still resolves to its base, which is the honest choice:
    # the baseline is approximate rather than absent, and the UI shows both
    # votes so the mismatch is visible.
    modified = describe_vote(resolve_vote("tuned", adaptive_margin=0.1), "tuned")
    assert "(modified)" in modified
    assert next(p for p in VOTE_PRESETS if modified.startswith(p)) == "tuned"
    assert srv.DEFAULT_VOTE_PRESET in VOTE_PRESETS


def test_the_endpoint_reads_only_the_columns_it_needs(tmp_path):
    """An assignments CSV for 14,000 slides is gigabytes of string columns. The
    check needs three columns; loading the rest would make a cheap read-only
    check expensive enough that nobody runs it."""
    frame = _cohort(shift=0.0)
    frame["samples"] = "S"
    frame["tiles"] = "1_1.jpeg"
    frame["leiden_2.5"] = 3
    frame["hpc_reference"] = "ref"
    path = tmp_path / "assignments.csv"
    frame.to_csv(path, index=False)

    wanted = ("neighbor_distance", "vote_margin", "slides")
    narrow = pd.read_csv(path, usecols=lambda c: c in wanted)
    assert set(narrow.columns) == set(wanted)
    # And the comparison gives the same answer from the narrow read.
    profile = _profile()
    assert (cs.compare(narrow, profile)["novelty_ratio"]
            == cs.compare(pd.read_csv(path), profile)["novelty_ratio"])


def test_the_api_client_posts_to_the_cohort_shift_endpoint(tmp_path):
    import sys as _sys
    _sys.path.insert(0, str(BACKEND.parent / "app"))
    import api_client

    sent = {}

    class _Spy(api_client.TileServerClient):
        def __init__(self):
            pass

        def _post_json(self, path, body, **kw):
            sent["path"], sent["body"] = path, body
            return {"level": "consistent"}

    _Spy().check_cohort_shift("run1")
    assert sent["path"] == "/dataset-jobs/run1/cohort-shift"
    assert sent["body"] == {"csv_path": None, "top_slides": 10}

    _Spy().check_cohort_shift("run1", csv_path="/x.csv", top_slides=3)
    assert sent["body"] == {"csv_path": "/x.csv", "top_slides": 3}


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_shift_test_"))
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
