"""Tiebreak rules for low-margin tiles.

These tiles are near-ties by definition, so any rule applied to them can lose a
tile as easily as win one. The tests that matter here are the ones proving the
accounting cannot flatter a rule: that a rule is never credited for a tile it
did not change, that breaking a correct tile is counted against it, and that a
rule which is pure noise comes out at roughly zero net rather than positive.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from tiebreak_experiment import RULES, _pick, cluster_centroids, compare  # noqa: E402
from validate_reference import load_reference  # noqa: E402

SCRIPT = BACKEND / "tiebreak_experiment.py"


def _write_overlapping(path: Path, *, n=12000, ncomp=6, nclust=8, k=25, seed=4,
                       spread=1.0, centre_scale=0.9):
    """Clusters that genuinely overlap, so there are low-margin tiles to break
    ties on. Well-separated ones produce an empty subset and prove nothing."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, nclust, n).astype(np.int64)
    centres = rng.standard_normal((nclust, ncomp)) * centre_scale
    vectors = centres[codes] + rng.standard_normal((n, ncomp)) * spread
    np.savez(
        path, reference=vectors.astype(np.float32),
        components=np.zeros((ncomp, ncomp), np.float32), codes=codes,
        categories=np.array([str(i) for i in range(nclust)]),
        n_neighbors=np.int64(k), meta=json.dumps({"groupby": "leiden_2.5"}),
    )


def test_rules_only_ever_return_winner_or_runner_up(tmp_path):
    """A tiebreak that could return a third cluster would not be a tiebreak,
    and would silently widen the change beyond what the report accounts for."""
    ref = tmp_path / "ov.npz"
    _write_overlapping(ref)
    reference = load_reference(ref)

    rng = np.random.default_rng(0)
    rowsn, k_base, k_expand, nclust = 200, 10, 30, 8
    winner = rng.integers(0, nclust, rowsn).astype(np.int64)
    runner = (winner + 1 + rng.integers(0, nclust - 1, rowsn)) % nclust
    labels_all = rng.integers(0, nclust, (rowsn, k_expand)).astype(np.int64)
    dist_all = np.sort(rng.random((rowsn, k_expand)), axis=1)
    queries = rng.standard_normal((rowsn, reference["vectors"].shape[1])).astype(np.float32)
    centroids = cluster_centroids(reference["vectors"], reference["codes"], nclust)

    for rule in RULES:
        chosen = _pick(rule, labels_base=labels_all[:, :k_base],
                       dist_base=dist_all[:, :k_base], labels_all=labels_all,
                       dist_all=dist_all, winner=winner, runner=runner,
                       queries=queries, centroids=centroids, distance_power=2.0)
        assert chosen.shape == winner.shape, rule
        assert np.isin(chosen, np.stack([winner, runner], axis=1)).all(), rule
        assert ((chosen == winner) | (chosen == runner)).all(), rule


def test_nearest_picks_the_candidate_owning_the_closest_neighbour(tmp_path):
    """Built so the answer is known: the runner-up owns column 0, which is the
    closest neighbour, so 'nearest' must switch to it."""
    winner = np.array([3, 3], dtype=np.int64)
    runner = np.array([5, 5], dtype=np.int64)
    # Row 0: runner-up owns the nearest neighbour. Row 1: winner does.
    labels_base = np.array([[5, 3, 3, 3], [3, 5, 5, 5]], dtype=np.int64)
    dist_base = np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]])
    chosen = _pick("nearest", labels_base=labels_base, dist_base=dist_base,
                   labels_all=labels_base, dist_all=dist_base, winner=winner,
                   runner=runner, queries=np.zeros((2, 2), np.float32),
                   centroids=np.zeros((8, 2), np.float32), distance_power=2.0)
    assert chosen.tolist() == [5, 3], chosen


def test_unknown_rule_is_rejected(tmp_path):
    try:
        _pick("coin-flip", labels_base=np.zeros((1, 2), np.int64),
              dist_base=np.zeros((1, 2)), labels_all=np.zeros((1, 2), np.int64),
              dist_all=np.zeros((1, 2)), winner=np.zeros(1, np.int64),
              runner=np.zeros(1, np.int64), queries=np.zeros((1, 2), np.float32),
              centroids=np.zeros((2, 2), np.float32), distance_power=2.0)
    except ValueError as e:
        assert "coin-flip" in str(e)
    else:
        raise AssertionError("an unknown rule must not silently pick something")


def test_net_accounting_matches_fixed_minus_broke(tmp_path):
    """The headline number is net. If fixed/broke/net could disagree with the
    subset accuracy they describe, a losing rule could read as a winner."""
    ref = tmp_path / "ov.npz"
    _write_overlapping(ref)
    result = compare(load_reference(ref), 4000, 0, 10, 30, 2.0, 0.5, 4096)

    n_low = result["n_low"]
    assert n_low > 50, n_low
    for rule, row in result["rules"].items():
        assert row["net"] == row["fixed"] - row["broke"], rule
        # subset accuracy must move by exactly net/n_low
        expected = result["subset_before"] + row["net"] / n_low
        assert abs(row["subset_after"] - expected) < 1e-9, (rule, row)
        # a rule cannot fix or break more tiles than it changed
        assert row["fixed"] + row["broke"] <= row["changed"], (rule, row)


def test_overall_accuracy_blends_untouched_tiles_correctly(tmp_path):
    """Tiles above the margin threshold must be carried through unchanged.
    Recomputing them instead would let a rule take credit for tiles it never
    saw."""
    ref = tmp_path / "ov.npz"
    _write_overlapping(ref)
    result = compare(load_reference(ref), 4000, 0, 10, 30, 2.0, 0.5, 4096)

    baseline, low = result["baseline"], result["low_mask"]
    untouched_correct = int(baseline["correct"][~low].sum())
    for rule, row in result["rules"].items():
        subset_correct = row["subset_after"] * result["n_low"]
        expected = (untouched_correct + subset_correct) / baseline["n"]
        assert abs(row["overall"] - expected) < 1e-9, (rule, row)


def test_ceiling_bounds_every_rule(tmp_path):
    """No A/B rule can beat the share of tiles whose true cluster is A or B.
    A result above the ceiling would mean the accounting is wrong."""
    ref = tmp_path / "ov.npz"
    _write_overlapping(ref)
    result = compare(load_reference(ref), 4000, 0, 10, 30, 2.0, 0.5, 4096)

    assert result["reachable"] <= result["n_low"]
    for rule, row in result["rules"].items():
        assert row["overall"] <= result["ceiling"] + 1e-9, (rule, row)
        assert row["subset_after"] <= result["reachable"] / result["n_low"] + 1e-9, rule


def test_no_low_margin_tiles_is_reported_not_crashed(tmp_path):
    """Well-separated clusters leave nothing to break ties on. That is a real
    answer — 'this lever does not apply here' — not an error."""
    ref = tmp_path / "sep.npz"
    _write_overlapping(ref, spread=0.25, centre_scale=25.0)
    result = compare(load_reference(ref), 2000, 0, 10, 30, 2.0, 0.0, 4096)
    assert result["rules"] == {}

    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--reference", str(ref), "--sample", "2000",
         "--margin-threshold", "0.0"],
        capture_output=True, text=True, cwd=str(BACKEND),
    )
    assert out.returncode == 0, out.stderr
    assert "nothing to break ties on" in out.stdout


def test_cli_runs_and_ranks_rules(tmp_path):
    ref = tmp_path / "ov.npz"
    _write_overlapping(ref)
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--reference", str(ref), "--sample", "4000",
         "--margin-threshold", "0.5", "--k-expand", "30"],
        capture_output=True, text=True, cwd=str(BACKEND),
    )
    assert out.returncode == 0, out.stderr
    for rule in RULES:
        assert rule in out.stdout, rule
    assert "fixed" in out.stdout and "broke" in out.stdout
    # Ranked by net descending, so the first data row is the best rule.
    assert "Best:" in out.stdout or "No rule wins" in out.stdout


def test_raising_the_flip_margin_only_ever_changes_fewer_tiles(tmp_path):
    """The flip margin exists to stop the rule overriding the vote on a lead of
    one part in a thousand. A higher bar must be strictly more conservative —
    if it could change MORE tiles it is not a confidence threshold at all."""
    ref = tmp_path / "ov.npz"
    _write_overlapping(ref)
    margins = (0.0, 0.05, 0.2, 0.5)
    result = compare(load_reference(ref), 4000, 0, 10, 30, 2.0, 0.5, 4096,
                     flip_margins=margins)

    for rule in RULES:
        changed = [result["sweep"][rule][m]["changed"] for m in margins]
        assert changed == sorted(changed, reverse=True), (rule, changed)
        # And at a bar no lead can clear, the vote stands untouched.
        untouched = compare(load_reference(ref), 4000, 0, 10, 30, 2.0, 0.5, 4096,
                            flip_margins=(1.01,))
        assert untouched["sweep"][rule][1.01]["changed"] == 0, rule
        assert untouched["sweep"][rule][1.01]["net"] == 0, rule
        assert abs(untouched["sweep"][rule][1.01]["overall"]
                   - untouched["overall_before"]) < 1e-12, rule


def test_flip_margin_zero_matches_the_plain_comparison(tmp_path):
    """flip_margin=0 is the original 'override on any lead' rule. It has to
    stay that, or the sweep's first column is not the baseline it is read as."""
    ref = tmp_path / "ov.npz"
    _write_overlapping(ref)
    reference = load_reference(ref)
    result = compare(reference, 4000, 0, 10, 30, 2.0, 0.5, 4096,
                     flip_margins=(0.0, 0.1))
    for rule in RULES:
        assert result["rules"][rule] == result["sweep"][rule][0.0], rule


def test_advantage_is_neutral_when_a_candidate_has_no_evidence(tmp_path):
    """A candidate absent from the neighbours scores -inf. That must read as
    'no evidence to override', never as a flip driven by an infinity."""
    from tiebreak_experiment import _advantage

    score_a = np.array([1.0, -np.inf, 2.0, -np.inf])
    score_b = np.array([2.0, 1.0, -np.inf, -np.inf])
    advantage = _advantage(score_a, score_b)
    assert np.isfinite(advantage).all(), advantage
    assert advantage[0] > 0          # B genuinely ahead
    assert (advantage[1:] == -1.0).all(), advantage[1:]

    # A rule scoring negated distances must still favour the closer candidate.
    closer_b = _advantage(np.array([-4.0]), np.array([-1.0]))
    assert closer_b[0] > 0, closer_b


def test_centroids_match_a_direct_mean(tmp_path):
    """cluster_centroids uses per-dimension bincounts for speed; it has to
    agree with the obvious computation."""
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 5, 500).astype(np.int64)
    vectors = rng.standard_normal((500, 7)).astype(np.float32)
    got = cluster_centroids(vectors, codes, 5)
    for code in range(5):
        expected = vectors[codes == code].mean(axis=0)
        assert np.allclose(got[code], expected, atol=1e-5), code


# --- the cached search must be the same search ---------------------------
#
# The cache turns an 18-minute search into nothing, which is only sound if the
# numbers are identical. If they were not, the cheap path would produce a whole
# report of plausible figures for a slightly different question.


def _without_timing(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "elapsed"}


def test_the_cached_path_gives_identical_results(tmp_path):
    ref = tmp_path / "ref.npz"
    _write_overlapping(ref)
    reference = load_reference(ref)
    cache = tmp_path / "nbr.npz"

    plain = compare(reference, 4000, 0, 10, 30, 2.0, 0.5, 4096)
    cached = compare(reference, 4000, 0, 10, 30, 2.0, 0.5, 4096, cache=cache)
    assert cache.is_file(), "the cache was never written"
    reused = compare(reference, 4000, 0, 10, 30, 2.0, 0.5, 4096, cache=cache)

    for other, label in ((cached, "first cached run"), (reused, "reused cache")):
        assert other["baseline"]["accuracy"] == plain["baseline"]["accuracy"], label
        assert np.array_equal(other["baseline"]["query_index"],
                              plain["baseline"]["query_index"]), label
        assert np.array_equal(other["baseline"]["predicted"],
                              plain["baseline"]["predicted"]), label
        assert other["n_low"] == plain["n_low"], label
        assert other["reachable"] == plain["reachable"], label
        for rule in RULES:
            # Everything but the timing, which is wall-clock and is the one
            # field the cache is supposed to change.
            assert _without_timing(other["rules"][rule]) \
                == _without_timing(plain["rules"][rule]), f"{label}: {rule}"


def test_a_cache_too_narrow_for_k_expand_is_refused(tmp_path):
    """A cache built at k_max=20 cannot answer --k-expand 30. Truncating the
    'restricted' rule to fewer neighbours than asked for would change its answer
    while reporting the k that was requested."""
    ref = tmp_path / "ref.npz"
    _write_overlapping(ref)
    reference = load_reference(ref)
    cache = tmp_path / "nbr.npz"

    # Seed the cache narrow, by asking for a small k_expand first.
    compare(reference, 4000, 0, 10, 12, 2.0, 0.5, 4096, cache=cache)
    stored_width = int(np.load(cache)["idx"].shape[1])
    assert stored_width == 12

    # Now ask for wider. It must re-search, not serve 12 columns as 30.
    wide = compare(reference, 4000, 0, 10, 30, 2.0, 0.5, 4096, cache=cache)
    plain = compare(reference, 4000, 0, 10, 30, 2.0, 0.5, 4096)
    assert int(np.load(cache)["idx"].shape[1]) == 30, "the cache was not rebuilt"
    for rule in RULES:
        assert _without_timing(wide["rules"][rule]) \
            == _without_timing(plain["rules"][rule]), rule


def test_a_cache_from_tune_classifier_is_reused_not_re_searched(tmp_path):
    """The whole point of the cache: one 18-minute search serves both tools.
    Both derive the query set from (sample, seed) independently, so if either
    changes how it samples, this stops being a reuse and becomes a silent
    re-search — correct, but paying the cost twice with no sign of it."""
    import io
    import contextlib
    import tune_classifier as tc

    ref = tmp_path / "ref.npz"
    _write_overlapping(ref)
    reference = load_reference(ref)
    cache = tmp_path / "shared.npz"

    # Written by the sweep, at k_max wide enough for the tiebreak's k_expand.
    tc.sweep(reference, 4000, 0, 30, 4096, [10], [2.0], [0], [0.0], [15],
             [False], cache)
    stored = np.load(cache)["query_index"]

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        result = compare(reference, 4000, 0, 10, 30, 2.0, 0.5, 4096, cache=cache)
    output = captured.getvalue()

    assert "reusing" in output, f"the sweep's cache was not reused:\n{output}"
    assert "does not match" not in output, output
    assert np.array_equal(result["baseline"]["query_index"], stored), \
        "the two tools disagree about which tiles (sample, seed) selects"


# --- the blind-swap accounting -------------------------------------------
#
# This is the "the truth is usually the runner-up, so just take the runner-up"
# idea. The 93% it rests on is conditioned on knowing the tile is wrong, which
# is what assignment does not know, so the accounting has to be explicit.


def test_a_blind_swap_breaks_every_correct_tile_in_the_band(tmp_path):
    """broke must equal the number of correct tiles in the band, exactly.
    Anything less means the swap is being credited for tiles it did not touch."""
    from tiebreak_experiment import blind_swap_bands
    ref = tmp_path / "ref.npz"
    _write_overlapping(ref)
    result = compare(load_reference(ref), 4000, 0, 10, 30, 2.0, 0.5, 4096)
    baseline = result["baseline"]
    margins, correct = baseline["margins"], baseline["correct"]

    rows = [r for r in blind_swap_bands(baseline) if r["n"]]
    assert rows, "no bands had any tiles"
    for row in rows:
        band = margins < row["threshold"]
        assert row["n"] == int(band.sum())
        assert row["broke"] == int(correct[band].sum())
        assert row["error_rate"] == float((~correct[band]).mean())
        # fixed can only ever be errors whose truth was the runner-up.
        assert row["fixed"] <= int((~correct[band]).sum())
        assert row["net"] == row["fixed"] - row["broke"]


def test_a_blind_swap_only_pays_in_a_majority_error_band(tmp_path):
    """The break-even. With r = P(truth is runner-up | wrong), net > 0 requires
    the band to be more than 1/(1+r) wrong. This asserts the arithmetic holds on
    real bands rather than restating the algebra."""
    from tiebreak_experiment import blind_swap_bands
    ref = tmp_path / "ref.npz"
    _write_overlapping(ref)
    result = compare(load_reference(ref), 4000, 0, 10, 30, 2.0, 0.5, 4096)
    baseline = result["baseline"]

    wrong = ~baseline["correct"]
    assert wrong.any(), "the fixture has no errors, so this proves nothing"
    r = float((baseline["runner_up"][wrong] == baseline["truth"][wrong]).mean())
    breakeven = 1.0 / (1.0 + r)

    paid = [row for row in blind_swap_bands(baseline) if row["n"] and row["net"] > 0]
    for row in paid:
        assert row["error_rate"] > breakeven, (
            f"band margin<{row['threshold']} nets {row['net']:+} at only "
            f"{row['error_rate']:.1%} error, below the {breakeven:.1%} break-even")


def test_bands_are_cumulative_and_monotone(tmp_path):
    """Reported as margin < t, so tile counts must only grow with t. A per-band
    (rather than cumulative) reading of the same table would be a different and
    much more optimistic claim."""
    from tiebreak_experiment import blind_swap_bands
    ref = tmp_path / "ref.npz"
    _write_overlapping(ref)
    baseline = compare(load_reference(ref), 4000, 0, 10, 30, 2.0,
                       0.5, 4096)["baseline"]
    rows = [r for r in blind_swap_bands(baseline) if r["n"]]
    counts = [r["n"] for r in rows]
    assert counts == sorted(counts)
    assert all(r["broke"] >= 0 and r["fixed"] >= 0 for r in rows)


# --- adaptive k vs a tiebreak rule, on the same band ---------------------
#
# On the production reference `restricted` and adaptive k both score 97.23%.
# That has two opposite readings -- one mechanism twice, or two mechanisms that
# happen to tie -- and only the cross-tab distinguishes them. So the cross-tab
# has to be right about which tiles each policy actually fixes.


def _band_result(tmp_path, **over):
    ref = tmp_path / "ref.npz"
    _write_overlapping(ref, n=20000, ncomp=8, nclust=12, spread=1.0,
                       centre_scale=1.4, **over)
    reference = load_reference(ref)
    return reference, compare(reference, 8000, 0, 10, 40, 3.0, 0.4, 4096,
                              cache=tmp_path / "nbr.npz")


def test_the_cross_tab_reproduces_the_rules_own_net(tmp_path):
    """compare_policies re-derives the rule's decision from the carried band.
    If it used different inputs -- the wrong k_base, distances instead of
    squared ones -- it would report a rule that was never in the table."""
    from tiebreak_experiment import compare_policies
    reference, result = _band_result(tmp_path)
    for rule in RULES:
        for margin in result["sweep"][rule]:
            cmp = compare_policies(result, reference, 25, 3.0, margin, rule)
            assert cmp["tiebreak"]["net"] == result["sweep"][rule][margin]["net"], \
                f"{rule} @ {margin}"
            assert cmp["tiebreak"]["fixed"] == result["sweep"][rule][margin]["fixed"]
            assert cmp["tiebreak"]["broke"] == result["sweep"][rule][margin]["broke"]


def test_the_two_by_two_partitions_the_band_exactly(tmp_path):
    """Every band tile lands in exactly one of the four cells. A cell that
    double-counts would overstate complementarity, which is the one conclusion
    this analysis is for."""
    from tiebreak_experiment import compare_policies
    reference, result = _band_result(tmp_path)
    cmp = compare_policies(result, reference, 25, 3.0, 0.1, "restricted")
    cells = (cmp["only_adaptive"] + cmp["only_tiebreak"]
             + cmp["both_correct"] + cmp["neither"])
    assert cells == cmp["n_band"], (cells, cmp["n_band"])


def test_the_oracle_union_bounds_both_policies_and_the_hybrid(tmp_path):
    """It is the ceiling on anything built from the pair, so nothing built from
    the pair may exceed it."""
    from tiebreak_experiment import best_hybrid, compare_policies
    reference, result = _band_result(tmp_path)
    cmp = compare_policies(result, reference, 25, 3.0, 0.1, "restricted")
    _, hybrid = best_hybrid(cmp)
    for row in (cmp["adaptive"], cmp["tiebreak"], hybrid):
        assert row["overall"] <= cmp["oracle_union"] + 1e-12


def test_the_hybrid_gate_is_not_the_flip_margin(tmp_path):
    """The first version gated on the flip margin, so at flip 0 the hybrid was
    adaptive k verbatim and looked like it added nothing. The gate is a
    different quantity -- how tied adaptive's own re-vote still is."""
    from tiebreak_experiment import compare_policies
    reference, result = _band_result(tmp_path)
    cmp = compare_policies(result, reference, 25, 3.0, 0.0, "restricted")
    nets = {g: row["net"] for g, row in cmp["hybrid_sweep"].items()}
    assert len(nets) > 1
    assert len(set(nets.values())) > 1, (
        f"every gate gives the same net ({nets}) — the gate is not doing "
        f"anything, which is what the flip-margin bug looked like")


def test_a_wider_hybrid_gate_only_ever_overrules_more(tmp_path):
    """The gate is a threshold on adaptive's margin, so raising it can only add
    tiles to the overruled set. A non-monotone changed-count means the gate is
    being applied to the wrong quantity."""
    from tiebreak_experiment import compare_policies
    reference, result = _band_result(tmp_path)
    cmp = compare_policies(result, reference, 25, 3.0, 0.1, "restricted")
    gates = sorted(cmp["hybrid_sweep"])
    overruled = [cmp["hybrid_sweep"][g] for g in gates]
    # 'changed' is against the ORIGINAL winner, so it is not itself monotone;
    # what must be monotone is how many rows the gate hands to the rule. Check
    # that directly by rebuilding the masks.
    from tiebreak_experiment import adaptive_choice
    band = result["band"]
    adaptive = adaptive_choice(band, reference["codes"],
                               len(reference["categories"]), 25, 3.0)
    counts = [int((adaptive["margin"] < g).sum()) for g in gates]
    assert counts == sorted(counts), counts
    assert overruled  # and the sweep actually produced rows


def test_adaptive_choice_matches_a_plain_wide_vote(tmp_path):
    """The cross-tab's adaptive arm must be the shipped policy, not an
    approximation of it: a wide vote over every cluster on the band's rows."""
    from tiebreak_experiment import adaptive_choice
    from assign_hpc_clusters import vote
    reference, result = _band_result(tmp_path)
    band = result["band"]
    n_clusters = len(reference["categories"])
    adaptive = adaptive_choice(band, reference["codes"], n_clusters, 25, 3.0)
    direct, margin, _ = vote(band["idx"][:, :25], band["dist_sq"][:, :25],
                             reference["codes"], n_clusters,
                             distance_weighted=True, distance_power=3.0)
    assert np.array_equal(adaptive["chosen"], direct)
    assert np.allclose(adaptive["margin"], margin)
    # And it is genuinely unrestricted: it may land outside the base top two.
    outside = ((adaptive["chosen"] != band["winner"])
               & (adaptive["chosen"] != band["runner"]))
    assert outside.any(), (
        "adaptive k never left the base top-2 on this fixture, so the test "
        "cannot tell it apart from an A/B rule")


# --- the disagreement profile --------------------------------------------
#
# This is the tool that decides whether the oracle union is reachable, so its
# failure mode is a false positive: declaring a selector exists because four
# quartiles of a few hundred tiles happened to slope. The verdict therefore has
# to be able to come out negative, and it has to distinguish a monotone trend
# from noise that merely spreads.


def test_the_two_kinds_of_disagreement_partition_it(tmp_path):
    """out-of-pair and same-pair must together be every disagreement. If they
    did not, one kind would be silently dropped from the analysis."""
    from tiebreak_experiment import disagreement_profile
    reference, result = _band_result(tmp_path)
    prof = disagreement_profile(result, reference, 25, 3.0, 0.1, "restricted")
    assert (prof["out_of_pair"].get("n", 0) + prof["same_pair"].get("n", 0)
            == prof["n_disagree"])


def test_out_of_pair_is_exactly_what_the_rule_cannot_reach(tmp_path):
    """An A/B rule can only ever return one of its two candidates, so an
    out-of-pair tile is by definition one adaptive k reached and it could not.
    Counting any same-pair tile there would overstate the cost of the A/B
    restriction, which is the one thing this cut is for."""
    from tiebreak_experiment import adaptive_choice, disagreement_profile
    reference, result = _band_result(tmp_path)
    band = result["band"]
    adaptive = adaptive_choice(band, reference["codes"],
                               len(reference["categories"]), 25, 3.0)
    outside = ((adaptive["chosen"] != band["winner"])
               & (adaptive["chosen"] != band["runner"]))
    prof = disagreement_profile(result, reference, 25, 3.0, 0.1, "restricted")
    # Every out-of-pair tile is a disagreement by construction, so the count of
    # tiles adaptive took outside the pair IS the out-of-pair disagreement count.
    assert prof["left_the_pair"] == int(outside.sum())
    assert prof["out_of_pair"]["n"] == int(outside.sum())
    assert prof["left_and_right"] == int(
        (outside & (adaptive["chosen"] == band["truth"])).sum())


def test_every_stratum_counts_only_decisive_tiles_in_its_share(tmp_path):
    """adaptive_share is a head-to-head share, so tiles where NEITHER policy is
    right must not sit in its denominator — they would drag every stratum
    toward each other and flatten a real signal into noise."""
    from tiebreak_experiment import disagreement_profile
    reference, result = _band_result(tmp_path)
    prof = disagreement_profile(result, reference, 25, 3.0, 0.1, "restricted")
    for rows in prof["strata"].values():
        for row in rows:
            if not row["n"]:
                continue
            decisive = row["adaptive_right"] + row["tiebreak_right"]
            assert row["n"] == decisive + row["neither"]
            if decisive:
                assert abs(row["adaptive_share"]
                           - row["adaptive_right"] / decisive) < 1e-12


def test_a_flat_stratum_is_reported_as_noise_not_a_selector(tmp_path):
    """The false positive this guards. Equal shares across strata must never
    read as a selector, however many tiles there are."""
    import io
    import contextlib
    from tiebreak_experiment import _report_stratum_verdict
    rows = [{"n": 400, "adaptive_right": 150, "tiebreak_right": 150,
             "neither": 100, "adaptive_share": 0.5, "lo": 0.0, "hi": 1.0}
            for _ in range(4)]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _report_stratum_verdict(rows)
    assert "no signal" in out.getvalue(), out.getvalue()


def test_a_small_spread_on_few_tiles_is_noise_not_a_selector(tmp_path):
    """15 points across four bins of ~30 decisive tiles is well within binomial
    noise. Judging on spread alone -- the first version of this -- called it a
    selector."""
    import io
    import contextlib
    from tiebreak_experiment import _report_stratum_verdict
    shares = (0.35, 0.42, 0.45, 0.50)
    rows = []
    for share in shares:
        right = int(round(share * 30))
        rows.append({"n": 40, "adaptive_right": right,
                     "tiebreak_right": 30 - right, "neither": 10,
                     "adaptive_share": right / 30, "lo": 0.0, "hi": 1.0})
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _report_stratum_verdict(rows)
    assert "no signal" in out.getvalue(), out.getvalue()


def test_a_non_monotone_trend_is_not_called_a_selector(tmp_path):
    """A real but disordered difference gives nothing to threshold on, so it
    must be reported differently from a usable trend."""
    import io
    import contextlib
    from tiebreak_experiment import _report_stratum_verdict
    shares = (0.30, 0.70, 0.32, 0.35)
    rows = []
    for share in shares:
        right = int(round(share * 400))
        rows.append({"n": 500, "adaptive_right": right,
                     "tiebreak_right": 400 - right, "neither": 100,
                     "adaptive_share": right / 400, "lo": 0.0, "hi": 1.0})
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _report_stratum_verdict(rows)
    text = out.getvalue()
    assert "non-monotone" in text and "no threshold" in text, text


def test_a_strong_monotone_trend_is_called_a_selector(tmp_path):
    """And the verdict must be able to come out positive, or it is just a
    rubber stamp in the other direction."""
    import io
    import contextlib
    from tiebreak_experiment import _report_stratum_verdict
    shares = (0.65, 0.55, 0.40, 0.25)
    rows = []
    for share in shares:
        right = int(round(share * 400))
        rows.append({"n": 500, "adaptive_right": right,
                     "tiebreak_right": 400 - right, "neither": 100,
                     "adaptive_share": right / 400, "lo": 0.0, "hi": 1.0})
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _report_stratum_verdict(rows)
    assert "this is a selector" in out.getvalue(), out.getvalue()


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_tiebreak_test_"))
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
