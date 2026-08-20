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
