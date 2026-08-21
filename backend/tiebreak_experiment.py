#!/usr/bin/env python3
"""Compare tiebreak rules for low-margin tiles, on the top two clusters only.

Why only the top two. The error diagnostics in validate_reference.py measured,
on the leiden_5.0 reference, that every error had the true cluster among the k
neighbours and 90% of them had it as the *runner-up* — median rank 2. So the
information needed to fix them is not another neighbour and not another space;
it is a better decision between two specific clusters that the vote scored
almost equally. That is a much narrower question than "which of 109 clusters",
and a rule can use evidence the vote throws away.

Each rule chooses between the winner A and the runner-up B, and only for tiles
whose vote_margin was below a threshold. Everything else keeps its answer.

    nearest      whichever of A/B owns the single closest neighbour. Uses no
                 extra search — the vote already had this and averaged it away.
    mean-dist    whichever of A/B has the lower *mean* distance over its own
                 neighbours. Sensitive to a cluster whose members sit at a
                 consistent distance rather than one lucky close point.
    restricted   re-query at a larger k but count only A and B, ignoring every
                 other cluster. This is the fix for why plain adaptive k
                 stalled: a bigger k did add A/B evidence, but it also added
                 noise from clusters that were never in contention.
    centroid     whichever cluster's centroid over the whole reference is
                 closer. The only rule here using global shape rather than the
                 local neighbourhood.

The number that matters is net, not fixed. A tiebreak that fixes 200 tiles and
breaks 180 is worthless, and reporting only the first would hide that — these
tiles are near-ties, so any rule is being applied to cases it can lose as
easily as win.

Usage:
    python tiebreak_experiment.py --reference ref_raw128.npz
    python tiebreak_experiment.py --reference ... --margin-threshold 0.25 --k-expand 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from assign_hpc_clusters import Searcher, _WEIGHT_EPS, vote  # noqa: E402
from tune_classifier import gather_neighbours  # noqa: E402
from validate_reference import describe_reference, leave_one_out, load_reference  # noqa: E402
from build_hpc_reference import HPC_REFERENCE_PATH  # noqa: E402

RULES = ("nearest", "mean-dist", "restricted", "centroid")

# Measured on the production reference: of the tiles the vote gets wrong,
# this fraction have the true cluster as the runner-up. Used only for the
# break-even figure printed before the sweep -- the report recomputes it
# from the sample in hand rather than trusting this.
_RUNNER_UP_SHARE = 0.93

# How far ahead the runner-up must be before the vote is overridden. 0.0 is
# "override on any lead at all", which is what a plain tiebreak does — and the
# reason it breaks correct tiles, since a lead of one part in a thousand is not
# evidence. Sweeping costs nothing once the scores exist.
FLIP_MARGINS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)


def cluster_centroids(vectors: np.ndarray, codes: np.ndarray,
                      n_clusters: int) -> np.ndarray:
    """Mean vector per cluster. One bincount per dimension rather than
    np.add.at, which is orders of magnitude slower at this size."""
    counts = np.maximum(np.bincount(codes, minlength=n_clusters), 1)
    sums = np.empty((n_clusters, vectors.shape[1]), dtype=np.float64)
    for dim in range(vectors.shape[1]):
        sums[:, dim] = np.bincount(codes, weights=vectors[:, dim].astype(np.float64),
                                   minlength=n_clusters)
    return (sums / counts[:, None]).astype(np.float32)


def _scores(rule: str, *, labels_base: np.ndarray, dist_base: np.ndarray,
            labels_all: np.ndarray, dist_all: np.ndarray, winner: np.ndarray,
            runner: np.ndarray, queries: np.ndarray, centroids: np.ndarray,
            distance_power: float) -> tuple[np.ndarray, np.ndarray]:
    """Score for A and for B under `rule`, where *higher is better*.

    Every rule returns a pair rather than a decision so that the flip margin
    below applies identically to all of them, and so a rule's confidence — not
    just its verdict — is available. Distance-based rules return negated
    distances to keep the direction uniform.
    """
    is_a = labels_base == winner[:, None]
    is_b = labels_base == runner[:, None]

    if rule == "nearest":
        # Neighbours come back sorted, so the smallest distance belonging to a
        # candidate is that candidate's closest. Absent candidates score -inf,
        # which the flip margin reads as "no evidence", not "evidence against".
        with np.errstate(invalid="ignore"):
            near_a = np.where(is_a.any(axis=1), np.min(np.where(is_a, dist_base, np.inf), axis=1), np.inf)
            near_b = np.where(is_b.any(axis=1), np.min(np.where(is_b, dist_base, np.inf), axis=1), np.inf)
        return -near_a, -near_b

    if rule == "mean-dist":
        with np.errstate(invalid="ignore"):
            mean_a = np.where(is_a.any(axis=1),
                              (dist_base * is_a).sum(axis=1) / np.maximum(is_a.sum(axis=1), 1),
                              np.inf)
            mean_b = np.where(is_b.any(axis=1),
                              (dist_base * is_b).sum(axis=1) / np.maximum(is_b.sum(axis=1), 1),
                              np.inf)
        return -mean_a, -mean_b

    if rule == "restricted":
        wide_a = labels_all == winner[:, None]
        wide_b = labels_all == runner[:, None]
        weights = 1.0 / (dist_all + _WEIGHT_EPS) ** distance_power
        return (weights * wide_a).sum(axis=1), (weights * wide_b).sum(axis=1)

    if rule == "centroid":
        return (-np.linalg.norm(queries - centroids[winner], axis=1),
                -np.linalg.norm(queries - centroids[runner], axis=1))

    raise ValueError(f"unknown rule {rule!r}")


def _advantage(score_a: np.ndarray, score_b: np.ndarray) -> np.ndarray:
    """B's lead over A, scaled to roughly [-1, 1] so one flip-margin threshold
    means a comparable thing across rules with different score units.

    For the weight-based rule this is the usual (b-a)/(a+b) share. For a
    negated-distance rule it works out to (d_a - d_b)/(d_a + d_b), the relative
    distance gap — also the right notion of "clearly closer". Rows where either
    side has no evidence at all resolve to -1: keep the vote's answer.
    """
    finite = np.isfinite(score_a) & np.isfinite(score_b)
    denominator = np.abs(score_a) + np.abs(score_b) + _WEIGHT_EPS
    with np.errstate(invalid="ignore"):
        advantage = (score_b - score_a) / denominator
    return np.where(finite, advantage, -1.0)


def _pick(rule: str, *, flip_margin: float = 0.0, **kwargs) -> np.ndarray:
    """The chosen cluster per row: winner unless the runner-up leads by more
    than flip_margin. flip_margin=0 overrides on any lead at all."""
    score_a, score_b = _scores(rule, **kwargs)
    return np.where(_advantage(score_a, score_b) > flip_margin,
                    kwargs["runner"], kwargs["winner"])


def compare(reference: dict, sample: int, seed: int, k_base: int, k_expand: int,
            distance_power: float, margin_threshold: float, batch: int,
            rules: tuple[str, ...] = RULES,
            flip_margins: tuple[float, ...] = FLIP_MARGINS,
            cache: Path | None = None) -> dict:
    """Baseline at k_base, then every rule applied to the low-margin tiles, at
    every flip margin.

    With `cache`, both searches come out of one cached neighbour matrix — the
    same file tune_classifier.py writes. Against the production reference the
    search is the entire cost (~18 min for 200,000 queries), and this experiment
    otherwise pays it twice: once for the baseline vote and once to expand the
    low-margin tiles to k_expand.
    """
    if cache is not None:
        # Reproduce leave_one_out's own sampling so the cached rows line up with
        # the query set it would have chosen. Derived here rather than passed in
        # because a mismatch is silent: every accuracy would be real, just for
        # different tiles.
        total = len(reference["vectors"])
        rng = np.random.default_rng(seed)
        query_index = (np.arange(total) if sample >= total
                       else np.sort(rng.choice(total, size=sample, replace=False)))
        idx_all, dist_all_sq = gather_neighbours(
            reference, query_index, max(k_base, k_expand), batch, cache)
        baseline = leave_one_out(reference, sample, k_base, batch, seed,
                                 distance_weighted=True,
                                 distance_power=distance_power,
                                 query_index=query_index,
                                 neighbours=(idx_all, dist_all_sq))
    else:
        idx_all = dist_all_sq = None
        baseline = leave_one_out(reference, sample, k_base, batch, seed,
                                 distance_weighted=True,
                                 distance_power=distance_power)

    low_mask = baseline["margins"] < margin_threshold
    if not low_mask.any():
        return {"baseline": baseline, "low_mask": low_mask, "rules": {}}

    vectors, codes = reference["vectors"], reference["codes"]
    rows = baseline["query_index"][low_mask]
    winner = baseline["predicted"][low_mask]
    runner = baseline["runner_up"][low_mask]
    truth = baseline["truth"][low_mask]

    queries = np.ascontiguousarray(vectors[rows])
    if idx_all is not None:
        # Already self-excluded, so the low-margin rows of the cached matrix are
        # exactly what a k_expand search would have returned for them.
        idx = idx_all[low_mask, :k_expand]
        dist_sq = dist_all_sq[low_mask, :k_expand]
        dist = np.sqrt(np.maximum(dist_sq, 0.0))
    else:
        searcher = Searcher(vectors)
        # k_expand + 1 for the same reason leave_one_out uses k + 1: every tile
        # is its own nearest neighbour and must be dropped.
        idx, dist = searcher.search(queries, k_expand + 1)

        self_mask = idx == rows[:, None]
        no_self = ~self_mask.any(axis=1)
        if no_self.any():
            self_mask[no_self, -1] = True
        keep = ~self_mask
        idx = idx[keep].reshape(len(rows), k_expand)
        dist_sq = dist[keep].reshape(len(rows), k_expand)
        dist = np.sqrt(np.maximum(dist_sq, 0.0))

    valid = idx >= 0
    labels_all = np.where(valid, codes[np.where(valid, idx, 0)], -1)
    # The first k_base columns are exactly the neighbours the baseline voted
    # on: same index, same order, just truncated.
    labels_base, dist_base = labels_all[:, :k_base], dist[:, :k_base]

    centroids = cluster_centroids(vectors, codes, len(reference["categories"]))

    before_correct = winner == truth
    # No rule choosing between A and B can help a tile whose true cluster is
    # neither. That is the ceiling for this whole approach, and it is worth
    # printing next to every result so a rule is not judged against 100%.
    reachable = (truth == winner) | (truth == runner)

    untouched_correct = int(baseline["correct"][~low_mask].sum())

    def _score_one(chosen: np.ndarray, elapsed: float) -> dict:
        after_correct = chosen == truth
        fixed = int((~before_correct & after_correct).sum())
        broke = int((before_correct & ~after_correct).sum())
        return {
            "fixed": fixed, "broke": broke, "net": fixed - broke,
            "subset_before": float(before_correct.mean()),
            "subset_after": float(after_correct.mean()),
            "overall": (untouched_correct + int(after_correct.sum())) / baseline["n"],
            "changed": int((chosen != winner).sum()),
            "elapsed": elapsed,
        }

    results, sweep = {}, {}
    for rule in rules:
        started = time.perf_counter()
        score_a, score_b = _scores(rule, labels_base=labels_base, dist_base=dist_base,
                                   labels_all=labels_all, dist_all=dist, winner=winner,
                                   runner=runner, queries=queries, centroids=centroids,
                                   distance_power=distance_power)
        advantage = _advantage(score_a, score_b)
        elapsed = time.perf_counter() - started
        # Scores are computed once, so sweeping the flip margin costs one
        # comparison per threshold — no extra search, no extra vote.
        sweep[rule] = {
            float(margin): _score_one(np.where(advantage > margin, runner, winner), elapsed)
            for margin in flip_margins
        }
        results[rule] = sweep[rule][float(flip_margins[0])]

    return {
        "baseline": baseline, "low_mask": low_mask, "rules": results, "sweep": sweep,
        # Everything a policy comparison needs, so it costs no second search.
        "band": {"rows": rows, "winner": winner, "runner": runner, "truth": truth,
                 "idx": idx, "dist_sq": dist_sq, "labels_all": labels_all,
                 "queries": queries, "centroids": centroids, "k_base": k_base,
                 "untouched_correct": untouched_correct,
                 "n_total": baseline["n"]},
        "flip_margins": tuple(float(m) for m in flip_margins),
        "n_low": int(low_mask.sum()), "reachable": int(reachable.sum()),
        "subset_before": float(before_correct.mean()),
        "overall_before": baseline["accuracy"],
        "ceiling": (untouched_correct + int(reachable.sum())) / baseline["n"],
    }


# What fraction of a band has to be wrong before swapping every tile in it to
# its runner-up pays off. Swapping fixes the wrong tiles whose truth IS the
# runner-up, and breaks every tile that was already right:
#
#     net per tile = p * r - (1 - p)      p = fraction wrong, r = P(truth is
#                                         runner-up | wrong) ~ 0.93
#     net > 0   <=>   p > 1 / (1 + r)
#
# At r = 0.93 that is p > 51.8%. The band has to be MAJORITY ERROR. This is the
# whole reason "the truth is usually the runner-up" does not license taking the
# runner-up: that 93% is conditioned on already knowing the tile is wrong, which
# at assignment time is exactly what is unknown.
def blind_swap_bands(baseline: dict, thresholds=(0.01, 0.02, 0.05, 0.10, 0.15,
                                                 0.25, 0.50, 0.75)) -> list[dict]:
    """Per margin band: how wrong it is, and what a blind swap would score.

    Reported cumulatively (margin < t) because that is the shape of rule anyone
    would actually ship — a single threshold below which the runner-up is taken.
    """
    margins, correct = baseline["margins"], baseline["correct"]
    truth, runner = baseline["truth"], baseline["runner_up"]
    rows = []
    for t in thresholds:
        band = margins < t
        n = int(band.sum())
        if not n:
            rows.append({"threshold": t, "n": 0})
            continue
        wrong = ~correct[band]
        # Swap every tile in the band to its runner-up.
        fixed = int((wrong & (runner[band] == truth[band])).sum())
        broke = int(correct[band].sum())
        rows.append({
            "threshold": t, "n": n,
            "share": n / len(margins),
            "error_rate": float(wrong.mean()),
            "fixed": fixed, "broke": broke, "net": fixed - broke,
        })
    return rows


def report_blind_swap(baseline: dict, rows: list[dict]) -> None:
    """The 'just take the runner-up' idea, priced."""
    reachable = 1.0 / (1.0 + _RUNNER_UP_SHARE) if _RUNNER_UP_SHARE else 1.0
    wrong = ~baseline["correct"]
    n_wrong = int(wrong.sum())
    if n_wrong:
        measured = float((baseline["runner_up"][wrong]
                          == baseline["truth"][wrong]).mean())
        breakeven = 1.0 / (1.0 + measured)
    else:
        measured, breakeven = 0.0, 1.0

    print(f"\n=== Taking the runner-up outright, by margin band ===")
    print(f"On this sample the true cluster is the runner-up for "
          f"{measured * 100:.0f}% of the {n_wrong:,} errors. So swapping a whole "
          f"band pays only where more than {breakeven * 100:.1f}% of that band is "
          f"already wrong — it fixes {measured * 100:.0f}% of the errors and breaks "
          f"every tile that was right.")
    print(f"\n  {'margin <':>9}  {'tiles':>8}  {'of all':>7}  {'wrong':>7}  "
          f"{'fixed':>6}  {'broke':>6}  {'net':>7}")
    for row in rows:
        if not row["n"]:
            print(f"  {row['threshold']:>9.2f}  {0:>8}       —        —"
                  f"       —       —        —")
            continue
        flag = "  <-- pays" if row["net"] > 0 else ""
        print(f"  {row['threshold']:>9.2f}  {row['n']:>8,}  "
              f"{row['share'] * 100:>6.1f}%  {row['error_rate'] * 100:>6.1f}%  "
              f"{row['fixed']:>6,}  {row['broke']:>6,}  {row['net']:>+7,}{flag}")

    best = max((r for r in rows if r["n"]), key=lambda r: r["net"], default=None)
    if best is None or best["net"] <= 0:
        print(f"\n  No band is majority-error, so a blind swap loses everywhere — "
              f"best is {best['net']:+,} at margin < {best['threshold']:g}. "
              f"The runner-up being usually right is not usable on its own; it "
              f"takes a rule that decides WHICH tiles to swap. That is what the "
              f"tiebreak rules below are, and the ceiling above is what they are "
              f"chasing." if best else "")
    else:
        print(f"\n  Margin < {best['threshold']:g} is {best['error_rate'] * 100:.1f}% "
              f"wrong, above the {breakeven * 100:.1f}% break-even, so a blind swap "
              f"there is worth {best['net']:+,} net. Still compare it against the "
              f"rules below, which should beat it by not swapping the tiles they "
              f"can tell are already right.")


def adaptive_choice(band: dict, codes: np.ndarray, n_clusters: int,
                    k_adaptive: int, distance_power: float) -> dict:
    """The shipped adaptive-k policy, evaluated on the same band.

    Not an A/B rule: it re-votes over *every* cluster at a wider k, so it can
    land on a third cluster the base vote had nowhere near the top. Which is
    exactly why it is worth cross-tabulating against a tiebreak rather than
    assuming the two are interchangeable because they score the same.
    """
    winners, margins, _, counts = vote(
        band["idx"][:, :k_adaptive], band["dist_sq"][:, :k_adaptive],
        codes, n_clusters, distance_weighted=True,
        distance_power=distance_power, return_counts=True)
    order = np.argsort(counts, axis=1)
    return {"chosen": winners, "margin": margins,
            "winner": order[:, -1], "runner": order[:, -2]}


def _band_score(band: dict, chosen: np.ndarray) -> dict:
    truth, winner = band["truth"], band["winner"]
    before, after = winner == truth, chosen == truth
    fixed = int((~before & after).sum())
    broke = int((before & ~after).sum())
    return {
        "fixed": fixed, "broke": broke, "net": fixed - broke,
        "changed": int((chosen != winner).sum()),
        "overall": (band["untouched_correct"] + int(after.sum())) / band["n_total"],
        "correct": after,
    }


# Thresholds for the hybrid gate: how tied adaptive k's own re-vote still has
# to be before the A/B rule is allowed to overrule it. Distinct from the flip
# margin, which is how far ahead B must be -- conflating the two makes the gate
# either never fire or fire on everything.
HYBRID_GATES = (0.02, 0.05, 0.10, 0.20, 0.35)


def compare_policies(result: dict, reference: dict, k_adaptive: int,
                     distance_power: float, flip_margin: float,
                     rule: str = "restricted",
                     hybrid_gates: tuple[float, ...] = HYBRID_GATES) -> dict:
    """Does the best tiebreak fix the same tiles adaptive k does?

    They score the same on production -- both 97.23% -- which has two very
    different explanations. Either they are one mechanism wearing two hats, in
    which case this line of work is finished, or they fix different tiles and
    the union is worth more than either. The 2x2 settles it, and the oracle
    union bounds anything built out of the pair.
    """
    band = result["band"]
    codes = reference["codes"]
    n_clusters = len(reference["categories"])

    adaptive = adaptive_choice(band, codes, n_clusters, k_adaptive, distance_power)
    # Exactly the inputs compare() gave the rule, so this reproduces its row in
    # the table rather than a near-miss of it. k_base comes from the band, not a
    # literal: the base vote's neighbours are its first k_base columns.
    k_base = band["k_base"]
    dist = np.sqrt(np.maximum(band["dist_sq"], 0.0))
    tiebreak_chosen = _pick(
        rule, flip_margin=flip_margin,
        labels_base=band["labels_all"][:, :k_base], dist_base=dist[:, :k_base],
        labels_all=band["labels_all"], dist_all=dist,
        winner=band["winner"], runner=band["runner"],
        queries=band["queries"], centroids=band["centroids"],
        distance_power=distance_power)

    a = _band_score(band, adaptive["chosen"])
    t = _band_score(band, tiebreak_chosen)
    before = band["winner"] == band["truth"]

    # An A/B rule can only ever reach a tile whose truth is one of the two it is
    # choosing between. Adaptive k has no such limit -- it votes over all 71 --
    # so the two have different ceilings as well as different answers.
    reach_ab = (band["truth"] == band["winner"]) | (band["truth"] == band["runner"])

    either = a["correct"] | t["correct"]
    both = a["correct"] & t["correct"]
    return {
        "adaptive": a, "tiebreak": t,
        "k_adaptive": k_adaptive, "flip_margin": flip_margin, "rule": rule,
        "n_band": len(band["truth"]),
        "before_correct": int(before.sum()),
        # Fixes each finds that the other does not.
        "only_adaptive": int((a["correct"] & ~t["correct"]).sum()),
        "only_tiebreak": int((t["correct"] & ~a["correct"]).sum()),
        "both_correct": int(both.sum()),
        "neither": int((~either).sum()),
        # The most any combination of exactly these two could reach.
        "oracle_union": (band["untouched_correct"] + int(either.sum())) / band["n_total"],
        "ab_reachable": int(reach_ab.sum()),
        "agree": int((adaptive["chosen"] == tiebreak_chosen).sum()),
        # The obvious hybrid: take adaptive k's answer, but let the A/B rule
        # overrule it where adaptive's own re-vote came out a near-tie too. The
        # gate is swept because there is no principled value for it, and best-of-
        # a-few is reported as exactly that.
        "hybrid_sweep": {
            float(gate): _band_score(band, np.where(
                adaptive["margin"] < gate, tiebreak_chosen, adaptive["chosen"]))
            for gate in hybrid_gates},
    }


def best_hybrid(cmp: dict) -> tuple[float, dict]:
    return max(cmp["hybrid_sweep"].items(), key=lambda kv: kv[1]["net"])


def report_policies(cmp: dict, overall_before: float) -> None:
    print(f"\n=== {cmp['rule']} (flip {cmp['flip_margin']:g}) vs adaptive "
          f"k={cmp['k_adaptive']}: the same tiles, or different ones? ===")
    a, t = cmp["adaptive"], cmp["tiebreak"]
    print(f"  {'policy':<22}  {'changed':>7}  {'fixed':>6}  {'broke':>6}  "
          f"{'net':>6}  {'overall':>8}")
    gate, hybrid = best_hybrid(cmp)
    for name, row in (("adaptive k", a), (cmp["rule"], t),
                      (f"hybrid (gate {gate:g})", hybrid)):
        print(f"  {name:<22}  {row['changed']:>7,}  {row['fixed']:>6,}  "
              f"{row['broke']:>6,}  {row['net']:>+6,}  "
              f"{row['overall'] * 100:>7.2f}% "
              f"({(row['overall'] - overall_before) * 100:+.2f})")
    print(f"\n  Hybrid by gate — how tied adaptive's own re-vote must still be "
          f"for the rule to overrule it:")
    print("    " + "".join(f"{g:>9.2f}" for g in sorted(cmp["hybrid_sweep"])))
    print("    " + "".join(f"{cmp['hybrid_sweep'][g]['net']:>+9,}"
                           for g in sorted(cmp["hybrid_sweep"])))
    print(f"    Best of {len(cmp['hybrid_sweep'])} gates, so biased upward — "
          f"confirm on another --seed.")

    n = cmp["n_band"]
    print(f"\n  Of the {n:,} tiles in the band, the two agree on "
          f"{cmp['agree']:,} ({cmp['agree'] / n * 100:.1f}%).")
    print(f"  Right after adaptive only : {cmp['only_adaptive']:,}")
    print(f"  Right after tiebreak only : {cmp['only_tiebreak']:,}")
    print(f"  Right after both          : {cmp['both_correct']:,}")
    print(f"  Right after neither       : {cmp['neither']:,}")
    print(f"\n  An oracle picking whichever of the two is right would reach "
          f"{cmp['oracle_union'] * 100:.2f}% "
          f"({(cmp['oracle_union'] - overall_before) * 100:+.2f}) — the ceiling on "
          f"anything built from this pair.")

    exclusive = cmp["only_adaptive"] + cmp["only_tiebreak"]
    best = max(a["net"], t["net"], hybrid["net"])
    headroom = cmp["oracle_union"] - max(a["overall"], t["overall"],
                                         hybrid["overall"])
    if exclusive < 0.15 * max(cmp["both_correct"], 1):
        print(f"\n  They overlap almost entirely ({exclusive:,} tiles decided "
              f"differently against {cmp['both_correct']:,} shared), so these are "
              f"one mechanism: re-decide the near-ties with more neighbours. "
              f"Combining them cannot help, and the remaining error needs "
              f"different evidence, not another way of counting the same "
              f"neighbours.")
    else:
        print(f"\n  {exclusive:,} tiles are decided differently, and the oracle "
              f"union leaves {headroom * 100:+.2f} over the best single policy "
              f"({best:+,} net). Worth building a selector — but only if some "
              f"computable signal separates them; the oracle is not one.")
    print(f"\n  Note the A/B rule can only reach {cmp['ab_reachable']:,} of "
          f"{n:,} band tiles by construction, while adaptive k votes over every "
          f"cluster and has no such ceiling.")


def disagreement_profile(result: dict, reference: dict, k_adaptive: int,
                         distance_power: float, flip_margin: float,
                         rule: str = "restricted") -> dict:
    """Where the two policies disagree, is there a signal saying which to trust?

    The oracle union is only reachable if something computable separates them.
    Two structurally different kinds of disagreement, which have to be counted
    apart or they average each other out:

      out-of-band   adaptive k picked a cluster that was neither the base
                    winner nor the base runner-up. The A/B rule could not have
                    chosen it at all, so this measures what the A/B restriction
                    costs -- or what adaptive k's extra freedom costs.
      same-pair     both were choosing between the same A and B and picked
                    differently. This is a straight contest between two
                    decision rules on identical options.

    Also profiled: within same-pair, whether the strength of either rule's own
    evidence predicts who is right. If it does, that is the selector. If both
    split near 50/50 in every stratum, the oracle union is not reachable and
    the honest move is to take the hybrid and stop.
    """
    band = result["band"]
    codes = reference["codes"]
    n_clusters = len(reference["categories"])
    k_base = band["k_base"]
    dist = np.sqrt(np.maximum(band["dist_sq"], 0.0))

    adaptive = adaptive_choice(band, codes, n_clusters, k_adaptive, distance_power)
    score_a, score_b = _scores(
        rule, labels_base=band["labels_all"][:, :k_base],
        dist_base=dist[:, :k_base], labels_all=band["labels_all"],
        dist_all=dist, winner=band["winner"], runner=band["runner"],
        queries=band["queries"], centroids=band["centroids"],
        distance_power=distance_power)
    advantage = _advantage(score_a, score_b)
    tiebreak = np.where(advantage > flip_margin, band["runner"], band["winner"])

    truth = band["truth"]
    disagree = adaptive["chosen"] != tiebreak
    in_pair = ((adaptive["chosen"] == band["winner"])
               | (adaptive["chosen"] == band["runner"]))

    def _split(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        if not n:
            return {"n": 0}
        a_right = int((adaptive["chosen"][mask] == truth[mask]).sum())
        t_right = int((tiebreak[mask] == truth[mask]).sum())
        return {"n": n, "adaptive_right": a_right, "tiebreak_right": t_right,
                "neither": n - a_right - t_right,
                "adaptive_share": a_right / max(a_right + t_right, 1)}

    out_of_pair = disagree & ~in_pair
    same_pair = disagree & in_pair

    # Does either rule's own confidence predict who wins the same-pair contest?
    # Stratified on both, because a selector only exists if some stratum is
    # lopsided -- a uniform 50/50 means the disagreement is irreducible.
    strata = {}
    if same_pair.any():
        for name, signal in (("adaptive margin", adaptive["margin"]),
                             ("rule advantage", np.abs(advantage))):
            edges = np.quantile(signal[same_pair], [0.0, 0.25, 0.5, 0.75, 1.0])
            rows = []
            for lo, hi, last in zip(edges[:-1], edges[1:],
                                    [False, False, False, True]):
                band_mask = same_pair & (signal >= lo) & (
                    (signal <= hi) if last else (signal < hi))
                row = _split(band_mask)
                row.update({"lo": float(lo), "hi": float(hi)})
                rows.append(row)
            strata[name] = rows

    return {
        "rule": rule, "flip_margin": flip_margin, "k_adaptive": k_adaptive,
        "n_band": len(truth), "n_disagree": int(disagree.sum()),
        "out_of_pair": _split(out_of_pair),
        "same_pair": _split(same_pair),
        "strata": strata,
        # How often adaptive k leaves the base top two at all, agreement aside.
        "left_the_pair": int((~in_pair).sum()),
        "left_and_right": int(((~in_pair) & (adaptive["chosen"] == truth)).sum()),
    }


def report_disagreement(prof: dict) -> None:
    print(f"\n=== Where they disagree, can anything tell us which to trust? ===")
    print(f"  {prof['n_disagree']:,} of {prof['n_band']:,} band tiles are decided "
          f"differently. Adaptive k left the base top-2 on "
          f"{prof['left_the_pair']:,} tiles and was right on "
          f"{prof['left_and_right']:,} of them "
          f"({prof['left_and_right'] / max(prof['left_the_pair'], 1) * 100:.1f}%).")

    print(f"\n  {'kind':<26}  {'tiles':>7}  {'adaptive':>9}  {'rule':>7}  "
          f"{'neither':>8}  {'adaptive wins':>14}")
    for label, key in (("out-of-pair (rule can't)", "out_of_pair"),
                       ("same-pair (head to head)", "same_pair")):
        row = prof[key]
        if not row["n"]:
            print(f"  {label:<26}  {0:>7}")
            continue
        print(f"  {label:<26}  {row['n']:>7,}  {row['adaptive_right']:>9,}  "
              f"{row['tiebreak_right']:>7,}  {row['neither']:>8,}  "
              f"{row['adaptive_share'] * 100:>13.1f}%")

    for name, rows in prof["strata"].items():
        print(f"\n  Same-pair disagreements by {name} (quartiles) — a selector "
              f"exists only if these are lopsided:")
        print(f"    {'range':>16}  {'tiles':>6}  {'adaptive':>9}  {'rule':>6}  "
              f"{'adaptive wins':>14}")
        for row in rows:
            if not row["n"]:
                continue
            print(f"    {row['lo']:>7.3f}-{row['hi']:<8.3f}  {row['n']:>6,}  "
                  f"{row['adaptive_right']:>9,}  {row['tiebreak_right']:>6,}  "
                  f"{row['adaptive_share'] * 100:>13.1f}%")
        _report_stratum_verdict(rows)


def _report_stratum_verdict(rows: list[dict]) -> None:
    """Is the trend across strata real, and is it usable?

    Spread alone is not enough. Each quartile here holds a few hundred decisive
    tiles, so a share of ~0.4 carries a standard error of several points and a
    15-point spread across four bins can be pure wobble — which is exactly what
    the adaptive-margin cut looks like while the rule-advantage cut is a clean
    monotone slide. A selector needs the trend to be BOTH bigger than its noise
    and ordered, because a non-monotone lurch gives nothing to threshold on.
    """
    usable = [r for r in rows if r.get("n") and
              (r["adaptive_right"] + r["tiebreak_right"]) > 0]
    if len(usable) < 2:
        return
    shares = [r["adaptive_share"] for r in usable]
    decisive = [r["adaptive_right"] + r["tiebreak_right"] for r in usable]
    lo, hi = min(range(len(shares)), key=shares.__getitem__), \
        max(range(len(shares)), key=shares.__getitem__)
    spread = shares[hi] - shares[lo]
    # Binomial standard error on the difference of two independent shares.
    se = sum(shares[i] * (1 - shares[i]) / max(decisive[i], 1) for i in (lo, hi)) ** 0.5
    sigma = spread / se if se else 0.0
    monotone = (shares == sorted(shares)) or (shares == sorted(shares, reverse=True))

    if sigma < 2.0:
        verdict = "within noise — no signal here"
    elif not monotone:
        verdict = ("real but not ordered, so there is no threshold to set — "
                   "look for a different cut")
    else:
        verdict = "monotone and beyond noise — this is a selector"
    print(f"    spread {spread * 100:.1f} points, {sigma:.1f} sigma, "
          f"{'monotone' if monotone else 'non-monotone'} — {verdict}.")


def report(result: dict, margin_threshold: float, k_base: int, k_expand: int) -> None:
    if not result["rules"]:
        print(f"\nNo tiles below margin {margin_threshold} — nothing to break ties on.")
        return

    n_low, n_total = result["n_low"], result["baseline"]["n"]
    print(f"\nLow-margin subset: {n_low:,} of {n_total:,} tiles "
          f"({n_low / n_total * 100:.1f}%) below margin {margin_threshold}, "
          f"{result['subset_before'] * 100:.1f}% correct at k={k_base}.")
    print(f"Of those, {result['reachable']:,} have the true cluster as winner or "
          f"runner-up — the most any A/B rule can get right "
          f"({result['reachable'] / n_low * 100:.1f}% of the subset).")
    print(f"Overall now {result['overall_before'] * 100:.2f}%; a perfect tiebreak "
          f"would give {result['ceiling'] * 100:.2f}%.")

    print(f"\nOverriding on any lead at all (flip margin 0):")
    print(f"  {'rule':<11}  {'changed':>7}  {'fixed':>6}  {'broke':>6}  {'net':>5}  "
          f"{'subset':>7}  {'overall':>8}")
    ordered = sorted(result["rules"].items(), key=lambda kv: -kv[1]["net"])
    for rule, row in ordered:
        delta = row["overall"] - result["overall_before"]
        print(f"  {rule:<11}  {row['changed']:>7,}  {row['fixed']:>6,}  "
              f"{row['broke']:>6,}  {row['net']:>+5,}  "
              f"{row['subset_after'] * 100:>6.1f}%  "
              f"{row['overall'] * 100:>7.2f}% ({delta * 100:+.2f})")

    # Net against how much of a lead the runner-up had to show. A rule whose
    # net climbs as the threshold rises was overriding on noise; one whose net
    # only falls was already only acting on real evidence.
    margins = result["flip_margins"]
    print(f"\nNet by flip margin — how far ahead the runner-up must be to override:")
    print("  " + f"{'rule':<11}" + "".join(f"{m:>8.2f}" for m in margins))
    for rule, _ in ordered:
        cells = "".join(f"{result['sweep'][rule][m]['net']:>+8,}" for m in margins)
        print(f"  {rule:<11}{cells}")

    flat = [(rule, margin, row)
            for rule, by_margin in result["sweep"].items()
            for margin, row in by_margin.items()]
    best_rule, best_margin, best_row = max(flat, key=lambda t: t[2]["net"])
    if best_row["net"] <= 0:
        print(f"\nNo rule wins at any flip margin: the best ({best_rule} at "
              f"{best_margin:g}) is {best_row['net']:+} net. These tiles are genuine "
              f"near-ties and the runner-up is not systematically the better answer — "
              f"a tiebreak is the wrong lever here.")
        return

    delta = best_row["overall"] - result["overall_before"]
    print(f"\nBest: {best_rule} at flip margin {best_margin:g} — {best_row['net']:+} net "
          f"({best_row['fixed']:,} fixed, {best_row['broke']:,} broken of "
          f"{best_row['changed']:,} changed), overall {best_row['overall'] * 100:.2f}% "
          f"({delta * 100:+.2f}).")
    # Net is a difference of paired counts, so the noise on it is set by how
    # many tiles disagreed, not by the sample size. Roughly sqrt(fixed+broke).
    discordant = best_row["fixed"] + best_row["broke"]
    noise = (discordant ** 0.5) if discordant else 0.0
    print(f"  {discordant:,} tiles changed verdict, so a rule doing nothing but "
          f"coin-flipping would land within about +/-{2 * noise:.0f} net. This is "
          f"{best_row['net'] / noise:.1f} sigma from that." if noise else "")
    print(f"  Confirm on a different --seed before believing it.")
    print(f"\nrestricted re-queried at k={k_expand}; every rule touched only the "
          f"{n_low:,} low-margin tiles, so the other "
          f"{n_total - n_low:,} answers are unchanged by construction.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", type=Path, default=HPC_REFERENCE_PATH,
                        help=f"Reference .npz. Defaults to the production one "
                             f"({HPC_REFERENCE_PATH}). Anything else is reported "
                             f"as not-production before any number is printed.")
    parser.add_argument("--sample", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-base", type=int, default=10,
                        help="k for the baseline vote, i.e. the current setting.")
    parser.add_argument("--k-expand", type=int, default=50,
                        help="k for the 'restricted' rule's second search. Larger "
                             "than adaptive k's would normally be, because only the "
                             "two candidate clusters are counted, so the extra "
                             "neighbours cannot add noise from a third.")
    parser.add_argument("--distance-power", type=float, default=2.0)
    parser.add_argument("--margin-threshold", type=float, default=0.25,
                        help="Only tiles below this vote_margin are re-decided.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--neighbours-cache", type=Path, default=None,
                        help="Reuse (or write) a neighbour matrix here — the same "
                             "file tune_classifier.py caches. The search is the "
                             "whole cost against the production reference, and "
                             "this experiment otherwise pays it twice. Must have "
                             "been built at k_max >= --k-expand for the same "
                             "--sample and --seed; a mismatch re-searches rather "
                             "than serving the wrong tiles.")
    parser.add_argument("--with-adaptive", type=int, default=0, metavar="K",
                        help="Also evaluate the shipped adaptive-k policy at this "
                             "k on the same band, and cross-tabulate it against "
                             "the best tiebreak rule. Free — same neighbours, no "
                             "extra search. Use it when a rule scores about the "
                             "same as adaptive k, to find out whether they are "
                             "fixing the same tiles or different ones.")
    parser.add_argument("--flip-margins", type=float, nargs="+", default=list(FLIP_MARGINS),
                        help="How far ahead the runner-up must be before the vote is "
                             "overridden. All are evaluated in one run — the scores are "
                             "computed once, so each extra threshold is free.")
    args = parser.parse_args()

    reference = load_reference(args.reference)
    describe_reference(args.reference, reference)
    result = compare(reference, args.sample, args.seed, args.k_base, args.k_expand,
                     args.distance_power, args.margin_threshold, args.batch_size,
                     flip_margins=tuple(args.flip_margins),
                     cache=args.neighbours_cache)
    # The cheap idea first: is any margin band wrong often enough that taking the
    # runner-up outright would pay? Answered before the rules, because if it did
    # pay there would be no need for a rule.
    baseline = result["baseline"]
    report_blind_swap(baseline, blind_swap_bands(baseline))
    report(result, args.margin_threshold, args.k_base, args.k_expand)

    if args.with_adaptive and result["rules"]:
        # Cross-tabulate against whichever rule/flip-margin actually won, not a
        # rule named up front: the point is to interrogate the best result.
        best_rule, best_margin, _ = max(
            ((rule, margin, row)
             for rule, by_margin in result["sweep"].items()
             for margin, row in by_margin.items()),
            key=lambda t: t[2]["net"])
        cmp = compare_policies(result, reference, args.with_adaptive,
                               args.distance_power, best_margin, best_rule)
        report_policies(cmp, result["overall_before"])
        # Only worth profiling if the two are actually complementary; if they
        # overlap there is nothing for a selector to select between.
        report_disagreement(disagreement_profile(
            result, reference, args.with_adaptive, args.distance_power,
            best_margin, best_rule))


if __name__ == "__main__":
    raise SystemExit(main())
