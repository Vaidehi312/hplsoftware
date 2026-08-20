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

from assign_hpc_clusters import Searcher, _WEIGHT_EPS  # noqa: E402
from validate_reference import describe_reference, leave_one_out, load_reference  # noqa: E402
from build_hpc_reference import HPC_REFERENCE_PATH  # noqa: E402

RULES = ("nearest", "mean-dist", "restricted", "centroid")

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
            flip_margins: tuple[float, ...] = FLIP_MARGINS) -> dict:
    """Baseline at k_base, then every rule applied to the low-margin tiles, at
    every flip margin."""
    baseline = leave_one_out(reference, sample, k_base, batch, seed,
                             distance_weighted=True, distance_power=distance_power)

    low_mask = baseline["margins"] < margin_threshold
    if not low_mask.any():
        return {"baseline": baseline, "low_mask": low_mask, "rules": {}}

    vectors, codes = reference["vectors"], reference["codes"]
    rows = baseline["query_index"][low_mask]
    winner = baseline["predicted"][low_mask]
    runner = baseline["runner_up"][low_mask]
    truth = baseline["truth"][low_mask]

    searcher = Searcher(vectors)
    queries = np.ascontiguousarray(vectors[rows])
    # k_expand + 1 for the same reason leave_one_out uses k + 1: every tile is
    # its own nearest neighbour and must be dropped.
    idx, dist = searcher.search(queries, k_expand + 1)

    self_mask = idx == rows[:, None]
    no_self = ~self_mask.any(axis=1)
    if no_self.any():
        self_mask[no_self, -1] = True
    keep = ~self_mask
    idx = idx[keep].reshape(len(rows), k_expand)
    dist = np.sqrt(np.maximum(dist[keep].reshape(len(rows), k_expand), 0.0))

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
        "flip_margins": tuple(float(m) for m in flip_margins),
        "n_low": int(low_mask.sum()), "reachable": int(reachable.sum()),
        "subset_before": float(before_correct.mean()),
        "overall_before": baseline["accuracy"],
        "ceiling": (untouched_correct + int(reachable.sum())) / baseline["n"],
    }


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
    parser.add_argument("--flip-margins", type=float, nargs="+", default=list(FLIP_MARGINS),
                        help="How far ahead the runner-up must be before the vote is "
                             "overridden. All are evaluated in one run — the scores are "
                             "computed once, so each extra threshold is free.")
    args = parser.parse_args()

    reference = load_reference(args.reference)
    describe_reference(args.reference, reference)
    result = compare(reference, args.sample, args.seed, args.k_base, args.k_expand,
                     args.distance_power, args.margin_threshold, args.batch_size,
                     flip_margins=tuple(args.flip_margins))
    report(result, args.margin_threshold, args.k_base, args.k_expand)


if __name__ == "__main__":
    raise SystemExit(main())
