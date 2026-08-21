#!/usr/bin/env python3
"""Sweep every settled classifier knob in one pass, on one search.

Each configuration used to cost its own leave-one-out run — 107s against the
2.5M-tile production reference — so comparing a dozen of them meant half an
hour and a spreadsheet. Almost all of that was the same work repeated: the
faiss search dominates, and it does not depend on the vote at all.

So this searches ONCE at k_max, keeps the neighbours, and re-votes. Every knob
in the sweep is then nearly free:

    k                a prefix slice of the same neighbour matrix
    distance_power   a re-vote over the same distances
    local_scaling    a re-vote with per-neighbour distances rescaled
    adaptive k       a re-vote of the low-margin rows at a wider prefix

Results are compared to the current production setting with McNemar, not with
independent confidence intervals. Every configuration is scored on the SAME
tiles, so the informative quantity is how many tiles changed verdict and which
way — a marginal CI on 20,000 tiles is far too pessimistic and would call a
real gain noise. What matters is fixed vs broke among the discordant pairs.

Usage:
    python tune_classifier.py --sample 20000
    python tune_classifier.py --sample 200000 --k-max 50
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from assign_hpc_clusters import (  # noqa: E402
    Searcher, compute_local_scale_for, vote,
)
from build_hpc_reference import HPC_REFERENCE_PATH  # noqa: E402
from validate_reference import describe_reference, load_reference  # noqa: E402

# The setting everything is measured against: what assignment does today.
BASELINE = {"k": 10, "distance_power": 2.0, "local_scaling": 0,
            "adaptive": 0.0, "class_weighted": False}


# Above this, local scaling is skipped rather than silently turning a
# three-minute sweep into an overnight one. It is a budget, not a limit on what
# is worth doing — the point is that the cost is stated and chosen.
_LOCAL_SCALE_BUDGET_SECONDS = 15 * 60


def _per_query_cost(n_queries: int, seconds: float) -> float:
    return seconds / max(n_queries, 1)


def _scale_for_rows(vectors: np.ndarray, rows: np.ndarray, r: int) -> np.ndarray:
    """Local density scale for just `rows`, returned as a full-length array.

    Rows that never appear as anyone's neighbour keep a scale of 1.0. vote()
    only ever indexes this by a neighbour it actually found, so those entries
    are never read — filling them is cheaper than carrying a mask around.
    """
    scale = np.ones(len(vectors), dtype=np.float32)
    scale[rows] = compute_local_scale_for(vectors, rows, r=r)
    return scale


def gather_neighbours(reference: dict, query_index: np.ndarray, k_max: int,
                      batch: int, cache: Path | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """The k_max nearest OTHER reference rows for each query, once.

    Searching k_max+1 and dropping the self-match means a prefix slice
    [:, :k] is exactly the k nearest non-self neighbours for any k <= k_max —
    which is what makes sweeping k free rather than another search each time.
    """
    # The search is essentially the entire cost of a sweep — 562s for 20,000
    # queries against the 2.5M production reference — and it depends only on
    # (reference, query_index, k_max), never on the vote. Cached, a re-sweep
    # with different knobs costs seconds instead of ten minutes.
    if cache is not None and cache.is_file():
        stored = np.load(cache)
        if (stored["query_index"].shape == query_index.shape
                and np.array_equal(stored["query_index"], query_index)
                and stored["idx"].shape[1] >= k_max):
            print(f"Search    : reusing {cache} "
                  f"({stored['idx'].shape[0]:,} queries x k={stored['idx'].shape[1]})")
            return stored["idx"][:, :k_max], stored["dist"][:, :k_max]
        print(f"Search    : {cache} does not match this sample/k_max — re-searching",
              file=sys.stderr)

    vectors = reference["vectors"]
    searcher = Searcher(vectors)
    idx = np.empty((len(query_index), k_max), dtype=np.int64)
    dist = np.empty((len(query_index), k_max), dtype=np.float32)

    started = time.perf_counter()
    for start in range(0, len(query_index), batch):
        stop = min(start + batch, len(query_index))
        rows = query_index[start:stop]
        found, distance = searcher.search(
            np.ascontiguousarray(vectors[rows]), k_max + 1
        )
        self_mask = found == rows[:, None]
        no_self = ~self_mask.any(axis=1)
        if no_self.any():
            self_mask[no_self, -1] = True
        idx[start:stop] = found[~self_mask].reshape(len(rows), k_max)
        dist[start:stop] = distance[~self_mask].reshape(len(rows), k_max)
    print(f"Search    : {len(query_index):,} queries x k={k_max} in "
          f"{time.perf_counter() - started:.1f}s "
          f"(every configuration below reuses this)")
    if cache is not None:
        np.savez(cache, idx=idx, dist=dist, query_index=query_index)
        print(f"            cached to {cache} — later sweeps skip the search")
    return idx, dist


def score(idx: np.ndarray, dist: np.ndarray, codes: np.ndarray, n_clusters: int,
          truth: np.ndarray, config: dict, local_scales: dict,
          class_weights: np.ndarray | None = None
          ) -> tuple[np.ndarray, np.ndarray]:
    """(correct, margins) for one configuration. No search, no I/O."""
    k = config["k"]
    scale = local_scales.get(config["local_scaling"])
    weights = class_weights if config.get("class_weighted") else None
    winners, margins, _ = vote(
        idx[:, :k], dist[:, :k], codes, n_clusters,
        distance_weighted=True, distance_power=config["distance_power"],
        local_scale=scale, class_weights=weights,
    )

    threshold = config["adaptive"]
    if threshold > 0:
        # Re-vote only the low-margin rows at the wider prefix. Same neighbours,
        # so this is the adaptive-k policy exactly, at no extra search cost.
        low = margins < threshold
        if low.any():
            wide = config["adaptive_k"]
            w2, m2, _ = vote(
                idx[low, :wide], dist[low, :wide], codes, n_clusters,
                distance_weighted=True, distance_power=config["distance_power"],
                local_scale=scale, class_weights=weights,
            )
            winners = winners.copy()
            winners[low] = w2
    return winners == truth, margins


def mcnemar(baseline_correct: np.ndarray, correct: np.ndarray) -> dict:
    """Paired comparison. Concordant tiles carry no information about which
    configuration is better, so they are dropped — which is exactly why a
    marginal confidence interval over all 20,000 is the wrong test here."""
    fixed = int((~baseline_correct & correct).sum())
    broke = int((baseline_correct & ~correct).sum())
    discordant = fixed + broke
    z = (fixed - broke) / np.sqrt(discordant) if discordant else 0.0
    return {"fixed": fixed, "broke": broke, "net": fixed - broke,
            "discordant": discordant, "z": float(z)}


def sweep(reference: dict, sample: int, seed: int, k_max: int, batch: int,
          ks, powers, scalings, adaptives, adaptive_ks,
          class_weightings, cache: Path | None = None
          ) -> tuple[list[dict], float]:
    """(one row per configuration, the baseline's accuracy).

    The baseline comes back alongside the rows because it is not one of them:
    it is scored whether or not the swept grid includes it, and every delta and
    McNemar figure in the rows is relative to it.
    """
    vectors, codes = reference["vectors"], reference["codes"]
    n_clusters = len(reference["categories"])
    total = len(vectors)

    rng = np.random.default_rng(seed)
    query_index = (np.arange(total) if sample >= total
                   else np.sort(rng.choice(total, size=sample, replace=False)))
    truth = codes[query_index]

    print(f"Reference : {total:,} tiles, {vectors.shape[1]} comps, "
          f"{n_clusters} clusters ({reference['groupby']})")
    search_started = time.perf_counter()
    idx, dist = gather_neighbours(reference, query_index, k_max, batch, cache)
    search_seconds = time.perf_counter() - search_started

    local_scales = {0: None}
    wanted = sorted({s for s in scalings if s})
    if wanted:
        # Only the reference rows that actually turn up as somebody's neighbour
        # need a density scale. Computing it for all of them is a self-search
        # over the whole reference — 19.5 hours at the production reference's
        # measured 28 ms/query — for a quantity the vote reads at ~1M positions.
        needed = np.unique(idx[idx >= 0])
        share = len(needed) / len(vectors)
        estimate = len(needed) * (elapsed_per_query := _per_query_cost(
            len(query_index), search_seconds))
        print(f"Local scale: needed for {len(needed):,} of {len(vectors):,} "
              f"reference rows ({share * 100:.1f}%), "
              f"~{estimate / 60:.0f} min at the measured search rate")
        if estimate > _LOCAL_SCALE_BUDGET_SECONDS:
            print(f"            SKIPPING — over the {_LOCAL_SCALE_BUDGET_SECONDS / 60:.0f} "
                  f"minute budget. Raise it with --local-scale-budget, or measure "
                  f"local scaling on its own with a smaller --sample.",
                  file=sys.stderr)
            scalings = [0]
        else:
            for r in wanted:
                started = time.perf_counter()
                local_scales[r] = _scale_for_rows(vectors, needed, r)
                print(f"            r={r} in {time.perf_counter() - started:.1f}s")

    # 1/count per cluster: a large cluster's neighbours count for less each, so
    # it cannot win a boundary tile purely by being more numerous nearby. This
    # was written into both scripts long ago and never once measured against a
    # real reference — the sweep is where that gets settled, since it is a
    # re-vote and costs nothing on top of the search.
    class_weights = 1.0 / np.maximum(np.bincount(codes, minlength=n_clusters), 1)

    configs = []
    for k, power, scaling, adaptive, wide, weighted in itertools.product(
        ks, powers, scalings, adaptives, adaptive_ks, class_weightings
    ):
        if adaptive > 0 and wide <= k:
            continue  # re-querying at a k no wider than the base is a no-op
        if adaptive == 0 and wide != adaptive_ks[0]:
            continue  # adaptive_k means nothing with the gate off; keep one
        configs.append({"k": k, "distance_power": power, "local_scaling": scaling,
                        "adaptive": adaptive, "adaptive_k": wide,
                        "class_weighted": weighted})

    baseline_config = dict(BASELINE, adaptive_k=adaptive_ks[0])
    baseline_correct, _ = score(idx, dist, codes, n_clusters, truth,
                                baseline_config, local_scales, class_weights)
    baseline_accuracy = float(baseline_correct.mean())
    print(f"Baseline  : k={BASELINE['k']}, distance^{BASELINE['distance_power']:g}, "
          f"no local scaling, no adaptive k -> "
          f"{baseline_accuracy * 100:.2f}%")

    results = []
    for config in configs:
        correct, _ = score(idx, dist, codes, n_clusters, truth, config,
                           local_scales, class_weights)
        results.append({**config, "accuracy": float(correct.mean()),
                        **mcnemar(baseline_correct, correct)})
    # Returned rather than looked up again from `results`. The baseline is scored
    # here unconditionally, but it is only *in* results when the swept grid
    # happens to contain it -- so narrowing the sweep to, say, --distance-power 3
    # alone used to kill the run with a bare StopIteration after paying for the
    # whole search. It is measured; it does not need finding.
    return results, baseline_accuracy


def report(results: list[dict], baseline_accuracy: float, top: int) -> None:
    ordered = sorted(results, key=lambda r: -r["accuracy"])
    print(f"\n{'k':>4} {'pow':>4} {'scale':>6} {'adapt':>6} {'wide':>5} {'cls':>4}  "
          f"{'accuracy':>9} {'delta':>7} {'fixed':>6} {'broke':>6} {'net':>6} "
          f"{'sigma':>6}")
    for row in ordered[:top]:
        delta = (row["accuracy"] - baseline_accuracy) * 100
        print(f"{row['k']:>4} {row['distance_power']:>4.0f} "
              f"{row['local_scaling'] or '-':>6} "
              f"{row['adaptive'] or '-':>6} "
              f"{(row['adaptive_k'] if row['adaptive'] else '-'):>5} "
              f"{('y' if row['class_weighted'] else '-'):>4}  "
              f"{row['accuracy'] * 100:>8.2f}% "
              f"{delta:>+7.2f} {row['fixed']:>6,} {row['broke']:>6,} "
              f"{row['net']:>+6,} {row['z']:>6.1f}")

    best = ordered[0]
    if best["net"] <= 0:
        print("\nNothing beats the current setting. That is a real answer: the "
              "knobs swept here are exhausted, and the remaining error needs a "
              "different kind of change, not a better value.")
        return
    parts = [f"k={best['k']}", f"distance^{best['distance_power']:g}"]
    if best["local_scaling"]:
        parts.append(f"local-scale r={best['local_scaling']}")
    if best["adaptive"]:
        parts.append(f"adaptive k={best['adaptive_k']} below margin {best['adaptive']:g}")
    if best["class_weighted"]:
        parts.append("class-weighted")
    print(f"\nBest: {', '.join(parts)}")
    print(f"  {best['fixed']:,} fixed, {best['broke']:,} broken, "
          f"{best['net']:+,} net over {best['discordant']:,} tiles that changed "
          f"verdict — {best['z']:.1f} sigma.")
    if abs(best["z"]) < 2:
        print("  Under 2 sigma: this is not distinguishable from chance. Do not "
              "adopt it on one seed.")
    else:
        print("  Confirm on a second --seed before adopting it: this is the best "
              "of many configurations scored on one sample, so its margin is "
              "biased upward even when the sign is right.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", type=Path, default=HPC_REFERENCE_PATH)
    parser.add_argument("--sample", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-max", type=int, default=50,
                        help="Widest neighbourhood searched. Every k in the sweep "
                             "and the adaptive re-query must fit inside it.")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 15, 25],
                        help="Base neighbourhood sizes to try.")
    parser.add_argument("--distance-power", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument("--local-scaling", type=int, nargs="+", default=[0],
                        help="0 disables it. Others are the r in the r-th "
                             "nearest-neighbour density scale.")
    parser.add_argument("--adaptive", type=float, nargs="+", default=[0.0, 0.10, 0.25],
                        help="vote_margin thresholds below which to re-query wider. "
                             "0 disables it.")
    parser.add_argument("--adaptive-k", type=int, nargs="+", default=[25],
                        help="Neighbourhood(s) the low-margin tiles are re-queried "
                             "at. Free to sweep — each is another prefix of the same "
                             "search.")
    parser.add_argument("--class-weighted", type=int, nargs="+", default=[0, 1],
                        help="Whether to scale each neighbour's vote by 1/(its "
                             "cluster's reference count). 0 off, 1 on, both to "
                             "compare. Never previously measured against a real "
                             "reference.")
    parser.add_argument("--neighbours-cache", type=Path, default=None,
                        help="Save/reuse the neighbour matrix here. The search "
                             "is the whole cost of a sweep and depends only on "
                             "the reference, the sample and --k-max, so a cached "
                             "one makes re-sweeping different knobs instant.")
    parser.add_argument("--local-scale-budget", type=float, default=15.0,
                        metavar="MINUTES",
                        help="Skip local scaling if estimating it would take "
                             "longer than this. It needs a self-search over every "
                             "reference row that appears as a neighbour, which on "
                             "a 2.5M reference is hours, not minutes.")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    if max(args.adaptive_k) > args.k_max:
        raise SystemExit(
            f"--adaptive-k {max(args.adaptive_k)} exceeds --k-max {args.k_max}; the "
            f"re-query would need neighbours the search never fetched."
        )
    if max(args.k) > args.k_max:
        raise SystemExit(f"--k values must all be <= --k-max ({args.k_max}).")

    reference = load_reference(args.reference)
    describe_reference(args.reference, reference)
    sample = args.sample if args.sample > 0 else len(reference["vectors"])

    global _LOCAL_SCALE_BUDGET_SECONDS
    _LOCAL_SCALE_BUDGET_SECONDS = args.local_scale_budget * 60
    results, baseline = sweep(
        reference, sample, args.seed, args.k_max, args.batch_size,
        args.k, args.distance_power, args.local_scaling,
        args.adaptive, args.adaptive_k,
        [bool(v) for v in args.class_weighted], args.neighbours_cache)
    report(results, baseline, args.top)


if __name__ == "__main__":
    raise SystemExit(main())
