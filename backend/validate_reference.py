#!/usr/bin/env python3
"""Measure how accurately k-NN recovers the reference's own Leiden labels.

The question this answers: given a tile whose true cluster is known, does the
k-NN vote used by assign_hpc_clusters.py return that cluster? Run against the
reference itself, leave-one-out, it puts a ceiling on what assignment can
achieve on new data — if k=250 cannot recover a reference tile's own label, no
amount of care on the query side will help.

Needs only the .npz from build_hpc_reference.py. No projections file, no
extraction run, no GPU. That is the point: it is the accuracy number available
before committing to anything else.

    sample N reference tiles ──> k+1 nearest neighbours ──> drop self
                                                             │
                            true Leiden label <── compare <──┘ majority vote

What it does and does not cover:

  covers      the neighbour search, the vote, the value of k, and whether the
              clusters are k-NN-separable in this PCA space at all. Also whether
              vote_margin is informative, by checking accuracy against it.
  misses      the projection of raw embeddings into the reference space, and the
              centering choice. Reference vectors are already in PCA space, so
              nothing here exercises project(). For that, run
              assign_hpc_clusters.py --validate-against with Kai's TCGA labels.

Self-matches are excluded deliberately. Every reference tile is its own nearest
neighbour at distance zero, so leaving them in inflates accuracy by roughly 1/k
and hides exactly the boundary cases worth seeing.

Usage:
    python validate_reference.py --reference hpc_reference_leiden_2p5_fold2.npz
    python validate_reference.py --reference ... --sample 50000 --k 250
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from assign_hpc_clusters import Searcher, compute_local_scale, vote  # noqa: E402
from build_hpc_reference import HPC_REFERENCE_PATH  # noqa: E402
from superclusters import load_mapping, report as report_superclusters, summarise  # noqa: E402

# Enough for a tight confidence interval on an accuracy near 0.9 (±0.4% at
# 20k), while staying seconds rather than minutes. The full 2.5M reference
# would be ~17 minutes.
_DEFAULT_SAMPLE = 20_000


def load_reference(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(
            f"No reference at {path}. Build it with build_hpc_reference.py first."
        )
    bundle = np.load(path, allow_pickle=False)
    meta = {}
    try:
        meta = json.loads(str(bundle["meta"]))
    except (ValueError, TypeError, KeyError):
        pass
    return {
        "vectors": bundle["reference"],
        "codes": bundle["codes"].astype(np.int64),
        "categories": bundle["categories"],
        "n_neighbors": int(bundle["n_neighbors"]),
        "groupby": meta.get("groupby", "leiden"),
        # Present only on a slide-holdout split. Carried through so
        # describe_reference cannot call such a reference "production": it has
        # the same groupby and the same 71 clusters, and differs only in having
        # whole slides missing.
        "holdout_slides": meta.get("holdout_slides") or [],
    }


# What the production reference looks like. A mismatch is not an error — an
# exploratory reference is a legitimate thing to measure — but it has to be
# stated, because the numbers are not comparable across references and that is
# very easy to forget once they are written down.
_PRODUCTION_GROUPBY = "leiden_2.5"
_PRODUCTION_CLUSTERS = 71


def describe_reference(path: Path, reference: dict) -> None:
    """Say which reference this is, loudly, before any number is printed.

    Every accuracy figure is relative to one reference. Two are in play: the
    production leiden_2.5 (71 clusters) and an exploratory leiden_5.0 (109
    clusters, a deliberately over-split QC pass), and a run against the second
    reads exactly like a run against the first once the command line has
    scrolled away.

    Both are built from **LATTICeA**, not TCGA — the production reference's
    meta["source"] is LATTICeA_5x_he_complete_surv_sex_filtered_leiden_2p5__
    fold2_subsample.h5ad. This comment used to say the production one was TCGA
    LUAD, which is wrong and cost real time: it makes the leave-one-out figure
    look like a within-TCGA number and makes the TCGA acceptance test look
    tautological, when in fact TCGA tiles are absent from the reference and that
    test is a genuine cross-cohort check. The HPCs were defined on LATTICeA and
    transferred onto TCGA, which is what assign_hpc_clusters.py
    --validate-against reproduces.
    """
    groupby = reference["groupby"]
    n_clusters = len(reference["categories"])
    print(f"Reference file: {path}")
    held = reference.get("holdout_slides") or []
    if held:
        # Checked before the production comparison, because a holdout split
        # passes that comparison: same groupby, same cluster count, just missing
        # whole slides. Silence here is how a deliberately crippled reference
        # gets used for a real assignment.
        print(
            f"                *** SLIDE-HOLDOUT REFERENCE — {len(held)} slide(s) "
            f"removed ***\n"
            f"                Built to measure closed-book accuracy. Cluster IDs "
            f"from it are NOT interchangeable with production ones and it must "
            f"never be used for a real assignment.\n"
            f"                Held out: {', '.join(held[:3])}"
            + (" ..." if len(held) > 3 else ""),
            file=sys.stderr,
        )
        return
    if groupby == _PRODUCTION_GROUPBY and n_clusters == _PRODUCTION_CLUSTERS:
        print(f"                production reference "
              f"({groupby}, {n_clusters} clusters)")
        return
    print(
        f"                *** NOT the production reference ***\n"
        f"                this is {groupby} with {n_clusters} clusters; production "
        f"is {_PRODUCTION_GROUPBY} with {_PRODUCTION_CLUSTERS}.\n"
        f"                Numbers from here are not comparable with production "
        f"ones and must not be quoted as if they were.",
        file=sys.stderr,
    )


def leave_one_out(reference: dict, sample: int, k: int,
                  batch: int, seed: int, distance_weighted: bool = False,
                  distance_power: float = 1.0, class_weighted: bool = False,
                  local_scaling: int = 0,
                  metric: str = "l2", query_index: np.ndarray | None = None,
                  neighbours: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """Leave-one-out over `sample` random reference rows (or exactly
    `query_index`, when given).

    query_index lets a caller re-query a *specific* set of rows — e.g. re-running
    the low-margin tiles from an earlier call at a different k — rather than a
    fresh random sample. `sample` and `seed` are ignored in that case: the
    caller already knows exactly which rows it wants, and re-deriving them from
    a seed would risk silently drifting from the set the caller actually meant.

    neighbours skips the search entirely, taking `(idx, dist)` a caller already
    has — `tune_classifier.gather_neighbours` caches exactly this, and the search
    is the whole cost of a run against the production reference. It must already
    have the self-match removed and be ordered by distance, with at least k
    columns and one row per query_index entry; anything else is refused rather
    than trusted, because neighbours for the wrong rows produce a complete,
    plausible set of accuracies for tiles nobody asked about. Distances are
    squared, as faiss returns them and as vote() expects.
    """
    vectors, codes = reference["vectors"], reference["codes"]
    n_clusters = len(reference["categories"])
    total = len(vectors)

    if query_index is not None:
        query_index = np.asarray(query_index, dtype=np.int64)
        if query_index.size and (query_index.min() < 0 or query_index.max() >= total):
            raise ValueError(
                f"query_index has values outside [0, {total}) — not rows of "
                f"this reference."
            )
    else:
        rng = np.random.default_rng(seed)
        if sample >= total:
            query_index = np.arange(total)
        else:
            query_index = np.sort(rng.choice(total, size=sample, replace=False))

    given_idx = given_dist = None
    if neighbours is not None:
        given_idx, given_dist = neighbours
        if given_idx.shape != given_dist.shape:
            raise ValueError(
                f"neighbours: idx {given_idx.shape} and dist {given_dist.shape} "
                f"disagree.")
        if given_idx.shape[0] != len(query_index):
            raise ValueError(
                f"neighbours has {given_idx.shape[0]} rows for "
                f"{len(query_index)} queries — these are not the same tiles.")
        if given_idx.shape[1] < k:
            raise ValueError(
                f"neighbours has only {given_idx.shape[1]} columns, so it cannot "
                f"answer k={k}. Re-search at k_max >= {k}.")

    searcher = None
    if given_idx is None:
        searcher = Searcher(vectors, metric=metric)
        print(f"Backend   : {searcher.backend}")
    else:
        print(f"Backend   : pre-computed neighbours "
              f"({given_idx.shape[0]:,} x {given_idx.shape[1]}), no search")
    print(f"Reference : {total:,} tiles, {vectors.shape[1]} comps, "
          f"{n_clusters} clusters ({reference['groupby']})")
    vote_desc = "+".join(filter(None, [
        f"distance^{distance_power:g}" if distance_weighted else None,
        "class" if class_weighted else None,
        f"local-scale(r={local_scaling})" if local_scaling else None,
    ])) or "unweighted"
    print(f"Queries   : {len(query_index):,} sampled, k={k} (self excluded), "
          f"{vote_desc} vote")

    # One extra self-search over the whole reference, seconds at this size. Done
    # once here rather than per batch: it is a property of the reference, not of
    # the query, so recomputing it per chunk would be both wasteful and a way for
    # two chunks to disagree.
    local_scale = None
    if local_scaling:
        local_scale = compute_local_scale(vectors, r=local_scaling)
        print(f"Local scale: r={local_scaling}, "
              f"median {np.median(local_scale):.3f}, "
              f"{np.percentile(local_scale, 5):.3f}-{np.percentile(local_scale, 95):.3f} "
              f"(5-95%)")

    class_weights = None
    if class_weighted:
        # Same 1/count correction as assign_hpc_clusters.py --class-weighted:
        # a large cluster's neighbours count for less per neighbour, so it
        # can't win a boundary tile purely by being more numerous nearby.
        class_weights = 1.0 / np.maximum(np.bincount(codes, minlength=n_clusters), 1)

    predicted = np.empty(len(query_index), dtype=np.int64)
    margins = np.empty(len(query_index), dtype=np.float32)
    # How many of the k neighbours actually carry the tile's true label. This
    # is what separates a fixable error from an unfixable one: zero means the
    # true cluster never appeared among the neighbours at all, so no vote rule
    # — weighting, class weights, a bigger k — can recover it. One or more
    # means the right answer was present and lost the vote, which is exactly
    # what those knobs can change.
    truth_neighbours = np.empty(len(query_index), dtype=np.int32)
    # Where the true cluster placed in the vote. 1 = it won. 2 = it was the
    # runner-up, which is the case a tiebreak between the top two could still
    # win back. Anything higher means a tiebreak cannot reach it, and the tile
    # needs more neighbours or a different space instead.
    truth_rank = np.empty(len(query_index), dtype=np.int32)
    # The cluster that came second. A tiebreak rule only ever chooses between
    # this and the winner, so it is the other half of what such a rule needs.
    runner_up = np.empty(len(query_index), dtype=np.int64)
    started = time.perf_counter()

    for start in range(0, len(query_index), batch):
        stop = min(start + batch, len(query_index))
        rows = query_index[start:stop]
        if given_idx is not None:
            # Already self-excluded and ordered, so a prefix is the k nearest.
            trimmed_idx = given_idx[start:stop, :k]
            trimmed_dist = given_dist[start:stop, :k]
        else:
            # k+1 so that dropping each tile's self-match still leaves k voters,
            # keeping the margin denominator comparable to a real assignment.
            idx, dist = searcher.search(np.ascontiguousarray(vectors[rows]), k + 1)

            # Mask the self-match rather than assuming it is column 0: with
            # duplicate vectors or ties it need not be.
            self_mask = idx == rows[:, None]
            # A row with no self-match (possible if k+1 exceeds the reference
            # size) would otherwise keep an extra neighbour; drop its last
            # column so every row votes on k.
            no_self = ~self_mask.any(axis=1)
            if no_self.any():
                self_mask[no_self, -1] = True
            keep = ~self_mask
            trimmed_idx = idx[keep].reshape(len(rows), k)
            trimmed_dist = dist[keep].reshape(len(rows), k)

        winners, margin, _, counts = vote(trimmed_idx, trimmed_dist, codes, n_clusters,
                                          distance_weighted=distance_weighted,
                                          distance_power=distance_power,
                                          class_weights=class_weights,
                                          local_scale=local_scale,
                                          return_counts=True)
        predicted[start:stop] = winners
        margins[start:stop] = margin
        row_truth = codes[rows]
        # faiss pads a short result with -1, and codes[-1] is the *last
        # reference row's* cluster — so counting without masking would credit
        # every padded slot to that one cluster and report tiles as having far
        # more same-label neighbours than exist. vote() guards this with
        # `valid`; this count has to guard it too.
        neighbour_valid = trimmed_idx >= 0
        truth_neighbours[start:stop] = (
            (codes[np.where(neighbour_valid, trimmed_idx, 0)] == row_truth[:, None])
            & neighbour_valid
        ).sum(axis=1)
        # Rank by "how many clusters scored strictly higher", so ties with the
        # true cluster count in its favour rather than against it — otherwise a
        # tile the vote genuinely tied would read as already lost.
        truth_score = np.take_along_axis(counts, row_truth[:, None], 1).ravel()
        truth_rank[start:stop] = (counts > truth_score[:, None]).sum(axis=1) + 1
        runner_up[start:stop] = np.argsort(counts, axis=1)[:, -2] if n_clusters > 1 \
            else winners

    elapsed = time.perf_counter() - started
    truth = codes[query_index]
    correct = predicted == truth

    return {
        "accuracy": float(correct.mean()),
        "n": len(query_index),
        "query_index": query_index,
        "correct": correct,
        "truth": truth,
        "predicted": predicted,
        "margins": margins,
        "truth_neighbours": truth_neighbours,
        "truth_rank": truth_rank,
        "runner_up": runner_up,
        "categories": reference["categories"],
        "n_clusters": n_clusters,
        "rate": len(query_index) / max(elapsed, 1e-9),
        "elapsed": elapsed,
    }


def load_holdout(path: Path, reference: dict) -> dict:
    """The held-out tiles, checked against the reference they were split from.

    A holdout only means anything paired with its own reduced reference. Given
    the full reference instead, every query would find itself at distance zero
    and the result would read as near-perfect accuracy — the most flattering
    possible wrong answer, which is exactly the shape of mistake this pipeline
    produces. So the pairing is verified rather than assumed.
    """
    if not path.is_file():
        raise SystemExit(f"No holdout at {path}. Build it with "
                         f"build_hpc_reference.py --holdout-slides N.")
    bundle = np.load(path, allow_pickle=False)
    meta = {}
    try:
        meta = json.loads(str(bundle["meta"]))
    except (ValueError, TypeError, KeyError):
        pass

    holdout = {
        "queries": bundle["queries"],
        "codes": bundle["codes"].astype(np.int64),
        "slides": bundle["slides"],
        "categories": bundle["categories"],
        "groupby": meta.get("groupby", "leiden"),
        "held_slides": meta.get("held_slides", []),
        "reference_rows": meta.get("reference_rows"),
        "unreachable_clusters": meta.get("unreachable_clusters", []),
    }

    if holdout["groupby"] != reference["groupby"]:
        raise SystemExit(
            f"Holdout labels are {holdout['groupby']} but the reference is "
            f"{reference['groupby']} — different clusterings entirely."
        )
    if len(holdout["categories"]) != len(reference["categories"]):
        raise SystemExit(
            f"Holdout has {len(holdout['categories'])} clusters, reference has "
            f"{len(reference['categories'])}. The codes index the category "
            f"table, so these would compare different clusters under the same "
            f"numbers."
        )
    if not np.array_equal(np.asarray(holdout["categories"], dtype=str),
                          np.asarray(reference["categories"], dtype=str)):
        raise SystemExit(
            "Holdout and reference category tables differ. Cluster codes mean "
            "different things on the two sides."
        )
    expected = holdout["reference_rows"]
    if expected is not None and int(expected) != len(reference["vectors"]):
        raise SystemExit(
            f"This holdout was split from a reference of {int(expected):,} rows "
            f"but the given reference has {len(reference['vectors']):,}. Pair it "
            f"with its own reduced reference — against the full one every query "
            f"finds itself and the accuracy is meaningless."
        )
    if not reference.get("holdout_slides"):
        raise SystemExit(
            "The given reference carries no holdout marker, so it is not the "
            "reduced reference this holdout was split from. Using the full "
            "reference here would score every tile against a copy of itself."
        )
    return holdout


def holdout_accuracy(reference: dict, holdout: dict, k: int, batch: int,
                     distance_weighted: bool = False,
                     distance_power: float = 1.0,
                     class_weighted: bool = False,
                     local_scaling: int = 0,
                     metric: str = "l2",
                     adaptive_margin: float = 0.0,
                     adaptive_k: int = 0) -> dict:
    """Closed-book accuracy: held-out slides against a reference without them.

    No self-match handling, and none needed — that is the whole difference from
    leave_one_out. These tiles are genuinely absent from the reference, so
    nothing has to be masked out and no slide-mate is available to vote for them.

    Returns the same shape leave_one_out does, so report() can print the same
    error budget, margin bands and per-cluster breakdown for both and the two
    numbers can be read side by side.
    """
    vectors, codes = reference["vectors"], reference["codes"]
    n_clusters = len(reference["categories"])
    queries, truth = holdout["queries"], holdout["codes"]

    searcher = Searcher(vectors, metric=metric)
    print(f"Backend   : {searcher.backend}")
    print(f"Reference : {len(vectors):,} tiles, {vectors.shape[1]} comps, "
          f"{n_clusters} clusters ({reference['groupby']})")
    print(f"Holdout   : {len(queries):,} tiles from "
          f"{len(holdout['held_slides'])} slide(s) never seen by the reference, "
          f"k={k}")

    local_scale = None
    if local_scaling:
        local_scale = compute_local_scale(vectors, r=local_scaling)
    class_weights = None
    if class_weighted:
        class_weights = 1.0 / np.maximum(np.bincount(codes, minlength=n_clusters), 1)

    k_search = max(k, adaptive_k) if adaptive_margin > 0 else k
    predicted = np.empty(len(queries), dtype=np.int64)
    margins = np.empty(len(queries), dtype=np.float32)
    truth_neighbours = np.empty(len(queries), dtype=np.int32)
    truth_rank = np.empty(len(queries), dtype=np.int32)
    n_revoted = 0

    started = time.perf_counter()
    for start in range(0, len(queries), batch):
        stop = min(start + batch, len(queries))
        idx, dist = searcher.search(
            np.ascontiguousarray(queries[start:stop]), k_search)
        winners, margin, _, counts = vote(
            idx[:, :k], dist[:, :k], codes, n_clusters,
            distance_weighted=distance_weighted, distance_power=distance_power,
            class_weights=class_weights, local_scale=local_scale,
            return_counts=True)

        if adaptive_margin > 0:
            low = margin < adaptive_margin
            if low.any():
                w2, m2, _, c2 = vote(
                    idx[low, :adaptive_k], dist[low, :adaptive_k], codes,
                    n_clusters, distance_weighted=distance_weighted,
                    distance_power=distance_power, class_weights=class_weights,
                    local_scale=local_scale, return_counts=True)
                winners, margin = winners.copy(), margin.copy()
                winners[low], margin[low] = w2, m2
                counts = counts.copy()
                counts[low] = c2
                n_revoted += int(low.sum())

        predicted[start:stop] = winners
        margins[start:stop] = margin
        row_truth = truth[start:stop]
        # Same -1 padding guard as leave_one_out: codes[-1] is a real cluster,
        # so counting without masking credits every padded slot to it.
        valid = idx[:, :k] >= 0
        truth_neighbours[start:stop] = (
            (codes[np.where(valid, idx[:, :k], 0)] == row_truth[:, None]) & valid
        ).sum(axis=1)
        truth_score = np.take_along_axis(counts, row_truth[:, None], 1).ravel()
        truth_rank[start:stop] = (counts > truth_score[:, None]).sum(axis=1) + 1

    elapsed = time.perf_counter() - started
    correct = predicted == truth
    if n_revoted:
        print(f"Adaptive  : re-voted {n_revoted:,} tiles "
              f"({n_revoted / len(queries) * 100:.1f}%) at k={adaptive_k}")
    return {
        "accuracy": float(correct.mean()),
        "n": len(queries),
        "correct": correct,
        "truth": truth,
        "predicted": predicted,
        "margins": margins,
        "truth_neighbours": truth_neighbours,
        "truth_rank": truth_rank,
        "categories": reference["categories"],
        "n_clusters": n_clusters,
        "rate": len(queries) / max(elapsed, 1e-9),
        "elapsed": elapsed,
        "held_slides": holdout["held_slides"],
        "slides": holdout["slides"],
    }


def report_holdout_extras(result: dict, holdout: dict) -> None:
    """What report() cannot say, because it does not know about slides."""
    slides = np.asarray(result["slides"])
    correct = result["correct"]
    print(f"\nPer held-out slide — a single slide's tiles are highly correlated, "
          f"so the spread across slides is the honest error bar, not the "
          f"binomial one above:")
    rows = []
    for slide in np.unique(slides):
        mask = slides == slide
        rows.append((str(slide), int(mask.sum()), float(correct[mask].mean())))
    rows.sort(key=lambda r: r[2])
    for name, n, acc in rows:
        print(f"    {acc * 100:>6.2f}%  {n:>7,} tiles  {name}")
    accuracies = [r[2] for r in rows]
    if len(accuracies) > 1:
        spread = max(accuracies) - min(accuracies)
        mean = sum(accuracies) / len(accuracies)
        variance = sum((a - mean) ** 2 for a in accuracies) / (len(accuracies) - 1)
        # Standard error over slides, which is the unit that actually varies.
        print(f"\n    unweighted slide mean {mean * 100:.2f}%, "
              f"between-slide SD {variance ** 0.5 * 100:.2f} points, "
              f"range {spread * 100:.1f} points")
        print(f"    Report the tile-weighted accuracy above as the headline, but "
              f"quote this spread with it: {len(accuracies)} slides is a small "
              f"sample and one atypical slide moves the total.")
    if holdout["unreachable_clusters"]:
        print(f"\n    Cluster(s) {holdout['unreachable_clusters']} exist only on "
              f"held-out slides, so no query could be assigned to them and their "
              f"tiles are counted wrong by construction. Rebuild with another "
              f"--holdout-seed to remove that penalty.")


def report(result: dict, worst: int) -> None:
    n, accuracy = result["n"], result["accuracy"]
    # Binomial standard error, so the number is read with the right precision
    # rather than to five decimal places it does not have.
    stderr = (accuracy * (1 - accuracy) / n) ** 0.5
    print(f"\nAccuracy  : {accuracy * 100:.2f}% ± {stderr * 196:.2f} "
          f"(95% CI, n={n:,})")
    print(f"            {int(result['correct'].sum()):,} of {n:,} recovered their "
          f"own Leiden label in {result['elapsed']:.1f}s "
          f"({result['rate']:,.0f} tiles/s)")

    # Per-cluster accuracy. A low overall number caused by two bad clusters is a
    # different problem from one spread evenly, and only this distinguishes them.
    #
    # For each cluster's wrong tiles, also the single most common wrong vote and
    # its share of those errors. Two clusters at 85% recovery are not the same
    # problem if one's errors scatter across a dozen neighbours and the other's
    # all land on one specific cluster — only the second is a merge candidate,
    # and neither guess is distinguishable from overall accuracy alone.
    truth, predicted, correct = result["truth"], result["predicted"], result["correct"]
    margins = result["margins"]
    truth_neighbours = result["truth_neighbours"]
    truth_rank = result["truth_rank"]
    categories, n_clusters = result["categories"], result["n_clusters"]

    # The ceiling. Errors where the true cluster never appeared among the k
    # neighbours cannot be recovered by any vote rule; they need a different
    # embedding space, a bigger k, or a merge. Everything else is a vote that
    # was lost with the right answer present — that is the addressable share,
    # and the honest upper bound on what tuning the vote can still buy.
    wrong_all = ~correct
    n_wrong_all = int(wrong_all.sum())
    if n_wrong_all:
        unreachable_all = int((truth_neighbours[wrong_all] == 0).sum())
        reachable_all = n_wrong_all - unreachable_all
        ceiling = (int(correct.sum()) + reachable_all) / n
        print(f"\nError budget: {n_wrong_all:,} wrong — {reachable_all:,} "
              f"({reachable_all / n_wrong_all * 100:.0f}%) had the true cluster among "
              f"the k neighbours and lost the vote, {unreachable_all:,} "
              f"({unreachable_all / n_wrong_all * 100:.0f}%) did not.")
        # An oracle bound, not a target: a rule that always picked the true
        # cluster when present would be reading the answer. What it says is
        # where the loss is — above this line the neighbours do not carry the
        # label, below it the vote is throwing it away.
        print(f"            An oracle picking the true cluster whenever it was present "
              f"would reach {ceiling * 100:.2f}%. Not achievable — it bounds how much "
              f"is lost in scoring rather than in the search.")

        # Of the errors where the true cluster was present, where did it place?
        # Runner-up is recoverable by a rule that only has to break the top two
        # apart. Rank 5 is not — no tiebreak reaches it, and those tiles need
        # more neighbours or a different space.
        ranks = truth_rank[wrong_all]
        runner_up = int((ranks == 2).sum())
        top3 = int((ranks <= 3).sum())
        print(f"            Of the wrong tiles, {runner_up:,} "
              f"({runner_up / n_wrong_all * 100:.0f}%) had the true cluster as "
              f"runner-up and {top3:,} ({top3 / n_wrong_all * 100:.0f}%) in the top 3 "
              f"(median rank {int(np.median(ranks))}) — the share a tiebreak could reach.")
    rows = []
    top_code_of = {}
    for code in range(n_clusters):
        mask = truth == code
        if mask.sum() == 0:
            continue
        wrong = mask & ~correct
        n_wrong = int(wrong.sum())
        confused_with, confused_share, top_code = None, 0.0, None
        if n_wrong > 0:
            vote_counts = np.bincount(predicted[wrong], minlength=n_clusters)
            top_code = int(np.argmax(vote_counts))
            confused_with = str(categories[top_code])
            confused_share = float(vote_counts[top_code] / n_wrong)
            top_code_of[code] = top_code
        wrong_margin = float(margins[wrong].mean()) if n_wrong > 0 else None
        correct_margin = float(margins[mask & correct].mean()) if (mask & correct).any() else None
        # Of this cluster's errors, how many had *no* same-cluster neighbour at
        # all among the k. Those are beyond any vote rule.
        unreachable = int((truth_neighbours[wrong] == 0).sum()) if n_wrong > 0 else 0
        rows.append((code, str(categories[code]), int(mask.sum()), float(correct[mask].mean()),
                     confused_with, confused_share, n_wrong, top_code, wrong_margin,
                     correct_margin, unreachable))
    rows.sort(key=lambda r: r[3])

    # Is weak recovery just small-reference-count noise, or does it track a
    # cluster's own size independent of that? A cluster with few reference
    # tiles has fewer plausible same-cluster neighbours to vote for it even
    # when the phenotype is perfectly separable, so size alone can explain
    # part of a low number without anything being "wrong" to fix.
    sizes = np.array([r[2] for r in rows], dtype=np.float64)
    accs = np.array([r[3] for r in rows], dtype=np.float64)
    if len(rows) > 2 and sizes.std() > 0 and accs.std() > 0:
        size_acc_corr = float(np.corrcoef(sizes, accs)[0, 1])
        print(f"\nCluster size vs. recovery: r = {size_acc_corr:+.2f} across {len(rows)} "
              f"clusters (near 0 = size isn't the story; negative = smaller clusters "
              f"recover worse).")

    print(f"\nWeakest {min(worst, len(rows))} clusters (recovery of their own tiles):")
    print(f"  {'cluster':>8}  {'n':>7}  {'recovered':>9}  top confusion")
    for (code, name, count, acc, confused_with, confused_share, n_wrong, top_code,
         wrong_margin, correct_margin, unreachable) in rows[:worst]:
        if confused_with is None:
            confusion = "(no errors)"
        else:
            mutual = top_code_of.get(top_code) == code
            confusion = (f"→ {confused_with} ({confused_share * 100:.0f}% of {n_wrong} wrong)"
                         f"{' [mutual]' if mutual else ''}")
        print(f"  {name:>8}  {count:>7,}  {acc * 100:>8.1f}%  {confusion}")
        if n_wrong == 0:
            continue
        # Reachable errors — the true cluster WAS among the k neighbours and
        # still lost the vote — are the ones a vote rule can win back. The rest
        # need a different space or a merge, not a better vote.
        reachable = n_wrong - unreachable
        print(f"           {reachable}/{n_wrong} errors had the true cluster among the "
              f"k neighbours ({'vote is losable — weighting/class weights can help' if reachable else 'true cluster never appears — no vote rule can fix these'})")
        if wrong_margin is None:
            continue
        if correct_margin is None:
            print(f"           wrong-tile margin {wrong_margin:.2f} (no correct tiles to "
                  f"compare against — this cluster is fully absorbed)")
            continue
        # Wrong tiles voted with near the same confidence as correct ones means
        # the model isn't hesitating on them — it's confidently picking the
        # wrong cluster, which margin-gated adaptive k cannot fix (it only
        # re-queries LOW-margin tiles). Wrong tiles with a visibly lower margin
        # than correct ones are boundary cases adaptive k should be catching.
        gap = correct_margin - wrong_margin
        style = "confidently wrong (adaptive-k won't catch this)" if gap < 0.1 \
            else "low-confidence / boundary (adaptive-k should help)"
        print(f"           wrong-tile margin {wrong_margin:.2f} vs. "
              f"correct-tile margin {correct_margin:.2f} — {style}")

    # Where the errors actually live. The weakest-cluster list ranks by *rate*,
    # which over-weights small clusters: a 60%-recovery cluster of 40 tiles is
    # 16 errors, while a 97% cluster of 3,000 is 90. Fixing the second moves the
    # overall number and the first does not, so rank the pairs by error count
    # too — that is the list to work down if the goal is overall accuracy.
    pair_counts = {}
    for t, p in zip(truth[~correct], predicted[~correct]):
        pair_counts[(int(t), int(p))] = pair_counts.get((int(t), int(p)), 0) + 1
    total_wrong = int((~correct).sum())
    if pair_counts:
        top_pairs = sorted(pair_counts.items(), key=lambda kv: -kv[1])[:worst]
        print(f"\nBiggest confusion pairs by error count ({total_wrong:,} errors total):")
        print(f"  {'true':>6} → {'predicted':<9}  {'errors':>7}  {'share':>6}  reverse")
        for (t, p), count in top_pairs:
            reverse = pair_counts.get((p, t), 0)
            print(f"  {str(categories[t]):>6} → {str(categories[p]):<9}  {count:>7,}  "
                  f"{count / total_wrong * 100:>5.1f}%  "
                  f"{reverse:,} the other way"
                  f"{' (symmetric — likely one phenotype split in two)' if reverse >= count * 0.5 else ''}")

    # Does vote_margin mean anything? If low-margin tiles are not wrong more
    # often, the confidence column in every assignment CSV is decoration, and
    # the low-confidence index on tile_registry is pointing at nothing.
    print("\nIs vote_margin informative?")
    edges = [0.0, 0.1, 0.25, 0.5, 0.75, 1.01]
    for low, high in zip(edges, edges[1:]):
        band = (margins >= low) & (margins < high)
        if band.sum() == 0:
            continue
        print(f"  margin {low:>4.2f}-{high if high <= 1 else 1.0:<4.2f}  "
              f"{band.sum():>7,} tiles  {correct[band].mean() * 100:>5.1f}% correct")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", type=Path, default=HPC_REFERENCE_PATH,
                        help=f"Reference .npz. Defaults to the production one "
                             f"({HPC_REFERENCE_PATH}). Anything else is reported "
                             f"as not-production before any number is printed.")
    parser.add_argument("--sample", type=int, default=_DEFAULT_SAMPLE,
                        help="Reference tiles to test. 0 or more than the reference "
                             "size tests all of them.")
    parser.add_argument("--k", type=int, default=None,
                        help="Neighbours to poll. Defaults to the reference's own "
                             "Leiden n_neighbors, which is what assignment uses.")
    parser.add_argument("--metric", default="l2", choices=["l2", "cosine"],
                        help="l2 (default): Euclidean distance. cosine: normalises "
                             "reference and query vectors to unit length first, so "
                             "only direction (not magnitude) decides neighbours. Run "
                             "once with each to see whether it actually helps before "
                             "using it for a real assignment.")
    parser.add_argument("--distance-weighted", action="store_true",
                        help="Weight each neighbour's vote by 1/distance instead of "
                             "one vote each. Run once with and once without to see "
                             "whether it actually helps before using it for a real "
                             "assignment.")
    parser.add_argument("--distance-power", type=float, default=1.0,
                        help="Exponent on the distance-weighted vote: "
                             "1/(distance+eps)**power. 2.0 makes the nearest few "
                             "neighbours matter much more. Only used with "
                             "--distance-weighted.")
    parser.add_argument("--class-weighted", action="store_true",
                        help="Scale each neighbour's vote by 1/(its cluster's "
                             "reference count), so a large cluster's neighbours don't "
                             "win a boundary tile just by being more numerous nearby. "
                             "Composes with --distance-weighted.")
    parser.add_argument("--local-scaling", type=int, default=0, metavar="R",
                        help="Divide each neighbour's distance by that neighbour's "
                             "own distance to its R-th nearest reference point, so "
                             "closeness is judged relative to local density. 0 (the "
                             "default) is off. 7 is a reasonable first try. The "
                             "reference's Leiden clusters were cut from a graph whose "
                             "weights carry exactly this per-point scale, which the "
                             "plain 1/d^p vote does not.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0,
                        help="Sampling seed, so a number can be reproduced.")
    parser.add_argument("--superclusters", action="store_true",
                        help="Also report accuracy after collapsing HPCs into the "
                             "four immune/architecture superclusters the survival "
                             "analysis uses. An HPC error inside one supercluster "
                             "changes nothing downstream, so this says how much of "
                             "the remaining error is worth chasing.")
    parser.add_argument("--worst", type=int, default=10,
                        help="How many weakest clusters to list.")
    parser.add_argument("--min-accuracy", type=float, default=None,
                        help="Exit non-zero below this, for use as a gate.")
    parser.add_argument("--holdout", type=Path, default=None,
                        help="A holdout .npz from build_hpc_reference.py "
                             "--holdout-slides, evaluated against the reduced "
                             "reference it was split from. This is the "
                             "closed-book number: the slides were never in the "
                             "reference, so nothing votes for a tile except "
                             "other slides. Leave-one-out is also reported, on "
                             "the same reduced reference, so the gap between "
                             "them is the size of the slide-mate leakage.")
    parser.add_argument("--adaptive-margin", type=float, default=0.0,
                        metavar="MARGIN",
                        help="Re-vote tiles below this margin at --adaptive-k. "
                             "Holdout evaluation only, so the shipped "
                             "configuration can be measured closed-book.")
    parser.add_argument("--adaptive-k", type=int, default=0, metavar="K",
                        help="The wider neighbourhood for --adaptive-margin.")
    args = parser.parse_args()

    if args.adaptive_margin > 0 and args.adaptive_k <= (args.k or 0):
        raise SystemExit(
            "--adaptive-margin needs --adaptive-k wider than --k, and an "
            "explicit --k to compare it against."
        )

    reference = load_reference(args.reference)
    describe_reference(args.reference, reference)
    k = args.k or reference["n_neighbors"]
    sample = args.sample if args.sample > 0 else len(reference["vectors"])

    vote_kwargs = dict(distance_weighted=args.distance_weighted,
                       distance_power=args.distance_power,
                       class_weighted=args.class_weighted,
                       local_scaling=args.local_scaling,
                       metric=args.metric)

    holdout_result = None
    if args.holdout is not None:
        holdout = load_holdout(args.holdout, reference)
        print("\n" + "=" * 72)
        print("CLOSED BOOK — held-out slides against a reference without them")
        print("=" * 72)
        holdout_result = holdout_accuracy(
            reference, holdout, k, args.batch_size,
            adaptive_margin=args.adaptive_margin, adaptive_k=args.adaptive_k,
            **vote_kwargs)
        report(holdout_result, args.worst)
        report_holdout_extras(holdout_result, holdout)
        print("\n" + "=" * 72)
        print("OPEN BOOK — leave-one-out on the same reduced reference")
        print("=" * 72)

    result = leave_one_out(reference, sample, k, args.batch_size, args.seed,
                           **vote_kwargs)
    report(result, args.worst)

    if holdout_result is not None:
        gap = (result["accuracy"] - holdout_result["accuracy"]) * 100
        print("\n" + "=" * 72)
        print(f"Leave-one-out {result['accuracy'] * 100:.2f}%  vs  "
              f"slide holdout {holdout_result['accuracy'] * 100:.2f}%  "
              f"— gap {gap:+.2f} points")
        print("=" * 72)
        print(
            "The gap IS the slide-mate leakage. Leave-one-out removes one tile "
            "and leaves its slide-mates in the reference, so it measures "
            "recovery from near-duplicates of the tile itself. The holdout "
            "number is the one that predicts a genuinely new slide, and "
            "therefore the one to quote for a new cohort."
        )
        if gap < 0:
            print(
                "  The holdout scored HIGHER, which leave-one-out cannot "
                "normally do. Either the held-out slides are unusually easy "
                "— check the per-slide spread above — or the two runs used "
                "different vote settings."
            )

    if args.superclusters:
        mapping = load_mapping()
        if mapping:
            report_superclusters(summarise(
                result["truth"], result["predicted"], result["categories"], mapping
            ))
        else:
            print("\nNo supercluster mapping available — HPL-LATTICeA/libraries/"
                  "supercluster_dictionary.py is not present in this checkout.",
                  file=sys.stderr)

    print(
        "\nThis measures the search, the vote and whether the clusters are "
        "k-NN-separable in this space. It does NOT cover projecting raw "
        "embeddings into it — for that, run assign_hpc_clusters.py "
        "--validate-against with labels you already trust."
    )

    if args.min_accuracy is not None and result["accuracy"] < args.min_accuracy:
        raise SystemExit(
            f"\nAccuracy {result['accuracy'] * 100:.2f}% is below the required "
            f"{args.min_accuracy * 100:.2f}%."
        )


if __name__ == "__main__":
    main()
