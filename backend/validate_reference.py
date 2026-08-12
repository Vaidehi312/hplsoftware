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

from assign_hpc_clusters import Searcher, vote  # noqa: E402

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
    }


def leave_one_out(reference: dict, sample: int, k: int,
                  batch: int, seed: int) -> dict:
    vectors, codes = reference["vectors"], reference["codes"]
    n_clusters = len(reference["categories"])
    total = len(vectors)

    rng = np.random.default_rng(seed)
    if sample >= total:
        query_index = np.arange(total)
    else:
        query_index = np.sort(rng.choice(total, size=sample, replace=False))

    searcher = Searcher(vectors)
    print(f"Backend   : {searcher.backend}")
    print(f"Reference : {total:,} tiles, {vectors.shape[1]} comps, "
          f"{n_clusters} clusters ({reference['groupby']})")
    print(f"Queries   : {len(query_index):,} sampled, k={k} (self excluded)")

    predicted = np.empty(len(query_index), dtype=np.int64)
    margins = np.empty(len(query_index), dtype=np.float32)
    started = time.perf_counter()

    for start in range(0, len(query_index), batch):
        stop = min(start + batch, len(query_index))
        rows = query_index[start:stop]
        # k+1 so that dropping each tile's self-match still leaves k voters,
        # keeping the margin denominator comparable to a real assignment.
        idx, dist = searcher.search(np.ascontiguousarray(vectors[rows]), k + 1)

        # Mask the self-match rather than assuming it is column 0: with
        # duplicate vectors, ties, or an approximate backend it need not be.
        self_mask = idx == rows[:, None]
        # A row with no self-match (possible under IVF) would otherwise keep an
        # extra neighbour; drop its last column so every row votes on k.
        no_self = ~self_mask.any(axis=1)
        if no_self.any():
            self_mask[no_self, -1] = True
        keep = ~self_mask
        trimmed_idx = idx[keep].reshape(len(rows), k)
        trimmed_dist = dist[keep].reshape(len(rows), k)

        winners, margin, _ = vote(trimmed_idx, trimmed_dist, codes, n_clusters)
        predicted[start:stop] = winners
        margins[start:stop] = margin

    elapsed = time.perf_counter() - started
    truth = codes[query_index]
    correct = predicted == truth

    return {
        "accuracy": float(correct.mean()),
        "n": len(query_index),
        "correct": correct,
        "truth": truth,
        "predicted": predicted,
        "margins": margins,
        "categories": reference["categories"],
        "n_clusters": n_clusters,
        "rate": len(query_index) / max(elapsed, 1e-9),
        "elapsed": elapsed,
    }


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
    truth, correct = result["truth"], result["correct"]
    rows = []
    for code in range(result["n_clusters"]):
        mask = truth == code
        if mask.sum() == 0:
            continue
        rows.append((str(result["categories"][code]), int(mask.sum()),
                     float(correct[mask].mean())))
    rows.sort(key=lambda r: r[2])

    print(f"\nWeakest {min(worst, len(rows))} clusters (recovery of their own tiles):")
    print(f"  {'cluster':>8}  {'n':>7}  {'recovered':>9}")
    for name, count, acc in rows[:worst]:
        print(f"  {name:>8}  {count:>7,}  {acc * 100:>8.1f}%")

    # Does vote_margin mean anything? If low-margin tiles are not wrong more
    # often, the confidence column in every assignment CSV is decoration, and
    # the low-confidence index on tile_registry is pointing at nothing.
    margins = result["margins"]
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
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=_DEFAULT_SAMPLE,
                        help="Reference tiles to test. 0 or more than the reference "
                             "size tests all of them.")
    parser.add_argument("--k", type=int, default=None,
                        help="Neighbours to poll. Defaults to the reference's own "
                             "Leiden n_neighbors, which is what assignment uses.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0,
                        help="Sampling seed, so a number can be reproduced.")
    parser.add_argument("--worst", type=int, default=10,
                        help="How many weakest clusters to list.")
    parser.add_argument("--min-accuracy", type=float, default=None,
                        help="Exit non-zero below this, for use as a gate.")
    args = parser.parse_args()

    reference = load_reference(args.reference)
    k = args.k or reference["n_neighbors"]
    sample = args.sample if args.sample > 0 else len(reference["vectors"])

    result = leave_one_out(reference, sample, k, args.batch_size, args.seed)
    report(result, args.worst)

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
