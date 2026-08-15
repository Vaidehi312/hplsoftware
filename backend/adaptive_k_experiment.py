"""Does re-querying ambiguous tiles at a larger k actually help?

Your own k-sweep already showed a *global* larger k hurts (k=250 -> 92.41%,
k=10 -> 96.13%), so uniformly raising k for every tile would likely repeat
that regression. The version with an actual mechanism behind it is
margin-gated: leave the confident majority alone (already ~100% correct in
the top margin band) and only re-query the tiles that came back ambiguous at
--k-base, this time at a larger --k-expand.

Two-pass, same rows both times:
  1. leave_one_out() at --k-base across --sample tiles (identical to
     validate_reference.py). vote_margin flags which of those tiles were
     ambiguous.
  2. The tiles below --margin-threshold get re-queried — the *exact* same
     reference rows, via query_index rather than a fresh random sample — at
     --k-expand. Comparing pass 1's accuracy on that subset against pass 2's
     answers the question directly: same tiles, only k differs.

Usage:
    python adaptive_k_experiment.py --reference ref_raw128.npz \
        --sample 20000 --seed 0 --k-base 10 --k-expand 25 \
        --distance-power 2 --margin-threshold 0.25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from validate_reference import load_reference, leave_one_out


def compare(reference: dict, sample: int, seed: int, k_base: int, k_expand: int,
           distance_power: float, margin_threshold: float, batch: int) -> dict:
    baseline = leave_one_out(
        reference, sample, k_base, batch, seed,
        distance_weighted=True, distance_power=distance_power,
    )
    low_mask = baseline["margins"] < margin_threshold
    low_index = baseline["query_index"][low_mask]

    result = {
        "baseline": baseline,
        "low_mask": low_mask,
        "low_index": low_index,
        "before_accuracy": None,
        "adaptive": None,
        "after_accuracy": None,
        "overall_before_accuracy": baseline["accuracy"],
        "overall_after_accuracy": None,
    }
    if not low_mask.any():
        return result

    result["before_accuracy"] = float(baseline["correct"][low_mask].mean())
    # sample/seed are ignored by leave_one_out() when query_index is given —
    # passed through only because the signature requires something there.
    adaptive = leave_one_out(
        reference, sample=len(low_index), k=k_expand, batch=batch, seed=seed,
        distance_weighted=True, distance_power=distance_power,
        query_index=low_index,
    )
    result["adaptive"] = adaptive
    result["after_accuracy"] = float(adaptive["correct"].mean())
    # The confident majority (~= baseline, untouched) plus the re-queried
    # subset (adaptive's own answers) — what the full sample's accuracy would
    # be if this margin-gated rule were the actual assignment policy.
    n_total = baseline["n"]
    n_high_correct = int(baseline["correct"][~low_mask].sum())
    n_low_correct = int(adaptive["correct"].sum())
    result["overall_after_accuracy"] = (n_high_correct + n_low_correct) / n_total
    return result


def report(result: dict, k_base: int, k_expand: int, margin_threshold: float) -> None:
    baseline, low_mask = result["baseline"], result["low_mask"]
    print(f"Baseline  : k={k_base}, {baseline['n']:,} tiles, "
          f"accuracy {baseline['accuracy'] * 100:.2f}%")

    if not low_mask.any():
        print(f"Low margin: no tiles below margin {margin_threshold} — nothing to re-query.")
        return

    print(f"Low margin: {low_mask.sum():,} tiles below margin {margin_threshold} "
          f"({result['before_accuracy'] * 100:.2f}% correct at k={k_base})")

    before, after = result["before_accuracy"], result["after_accuracy"]
    delta = after - before
    print(f"\nSame {low_mask.sum():,} tiles, re-queried at k={k_expand}:")
    print(f"  before (k={k_base})  : {before * 100:.2f}% correct")
    print(f"  after  (k={k_expand}): {after * 100:.2f}% correct")
    print(f"  delta               : {delta * 100:+.2f} points")

    overall_before = result["overall_before_accuracy"]
    overall_after = result["overall_after_accuracy"]
    print(f"\nWhole {baseline['n']:,}-tile sample if this margin-gated rule were "
          f"the actual policy (confident tiles unchanged, only the "
          f"{low_mask.sum():,} ambiguous ones re-queried at k={k_expand}):")
    print(f"  k={k_base} only            : {overall_before * 100:.2f}%")
    print(f"  margin-gated adaptive k  : {overall_after * 100:.2f}%")
    print(f"  delta                    : {(overall_after - overall_before) * 100:+.2f} points")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=20_000,
                        help="Reference tiles in the baseline (--k-base) pass.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-base", type=int, default=10,
                        help="k for the first pass, over --sample tiles.")
    parser.add_argument("--k-expand", type=int, default=25,
                        help="k for the second pass, over only the tiles that "
                             "came back below --margin-threshold.")
    parser.add_argument("--distance-power", type=float, default=2.0,
                        help="Distance-weighting exponent, used in both passes.")
    parser.add_argument("--margin-threshold", type=float, default=0.25,
                        help="Tiles with vote_margin below this get re-queried.")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    reference = load_reference(args.reference)
    result = compare(reference, args.sample, args.seed, args.k_base, args.k_expand,
                     args.distance_power, args.margin_threshold, args.batch_size)
    report(result, args.k_base, args.k_expand, args.margin_threshold)


if __name__ == "__main__":
    main()
