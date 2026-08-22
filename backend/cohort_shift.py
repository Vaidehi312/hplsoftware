#!/usr/bin/env python3
"""Is a new cohort's tissue actually represented in the reference?

The pipeline's accuracy has only ever been measured within LATTICeA — the cohort
the 71 HPCs were defined on. A slide-level holdout showed a genuinely unseen
LATTICeA slide costs nothing (97.33% against 97.27%), so the classifier
generalises fine to new *slides*. What is unmeasured, and unmeasurable without
labels, is a new *cohort*: a different scanner, stain protocol and microns-per-
pixel.

That gap cannot be closed by accuracy, because a new cohort has no ground truth.
It can be probed by distance, and `migrate_tile_registry_confidence.sql` said so
when the column was added:

    hpc_neighbor_distance ... is also the cheapest available probe for cohort
    shift: if a new cohort's tiles sit systematically further from the reference
    than TCGA's did, that is a batch effect visible as a distance before it turns
    into a cluster-proportion difference that looks biological.

That last clause is the whole point. k-NN assigns every tile to its nearest
cluster however far away that is, and reports nothing. So a shifted cohort still
produces a complete assignments CSV, per-slide proportions that differ from
TCGA's, and a survival analysis that reads those differences as biology. Nothing
crashes and no completeness check fires — the failure mode this codebase is
written against.

Two numbers per tile, both already in the assignments CSV, and they fail
independently:

    neighbor_distance   how far the nearest reference tissue is. NOVELTY. High
                        means nothing in the reference looks like this tile,
                        whatever cluster it was assigned to.
    vote_margin         how close the runner-up was. AMBIGUITY. Low means the
                        tile sits between two clusters.

A shifted cohort shows up in the first. A cohort that is represented but sits on
cluster boundaries shows up in the second. Both are worth knowing and they are
not the same problem.

Comparison is against a profile of the reference's OWN distribution, built by
validate_reference.py --save-profile. A profile is pinned to one reference AND
one vote configuration, because the distance depends on k and the margin depends
on every knob — comparing across configurations would manufacture a shift out of
a settings difference.

Usage:
    # once per (reference, vote) pair
    python validate_reference.py --reference <ref>.npz --sample 200000 \\
      --k 10 --distance-weighted --distance-power 3 \\
      --adaptive-margin 0.15 --adaptive-k 25 \\
      --save-profile reference_profile_tuned.json

    # then per dataset, free
    python cohort_shift.py --assignments <dataset>_hpc_assignments.csv \\
      --profile reference_profile_tuned.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Quantiles stored in a profile. Dense in the tail because that is where novelty
# lives: the median distance barely moves under a shift while the 99th percentile
# moves a lot.
PROFILE_LEVELS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999)

# A tile is "beyond the reference's envelope" above this reference percentile. By
# construction 1% of reference tiles are, so the expected count for an unshifted
# cohort is 1% and the reported figure is a ratio against that — which makes the
# number mean the same thing whatever the reference's absolute distances are.
ENVELOPE_LEVEL = 0.99

# Two signals, deliberately, because they are scaled very differently.
#
# novelty_ratio counts tiles past the reference's 99th percentile, so it is
# sensitive but explosive: a shift of one standard deviation in distance sends it
# past 5x while the cohort is still largely inside the reference's range, and a
# handful of bad slides send it past 20x while 70% of the cohort is fine.
#
# median_percentile asks where the cohort's TYPICAL tile sits in the reference
# distribution. It is bounded by construction, so it separates "mostly fine with
# a tail" from "the whole cohort has moved" — which are different problems with
# different actions.
#
# So the cohort-level verdict keys on the median and the ratio only raises a
# notice. Neither is a decision threshold: a genuinely different tissue
# composition moves both without anything being technically wrong, and a real
# batch effect can hide under both.
_NOTICE_RATIO = 2.0
_NOTICE_PERCENTILE = 65.0
_ALARM_PERCENTILE = 85.0
# Above this spread in per-slide "beyond the envelope" share, the problem is
# some slides rather than the cohort, and the action is to look at them.
_SLIDE_CONCENTRATION = 0.10


def default_profile_path(reference: Path, preset: str) -> Path:
    """Where the profile for (this reference, this vote preset) lives.

    Keyed on the preset as well as the reference, because a profile built under
    the legacy unweighted vote is not a valid baseline for a tuned assignment:
    the distances come from a different k and the margins from a different
    weighting, so comparing across them would manufacture a shift out of a
    settings difference. Separate files make picking the wrong one take a
    deliberate act.
    """
    return reference.with_name(f"{reference.stem}_profile_{preset}.json")


def build_profile(distances: np.ndarray, margins: np.ndarray,
                 codes: np.ndarray, categories, *, reference: str,
                 reference_rows: int, groupby: str, vote: str,
                 seed: int, levels=PROFILE_LEVELS) -> dict:
    """The reference's own distance and margin distribution, as quantiles.

    Quantiles rather than the raw arrays: 200,000 float32s is 800 KB per column
    and nothing downstream needs more resolution than this. Cluster proportions
    go in too — not as a shift detector, since a new cohort's composition is
    legitimately different, but because seeing the two side by side is how anyone
    judges whether a proportion difference is plausible.
    """
    finite_distance = distances[np.isfinite(distances)]
    proportions = np.bincount(codes, minlength=len(categories)) / max(len(codes), 1)
    return {
        "reference": reference,
        "reference_rows": int(reference_rows),
        "groupby": groupby,
        # The configuration these numbers describe. A profile is meaningless
        # without it: the same reference under a different k gives different
        # distances, and under a different weighting different margins.
        "vote": vote,
        "n_sampled": int(len(distances)),
        "seed": int(seed),
        "levels": [float(x) for x in levels],
        "neighbor_distance": [float(x) for x in
                              np.quantile(finite_distance, levels)],
        "vote_margin": [float(x) for x in np.quantile(margins, levels)],
        "low_margin_share": {
            "0.10": float((margins < 0.10).mean()),
            "0.25": float((margins < 0.25).mean()),
        },
        "cluster_proportion": {str(categories[i]): float(p)
                               for i, p in enumerate(proportions)},
    }


def save_profile(profile: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, sort_keys=True))


def load_profile(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(
            f"No profile at {path}. Build one with validate_reference.py "
            f"--save-profile, at the same vote settings the assignment used."
        )
    profile = json.loads(path.read_text())
    missing = [key for key in ("levels", "neighbor_distance", "vote_margin",
                               "vote", "reference")
               if key not in profile]
    if missing:
        raise SystemExit(f"{path} is missing {missing} — not a reference profile.")
    return profile


def _interp_percentile(profile_quantiles, levels, value: float) -> float:
    """Where `value` sits in the reference distribution, as a percentile.

    Linear interpolation between stored quantiles, clamped at both ends. Beyond
    the top stored quantile it returns that level rather than extrapolating: the
    honest answer for a tile past the reference's 99.9th percentile is "at least
    99.9th", not a fabricated 99.9997th.
    """
    q = np.asarray(profile_quantiles, dtype=float)
    lv = np.asarray(levels, dtype=float)
    return float(np.interp(value, q, lv, left=lv[0], right=lv[-1]))


def compare(frame: pd.DataFrame, profile: dict) -> dict:
    """The new cohort against the reference's own distribution."""
    for column in ("neighbor_distance", "vote_margin"):
        if column not in frame.columns:
            raise SystemExit(
                f"The assignments CSV has no '{column}' column, so there is "
                f"nothing to compare. It is written by assign_hpc_clusters.py; "
                f"found {list(frame.columns)}."
            )

    levels = profile["levels"]
    ref_distance = profile["neighbor_distance"]
    envelope = float(np.interp(ENVELOPE_LEVEL, levels, ref_distance))

    distance = frame["neighbor_distance"].to_numpy(dtype=float)
    margin = frame["vote_margin"].to_numpy(dtype=float)
    finite = np.isfinite(distance)

    beyond = float((distance[finite] > envelope).mean()) if finite.any() else 0.0
    expected = 1.0 - ENVELOPE_LEVEL
    result = {
        "n": int(len(frame)),
        "n_finite_distance": int(finite.sum()),
        "envelope": envelope,
        "envelope_level": ENVELOPE_LEVEL,
        "beyond_envelope": beyond,
        "expected_beyond": expected,
        # The headline. 1.0 means the cohort's tail matches the reference's; 10
        # means ten times as many tiles sit outside it as should.
        "novelty_ratio": beyond / expected if expected else float("nan"),
        "levels": levels,
        "reference_distance": ref_distance,
        "cohort_distance": [float(x) for x in
                            np.quantile(distance[finite], levels)]
        if finite.any() else [],
        "reference_margin": profile["vote_margin"],
        "cohort_margin": [float(x) for x in np.quantile(margin, levels)],
        "reference_low_margin": profile.get("low_margin_share", {}),
        "cohort_low_margin": {
            "0.10": float((margin < 0.10).mean()),
            "0.25": float((margin < 0.25).mean()),
        },
        # Where the cohort's median tile sits in the reference's distribution. 50
        # means the two distributions are centred together; 90 means the typical
        # new tile is further from the reference than 90% of reference tiles.
        "median_percentile": _interp_percentile(
            ref_distance, levels, float(np.median(distance[finite]))
        ) * 100 if finite.any() else float("nan"),
        "vote": profile.get("vote"),
        "reference": profile.get("reference"),
    }

    if "slides" in frame.columns and finite.any():
        # Per slide, because a shift is often a handful of slides — a bad stain
        # batch, one scanner — rather than the cohort. Removing those is a
        # different action from rejecting the cohort, and only this distinguishes
        # them.
        per_slide = []
        slides = frame.loc[finite, "slides"].to_numpy()
        slide_distance = distance[finite]
        slide_margin = margin[finite]
        for slide in pd.unique(slides):
            mask = slides == slide
            per_slide.append({
                "slide": str(slide),
                "n": int(mask.sum()),
                "beyond": float((slide_distance[mask] > envelope).mean()),
                "median_distance": float(np.median(slide_distance[mask])),
                "low_margin": float((slide_margin[mask] < 0.10).mean()),
            })
        per_slide.sort(key=lambda row: -row["beyond"])
        result["per_slide"] = per_slide
    return result


def slide_concentration(result: dict) -> float | None:
    """Spread in per-slide "beyond the envelope" share, or None without slides.

    Distinguishes a cohort that has moved from a cohort containing some bad
    slides. Those need different actions -- extend the reference, versus drop or
    re-scan a few slides -- and the ratio alone cannot tell them apart.
    """
    rows = result.get("per_slide")
    if not rows or len(rows) < 2:
        return None
    shares = [row["beyond"] for row in rows]
    return max(shares) - min(shares)


def verdict(result: dict) -> tuple[str, str]:
    """(level, sentence). level is one of consistent / notice / alarm."""
    ratio = result["novelty_ratio"]
    median = result["median_percentile"]
    spread = slide_concentration(result)
    concentrated = spread is not None and spread > _SLIDE_CONCENTRATION

    where = (
        f"The cohort's median tile sits at the reference's {median:.0f}th "
        f"percentile, and {ratio:.1f}x as many tiles as expected fall outside "
        f"its {result['envelope_level'] * 100:.0f}th."
    )
    localise = (
        " The per-slide spread is wide, so this is some slides rather than the "
        "cohort — look at the worst ones before concluding anything about the "
        "cohort, and consider dropping or re-scanning them."
        if concentrated else
        " The per-slide spread is narrow, so it is a property of the cohort and "
        "not of a few slides."
    ) if spread is not None else ""

    if not np.isfinite(median):
        return "consistent", "No finite distances to compare."

    if median >= _ALARM_PERCENTILE:
        return "alarm", (
            f"{where}{localise} The typical tile of this cohort is further from "
            f"the reference than most reference tiles are from each other, so "
            f"its assignments are extrapolation. Every tile still got a cluster "
            f"— k-NN always returns its nearest, however far — and the per-slide "
            f"proportions will differ in ways that look biological and are not. "
            f"Do not read the HPC proportions as biology until this is resolved."
        )
    if ratio >= _NOTICE_RATIO or median >= _NOTICE_PERCENTILE:
        return "notice", (
            f"{where}{localise} The cohort is largely inside the reference's "
            f"range but has a heavier tail than it should. Worth understanding "
            f"before per-slide proportions are quoted; it can equally be a "
            f"genuinely different tissue composition, which is a finding rather "
            f"than a fault."
        )
    return "consistent", (
        f"{where} That is what an unshifted cohort looks like. Nothing here "
        f"suggests the reference fails to represent this tissue — which is not "
        f"the same as the accuracy being equal, only that the cheap probe finds "
        f"nothing."
    )


def report(result: dict, top_slides: int = 10) -> None:
    print(f"Reference profile: {result.get('reference')}")
    print(f"  vote            {result.get('vote')}")
    print(f"Cohort           : {result['n']:,} tiles")
    print()
    print(f"Novelty — how far the cohort sits from the reference")
    print(f"  reference {result['envelope_level'] * 100:.0f}th percentile "
          f"distance   {result['envelope']:.4f}")
    print(f"  cohort tiles beyond it                "
          f"{result['beyond_envelope'] * 100:6.2f}%  "
          f"(expected {result['expected_beyond'] * 100:.0f}%)")
    print(f"  ratio                                 "
          f"{result['novelty_ratio']:6.1f}x")
    print(f"  cohort's median tile sits at the reference's "
          f"{result['median_percentile']:.0f}th percentile")

    print(f"\nDistance by quantile")
    print(f"  {'quantile':>9}  {'reference':>10}  {'cohort':>10}  {'ratio':>7}")
    for level, ref, got in zip(result["levels"], result["reference_distance"],
                               result["cohort_distance"]):
        ratio = got / ref if ref else float("nan")
        print(f"  {level * 100:>8.1f}%  {ref:>10.4f}  {got:>10.4f}  {ratio:>6.2f}x")

    print(f"\nAmbiguity — low-margin share, reference vs cohort")
    for key in ("0.10", "0.25"):
        ref = result["reference_low_margin"].get(key)
        got = result["cohort_low_margin"][key]
        if ref is None:
            print(f"  margin < {key}   cohort {got * 100:5.1f}%  "
                  f"(profile has no reference figure)")
            continue
        print(f"  margin < {key}   reference {ref * 100:5.1f}%   "
              f"cohort {got * 100:5.1f}%   "
              f"{got / ref if ref else float('nan'):.1f}x")
    print("  A high low-margin share with normal distances is a different "
          "problem from novelty: the tissue IS represented, it just sits on "
          "cluster boundaries. Stage 5's --min-margin is the lever for that; "
          "novelty has no lever short of extending the reference.")

    if result.get("per_slide"):
        rows = result["per_slide"]
        print(f"\nWorst {min(top_slides, len(rows))} slides by share beyond the "
              f"envelope (of {len(rows)}):")
        print(f"  {'beyond':>7}  {'median d':>9}  {'low margin':>10}  "
              f"{'tiles':>7}  slide")
        for row in rows[:top_slides]:
            print(f"  {row['beyond'] * 100:>6.1f}%  {row['median_distance']:>9.4f}  "
                  f"{row['low_margin'] * 100:>9.1f}%  {row['n']:>7,}  "
                  f"{row['slide']}")
        shares = [row["beyond"] for row in rows]
        if len(shares) > 1:
            spread = max(shares) - min(shares)
            print(f"\n  spread across slides {spread * 100:.1f} points — "
                  + ("concentrated in some slides, so look at those rather than "
                     "the cohort" if spread > 0.10 else
                     "evenly spread, so this is a property of the cohort and not "
                     "of a few slides"))

    level, sentence = verdict(result)
    print(f"\nVerdict: {level.upper()}")
    print(f"  {sentence}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--assignments", required=True, type=Path,
                        help="Assignments CSV from assign_hpc_clusters.py.")
    parser.add_argument("--profile", required=True, type=Path,
                        help="Reference profile from validate_reference.py "
                             "--save-profile, built at the SAME vote settings.")
    parser.add_argument("--top-slides", type=int, default=10)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    frame = pd.read_csv(args.assignments)
    result = compare(frame, profile)
    report(result, args.top_slides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
