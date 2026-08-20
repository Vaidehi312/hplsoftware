#!/usr/bin/env python3
"""Group HPCs into the superclusters the downstream analysis actually uses.

A 71-way accuracy figure treats every confusion as equally costly. The survival
work does not: libraries/supercluster_dictionary.py collapses HPCs into four
immune/architecture classes, and hpc_annotations.py takes its majority vote at
that level. So an error between two HPCs in the SAME supercluster changes no
downstream number at all.

That matters for how much of the remaining error is worth chasing. On the
production reference the weakest cluster is 69, and its main confusion partner
is 67 — both "Hot, cohesive". Every one of those errors is invisible to the
analysis it feeds.

The mapping is imported from Kai's file rather than restated here, so it cannot
drift. Note it is deliberately partial: it names the ~25 HPCs the survival work
uses and returns "Unknown" for the rest, which is why coverage is reported
alongside any accuracy computed from it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UPSTREAM = Path(__file__).resolve().parents[1] / "HPL-LATTICeA" / "libraries"

UNKNOWN = "Unknown"


def load_mapping(version: str = "v1") -> dict[str, str] | None:
    """{"69": "Hot, cohesive", ...} from the upstream dictionary, or None if the
    subtree is not present (it is a git subtree, not a dependency)."""
    if not (_UPSTREAM / "supercluster_dictionary.py").is_file():
        return None
    if str(_UPSTREAM) not in sys.path:
        sys.path.insert(0, str(_UPSTREAM))
    try:
        import supercluster_dictionary as upstream
    except ImportError:
        return None
    assign = (upstream.assign_supercluster if version == "v1"
              else upstream.assign_supercluster_v2)
    # The functions take "HPC 69"; probe a generous range and keep what lands.
    mapping = {}
    for hpc in range(200):
        group = assign(f"HPC {hpc}")
        if group != UNKNOWN:
            mapping[str(hpc)] = group
    return mapping


def supercluster_codes(categories, mapping: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    """Per-cluster supercluster code, and the group names.

    Unmapped clusters get -1 rather than a shared "Unknown" code. Pooling them
    would be worse than dropping them: two unmapped HPCs confused for each other
    would count as a supercluster-level success purely because neither is
    mapped, which is the opposite of what this measures.
    """
    groups = sorted({g for g in mapping.values()})
    index = {g: i for i, g in enumerate(groups)}
    codes = np.full(len(categories), -1, dtype=np.int64)
    for position, name in enumerate(categories):
        group = mapping.get(str(name))
        if group is not None:
            codes[position] = index[group]
    return codes, groups


def summarise(truth: np.ndarray, predicted: np.ndarray, categories,
              mapping: dict[str, str]) -> dict:
    """How much of the error survives collapsing HPCs to superclusters."""
    codes, groups = supercluster_codes(categories, mapping)
    mapped = codes[truth] >= 0
    wrong = truth != predicted

    # Errors where BOTH sides are mapped are the only ones this can classify.
    classifiable = wrong & mapped & (codes[predicted] >= 0)
    within = classifiable & (codes[truth] == codes[predicted])

    mapped_total = int(mapped.sum())
    mapped_wrong = int((wrong & mapped).sum())
    mapped_wrong_across = int((wrong & mapped & ~(codes[truth] == codes[predicted])).sum())

    return {
        "groups": groups,
        "n_mapped_clusters": int((codes >= 0).sum()),
        "n_clusters": len(categories),
        "tiles_mapped": mapped_total,
        "tiles_total": len(truth),
        "hpc_accuracy_mapped": (1 - mapped_wrong / mapped_total) if mapped_total else float("nan"),
        "supercluster_accuracy_mapped": (
            (1 - mapped_wrong_across / mapped_total) if mapped_total else float("nan")
        ),
        "errors_mapped": mapped_wrong,
        "errors_within_supercluster": int(within.sum()),
        "errors_classifiable": int(classifiable.sum()),
    }


def report(summary: dict) -> None:
    print(f"\nSupercluster view ({summary['n_mapped_clusters']} of "
          f"{summary['n_clusters']} HPCs are mapped; "
          f"{summary['tiles_mapped']:,} of {summary['tiles_total']:,} tiles):")
    print(f"  groups                    {', '.join(summary['groups'])}")
    print(f"  HPC-level accuracy        "
          f"{summary['hpc_accuracy_mapped'] * 100:.2f}%   (71-way, mapped tiles only)")
    print(f"  supercluster accuracy     "
          f"{summary['supercluster_accuracy_mapped'] * 100:.2f}%   "
          f"(an HPC error inside one group is not an error here)")
    if summary["errors_classifiable"]:
        share = summary["errors_within_supercluster"] / summary["errors_classifiable"]
        print(f"  of {summary['errors_classifiable']:,} errors between two mapped HPCs, "
              f"{summary['errors_within_supercluster']:,} ({share * 100:.0f}%) stay "
              f"inside one supercluster")
    print("  Those cost nothing downstream: hpc_annotations.py takes its "
          "majority at supercluster level, so the analysis cannot see them.")
