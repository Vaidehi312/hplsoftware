#!/usr/bin/env python3
"""Why a cluster recovers badly — structure, not another accuracy number.

validate_reference.py answers "which clusters are weak and what do they get
confused with". This answers the next question: *why*, in terms that imply an
action.

The starting point is a measurement that rules most explanations out. On
ref_raw128, every single error had the true cluster among the k neighbours and
90% had it as the runner-up. So weakness is not "the reference has never seen
this morphology" and not "k is too small" — the right answer was retrieved and
then lost. What loses it is how the cluster sits relative to its neighbours in
the embedding, and that is what is measured here.

Six numbers per cluster, chosen so they separate causes that need OPPOSITE
responses:

  purity        mean share of a tile's k neighbours carrying its own label.
                Local agreement, independent of who won any vote.
  core          share of tiles whose neighbourhood is mostly its own cluster.
                A cluster with a core has an unambiguous centre and a fuzzy
                edge; one without is fuzzy all the way through — which is a
                different problem with a different fix.
  radius        median distance from the cluster's tiles to its medoid.
                Medoid, not centroid: a Leiden cluster is a graph community and
                need not be convex, so its mean can land somewhere no tile is.
  separation    distance from the medoid to the nearest other cluster's medoid,
                over its own radius. Below 1 means the nearest other cluster is
                closer than the cluster's own typical member.
  partner       which other cluster supplies the most foreign neighbours.
                Geometric — computed over ALL tiles, not just wrong ones, so it
                shows the overlap whether or not it has cost a vote yet.
  mutual        whether that partner points back. Mutual overlap is two labels
                on one region; one-way is a cluster sitting inside another.

Those combine into a named diagnosis, because the action differs:

  duplicate   high mutual overlap, low separation. One morphology split in two.
              -> merging them is a labelling decision, not a classifier fix.
  engulfed    one-way overlap into a much larger partner, separation < 1.
              -> the small cluster lives inside the big one's territory. Only
                 more discriminative features can fix this; no vote rule will.
  diffuse     low core with no dominant partner. No centre to speak of.
              -> the cluster may not be one thing. Consider splitting it.
  boundary    healthy core, purity lost only at the edge.
              -> normal. This is what a good cluster with neighbours looks like,
                 and the tiles it loses are genuinely ambiguous.

Usage:
    python cluster_diagnostics.py --reference ref_raw128.npz
    python cluster_diagnostics.py --reference ... --worst 15 --k 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from assign_hpc_clusters import Searcher  # noqa: E402
from validate_reference import describe_reference, load_reference  # noqa: E402
from build_hpc_reference import HPC_REFERENCE_PATH  # noqa: E402

# A tile is "core" when this share of its neighbours carry its own label. 0.8 is
# a deliberate round number; the metric is used comparatively between clusters,
# not against an absolute standard.
_CORE_THRESHOLD = 0.8


# Words that carry no morphology, so two descriptions differing only by them
# are the same description. Deliberately short — the point is to catch
# "stroma with pigment" vs "pigment and stroma", not to do NLP.
_FILLER = {"and", "with", "of", "the", "a", "mostly", "areas", "some"}


def _normalise_description(text: str) -> frozenset:
    """A description reduced to its content words, order-insensitive.

    Byte comparison misses the cases that matter most. In the real review sheet
    cluster 57 is "stroma with pigment" and 89 is "pigment and stroma" — the
    same morphology written two ways, and by construction one of the strongest
    duplicate signals available.
    """
    words = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower()).split()
    return frozenset(w for w in words if w not in _FILLER)


def _same_description(descriptions: dict, code: int, partner) -> bool:
    if partner is None or code not in descriptions or partner not in descriptions:
        return False
    a, b = _normalise_description(descriptions[code]), _normalise_description(descriptions[partner])
    if not a or not b:
        return False
    return a == b


def load_descriptions(path: Path) -> tuple[dict, set]:
    """Kai's cluster review sheet: cluster, description, remove.

    Optional, and the reason it matters is that the leiden_5.0 clustering this
    reference is built on was never a taxonomy. Per HPL-LATTICeA/README.md's
    "Background and artefact removal" section, it exists to find junk tiles to
    delete; the biology clustering runs afterwards at leiden 2.5 on the filtered
    data. So duplicate descriptions are expected here by construction, and a
    "confusion" between two clusters describing the same thing is not an error.
    """
    import csv
    descriptions, removed = {}, set()
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                code = int(row["cluster"])
            except (KeyError, ValueError, TypeError):
                continue
            descriptions[code] = (row.get("description") or "").strip()
            if (row.get("remove") or "").strip():
                removed.add(code)
    return descriptions, removed


def medoids(vectors: np.ndarray, codes: np.ndarray, n_clusters: int,
            rng: np.random.Generator, sample_per_cluster: int = 400) -> np.ndarray:
    """The most central actual member of each cluster.

    A Leiden cluster is a graph community, so it need not be convex or even
    contiguous in the embedding — its centroid can land in empty space, or
    inside a different cluster entirely, which would make every distance
    computed from it meaningless. A medoid is always a real tile.

    Computed on a per-cluster subsample: exact medoid is O(n^2) per cluster and
    the ranking this feeds does not need that precision.
    """
    result = np.zeros((n_clusters, vectors.shape[1]), dtype=np.float32)
    for code in range(n_clusters):
        members = np.flatnonzero(codes == code)
        if members.size == 0:
            continue
        if members.size > sample_per_cluster:
            members = rng.choice(members, size=sample_per_cluster, replace=False)
        block = vectors[members].astype(np.float32)
        # Sum of distances to every other sampled member; smallest wins.
        sq = ((block[:, None, :] - block[None, :, :]) ** 2).sum(axis=2)
        result[code] = block[np.argsort(sq.sum(axis=1))[0]]
    return result


def analyse(reference: dict, k: int, sample: int, seed: int,
            batch: int = 4096, descriptions: dict | None = None,
            removed: set | None = None) -> list[dict]:
    """Per-cluster structure. No labels are predicted here — this deliberately
    does not re-run the vote, so the numbers cannot be a restatement of
    accuracy."""
    vectors, codes = reference["vectors"], reference["codes"]
    categories = reference["categories"]
    n_clusters = len(categories)
    total = len(vectors)
    rng = np.random.default_rng(seed)

    query_index = (np.arange(total) if sample >= total
                   else np.sort(rng.choice(total, size=sample, replace=False)))

    searcher = Searcher(vectors)
    print(f"Reference : {total:,} tiles, {vectors.shape[1]} comps, "
          f"{n_clusters} clusters ({reference['groupby']})")
    print(f"Sampling  : {len(query_index):,} tiles, k={k} (self excluded)")

    own_share = np.empty(len(query_index), dtype=np.float32)
    # foreign[c, d] = how many of cluster c's neighbours belong to cluster d.
    foreign = np.zeros((n_clusters, n_clusters), dtype=np.int64)

    started = time.perf_counter()
    for start in range(0, len(query_index), batch):
        stop = min(start + batch, len(query_index))
        rows = query_index[start:stop]
        idx, _ = searcher.search(np.ascontiguousarray(vectors[rows]), k + 1)

        self_mask = idx == rows[:, None]
        no_self = ~self_mask.any(axis=1)
        if no_self.any():
            self_mask[no_self, -1] = True
        trimmed = idx[~self_mask].reshape(len(rows), k)

        valid = trimmed >= 0
        labels = np.where(valid, codes[np.where(valid, trimmed, 0)], -1)
        truth = codes[rows]
        own = (labels == truth[:, None]) & valid
        own_share[start:stop] = own.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)

        # Count who the foreign neighbours belong to, per source cluster.
        others = valid & ~own
        np.add.at(foreign, (np.repeat(truth, k)[others.ravel()],
                            labels.ravel()[others.ravel()]), 1)

    print(f"Neighbourhoods scanned in {time.perf_counter() - started:.1f}s")
    centres = medoids(vectors, codes, n_clusters, rng)

    # Medoid-to-medoid distances, for separation.
    between = np.sqrt(((centres[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(between, np.inf)

    truth_all = codes[query_index]
    rows_out = []
    for code in range(n_clusters):
        mask = truth_all == code
        if not mask.any():
            continue
        member_rows = query_index[mask]
        radius = float(np.median(
            np.sqrt(((vectors[member_rows] - centres[code]) ** 2).sum(axis=1))
        ))
        nearest = int(np.argmin(between[code]))
        nearest_distance = float(between[code, nearest])

        counts = foreign[code].copy()
        counts[code] = 0
        total_foreign = counts.sum()
        partner = int(np.argmax(counts)) if total_foreign else None
        partner_share = float(counts[partner] / total_foreign) if partner is not None and total_foreign else 0.0
        # How far ahead of the NEXT overlapping cluster the partner is. Raw
        # share cannot be compared across references: with 109 clusters the
        # foreign neighbours spread thin and 17% is enormous, while with 3
        # clusters 17% is nothing. A ratio against the runner-up is on the same
        # scale either way, which is what makes one threshold work for both.
        ordered = np.sort(counts)[::-1]
        second = float(ordered[1] / total_foreign) if total_foreign and len(ordered) > 1 else 0.0
        dominance = partner_share / max(second, 1e-9) if partner_share else 0.0

        rows_out.append({
            "code": code,
            "name": str(categories[code]),
            "n": int((codes == code).sum()),
            "purity": float(own_share[mask].mean()),
            "core": float((own_share[mask] >= _CORE_THRESHOLD).mean()),
            "radius": radius,
            "separation": nearest_distance / radius if radius > 0 else np.inf,
            "nearest": int(nearest),
            "partner": partner,
            "partner_share": partner_share,
            "partner_second_share": second,
            "dominance": dominance,
        })

    # Mutuality needs every cluster's partner known first.
    partner_of = {r["code"]: r["partner"] for r in rows_out}
    sizes = {r["code"]: r["n"] for r in rows_out}
    # Calibrated against this reference rather than a constant: "weak core" only
    # means anything relative to what the other clusters here manage.
    core_floor = float(np.percentile([r["core"] for r in rows_out], 10))
    for row in rows_out:
        row["mutual"] = (row["partner"] is not None
                         and partner_of.get(row["partner"]) == row["code"])
        row["partner_ratio"] = (sizes.get(row["partner"], 0) / max(row["n"], 1)
                                if row["partner"] is not None else 0.0)
        row["same_description"] = _same_description(
            descriptions, row["code"], row["partner"]
        ) if descriptions else False
        row["description"] = (descriptions or {}).get(row["code"], "")
        row["partner_description"] = (descriptions or {}).get(row["partner"], "")
        row["diagnosis"], row["action"] = _diagnose(row, core_floor)
    return rows_out


def _diagnose(row: dict, core_floor: float) -> tuple[str, str]:
    """Name the cause, and what to do about it.

    Thresholds are on scale-free quantities, because absolute ones do not
    survive a change in cluster count. `dominance` is the partner's share of
    foreign neighbours over the runner-up's, and `separation` is already a
    ratio; `core_floor` is a percentile of this reference's own core
    distribution rather than a constant. An earlier version used raw shares and
    classified all 109 clusters identically — with 109 clusters the foreign
    neighbours spread so thin that no absolute share threshold could ever fire.

    Ordered most-specific first. Each class implies a different response, so a
    cluster in the wrong one sends someone off to do the wrong thing.
    """
    if row.get("same_description"):
        return ("duplicate",
                "its overlap partner carries the same morphology description — "
                "these are one cluster split in two, and the 'errors' between "
                "them are not errors")
    if row["mutual"] and row["dominance"] >= 1.6 and row["separation"] < 1.15:
        return ("duplicate",
                "two clusters sitting on one region and each other's main "
                "overlap — merging is a labelling call, not a classifier fix")
    if (not row["mutual"] and row["separation"] < 1.0
            and row["partner_ratio"] > 2.0 and row["dominance"] >= 1.6):
        return ("engulfed",
                "sits inside a much larger cluster's territory — no vote rule "
                "recovers this, it needs more discriminative features")
    # Both conditions, and an absolute floor as well as a relative one. A
    # reference whose clusters are all clean has a 10th percentile core of 1.0,
    # and "core <= that" is then true of every cluster including the perfect
    # ones — which is the opposite of diffuse. A cluster with no foreign
    # neighbours at all is isolated, not formless.
    if (row["core"] < 0.5 and row["core"] <= core_floor
            and row["dominance"] < 1.4 and row["partner"] is not None):
        return ("diffuse",
                "weak core and no single overlapping neighbour — may not be one "
                "thing; consider splitting")
    return ("boundary",
            "healthy core, purity lost only at the edge — normal, and the "
            "tiles it loses are genuinely ambiguous")


def report(rows: list[dict], worst: int, removed: set | None = None) -> None:
    removed = removed or set()
    rows = sorted(rows, key=lambda r: r["purity"])
    has_descriptions = any(r.get("description") for r in rows)

    print(f"\nWeakest {min(worst, len(rows))} clusters by neighbourhood purity:")
    print(f"  {'cluster':>8} {'n':>7} {'purity':>7} {'core':>6} {'sep':>6} "
          f"{'partner':>8} {'dom':>5}  diagnosis")
    for row in rows[:worst]:
        partner = ("-" if row["partner"] is None
                   else f"{row['partner']}{'*' if row['mutual'] else ''}")
        junk = " [remove]" if row["code"] in removed else ""
        print(f"  {row['name']:>8} {row['n']:>7,} {row['purity'] * 100:>6.1f}% "
              f"{row['core'] * 100:>5.0f}% {row['separation']:>6.2f} "
              f"{partner:>8} {row['dominance']:>5.1f}  {row['diagnosis']}{junk}")
        if has_descriptions and row.get("description"):
            partner_junk = " [remove]" if row["partner"] in removed else ""
            same = "  <-- same morphology" if row.get("same_description") else ""
            print(f"           {row['description']!r} -> "
                  f"{row['partner_description']!r}{partner_junk}{same}")
    print("  (* = the partner points back too; dom = partner's share of foreign "
          "neighbours over the runner-up's)")

    print("\nWhat each diagnosis means:")
    seen = {}
    for row in rows[:worst]:
        seen.setdefault(row["diagnosis"], row["action"])
    for diagnosis, action in seen.items():
        print(f"  {diagnosis:<10} {action}")

    counts = {}
    for row in rows:
        counts[row["diagnosis"]] = counts.get(row["diagnosis"], 0) + 1
    print("\nAcross all clusters: " + ", ".join(
        f"{n} {d}" for d, n in sorted(counts.items(), key=lambda kv: -kv[1])))

    if removed:
        present = sorted(r["code"] for r in rows if r["code"] in removed)
        print(f"\n{len(present)} of these clusters are flagged remove=1 in the review "
              f"sheet: {present}")
        print("  Those are background / edge / ink / out-of-focus / artefact classes. "
              "This reference still contains them, so Stage 4 can assign a production "
              "tile to one. That is worth acting on regardless of accuracy.")

    print(
        "\nNote on why this is not 100%. Every error measured had the true cluster "
        "among its neighbours and 90% had it as runner-up, so nothing here is a "
        "retrieval failure. A Leiden cluster is a cut through a continuous density, "
        "and a tile on that cut has no unambiguous label — reproducing it exactly "
        "would mean reproducing the clustering's own arbitrary boundary decisions. "
        "'boundary' clusters are working as intended; 'duplicate' and 'engulfed' are "
        "where the labelling, not the classifier, is what limits accuracy."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", type=Path, default=HPC_REFERENCE_PATH,
                        help=f"Reference .npz. Defaults to the production one "
                             f"({HPC_REFERENCE_PATH}). Anything else is reported "
                             f"as not-production before any number is printed.")
    parser.add_argument("--k", type=int, default=10,
                        help="Neighbourhood size, matching what assignment uses.")
    parser.add_argument("--sample", type=int, default=40_000,
                        help="Tiles to scan. 0 or more than the reference size "
                             "scans all of them.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worst", type=int, default=15)
    parser.add_argument("--descriptions", type=Path, default=None,
                        help="Kai's cluster review CSV (cluster, description, "
                             "remove) for this resolution and fold, e.g. "
                             "231212_DGX10e_removal_5p0_f0.csv. With it, a cluster "
                             "whose overlap partner carries the same morphology is "
                             "named as a duplicate outright — which is the single "
                             "most useful signal here, since the leiden_5.0 "
                             "clustering was a QC pass, not a taxonomy.")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    reference = load_reference(args.reference)
    describe_reference(args.reference, reference)
    sample = args.sample if args.sample > 0 else len(reference["vectors"])
    descriptions, removed = ({}, set())
    if args.descriptions:
        if not args.descriptions.is_file():
            raise SystemExit(f"No such file: {args.descriptions}")
        descriptions, removed = load_descriptions(args.descriptions)
        print(f"Review    : {len(descriptions)} described clusters, "
              f"{len(removed)} flagged remove=1")
    rows = analyse(reference, args.k, sample, args.seed, args.batch_size,
                   descriptions=descriptions or None, removed=removed or None)
    report(rows, args.worst, removed=removed)


if __name__ == "__main__":
    raise SystemExit(main())
