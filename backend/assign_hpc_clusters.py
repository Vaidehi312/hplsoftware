"""Assign HPL cluster IDs to new embeddings by k-NN vote, with confidence.

Reproduces what sc.tl.ingest does for label transfer — a k-nearest-neighbour
majority vote in the reference PCA space
(HPL-LATTICeA/models/clustering/leiden_representations.py:244) — and adds the
one thing ingest does not report: how far each tile sits from the reference
manifold. Every tile gets a cluster either way, so without that number there is
no way to tell a confident assignment from a tile the reference has never seen
anything like.

Two confidence columns rather than one, because they fail independently:

  vote_margin        (top votes - runner-up votes) / k. Between-cluster
                     ambiguity. Low means the tile sits on a boundary.
  neighbor_distance  Mean distance to the k neighbours. Novelty. High means
                     nothing in the reference looks like this tile.

A tile can be deep inside one cluster yet far from everything (unseen
morphology: high margin, high distance), or close to the manifold but between
two clusters (low margin, low distance). One scalar cannot say which, and
collapsing them is how a "confidence" column stops being interpretable.

Speed note, measured on a 360,667 x 128 reference at k=250: 89% of the runtime
of the NumPy path is top-k selection, not distance computation, and it scales
with reference size only — float16 is ~30x slower (no native fp16 BLAS), and
truncating dimensions or lowering k changes nothing. So faiss is worth having
for its selection kernels, and the only other lever is fewer candidates.

Usage:
    python assign_hpc_clusters.py \
        --reference hpc_reference_leiden_2p5_fold2.npz \
        --h5        <DIR>/hdf5_<dataset>_he_filtered.h5 \
        --out       <dataset>_hpc_assignments.csv

    # Acceptance test: reproduce labels you already have
    python assign_hpc_clusters.py --reference ... --h5 <TCGA reps>.h5 \
        --out /tmp/check.csv \
        --validate-against TCGA_LUAD_5x_he_train_filtered_leiden_2p5__fold2.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Imported rather than re-defaulted here, so the path this reads from is the
# same object the builder writes to. Override with the HPC_REFERENCE_PATH
# environment variable.
from build_hpc_reference import HPC_REFERENCE_PATH

META_FIELDS = ("samples", "slides", "tiles")
SET_PREFIXES = ("train_", "valid_", "test_", "additional_")


# --------------------------------------------------------------------------- #
# Reading the new representations
# --------------------------------------------------------------------------- #

def read_representations(h5_path: Path, rep_key: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Embeddings plus their (samples, slides, tiles) metadata.

    Matches representations_to_frame() in HPL's data_processing.py — first key
    *containing* rep_key wins, and set prefixes are stripped off metadata names —
    without importing it. That module pulls in skbio, matplotlib, seaborn and
    anndata at import time, which is a heavy and fragile chain to take on for
    reading four datasets out of an HDF5 file. Matching its semantics keeps
    --rep-key meaning the same thing it means to the HPL scripts.
    """
    with h5py.File(h5_path, "r") as content:
        keys = list(content.keys())

        matching = [key for key in keys if rep_key in key]
        if not matching:
            raise KeyError(
                f"No dataset containing '{rep_key}' in {h5_path}. Present: {keys}"
            )
        if len(matching) > 1:
            print(
                f"[warn] several datasets match '{rep_key}' ({matching}); "
                f"using '{matching[0]}' — the same one the HPL scripts would take.",
                file=sys.stderr,
            )
        embeddings = np.asarray(content[matching[0]][:], dtype=np.float32)

        columns = {}
        for key in keys:
            if "latent" in key:
                continue
            name = key
            for prefix in SET_PREFIXES:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            if name not in META_FIELDS:
                continue
            # Only the three metadata fields are read, by name. HPL's version
            # loads *every* non-latent key into a DataFrame column, which is why
            # handing it a tiles .h5 explodes on the 4-D `img` array; naming the
            # fields wanted makes that impossible here.
            values = content[key][:]
            columns[name] = np.asarray(
                [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]
            )

    missing = [f for f in META_FIELDS if f not in columns]
    if missing:
        raise KeyError(
            f"{h5_path} has no {missing} dataset(s) — the output could not be "
            f"joined back to anything. Found: {sorted(columns)}"
        )

    frame = pd.DataFrame({field: columns[field] for field in META_FIELDS})
    if len(frame) != len(embeddings):
        raise ValueError(
            f"{len(embeddings)} embeddings but {len(frame)} metadata rows in {h5_path}."
        )
    return embeddings, frame


# --------------------------------------------------------------------------- #
# Projection into the reference space
# --------------------------------------------------------------------------- #

def project(embeddings: np.ndarray, components: np.ndarray,
            reference_mean: np.ndarray | None, centering: str) -> np.ndarray:
    """Raw embeddings -> reference PCA space.

    Getting this wrong is the failure that degrades everything downstream while
    still producing plausible cluster IDs, so the choice is explicit rather than
    assumed.

    centering="query" mirrors scanpy's Ingest._pca, which centres the *new* batch
    by its own mean before projecting. That is what reproducing ingest means, and
    it is also why a cohort with a different mean embedding gets shifted relative
    to the reference — a real quirk of ingest, not of this script.

    centering="reference" uses the mean stored with the reference, keeping every
    cohort in one fixed frame. Arguably more correct; it just is not what ingest
    did to produce the labels already in your KB.

    Note this cannot be settled by the acceptance test: validating on the same
    tiles the reference was built from makes query mean == reference mean, so all
    modes agree there and disagree only on genuinely new cohorts.
    """
    if embeddings.shape[1] != components.shape[0]:
        raise ValueError(
            f"Embeddings are {embeddings.shape[1]}-dimensional but the reference "
            f"PCA basis expects {components.shape[0]}. Different encoder, "
            f"different rep_key, or the wrong reference."
        )

    X = np.asarray(embeddings, dtype=np.float32)
    if centering == "query":
        X = X - X.mean(axis=0, keepdims=True)
    elif centering == "reference":
        if reference_mean is None:
            raise ValueError(
                "centering='reference' needs a mean stored in the reference .npz, "
                "and this one has none. Use --centering query."
            )
        X = X - reference_mean.reshape(1, -1)
    elif centering != "none":
        raise ValueError(f"Unknown centering mode: {centering}")

    return np.ascontiguousarray(X @ components, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Neighbour search
# --------------------------------------------------------------------------- #

def _search_numpy(reference: np.ndarray, queries: np.ndarray, k: int,
                  chunk: int = 40_000) -> tuple[np.ndarray, np.ndarray]:
    """Exact search, chunked over the reference.

    Chunked because the full distance block is len(queries) x len(reference):
    at 5,000 queries against 360k references that is 7.2 GB in float32, which
    thrashes long before it finishes. Chunking bounds it to chunk-width.
    """
    n = reference.shape[0]
    ref_sq = (reference ** 2).sum(axis=1)
    best_d = np.full((len(queries), k), np.inf, dtype=np.float32)
    best_i = np.zeros((len(queries), k), dtype=np.int64)

    for start in range(0, n, chunk):
        block = reference[start:start + chunk]
        # Squared distance without the per-query term, which is constant across
        # the row and so cannot affect the ranking. Added back by the caller.
        partial = ref_sq[start:start + chunk] - 2.0 * (queries @ block.T)
        width = min(k, partial.shape[1])
        idx = np.argpartition(partial, width - 1, axis=1)[:, :width]
        merged_d = np.concatenate([best_d, np.take_along_axis(partial, idx, 1)], axis=1)
        merged_i = np.concatenate([best_i, idx + start], axis=1)
        keep = np.argpartition(merged_d, k - 1, axis=1)[:, :k]
        best_d = np.take_along_axis(merged_d, keep, 1)
        best_i = np.take_along_axis(merged_i, keep, 1)

    query_sq = (queries ** 2).sum(axis=1, keepdims=True)
    return best_i, np.maximum(best_d + query_sq, 0.0)


class Searcher:
    """k-NN search over the reference, faiss-backed when available."""

    def __init__(self, reference: np.ndarray, backend: str, nlist: int, nprobe: int):
        self.reference = np.ascontiguousarray(reference, dtype=np.float32)
        self.index = None
        self.backend = "numpy"

        if backend == "numpy":
            return

        try:
            import faiss
        except ImportError:
            if backend != "auto":
                raise SystemExit(
                    f"--backend {backend} needs faiss, which is not installed. "
                    f"Use --backend numpy (exact, ~191 tiles/s at 360k reference)."
                )
            print("[info] faiss not installed; using the exact NumPy path.",
                  file=sys.stderr)
            return

        dim = self.reference.shape[1]
        if backend == "faiss-ivf":
            quantiser = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantiser, dim, nlist)
            index.train(self.reference)
            index.add(self.reference)
            index.nprobe = nprobe
            self.backend = f"faiss-ivf(nlist={nlist},nprobe={nprobe})"
        else:
            # Exact by default: an approximate index would put an approximation
            # inside the one number this script is judged on (agreement with the
            # labels already in the KB), for a speedup that is not needed until
            # measured to be.
            index = faiss.IndexFlatL2(dim)
            index.add(self.reference)
            self.backend = "faiss-flat"
        self.index = index

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if self.index is None:
            return _search_numpy(self.reference, queries, k)
        distances, indices = self.index.search(queries, k)
        # faiss can return -1 when an IVF probe finds fewer than k candidates.
        # Left as -1 would silently index the last reference row and vote for
        # whatever cluster it belongs to.
        return indices, distances


# --------------------------------------------------------------------------- #
# Voting
# --------------------------------------------------------------------------- #

def vote(neighbour_indices: np.ndarray, neighbour_distances: np.ndarray,
         codes: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Majority label, margin, and mean neighbour distance per query."""
    rows, k = neighbour_indices.shape
    valid = neighbour_indices >= 0
    safe = np.where(valid, neighbour_indices, 0)
    labels = codes[safe]

    # One bincount over row-offset label ids beats a per-row loop; at 250
    # neighbours x 71 clusters the dense count matrix is small.
    offsets = labels + (np.arange(rows, dtype=np.int64)[:, None] * n_clusters)
    counts = np.bincount(
        offsets[valid].ravel(), minlength=rows * n_clusters
    ).reshape(rows, n_clusters)

    order = np.argsort(counts, axis=1)
    winner = order[:, -1]
    top = np.take_along_axis(counts, winner[:, None], 1).ravel()
    runner_up = np.take_along_axis(counts, order[:, -2][:, None], 1).ravel() if n_clusters > 1 else 0

    found = valid.sum(axis=1)
    # Divide by neighbours actually found, not the requested k, so an IVF probe
    # that returned fewer does not read as a weaker margin than it is.
    denominator = np.maximum(found, 1)
    margin = (top - runner_up) / denominator

    distances = np.sqrt(np.maximum(neighbour_distances, 0.0))
    mean_distance = np.where(
        found > 0,
        np.sum(np.where(valid, distances, 0.0), axis=1) / denominator,
        np.nan,
    )
    return winner, margin.astype(np.float32), mean_distance.astype(np.float32)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def assign(args) -> pd.DataFrame:
    if not args.reference.is_file():
        raise SystemExit(
            f"No reference artifact at {args.reference}.\n"
            f"Build it once from the Leiden config:\n"
            f"  python build_hpc_reference.py --h5ad "
            f"<DIR>/rapids_2p5m/adatas/<stem>_leiden_2p5__fold2.h5ad\n"
            f"or point HPC_REFERENCE_PATH at an existing one."
        )

    bundle = np.load(args.reference, allow_pickle=False)
    meta = json.loads(str(bundle["meta"]))
    reference = bundle["reference"]
    components = bundle["components"]
    codes = bundle["codes"].astype(np.int64)
    categories = bundle["categories"]
    reference_mean = bundle["mean"] if "mean" in bundle.files else None
    k = args.k or int(bundle["n_neighbors"])
    groupby = meta.get("groupby", "leiden")

    if k > len(reference):
        raise SystemExit(f"k={k} exceeds the {len(reference)} reference tiles.")

    print(f"Reference : {len(reference):,} tiles, {reference.shape[1]} comps, "
          f"{len(categories)} clusters, k={k} ({groupby})")

    embeddings, frame = read_representations(args.h5, args.rep_key)
    if args.limit:
        embeddings, frame = embeddings[:args.limit], frame.iloc[:args.limit].copy()
    print(f"Queries   : {len(embeddings):,} tiles, {embeddings.shape[1]}-dim")

    queries = project(embeddings, components, reference_mean, args.centering)

    searcher = Searcher(reference, args.backend, args.nlist, args.nprobe)
    print(f"Backend   : {searcher.backend}, centering={args.centering}")

    winners = np.empty(len(queries), dtype=np.int64)
    margins = np.empty(len(queries), dtype=np.float32)
    distances = np.empty(len(queries), dtype=np.float32)

    started = time.perf_counter()
    for start in range(0, len(queries), args.batch_size):
        stop = min(start + args.batch_size, len(queries))
        idx, dist = searcher.search(queries[start:stop], k)
        w, m, d = vote(idx, dist, codes, len(categories))
        winners[start:stop], margins[start:stop], distances[start:stop] = w, m, d
        if args.progress and (start // args.batch_size) % args.progress == 0:
            done = stop
            rate = done / (time.perf_counter() - started)
            print(f"  {done:,}/{len(queries):,}  {rate:,.0f} tiles/s", flush=True)
    elapsed = time.perf_counter() - started
    print(f"Assigned  : {len(queries):,} tiles in {elapsed:.1f}s "
          f"({len(queries)/max(elapsed, 1e-9):,.0f} tiles/s)")

    frame[groupby] = [categories[w] for w in winners]
    frame["vote_margin"] = margins
    frame["neighbor_distance"] = distances
    # Which reference produced these IDs. Cluster numbers only mean something
    # relative to one reference plus one encoder checkpoint, and once the CSV is
    # merged into tile_registry there is otherwise nothing to tell assignments
    # from two different references apart.
    frame["hpc_reference"] = args.reference.stem
    return frame


def validate(frame: pd.DataFrame, truth_path: Path, groupby: str) -> bool:
    """Agreement against labels produced by the reference implementation.

    Joined on (slides, tiles), never on row order — the .h5 and the CSV are
    written by different programs and there is no guarantee they agree on it.
    """
    truth = pd.read_csv(truth_path)
    if groupby not in truth.columns:
        raise SystemExit(f"{truth_path} has no '{groupby}' column: {list(truth.columns)}")

    merged = frame.merge(
        truth[["slides", "tiles", groupby]].rename(columns={groupby: "expected"}),
        on=["slides", "tiles"], how="inner", validate="one_to_one",
    )
    if merged.empty:
        raise SystemExit(
            "No (slides, tiles) pairs matched. Check that both sides use the same "
            "tile naming — the CSV's tiles look like '18_15.jpeg'."
        )

    got = pd.to_numeric(merged[groupby], errors="coerce")
    want = pd.to_numeric(merged["expected"], errors="coerce")
    same = got == want
    agreement = same.mean()

    print(f"\nValidation: {len(merged):,} of {len(frame):,} tiles matched by "
          f"(slides, tiles)")
    print(f"  agreement {agreement*100:.3f}%  ({int(same.sum()):,} / {len(merged):,})")

    if not same.all():
        wrong = merged.loc[~same]
        # This path is meant to *be* k-NN, not approximate it, so a disagreement
        # is a defect. It should at least be confined to boundary tiles: a
        # high-margin disagreement means the space or k is wrong, not that the
        # tile was ambiguous.
        print(f"  disagreements {len(wrong):,}; "
              f"vote_margin median {wrong['vote_margin'].median():.3f} "
              f"vs {merged.loc[same, 'vote_margin'].median():.3f} for agreements")
        worst = wrong.nlargest(min(5, len(wrong)), "vote_margin")
        print("  highest-margin disagreements (these are the concerning ones):")
        for _, row in worst.iterrows():
            print(f"    {row['slides']}/{row['tiles']}: got {row[groupby]} "
                  f"want {row['expected']} margin {row['vote_margin']:.3f}")
    return bool(agreement >= 0.99)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", type=Path, default=HPC_REFERENCE_PATH,
                        help=f".npz from build_hpc_reference.py. Defaults to "
                             f"HPC_REFERENCE_PATH ({HPC_REFERENCE_PATH}).")
    parser.add_argument("--h5", required=True, type=Path,
                        help="Representations .h5 (not a tiles .h5)")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV")
    parser.add_argument("--rep-key", default="z_latent")
    parser.add_argument("--k", type=int, default=None,
                        help="Override the reference's own n_neighbors. Changing it "
                             "makes this a different function than the one that "
                             "produced your existing labels.")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "faiss", "faiss-ivf", "numpy"])
    parser.add_argument("--nlist", type=int, default=600,
                        help="IVF cells; ~sqrt(n_reference) is the usual choice.")
    parser.add_argument("--nprobe", type=int, default=32,
                        help="IVF cells probed. Raise until validation agreement "
                             "plateaus, then stop.")
    parser.add_argument("--centering", default="query",
                        choices=["query", "reference", "none"],
                        help="query (default) mirrors scanpy ingest. See project().")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None,
                        help="Assign only the first N tiles, for a quick check.")
    parser.add_argument("--progress", type=int, default=0,
                        help="Print progress every N batches (0 = silent).")
    parser.add_argument("--validate-against", type=Path, default=None,
                        help="CSV of known labels; report agreement and exit "
                             "non-zero below 99%%.")
    args = parser.parse_args()

    frame = assign(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Written   : {args.out}")

    # By name, not position — the frame has grown a column before now.
    groupby = next(c for c in frame.columns
                   if c not in ("samples", "slides", "tiles", "vote_margin",
                                "neighbor_distance", "hpc_reference"))
    counts = frame[groupby].value_counts(normalize=True)
    print(f"Largest clusters: " + ", ".join(
        f"{i}={v*100:.1f}%" for i, v in counts.head(5).items()
    ))
    print(f"vote_margin       median {frame['vote_margin'].median():.3f}, "
          f"{(frame['vote_margin'] < 0.1).mean()*100:.1f}% below 0.1")
    print(f"neighbor_distance median {frame['neighbor_distance'].median():.3f}, "
          f"p95 {frame['neighbor_distance'].quantile(0.95):.3f}")

    if args.validate_against:
        if not validate(frame, args.validate_against, groupby):
            raise SystemExit(
                "\nAgreement below 99% — treat as a defect, not drift. Check the "
                "PCA projection (--centering), k, and that this is the reference "
                "the existing labels came from."
            )
        print("\nAgreement acceptable.")


if __name__ == "__main__":
    main()
