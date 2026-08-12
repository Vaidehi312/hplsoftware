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

Speed note, measured on a 360,667 x 128 reference at k=250: 89% of an exact
search's runtime is top-k selection, not distance computation, and it scales
with reference size only — float16 is ~30x slower (no native fp16 BLAS), and
truncating dimensions or lowering k changes nothing. So faiss's selection
kernels are why this is fast at all, and the only other lever is fewer
candidates.

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

def _resolve_datasets(content: h5py.File, rep_key: str) -> tuple[str, dict[str, str]]:
    """Pick the embedding dataset and the metadata datasets, by HPL's rules.

    Split out from reading so every pass over the file agrees on which datasets
    it is looking at. Two passes that resolved `--rep-key` independently could
    disagree if a file held more than one match, and the mean would then be
    computed over different vectors than the ones assigned.
    """
    keys = list(content.keys())
    matching = [key for key in keys if rep_key in key]
    if not matching:
        raise KeyError(f"No dataset containing '{rep_key}' in {content.filename}. Present: {keys}")
    if len(matching) > 1:
        print(
            f"[warn] several datasets match '{rep_key}' ({matching}); "
            f"using '{matching[0]}' — the same one the HPL scripts would take.",
            file=sys.stderr,
        )

    meta_keys: dict[str, str] = {}
    for key in keys:
        if "latent" in key:
            continue
        name = key
        for prefix in SET_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        # Only the three metadata fields are read, by name. HPL's version loads
        # *every* non-latent key into a DataFrame column, which is why handing it
        # a tiles .h5 explodes on the 4-D `img` array; naming the fields wanted
        # makes that impossible here.
        if name in META_FIELDS:
            meta_keys[name] = key

    missing = [f for f in META_FIELDS if f not in meta_keys]
    if missing:
        raise KeyError(
            f"{content.filename} has no {missing} dataset(s) — the output could not "
            f"be joined back to anything. Found: {sorted(meta_keys)}"
        )
    return matching[0], meta_keys


def _decode_column(values: np.ndarray) -> np.ndarray:
    """Bytes-or-whatever HDF5 gave us -> str, without a per-tile Python loop.

    The list comprehension this replaces ran once per tile per field: three
    passes over 14M rows is ~42M interpreter iterations for work numpy does in C.
    """
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8")
    if values.dtype.kind in ("U", "O"):
        return values.astype(str)
    return values.astype(str)


def read_metadata_frame(h5_path: Path, meta_keys: dict[str, str],
                        lo: int, hi: int) -> pd.DataFrame:
    """The (samples, slides, tiles) columns for rows [lo, hi)."""
    with h5py.File(h5_path, "r") as content:
        columns = {
            field: _decode_column(np.asarray(content[key][lo:hi]))
            for field, key in meta_keys.items()
        }
    return pd.DataFrame({field: columns[field] for field in META_FIELDS})


def query_row_count(h5_path: Path, rep_key: str) -> int:
    with h5py.File(h5_path, "r") as content:
        rep_name, meta_keys = _resolve_datasets(content, rep_key)
        rows = int(content[rep_name].shape[0])
        for field, key in meta_keys.items():
            if int(content[key].shape[0]) != rows:
                raise ValueError(
                    f"{rows} embeddings but {content[key].shape[0]} '{field}' rows "
                    f"in {h5_path}."
                )
    return rows


def iter_embedding_chunks(h5_path: Path, rep_key: str, chunk: int,
                          lo: int = 0, hi: int | None = None):
    """Yield (start, stop, embeddings) over rows [lo, hi) in `chunk`-row pieces.

    The whole point of this module's rework: the previous code read the entire
    embedding dataset with a single [:], which is ~14 GB at 14M tiles in 128
    dimensions and ~86 GB with --rep-key h_latent (1536-d). Streaming bounds
    peak memory to the chunk regardless of input size.
    """
    with h5py.File(h5_path, "r") as content:
        rep_name, _ = _resolve_datasets(content, rep_key)
        dataset = content[rep_name]
        end = dataset.shape[0] if hi is None else min(hi, dataset.shape[0])
        for start in range(lo, end, chunk):
            stop = min(start + chunk, end)
            yield start, stop, np.asarray(dataset[start:stop], dtype=np.float32)


def compute_query_mean(h5_path: Path, rep_key: str, chunk: int,
                       total_rows: int) -> np.ndarray:
    """Mean over *every* query row, accumulated in float64.

    Deliberately over the whole query set and never over a shard. --centering
    query mirrors scanpy's Ingest._pca, which centres the new batch by its own
    mean, so the mean is a property of the file rather than of the slice being
    worked on. A shard that computed its own mean would project into a slightly
    different frame and produce different cluster labels — a well-formed CSV of
    silently wrong IDs. This is why sharding passes --query-mean instead.

    float64 accumulation because a float32 running sum over millions of rows
    loses low-order bits, and the mean feeds every projected coordinate.
    """
    if total_rows == 0:
        raise ValueError(f"{h5_path} holds no embeddings.")
    total = None
    seen = 0
    for _, _, block in iter_embedding_chunks(h5_path, rep_key, chunk, 0, total_rows):
        contribution = block.astype(np.float64).sum(axis=0)
        total = contribution if total is None else total + contribution
        seen += len(block)
    if seen != total_rows:
        raise ValueError(f"Expected {total_rows} rows for the mean, streamed {seen}.")
    return (total / seen).astype(np.float32)


# --------------------------------------------------------------------------- #
# Projection into the reference space
# --------------------------------------------------------------------------- #

def project(embeddings: np.ndarray, components: np.ndarray,
            reference_mean: np.ndarray | None, centering: str,
            query_mean: np.ndarray | None = None) -> np.ndarray:
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
        # query_mean is required rather than derived from `embeddings`, because
        # this function is now handed one chunk at a time. Centring a chunk by
        # its own mean would make a tile's coordinates depend on which chunk it
        # landed in — the exact bug streaming could have introduced. The caller
        # computes the mean over the whole query set first (compute_query_mean).
        if query_mean is None:
            raise ValueError(
                "centering='query' needs the mean over all queries. Call "
                "compute_query_mean() and pass query_mean; deriving it from a "
                "chunk would make results depend on the chunk boundaries."
            )
        X = X - np.asarray(query_mean, dtype=np.float32).reshape(1, -1)
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

class Searcher:
    """Exact k-NN search over the reference via a faiss flat index.

    No approximate index is offered: faiss-ivf was measured against this
    reference to agree with the true nearest neighbour only 33% of the time,
    for no speed gain, which means it changes cluster labels rather than just
    trading accuracy for speed. faiss is required rather than optional so that
    choice cannot be made by accident.
    """

    def __init__(self, reference: np.ndarray):
        self.reference = np.ascontiguousarray(reference, dtype=np.float32)
        try:
            import faiss
        except ImportError as e:
            raise SystemExit(
                "faiss is required (pip install faiss-cpu) — there is no "
                "fallback search path."
            ) from e

        index = faiss.IndexFlatL2(self.reference.shape[1])
        index.add(self.reference)
        self.index = index
        self.backend = "faiss-flat"

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        distances, indices = self.index.search(queries, k)
        # faiss returns -1 when k exceeds the number of vectors in the index.
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

def _load_reference(reference_path: Path) -> dict:
    bundle = np.load(reference_path, allow_pickle=False)
    meta = json.loads(str(bundle["meta"]))
    return {
        "reference": bundle["reference"],
        "components": bundle["components"],
        "codes": bundle["codes"].astype(np.int64),
        "categories": bundle["categories"],
        "mean": bundle["mean"] if "mean" in bundle.files else None,
        "n_neighbors": int(bundle["n_neighbors"]),
        "groupby": meta.get("groupby", "leiden"),
    }


def resolve_query_mean(args, total_rows: int) -> np.ndarray | None:
    """The mean to centre queries by, or None when centering needs no mean.

    Three ways in, and the order matters. An explicitly supplied --query-mean
    wins, because that is how every shard of a sharded run ends up in the same
    frame. Otherwise it is computed over the whole file. --centering reference
    and none never need one.
    """
    if args.centering == "query":
        if args.query_mean is not None:
            mean = np.load(args.query_mean).astype(np.float32).ravel()
            print(f"Query mean: loaded from {args.query_mean}")
            return mean
        return compute_query_mean(args.h5, args.rep_key, args.chunk_size, total_rows)
    return None


def assign(args) -> dict:
    """Stream the queries, assign each chunk, append to the CSV as we go.

    Returns summary statistics rather than a DataFrame. The frame used to be
    held whole and written at the end, which put a hard ceiling on input size
    for no benefit — every row is final the moment its chunk is voted on.

    vote_margin and neighbor_distance are still kept in memory: two float32
    arrays are 8 bytes a tile (112 MB at 14M), which buys exact medians and
    percentiles in the summary without re-reading the CSV. The string columns,
    which are what actually cost gigabytes, are written and dropped.
    """
    if not args.reference.is_file():
        raise SystemExit(
            f"No reference artifact at {args.reference}.\n"
            f"Build it once from the Leiden config:\n"
            f"  python build_hpc_reference.py --h5ad "
            f"<DIR>/rapids_2p5m/adatas/<stem>_leiden_2p5__fold2.h5ad\n"
            f"or point HPC_REFERENCE_PATH at an existing one."
        )

    ref = _load_reference(args.reference)
    reference, components = ref["reference"], ref["components"]
    codes, categories = ref["codes"], ref["categories"]
    k = args.k or ref["n_neighbors"]
    groupby = ref["groupby"]
    category_lookup = np.asarray(categories)

    if k > len(reference):
        raise SystemExit(f"k={k} exceeds the {len(reference)} reference tiles.")

    print(f"Reference : {len(reference):,} tiles, {reference.shape[1]} comps, "
          f"{len(categories)} clusters, k={k} ({groupby})")

    total_rows = query_row_count(args.h5, args.rep_key)
    if args.limit:
        total_rows = min(total_rows, args.limit)

    # The slice this process is responsible for. The mean above is always over
    # the whole file; only the assignment is sliced.
    lo = 0 if args.row_start is None else max(0, min(args.row_start, total_rows))
    hi = total_rows if args.row_stop is None else max(lo, min(args.row_stop, total_rows))
    if lo == hi:
        raise SystemExit(f"Empty row range [{lo}, {hi}) — nothing to assign.")
    sharded = (lo, hi) != (0, total_rows)
    print(f"Queries   : {total_rows:,} tiles total"
          + (f", assigning rows [{lo:,}, {hi:,})" if sharded else ""))

    query_mean = resolve_query_mean(args, total_rows)

    with h5py.File(args.h5, "r") as content:
        _, meta_keys = _resolve_datasets(content, args.rep_key)

    searcher = Searcher(reference)
    print(f"Backend   : {searcher.backend}, centering={args.centering}")

    out_path = args.out
    if sharded:
        out_path = out_path.with_name(f"{out_path.stem}.rows{lo}-{hi}{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_assigned = hi - lo
    margins = np.empty(n_assigned, dtype=np.float32)
    distances = np.empty(n_assigned, dtype=np.float32)
    cluster_counts = np.zeros(len(categories), dtype=np.int64)

    # Written to a temporary name and renamed at the end: a run killed partway
    # would otherwise leave a valid-looking CSV holding some of the tiles, and
    # nothing downstream inspects row counts before merging.
    tmp_path = out_path.with_name(out_path.name + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()

    started = time.perf_counter()
    written = 0
    try:
        for start, stop, block in iter_embedding_chunks(
            args.h5, args.rep_key, args.chunk_size, lo, hi
        ):
            queries = project(block, components, ref["mean"], args.centering,
                              query_mean=query_mean)
            offset = start - lo
            for bstart in range(0, len(queries), args.batch_size):
                bstop = min(bstart + args.batch_size, len(queries))
                idx, dist = searcher.search(queries[bstart:bstop], k)
                w, m, d = vote(idx, dist, codes, len(categories))
                margins[offset + bstart:offset + bstop] = m
                distances[offset + bstart:offset + bstop] = d
                np.add.at(cluster_counts, w, 1)
                if bstart == 0:
                    chunk_winners = np.empty(len(queries), dtype=np.int64)
                chunk_winners[bstart:bstop] = w

            frame = read_metadata_frame(args.h5, meta_keys, start, stop)
            # Vectorised label lookup, replacing a per-tile list comprehension.
            frame[groupby] = category_lookup[chunk_winners]
            frame["vote_margin"] = margins[offset:offset + len(queries)]
            frame["neighbor_distance"] = distances[offset:offset + len(queries)]
            # Which reference produced these IDs. Cluster numbers only mean
            # something relative to one reference plus one encoder checkpoint,
            # and once the CSV is merged into tile_registry there is otherwise
            # nothing to tell assignments from two different references apart.
            frame["hpc_reference"] = args.reference.stem
            frame.to_csv(tmp_path, mode="a", header=(written == 0), index=False)
            written += len(frame)

            if args.progress and written % max(args.progress, 1) < len(frame):
                rate = written / max(time.perf_counter() - started, 1e-9)
                print(f"  {written:,}/{n_assigned:,}  {rate:,.0f} tiles/s", flush=True)

        if written != n_assigned:
            raise RuntimeError(f"Wrote {written} rows for a range of {n_assigned}.")
        tmp_path.replace(out_path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    elapsed = time.perf_counter() - started
    print(f"Assigned  : {written:,} tiles in {elapsed:.1f}s "
          f"({written/max(elapsed, 1e-9):,.0f} tiles/s)")
    print(f"Written   : {out_path}")

    return {
        "out_path": out_path,
        "groupby": groupby,
        "rows": written,
        "margins": margins,
        "distances": distances,
        "cluster_counts": cluster_counts,
        "categories": categories,
        "sharded": sharded,
    }


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
    parser.add_argument("--out", type=Path, default=None, help="Output CSV")
    parser.add_argument("--rep-key", default="z_latent")
    parser.add_argument("--k", type=int, default=None,
                        help="Neighbours to poll. Defaults to the reference's own "
                             "Leiden n_neighbors, which is what ingest used.")
    parser.add_argument("--centering", default="query",
                        choices=["query", "reference", "none"],
                        help="How queries are centred before projection. 'query' "
                             "reproduces sc.tl.ingest and is what produced the "
                             "labels already in the KB.")
    # Rows read from the .h5 at a time. Bounds peak memory independently of the
    # search batch below: 32,768 rows of 128-d float32 is 16 MB, of 1536-d
    # h_latent is 200 MB.
    parser.add_argument("--chunk-size", type=int, default=32_768,
                        help="Embedding rows read per chunk (memory ceiling).")
    # Queries per faiss search call. Raised from 4,096: larger batches amortise
    # the per-call overhead, and search is per-query independent so this cannot
    # change an assignment.
    parser.add_argument("--batch-size", type=int, default=16_384,
                        help="Queries per k-NN search call.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Assign only the first N tiles.")
    parser.add_argument("--progress", type=int, default=0,
                        help="Print progress every N tiles (0 = silent).")
    parser.add_argument("--validate-against", type=Path, default=None,
                        help="CSV of known labels; report agreement and exit "
                             "non-zero below 99%%.")
    # --- sharding -------------------------------------------------------------
    # Splitting this stage across jobs is only safe if every shard centres on the
    # same mean (see compute_query_mean). --precompute-mean produces that mean
    # once; --query-mean hands it to each shard; --row-start/--row-stop select
    # the slice. Using --row-start under 'query' centering without --query-mean
    # is refused below rather than silently producing different labels.
    parser.add_argument("--precompute-mean", type=Path, default=None,
                        help="Compute the query mean over the whole .h5, write it "
                             "to this .npy, and exit without assigning.")
    parser.add_argument("--query-mean", type=Path, default=None,
                        help="Use this .npy as the query mean instead of computing "
                             "one. Required when sharding under 'query' centering.")
    parser.add_argument("--row-start", type=int, default=None,
                        help="First tile to assign. Output goes to "
                             "<out>.rows<lo>-<hi>.csv.")
    parser.add_argument("--row-stop", type=int, default=None,
                        help="One past the last tile to assign.")
    args = parser.parse_args()

    if args.precompute_mean is not None:
        total_rows = query_row_count(args.h5, args.rep_key)
        if args.limit:
            total_rows = min(total_rows, args.limit)
        mean = compute_query_mean(args.h5, args.rep_key, args.chunk_size, total_rows)
        args.precompute_mean.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.precompute_mean, mean)
        print(f"Query mean over {total_rows:,} tiles -> {args.precompute_mean} "
              f"({mean.shape[0]} dims)")
        return

    if args.out is None:
        parser.error("--out is required unless --precompute-mean is given")

    sharding = args.row_start is not None or args.row_stop is not None
    if sharding and args.centering == "query" and args.query_mean is None:
        # The failure this prevents is invisible: each shard would centre on its
        # own slice's mean, project into a slightly different space, and emit a
        # perfectly well-formed CSV of different cluster IDs.
        parser.error(
            "Sharding with --centering query needs --query-mean, or every shard "
            "centres on its own slice and produces different labels. Run with "
            "--precompute-mean first, then pass that file to each shard."
        )

    stats = assign(args)

    counts = stats["cluster_counts"]
    total = max(counts.sum(), 1)
    order = np.argsort(counts)[::-1][:5]
    print("Largest clusters: " + ", ".join(
        f"{stats['categories'][i]}={counts[i]/total*100:.1f}%" for i in order
    ))
    margins, distances = stats["margins"], stats["distances"]
    print(f"vote_margin       median {np.median(margins):.3f}, "
          f"{(margins < 0.1).mean()*100:.1f}% below 0.1")
    finite = distances[np.isfinite(distances)]
    if len(finite):
        print(f"neighbor_distance median {np.median(finite):.3f}, "
              f"p95 {np.quantile(finite, 0.95):.3f}")

    if args.validate_against:
        if stats["sharded"]:
            print("\n[warn] --validate-against skipped: this run assigned one shard, "
                  "and agreement is only meaningful over the whole set.",
                  file=sys.stderr)
            return
        # Read back rather than kept in memory — the whole point of streaming is
        # not holding every row, and the CSV on disk is the artifact anyway.
        frame = pd.read_csv(stats["out_path"])
        if not validate(frame, args.validate_against, stats["groupby"]):
            raise SystemExit(
                "\nAgreement below 99% — treat as a defect, not drift. Check the "
                "PCA projection (--centering), k, and that this is the reference "
                "the existing labels came from."
            )
        print("\nAgreement acceptable.")


if __name__ == "__main__":
    main()
