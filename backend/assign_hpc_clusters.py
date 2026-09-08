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

_COSINE_EPS = 1e-12  # guards a zero-norm vector, which cosine has no direction for


class Searcher:
    """Exact k-NN search over the reference via a faiss flat index.

    No approximate index is offered: faiss-ivf was measured against this
    reference to agree with the true nearest neighbour only 33% of the time,
    for no speed gain, which means it changes cluster labels rather than just
    trading accuracy for speed. faiss is required rather than optional so that
    choice cannot be made by accident.

    metric="cosine" normalises reference and query vectors to unit length and
    searches with IndexFlatIP (inner product), then converts the returned
    similarity back to a squared-Euclidean-equivalent distance —
    ||a-b||^2 = 2 - 2cos(theta) for unit vectors — so vote()'s sqrt(distance),
    margin, and neighbor_distance need no changes downstream. Still exact:
    this is a different metric on the same flat index, not an approximation.
    """

    def __init__(self, reference: np.ndarray, metric: str = "l2",
                 device: str = "cpu"):
        if metric not in ("l2", "cosine"):
            raise ValueError(f"Unknown metric: {metric!r}. Use 'l2' or 'cosine'.")
        if device not in ("cpu", "gpu"):
            raise ValueError(f"Unknown device: {device!r}. Use 'cpu' or 'gpu'.")
        self.metric = metric
        self.device = device
        self.reference = np.ascontiguousarray(reference, dtype=np.float32)
        try:
            import faiss
        except ImportError as e:
            raise SystemExit(
                "faiss is required (pip install faiss-cpu) — there is no "
                "fallback search path."
            ) from e

        if metric == "cosine":
            normed = self._normalize(self.reference)
            index = faiss.IndexFlatIP(normed.shape[1])
            index.add(normed)
        else:
            index = faiss.IndexFlatL2(self.reference.shape[1])
            index.add(self.reference)

        self.backend = "faiss-flat" if metric == "l2" else "faiss-flat-cosine"
        if device == "gpu":
            # Still the same exhaustive scan — GpuIndexFlat compares every query
            # against every reference vector, exactly as IndexFlat does. That is
            # what makes this admissible where faiss-ivf was not: ivf changed
            # which neighbour came back (33% agreement), and this changes only
            # the hardware doing the arithmetic.
            #
            # Refused rather than quietly falling back to CPU. A silent fallback
            # here is a ~30x slowdown that shows up as nothing but wall clock,
            # which is the same class of bug as the --cleanenv thread count.
            if not hasattr(faiss, "StandardGpuResources"):
                raise SystemExit(
                    "--device gpu needs a faiss build with GPU support, and this "
                    "one has none (faiss.StandardGpuResources is missing). The "
                    "container's extras hold faiss-cpu; install the GPU extras "
                    "with `python submit_feature_extraction.py "
                    "--bootstrap-extras-gpu` and submit with --device gpu, or "
                    "drop the flag to search on CPU."
                )
            try:
                self._gpu_resources = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, index)
            except Exception as e:  # noqa: BLE001 — surfaced, never downgraded
                raise SystemExit(
                    f"--device gpu was asked for and the index could not be "
                    f"moved to GPU 0: {type(e).__name__}: {e}. Refusing rather "
                    f"than searching on CPU at a thirtieth of the speed without "
                    f"saying so."
                ) from e
            self._verify_matches_cpu(index, gpu_index)
            index = gpu_index
            self.backend += "-gpu"
        self.index = index

    def _verify_matches_cpu(self, cpu_index, gpu_index, sample: int = 2048) -> None:
        """Refuse a GPU index that does not agree with the CPU one.

        Not defensive programming — a measured necessity. A faiss build can
        expose StandardGpuResources, accept index_cpu_to_gpu without error, and
        then return entirely different neighbours: on faiss 1.15.0 with
        get_num_gpus() reporting 1 and no usable device, this returned the right
        *shape* of answer with 18% of the top-1 neighbours correct and distances
        wrong by three orders of magnitude.

        Nothing downstream could catch that. Every tile would get a cluster ID,
        every margin would look plausible, the CSV would have no missing values,
        and the labels would be noise. So the capability is tested by doing the
        search rather than by asking whether it is available — one second on a
        sample of the reference, against the CPU index that is already built.

        Queries are reference rows, which makes the answer checkable on its own
        terms as well: a vector's nearest neighbour is itself, at distance zero.
        """
        rows = min(sample, self.reference.shape[0])
        if not rows:
            return
        probe = self.reference[:rows]
        if self.metric == "cosine":
            probe = self._normalize(probe)
        k = min(10, self.reference.shape[0])

        cpu_distances, cpu_indices = cpu_index.search(probe, k)
        gpu_distances, gpu_indices = gpu_index.search(probe, k)

        top1 = float((cpu_indices[:, 0] == gpu_indices[:, 0]).mean())
        # Absolute tolerance rather than relative: these are squared distances
        # whose scale is set by the embedding, and float32 accumulation order
        # differs legitimately between the two implementations.
        worst = float(np.abs(cpu_distances - gpu_distances).max())
        scale = max(float(np.abs(cpu_distances).max()), 1.0)

        if top1 < 0.999 or worst > 1e-2 * scale:
            raise SystemExit(
                f"REFUSING --device gpu: the GPU index disagrees with the CPU "
                f"index on this machine. Of {rows:,} reference vectors searched "
                f"against the index they came from, {top1 * 100:.1f}% returned "
                f"themselves as the nearest neighbour (must be ~100%), and the "
                f"largest distance difference was {worst:.3g} against a scale of "
                f"{scale:.3g}.\n\n"
                f"This is a broken or stub GPU faiss build, not a precision "
                f"difference. Every cluster ID it produced would be wrong while "
                f"looking entirely well-formed. Search on CPU (--device cpu) "
                f"until the container's GPU faiss is fixed."
            )

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.ascontiguousarray(vectors / np.maximum(norms, _COSINE_EPS), dtype=np.float32)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if self.metric == "cosine":
            similarities, indices = self.index.search(self._normalize(queries), k)
            # faiss pads a missing neighbour (index -1, see below) with a
            # sentinel similarity near float32's minimum, which overflows
            # 2 - 2*sim before the clip below. Harmless — those slots are
            # masked out downstream by index >= 0 — but errstate keeps the
            # overflow from printing a warning on every such (valid) call.
            with np.errstate(over="ignore"):
                # Clip for float error near cos(theta)=1 (distance 0), not
                # because a real cosine similarity can exceed 1.
                distances = np.clip(2.0 - 2.0 * similarities, 0.0, None)
        else:
            distances, indices = self.index.search(queries, k)
        # faiss returns -1 when k exceeds the number of vectors in the index.
        # Left as -1 would silently index the last reference row and vote for
        # whatever cluster it belongs to.
        return indices, distances


# --------------------------------------------------------------------------- #
# Voting
# --------------------------------------------------------------------------- #

_WEIGHT_EPS = 1e-6  # avoids a divide-by-zero for a duplicate-vector neighbour at distance 0
# Floor on a local density scale. A reference point sitting on duplicates has a
# scale of zero, and dividing by it would send every distance to infinity.
_SCALE_EPS = 1e-6

def compute_local_scale_for(vectors: np.ndarray, rows: np.ndarray, r: int = 7,
                            batch: int = 8192) -> np.ndarray:
    """Local density scale for a chosen subset of reference rows.

    The index is still the whole reference — a density scale computed against a
    subsample would be a different quantity — but only `rows` are queried. That
    matters a great deal: at the production reference's measured 28 ms/query,
    scaling every one of 2.5M rows is 19.5 hours, while the ~1M distinct rows a
    20,000-tile sweep actually touches is a fraction of that, and a smaller
    sample is minutes.
    """
    if r < 1:
        raise ValueError(f"r must be at least 1, got {r}")
    rows = np.asarray(rows, dtype=np.int64)
    searcher = Searcher(vectors)
    scale = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), batch):
        stop = min(start + batch, len(rows))
        block_rows = rows[start:stop]
        idx, dist = searcher.search(
            np.ascontiguousarray(vectors[block_rows]), r + 1
        )
        self_mask = idx == block_rows[:, None]
        no_self = ~self_mask.any(axis=1)
        if no_self.any():
            self_mask[no_self, -1] = True
        kept = np.sqrt(np.maximum(dist[~self_mask].reshape(len(block_rows), r), 0.0))
        scale[start:stop] = kept[:, -1]
    return scale


def compute_local_scale(vectors: np.ndarray, r: int = 7,
                        batch: int = 8192) -> np.ndarray:
    """Each reference point's local density scale: its distance to its r-th
    nearest *other* reference point.

    This is the same quantity UMAP calls sigma, arrived at cheaply. UMAP solves
    a binary search per point so the neighbourhood's entropy equals log2(k);
    the r-th neighbour distance is the standard cheap stand-in, and it captures
    the property that matters here — that a point in a dense region has a small
    scale and one in a sparse region a large one.

    r is the *other* points, so the search asks for r+1 and drops the
    self-match. A reference with duplicate vectors can legitimately produce a
    scale of zero; vote() floors it rather than dividing by it.
    """
    if r < 1:
        raise ValueError(f"r must be at least 1, got {r}")
    searcher = Searcher(vectors)
    scale = np.empty(len(vectors), dtype=np.float32)
    for start in range(0, len(vectors), batch):
        stop = min(start + batch, len(vectors))
        rows = np.arange(start, stop)
        idx, dist = searcher.search(np.ascontiguousarray(vectors[start:stop]), r + 1)
        # Mask the self-match by index rather than assuming column 0: with
        # duplicate vectors the tie order is not guaranteed.
        self_mask = idx == rows[:, None]
        no_self = ~self_mask.any(axis=1)
        if no_self.any():
            self_mask[no_self, -1] = True
        kept = np.sqrt(np.maximum(dist[~self_mask].reshape(len(rows), r), 0.0))
        scale[start:stop] = kept[:, -1]
    return scale


def vote(neighbour_indices: np.ndarray, neighbour_distances: np.ndarray,
         codes: np.ndarray, n_clusters: int,
         distance_weighted: bool = False, distance_power: float = 1.0,
         class_weights: np.ndarray | None = None,
         local_scale: np.ndarray | None = None,
         return_counts: bool = False) -> tuple[np.ndarray, ...]:
    """Majority (or weighted) label, margin, and mean neighbour distance.

    distance_weighted gives neighbour i a vote of 1/(distance_i + eps)**power
    instead of one vote each — the fix for a boundary tile where an unweighted
    count among k neighbours ties or nearly ties between two clusters that are
    not actually equally close. distance_power > 1 (e.g. 2) makes the nearest
    few neighbours matter much more than the rest; 1.0 is the default weighting.

    class_weights, one entry per cluster code (pass 1/reference_count), further
    scales each neighbour's vote by its cluster's weight — the fix for a large
    cluster winning a boundary tile simply by having more points nearby, not by
    being genuinely closer. It composes with distance_weighted rather than
    replacing it: a neighbour's final weight is the distance term times its
    cluster's weight.

    All three are opt-in and independent. With none, every valid neighbour's
    weight is exactly 1, so counts and margin come out bit-for-bit what this
    returned before any of them existed.

    return_counts appends the full per-cluster vote-weight matrix to the return
    tuple. Diagnostics need it to ask where the *true* cluster placed on a tile
    that came out wrong — runner-up is a different problem from also-ran, and
    only the losing scores distinguish them. Off by default so the assignment
    path never materialises a rows x n_clusters array it does not read.
    """
    rows, k = neighbour_indices.shape
    valid = neighbour_indices >= 0
    safe = np.where(valid, neighbour_indices, 0)
    labels = codes[safe]
    distances = np.sqrt(np.maximum(neighbour_distances, 0.0))

    # local_scale divides each neighbour's distance by that neighbour's own
    # local density scale, so "close" is judged relative to how tightly packed
    # the reference is where that neighbour sits, not on an absolute ruler.
    #
    # This is the asymmetry the reference's own construction creates. Leiden
    # partitioned a UMAP fuzzy-simplicial-set graph whose edge weights are
    # exp(-(d - rho_i)/sigma_i), with sigma_i solved per reference point so
    # every neighbourhood carries equal entropy. 1/(d+eps)^p has no per-point
    # scale at all, so where A and B differ in local density the denser one
    # wins on sheer numerosity of nearby points rather than on being the
    # better answer.
    #
    # Weighting only. mean_distance below stays on raw distances, because it is
    # written to the assignments CSV as neighbor_distance and consumed as a real
    # distance — rescaling it would silently change what that column means.
    weight_distances = distances
    if local_scale is not None:
        weight_distances = distances / np.maximum(local_scale[safe], _SCALE_EPS)

    if distance_weighted:
        weights = np.where(
            valid, 1.0 / (weight_distances + _WEIGHT_EPS) ** distance_power, 0.0
        )
    else:
        weights = valid.astype(np.float64)

    if class_weights is not None:
        # Invalid slots already carry weight 0 above, so multiplying by
        # whatever class_weights[labels] resolves to there (a placeholder
        # cluster, since safe replaced -1 with 0) stays 0 rather than leaking
        # a spurious weight in.
        weights = weights * class_weights[labels]

    # One bincount over row-offset label ids beats a per-row loop; at 250
    # neighbours x 71 clusters the dense count matrix is small.
    offsets = labels + (np.arange(rows, dtype=np.int64)[:, None] * n_clusters)
    counts = np.bincount(
        offsets[valid].ravel(), weights=weights[valid].ravel(), minlength=rows * n_clusters
    ).reshape(rows, n_clusters)

    order = np.argsort(counts, axis=1)
    winner = order[:, -1]
    top = np.take_along_axis(counts, winner[:, None], 1).ravel()
    runner_up = np.take_along_axis(counts, order[:, -2][:, None], 1).ravel() if n_clusters > 1 else 0

    found = valid.sum(axis=1)
    # Divide by total vote weight, not neighbour count, so margin means the
    # same thing in both modes: the winner's share of the vote over the
    # runner-up. In unweighted mode weight sums to found, so this is
    # unchanged from before — a faiss result padded with -1 (k larger than
    # the index) still doesn't read as a weaker margin than it is.
    total_weight = np.maximum(weights.sum(axis=1), 1e-12)
    margin = (top - runner_up) / total_weight

    denominator = np.maximum(found, 1)
    mean_distance = np.where(
        found > 0,
        np.sum(np.where(valid, distances, 0.0), axis=1) / denominator,
        np.nan,
    )
    if return_counts:
        return (winner, margin.astype(np.float32), mean_distance.astype(np.float32),
                counts)
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

    # Adaptive k: tiles whose base-k vote is nearly tied get re-voted at a
    # wider neighbourhood. Refuse a setting that cannot do anything rather than
    # accepting it and producing a normal-looking CSV that quietly ignored it.
    adaptive_margin = getattr(args, "adaptive_margin", 0.0) or 0.0
    adaptive_k = getattr(args, "adaptive_k", 0) or 0
    if adaptive_margin > 0:
        if adaptive_k <= k:
            raise SystemExit(
                f"--adaptive-k {adaptive_k} is not wider than k={k}, so the "
                f"re-vote would see the same neighbours and change nothing. "
                f"Raise it above k, or drop --adaptive-margin."
            )
        if adaptive_k > len(reference):
            raise SystemExit(
                f"--adaptive-k {adaptive_k} exceeds the {len(reference)} "
                f"reference tiles."
            )
    elif adaptive_k and adaptive_k != k:
        raise SystemExit(
            f"--adaptive-k {adaptive_k} does nothing without a positive "
            f"--adaptive-margin to gate it. Set the margin, or drop the k."
        )

    # One search at the wider width, voted on a prefix. The flat-L2 scan over
    # the reference is the cost and does not depend on k -- only the top-k
    # selection does -- so searching wider once is far cheaper than searching
    # twice, and gives the base and the re-vote the same neighbours.
    k_search = max(k, adaptive_k) if adaptive_margin > 0 else k

    print(f"Reference : {len(reference):,} tiles, {reference.shape[1]} comps, "
          f"{len(categories)} clusters, k={k} ({groupby})")
    if adaptive_margin > 0:
        print(f"Adaptive  : re-vote at k={adaptive_k} below margin "
              f"{adaptive_margin:g} (searching k={k_search})")

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

    searcher = Searcher(reference, metric=args.metric, device=args.device)
    print(f"Backend   : {searcher.backend}, centering={args.centering}")

    # A property of the reference, so computed once here rather than per chunk.
    local_scale = None
    if args.local_scaling:
        local_scale = compute_local_scale(reference, r=args.local_scaling)
        print(f"Local scale : r={args.local_scaling}, "
              f"median {np.median(local_scale):.3f}")

    class_weights = None
    if args.class_weighted:
        # 1/count so a cluster with more reference points gets less weight per
        # neighbour — correcting for a large cluster winning a boundary tile by
        # being more numerous nearby, not by being genuinely closer.
        class_weights = 1.0 / np.maximum(np.bincount(codes, minlength=len(categories)), 1)

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
    n_revoted = 0
    try:
        for start, stop, block in iter_embedding_chunks(
            args.h5, args.rep_key, args.chunk_size, lo, hi
        ):
            queries = project(block, components, ref["mean"], args.centering,
                              query_mean=query_mean)
            offset = start - lo
            for bstart in range(0, len(queries), args.batch_size):
                bstop = min(bstart + args.batch_size, len(queries))
                idx, dist = searcher.search(queries[bstart:bstop], k_search)
                w, m, d = vote(idx[:, :k], dist[:, :k], codes, len(categories),
                               distance_weighted=args.distance_weighted,
                               distance_power=args.distance_power,
                               class_weights=class_weights,
                               local_scale=local_scale)
                if adaptive_margin > 0:
                    low = m < adaptive_margin
                    if low.any():
                        w2, m2, d2 = vote(
                            idx[low, :adaptive_k], dist[low, :adaptive_k],
                            codes, len(categories),
                            distance_weighted=args.distance_weighted,
                            distance_power=args.distance_power,
                            class_weights=class_weights,
                            local_scale=local_scale)
                        # The recorded margin has to describe the vote that
                        # produced the recorded label, or vote_margin measures
                        # a vote that was thrown away -- and Stage 5's
                        # --min-margin would then drop exactly the tiles this
                        # was added to rescue.
                        w, m, d = w.copy(), m.copy(), d.copy()
                        w[low], m[low], d[low] = w2, m2, d2
                        n_revoted += int(low.sum())
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
    if adaptive_margin > 0:
        print(f"Re-voted  : {n_revoted:,} tiles ({n_revoted / max(written, 1) * 100:.1f}%) "
              f"at k={adaptive_k}")
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
        "revoted": n_revoted,
    }


def _dedupe_truth(truth: pd.DataFrame, groupby: str
                  ) -> tuple[pd.DataFrame, list[tuple], int]:
    """Make (slides, tiles) unique in the truth file, and say what was dropped.

    Kai's TCGA label CSV lists 100 tiles twice, 96 of them with two *different*
    Leiden labels — all on one slide, which looks like a slide tiled twice and
    clustered independently each time. The merge below asks for one_to_one
    precisely so a duplicated key cannot silently multiply rows, so before this
    the whole acceptance test died on a pandas MergeError naming neither the file
    nor the slide.

    Dropping duplicates blindly would be worse than crashing: whichever row came
    first would become "the" truth, and 96 tiles would be scored against a label
    chosen by CSV ordering. So:

      * duplicated with the same label — redundant, keep one, nothing changes.
      * duplicated with different labels — there is no fact to be right about,
        so the tile is excluded from the comparison and counted out loud.

    Returns (unique truth, ambiguous keys, redundant row count).
    """
    key = ["slides", "tiles"]
    duplicated = truth[truth.duplicated(key, keep=False)]
    if duplicated.empty:
        return truth, [], 0

    labels_per_key = duplicated.groupby(key)[groupby].nunique()
    ambiguous_keys = labels_per_key[labels_per_key > 1].index
    ambiguous = list(ambiguous_keys)

    cleaned = truth
    if len(ambiguous):
        # An index-based anti-join, so a tile is dropped by its (slides, tiles)
        # pair rather than by position.
        flat = set(ambiguous)
        keep = ~pd.MultiIndex.from_frame(truth[key]).isin(flat)
        cleaned = truth[keep]
    redundant = int(len(cleaned) - len(cleaned.drop_duplicates(key)))
    cleaned = cleaned.drop_duplicates(key)
    return cleaned, ambiguous, redundant


def validate(frame: pd.DataFrame, truth_path: Path, groupby: str) -> bool:
    """Agreement against labels produced by the reference implementation.

    Joined on (slides, tiles), never on row order — the .h5 and the CSV are
    written by different programs and there is no guarantee they agree on it.
    """
    truth = pd.read_csv(truth_path)
    if groupby not in truth.columns:
        raise SystemExit(f"{truth_path} has no '{groupby}' column: {list(truth.columns)}")

    truth, ambiguous, redundant = _dedupe_truth(truth, groupby)

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
    if redundant:
        print(f"  {redundant:,} tile(s) listed more than once with the SAME label "
              f"— deduplicated, no effect on the number below.")
    if ambiguous:
        print(f"  {len(ambiguous):,} tile(s) listed more than once with DIFFERENT "
              f"labels — excluded, since there is no label to be right about. "
              f"Slides affected: {', '.join(sorted({s for s, _ in ambiguous})[:3])}"
              + (" ..." if len({s for s, _ in ambiguous}) > 3 else ""))
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
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"],
                        help="Where the exact flat search runs. GPU is the same "
                             "exhaustive scan on faster hardware — not an "
                             "approximation — but it needs a faiss build with "
                             "GPU support (see --bootstrap-extras-gpu). Defaults "
                             "to cpu so no existing submission changes hardware, "
                             "and refuses rather than falling back if the GPU is "
                             "unavailable.")
    parser.add_argument("--metric", default="l2", choices=["l2", "cosine"],
                        help="l2 (default): Euclidean distance, what ingest uses. "
                             "cosine: direction only, ignoring magnitude — validate "
                             "with validate_reference.py --metric cosine before using "
                             "this for a real assignment, same as k or the weighting "
                             "flags — it changes cluster labels.")
    parser.add_argument("--distance-weighted", action="store_true",
                        help="Weight each neighbour's vote by 1/distance instead of "
                             "one vote each. Validate with validate_reference.py "
                             "--distance-weighted before using this for a real "
                             "assignment — it changes cluster labels, same as k does.")
    parser.add_argument("--distance-power", type=float, default=1.0,
                        help="Exponent on the distance-weighted vote: "
                             "1/(distance+eps)**power. 2.0 makes the nearest few "
                             "neighbours matter much more. Only used with "
                             "--distance-weighted.")
    parser.add_argument("--class-weighted", action="store_true",
                        help="Scale each neighbour's vote by 1/(its cluster's "
                             "reference count), so a large cluster's neighbours don't "
                             "win a boundary tile just by being more numerous nearby. "
                             "Composes with --distance-weighted. Validate first, same "
                             "as that flag.")
    parser.add_argument("--adaptive-margin", type=float, default=0.0,
                        metavar="MARGIN",
                        help="Re-vote any tile whose base-k vote_margin falls "
                             "below this at the wider --adaptive-k. 0 (default) "
                             "disables it. Measured on the production reference "
                             "at 200,000 tiles: --k 10 --distance-power 3 with "
                             "--adaptive-margin 0.1 --adaptive-k 25 took "
                             "96.78%% to 97.23%% (1,925 fixed, 1,030 broken, "
                             "16.5 sigma by McNemar). Costs nothing extra in "
                             "search: the wider neighbours come from the same "
                             "scan, and only ~9%% of tiles are re-voted.")
    parser.add_argument("--adaptive-k", type=int, default=0, metavar="K",
                        help="The wider neighbourhood the low-margin tiles are "
                             "re-voted at. Must exceed --k; refused otherwise, "
                             "since the re-vote would see the same neighbours.")
    parser.add_argument("--local-scaling", type=int, default=0, metavar="R",
                        help="Divide each neighbour's distance by that neighbour's "
                             "own distance to its R-th nearest reference point, so "
                             "closeness is judged relative to local density rather "
                             "than on an absolute ruler. 0 (the default) is off. "
                             "Validate with validate_reference.py --local-scaling "
                             "before using it for a real assignment.")
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
