"""Extract the HPL cluster reference from a Leiden .h5ad into a compact .npz.

The .h5ad in <results>/<meta_field>/adatas/ is what defines the HPCs: it holds
the reference embedding, the PCA basis, and each reference tile's Leiden label.
sc.tl.ingest reads it to assign new tiles (see
HPL-LATTICeA/models/clustering/leiden_representations.py:244). assign_hpc_clusters.py
reproduces that vote, and this script gives it everything it needs in one file.

Read with h5py rather than anndata deliberately:

  * .h5ad is plain HDF5, so the fields needed here are directly addressable.
    With --embedding-space pca (the default), `X` — the largest array in the
    file — is never touched.
  * It removes anndata from the runtime dependencies. The two requirement files
    in this repo pin 0.8.0 and 0.10.4, whose on-disk categorical layouts differ,
    and reading the fields directly is what lets one script handle both.

--embedding-space {pca,raw} controls what goes into the reference:

  pca   (default) obsm/X_pca, 127-d, with varm/PCs as the projection basis.
        This is what sc.tl.ingest actually clusters on.
  raw   the encoder's own 128-d z_latent, straight from X, skipping the PCA
        step entirely. components comes back as an identity matrix so
        project()/Searcher/vote() in assign_hpc_clusters.py need no changes —
        a raw embedding "projects" into itself. There is no PCA mean to
        subtract for this space, so the .npz carries no `mean` key; assigning
        against it for real needs --centering none.

  The two are comparable with validate_reference.py against the *same*
  --groupby, since leave-one-out only exercises the neighbour search and vote
  in whichever space the reference vectors are in — it never calls project().

Usage:
    python build_hpc_reference.py \
        --h5ad   <DIR>/rapids_2p5m/adatas/<stem>_leiden_2p5__fold2.h5ad \
        --out    hpc_reference_leiden_2p5_fold2.npz

    python build_hpc_reference.py \
        --h5ad   <DIR>/rapids_2p5m/adatas/<stem>_leiden_2p5__fold2.h5ad \
        --out    hpc_reference_leiden_2p5_fold2_raw128.npz \
        --embedding-space raw
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

# Where the reference artifact lives, following the same
# Path(os.getenv(NAME, default)) convention as TISSUE_MASK_DIR /
# PROCESSED_TILES_DIR / HPL_DATASETS_ROOT in tile_server_v2_.py.
#
# Defined here — in the script that *writes* it — and imported by
# assign_hpc_clusters.py, rather than defaulted separately in each. Two
# independent defaults for one file is how the assigner ends up silently
# reading a stale reference while the builder writes a fresh one somewhere
# else, and cluster IDs from two different references are indistinguishable
# once they are in the registry.
HPC_REFERENCE_PATH = Path(os.getenv(
    "HPC_REFERENCE_PATH",
    str(Path(__file__).resolve().parent / "hpc_reference_leiden_2p5_fold2.npz"),
))


def _decode(values) -> np.ndarray:
    """HDF5 string arrays come back as bytes; categories must be real strings."""
    out = []
    for value in values:
        out.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return np.asarray(out, dtype=object)


def read_categorical(content: h5py.File, column: str) -> tuple[np.ndarray, np.ndarray]:
    """(codes, categories) for an obs column, across anndata's layouts.

    anndata >=0.7 stores a categorical as a group holding 'codes' and
    'categories'; older files keep the codes as a bare dataset under obs and the
    labels in a shared obs/__categories group. Both appear in this project's
    pinned versions, and which one a reference config uses is a property of
    whoever wrote it, not of the code reading it.
    """
    obs = content["obs"]
    if column not in obs:
        available = sorted(obs.keys())
        raise KeyError(f"obs/{column} not found. Available obs columns: {available}")

    node = obs[column]
    if isinstance(node, h5py.Group):
        if "codes" not in node or "categories" not in node:
            raise KeyError(f"obs/{column} is a group without codes/categories")
        return node["codes"][:], _decode(node["categories"][:])

    codes = node[:]
    legacy = obs.get("__categories")
    if legacy is not None and column in legacy:
        return codes, _decode(legacy[column][:])

    # Not categorical at all — a plain integer or string column of labels. Build
    # the category table so downstream code has one representation to handle.
    categories, inverse = np.unique(_decode(codes), return_inverse=True)
    return inverse, categories


def read_slide_names(h5ad_path: Path, column: str = "slides") -> np.ndarray:
    """One slide name per reference row, for splitting the reference by slide."""
    with h5py.File(h5ad_path, "r") as content:
        codes, categories = read_categorical(content, column)
    codes = np.asarray(codes, dtype=np.int64)
    if (codes < 0).any():
        raise SystemExit(
            f"obs/{column} has {int((codes < 0).sum()):,} missing values. A tile "
            f"with no slide can be neither held out nor deliberately kept, and "
            f"guessing either way risks leaving part of a held-out slide in the "
            f"reference — which is the one thing this split exists to prevent."
        )
    return np.asarray(categories, dtype=object)[codes]


def hold_out_slides(artifact: dict, slide_names: np.ndarray, n_slides: int,
                    seed: int = 0) -> tuple[dict, dict]:
    """Split whole slides out of a reference, for a closed-book accuracy test.

    Whole slides, not random tiles, and that is the entire point. Leave-one-out
    removes one tile and leaves its ~5,000 slide-mates in the reference — same
    scanner, same staining, often contiguous tissue — so a tile is largely
    classified by near-duplicates of itself. The resulting number is optimistic
    by an unknown amount relative to what a genuinely new slide would score,
    which is the thing anyone actually wants to know before running a new cohort.

    Returns (reduced reference, holdout). Three things are deliberately NOT
    recomputed for the reduced side:

      categories  the codes index this table, so renumbering it would silently
                  reassign every cluster ID on both sides of the split.
      mean        the PCA mean is a property of the original fit, not of the rows
                  kept. Recomputing it from a subset would move the space the
                  queries are already expressed in.
      n_neighbors a property of the graph the reference was clustered on.
    """
    rows = int(artifact["reference"].shape[0])
    if len(slide_names) != rows:
        raise SystemExit(
            f"{len(slide_names):,} slide names for {rows:,} reference rows — "
            f"these do not describe the same .h5ad."
        )

    unique = np.unique(slide_names)          # sorted, so the seed alone decides
    if n_slides < 1:
        raise SystemExit("--holdout-slides must be at least 1.")
    if n_slides >= len(unique):
        raise SystemExit(
            f"--holdout-slides {n_slides} of only {len(unique)} slides would "
            f"leave nothing to classify against."
        )

    rng = np.random.default_rng(seed)
    held = np.sort(rng.choice(unique, size=n_slides, replace=False))
    in_holdout = np.isin(slide_names, held)
    keep = ~in_holdout
    if not in_holdout.any() or not keep.any():
        raise SystemExit("The split left one side empty.")

    reduced = dict(artifact)
    reduced["reference"] = artifact["reference"][keep]
    reduced["codes"] = np.asarray(artifact["codes"])[keep]
    reduced["holdout_slides"] = [str(s) for s in held]

    holdout = {
        "queries": artifact["reference"][in_holdout],
        "codes": np.asarray(artifact["codes"])[in_holdout],
        "slides": np.asarray([str(s) for s in slide_names[in_holdout]]),
        "categories": np.asarray([str(c) for c in artifact["categories"]]),
        "groupby": artifact["groupby"],
        "source": artifact["source"],
        "held_slides": [str(s) for s in held],
        "reference_rows": int(reduced["reference"].shape[0]),
    }

    # A cluster that exists only on a held-out slide is now unreachable: no
    # query can be assigned to it, so every one of its held-out tiles is wrong
    # by construction and the accuracy figure carries that as if it were a
    # classifier error. Reported rather than silently absorbed.
    kept_clusters = set(np.unique(reduced["codes"]).tolist())
    lost = sorted(set(np.unique(holdout["codes"]).tolist()) - kept_clusters)
    holdout["unreachable_clusters"] = lost
    return reduced, holdout


def save_holdout(holdout: dict, out_path: Path) -> None:
    """The held-out tiles, as their own file.

    Not extra keys on the reference .npz: readers there are pinned by
    test_reference_keys_match_the_builder precisely so they cannot drift, and a
    reference carrying its own test set is a reference somebody will eventually
    assign against by accident.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        queries=holdout["queries"],
        codes=np.asarray(holdout["codes"]),
        slides=holdout["slides"],
        categories=holdout["categories"],
        meta=np.asarray(json.dumps({
            "groupby": holdout["groupby"],
            "source": holdout["source"],
            "held_slides": holdout["held_slides"],
            "n_held_slides": len(holdout["held_slides"]),
            "n_queries": int(holdout["queries"].shape[0]),
            "n_clusters": int(len(holdout["categories"])),
            "reference_rows": holdout["reference_rows"],
            "unreachable_clusters": holdout["unreachable_clusters"],
        })),
    )


def _read_raw_embedding(content: h5py.File) -> np.ndarray:
    if "X" not in content:
        raise KeyError("X missing — no raw embedding to read.")
    node = content["X"]
    if isinstance(node, h5py.Group):
        raise KeyError(
            "X is a group, not a dataset — this .h5ad stores X sparse (typical "
            "of raw counts, not a dense self-supervised latent). --embedding-space "
            "raw only handles a dense X; add scipy.sparse reconstruction here if "
            "this reference genuinely needs it."
        )
    return np.asarray(node[:], dtype=np.float32)


def build(h5ad_path: Path, groupby: str, embedding_space: str = "pca") -> dict:
    with h5py.File(h5ad_path, "r") as content:
        mean = None
        if embedding_space == "pca":
            if "obsm/X_pca" not in content:
                raise KeyError(
                    "obsm/X_pca missing — this .h5ad was not written with a PCA "
                    "embedding, so there is no space to project queries into."
                )
            if "varm/PCs" not in content:
                raise KeyError(
                    "varm/PCs missing — without the PCA basis, new embeddings cannot "
                    "be placed in the same space the reference was clustered in."
                )

            reference = np.asarray(content["obsm/X_pca"][:], dtype=np.float32)
            # varm/PCs is (n_vars, n_comps): the matrix raw embeddings multiply by.
            components = np.asarray(content["varm/PCs"][:], dtype=np.float32)
        elif embedding_space == "raw":
            reference = _read_raw_embedding(content)
            # Identity, not the PCA basis: a raw embedding is already in its own
            # space, so project()'s `embeddings @ components` must be a no-op.
            components = np.eye(reference.shape[1], dtype=np.float32)
        else:
            raise ValueError(f"Unknown embedding_space: {embedding_space!r}")

        codes, categories = read_categorical(content, groupby)

        # k is a property of the graph the reference was clustered on, not a
        # tuning knob — a vote over a different neighbourhood size is a different
        # function. sc.pp.neighbors records it here, so read it rather than
        # defaulting; a config without it is one whose clustering cannot be
        # faithfully reproduced, and guessing would hide that.
        params = content.get("uns/nn_leiden/params")
        if params is None or "n_neighbors" not in params:
            raise KeyError(
                "uns/nn_leiden/params/n_neighbors missing — cannot know the "
                "neighbourhood size this reference was built with."
            )
        n_neighbors = int(np.asarray(params["n_neighbors"]).reshape(-1)[0])

        # The mean subtracted before projection. scanpy's PCA centres by default
        # and does not always persist the mean; when it is absent the reference
        # coordinates themselves are the ground truth used to recover it below.
        # Only applies to the pca space — a raw z_latent was never centred by
        # scanpy's PCA step, so there is nothing to look up here.
        if embedding_space == "pca":
            for key in ("uns/pca/mean", "varm/mean", "uns/pca/mean_"):
                if key in content:
                    mean = np.asarray(content[key][:], dtype=np.float32).reshape(-1)
                    break

    embedding_key = "obsm/X_pca" if embedding_space == "pca" else "X"
    if reference.shape[0] != codes.shape[0]:
        raise ValueError(
            f"{embedding_key} has {reference.shape[0]} rows but obs/{groupby} has "
            f"{codes.shape[0]} — the reference and its labels disagree."
        )
    if components.shape[1] != reference.shape[1]:
        raise ValueError(
            f"components gives {components.shape[1]} dimensions but {embedding_key} "
            f"has {reference.shape[1]} dimensions."
        )

    return {
        "reference": reference,
        "components": components,
        "codes": np.asarray(codes, dtype=np.int32),
        "categories": np.asarray(categories, dtype=object),
        "n_neighbors": n_neighbors,
        "mean": mean,
        "groupby": groupby,
        "source": str(h5ad_path),
        "embedding_space": embedding_space,
    }


def save(artifact: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference": artifact["reference"],
        "components": artifact["components"],
        "codes": artifact["codes"],
        # Saved as plain strings so the .npz loads without allow_pickle, which
        # assign_hpc_clusters.py would otherwise have to enable to read labels.
        "categories": np.asarray([str(c) for c in artifact["categories"]]),
        "n_neighbors": np.asarray(artifact["n_neighbors"]),
        "meta": np.asarray(
            json.dumps(
                {
                    "groupby": artifact["groupby"],
                    "source": artifact["source"],
                    "n_reference": int(artifact["reference"].shape[0]),
                    "n_components": int(artifact["reference"].shape[1]),
                    "n_vars": int(artifact["components"].shape[0]),
                    "n_clusters": int(len(artifact["categories"])),
                    "has_mean": artifact["mean"] is not None,
                    "embedding_space": artifact.get("embedding_space", "pca"),
                    # Present only on a slide-holdout split. Without it such a
                    # reference is indistinguishable from production — same
                    # groupby, same 71 clusters, just fewer rows — and would be
                    # reported as production by describe_reference while
                    # missing whole slides.
                    **({"holdout_slides": artifact["holdout_slides"]}
                       if artifact.get("holdout_slides") else {}),
                }
            )
        ),
    }
    if artifact["mean"] is not None:
        payload["mean"] = artifact["mean"]
    np.savez(out_path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--h5ad", required=True, type=Path, help="Reference .h5ad.")
    parser.add_argument(
        "--out", type=Path, default=HPC_REFERENCE_PATH,
        help=f"Output .npz. Defaults to HPC_REFERENCE_PATH ({HPC_REFERENCE_PATH}), "
             f"which is also where assign_hpc_clusters.py looks by default.",
    )
    parser.add_argument(
        "--groupby",
        default=None,
        help="obs column holding the labels. Defaults to the resolution parsed "
             "from the filename (e.g. ..._leiden_2p5__fold2.h5ad -> leiden_2.5).",
    )
    parser.add_argument(
        "--embedding-space", default="pca", choices=["pca", "raw"],
        help="pca (default): obsm/X_pca, the space sc.tl.ingest actually "
             "clustered on. raw: the encoder's own z_latent from X, skipping "
             "PCA — comparable to 'pca' via validate_reference.py on the same "
             "--groupby, since leave-one-out never calls project().",
    )
    parser.add_argument(
        "--holdout-slides", type=int, default=0, metavar="N",
        help="Build a reference with N whole slides removed, and write those "
             "slides' tiles to --holdout-out as a held-out test set. Whole "
             "slides rather than random tiles: leave-one-out leaves a tile's "
             "~5,000 slide-mates in the reference, so it measures recovery from "
             "near-duplicates of itself and overstates what a genuinely new "
             "slide would score. 0 (default) builds the full reference.",
    )
    parser.add_argument(
        "--holdout-out", type=Path, default=None,
        help="Where the held-out tiles go. Defaults to the reference's own name "
             "with '_holdout' appended. A separate file, not extra keys on the "
             "reference: a reference carrying its own test set is one somebody "
             "eventually assigns against by accident.",
    )
    parser.add_argument(
        "--holdout-seed", type=int, default=0,
        help="Which slides get held out. Report it with the result — a single "
             "20-slide draw is one sample, not a population.",
    )
    parser.add_argument(
        "--holdout-column", default="slides",
        help="obs column naming each tile's slide.",
    )
    args = parser.parse_args()

    if args.holdout_out and not args.holdout_slides:
        raise SystemExit(
            "--holdout-out does nothing without --holdout-slides. Set the "
            "number of slides to hold out, or drop the path."
        )

    groupby = args.groupby
    if groupby is None:
        # Filenames carry the resolution with '.' replaced by 'p' (see
        # leiden_representations.py:290), so invert that to get the obs column.
        stem = args.h5ad.name
        marker = "_leiden_"
        if marker not in stem:
            raise SystemExit(
                f"Cannot infer --groupby from '{stem}' — pass it explicitly."
            )
        resolution = stem.split(marker)[1].split("__")[0]
        groupby = "leiden_" + resolution.replace("p", ".")

    artifact = build(args.h5ad, groupby, embedding_space=args.embedding_space)

    holdout = None
    if args.holdout_slides:
        full_rows = int(artifact["reference"].shape[0])
        slide_names = read_slide_names(args.h5ad, args.holdout_column)
        artifact, holdout = hold_out_slides(
            artifact, slide_names, args.holdout_slides, args.holdout_seed)
        holdout_path = args.holdout_out or args.out.with_name(
            f"{args.out.stem}_holdout{args.out.suffix}")
        save_holdout(holdout, holdout_path)

    save(artifact, args.out)

    if holdout is not None:
        print(f"Holdout written to {holdout_path}")
        print(f"  slides held out {len(holdout['held_slides'])} "
              f"(seed {args.holdout_seed})")
        print(f"  tiles held out  {holdout['queries'].shape[0]:,} of "
              f"{full_rows:,} ({holdout['queries'].shape[0] / full_rows * 100:.1f}%)")
        print(f"  first few       {', '.join(holdout['held_slides'][:3])}"
              + (" ..." if len(holdout["held_slides"]) > 3 else ""))
        if holdout["unreachable_clusters"]:
            # These tiles cannot be got right by any classifier now, so the
            # accuracy figure would carry them as errors. Said here, before the
            # number exists, rather than left to explain it afterwards.
            print(f"  WARNING: cluster(s) {holdout['unreachable_clusters']} exist "
                  f"only on held-out slides, so no query can be assigned to them "
                  f"and their held-out tiles are wrong by construction. Consider "
                  f"another --holdout-seed.")
        print()

    print(f"Reference written to {args.out}")
    if holdout is not None:
        print(f"  *** SLIDE-HOLDOUT REFERENCE — not for real assignments ***")
    print(f"  embedding space {artifact['embedding_space']}")
    print(f"  labels from     obs/{groupby}")
    print(f"  reference tiles {artifact['reference'].shape[0]:,}")
    print(f"  dimensions      {artifact['reference'].shape[1]}")
    print(f"  input dims      {artifact['components'].shape[0]}")
    print(f"  clusters        {len(artifact['categories'])}")
    print(f"  n_neighbors     {artifact['n_neighbors']}")
    print(f"  stored mean     {'yes' if artifact['mean'] is not None else 'no (recovered at assign time)'}")


if __name__ == "__main__":
    main()
