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
    args = parser.parse_args()

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
    save(artifact, args.out)

    print(f"Reference written to {args.out}")
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
