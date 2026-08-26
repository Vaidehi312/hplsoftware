"""Pre-flight and verification helpers for HPL cluster assignment.

The assignment itself is done by HPL-LATTICeA/run_representationsleiden_assignment.py.
That script derives every path by string-splitting its arguments and swallows per-fold
exceptions with a printed warning, so a wrong path exits 0 with no output file. This
script checks the inputs before you submit, and checks the output afterwards.

    python hpl_assignment_helper.py preflight \
        --adatas-dir /path/to/..._filtered/rapids_2p5m/adatas \
        --new-h5     /path/to/..._filtered/hdf5_mydataset_he_filtered.h5

    python hpl_assignment_helper.py verify \
        --assigned-csv /path/to/adatas/mydataset_he_filtered_leiden_2p5__fold2.csv \
        --new-h5       /path/to/hdf5_mydataset_he_filtered.h5
"""

import argparse
import glob
import os
import sys

import h5py

FAILED = []


def ok(msg):
    print("  [ok]   %s" % msg)


def warn(msg):
    print("  [warn] %s" % msg)


def fail(msg):
    print("  [FAIL] %s" % msg)
    FAILED.append(msg)


def h5ad_obs_categories(content, groupby):
    """Category labels for an .h5ad obs column, across anndata layouts."""
    obs = content["obs"]
    if groupby not in obs:
        return None
    node = obs[groupby]
    if isinstance(node, h5py.Group) and "categories" in node:
        return node["categories"][:]
    if "__categories" in obs and groupby in obs["__categories"]:
        return obs["__categories"][groupby][:]
    return None


def inspect_reference(h5ad_path, groupby):
    """Read the reference config's shape and check it carries what ingest needs."""
    n_vars = None
    with h5py.File(h5ad_path, "r") as content:
        # sc.tl.ingest(embedding_method='pca') needs the reference PCA basis.
        if "varm/PCs" in content:
            n_vars = content["varm/PCs"].shape[0]
            ok("varm/PCs present, %s reference dimensions" % n_vars)
        else:
            fail("varm/PCs missing - ingest cannot project onto the reference PCA")

        if "obsm/X_pca" in content:
            ok("obsm/X_pca present, %s reference tiles" % content["obsm/X_pca"].shape[0])
        else:
            fail("obsm/X_pca missing - ingest has no reference embedding to map into")

        # leiden_representations.py passes neighbors_key='nn_leiden'.
        if "uns/nn_leiden" in content:
            params = content.get("uns/nn_leiden/params")
            n_neighbors = params["n_neighbors"][()] if params and "n_neighbors" in params else "?"
            ok("uns/nn_leiden present, n_neighbors=%s" % n_neighbors)
        else:
            fail("uns/nn_leiden missing - ingest is called with neighbors_key='nn_leiden'")

        categories = h5ad_obs_categories(content, groupby)
        if categories is None:
            fail("obs/%s missing - nothing for ingest to transfer" % groupby)
        else:
            ok("obs/%s present, %s clusters" % (groupby, len(categories)))

    return n_vars


def inspect_new_h5(new_h5, rep_key, ref_n_vars):
    with h5py.File(new_h5, "r") as content:
        keys = list(content.keys())
        print("  datasets: %s" % keys)

        # representations_to_frame takes the FIRST key containing rep_key.
        rep_keys = [k for k in keys if rep_key in k]
        if not rep_keys:
            fail("no dataset containing '%s' - check --rep_key" % rep_key)
            return
        if len(rep_keys) > 1:
            warn("several %s datasets %s - the script silently takes '%s'"
                 % (rep_key, rep_keys, rep_keys[0]))

        shape = content[rep_keys[0]].shape
        ok("representations '%s' shape %s" % (rep_keys[0], shape))
        if ref_n_vars is not None and len(shape) > 1 and shape[1] != ref_n_vars:
            fail("dimension mismatch: new is %s, reference is %s - ingest requires "
                 "matching var_names" % (shape[1], ref_n_vars))

        # Every non-latent key is loaded into a DataFrame column; 4-D images break that.
        for key in keys:
            if "latent" in key:
                continue
            if content[key].ndim > 1:
                fail("dataset '%s' has shape %s - use the representations .h5 from "
                     "run_representationspathology_projection.py, not the tiles .h5"
                     % (key, content[key].shape))

        for field in ("tiles", "slides", "samples"):
            if not any(k.replace("train_", "").replace("valid_", "").replace("test_", "") == field
                       for k in keys):
                warn("no '%s' metadata - the output CSV will not carry it" % field)


def preflight(args):
    res_tag = ("%s" % args.resolution).replace(".", "p")
    groupby = "leiden_%s" % args.resolution
    suffix = "_leiden_%s__fold%s" % (res_tag, args.fold)

    adatas_dir = os.path.abspath(args.adatas_dir)
    print("\n== reference config ==")
    if not os.path.isdir(adatas_dir):
        fail("adatas dir does not exist: %s" % adatas_dir)
        return
    ok("adatas dir: %s" % adatas_dir)

    # Invert rule 2: the script builds the filename it looks for from --h5_complete_path,
    # so read the real filename and work backwards to the argument that reproduces it.
    stem = None
    for candidate in sorted(glob.glob(os.path.join(adatas_dir, "*%s*.h5ad" % suffix))):
        base = os.path.basename(candidate)
        for tail in (suffix + "_subsample.h5ad", suffix + ".h5ad"):
            if base.endswith(tail):
                stem = base[: -len(tail)]
                ok("found %s" % base)
                break
        if stem:
            reference = candidate
            break

    if stem is None:
        found = [os.path.basename(p) for p in glob.glob(os.path.join(adatas_dir, "*.h5ad"))]
        fail("no .h5ad ending in '%s[_subsample].h5ad'. Present: %s"
             % (suffix, found if found else "nothing"))
        return

    # Invert rule 1: adatas dir is <parent>/<meta_field>/adatas.
    meta_dir = os.path.dirname(adatas_dir)
    parent = os.path.dirname(meta_dir)
    if os.path.basename(meta_dir) != args.meta_field:
        fail("--meta_field must be '%s' to match the directory layout, not '%s'"
             % (os.path.basename(meta_dir), args.meta_field))
    if "hdf5_" in parent:
        fail("path contains 'hdf5_' above the filename (%s); the script's split() "
             "will mis-derive the adatas dir" % parent)

    h5_complete_path = os.path.join(parent, "hdf5_%s.h5" % stem)
    ok("--h5_complete_path must be: %s" % h5_complete_path)
    if not os.path.exists(h5_complete_path):
        ok("(that file need not exist - it is only string-split for path derivation)")

    ref_n_vars = inspect_reference(reference, groupby)

    print("\n== new representations ==")
    new_h5 = os.path.abspath(args.new_h5)
    if not os.path.isfile(new_h5):
        fail("new .h5 does not exist: %s" % new_h5)
        return
    if not os.path.basename(new_h5).startswith("hdf5_"):
        fail("filename must start with 'hdf5_' - the script does "
             "split('/hdf5_')[1] and will raise IndexError")
    inspect_new_h5(new_h5, args.rep_key, ref_n_vars)

    print("\n== result ==")
    if FAILED:
        print("  %s blocking problem(s); fix before submitting." % len(FAILED))
        return
    out_csv = os.path.join(adatas_dir, "%s%s.csv"
                           % (os.path.basename(new_h5)[len("hdf5_"):].split(".h5")[0], suffix))
    print("  All checks passed. Run, from inside HPL-LATTICeA/:\n")
    print("    python ./run_representationsleiden_assignment.py \\")
    print("      --meta_field         %s \\" % args.meta_field)
    print("      --resolution         %s \\" % args.resolution)
    print("      --rep_key            %s \\" % args.rep_key)
    print("      --folds_pickle       %s \\" % args.folds_pickle)
    print("      --h5_complete_path   %s \\" % h5_complete_path)
    print("      --h5_additional_path %s" % new_h5)
    print("\n  Warnings for the other folds are expected - the config holds fold %s only."
          % args.fold)
    print("  Output CSV: %s" % out_csv)
    print("  Then: python hpl_assignment_helper.py verify --assigned-csv <that> "
          "--new-h5 %s" % new_h5)


def verify(args):
    import pandas as pd

    groupby = "leiden_%s" % args.resolution

    print("\n== assigned output ==")
    if not os.path.isfile(args.assigned_csv):
        fail("no output CSV at %s - the script exits 0 even when every fold fails, "
             "so re-read the log for the fold %s warning" % (args.assigned_csv, args.fold))
        return
    assigned = pd.read_csv(args.assigned_csv)
    if groupby not in assigned.columns:
        fail("column '%s' missing; columns are %s" % (groupby, list(assigned.columns)))
        return
    ok("%s rows, %s columns" % (len(assigned), len(assigned.columns)))

    # Tiles are chunked at 500k and re-concatenated; a short file means a lost chunk.
    with h5py.File(args.new_h5, "r") as content:
        rep_keys = [k for k in content.keys() if args.rep_key in k]
        n_tiles = content[rep_keys[0]].shape[0] if rep_keys else None
    if n_tiles is None:
        warn("could not read tile count from %s" % args.new_h5)
    elif len(assigned) == n_tiles:
        ok("row count matches the %s tiles in the .h5" % n_tiles)
    else:
        fail("row count %s != %s tiles in the .h5 - a 500k chunk was lost"
             % (len(assigned), n_tiles))

    new_ids = set(pd.to_numeric(assigned[groupby], errors="coerce").dropna().astype(int))

    if os.path.isfile(args.reference_csv):
        print("\n== against %s ==" % os.path.basename(args.reference_csv))
        ref = pd.read_csv(args.reference_csv)
        ref_ids = set(pd.to_numeric(ref[groupby], errors="coerce").dropna().astype(int))
        extra = new_ids - ref_ids
        if extra:
            fail("cluster IDs %s absent from the reference - wrong config or resolution"
                 % sorted(extra))
        else:
            ok("all %s cluster IDs are within the reference's %s"
               % (len(new_ids), len(ref_ids)))
        if ref_ids - new_ids:
            warn("%s reference clusters unused: %s"
                 % (len(ref_ids - new_ids), sorted(ref_ids - new_ids)))

        # A collapsed or wildly different distribution means an encoder/preprocessing
        # mismatch, not a biological finding.
        new_p = assigned[groupby].value_counts(normalize=True)
        ref_p = ref[groupby].value_counts(normalize=True)
        shared = new_p.index.intersection(ref_p.index)
        drift = (new_p[shared] - ref_p[shared]).abs().sum() / 2
        print("  total variation distance vs reference: %.3f" % drift)
        if drift > 0.35:
            warn("distributions differ substantially - check checkpoint, magnification "
                 "(5x / 1.8 um-per-px) and tile size (224) match the original run")
        else:
            ok("cluster proportions are broadly comparable")

        top = new_p.head(3)
        print("  largest clusters: %s"
              % ", ".join("%s=%.1f%%" % (i, v * 100) for i, v in top.items()))
        if top.iloc[0] > 0.5:
            warn("cluster %s holds %.0f%% of tiles - suspicious collapse"
                 % (top.index[0], top.iloc[0] * 100))
    else:
        warn("reference CSV not found at %s - skipped comparison" % args.reference_csv)

    print("\n== result ==")
    if FAILED:
        print("  %s problem(s) found." % len(FAILED))
    else:
        print("  Assignment looks sound. Spot-check tiles per cluster with")
        print("  HPL-LATTICeA/utilities/visualizations/cluster_images.py before relying on it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--resolution", default="2.5", help="Leiden resolution (default 2.5).")
    common.add_argument("--fold", default="2", help="Fold held by the config (default 2).")
    common.add_argument("--rep-key", default="z_latent", help="Representation key.")

    p = sub.add_parser("preflight", parents=[common], help="Check inputs before submitting.")
    p.add_argument("--adatas-dir", required=True, help="Directory holding the reference .h5ad.")
    p.add_argument("--new-h5", required=True, help="New representations .h5 to assign.")
    p.add_argument("--meta-field", default="rapids_2p5m", dest="meta_field")
    p.add_argument("--folds-pickle", default="./utilities/fold_creation/lattice_5x_folds.pkl")
    p.set_defaults(func=preflight)

    v = sub.add_parser("verify", parents=[common], help="Check the assignment output.")
    v.add_argument("--assigned-csv", required=True, help="CSV written into adatas/.")
    v.add_argument("--new-h5", required=True, help="The .h5 that was assigned.")
    v.add_argument("--reference-csv",
                   default="TCGA_LUAD_5x_he_train_filtered_leiden_2p5__fold2.csv",
                   help="Existing assignments to compare distributions against.")
    v.set_defaults(func=verify)

    args = parser.parse_args()
    args.func(args)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
