#!/usr/bin/env python3
"""Submit a Slurm job that assigns HPL cluster IDs to a run's tile embeddings.

Stage 3. Takes the projections .h5 that feature extraction produced, runs
assign_hpc_clusters.py against the reference built by build_hpc_reference.py,
and writes a per-tile CSV of cluster IDs with two confidence columns.

No GPU. faiss's k-NN search is CPU work and the reference is a few hundred MB,
so this asks for cores and memory instead — which also means it can run on a
CPU partition while the GPU queue is busy.

Two things are worth knowing before reading further.

The reference is not optional and not inferable. Cluster IDs mean nothing
except relative to one reference (one Leiden run, one fold) plus the encoder
checkpoint the embeddings came from. Assigning against the wrong reference
produces a complete, well-formed CSV of IDs that silently do not correspond to
the hpc_dictionary the UI joins against, so the reference is checked at submit
time and recorded with the results.

And unlike feature extraction, this stage is cheap to redo — it reads
embeddings rather than images, and takes minutes rather than hours. So it does
not go to the lengths extraction does to avoid recomputation; it overwrites its
own output on request instead.

Usage:
    python submit_cluster_assignment.py
        --projections-h5 <results>/BarlowTwins_3/DS/h224_w224_n3_zdim128/hdf5_DS_he_train.h5
        --out /path/to/DS_hpc_assignments.csv
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import h5py

from submit_feature_extraction import (
    CONTAINER_EXTRAS,
    shard_ranges,
    MERGE_PARTITION,
    SINGULARITY_BIN,
    SINGULARITY_IMAGE,
    _bind_args,
    _check_container_extras,
    _check_singularity_image,
)
from submit_mask_tile_slurm import _run_sbatch_with_retry

ASSIGN_SCRIPT = "assign_hpc_clusters.py"

# Default location of the reference artifact, matching the constant
# build_hpc_reference.py writes to. Imported lazily in _reference_path()
# because that module is a sibling script, not a package, and a hard import at
# module scope would make this file unimportable wherever it is absent.
_REFERENCE_ENV = "HPC_REFERENCE_PATH"

# Modules assign_hpc_clusters.py needs at runtime. faiss is deliberately not in
# here: the script falls back to an exact NumPy path without it, more slowly
# but with identical results, and refusing to run would turn an optimisation
# into a hard dependency. Whether it was actually used is reported by the job.
_REQUIRED_MODULES = ("numpy", "pandas", "h5py")

# Exactly what build_hpc_reference.save() writes. Listed rather than guessed:
# an earlier version of this check looked for a "labels" key that the builder
# has never written, so a perfectly good reference was rejected as malformed.
# test_reference_keys_match_the_builder keeps the two in step by round-tripping
# through the real save().
_REFERENCE_KEYS = {"reference", "components", "codes", "categories", "n_neighbors", "meta"}


def _reference_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    env = os.getenv(_REFERENCE_ENV)
    if env:
        return Path(env)
    try:
        from build_hpc_reference import HPC_REFERENCE_PATH
        return HPC_REFERENCE_PATH
    except ImportError:
        return Path(__file__).resolve().with_name("hpc_reference_leiden_2p5_fold2.npz")


def check_reference(reference: Path) -> dict:
    """Refuse to submit against a reference that is missing or not one.

    Returns what the .npz says about itself, so the submitter can record which
    reference produced an assignment. Cluster IDs from two references are
    indistinguishable once they are in the registry, which is the whole reason
    tile_registry has an hpc_reference column.
    """
    if not reference.is_file():
        raise FileNotFoundError(
            f"HPC reference not found: {reference}. Build it once from the Leiden "
            f".h5ad, e.g.\n"
            f"  python build_hpc_reference.py \\\n"
            f"      --h5ad '<...>/cluster reference/LATTICeA_5x_he_complete_surv_sex"
            f"_filtered_leiden_2p5__fold2_subsample.h5ad' \\\n"
            f"      --out {reference}\n"
            f"Or set {_REFERENCE_ENV} to an existing one."
        )

    # The .h5ad is the *source* for the reference, not the reference. Pointing
    # at it is the natural mistake — it is the file you have, its name contains
    # "leiden_2p5__fold2", and the UI asks for a path. numpy's own complaint
    # ("This file contains pickled (object) data") describes the byte format it
    # failed to parse and says nothing about the missing build step, so the case
    # is detected here and named.
    with reference.open("rb") as fh:
        magic = fh.read(8)
    if magic == b"\x89HDF\r\n\x1a\n" or reference.suffix == ".h5ad":
        raise ValueError(
            f"{reference} is an HDF5/.h5ad file, not a reference .npz. The .h5ad is "
            f"what the reference is *built from* — convert it once:\n"
            f"  python build_hpc_reference.py \\\n"
            f"      --h5ad {reference} \\\n"
            f"      --out {_reference_path().name}\n"
            f"then leave the reference field blank to use it, or give the .npz path."
        )

    import json

    import numpy as np
    try:
        with np.load(reference, allow_pickle=False) as npz:
            keys = set(npz.files)
            missing = _REFERENCE_KEYS - keys
            if missing:
                raise ValueError(
                    f"{reference} is missing {sorted(missing)} (it has {sorted(keys)}) "
                    f"— it does not look like a build_hpc_reference.py artifact. "
                    f"Rebuild it."
                )
            meta = {}
            try:
                meta = json.loads(str(npz["meta"]))
            except (ValueError, TypeError):
                pass
            info = {
                "reference_path": str(reference),
                "reference_rows": int(npz["reference"].shape[0]),
                "reference_dims": int(npz["reference"].shape[1]),
                "n_clusters": int(len(npz["categories"])),
                "groupby": meta.get("groupby"),
                "k": int(npz["n_neighbors"]),
                # No stored mean means --centering reference is unavailable, so
                # sharding has to go through --query-mean rather than the
                # shard-independent reference frame. Worth surfacing, since the
                # alternative is finding out from a failed job.
                "has_mean": "mean" in keys,
            }
    except (OSError, ValueError) as e:
        if isinstance(e, ValueError) and "does not look like" in str(e):
            raise
        raise ValueError(f"Could not read {reference} as an .npz: {e}") from e
    return info


def check_projections(path: Path, rep_key: str = "z_latent") -> int:
    """Confirm the input holds embeddings of the expected kind, and count them.

    Named datasets rather than any-h5-will-do because the failure otherwise
    lands inside assign_hpc_clusters.py after a queue wait, reported as a
    KeyError against a name the reader has no reason to recognise.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Projections file not found: {path}. Feature extraction (Stage 2) "
            f"produces this; check it finished."
        )
    try:
        with h5py.File(path, "r") as f:
            matches = [k for k in f.keys() if k.endswith(rep_key)]
            if not matches:
                raise KeyError(
                    f"{path} has no dataset ending in '{rep_key}' (found: "
                    f"{sorted(f.keys())}). This is the encoder's output file, so "
                    f"either extraction wrote something unexpected or this is the "
                    f"packaged input .h5 rather than the projections."
                )
            rows = int(f[matches[0]].shape[0])
    except OSError as e:
        raise ValueError(f"Could not open {path} as HDF5: {e}") from e
    if rows == 0:
        raise ValueError(f"{path} holds zero embeddings — nothing to assign.")
    return rows


def _import_check_python(reference_in_job: str) -> str:
    """Import probe the job runs before touching the reference.

    Same reasoning as the extraction job's: these are container-provided, and
    finding out after the queue wait that pandas is absent costs an allocation
    to learn a one-line fact.
    """
    return (
        "import sys\n"
        "missing = []\n"
        f"for name in {list(_REQUIRED_MODULES)!r}:\n"
        "    try:\n"
        "        __import__(name)\n"
        "    except ImportError as e:\n"
        "        missing.append(f'{name} ({e})')\n"
        "if missing:\n"
        "    print('FATAL: packages missing inside the container: ' + ', '.join(missing),\n"
        "          file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "try:\n"
        "    import faiss\n"
        "    print('faiss:', faiss.__version__ if hasattr(faiss, '__version__') else 'present',\n"
        "          flush=True)\n"
        "except ImportError:\n"
        "    print('faiss: not installed - falling back to the exact NumPy path, '\n"
        "          'which is slower but returns identical assignments.', flush=True)\n"
        "print('container packages: ok', flush=True)\n"
    )


def _build_assignment_command(
    *,
    singularity_bin: str,
    singularity_image: Path,
    extras_dir: Path,
    assign_script: Path,
    reference: Path,
    projections_h5: Path,
    out_csv: Path,
    rep_key: str,
    k: int | None,
    backend: str,
    batch_size: int,
    validate_against: Path | None,
    query_mean: Path | None = None,
    shard_bounds: list[tuple[int, int]] | None = None,
) -> str:
    """Shell command the Slurm --wrap runs.

    No --nv: this is CPU work, and requesting the GPU runtime for it would put
    the job behind every GPU job in the queue for no benefit.
    """
    paths = [assign_script.parent, reference.parent, projections_h5, out_csv.parent,
             singularity_image, extras_dir]
    if validate_against is not None:
        paths.append(validate_against.parent)
    binds = _bind_args(*paths)

    real = os.path.realpath
    args = [
        f"--reference {shlex.quote(real(reference))}",
        f"--h5 {shlex.quote(real(projections_h5))}",
        f"--out {shlex.quote(real(out_csv))}",
        f"--rep-key {shlex.quote(rep_key)}",
        f"--backend {shlex.quote(backend)}",
        f"--batch-size {batch_size}",
        # Progress every N tiles, so a long run is visibly alive in the log
        # rather than silent until it finishes.
        "--progress 50000",
    ]
    if k is not None:
        args.append(f"--k {k}")
    if validate_against is not None:
        args.append(f"--validate-against {shlex.quote(real(validate_against))}")

    # Sharding. The mean is supplied rather than computed per task because
    # --centering query centres on the mean of *all* queries: a task that
    # computed its own would project into a slightly different space and emit a
    # well-formed CSV of different cluster IDs. assign_hpc_clusters.py refuses
    # the combination outright, so this is belt and braces on a guard that
    # already exists.
    shard_preamble = ""
    if query_mean is not None:
        args.append(f"--query-mean {shlex.quote(real(query_mean))}")
    if shard_bounds is not None:
        starts = " ".join(str(lo) for lo, _ in shard_bounds)
        stops = " ".join(str(hi) for _, hi in shard_bounds)
        shard_preamble = (
            f"SHARD_STARTS=({starts}); SHARD_STOPS=({stops}); "
            'ROW_START="${SHARD_STARTS[$SLURM_ARRAY_TASK_ID]}"; '
            'ROW_STOP="${SHARD_STOPS[$SLURM_ARRAY_TASK_ID]}"; '
            'echo "=== Shard $SLURM_ARRAY_TASK_ID: rows $ROW_START-$ROW_STOP ==="; '
        )
        args.append("--row-start $ROW_START --row-stop $ROW_STOP")

    inner = (
        "set -euo pipefail; "
        f"export PYTHONPATH={shlex.quote(real(extras_dir))}${{PYTHONPATH:+:$PYTHONPATH}}; "
        'export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig-$$"; mkdir -p "$MPLCONFIGDIR"; '
        # Single-threaded BLAS. faiss and numpy both spawn threads sized to the
        # machine, not to the cpuset Slurm gave us, and oversubscribing a
        # shared node is slower than the serial path as well as antisocial.
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"; '
        'export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"; '
        'export MKL_NUM_THREADS="$OMP_NUM_THREADS"; '
        "echo '=== Container packages ==='; "
        f"python -c {shlex.quote(_import_check_python(real(reference)))}; "
        f"{shard_preamble}"
        "echo '=== Cluster assignment ==='; "
        f"python {shlex.quote(real(assign_script))} {' '.join(args)}"
    )
    return " ".join([
        shlex.quote(singularity_bin), "exec", "--cleanenv", *binds,
        shlex.quote(str(singularity_image)), "bash", "-lc", shlex.quote(inner),
    ])


def _build_simple_command(
    *,
    singularity_bin: str,
    singularity_image: Path,
    extras_dir: Path,
    script: Path,
    args: list[str],
    extra_binds: list[Path],
    banner: str,
) -> str:
    """One-liner container invocation, shared by the mean and merge jobs.

    Both are small, CPU-only, single-purpose steps around the array; giving each
    its own bespoke builder would be three near-identical functions.
    """
    binds = _bind_args(script.parent, extras_dir, singularity_image, *extra_binds)
    real = os.path.realpath
    inner = (
        "set -euo pipefail; "
        f"export PYTHONPATH={shlex.quote(real(extras_dir))}${{PYTHONPATH:+:$PYTHONPATH}}; "
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"; '
        f"echo '=== {banner} ==='; "
        f"python {shlex.quote(real(script))} {' '.join(args)}"
    )
    return " ".join([
        shlex.quote(singularity_bin), "exec", "--cleanenv", *binds,
        shlex.quote(str(singularity_image)), "bash", "-lc", shlex.quote(inner),
    ])


def submit_cluster_assignment_job(
    projections_h5: Path,
    out_csv: Path,
    *,
    reference: Path | None = None,
    depends_on_job_id: str | None = None,
    rep_key: str = "z_latent",
    k: int | None = None,
    backend: str = "auto",
    batch_size: int = 16_384,
    validate_against: Path | None = None,
    shards: int = 1,
    centering: str = "query",
    partition: str = MERGE_PARTITION,
    cpus: int = 16,
    memory: str = "64G",
    time_limit: str = "04:00:00",
    job_name: str = "hpl_cluster_assign",
    notify_email: str | None = None,
    singularity_image: Path = SINGULARITY_IMAGE,
    singularity_bin: str = SINGULARITY_BIN,
    extras_dir: Path = CONTAINER_EXTRAS,
    overwrite: bool = False,
) -> dict:
    """Submit Stage 3 for one projections file.

    depends_on_job_id chains this behind extraction (or behind a shard merge),
    so the whole pipeline can be queued in one go. afterok rather than afterany:
    assigning clusters to a projections file that extraction failed to finish
    would read zero-filled rows and produce confident-looking nonsense.
    """
    reference = _reference_path(reference)
    reference_info = check_reference(reference)

    # Skipped when chained: the file will not exist yet, because the job that
    # writes it has not run. The dependency is what guarantees it later.
    rows = None
    if depends_on_job_id is None:
        rows = check_projections(projections_h5, rep_key)

    if out_csv.exists() and not overwrite:
        raise FileExistsError(
            f"{out_csv} already exists. This stage is cheap to redo — delete it, or "
            f"resubmit with overwrite enabled, if you mean to replace it."
        )

    _check_singularity_image(singularity_image, singularity_bin)
    _check_container_extras(extras_dir, singularity_image, singularity_bin)

    if shards > 1 and depends_on_job_id is not None:
        # The array size has to be known at sbatch time, and it comes from the
        # projections file's row count — which does not exist yet when this is
        # chained behind extraction. Refused rather than guessed.
        raise ValueError(
            "Cannot combine --shards with --depends-on-job-id: the number of rows "
            "to split is read from the projections file, which the job it depends "
            "on has not written yet. Either submit unsharded and chained, or wait "
            "for extraction to finish and then submit sharded."
        )

    script_path = Path(__file__).resolve()
    backend_dir = script_path.parent
    log_dir = backend_dir / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- sharding -----------------------------------------------------------
    # Three jobs rather than one: compute the query mean over the whole file,
    # then an array of assign tasks all handed that same mean, then a merge.
    #
    # The mean job exists solely because --centering query centres on the mean of
    # every query. Letting each task compute its own would put each shard in a
    # slightly different space and produce a well-formed CSV of different cluster
    # IDs — the one failure here that no downstream check would catch.
    shard_bounds = None
    mean_path = None
    mean_job_id = None
    if shards > 1:
        shard_bounds = shard_ranges(rows, shards)
        mean_path = out_csv.with_name(f"{out_csv.stem}.query_mean.npy")

        if centering == "query":
            mean_command = _build_simple_command(
                singularity_bin=singularity_bin, singularity_image=singularity_image,
                extras_dir=extras_dir, script=backend_dir / ASSIGN_SCRIPT,
                args=[
                    f"--reference {shlex.quote(os.path.realpath(reference))}",
                    f"--h5 {shlex.quote(os.path.realpath(projections_h5))}",
                    f"--rep-key {shlex.quote(rep_key)}",
                    f"--precompute-mean {shlex.quote(os.path.realpath(mean_path.parent) + '/' + mean_path.name)}",
                ],
                extra_binds=[projections_h5, reference.parent, out_csv.parent],
                banner="Query mean",
            )
            mean_sbatch = [
                "sbatch", f"--job-name={job_name}_mean",
                f"--partition={partition}", "--cpus-per-task=4", "--mem=16G",
                "--time=02:00:00",
                f"--output={log_dir}/hpl_assign_mean_%j.out",
                f"--error={log_dir}/hpl_assign_mean_%j.err",
                f"--chdir={backend_dir}",
                "--wrap", f"bash -lc {shlex.quote(mean_command)}",
            ]
            try:
                mean_result = _run_sbatch_with_retry(mean_sbatch)
            except subprocess.CalledProcessError as e:
                reason = (e.stderr or "").strip() or (e.stdout or "").strip() or "no output"
                raise RuntimeError(f"Could not submit the query-mean job: {reason}") from e
            match = re.search(r"Submitted batch job (\d+)", mean_result.stdout or "")
            mean_job_id = match.group(1) if match else None
            if not mean_job_id:
                raise RuntimeError(
                    "The query-mean job was submitted but sbatch reported no job ID, "
                    "so the shard array cannot be made to depend on it. Without that "
                    "dependency the shards would read a mean file that does not exist "
                    "yet."
                )
        else:
            # reference / none centering needs no shared mean: both are
            # shard-independent by construction.
            mean_path = None

    command = _build_assignment_command(
        singularity_bin=singularity_bin,
        singularity_image=singularity_image,
        extras_dir=extras_dir,
        assign_script=backend_dir / ASSIGN_SCRIPT,
        reference=reference,
        projections_h5=projections_h5,
        out_csv=out_csv,
        rep_key=rep_key,
        k=k,
        backend=backend,
        batch_size=batch_size,
        validate_against=validate_against,
        query_mean=mean_path,
        shard_bounds=shard_bounds,
    )

    sbatch_command = [
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={partition}",
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        f"--time={time_limit}",
        *([f"--array=0-{shards - 1}"] if shard_bounds else []),
        # afterok on the mean job when sharding: a shard that ran before the mean
        # file existed would fail on a missing --query-mean, and one that somehow
        # read a stale mean would silently disagree with its siblings.
        *([f"--dependency=afterok:{mean_job_id or depends_on_job_id}"]
          if (mean_job_id or depends_on_job_id) else []),
        f"--output={log_dir}/hpl_assign_%j.out",
        f"--error={log_dir}/hpl_assign_%j.err",
        f"--chdir={backend_dir}",
        *([f"--mail-user={notify_email}", "--mail-type=END,FAIL"] if notify_email else []),
        "--wrap", f"bash -lc {shlex.quote(command)}",
    ]

    info = {
        "projections_h5": str(projections_h5),
        "out_csv": str(out_csv),
        "embeddings": rows,
        "assignment_job_id": None,
        "shards": shards,
        "shard_bounds": shard_bounds,
        "mean_job_id": mean_job_id,
        "merge_job_id": None,
        "sbatch_command": shlex.join(sbatch_command),
        **reference_info,
    }

    try:
        result = _run_sbatch_with_retry(sbatch_command)
    except subprocess.CalledProcessError as e:
        reason = (e.stderr or "").strip() or (e.stdout or "").strip() or "no output from sbatch"
        raise RuntimeError(f"sbatch failed (exit {e.returncode}): {reason}") from e

    stdout = (result.stdout or "").strip()
    info["sbatch_stdout"] = stdout
    match = re.search(r"Submitted batch job (\d+)", stdout)
    if match:
        info["assignment_job_id"] = match.group(1)

    if shard_bounds:
        if not info["assignment_job_id"]:
            raise RuntimeError(
                "The shard array was submitted but sbatch reported no job ID, so the "
                "merge cannot depend on it. Merge by hand once the array finishes:\n"
                f"  python {backend_dir / 'merge_assignment_shards.py'} "
                f"--output {out_csv} --expected-rows {rows} --cleanup"
            )
        merge_command = _build_simple_command(
            singularity_bin=singularity_bin, singularity_image=singularity_image,
            extras_dir=extras_dir, script=backend_dir / "merge_assignment_shards.py",
            args=[
                f"--output {shlex.quote(os.path.realpath(out_csv.parent) + '/' + out_csv.name)}",
                # Checked against the projections file rather than believed from
                # the parts: a missing final shard leaves no gap to detect.
                f"--expected-rows {rows}",
                "--cleanup",
            ],
            extra_binds=[out_csv.parent],
            banner="Merging shards",
        )
        merge_sbatch = [
            "sbatch", f"--job-name={job_name}_merge",
            f"--partition={partition}", "--cpus-per-task=2", "--mem=8G",
            "--time=02:00:00",
            # afterok, not afterany: concatenating around a failed shard is the
            # silent corruption this whole path is built to avoid.
            f"--dependency=afterok:{info['assignment_job_id']}",
            f"--output={log_dir}/hpl_assign_merge_%j.out",
            f"--error={log_dir}/hpl_assign_merge_%j.err",
            f"--chdir={backend_dir}",
            "--wrap", f"bash -lc {shlex.quote(merge_command)}",
        ]
        try:
            merge_result = _run_sbatch_with_retry(merge_sbatch)
        except subprocess.CalledProcessError as e:
            reason = (e.stderr or "").strip() or (e.stdout or "").strip() or "no output"
            raise RuntimeError(
                f"The shard array is job {info['assignment_job_id']}, but the merge "
                f"job could not be submitted: {reason}. Merge by hand once it finishes."
            ) from e
        merge_match = re.search(r"Submitted batch job (\d+)", merge_result.stdout or "")
        if merge_match:
            info["merge_job_id"] = merge_match.group(1)

    return info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit a Slurm job assigning HPL cluster IDs to tile embeddings.",
    )
    parser.add_argument("--projections-h5", type=Path, required=True,
                        help="Feature extraction's output for this dataset.")
    parser.add_argument("--out", dest="out_csv", type=Path, required=True,
                        help="Where the per-tile assignment CSV goes.")
    parser.add_argument("--reference", type=Path, default=None,
                        help=f"Reference .npz. Defaults to ${_REFERENCE_ENV} or the "
                             f"path build_hpc_reference.py writes.")
    parser.add_argument("--depends-on-job-id", type=str, default=None)
    parser.add_argument("--rep-key", type=str, default="z_latent")
    parser.add_argument("--k", type=int, default=None,
                        help="Neighbours to poll. Defaults to the reference's own "
                             "Leiden n_neighbors, which is what ingest used.")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "faiss", "faiss-ivf", "numpy"])
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--validate-against", type=Path, default=None,
                        help="A CSV of known labels to check the assignment reproduces. "
                             "Use Kai's TCGA transfer as an acceptance test.")
    parser.add_argument("--partition", type=str, default=MERGE_PARTITION)
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--memory", type=str, default="64G")
    parser.add_argument("--time-limit", type=str, default="04:00:00")
    parser.add_argument("--notify-email", type=str, default=None)
    parser.add_argument("--shards", type=int, default=1,
                        help="Split the assignment across N array tasks. A mean job "
                             "runs first so every shard centres identically, then a "
                             "merge job concatenates the parts.")
    parser.add_argument("--centering", type=str, default="query",
                        choices=["query", "reference", "none"])
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing output CSV.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        info = submit_cluster_assignment_job(
            projections_h5=args.projections_h5,
            out_csv=args.out_csv,
            reference=args.reference,
            depends_on_job_id=args.depends_on_job_id,
            rep_key=args.rep_key,
            k=args.k,
            backend=args.backend,
            batch_size=args.batch_size,
            validate_against=args.validate_against,
            shards=args.shards,
            centering=args.centering,
            partition=args.partition,
            cpus=args.cpus,
            memory=args.memory,
            time_limit=args.time_limit,
            notify_email=args.notify_email,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, KeyError) as e:
        print(f"Not submitted: {e}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Reference:        {info['reference_path']}")
    print(f"  rows x dims     {info['reference_rows']:,} x {info['reference_dims']}")
    print(f"  clusters        {info['n_clusters']}")
    if info["embeddings"] is not None:
        print(f"Embeddings:       {info['embeddings']:,}")
    print(f"Output CSV:       {info['out_csv']}")
    if info["shards"] > 1:
        b = info["shard_bounds"]
        print(f"Shards:           {info['shards']}  "
              f"(rows {b[0][0]}-{b[0][1]} ... {b[-1][0]}-{b[-1][1]})")
        print(f"Mean job ID:      {info['mean_job_id']}")
        print(f"Merge job ID:     {info['merge_job_id']}")
    print(f"Slurm job ID:     {info['assignment_job_id']}")


if __name__ == "__main__":
    main()
