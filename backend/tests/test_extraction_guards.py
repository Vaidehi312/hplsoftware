"""Tests for the feature-extraction guards: what stops a retry dead-ending.

Runs under pytest, and also standalone with `python3 test_extraction_guards.py`
— same reasoning as test_dataset_rollup.py. Unlike that suite these cases do
touch the filesystem and HDF5, because the thing under test *is* a file's
integrity; they never touch Slurm (nothing here reaches sbatch) or Postgres.

The failure these guard against: real_encode_contrastive_from_checkpoint() in
the HPL repo skips encoding entirely when its output file already exists, and
its skip path then raises UnboundLocalError (it reads `key_shape`, assigned
only in the encoding branch). The encoder also creates that output with
mode='w' *before* encoding anything. So any interrupted attempt leaves a file
that makes every later attempt fail in seconds, with an error naming a
variable rather than the leftover file — and extraction has no resume, so
nothing else was ever going to notice.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shlex  # noqa: E402

from submit_feature_extraction import (  # noqa: E402
    GPU_GRES,
    GPU_PREFERENCE,
    _build_extraction_command,
    _check_checkpoint,
    _check_container_extras,
    _check_hpl_repo_dir,
    _gpu_probe_python,
    _input_h5_rows,
    expected_extraction_output_path,
    submit_feature_extraction_job,
    validate_extraction_output,
)

ROWS = 7
Z_DIM, H_DIM = 128, 1536
CARRIED = ("samples", "slides", "tiles")


def write_input(path, rows=ROWS):
    """A packaged .h5 as make_hpl_hdf5.package_slides_to_h5 leaves it."""
    with h5py.File(path, "w") as f:
        f.create_dataset("img", (rows, 224, 224, 3), dtype="uint8")
        for name in CARRIED:
            ds = f.create_dataset(name, (rows,), dtype="S12")
            if rows:
                ds[:] = np.array([b"slide_01"] * rows, dtype="S12")


def write_output(path, rows=ROWS, latents=True, meta=True, meta_rows=None):
    """A projections .h5 as the encoder leaves it. latents=False, meta=False
    is the shape of the real leftover: mode='w' ran, nothing else did."""
    with h5py.File(path, "w") as f:
        if latents:
            f.create_dataset("img_h_latent", (rows, H_DIM), dtype=np.float32)
            f.create_dataset("img_z_latent", (rows, Z_DIM), dtype=np.float32)
        if meta:
            mr = rows if meta_rows is None else meta_rows
            for name in CARRIED:
                ds = f.create_dataset(name, (mr,), dtype="S12")
                if mr:
                    ds[:] = np.array([b"slide_01"] * mr, dtype="S12")


# --- validate_extraction_output ------------------------------------------

def test_complete_output_validates(tmp_path):
    p = tmp_path / "complete.h5"
    write_output(p)
    assert validate_extraction_output(p) == (True, "")


def test_bare_mode_w_leftover_rejected(tmp_path):
    """The exact file a job killed during model restore leaves behind."""
    p = tmp_path / "empty.h5"
    h5py.File(p, "w").close()
    ok, reason = validate_extraction_output(p)
    assert not ok and "never wrote embeddings" in reason


def test_died_before_metadata_copy_rejected(tmp_path):
    """The encoder writes latents first, then copies samples/slides/tiles, so
    latents-without-metadata means it died between the two."""
    p = tmp_path / "latents_only.h5"
    write_output(p, meta=False)
    ok, reason = validate_extraction_output(p)
    assert not ok and "before copying" in reason


def test_zero_embeddings_rejected(tmp_path):
    p = tmp_path / "zero.h5"
    write_output(p, rows=0)
    assert not validate_extraction_output(p)[0]


def test_length_disagreement_rejected(tmp_path):
    p = tmp_path / "mismatch.h5"
    write_output(p, meta_rows=3)
    ok, reason = validate_extraction_output(p)
    assert not ok and "disagree" in reason


def test_truncated_file_rejected(tmp_path):
    """HDF5 validates the superblock on open, so a file cut partway through
    its data opens cleanly — reading the last row is what catches it."""
    p = tmp_path / "truncated.h5"
    write_output(p)
    with open(p, "r+b") as fh:
        fh.truncate(os.path.getsize(p) // 2)
    assert not validate_extraction_output(p)[0]


def test_output_from_a_different_input_rejected(tmp_path):
    """Same basename, different source .h5 — the one corruption reading rows
    cannot catch, since every row present is perfectly readable."""
    p = tmp_path / "complete.h5"
    write_output(p)
    ok, reason = validate_extraction_output(p, expected_rows=ROWS + 1)
    assert not ok and "different input file" in reason
    assert validate_extraction_output(p, expected_rows=ROWS)[0]


def test_missing_file_rejected(tmp_path):
    assert not validate_extraction_output(tmp_path / "nope.h5")[0]


def test_input_row_count(tmp_path):
    p = tmp_path / "input.h5"
    write_input(p)
    assert _input_h5_rows(p) == ROWS
    # Never gates anything on its own, so an unreadable input is None rather
    # than an error — the caller validates the input properly.
    assert _input_h5_rows(tmp_path / "nope.h5") is None


# --- _check_hpl_repo_dir -------------------------------------------------

def test_repo_dir_with_filename_appended(tmp_path):
    """The original failure: HPL_REPO_DIR set to a file inside the repo. The
    hint matters more than the rejection — the path looks right at a glance."""
    repo = tmp_path / "HPL-LATTICeA"
    repo.mkdir()
    (repo / "run_representationspathology_projection.py").touch()
    _check_hpl_repo_dir(repo)  # the correct value: no raise

    try:
        _check_hpl_repo_dir(repo / "submit_feature_extraction.py")
        raise AssertionError("expected NotADirectoryError")
    except NotADirectoryError as e:
        assert "filename got appended" in str(e) and str(repo) in str(e)


def test_dir_that_is_not_the_clone_rejected(tmp_path):
    """Catches pointing at the parent, or at backend/ — both are real
    directories, so is_dir() alone would have passed them through to a job
    that fails only after a GPU queue wait."""
    try:
        _check_hpl_repo_dir(tmp_path)
        raise AssertionError("expected NotADirectoryError")
    except NotADirectoryError as e:
        assert "run_representationspathology_projection.py" in str(e)


# --- submit-time stale-output handling -----------------------------------
# None of these reach sbatch: every case must be decided before submission.

def _fake_repo(tmp_path):
    repo = tmp_path / "HPL-LATTICeA"
    repo.mkdir()
    (repo / "run_representationspathology_projection.py").touch()
    return repo


def _fake_extras(parent):
    """An extras directory the submitter will accept. Only the presence of the
    installed package is checked, so an empty skimage/ is enough."""
    extras = parent / "extras-py38"
    (extras / "skimage").mkdir(parents=True, exist_ok=True)
    return extras


def _submit(repo, inp, **kw):
    """Returns the message from whatever the submitter refused with, or
    'SUBMITTED' if it got as far as trying to run sbatch."""
    # A zero-byte stand-in is enough: the submitter only checks that the
    # image path exists before handing off to sbatch (which we never reach
    # with a real binary here).
    image = inp.parent / "fake.sif"
    if not image.exists():
        image.touch()
    kw.setdefault("singularity_image", image)
    kw.setdefault("singularity_bin", sys.executable)  # any existing file
    # Populated by default so these tests exercise the guard they are named
    # for rather than stopping at the container-extras check.
    kw.setdefault("extras_dir", _fake_extras(inp.parent))

    # A checkpoint the submitter will accept, so these tests exercise the
    # stale-output guards rather than stopping at checkpoint validation.
    checkpoint = kw.pop("checkpoint", None)
    if checkpoint is None:
        weights = inp.parent / "weights"
        weights.mkdir(exist_ok=True)
        (weights / "BarlowTwins_3.ckt.index").touch()
        checkpoint = str(weights / "BarlowTwins_3.ckt")

    try:
        submit_feature_extraction_job(
            real_hdf5_path=inp, checkpoint=checkpoint, dataset_name="TestDS",
            hpl_repo_dir=repo, **kw,
        )
        return "SUBMITTED"
    except FileExistsError as e:
        return str(e)
    except Exception:
        # sbatch missing / failing means the guards let it through, which is
        # the outcome under test here.
        return "SUBMITTED"


def test_complete_output_is_not_resubmitted(tmp_path):
    """Resubmitting over a finished extraction burns a GPU allocation to
    produce nothing — the encoder would skip its work and crash."""
    repo, inp = _fake_repo(tmp_path), tmp_path / "hdf5_TestDS_he_train.h5"
    write_input(inp)
    out = expected_extraction_output_path(repo, "BarlowTwins_3", "TestDS", inp)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_output(out)

    msg = _submit(repo, inp)
    assert "already completed" in msg
    assert out.is_file(), "a complete output must never be deleted"


def test_stale_output_cleared_then_submitted(tmp_path):
    repo, inp = _fake_repo(tmp_path), tmp_path / "hdf5_TestDS_he_train.h5"
    write_input(inp)
    out = expected_extraction_output_path(repo, "BarlowTwins_3", "TestDS", inp)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_output(out, latents=False, meta=False)

    assert _submit(repo, inp) == "SUBMITTED"
    assert not out.exists(), "the leftover must be gone before the job runs"


def test_stale_output_kept_when_asked(tmp_path):
    repo, inp = _fake_repo(tmp_path), tmp_path / "hdf5_TestDS_he_train.h5"
    write_input(inp)
    out = expected_extraction_output_path(repo, "BarlowTwins_3", "TestDS", inp)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_output(out, latents=False, meta=False)

    msg = _submit(repo, inp, clear_stale_output=False)
    assert "unusable file" in msg
    assert out.is_file()


def test_output_from_another_input_is_cleared(tmp_path):
    repo, inp = _fake_repo(tmp_path), tmp_path / "hdf5_TestDS_he_train.h5"
    write_input(inp)
    out = expected_extraction_output_path(repo, "BarlowTwins_3", "TestDS", inp)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_output(out, rows=ROWS + 3)   # complete, but not of this input

    assert _submit(repo, inp) == "SUBMITTED"
    assert not out.exists()


# --- GPU targeting and container paths -----------------------------------
# The failures here cost hours each and both looked like success at submit
# time: one ran to completion on a CPU, the other reached the GPU and died on
# `cd`. Neither is visible without reading a Slurm log.

def _cluster_layout(tmp_path):
    """The layout that broke job 1214773: the repo reachable by two names,
    one of which is a symlink into another filesystem."""
    real = tmp_path / "cephfs" / "users" / "vpandya"
    repo = real / "Work" / "HPL-LATTICeA"
    repo.mkdir(parents=True)
    (repo / "run_representationspathology_projection.py").touch()
    (real / "Work" / "model_input").mkdir(parents=True)
    h5 = real / "Work" / "model_input" / "hdf5_X_he_train.h5"
    h5.touch()
    ckpt = real / "BarlowTwins_3.ckt"
    ckpt.touch()
    sif = real / "tf.sif"
    sif.touch()
    extras = real / "extras-py38"
    (extras / "skimage").mkdir(parents=True)

    home = tmp_path / "hpc-home" / "users" / "vpandya"
    home.mkdir(parents=True)
    (home / "long-term-scratch").symlink_to(real)
    link = home / "long-term-scratch"
    return {
        "sif": sif,
        "ckpt": ckpt,
        "extras": extras,
        "linked_repo": link / "Work" / "HPL-LATTICeA",
        "linked_h5": link / "Work" / "model_input" / "hdf5_X_he_train.h5",
    }


def _command_for(layout):
    return _build_extraction_command(
        singularity_bin="/usr/bin/singularity",
        singularity_image=layout["sif"],
        hpl_repo_dir=layout["linked_repo"],
        real_hdf5_path=layout["linked_h5"],
        checkpoint=str(layout["ckpt"]),
        dataset_name="X",
        model="BarlowTwins_3",
        marker="he",
        z_dim=128,
        img_size=224,
        batch_size=64,
        extras_dir=layout["extras"],
    )


def test_symlinked_repo_is_entered_by_its_real_path(tmp_path):
    """A symlink inside a container resolves against the container, where the
    target is absent unless bound. Job 1214773 registered an H200 and then
    failed on `cd` into a directory that lists fine from the login node."""
    layout = _cluster_layout(tmp_path)
    cmd = _command_for(layout)
    real_repo = os.path.realpath(layout["linked_repo"])

    assert f"cd {shlex.quote(real_repo)}" in cmd
    # The link's own spelling must not be what the job cds into.
    assert f"cd {shlex.quote(str(layout['linked_repo']))}" not in cmd


def test_both_spellings_of_every_input_are_bound(tmp_path):
    """Binding only the /hpc-home side is what left the target invisible."""
    layout = _cluster_layout(tmp_path)
    parts = shlex.split(_command_for(layout))
    sources = {p.split(":", 1)[0] for i, p in enumerate(parts)
               if i and parts[i - 1] == "--bind"}

    for path in (layout["linked_repo"], layout["linked_h5"], layout["ckpt"]):
        real = os.path.realpath(path)
        assert any(real == s or real.startswith(s.rstrip("/") + "/") for s in sources), \
            f"nothing binds the real location of {path}"


def test_paths_are_checked_before_the_encoder_starts(tmp_path):
    """So the next bind mistake names itself instead of surfacing as a bare
    'No such file or directory' against a path that plainly exists."""
    cmd = _command_for(_cluster_layout(tmp_path))
    assert "not visible inside the container" in cmd
    # Ordering matters: a check after the encoder starts is not a check.
    assert cmd.index("not visible inside the container") < cmd.index("=== Feature extraction ===")


def _rejects(checkpoint) -> str:
    """Return the rejection message, or fail if it was accepted."""
    try:
        _check_checkpoint(str(checkpoint))
    except FileNotFoundError as e:
        return str(e)
    raise AssertionError(f"expected {checkpoint!r} to be rejected")


def test_relative_checkpoint_is_rejected_not_resolved(tmp_path):
    """The exact shape of the 1214781 failure. A path missing its leading '/'
    was joined to the server's cwd, producing
    .../Work/backend/mnt/cephfs-lts/... — wrong, and wrong in a way that reads
    like a real path."""
    del tmp_path
    msg = _rejects("mnt/cephfs-lts/long-term-scratch/users/vpandya/w/BT_3.ckt")
    assert "absolute" in msg
    # The fix should be in the message, not left as an exercise.
    assert "/mnt/cephfs-lts" in msg


def test_tensorflow_checkpoint_prefix_is_accepted(tmp_path):
    """BarlowTwins_3.ckt names a set of sidecar files and usually does not
    exist itself. Requiring is_file() would reject every working checkpoint."""
    weights = tmp_path / "weights"
    weights.mkdir()
    for suffix in (".index", ".data-00000-of-00001", ".meta"):
        (weights / f"BarlowTwins_3.ckt{suffix}").touch()

    _check_checkpoint(str(weights / "BarlowTwins_3.ckt"))  # must not raise


def test_missing_checkpoint_names_what_is_actually_there(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "BarlowTwins_3.ckt.index").touch()

    assert "BarlowTwins_3.ckt" in _rejects(weights / "BarlowTwins_5.ckt")


def test_missing_checkpoint_directory_says_so(tmp_path):
    msg = _rejects(tmp_path / "nope" / "BarlowTwins_3.ckt")
    assert "directory does not exist" in msg


def test_bad_checkpoint_stops_the_submission(tmp_path):
    """Validation has to sit in the submitter, not only in the UI: the whole
    point is that no GPU is allocated for a path that cannot work."""
    repo, inp = _fake_repo(tmp_path), tmp_path / "hdf5_TestDS_he_train.h5"
    write_input(inp)
    try:
        submit_feature_extraction_job(
            real_hdf5_path=inp, checkpoint="relative/BarlowTwins_3.ckt",
            dataset_name="TestDS", hpl_repo_dir=repo,
        )
        raise AssertionError("expected the submission to be refused")
    except FileNotFoundError as e:
        assert "absolute" in str(e)


def test_container_checks_the_checkpoint_directory(tmp_path):
    """Checking the prefix itself would fail a valid checkpoint inside the
    job, after the GPU has already been allocated."""
    layout = _cluster_layout(tmp_path)
    cmd = _command_for(layout)
    real_ckpt = os.path.realpath(layout["ckpt"])
    assert os.path.dirname(real_ckpt) in cmd
    assert f'-e {shlex.quote(real_ckpt)}' not in cmd


def test_gres_names_the_gpu_model(tmp_path):
    """Bare gpu:1 is satisfiable by any card on the partition, including ones
    whose CUDA this container cannot use — the misconfig that put a run on CPU
    for 8.6 hours. Every type the submitter can request must be named."""
    del tmp_path
    assert "h200" in GPU_GRES.lower(), GPU_GRES
    assert GPU_PREFERENCE, "the preference list must not be empty"
    for gpu_type in GPU_PREFERENCE:
        assert gpu_type and ":" not in gpu_type, gpu_type


def test_h200_is_preferred_but_not_required(tmp_path):
    """H200 first, then next-best. The fallback matters because the H200 queue
    is often full, and after the read-path work extraction is decode-bound at
    ~1.4k tiles/s — below what any of these cards encode — so an H100 or A100
    finishes in about the same wall clock."""
    del tmp_path
    assert GPU_PREFERENCE[0] == "nvidia_h200", GPU_PREFERENCE
    assert len(GPU_PREFERENCE) > 1, "a preference list of one cannot fall back"


def _select_with_sinfo(monkeypatched_output, *, returncode=0, raises=None):
    """select_gpu_gres against a stubbed sinfo."""
    import submit_feature_extraction as sfe

    class _Result:
        def __init__(self, out):
            self.stdout, self.stderr, self.returncode = out, "", returncode

    original = sfe.subprocess.run
    try:
        if raises is not None:
            def _run(*a, **k):
                raise raises
            sfe.subprocess.run = _run
        else:
            sfe.subprocess.run = lambda *a, **k: _Result(monkeypatched_output)
        return sfe.select_gpu_gres("gpu")
    finally:
        sfe.subprocess.run = original


def test_best_free_gpu_is_chosen(tmp_path):
    del tmp_path
    # H200 free -> take it.
    gres, _ = _select_with_sinfo("gpu:nvidia_h200:4|idle|2\ngpu:nvidia_a100_80gb_pcie:8|idle|3\n")
    assert gres == "gpu:nvidia_h200:1", gres

    # H200 fully allocated -> next preference that is free.
    gres, _ = _select_with_sinfo(
        "gpu:nvidia_h200:4|alloc|2\ngpu:nvidia_h100_80gb_hbm3:4|idle|1\ngpu:nvidia_a100_80gb_pcie:8|idle|3\n"
    )
    assert gres == "gpu:nvidia_h100_80gb_hbm3:1", gres

    # Nothing idle, but an A100 node is partly free -> better than waiting.
    gres, _ = _select_with_sinfo(
        "gpu:nvidia_h200:4|alloc|2\ngpu:nvidia_h100_80gb_hbm3:4|alloc|1\ngpu:nvidia_a100_80gb_pcie:8|mix|3\n"
    )
    assert gres == "gpu:nvidia_a100_80gb_pcie:1", gres


def test_nothing_free_queues_for_the_best(tmp_path):
    """Requesting the fastest card and waiting beats silently downgrading, since
    this cannot compare queue depths."""
    del tmp_path
    gres, reason = _select_with_sinfo(
        "gpu:nvidia_h200:4|alloc|2\ngpu:nvidia_a100_80gb_pcie:8|alloc|3\n"
    )
    assert gres == "gpu:nvidia_h200:1", gres
    assert "queueing" in reason


def test_unresponsive_nodes_do_not_count_as_free(tmp_path):
    """sinfo's '*' means the node is not answering, so 'idle*' is idle and
    unschedulable. Counting it would keep picking a dead H200 over a live
    A100 — a job that queues forever rather than one that runs."""
    del tmp_path
    gres, _ = _select_with_sinfo("gpu:nvidia_h200:4|idle*|2\ngpu:nvidia_a100_80gb_pcie:8|idle|3\n")
    assert gres == "gpu:nvidia_a100_80gb_pcie:1", gres


def test_unknown_cluster_types_are_used_rather_than_invented(tmp_path):
    """Requesting a type the cluster has never heard of is rejected by sbatch
    outright, so an unrecognised partition falls back to what is actually there
    and says how to configure it."""
    del tmp_path
    gres, reason = _select_with_sinfo("gpu:tesla_v100:2|idle|5\n")
    assert gres == "gpu:tesla_v100:1", gres
    assert "HPL_GPU_PREFERENCE" in reason


def test_missing_sinfo_is_not_read_as_no_gpus(tmp_path):
    """Same distinction the tiling code draws for sacct: "cannot ask" must not
    become a positive claim. Ask for the first preference and let Slurm decide."""
    del tmp_path
    for kwargs in ({"returncode": 1}, {"raises": OSError("sinfo not found")}):
        gres, reason = _select_with_sinfo("", **kwargs)
        assert gres == "gpu:nvidia_h200:1", gres
        assert "unavailable" in reason


# Verbatim `sinfo -p gpu -o "%N %G %t %D"` from the cluster, including the
# duplicated type entries and the shard: lines. Kept literal because both of
# those broke a first attempt at this: the duplicates double-counted nodes, and
# guessed type names ("nvidia_h100") matched nothing here, which made the whole
# fallback inert while looking like it worked.
REAL_SINFO = (
    "gpu:nvidia_h100_80gb_hbm3:4(S:0-1),shard:nvidia_h100_80gb_hbm3:240(S:0-1),"
    "gpu:nvidia_h100_80gb_hbm3:nvidia_h100_80gb_hbm3:4|mix|1\n"
    "gpu:nvidia_h200:8(S:0-1),shard:nvidia_h200:1144(S:0-1),"
    "gpu:nvidia_h200:nvidia_h200:8|alloc|2\n"
    "gpu:nvidia_h100_pcie:3(S:0-1),shard:nvidia_h100_pcie:240(S:0-1),"
    "gpu:nvidia_h100_pcie:nvidia_h100_pcie:3|idle|1\n"
    "gpu:nvidia_a100_80gb_pcie:2(S:0),shard:nvidia_a100_80gb_pcie:160(S:0),"
    "gpu:nvidia_a100_80gb_pcie:nvidia_a100_80gb_pcie:2|idle|1\n"
)


def test_real_cluster_types_are_the_defaults(tmp_path):
    """Every default preference must exist on the cluster, or the fallback is
    inert: a list of absent names falls through to "queue for the first one",
    which is precisely the behaviour it was added to replace."""
    del tmp_path
    from submit_feature_extraction import discover_gpu_types
    import submit_feature_extraction as sfe

    original = sfe.subprocess.run

    class _R:
        stdout, stderr, returncode = REAL_SINFO, "", 0

    try:
        sfe.subprocess.run = lambda *a, **k: _R()
        available = discover_gpu_types("gpu")
    finally:
        sfe.subprocess.run = original

    for gpu_type in GPU_PREFERENCE:
        assert gpu_type in available, (
            f"{gpu_type} is in the default preference but not on the cluster; "
            f"it has {sorted(available)}"
        )


def test_node_counts_are_not_doubled_by_duplicate_gres_entries(tmp_path):
    """This cluster lists each type twice per line ("gpu:X:4" and "gpu:X:X:4").
    Counting both would report two idle nodes where there is one, so the reason
    printed at submit time would overstate what is free."""
    del tmp_path
    import submit_feature_extraction as sfe

    original = sfe.subprocess.run

    class _R:
        stdout, stderr, returncode = REAL_SINFO, "", 0

    try:
        sfe.subprocess.run = lambda *a, **k: _R()
        available = sfe.discover_gpu_types("gpu")
    finally:
        sfe.subprocess.run = original

    # Straight from the NODES column of the sinfo output above.
    assert available["nvidia_h100_80gb_hbm3"]["total"] == 1
    assert available["nvidia_h200"]["total"] == 2
    assert available["nvidia_h100_pcie"]["total"] == 1
    assert available["nvidia_a100_80gb_pcie"]["total"] == 1
    # shard: entries are MPS slices, not whole GPUs, and must never be requested.
    assert not any("shard" in name for name in available)


def test_busy_h200_falls_back_on_the_real_cluster(tmp_path):
    """The case that prompted this: both H200 nodes allocated, an H100 PCIe node
    idle. Queueing for the H200 was the old behaviour and the thing to avoid."""
    del tmp_path
    gres, reason = _select_with_sinfo(REAL_SINFO)
    assert gres == "gpu:nvidia_h100_pcie:1", gres
    assert "idle" in reason

    # And when the H200s free up it goes straight back to them.
    gres, _ = _select_with_sinfo(REAL_SINFO.replace("|alloc|2", "|idle|2"))
    assert gres == "gpu:nvidia_h200:1", gres


def test_ambiguous_type_name_is_not_guessed(tmp_path):
    """"nvidia_h100" matches both the SXM and PCIe cards here, which differ in
    bandwidth. Resolving it either way would be a silent performance decision."""
    del tmp_path
    from submit_feature_extraction import _match_gpu_type
    available = {"nvidia_h100_pcie": {}, "nvidia_h100_80gb_hbm3": {}, "nvidia_h200": {}}
    assert _match_gpu_type("nvidia_h100", available) is None
    # Unambiguous substrings still resolve, so a loosely-written list works.
    assert _match_gpu_type("h200", available) == "nvidia_h200"
    assert _match_gpu_type("nvidia_h200", available) == "nvidia_h200"


def test_explicit_gres_is_never_second_guessed(tmp_path):
    """An operator naming a type has a reason; probing would override it."""
    del tmp_path
    from submit_feature_extraction import select_gpu_gres
    gres, reason = select_gpu_gres("gpu", explicit="gpu:tesla_v100:2")
    assert gres == "gpu:tesla_v100:2"
    assert "explicit" in reason


def test_probe_refuses_to_continue_without_a_gpu(tmp_path):
    """The CPU fallback is silent and correct-looking: TF logs the card, fails
    to load its CUDA libs, and encodes anyway. One run burned 8.6 hours that
    way before hitting the wall clock."""
    del tmp_path
    probe = _gpu_probe_python()
    assert "sys.exit(1)" in probe
    assert "is_gpu_available" in probe


def test_probe_runs_before_the_encoder(tmp_path):
    cmd = _command_for(_cluster_layout(tmp_path))
    assert cmd.index("TensorFlow GPU probe") < cmd.index("=== Feature extraction ===")
    # Without this the probe's exit status is discarded and encoding proceeds.
    assert "set -euo pipefail" in cmd


def test_walltime_is_two_days_and_reaches_sbatch(tmp_path):
    """The encoder has no resume — it opens its output with mode='w' and starts
    from row zero — so a run killed by the wall clock at 99% has produced
    nothing and costs the whole allocation again. Overshooting the limit only
    costs backfill position, since Slurm bills what is used."""
    del tmp_path
    from submit_feature_extraction import _DEFAULT_TIME_LIMIT, build_parser

    assert _DEFAULT_TIME_LIMIT == "2-00:00:00", _DEFAULT_TIME_LIMIT
    # Both entry points, so the CLI and the server-side call agree.
    assert build_parser().get_default("time_limit") == _DEFAULT_TIME_LIMIT
    import inspect
    from submit_feature_extraction import submit_feature_extraction_job
    got = inspect.signature(submit_feature_extraction_job).parameters["time_limit"].default
    assert got == _DEFAULT_TIME_LIMIT, got


def test_container_flags_present(tmp_path):
    """--nv injects the driver's libcuda; without it the container sees no GPU
    at all. --cleanenv keeps the host's CUDA 10 stubs out of LD_LIBRARY_PATH,
    which is what the conda env poisoned."""
    cmd = _command_for(_cluster_layout(tmp_path))
    assert " --nv " in cmd
    assert " --cleanenv " in cmd


# --- packages the NGC image does not ship --------------------------------
# The shape of the 1214883 failure: the GPU probe passed, every path resolved,
# and the job died importing scikit-image — which the image has never had and
# HPL-LATTICeA imports at module scope (models/data_augmentation.py).

def test_extras_are_on_pythonpath_inside_the_job(tmp_path):
    """--cleanenv wipes the environment, so PYTHONPATH has to be set inside
    the container. By its real path, for the same reason `cd` is."""
    layout = _cluster_layout(tmp_path)
    cmd = _command_for(layout)
    real_extras = os.path.realpath(layout["extras"])

    assert f"export PYTHONPATH={shlex.quote(real_extras)}" in cmd
    # Appended, not replaced: clobbering an image-set PYTHONPATH would take
    # away modules rather than add them.
    assert "${PYTHONPATH:+:$PYTHONPATH}" in cmd


def test_extras_directory_is_bound_into_the_job(tmp_path):
    """On PYTHONPATH but not bound is a directory that does not exist in
    there — the same mistake that made the repo invisible in 1214773."""
    layout = _cluster_layout(tmp_path)
    parts = shlex.split(_command_for(layout))
    sources = {p.split(":", 1)[0] for i, p in enumerate(parts)
               if i and parts[i - 1] == "--bind"}
    real = os.path.realpath(layout["extras"])
    assert any(real == s or real.startswith(s.rstrip("/") + "/") for s in sources)


def test_missing_packages_are_reported_before_the_encoder_starts(tmp_path):
    """Otherwise the report is a ModuleNotFoundError from four frames inside a
    third-party import chain, arriving after the GPU allocation."""
    cmd = _command_for(_cluster_layout(tmp_path))
    assert "packages missing inside the container" in cmd
    assert cmd.index("packages missing inside the container") < \
        cmd.index("=== Feature extraction ===")


def test_unpopulated_extras_directory_stops_the_submission(tmp_path):
    """The check is worth having at submit time too: a queue wait to be told
    a package is missing is a queue wait spent learning nothing."""
    empty = tmp_path / "extras-py38"
    empty.mkdir()
    for extras in (empty, tmp_path / "never-created"):
        try:
            _check_container_extras(extras, tmp_path / "tf.sif", "/usr/bin/singularity")
        except FileNotFoundError as e:
            # The fix belongs in the message; looking it up is the slow part.
            assert "--bootstrap-extras" in str(e)
            assert "scikit-image" in str(e)
        else:
            raise AssertionError(f"expected {extras} to be rejected")


def test_extras_install_does_not_shadow_the_image_numpy(tmp_path):
    """PYTHONPATH wins over the image's site-packages, so a numpy pulled in as
    a transitive dependency would replace the one TF 1.15 was built against.
    --no-deps is what keeps that from happening."""
    del tmp_path
    from submit_feature_extraction import _CONTAINER_EXTRA_PACKAGES, _extras_bootstrap_hint

    hint = _extras_bootstrap_hint(Path("/x/extras"), Path("/x/tf.sif"), "singularity")
    assert "--no-deps" in hint
    installed = {p.split("==")[0].lower() for p in _CONTAINER_EXTRA_PACKAGES}
    assert not installed & {"numpy", "scipy", "tensorflow", "h5py", "matplotlib"}


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_extract_test_"))
        try:
            fn(tmp_path)
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
