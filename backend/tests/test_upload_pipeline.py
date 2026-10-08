"""A single uploaded slide is a one-slide pipeline run, all the way to the KB.

Uploading a slide used to mask, tile and package it inside the server and
then wait for someone to click through feature extraction and classification
— with an empty checkpoint box, the one place a typed path reached the
encoder. It is now submitted as the same Nextflow run Run HPL starts, over the
upload's own directory, so Stages 1-4 happen by themselves.

What goes wrong quietly, which is why it is tested here rather than left to
the first upload someone tries:

  * a re-upload over the previous upload's tiles. The tiling task keeps any
    slide whose tiles already validate, so the new run would classify the
    *old* file's tiles under the new file's name — every label plausible,
    every one for the wrong slide;
  * the cohort each upload is registered under. register_dataset.commit()
    scopes --replace to a dataset_id and DELETEs what it finds there, so one
    shared "UPLOADED" cohort would mean registering the second uploaded slide
    either refused outright or deleted the first one's rows;
  * the name the raw file is saved under. Registration and the tiler both
    find each slide's id via slide_id_from_raw_path(), and a name it cannot
    recover the slide_id from does not fail — it registers a cohort with no
    wsi_registry row, and the viewer 404s on a slide sitting right there;
  * the sentinel job id standing in for a stage that ran in the server, which
    uploads from before this still carry. sacct rejects a whole call for one
    id it does not recognise, so a sentinel reaching it would report "can't
    reach Slurm" for every real run in the same listing.
"""

import inspect
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
APP = BACKEND.parent / "app" / "app_v28.py"

# Standalone mode is how this suite runs on the cluster, where nothing else has
# put backend/ on the path.
sys.path.insert(0, str(BACKEND))

import tile_server_v2_ as srv  # noqa: E402
from slide_naming import slide_id_from_raw_path  # noqa: E402
from tile_mask import run_tissue_detection  # noqa: E402
from auto_tile_from_mask import tile_slide_from_mask  # noqa: E402

_SERVER_SOURCE = (BACKEND / "tile_server_v2_.py").read_text()


class _Patched:
    """Set module attributes for the duration of a block and put them back."""

    def __init__(self, module, **attrs):
        self.module, self.attrs, self.original = module, attrs, {}

    def __enter__(self):
        for name, value in self.attrs.items():
            self.original[name] = getattr(self.module, name)
            setattr(self.module, name, value)
        return self.module

    def __exit__(self, *exc):
        for name, value in self.original.items():
            setattr(self.module, name, value)
        return False


def _slurm_must_not_be_called(*args, **kwargs):
    raise AssertionError("Slurm was queried about a stage that never went to Slurm")


# --- the sentinel for a stage that ran in this process ---------------------

def test_an_in_process_stage_reads_as_completed(_tmp=None):
    """The gate every later stage gates on. Without it an uploaded slide's
    packaging has no Slurm state, and Stage 3 stays permanently blocked behind
    a .h5 that is sitting on disk."""
    with _Patched(srv, _run_slurm=_slurm_must_not_be_called,
                  _slurm_jobs_live_states=_slurm_must_not_be_called):
        assert srv._get_slurm_job_state(srv._local_job_id("packaging")) == "COMPLETED"


def test_a_real_job_id_is_still_asked_about(_tmp=None):
    """The companion: proves the shortcut above is narrow. A real id must still
    be resolved against Slurm, and an unreachable Slurm must still come back
    unknown rather than as a cheerful COMPLETED."""
    asked = []

    def _record(cmd, timeout):
        asked.append(cmd)
        return None

    with _Patched(srv, _run_slurm=_record, _slurm_jobs_live_states=lambda ids: []):
        assert srv._get_slurm_job_state("9876543") is None
    assert asked, "a real job id never reached Slurm"


def test_a_sentinel_is_kept_out_of_the_sacct_call(_tmp=None):
    """The blast radius this prevents: sacct rejects the whole call for one
    unknown id, so a single upload run would have blanked the states of every
    real run listed beside it."""
    asked = []

    def _record(cmd, timeout):
        asked.append(" ".join(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="9876543|COMPLETED\n", stderr="")

    local = srv._local_job_id("tiling")
    with _Patched(srv, _run_slurm=_record):
        states = srv._slurm_states_by_job([local, "9876543"])

    assert states[local] == {"COMPLETED"}
    assert states["9876543"] == {"COMPLETED"}
    assert asked and local not in asked[0], f"the sentinel reached sacct: {asked}"
    assert "9876543" in asked[0], "the real id was dropped along with the sentinel"


def test_one_sentinel_counts_as_one_finished_task(_tmp=None):
    """tiling_complete is "no in-flight state present" over these counts, so an
    upload whose Stage 1 never went through sbatch has to answer here."""
    with _Patched(srv, _run_slurm=_slurm_must_not_be_called):
        counts = srv._get_slurm_array_state_counts([srv._local_job_id("tiling")])
    assert counts == {"COMPLETED": 1}
    assert not (set(counts) & srv.IN_FLIGHT_SLURM_STATES)


def test_a_real_job_with_no_accounting_row_is_not_answered_by_a_sentinel(_tmp=None):
    """Proves the sentinel's count is added to sacct's answer rather than
    standing in for it: a real job Slurm cannot account for must still fall
    through to the live queue, even when a sentinel sits beside it."""
    consulted = []

    with _Patched(
        srv,
        _run_slurm=lambda cmd, timeout: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        _slurm_jobs_live_states=lambda ids: consulted.append(ids) or ["RUNNING"],
    ):
        counts = srv._get_slurm_array_state_counts([srv._local_job_id("tiling"), "9876543"])

    assert consulted == [["9876543"]], consulted
    assert counts == {"COMPLETED": 1, "RUNNING": 1}, counts


def test_a_sentinel_does_not_blank_the_dataset_listing(_tmp=None):
    """The listing asks squeue about every run's jobs in chunks, and squeue
    fails the whole chunk for one id the controller never had. An upload run
    sharing a chunk with fifty real ones must not be what makes them all read
    as "no record"."""
    asked = []

    def _fake_squeue(cmd, **kwargs):
        asked.append(" ".join(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="9876543|RUNNING\n", stderr="")

    local = srv._local_job_id("tiling")
    with _Patched(subprocess, run=_fake_squeue):
        states = srv._squeue_states_by_job([local, "9876543"])

    assert states[local] == {"COMPLETED"}
    assert states["9876543"] == {"RUNNING"}
    assert asked and local not in asked[0], f"the sentinel reached squeue: {asked}"


def test_a_sentinel_is_never_handed_to_scancel(_tmp=None):
    """scancel rejects the call for an id it does not know, which would take
    the run's real jobs down with it into scancel_error."""
    real, local = srv._split_local_job_ids(["123", srv._local_job_id("tiling"), "456"])
    assert real == ["123", "456"]
    assert local == [srv._local_job_id("tiling")]


# --- one cohort per uploaded slide ----------------------------------------

def test_each_uploaded_slide_gets_its_own_cohort(_tmp=None):
    """Two uploads must not share a dataset_id. register_dataset.commit()
    refuses to re-register an occupied cohort without --replace, and --replace
    DELETEs every row under that dataset_id — so a shared cohort would mean the
    second uploaded slide could only be registered by deleting the first."""
    assert srv.upload_dataset_name("slide-a") != srv.upload_dataset_name("slide-b")
    assert srv.upload_dataset_name("slide-a") == "UPLOADED_SLIDE-A"


def test_the_cohort_name_is_also_the_tile_folder(_tmp=None):
    """Registration reads Stage 1's _tile_metadata.csv out of
    tile_dir/<the run's dataset_name>/<slide_id>/. The upload pipeline writes
    its tiles under the same name, which is what makes registering an upload
    need no special case at all.

    Also what a re-upload moves aside: the pipeline's tiler writes exactly the
    folders _move_upload_tiles_aside moves, and a mismatch would leave the old
    tiles where the new run adopts them."""
    from submit_mask_tile_slurm import slide_output_paths

    masks, tiles = Path("/m"), Path("/t")
    name = srv.upload_dataset_name("TCGA-55-7574")
    saved = Path(f"/raw/uuid/TCGA-55-7574_{'0' * 8}-{'0' * 4}-{'0' * 4}-{'0' * 4}-{'0' * 12}_x.svs")
    paths = slide_output_paths(saved, masks, tiles, name)
    assert paths["dataset_tile_dir"] == tiles / name
    assert paths["dataset_mask_dir"] == masks / name
    assert paths["slide_tile_dir"] == tiles / name / "TCGA-55-7574"


def test_an_upload_from_before_this_is_still_recognised_as_an_upload(_tmp=None):
    """Rows written before per-slide cohorts carry the bare "UPLOADED". Reading
    one as somebody else's dataset would turn re-uploading such a slide into a
    hard 409 about a cohort collision that does not exist."""
    assert srv._is_upload_dataset_id("UPLOADED")
    assert srv._is_upload_dataset_id(srv.upload_dataset_name("TCGA-55-7574"))


def test_a_curated_cohort_is_not_mistaken_for_an_upload(_tmp=None):
    """The half that guards a real slide: an upload must never be allowed to
    repoint a curated cohort's registry row at whatever was just uploaded."""
    for dataset_id in ("TCGA_LUAD_5x", "Radiogenomics", "", None):
        assert not srv._is_upload_dataset_id(dataset_id), dataset_id


def test_the_upload_registers_the_slide_under_that_cohort(_tmp=None):
    """The viewer needs a wsi_registry row immediately, and Stage 5 refuses to
    register a slide that already belongs to a *different* dataset_id — so the
    row written at upload time has to carry the cohort Stage 5 will use."""
    call = _SERVER_SOURCE[_SERVER_SOURCE.index("        _register_uploaded_slide("):]
    call = call[:call.index("\n\n")]
    assert "upload_dataset_name(safe_user_slide_id)" in call, call


# --- the name the raw file is saved under ---------------------------------

def test_the_saved_name_gives_the_slide_id_back(_tmp=None):
    """Registration matches raw files by slide_id_from_raw_path(). A name it
    cannot parse does not fail: the cohort registers with no wsi_registry row
    and every tile of that slide 404s in the viewer."""
    uuid4 = "0f4f2f8c-1a2b-4c3d-8e9f-abcdef012345"
    saved = Path(f"/scratch/uploads/raw/{uuid4}/TCGA-55-7574_{uuid4}_original_scan.svs")
    assert slide_id_from_raw_path(saved) == "TCGA-55-7574"


def test_the_name_without_the_uuid_does_not_parse(_tmp=None):
    """Why the line above is load-bearing rather than decorative — this is the
    shape uploads were saved under, and the whole stem comes back as the slide
    id."""
    saved = Path("/scratch/uploads/raw/0f4f2f8c/TCGA-55-7574_original_scan.svs")
    assert slide_id_from_raw_path(saved) != "TCGA-55-7574"


def test_the_endpoint_saves_under_the_parseable_name(_tmp=None):
    """Pins the two together: the test above proves the convention, this proves
    /upload-slide still writes it."""
    assert 'save_dir / f"{safe_user_slide_id}_{internal_id}_{safe_filename}"' in _SERVER_SOURCE


# --- the upload's run is a pipeline run -----------------------------------

class _Refused(Exception):
    pass


def _dirs(tmp_path):
    masks, tiles = tmp_path / "masks", tmp_path / "tiles"
    return masks, tiles


def _previous_upload(tmp_path, slide_id="TCGA-55-7574"):
    """The mask and tiles a previous upload of this slide_id left on disk,
    plus a neighbouring cohort's that must never be touched."""
    masks, tiles = _dirs(tmp_path)
    name = srv.upload_dataset_name(slide_id)
    for root in (masks, tiles):
        (root / name / slide_id).mkdir(parents=True)
        (root / name / slide_id / "old.jpeg").write_bytes(b"old")
        (root / "TCGA").mkdir()
    return name


def _start(tmp_path, create=None, update=None):
    """_start_upload_pipeline_run with the run creation stubbed out."""
    masks, tiles = _dirs(tmp_path)
    calls = []

    def _fake_create(req, raw_dir, dataset_name):
        calls.append((req, raw_dir, dataset_name))
        if create is not None:
            return create(req, raw_dir, dataset_name)
        return {"submission_id": "run-1"}, ("run-1",)

    with _Patched(srv, TISSUE_MASK_DIR=masks, PROCESSED_TILES_DIR=tiles,
                  _create_pipeline_run=_fake_create,
                  _update_dataset_run=update or (lambda *a, **k: None)):
        response, submission = srv._start_upload_pipeline_run(
            "TCGA-55-7574", tmp_path / "raw" / "uuid-dir")
    return calls, response, submission


def test_an_upload_is_submitted_as_a_pipeline_run_over_its_own_directory(_tmp=None):
    """The whole change: one run, over a directory holding exactly this slide,
    under the slide's own cohort, with every setting the server's — the
    checkpoint in particular, which is no longer anybody's to type."""
    tmp_path = Path(tempfile.mkdtemp(prefix="hpl_upload_test_"))
    calls, response, submission = _start(tmp_path)

    (req, raw_dir, dataset_name), = calls
    assert raw_dir == tmp_path / "raw" / "uuid-dir"
    assert dataset_name == srv.upload_dataset_name("TCGA-55-7574")
    assert req.dataset_name == dataset_name
    assert req.checkpoint is None and req.reference is None, "not the server's defaults"
    assert req.vote_preset == srv.DEFAULT_VOTE_PRESET
    assert not req.sample_size and not req.slide_names
    # A re-upload's earlier .h5/projections/assignments are moved aside, not
    # refused: the upload has no other button to press.
    assert req.move_existing_outputs is True
    assert submission == ("run-1",)


def test_a_reupload_moves_the_old_tiles_aside(_tmp=None):
    tmp_path = Path(tempfile.mkdtemp(prefix="hpl_upload_test_"))
    name = _previous_upload(tmp_path)
    _, response, _ = _start(tmp_path)

    masks, tiles = _dirs(tmp_path)
    for root in (masks, tiles):
        assert not (root / name).exists(), "the new run would adopt these tiles"
        kept = list((root / ".superseded").glob(f"{name}-*/TCGA-55-7574/old.jpeg"))
        assert kept, "moved, never deleted"
        assert (root / "TCGA").is_dir(), "another cohort was touched"
    assert len(response["superseded_tiles"]) == 2


def test_the_moved_tiles_are_not_offered_as_a_dataset_folder(_tmp=None):
    tmp_path = Path(tempfile.mkdtemp(prefix="hpl_upload_test_"))
    _previous_upload(tmp_path)
    _start(tmp_path)
    with _Patched(srv, PROCESSED_TILES_DIR=_dirs(tmp_path)[1]):
        names = srv.list_tile_dataset_names()["dataset_names"]
    assert names == ["TCGA"], names


def test_a_refused_run_leaves_the_old_tiles_where_they_were(_tmp=None):
    """The move comes after the run exists. Moved first, a refusal (a missing
    checkpoint, say) would leave the previous upload's run pointing at tiles
    that are no longer there."""
    tmp_path = Path(tempfile.mkdtemp(prefix="hpl_upload_test_"))
    name = _previous_upload(tmp_path)

    def _refuse(*args):
        raise _Refused("checkpoint not found")

    try:
        _start(tmp_path, create=_refuse)
    except _Refused:
        pass
    else:
        raise AssertionError("the refusal did not reach the caller")
    for root in _dirs(tmp_path):
        assert (root / name / "TCGA-55-7574" / "old.jpeg").is_file()


def test_tiles_that_cannot_be_moved_stop_the_run(_tmp=None):
    """If the old tiles stay, the run must not start: its tiling would keep
    them. The row says so, and the caller never queues the submission."""
    tmp_path = Path(tempfile.mkdtemp(prefix="hpl_upload_test_"))
    _previous_upload(tmp_path)
    updates = []

    def _cannot_move(*args, **kwargs):
        raise OSError("read-only file system")

    with _Patched(srv, _move_upload_tiles_aside=_cannot_move):
        try:
            _start(tmp_path, update=lambda sid, **fields: updates.append((sid, fields)))
        except srv.HTTPException as e:
            assert "earlier tiles" in e.detail
        else:
            raise AssertionError("a run was handed back over tiles it would adopt")
    assert updates and updates[0][0] == "run-1"
    assert updates[0][1]["status"] == "error"


def test_the_endpoint_queues_the_run_it_created(_tmp=None):
    upload = _SERVER_SOURCE[_SERVER_SOURCE.index("async def upload_slide("):]
    upload = upload[:upload.index("\n@app.get")]
    assert "_start_upload_pipeline_run(safe_user_slide_id, save_dir)" in upload
    assert "background_tasks.add_task(_run_pipeline_submission, *submission)" in upload
    # save_dir, not the shared UPLOAD_RAW_DIR: discovery walks raw_dir, and
    # the pool holds every upload ever made.
    assert "_start_upload_pipeline_run(safe_user_slide_id, UPLOAD_RAW_DIR" not in upload


# --- a re-upload while the last run is still going --------------------------

def _in_flight(state):
    row = {"submission_id": "run-0", "job_id": "nf:run-0:tiling",
           "h5_job_id": "nf:run-0:packaging", "extraction_job_id": "nf:run-0:extraction",
           "assignment_job_id": "nf:run-0:assignment"}
    with _Patched(srv,
                  _upload_run_for_slide=lambda slide_id: {"submission_id": "run-0"},
                  _get_dataset_run_row=lambda sid: dict(row),
                  _get_slurm_job_state=lambda job_id: state):
        return srv._upload_run_in_flight("TCGA-55-7574")


def test_a_reupload_waits_for_a_running_run(_tmp=None):
    assert "run-0" in _in_flight("RUNNING")
    assert "run-0" in _in_flight("PENDING")


def test_a_reupload_waits_when_slurm_cannot_say(_tmp=None):
    """Unreachable is not "nothing running" — moving tiles out from under a
    live tiling task is the worse outcome."""
    assert "unknown" in _in_flight(None)


def test_a_finished_or_stopped_run_does_not_block_a_reupload(_tmp=None):
    for state in ("COMPLETED", "FAILED", "CANCELLED", ""):
        assert _in_flight(state) is None, state


def test_the_in_flight_check_comes_before_the_overwrite_prompt(_tmp=None):
    """Otherwise the user confirms the overwrite and is refused anyway."""
    upload = _SERVER_SOURCE[_SERVER_SOURCE.index("async def upload_slide("):]
    assert upload.index('"upload_run_in_progress"') < upload.index('"slide_id_exists"')


# --- the upload panel's status, read off the pipeline -----------------------

def _progress(tmp_path, head_state, started=(), done=(), row=None):
    from hpl_nf_state import mark_done, mark_started
    run_dir = tmp_path / "run-1"
    for stage in started:
        mark_started(run_dir, stage)
    for stage in done:
        mark_started(run_dir, stage)
        mark_done(run_dir, stage, {"rows": 1})
    row = row or {"submission_id": "run-1", "status": "submitted",
                  "job_id": "nf:run-1:tiling"}
    with _Patched(srv, HPL_NF_RESULTS_ROOT=tmp_path,
                  _get_dataset_run_row=lambda sid: dict(row),
                  _nf_head_state=lambda sid: head_state):
        return srv._upload_pipeline_progress("run-1")


def test_the_status_follows_the_pipeline(_tmp=None):
    def fresh():
        return Path(tempfile.mkdtemp(prefix="hpl_upload_test_"))

    assert _progress(fresh(), "PENDING")["status"] == "queued"
    assert _progress(fresh(), "RUNNING", started=["tiling"])["status"] == "tiling"
    assert _progress(fresh(), "RUNNING", done=["tiling"],
                     started=["packaging"])["status"] == "packaging"
    assert _progress(fresh(), "RUNNING", done=["tiling", "packaging"],
                     started=["extraction"])["status"] == "feature extraction"
    assert _progress(fresh(), "RUNNING", done=["tiling", "packaging", "extraction"],
                     started=["assignment"])["status"] == "classification"
    assert _progress(fresh(), "COMPLETED",
                     done=list(srv.NF_STAGES))["status"] == "done"


def test_a_run_that_stopped_is_an_error_naming_the_stage(_tmp=None):
    """The guard can come out bad: a head job that ended without the
    extraction marker must not read as done, or as still going."""
    tmp_path = Path(tempfile.mkdtemp(prefix="hpl_upload_test_"))
    progress = _progress(tmp_path, "COMPLETED", done=["tiling", "packaging"])
    assert progress["status"] == "error", progress
    assert "feature extraction" in progress["error"]


def test_a_refused_submission_is_an_error(_tmp=None):
    tmp_path = Path(tempfile.mkdtemp(prefix="hpl_upload_test_"))
    progress = _progress(tmp_path, None, row={
        "submission_id": "run-1", "status": "error", "job_id": "nf:run-1:tiling",
        "error": "sbatch: invalid partition"})
    assert progress == {"status": "error", "error": "sbatch: invalid partition"}


def test_an_older_upload_keeps_the_status_it_recorded(_tmp=None):
    """Uploads from before this ran Stages 1-2 in the server and wrote their
    progress to wsi_registry; nothing here may overwrite it."""
    tmp_path = Path(tempfile.mkdtemp(prefix="hpl_upload_test_"))
    assert _progress(tmp_path, None, row={
        "submission_id": "run-1", "status": "completed",
        "job_id": srv._local_job_id("tiling")}) is None


# --- the UI's route into the run ------------------------------------------

def test_the_processing_status_names_the_run(_tmp=None):
    """The upload response carries submission_id too, but it is gone the moment
    the page reloads — this endpoint is what the UI polls, so the run has to be
    reachable from it or the pipeline view is only available to the tab that
    started the upload."""
    payload = srv.slide_processing_status("TCGA-55-7574")
    assert "submission_id" in payload
    assert payload["dataset_name"] == srv.upload_dataset_name("TCGA-55-7574")
    assert "status" in payload


def test_the_upload_response_carries_the_run(_tmp=None):
    upload = _SERVER_SOURCE[_SERVER_SOURCE.index("async def upload_slide("):]
    upload = upload[:upload.index("\n@app.get")]
    assert '"submission_id": submission_id' in upload
    assert '"pipeline_error": pipeline_error' in upload


def _pipeline_steps():
    """app_v28._pipeline_steps, exec'd with stubs — same technique
    test_pipeline_steps.py uses, since importing that module pulls in
    streamlit."""
    source = APP.read_text()
    start = source.index("def _pipeline_steps")
    end = source.index("\ndef ", start)
    namespace = {
        "_SLURM_IN_FLIGHT": srv.IN_FLIGHT_SLURM_STATES,
        "_test_packaging_note": lambda status: "",
        "_human_bytes": lambda n: f"{n} B",
        "re": re,
    }
    exec(compile(source[start:end], "probe", "exec"), namespace)
    return namespace["_pipeline_steps"]


def test_the_stepper_offers_feature_extraction_for_an_older_uploaded_slide(_tmp=None):
    """An upload from before uploads were pipeline runs: Stages 1 and 2 read as
    done and Stage 3 is still the next thing to click."""
    steps = {s["key"]: s for s in _pipeline_steps()({
        "status": "completed",
        "total_slides": 1, "succeeded": 1, "tiling_complete": True,
        "h5_job_id": srv._local_job_id("packaging"),
        "h5_slurm_state": "COMPLETED",
        "h5_ready": True,
    })}
    assert steps["tiling"]["state"] == "done", steps["tiling"]
    assert steps["packaging"]["state"] == "done", steps["packaging"]
    assert steps["extraction"]["state"] == "action", steps["extraction"]


def test_the_upload_panel_renders_that_stepper(_tmp=None):
    """A run nothing renders is a run nobody can advance — the failure this
    whole change is about."""
    app_source = APP.read_text()
    assert "_render_upload_pipeline(processing_slide_id" in app_source
    panel = app_source[app_source.index("def _render_upload_pipeline"):]
    panel = panel[:panel.index("\ndef ", 1)]
    assert "_render_job_progress(" in panel
    assert 'status_payload.get("submission_id")' in panel


def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_upload_test_"))
        try:
            fn(tmp_path)
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
