"""Stage 7's live view: which head job is carrying the run, and how far it is.

Two things the stage got wrong on a real run (2026-09-30, 7,221 slides):

  * the head chain. The row recorded head job 1255076, which reached its
    two-day walltime; a head job and six standbys submitted by hand under the
    run's job name carried on. The stage asked only about the recorded id, so it
    read TIMEOUT — ended, with a Retry form — over a running pipeline, and a
    Retry would have queued a second head job over the live one, which is what
    the in-flight guard exists to stop;
  * progress. The head job's state says the pipeline is alive and nothing
    about how far through it is, or why 192 GPU tasks sat PENDING (the
    account's GPU cap, which reads like a stall).

Slurm is stubbed at _run_slurm, the one place the server shells out for these
queries, so the parsing of squeue's and sacct's output is exercised too.
"""

import contextlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tile_server_v2_ as srv  # noqa: E402
from test_anorak_server import _FakeSubmit, _Patched, _refused, _setup  # noqa: E402

HEAD_NAME = "anorak_sub1"


def _clear_caches():
    for cache in (srv._ANORAK_HEAD_CACHE, srv._ANORAK_QUEUE_CACHE,
                  srv._ANORAK_TRACE_CACHE, srv._ANORAK_SLIDE_COUNT_CACHE):
        cache.clear()


class _FakeSlurm:
    """_run_slurm for the three queries Stage 7 makes. None for a query means
    that tool is unreachable, which _run_slurm reports as None."""

    def __init__(self, *, heads="", ended="", tasks=""):
        self.heads, self.ended, self.tasks = heads, ended, tasks
        self.calls = []

    def __call__(self, cmd, timeout):
        self.calls.append(cmd)
        if cmd[0] == "squeue" and "-n" in cmd:
            assert cmd[cmd.index("-n") + 1] == HEAD_NAME
            out = self.heads
        elif cmd[0] == "sacct":
            assert cmd[cmd.index("--name") + 1] == HEAD_NAME
            out = self.ended
        elif cmd[0] == "squeue":
            out = self.tasks
        else:
            raise AssertionError(f"unexpected Slurm call {cmd}")
        return None if out is None else SimpleNamespace(stdout=out, returncode=0, stderr="")


# The live run as squeue had it: the hand-submitted head and six standbys.
LIVE_CHAIN = "1264266|RUNNING|1-10:21:07|None|\n" + "".join(
    f"{j}|PENDING|2-00:00:00|Dependency|afternotok:{j - 1}(unfulfilled)\n"
    for j in range(1264267, 1264273))


def _patches(row, recorded, slurm, submit=None):
    """recorded maps a recorded job id to what _get_slurm_job_state says of it."""
    _clear_caches()
    return _Patched(
        srv,
        _slurm_submission_lock=contextlib.nullcontext,
        _get_dataset_run_row=lambda sid: dict(row),
        _get_slurm_job_state=lambda job_id: recorded.get(job_id),
        _run_slurm=slurm,
        submit_anorak_job=submit or _FakeSubmit(),
        _update_dataset_run_best_effort=lambda sid, **f: None,
        _record_run_job=lambda *a, **k: None,
    )


# --- head-job discovery -------------------------------------------------------

def test_a_head_job_found_by_name_is_in_flight_and_blocks_a_resubmission(tmp_path):
    """The recorded job TIMEOUT, a live one under the run's name: in flight,
    and a Retry is refused before submit_anorak_job can rewrite slide_list.csv.
    Fails with the lookup removed — the stage reads TIMEOUT and accepts."""
    row, source, _ = _setup(tmp_path, anorak_job_id="1255076")
    submit = _FakeSubmit()
    slurm = _FakeSlurm(heads=LIVE_CHAIN, ended="1255076|TIMEOUT\n")
    with _patches(row, {"1255076": "TIMEOUT"}, slurm, submit):
        fields = srv._anorak_status_fields(row)
        err = _refused(lambda: srv.start_anorak_job(
            "sub1", srv.AnorakRequest(slides_csv=str(source), overwrite=True)))
    assert fields["anorak_in_flight"] is True
    assert fields["anorak_slurm_state"] == "RUNNING"
    assert fields["anorak_submit_blocked"]
    assert err.status_code == 400 and "already running" in err.detail
    assert "1264266" in err.detail
    assert submit.calls == [], "a second head job was submitted over the live one"


def test_the_live_head_its_time_left_and_the_standbys_are_reported(tmp_path):
    row, _, _ = _setup(tmp_path, anorak_job_id="1255076")
    slurm = _FakeSlurm(heads=LIVE_CHAIN, ended="1255076|TIMEOUT\n")
    with _patches(row, {"1255076": "TIMEOUT"}, slurm):
        fields = srv._anorak_status_fields(row)
    assert fields["anorak_head_job_id"] == "1264266"
    assert fields["anorak_head_state"] == "RUNNING"
    assert fields["anorak_head_time_left"] == "1-10:21:07"
    assert fields["anorak_standbys_queued"] == 6
    assert fields["anorak_standby_job_ids"] == [str(j) for j in range(1264267, 1264273)]
    assert fields["anorak_recorded_state"] == "TIMEOUT"
    assert fields["anorak_chain_took_over"] is True
    # The recorded id is still what the row says; the live one is beside it.
    assert fields["anorak_job_id"] == "1255076"


def test_no_standby_is_reported_as_none(tmp_path):
    row, _, _ = _setup(tmp_path, anorak_job_id="1264266")
    slurm = _FakeSlurm(heads="1264266|RUNNING|0-03:00:00|None|\n", ended="")
    with _patches(row, {"1264266": "RUNNING"}, slurm):
        fields = srv._anorak_status_fields(row)
    assert fields["anorak_standbys_queued"] == 0
    assert fields["anorak_chain_took_over"] is False
    assert fields["anorak_head_job_id"] == "1264266"


def test_squeue_unreachable_by_name_stays_unknown(tmp_path):
    """A recorded TIMEOUT plus a by-name lookup that could not be made is not
    "ended": a head job submitted by hand may be running. Unknown, and a
    resubmission is refused with 503. Fails if a failed lookup reads as []."""
    row, source, _ = _setup(tmp_path, anorak_job_id="1255076")
    submit = _FakeSubmit()
    slurm = _FakeSlurm(heads=None, ended=None, tasks=None)
    with _patches(row, {"1255076": "TIMEOUT"}, slurm, submit):
        fields = srv._anorak_status_fields(row)
        err = _refused(lambda: srv.start_anorak_job(
            "sub1", srv.AnorakRequest(slides_csv=str(source))))
    assert fields["anorak_slurm_state"] is None
    assert fields["anorak_state_unknown"] is True
    assert fields["anorak_head_discovery_failed"] is True
    assert err.status_code == 503
    assert submit.calls == []


def test_sacct_unreachable_with_nothing_live_stays_unknown(tmp_path):
    """squeue answers (nothing live) but sacct cannot say whether a later head
    job finished the run: unknown, not the recorded TIMEOUT."""
    row, _, _ = _setup(tmp_path, anorak_job_id="1255076")
    with _patches(row, {"1255076": "TIMEOUT"}, _FakeSlurm(heads="", ended=None)):
        assert srv._anorak_run_state(row) is None


def test_a_later_head_job_that_completed_finishes_the_run_but_an_earlier_one_does_not(tmp_path):
    """The chain a hand-submitted head carried finished: COMPLETED, although the
    recorded job timed out. A COMPLETED head job from before the recorded one
    is an earlier attempt and says nothing about this one."""
    row, _, _ = _setup(tmp_path, anorak_job_id="1255076")
    with _patches(row, {"1255076": "TIMEOUT"},
                  _FakeSlurm(heads="", ended="1255076|TIMEOUT\n1264266|COMPLETED\n")):
        assert srv._anorak_run_state(row) == "COMPLETED"
    with _patches(row, {"1255076": "TIMEOUT"},
                  _FakeSlurm(heads="", ended="1200000|COMPLETED\n1255076|TIMEOUT\n")):
        assert srv._anorak_run_state(row) == "TIMEOUT"


def test_the_submit_path_does_not_trust_a_cached_none(tmp_path):
    """A status poll caches "nothing live" for a few seconds; a head job
    submitted by hand inside that window must still block a submission."""
    row, source, _ = _setup(tmp_path, anorak_job_id="1255076")
    submit = _FakeSubmit()
    slurm = _FakeSlurm(heads="", ended="1255076|TIMEOUT\n")
    with _patches(row, {"1255076": "TIMEOUT"}, slurm, submit):
        assert srv._anorak_run_state(row) == "TIMEOUT"   # cached now
        slurm.heads = "1264266|PENDING|2-00:00:00|Priority|\n"
        err = _refused(lambda: srv.start_anorak_job(
            "sub1", srv.AnorakRequest(slides_csv=str(source))))
    assert err.status_code == 400 and submit.calls == []


# --- trace.txt ----------------------------------------------------------------

TRACE_ROWS = [
    # task_id, name, status
    ("1", "TILE_SLIDE (A)", "CACHED"),
    ("2", "TILE_SLIDE (B)", "COMPLETED"),
    ("3", "TILE_SLIDE (C)", "COMPLETED"),
    # Finished once, then a later attempt ended ABORTED (a head job stopped
    # while re-running it): still done — its output exists — so "done" is any
    # success, not the latest status.
    ("40", "TILE_SLIDE (C)", "ABORTED"),
    # A retry that succeeded: done, whatever the failed attempt said.
    ("9", "PREDICT_GP (A)", "COMPLETED"),
    ("5", "PREDICT_GP (A)", "FAILED"),
    # A retry still running, written *before* the failure it retries: latest
    # is by task_id, not by file order.
    ("12", "PREDICT_GP (B)", "RUNNING"),
    ("7", "PREDICT_GP (B)", "FAILED"),
    # Retries exhausted.
    ("8", "PREDICT_GP (C)", "FAILED"),
    ("10", "ANORAK:SS1_STITCH (A)", "SUBMITTED"),
]


def _trace(path: Path, order=("task_id", "hash", "native_id", "name", "status", "exit"),
           partial_tail=True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(order)]
    for task_id, name, status in TRACE_ROWS:
        values = {"task_id": task_id, "hash": "ab/cdef12", "native_id": "99" + task_id,
                  "name": name, "status": status, "exit": "0", "duration": "1m",
                  "realtime": "50s"}
        lines.append("\t".join(values.get(col, "-") for col in order))
    text = "\n".join(lines) + "\n"
    if partial_tail:
        text += "13\tab/12"   # a line still being written
    path.write_text(text)
    return path


EXPECTED = {
    "TILE_SLIDE": {"done": 3, "running": 0, "waiting": 0, "failed": 0, "tasks": 3},
    "PREDICT_GP": {"done": 1, "running": 1, "waiting": 0, "failed": 1, "tasks": 3},
    "SS1_STITCH": {"done": 0, "running": 0, "waiting": 1, "failed": 0, "tasks": 1},
}


def test_trace_counts_cached_completed_and_retries(tmp_path):
    """Fails if a FAILED attempt counts against a task a retry finished, or if
    latest-attempt is taken by file order."""
    _clear_caches()
    parsed = srv._anorak_parse_trace(_trace(tmp_path / "trace.txt"))
    assert parsed["available"] is True
    assert parsed["processes"] == EXPECTED
    assert parsed["skipped_rows"] == 1


def test_trace_columns_are_found_by_header_not_position(tmp_path):
    _clear_caches()
    shuffled = ("status", "duration", "exit", "name", "realtime", "hash", "task_id", "native_id")
    parsed = srv._anorak_parse_trace(_trace(tmp_path / "trace.txt", order=shuffled))
    assert parsed["processes"] == EXPECTED


def test_a_missing_or_headerless_trace_is_reported_not_counted(tmp_path):
    _clear_caches()
    missing = srv._anorak_parse_trace(tmp_path / "nope" / "trace.txt")
    assert missing["available"] is False and "does not exist" in missing["error"]
    bad = tmp_path / "bad.txt"
    bad.write_text("a\tb\n1\t2\n")
    parsed = srv._anorak_parse_trace(bad)
    assert parsed["available"] is False and "'name'" in parsed["error"]


def test_the_parse_is_cached_and_redone_when_the_file_changes(tmp_path):
    _clear_caches()
    path = _trace(tmp_path / "trace.txt", partial_tail=False)
    first = srv._anorak_parse_trace(path)
    assert srv._anorak_parse_trace(path) is first
    with path.open("a") as fh:
        fh.write("20\tab/1\t1\tTILE_SLIDE (D)\tCOMPLETED\t0\n")
    assert srv._anorak_parse_trace(path)["processes"]["TILE_SLIDE"]["done"] == 4


def test_totals_come_from_the_runs_slide_list(tmp_path):
    """Per-slide steps out of slide_list.csv's rows, TUMOUR_GRADE out of 1; a
    PUBLISH_SLIDE the run never ran is not shown as stuck at 0."""
    _clear_caches()
    out = tmp_path / "out"
    _trace(out / "pipeline_info" / "trace.txt")
    (out / "slide_list.csv").write_text(
        'slide_id,samples\nA,T1\nB,T1\n"C, with a comma",T2\nD,T3\nE,T3\n')
    (out / "ss1_final").mkdir()
    for name in ("A", "B"):
        (out / "ss1_final" / name).mkdir()
    row = {"anorak_slide_list": str(out / "slide_list.csv"), "anorak_slides": 999}
    progress = srv._anorak_progress(row, out, in_flight=False)
    steps = {s["process"]: s for s in progress["steps"]}
    assert progress["slides"] == 5
    assert steps["TILE_SLIDE"]["total"] == 5 and steps["TILE_SLIDE"]["done"] == 3
    assert steps["TUMOUR_GRADE"]["total"] == 1
    assert "PUBLISH_SLIDE" not in steps
    assert progress["stitched_slides"] == 2
    assert progress["counts_source"] == "trace"


# --- task jobs and why they wait ------------------------------------------------

def test_task_jobs_are_matched_by_work_dir_and_their_reasons_explained(tmp_path):
    """Another run's nf-PREDICT_GP jobs carry the same names; only the working
    directory tells them apart. Fails if matched by name, or by a prefix that
    also takes <out>/work2."""
    _clear_caches()
    out = tmp_path / "out"
    work = out / "work"
    other = tmp_path / "other" / "work"
    tasks = "".join([
        *(f"{3000 + i}|PENDING|AssocGrpGRES|{work}/ab/{i:04d}|nf-PREDICT_GP_(S{i})\n"
          for i in range(192)),
        f"4000|RUNNING|None|{work}/cd/0001|nf-PREDICT_GP_(S900)\n",
        f"4001|PENDING|Priority|{work}/cd/0002|nf-SS1_STITCH_(S901)\n",
        # Not this run's:
        *(f"{5000 + i}|PENDING|AssocGrpGRES|{other}/ab/{i:04d}|nf-PREDICT_GP_(S{i})\n"
          for i in range(50)),
        f"6000|PENDING|Resources|{out}/work2/ab/1|nf-PREDICT_GP_(X)\n",
        f"6001|RUNNING|None|{out}|anorak_sub1\n",   # the head job itself
    ])
    with _Patched(srv, _run_slurm=_FakeSlurm(tasks=tasks)):
        queue = srv._anorak_task_queue(out)
    assert queue["jobs"] == 194
    assert queue["by_state"] == {"PENDING": 193, "RUNNING": 1}
    assert queue["by_process"]["PREDICT_GP"] == {"running": 1, "waiting": 192}
    assert queue["waiting"][0]["reason"] == "AssocGrpGRES"
    assert queue["waiting"][0]["count"] == 192
    assert "GPU limit" in queue["waiting"][0]["explanation"]
    assert {w["reason"] for w in queue["waiting"]} == {"AssocGrpGRES", "Priority"}


def test_while_in_flight_running_and_waiting_come_from_the_queue(tmp_path):
    _clear_caches()
    out = tmp_path / "out"
    _trace(out / "pipeline_info" / "trace.txt")
    (out / "slide_list.csv").write_text("slide_id,samples\nA,T\nB,T\nC,T\n")
    tasks = f"1|PENDING|AssocGrpGRES|{out}/work/ab/1|nf-PREDICT_GP_(C)\n"
    with _Patched(srv, _run_slurm=_FakeSlurm(tasks=tasks)):
        progress = srv._anorak_progress({}, out, in_flight=True)
    steps = {s["process"]: s for s in progress["steps"]}
    assert progress["counts_source"] == "squeue"
    assert steps["PREDICT_GP"]["waiting"] == 1 and steps["PREDICT_GP"]["running"] == 0
    assert steps["PREDICT_GP"]["done"] == 1 and steps["PREDICT_GP"]["failed"] == 1
    assert progress["task_queue"]["waiting"][0]["reason"] == "AssocGrpGRES"


def test_an_unreachable_queue_is_none_not_empty(tmp_path):
    _clear_caches()
    with _Patched(srv, _run_slurm=_FakeSlurm(tasks=None)):
        assert srv._anorak_task_queue(tmp_path / "out") is None


# --- Streamlit, by source, as test_anorak_server does -------------------------

def test_the_streamlit_stage_shows_the_head_chain_and_progress(_tmp=None):
    app = (BACKEND.parent / "app" / "app_v28.py").read_text()
    jsx = (BACKEND.parent / "frontend" / "src" / "components" / "pipeline"
           / "stages" / "AnorakStage.jsx").read_text()
    for field in ("anorak_head_job_id", "anorak_head_time_left", "anorak_standby_job_ids",
                  "anorak_chain_took_over", "anorak_progress", "stitched_slides", "task_queue"):
        assert field in app, field
        assert field in jsx, field
    sentence = ("If this head job reaches its time limit, the next standby resumes the run; "
                "finished slides are kept.")
    assert sentence in app.replace('"\n    "', "")
    assert sentence in jsx
    assert "No standby head job is queued: nothing will take over" in app
    # Both the Stage 7 view and the ANORAK-run view carry it.
    assert app.count("_render_anorak_head_chain(") == 3
    assert app.count("_render_anorak_progress(status)") >= 3


def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="anorak_progress_test_"))
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
