#!/usr/bin/env python3
"""Write a stage's outcome back onto slurm_dataset_runs, from inside a job.

Stages 5 and 6 run as Slurm jobs now, so the process that knows whether the
write committed is the job, not the server. This is how it says so.

Two things it does not do, both deliberate:

  * It never raises into the caller's exit status. Every caller reaches it on
    the far side of a committed Knowledge Bank transaction, so failing the job
    because the bookkeeping row could not be updated would report a write that
    happened as a write that did not — the more misleading of the two errors. It
    prints loudly instead, and the next preview will show the rows are there.

  * It does not import the tile server. That module builds a FastAPI app and
    opens HDF5 handles at import time, which is not something a batch job should
    inherit to run one UPDATE.

Run tracking lives in the production database whichever Knowledge Bank the rows
went into, which is why the database name is passed explicitly rather than read
from DB_NAME — the job has DB_NAME pointed at the KB it is writing.
"""

from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine, text


def run_engine(run_db_name: str):
    user = os.getenv("DB_USER", os.getenv("USER", ""))
    password = os.getenv("DB_PASS", "")
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = os.getenv("DB_PORT", "5432")
    return create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{run_db_name}",
        pool_pre_ping=True,
    )


def record_run(run_db_name: str, submission_id: str, **fields) -> bool:
    """UPDATE slurm_dataset_runs for one run. True if it took, False if not."""
    if not fields:
        return True
    try:
        engine = run_engine(run_db_name)
        set_clause = ", ".join(f"{k} = :{k}" for k in fields)
        with engine.begin() as conn:
            conn.execute(
                text(f"UPDATE slurm_dataset_runs SET {set_clause} "
                     f"WHERE submission_id = :submission_id"),
                {**fields, "submission_id": submission_id},
            )
        return True
    except Exception as e:  # noqa: BLE001 — reported, never fatal; see docstring
        print(
            f"\nWARNING: could not record this stage's outcome against run "
            f"{submission_id} in {run_db_name}: {type(e).__name__}: {e}\n"
            f"The Knowledge Bank write itself is unaffected — whatever this job "
            f"printed above is what happened. The run will show the stage as not "
            f"done, and a preview will report the rows as already present.",
            file=sys.stderr,
        )
        return False
