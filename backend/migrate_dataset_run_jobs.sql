-- ============================================================
-- Per-job history for dataset runs.
-- Run once on the HPCC PostgreSQL instance.
--
-- Why: slurm_dataset_runs holds one column per stage — job_id, h5_job_id,
-- test_h5_job_id, extraction_job_id — so each stage remembers only its most
-- recent attempt. Repackaging, or a second test packaging with a different
-- sample, overwrites the previous job id and output path with no trace that it
-- ever ran.
--
-- Tiling looked like the exception only by accident: a resume mints a new
-- submission_id and therefore a whole new run row, so each tiling attempt
-- survives as its own entry in "Recent dataset jobs". Every other stage reuses
-- the row it started from.
--
-- This table records one row per submitted Slurm job, so history is a query
-- rather than a side effect of how a stage happens to be re-run.
--
-- Deliberately additive. The single-slot columns on slurm_dataset_runs stay
-- exactly as they are and remain what the packaging retry guard and the
-- h5_ready / extraction_ready gates read. Nothing changes behaviourally —
-- this only makes the attempts visible. Reading gating state from a history
-- table instead would mean deciding which of several rows is authoritative,
-- which is precisely the question the single slot already answers.
-- ============================================================

CREATE TABLE IF NOT EXISTS slurm_dataset_run_jobs (
    id SERIAL PRIMARY KEY,
    submission_id TEXT NOT NULL,
    -- tiling | packaging | packaging_test | extraction | extraction_test
    stage TEXT NOT NULL,
    -- Comma-joined for an array split across batches, matching
    -- slurm_dataset_runs.job_id. One row per *attempt* rather than per batch
    -- task: an attempt is the unit a reader is looking for, and 15 rows for one
    -- tiling submission would bury the four other stages.
    job_id TEXT NOT NULL,
    output_path TEXT,
    -- What the attempt was asked to do (sample_size, scope, seed, checkpoint).
    -- Without it, two test-packaging rows are indistinguishable but for their
    -- job ids, which is not what anyone is trying to tell apart.
    params JSONB,
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Makes recording idempotent: the tiling path can re-persist the same job ids
-- after a crash-recovery lookup, and a retry that resolves to an
-- already-submitted job must not add a duplicate row.
CREATE UNIQUE INDEX IF NOT EXISTS slurm_dataset_run_jobs_unique_idx
    ON slurm_dataset_run_jobs (submission_id, stage, job_id);

CREATE INDEX IF NOT EXISTS slurm_dataset_run_jobs_run_idx
    ON slurm_dataset_run_jobs (submission_id, submitted_at DESC);
