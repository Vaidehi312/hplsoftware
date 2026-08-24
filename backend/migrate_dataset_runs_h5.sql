-- ============================================================
-- Adds tracking for the automatic .h5 packaging job that gets chained onto
-- a dataset's tiling array via Slurm's own --dependency mechanism.
-- Run once on the HPCC PostgreSQL instance, after migrate_dataset_runs_async.sql.
-- ============================================================

ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS h5_job_id TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS h5_output_path TEXT;
