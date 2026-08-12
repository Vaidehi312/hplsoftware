-- ============================================================
-- Stage 5 (Knowledge Bank load) tracking on slurm_dataset_runs.
-- Run once on the HPCC PostgreSQL instance, after migrate_dataset_runs_assignment.sql.
--
-- Unlike stages 1-4 this stage submits no Slurm job — load_hpc_assignments.py's
-- logic runs in-process inside the server and either commits in one
-- transaction or doesn't, so there is no job_id/slurm_state pair to track.
-- kb_load_done is the single fact the UI gates the "loaded" state on.
--
-- kb_load_reference mirrors assignment_reference for the same reason:
-- tile_registry.hpc_id is only meaningful relative to one reference, and this
-- records which one this run's load actually committed, independent of
-- whichever reference the run's Stage 4 output happens to point at right now.
-- ============================================================

ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS kb_load_done BOOLEAN DEFAULT FALSE;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS kb_load_at TIMESTAMPTZ;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS kb_load_rows INTEGER;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS kb_load_reference TEXT;
