-- ============================================================
-- Dataset registration tracking on slurm_dataset_runs.
-- Run once on the HPCC PostgreSQL instance, after
-- migrate_dataset_runs_kb_load.sql.
--
-- Registration is the step that creates a cohort's identity rows —
-- wsi_registry, wsi_metadata, dataset_config, tile_coordinates and
-- tile_registry — before Stage 5 fills in hpc_id. It existed only as a
-- hand-run CLI (register_dataset.py), so the UI could not show it, could not
-- gate on it, and had no way to say whether it had happened. Stage 5's own
-- 95% match-rate guard is what caught its absence, at 0%: with no rows to
-- UPDATE the load refuses, correctly, and says nothing about why.
--
-- Like Stage 5 and unlike stages 1-4, this submits no Slurm job — it runs
-- in-process in the server and either commits in one transaction or does not,
-- so there is no job_id/slurm_state pair to track.
--
-- registration_dataset_id is recorded separately from the run's own
-- dataset_name because they are different things: dataset_name is the folder
-- of slides on scratch, registration_dataset_id is the cohort key every KB row
-- is scoped by. They are usually related and are not required to match, and a
-- later --replace has to target the one that was actually written.
--
-- registration_raw_dir records where the slide files were found. wsi_registry
-- stores an absolute path per slide, so if the raw directory is ever moved
-- every one of those paths is stale; this is what says which run to
-- re-register.
-- ============================================================

ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS registration_done BOOLEAN DEFAULT FALSE;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS registration_at TIMESTAMPTZ;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS registration_dataset_id TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS registration_raw_dir TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS registration_rows JSONB;
