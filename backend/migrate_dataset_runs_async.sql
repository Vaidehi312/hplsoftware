-- ============================================================
-- Makes dataset job submission asynchronous.
-- Run once on the HPCC PostgreSQL instance, after migrate_dataset_runs.sql.
--
-- Why: POST /dataset-jobs used to discover every slide in a dataset
-- directory (a recursive filesystem walk) synchronously within the HTTP
-- request, before submitting to Slurm. On a large or network-mounted
-- dataset that walk alone can exceed any reasonable client timeout. The
-- old schema's primary key was job_id — but job_id isn't known until
-- *after* discovery finishes and sbatch succeeds, so there was no way to
-- hand the client an identifier to poll before that slow part completes.
--
-- submission_id is generated immediately, before discovery starts, and
-- becomes the new primary key. job_id, manifest_path, and total_slides
-- are now nullable — they get filled in once the background task
-- (discovery + sbatch) actually finishes.
-- ============================================================

ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS submission_id TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'submitted';
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS error TEXT;

-- Backfill submission_id for any rows inserted before this migration.
UPDATE slurm_dataset_runs SET submission_id = job_id WHERE submission_id IS NULL;
ALTER TABLE slurm_dataset_runs ALTER COLUMN submission_id SET NOT NULL;

ALTER TABLE slurm_dataset_runs DROP CONSTRAINT IF EXISTS slurm_dataset_runs_pkey;
ALTER TABLE slurm_dataset_runs ADD PRIMARY KEY (submission_id);

ALTER TABLE slurm_dataset_runs ALTER COLUMN job_id DROP NOT NULL;
ALTER TABLE slurm_dataset_runs ALTER COLUMN manifest_path DROP NOT NULL;
ALTER TABLE slurm_dataset_runs ALTER COLUMN total_slides DROP NOT NULL;
