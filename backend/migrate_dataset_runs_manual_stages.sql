-- Packaging and feature extraction stop auto-chaining and become
-- user-triggered UI steps. is_subset/partition/notify_email are persisted
-- at submission time so the later manual /package and /resume calls can
-- reuse them instead of guessing; extraction_* columns track the new
-- feature-extraction stage the same way h5_job_id/h5_output_path already
-- track packaging.
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS is_subset BOOLEAN DEFAULT FALSE;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS partition TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS notify_email TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS extraction_job_id TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS extraction_output_path TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS extraction_checkpoint TEXT;
