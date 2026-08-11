-- ============================================================
-- Stage 4 (HPC cluster assignment) tracking on slurm_dataset_runs.
-- Run once on the HPCC PostgreSQL instance.
--
-- Mirrors the extraction_* columns exactly, because the UI gates each stage on
-- "is there a job id" plus "is there a usable output", and a stage that stores
-- those two facts differently from its neighbours needs its own special case
-- in every place that reads them.
--
-- assignment_reference records which reference .npz produced the CSV. Cluster
-- IDs are only meaningful relative to one Leiden run plus the encoder
-- checkpoint the embeddings came from, so two runs assigned against different
-- references hold IDs that cannot be compared — and nothing in the CSV itself
-- says which one it was. Same reasoning as tile_registry.hpc_reference; this
-- is the run-level counterpart.
-- ============================================================

ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS assignment_job_id TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS assignment_output_path TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS assignment_reference TEXT;
