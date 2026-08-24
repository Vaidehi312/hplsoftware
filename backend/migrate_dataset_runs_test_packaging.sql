-- ============================================================
-- Make test packaging jobs observable across page reloads.
-- Run once on the HPCC PostgreSQL instance.
--
-- Why: start_test_packaging_job() submits a real Slurm job and returns its id
-- and output path, but persists neither. The only record was the Streamlit
-- session_state entry the UI happened to be holding, so a reload — or a second
-- browser tab, or a server restart — lost the job entirely. The job kept
-- running and the .h5 kept being written; there was simply nothing left that
-- knew about it, so the panel showed no Slurm state, no completion, and no
-- path to hand to feature extraction.
--
-- Deliberately separate columns rather than reusing h5_job_id/h5_output_path.
-- Those are what the packaging retry guard and the stage-3 h5_ready gate read,
-- and a test run must not satisfy either: it packages a deliberate subset, so
-- treating it as the run's packaging output would let feature extraction be
-- launched over a sample while presenting it as the full dataset. Keeping the
-- two apart is what preserves "a test attempt has no bearing on the real run"
-- while still letting the UI report on it.
-- ============================================================

ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS test_h5_job_id TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS test_h5_output_path TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS test_h5_submitted_at TIMESTAMPTZ;

-- What the sample was, so the UI can say "this job is from an earlier setup"
-- after a reload has discarded the form state it used to compare against.
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS test_h5_params JSONB;
