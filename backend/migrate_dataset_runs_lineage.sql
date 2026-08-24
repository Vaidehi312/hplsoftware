-- ============================================================
-- Record which run a resume came from.
-- Run once on the HPCC PostgreSQL instance.
--
-- Why: POST /dataset-jobs/{id}/resume does not continue the run it was called
-- on. It works out which slides never got tiled and calls
-- _start_dataset_submission for just those, which mints a brand-new
-- submission_id. The response says resumed_from_submission_id, but nothing
-- stored it, so the relationship existed only in that one HTTP reply.
--
-- The consequence is a dataset that took three resumes to tile appearing as
-- four unrelated rows, each with its own single-slot job columns, none of them
-- able to say whether the dataset is finished. GET /datasets already puts them
-- back together by grouping on (raw_dir, dataset_name) — which works, because
-- resume passes both straight through — but that grouping cannot tell a resume
-- apart from the deliberate second run the UI offers under "Submit a separate
-- NEW run for this same path anyway". Those two mean opposite things: one is a
-- continuation of work, the other is an independent experiment that happens to
-- share a directory.
--
-- Nullable with no backfill, and no foreign key. Every existing row predates
-- this and genuinely has no recorded parent — writing a guess into them would
-- turn "we don't know" into a claim. Consumers must treat NULL as unknown
-- rather than as "not a resume", and fall back to the (raw_dir, dataset_name)
-- grouping, which is what GET /datasets does. No FK because the parent run is
-- only ever read for display; a parent deleted from the table should not take
-- its resumes down with it or block the delete.
-- ============================================================

ALTER TABLE slurm_dataset_runs
    ADD COLUMN IF NOT EXISTS resumed_from_submission_id TEXT;

-- Answers "what did this run spawn?" without scanning the table. Partial,
-- because the overwhelming majority of rows are not resumes and indexing their
-- NULLs would cost space for a lookup nobody performs.
CREATE INDEX IF NOT EXISTS idx_dataset_runs_resumed_from
    ON slurm_dataset_runs (resumed_from_submission_id)
    WHERE resumed_from_submission_id IS NOT NULL;
