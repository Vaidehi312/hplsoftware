-- ============================================================
-- Which Knowledge Bank each stage wrote to.
-- Run once on the HPCC PostgreSQL instance, after
-- migrate_dataset_runs_registration.sql.
--
-- The server can now write a cohort into either hpl_kb or hpl_kb_test, but
-- run tracking itself stays in production for every stage — a run is one run
-- regardless of which KB it filled, and splitting slurm_dataset_runs across
-- two databases would mean the pipeline UI could not list its own history.
--
-- That is the reason these columns have to exist. With the target recorded
-- nowhere, `registration_done = true` says a cohort was registered and not
-- where, so the UI would show a run as finished while the production KB it
-- appears to describe is empty. This is the same class of problem as
-- kb_load_reference: a boolean that is only meaningful next to the thing it
-- was true of.
--
-- Defaulting to NULL rather than 'production': a NULL here means "this run
-- predates targets", which is not quite the same claim as "this run wrote to
-- production" even though both read that way today. The server coalesces it
-- for display; the column keeps the distinction.
-- ============================================================

ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS registration_kb_target TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS kb_load_kb_target TEXT;
