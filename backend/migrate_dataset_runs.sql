-- ============================================================
-- Tracks dataset-wide Slurm masking+tiling jobs submitted from the UI.
-- Run once on the HPCC PostgreSQL instance.
--
-- Why: submit_mask_tile_slurm.py submits one Slurm array job per dataset.
-- This table is the durable record of "what got submitted, with what job
-- ID, pointed at which manifest" so the UI can list past/active runs and
-- poll their status (via sacct + the per-slide summary files) later,
-- across server restarts.
-- ============================================================

CREATE TABLE IF NOT EXISTS slurm_dataset_runs (
    job_id TEXT PRIMARY KEY,
    raw_dir TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    mask_dir TEXT NOT NULL,
    tile_dir TEXT NOT NULL,
    total_slides INTEGER NOT NULL,
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
