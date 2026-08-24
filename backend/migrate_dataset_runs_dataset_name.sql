-- ============================================================
-- Adds an explicit dataset_name to each dataset run, so tiles for that run
-- land in tile_dir/<dataset_name>/<slide_id>/ instead of flat under
-- tile_dir/<slide_id>/ (TCGA and Radiogenomics were landing in the same
-- folder and getting mixed together). Run once on the HPCC PostgreSQL
-- instance, after migrate_dataset_runs_manual_stages.sql.
--
-- NULL for rows created before this migration (and for any future row
-- where the user didn't type a name) — callers fall back to
-- raw_dir's own folder name in that case, same as submit_mask_tile_slurm.py
-- already did before this column existed.
-- ============================================================

ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS dataset_name TEXT;
