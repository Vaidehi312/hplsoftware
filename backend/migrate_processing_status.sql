-- ============================================================
-- Adds durable post-upload pipeline status to wsi_registry.
-- Run once on the HPCC PostgreSQL instance.
--
-- Why: tile_server_v2's mask+tile background job used to track status in an
-- in-process dict, which is lost on every restart (e.g. uvicorn --reload
-- picking up a source change mid-job) and isn't shared across workers.
-- Storing it in wsi_registry survives both.
-- ============================================================

ALTER TABLE wsi_registry ADD COLUMN IF NOT EXISTS processing_status VARCHAR(20);
ALTER TABLE wsi_registry ADD COLUMN IF NOT EXISTS processing_error TEXT;
ALTER TABLE wsi_registry ADD COLUMN IF NOT EXISTS processing_updated_at TIMESTAMPTZ;
