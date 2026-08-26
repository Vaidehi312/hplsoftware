-- ============================================================
-- PostgreSQL Index Migration for HPC Tile Server
-- Run once on the HPCC PostgreSQL instance to speed up queries.
-- ============================================================

-- 1. tile_coordinates — the most queried table
--    Normalise slides/slide_tile to uppercase once, then index plain columns.
UPDATE tile_coordinates SET slides = UPPER(TRIM(slides))         WHERE slides IS NOT NULL;
UPDATE tile_coordinates SET slide_tile = UPPER(TRIM(slide_tile)) WHERE slide_tile IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tc_slides      ON tile_coordinates(slides);
CREATE INDEX IF NOT EXISTS idx_tc_slide_tile  ON tile_coordinates(slide_tile);
CREATE INDEX IF NOT EXISTS idx_tc_x_y_native  ON tile_coordinates(x_native, y_native);

-- 2. tile_registry — joined on slide_tile and queried by slides/hpc_id
--    Add slide_tile column if it doesn't exist yet.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'tile_registry' AND column_name = 'slide_tile'
    ) THEN
        ALTER TABLE tile_registry ADD COLUMN slide_tile VARCHAR(250);
        UPDATE tile_registry SET slide_tile = UPPER(TRIM(slides || '_' || tiles));
    END IF;
END$$;

UPDATE tile_registry SET slides     = UPPER(TRIM(slides))     WHERE slides IS NOT NULL;
UPDATE tile_registry SET slide_tile = UPPER(TRIM(slide_tile)) WHERE slide_tile IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tr_slide_tile ON tile_registry(slide_tile);
CREATE INDEX IF NOT EXISTS idx_tr_slides     ON tile_registry(slides);
CREATE INDEX IF NOT EXISTS idx_tr_hpc_id     ON tile_registry(hpc_id);

-- 3. wsi_registry — looked up by slide_id on every request
UPDATE wsi_registry SET slide_id = UPPER(TRIM(slide_id)) WHERE slide_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_wsi_slide_id ON wsi_registry(slide_id);

-- 4. hpl_profile_proportion — queried per slide
UPDATE hpl_profile_proportion SET slides = UPPER(TRIM(slides)) WHERE slides IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_hpp_slides ON hpl_profile_proportion(slides);
CREATE INDEX IF NOT EXISTS idx_hpp_hpc_id ON hpl_profile_proportion(hpc_id);

-- 5. hpl_profile_summary — queried per slide and sample
UPDATE hpl_profile_summary SET slides  = UPPER(TRIM(slides))  WHERE slides IS NOT NULL;
UPDATE hpl_profile_summary SET samples = UPPER(TRIM(samples)) WHERE samples IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_hps_slides  ON hpl_profile_summary(slides);
CREATE INDEX IF NOT EXISTS idx_hps_samples ON hpl_profile_summary(samples);

-- 6/7. hpc_dictionary and hpc_survival_analysis.
-- Guarded by table rather than stated flat, because these are two of the four
-- reference tables with no CREATE TABLE anywhere in git (see
-- migrate_kb_base_tables.sql's header). CREATE INDEX IF NOT EXISTS still raises
-- when the TABLE is missing, so on an empty database these two lines were the
-- last thing stopping migrate_all.sql from running to completion — found by
-- executing it, which nothing had done before 2026-08-26.
DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['hpc_dictionary', 'hpc_survival_analysis'] LOOP
        IF EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = current_schema() AND table_name = t) THEN
            EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I (hpc_id)',
                           'idx_' || CASE t WHEN 'hpc_dictionary' THEN 'hd' ELSE 'hsa' END
                           || '_hpc_id', t);
        ELSE
            RAISE NOTICE '% absent; its index was not created.', t;
        END IF;
    END LOOP;
END$$;

-- Done. Run `ANALYZE;` after to refresh planner statistics.
ANALYZE;
