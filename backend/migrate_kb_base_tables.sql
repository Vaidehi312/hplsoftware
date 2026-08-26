-- ============================================================
-- The base tables no other SQL in this repository creates.
--
--   psql -h <socket> -d hpl_kb -f backend/migrate_kb_base_tables.sql
--
-- Why this exists. migrate_all.sql says "running it against a fresh one builds
-- the whole schema". Before this file, that was false. Counting CREATE TABLE
-- across the entire repository:
--
--   schema.sql   hpc_dictionary, hpc_malignant_details, hpc_non_malignant_details,
--                hpc_survival_analysis, hpl_profile_proportion, hpl_profile_summary,
--                tile_registry                                             (7)
--   migrations   slurm_dataset_runs, slurm_dataset_run_jobs                (2)
--
-- Nine of the live database's seventeen tables. The other eight —
-- dataset_config, h_latent_vectors, slide_hpc_membership, tile_coordinates,
-- tile_hpc_heatmap, tile_hpc_heatmap_old, wsi_metadata, wsi_registry — existed
-- only in the live database, created by hand or by a notebook's pandas
-- to_sql(). So migrate_all.sql on an empty database stopped at
-- migrate_indexes.sql's `UPDATE tile_coordinates ...` (ON_ERROR_STOP is on),
-- and the failure named an index migration rather than the missing table.
--
-- And schema.sql is worse than absent, because it looks authoritative. It is a
-- pg_dump taken 2025-10-23 and never regenerated; it disagrees with the live
-- database on the one table this pipeline writes most:
--
--   schema.sql tile_registry     live tile_registry
--   ------------------------     ------------------
--   id integer NOT NULL (PK)     (no id column)
--   hpc_id character varying     hpc_id integer  -> FK hpc_dictionary(hpc_id)
--   samples varchar(100)         samples varchar
--   -                            slide_tile varchar NOT NULL (PK)
--   -                            dataset_id text NOT NULL
--   -                            hpc_vote_margin, hpc_neighbor_distance,
--                                hpc_assigned_at, hpc_reference
--
-- A database built from schema.sql would take every INSERT in
-- register_dataset.py and every UPDATE in load_hpc_assignments.py and fail on
-- a column that is not there — or, for hpc_id, quietly accept integers into a
-- varchar. schema.sql now carries a header saying so; this file is what should
-- be trusted instead.
--
-- Definitions here were transcribed from `\d` against the live database
-- (hpl_kb, PostgreSQL 18.0) on 2026-08-26, which is the first complete record
-- of these tables that has ever existed outside the server.
--
-- Idempotent, like every other migration: CREATE TABLE IF NOT EXISTS,
-- ADD COLUMN IF NOT EXISTS, CREATE INDEX IF NOT EXISTS throughout. Running it
-- against the live database is a no-op.
--
-- ORDER: this must run FIRST in migrate_all.sql. migrate_indexes.sql indexes
-- and rewrites tile_coordinates and wsi_registry; migrate_processing_status.sql
-- adds columns to wsi_registry. Both assume the tables are already there.
--
-- STILL NOT COVERED, and deliberately not guessed at: the four cluster
-- reference tables (hpc_dictionary, hpc_malignant_details,
-- hpc_non_malignant_details, hpc_survival_analysis) and the slide_metadata
-- view. schema.sql defines the four, but its tile_registry is demonstrably a
-- decade of drift out of date, so its hpc_dictionary cannot be trusted either
-- — and it must have drifted, because the live tile_registry.hpc_id is an
-- integer with an FK to hpc_dictionary(hpc_id), which schema.sql declares
-- varchar(100). Getting these right needs `\d hpc_dictionary`,
-- `\d hpc_malignant_details`, `\d hpc_non_malignant_details`,
-- `\d hpc_survival_analysis` and `\d+ slide_metadata` from the live database.
-- Until then a fresh checkout still cannot build a complete schema, and that
-- is stated here rather than papered over.
-- ============================================================

\set ON_ERROR_STOP on

-- ------------------------------------------------------------
-- 0. pgvector, for h_latent_vectors.embedding.
--
-- CREATE EXTENSION needs privileges an ordinary role may not have, and
-- h_latent_vectors is not on any pipeline path (nothing in backend/ or in the
-- current UI reads or writes it — see KB_TABLE_COVERAGE). So a database
-- without pgvector is not a broken database, and this must not stop the run.
-- ------------------------------------------------------------
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgvector unavailable (%); skipping h_latent_vectors.', SQLERRM;
END$$;


-- ------------------------------------------------------------
-- 1. dataset_config — per-cohort tiling geometry.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dataset_config (
    dataset_id      text             NOT NULL,
    target_mpp      double precision NOT NULL,
    tile_size_5x_px integer          NOT NULL,
    CONSTRAINT dataset_config_pkey PRIMARY KEY (dataset_id)
);


-- ------------------------------------------------------------
-- 2. tile_coordinates — where each tile sits on its slide.
--
-- "row" is quoted because ROW is a keyword. PostgreSQL classes it as
-- col_name_keyword, so `row integer` does parse and register_dataset.py's
-- unquoted `INSERT INTO tile_coordinates (..., col, row, ...)` is correct as
-- written; quoting here is belt-and-braces and produces the identical
-- lower-case identifier.
--
-- Indexes idx_tc_slides / idx_tc_slide_tile / idx_tc_x_y_native are left to
-- migrate_indexes.sql, which also normalises the casing they depend on.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tile_coordinates (
    slides     text,
    tiles      text,
    slide_tile text NOT NULL,
    col        integer,
    "row"      integer,
    x_5x       integer,
    y_5x       integer,
    x_native   integer,
    y_native   integer,
    h5_index   integer,
    dataset_id text,
    CONSTRAINT tile_coordinates_pkey PRIMARY KEY (slide_tile)
);


-- ------------------------------------------------------------
-- 3. wsi_registry — one row per slide file on disk.
--
-- processing_status / processing_error / processing_updated_at are added by
-- migrate_processing_status.sql and are not repeated here, so that file stays
-- the single place they are described.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS wsi_registry (
    slide_id        text NOT NULL,
    sample_id       text,
    file_uuid       text,
    filename        text,
    hpc_path        text NOT NULL,
    file_size_bytes bigint,
    mtime_utc       timestamp without time zone,
    added_at        timestamp without time zone,
    dataset_id      text NOT NULL,
    CONSTRAINT wsi_registry_pkey PRIMARY KEY (slide_id)
);


-- ------------------------------------------------------------
-- 4. wsi_metadata — what OpenSlide reports about each slide.
--
-- mpp_x / mpp_y / objective_power are the numbers every coordinate conversion
-- in the tile server depends on. Nothing writes this table today; see
-- KB_TABLE_COVERAGE.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS wsi_metadata (
    slide_id               text NOT NULL,
    sample_id              text,
    level_count            integer,
    level_dimensions_json  jsonb,
    level_downsamples_json jsonb,
    mpp_x                  real,
    mpp_y                  real,
    objective_power        real,
    vendor                 text,
    scanner_model          text,
    scanner_date           timestamp without time zone,
    tile_width             integer,
    tile_height            integer,
    quickhash              text,
    dataset_id             text NOT NULL,
    CONSTRAINT wsi_metadata_pkey PRIMARY KEY (slide_id)
);


-- ------------------------------------------------------------
-- 5. tile_hpc_heatmap — per-tile cluster probabilities for the viewer.
--
-- 71 columns p_hpc_0 .. p_hpc_70, one per HPC, plus the argmax and its
-- probability. Built with a loop rather than 71 written-out lines so that the
-- count is stated once and cannot drift between the two tables.
--
-- The constraint names look transposed and are NOT a typo here — they are
-- copied from the live database, where tile_hpc_heatmap's primary key is
-- named hpc_heatmap_pkey and tile_hpc_heatmap_old's is named
-- tile_hpc_heatmap_pkey. That is what a rename of the live table under a
-- running system leaves behind. Reproduced exactly so a rebuilt database
-- matches the one in use.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tile_hpc_heatmap (
    slides        text,
    slide_tile    text NOT NULL,
    x_native      integer,
    y_native      integer,
    slide_id      text,
    hpc_top1      text,
    hpc_top1_prob real,
    CONSTRAINT hpc_heatmap_pkey PRIMARY KEY (slide_tile)
);

CREATE TABLE IF NOT EXISTS tile_hpc_heatmap_old (
    slides        text,
    slide_tile    text NOT NULL,
    x_native      integer,
    y_native      integer,
    slide_id      text,
    hpc_top1      text,
    hpc_top1_prob real,
    CONSTRAINT tile_hpc_heatmap_pkey PRIMARY KEY (slide_tile)
);

DO $$
DECLARE
    i integer;
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['tile_hpc_heatmap', 'tile_hpc_heatmap_old'] LOOP
        FOR i IN 0..70 LOOP
            EXECUTE format(
                'ALTER TABLE %I ADD COLUMN IF NOT EXISTS %I real', t, 'p_hpc_' || i
            );
        END LOOP;
    END LOOP;
END$$;


-- ------------------------------------------------------------
-- 6. slide_hpc_membership — which HPCs appear on which slide.
--
-- Note it has no dataset_id, so unlike every other per-slide table here it
-- cannot be scoped to a cohort: two datasets sharing a slide_id share a row.
-- Recorded as-is; changing it is a decision, not a transcription.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS slide_hpc_membership (
    slide_id text    NOT NULL,
    hpc_id   integer NOT NULL,
    CONSTRAINT slide_hpc_membership_pkey PRIMARY KEY (slide_id, hpc_id)
);

CREATE INDEX IF NOT EXISTS idx_shm_slide ON slide_hpc_membership(slide_id);
CREATE INDEX IF NOT EXISTS idx_shm_hpc   ON slide_hpc_membership(hpc_id);


-- ------------------------------------------------------------
-- 7. h_latent_vectors — per-tile encoder embeddings.
--
-- Created only if pgvector is present (section 0). The live column is
-- vector(1024) although the HPL encoder emits 128-D latents; the width is
-- transcribed from the live table, not inferred from the encoder.
-- ------------------------------------------------------------
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        CREATE TABLE IF NOT EXISTS h_latent_vectors (
            id        serial PRIMARY KEY,
            sample_id varchar(80),
            slide_id  varchar(120),
            tile_name varchar(120),
            hpc_id    integer,
            embedding vector(1024)
        );
    ELSE
        RAISE NOTICE 'pgvector absent; h_latent_vectors not created.';
    END IF;
END$$;


-- ------------------------------------------------------------
-- 8. dataset_id, the column that scopes every cohort — and that no migration
--    in this repository has ever created.
--
-- It is NOT NULL on tile_registry, wsi_registry, wsi_metadata,
-- hpl_profile_proportion and hpl_profile_summary in the live database, and
-- present on tile_coordinates. Every guard in register_dataset.py and
-- load_hpc_assignments.py is scoped by it. It appears in no CREATE TABLE and
-- no ALTER TABLE in git: it was added to the live database by hand, and a
-- database built from schema.sql has none of it.
--
-- Added here as nullable, then tightened to NOT NULL only where the table is
-- already free of NULLs. Adding NOT NULL to a populated table with existing
-- NULLs would fail the whole migration, and back-filling a cohort id is a
-- judgement about which cohort those rows belong to — not something a
-- migration can decide.
-- ------------------------------------------------------------
DO $$
DECLARE
    t         text;
    null_rows bigint;
BEGIN
    FOREACH t IN ARRAY ARRAY['tile_registry', 'tile_coordinates', 'wsi_registry',
                             'wsi_metadata', 'hpl_profile_proportion',
                             'hpl_profile_summary'] LOOP
        IF NOT EXISTS (SELECT 1 FROM information_schema.tables
                       WHERE table_name = t) THEN
            CONTINUE;
        END IF;

        EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS dataset_id text', t);

        -- tile_coordinates is nullable in the live database; leave it that way.
        IF t = 'tile_coordinates' THEN
            CONTINUE;
        END IF;

        EXECUTE format('SELECT COUNT(*) FROM %I WHERE dataset_id IS NULL', t)
            INTO null_rows;
        IF null_rows = 0 THEN
            EXECUTE format('ALTER TABLE %I ALTER COLUMN dataset_id SET NOT NULL', t);
        ELSE
            RAISE NOTICE '% has % row(s) with no dataset_id; leaving the column '
                         'nullable. Assign them a cohort, then re-run.', t, null_rows;
        END IF;
    END LOOP;
END$$;

CREATE INDEX IF NOT EXISTS idx_tile_registry_dataset
    ON tile_registry(dataset_id);
CREATE INDEX IF NOT EXISTS idx_wsi_registry_dataset
    ON wsi_registry(dataset_id);
CREATE INDEX IF NOT EXISTS idx_wsi_metadata_dataset
    ON wsi_metadata(dataset_id);
CREATE INDEX IF NOT EXISTS idx_hpl_profile_proportion_dataset
    ON hpl_profile_proportion(dataset_id);
CREATE INDEX IF NOT EXISTS idx_hpl_profile_summary_dataset
    ON hpl_profile_summary(dataset_id);

\echo '== base tables done =='
