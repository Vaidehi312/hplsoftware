-- ============================================================
-- Is this database ready to be used as a Knowledge Bank?
--
--   psql -h <socket> -d hpl_kb_test -f backend/check_kb_ready.sql
--
-- Read-only. Answers the question "I already have the schema" leaves open,
-- which is not whether the tables exist but whether the four things a cohort
-- actually needs are present: the tables it writes, the columns those writes
-- name, and — the one that is easy to miss — DATA in the cluster reference
-- tables, which are read by the tile overlay and the chatbot and are never
-- created by migrate_all.sql.
--
-- A schema-only copy of production passes every "does the table exist" check
-- and still answers every clinical question with blanks, because
-- hpc_dictionary is empty.
-- ============================================================

\pset footer off

\echo ''
\echo '=== 1. Tables a cohort writes or reads ==='
WITH required(name, why) AS (VALUES
    ('wsi_registry',           'slide paths — the viewer 404s without it'),
    ('wsi_metadata',           'optional slide headers'),
    ('dataset_config',         'per-cohort tiling geometry'),
    ('tile_coordinates',       'tile positions — drives tiles_meta'),
    ('tile_registry',          'tile identity + hpc_id'),
    ('hpl_profile_summary',    'per-slide aggregate the chatbot reads'),
    ('hpl_profile_proportion', 'per-slide aggregate the chatbot reads'),
    ('slide_hpc_membership',   'read by the chatbot via a dynamic table scan'),
    ('hpc_dictionary',         'the 71 clusters — REFERENCE DATA'),
    ('hpc_malignant_details',  'cluster annotation — REFERENCE DATA'),
    ('hpc_non_malignant_details','cluster annotation — REFERENCE DATA'),
    ('hpc_survival_analysis',  'Cox coefficients — REFERENCE DATA'),
    ('tile_hpc_heatmap',       'probability overlay; nothing writes it anywhere')
)
SELECT r.name,
       CASE WHEN t.table_name IS NULL THEN 'MISSING' ELSE 'ok' END AS status,
       r.why
FROM required r
LEFT JOIN information_schema.tables t
       ON t.table_name = r.name AND t.table_schema = current_schema()
ORDER BY (t.table_name IS NOT NULL), r.name;

\echo ''
\echo '=== 2. dataset_id — every cohort guard is scoped by it ==='
WITH required(name) AS (VALUES
    ('tile_registry'), ('tile_coordinates'), ('wsi_registry'),
    ('wsi_metadata'), ('hpl_profile_summary'), ('hpl_profile_proportion')
)
SELECT r.name,
       CASE WHEN c.column_name IS NULL THEN 'MISSING dataset_id' ELSE 'ok' END AS status
FROM required r
LEFT JOIN information_schema.columns c
       ON c.table_name = r.name AND c.column_name = 'dataset_id'
      AND c.table_schema = current_schema()
ORDER BY (c.column_name IS NOT NULL), r.name;

\echo ''
\echo '=== 3. Reference data — schema alone is not enough ==='
\echo '    (0 rows here means the viewer and chatbot answer with blanks)'
SELECT 'hpc_dictionary'            AS table_name, count(*) AS rows, 71 AS expected FROM hpc_dictionary
UNION ALL SELECT 'hpc_malignant_details',     count(*), 27 FROM hpc_malignant_details
UNION ALL SELECT 'hpc_non_malignant_details', count(*), 44 FROM hpc_non_malignant_details
UNION ALL SELECT 'hpc_survival_analysis',     count(*), NULL FROM hpc_survival_analysis;

\echo ''
\echo '=== 4. What is already in here, by cohort ==='
SELECT dataset_id, count(*) AS tile_registry_rows
FROM tile_registry GROUP BY 1 ORDER BY 1;

\echo ''
\echo 'Anything reading MISSING above: run backend/migrate_all.sql against this database.'
\echo 'Reference tables at 0 rows: pg_dump -t them from production and pg_restore here.'
\echo ''
