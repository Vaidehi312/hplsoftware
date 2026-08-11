-- ============================================================
-- Confidence columns for tile -> HPC cluster assignments.
-- Run once on the HPCC PostgreSQL instance.
--
-- Why: hpc_id records which cluster a tile was assigned to, but not how much
-- to trust it. sc.tl.ingest assigns every tile to its nearest cluster
-- unconditionally, however far that tile sits from anything in the reference,
-- and reports no measure of that distance. A tile of novel morphology, an
-- artefact that survived filtering, and a textbook example of a cluster are
-- all stored identically today.
--
-- assign_hpc_clusters.py produces two numbers per tile, kept as separate
-- columns because they fail independently and cannot be combined without
-- losing the distinction:
--
--   hpc_vote_margin        (top votes - runner-up votes) / k. Between-cluster
--                          ambiguity — low means the tile is on a boundary
--                          between two clusters.
--   hpc_neighbor_distance  Mean distance to the k nearest reference tiles.
--                          Novelty — high means nothing in the reference
--                          resembles this tile, whatever its margin says.
--
-- The second is also the cheapest available probe for cohort shift: if a new
-- cohort's tiles sit systematically further from the reference than TCGA's
-- did, that is a batch effect visible as a distance before it turns into a
-- cluster-proportion difference that looks biological.
--
-- All nullable. The rows already carrying hpc_id predate this and have no
-- scores, and a NULL has to stay distinguishable from a genuinely confident
-- assignment — defaulting to 0 or 1 would silently label the entire existing
-- registry as either maximally suspect or fully trusted.
--
-- Note hpc_id itself stays character varying(100). The serving join in
-- tile_server_v2_.py (tile_registry -> hpc_dictionary) and the KB CSVs depend
-- on that type; "fixing" it to integer here would break both.
-- ============================================================

ALTER TABLE tile_registry ADD COLUMN IF NOT EXISTS hpc_vote_margin REAL;
ALTER TABLE tile_registry ADD COLUMN IF NOT EXISTS hpc_neighbor_distance REAL;
ALTER TABLE tile_registry ADD COLUMN IF NOT EXISTS hpc_assigned_at TIMESTAMPTZ;

-- Which reference produced the assignment. Cluster IDs are only meaningful
-- relative to one reference config plus one encoder checkpoint, so a registry
-- holding assignments from more than one is otherwise indistinguishable from
-- one holding a single self-consistent set.
ALTER TABLE tile_registry ADD COLUMN IF NOT EXISTS hpc_reference TEXT;

-- Supports "show me the tiles this pipeline is least sure about" without a
-- full scan of a 14M-row registry. Partial, because fully-confident tiles are
-- the overwhelming majority and are never the ones being looked for.
CREATE INDEX IF NOT EXISTS tile_registry_low_confidence_idx
    ON tile_registry (hpc_vote_margin)
    WHERE hpc_vote_margin IS NOT NULL AND hpc_vote_margin < 0.1;
