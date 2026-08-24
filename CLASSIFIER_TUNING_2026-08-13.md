# Classifier tuning + Knowledge Bank automation — session log

**Date:** 2026-08-13
**Scope:** `backend/assign_hpc_clusters.py`, `backend/validate_reference.py`, `backend/load_hpc_assignments.py`, `backend/tile_server_v2_.py`, `app/api_client.py`, `app/app_v28.py`
**Trigger:** improving k-NN classification accuracy (`Backend: faiss-flat` leave-one-out was 92.41%), which led to a faiss-only cleanup, a Stage 5 (Knowledge Bank load) automation pass, and — while testing a load against a new dataset — the discovery that nothing in this pipeline has ever automatically inserted a row into `tile_coordinates`, `tile_registry`, or the embeddings table.

---

## TL;DR

Classification accuracy went from **92.41% → 96.13% (smaller k) → 96.65% (+ distance-weighting)** on the same 20,000-tile leave-one-out sample, all measured, not assumed. Stage 5 (loading assignments into the KB) is now UI-driven with a confidence threshold and a path for untracked test output. One real, load-bearing gap was found and documented but **not fixed**: a brand-new dataset has no automated way into the Knowledge Bank's identity tables — every row there today came from hand-run notebooks. A second round of tuning experiments (stronger distance weighting, cosine similarity, raw embeddings, a low-margin fallback classifier) is **in progress, not finished** — see [In progress](#in-progress--not-yet-built).

**2026-08-15 continuation** (see [§9–12](#9-raw-128-d-embeddings-vs-pca-127-d--measured-no-meaningful-difference)): against a *different*, exploratory reference (`..._leiden_5p0__fold0_subsample.h5ad`, 250,000 tiles, 109 clusters — **not** the production `leiden_2.5` reference the numbers above came from, so none of today's percentages are directly comparable to 92.41/96.13/96.65 above), raw-128-D vs PCA-127-D made no measurable difference, `distance_power=2` improved accuracy again (96.75% → 97.34%), cosine similarity measurably hurt (96.75% → 95.51%) for a reason now confirmed in the encoder's own clustering code, and **margin-gated adaptive k — re-querying only the tiles whose vote came back ambiguous, at a larger k — is the best result of the whole session** (97.34% → 97.95%, touching only 4% of tiles). None of this section is yet validated against the real production `leiden_2.5` reference.

---

## What was tried, in order

### 1. faiss-only backend — kept, committed
Removed the numpy fallback and the `faiss-ivf` approximate path from `Searcher`/`vote()` in `assign_hpc_clusters.py`. faiss is now a hard dependency (`backend/requirements.txt`, `submit_cluster_assignment.py`'s `_REQUIRED_MODULES`) instead of an optional accelerant — `auto`/`faiss`/`numpy` all returned identical assignments at different speeds, and `faiss-ivf` was already known to change labels for no speed gain, so the "choice" was three ways to get the same answer and one way to get a different one.

Committed: `297085a` "Make k-NN search faiss-only and give Stage 5 a UI"

### 2. k tuning — kept, k=10 is now the working default
Original `k=250` mirrors the reference's own stored `n_neighbors` (tied to how the Leiden graph itself was built, not an arbitrary knob). Swept `k = 10, 25, 50, 100, 250` via `validate_reference.py --k <n>`, same 20,000-tile sample each time (fixed seed):

| k | accuracy |
|---|---|
| 250 (original) | 92.41% ± 0.37 |
| 10 | **96.13% ± 0.27** |

Cross-checked against the k=250 per-cluster breakdown: correlation between a cluster's k=250 accuracy and how much it improved at k=10 was **-0.906** — the weakest clusters improved the most, consistent with genuine boundary noise rather than a k=250-specific artifact. Cluster-size vs. own-accuracy correlation was ~0 at both k values, ruling out "small clusters just have too few votes" as the mechanism.

### 3. Distance-weighted voting — kept, opt-in
`vote()` in `assign_hpc_clusters.py` gained a `distance_weighted` flag: each neighbour's vote becomes `1/(distance + eps)` instead of 1. Unweighted (default) is bit-for-bit unchanged — verified by the full test suite plus a synthetic case (4 distant same-cluster neighbours beating 1 close one unweighted, losing once weighted).

Measured at k=10: **96.65% ± 0.25**, vs 96.13% unweighted — CIs just touch (96.40% both edges), so real but modest on its own.

### 4. Confusion-matrix reporting — kept
`validate_reference.py`'s per-cluster table now shows, for each cluster's wrong tiles, the single most common wrong vote and its share of those errors — not just right/wrong. This is what actually distinguishes "these two clusters are confused with each other" (a real merge candidate) from "this cluster's errors are scattered" (a different problem).

Found at k=10 distance-weighted: three reciprocal pairs (19↔63, 64↔27, 20↔51) each other's #1 confusion target in both directions. Two of three independently confirmed by matching human annotation in `hpc_dict_main.csv` (19/63 both "Non-neoplastic pathological lung, unspecified"; 20/51 both "Normal/near-normal lung") — found from model behaviour first, corroborated by annotation second, the reverse (and more rigorous) order from an earlier, retracted attempt that pattern-matched on annotations alone and found no correlation with actual confusability.

Also found: 6 clusters (16, 11, 15, 22, 23, 57) each absorb the #1 wrong-vote share from 3+ smaller neighbouring clusters, all at or above median cluster size — see item 5.

### 5. Class-frequency weighting — tried → reverted → re-added, **not yet re-validated**
Built once (per-cluster `1/reference_count` vote weight, composable with distance-weighting), then **explicitly reverted** on direct instruction ("no just keep distance-weighted, this impacts the classifier ever more!"), fully removed and confirmed via `grep` + full test suite.

Later **re-requested** as part of a broader experiment list, on the strength of the confusion-matrix finding in item 4 (large clusters acting as "gravity wells" for smaller neighbours' errors) — a different, more specific piece of evidence than what motivated the original removal (that check only showed cluster size doesn't predict a cluster's *own* accuracy, which is a different question from whether it absorbs *others'* errors).

Current state: `class_weights` param and `--class-weighted` flag are back in both scripts, composable with `distance_power` (below). **Not yet re-run against the real reference** — the original revert happened before it was ever measured against the confusion-matrix evidence, so this is a genuine open experiment, not a settled reversal.

### 6. Manifest bug in `/package` and `/package-test` — fixed, kept
Both endpoints refused `scope="tiled"` ("package everything on disk, regardless of which run tiled it") whenever the *run's own original* tiling manifest was missing from disk — even though that scope builds a brand-new manifest from live disk coverage and never reads the original file at all. Fixed: the existence check now only applies when `scope != "tiled"`. Verified by simulating the guard's boolean logic against 5 cases (including the exact bug scenario) plus the full test suite.

### 7. Stage 5 (Knowledge Bank load) UI automation — kept
- `min_margin` threshold excludes low-confidence tiles from `hpl_profile_proportion`/`hpl_profile_summary` only — `tile_registry` keeps every tile's own `hpc_id`/margin regardless. `total_tiles` shrinks with the exclusion, deliberately, so it always matches the population the proportions were computed from. Default 0 (no behaviour change) unless set.
- `csv_path` override lets output from Stage 4's untracked "Test on a sample .h5" mode be loaded — this is a **real write** to the KB, not a dry run like the other test paths in this pipeline; it just doesn't mark a specific run's `kb_load_done`, since the CSV may not be that run's own output.
- Preview-then-commit flow throughout; never auto-commits. `/kb-load` calls
  `load_hpc_assignments.py`'s own functions in-process rather than reimplementing them, so
  every CLI guard (95% match rate, unknown cluster IDs) applies unchanged.

**⚠️ Pending deploy step:** `backend/migrate_dataset_runs_kb_load.sql` (committed in
`297085a`) still has to be run against the production Postgres:

```bash
psql "$DATABASE_URL" -f backend/migrate_dataset_runs_kb_load.sql
```

Idempotent (`ADD COLUMN IF NOT EXISTS`). Without it Stage 5 still *works*, but nothing ever
shows as loaded — the code reads `kb_load_*` with `.get()` and degrades silently, so every
run looks permanently "ready to load". Check `migrate_dataset_runs_assignment.sql` ran first.

**Evidence behind choosing a `min_margin`** — leave-one-out, k=10, distance-weighted:

| vote_margin | correct |
|---|---|
| 0.00–0.10 | 61.7% |
| 0.10–0.25 | 81.2% |
| 0.25–0.50 | 95.5% |
| 0.50–0.75 | 99.5% |
| 0.75–1.00 | 100.0% |

A tile under 0.10 margin is close to a coin flip; over 0.75 it was never wrong in the
sample. 0.25 is the natural cut if one is wanted.

### 8. Knowledge Bank onboarding gap — **documented, not fixed**
Attempting to load a "Radiogenomics" test-mode assignment into the KB refused at **0% match rate**. Root cause, confirmed by reading the actual code paths (not guessed): **nothing in this repository ever `INSERT`s into `tile_coordinates`, `tile_registry`, or the embeddings table (`h_latent_vectors`, confirmed against the real `\dt` output — 26 relations, matching `CLAUDE.md`'s count exactly).** Every row in those tables today came from hand-run notebooks (`Filling_out_kb.ipynb`, `KB_int.ipynb`) — export to CSV, manually deduplicate (100 duplicate `slide_tile` keys found by hand in a 360,667-row file), `psql` import. Stage 5's `load()` only ever `UPDATE`s `tile_registry.hpc_id`; it was never meant to create the row.

What *is* automated and fully sufficient today: `hpl_profile_proportion`/`hpl_profile_summary` (Stage 5 always inserts/refreshes these — no identity data needed, they're derived entirely from the assignment CSV).

Everything needed for automatic registration already exists on disk after Stages 1–2, traced field-by-field:
- `tile_coordinates` (`slide_tile, col, row, x_5x, y_5x, x_native, y_native`) — written per slide by `auto_tile_from_mask.py`'s own `_tile_metadata.csv`, already, today.
- `tile_registry` identity fields (`image_index`, `h5_source_path`) — fixed the moment `make_hpl_hdf5.py` writes the packaged `.h5` (row position in its own `samples`/`slides`/`tiles` arrays).

Plan (not built): 1) get the real column-level schema for `tile_coordinates`/`h_latent_vectors` (only table *names* have been confirmed via `\dt`, not `\d`); 2) build a registration step reading the above; 3) same dry-run-preview / refuse-loud guards Stage 5 already has.

Full 26-relation mapping (what's touched by a run, what's static cluster reference, what's a different feature, what's legacy, what's unreferenced anywhere) is in two published flowchart artifacts (see below) rather than repeated here.

---

## 2026-08-15 continuation

All three measurements below use one exploratory reference, built from
`LATTICeA_5x_he_complete_surv_sex_leiden_5p0__fold0_subsample.h5ad`: 250,000
tiles, 109 clusters, `leiden_5.0`, `n_neighbors=250`. This is **not** the
production `leiden_2.5` reference items 1–8 above were measured against
(different resolution, different fold, a named subsample, a different cluster
count) — the percentages in this section are internally comparable to each
other, not to 92.41/96.13/96.65 above.

Same 20,000-tile sample, `--seed 0`, `--k 10` throughout, so only the one
thing under test changes between rows:

| variant | accuracy | vs. baseline |
|---|---|---|
| unweighted, PCA-127 | 96.74% ± 0.25 | — |
| unweighted, raw-128 | 96.75% ± 0.25 | baseline for this section |
| distance-weighted, `distance_power=2`, raw-128 | **97.34% ± 0.22** | real improvement |
| cosine metric, unweighted, raw-128 | 95.51% ± 0.29 | real regression |

### 9. Raw 128-D embeddings vs PCA 127-D — measured, no meaningful difference
Added `--embedding-space {pca,raw}` to `build_hpc_reference.py`: `raw` reads
`X` (the encoder's own `z_latent`) directly instead of `obsm/X_pca`, with an
identity `components` matrix so `project()`/`Searcher`/`vote()` in
`assign_hpc_clusters.py` need zero changes.

96.74% (PCA-127) vs 96.75% (raw-128) — a 3-tile difference out of 20,000,
CIs almost fully overlapping, and the weakest-clusters table identical down
to the exact confusion pairs. Root cause checked directly rather than assumed:
[`leiden_representations.py:196`](HPL-LATTICeA/models/clustering/leiden_representations.py:196)
calls `sc.pp.neighbors(..., n_pcs=adata.X.shape[1] - 1, ...)` — PCA here drops
exactly **one** of 128 dimensions (the lowest-variance axis), so it was never
going to change k-NN neighbourhoods much. This is a property of this
reference's own construction, not a general "PCA doesn't matter" claim.

### 10. Stronger distance weighting (`distance_power`) — kept, `power=2` is the plateau
`power=2`: 96.75% → **97.34%** (19,351 → 19,468 of 20,000; CIs [96.50, 97.00]
vs [97.12, 97.56] — no overlap). Larger jump than `distance_power=1`'s effect
on the old PCA-127 reference (96.13% → 96.65%), consistent with "weight the
nearest neighbours even more" continuing to help. Not uniform, though:
**cluster 95 got worse** (91.2% → 87.7% recovery) even as the overall number
improved — the confusion-matrix reporting from item 4 is what caught this;
overall accuracy alone would have hidden it.

`power=3`, same sample/seed/k: **97.27% ± 0.23** (19,453/20,000) — CI
[97.04, 97.50] overlaps `power=2`'s [97.12, 97.56] almost entirely, so this is
noise, not a further gain (if anything a hair lower). Cluster 95 recovered
slightly better than at `power=2` (88.6% vs 87.7%) but still worse than
unweighted (91.2%) — the regression on that cluster isn't power-specific, it's
a cost of distance-weighting generally. **`power=2` is the working recommendation;
pushing higher buys nothing further on this reference.**

### 11. Cosine similarity — measured, real regression, root cause found
96.75% → **95.51%** (19,351 → 19,102; CIs [96.50, 97.00] vs
[95.22, 95.80] — no overlap). Added `metric={l2,cosine}` to `Searcher` in
`assign_hpc_clusters.py`: cosine mode L2-normalises reference and query
vectors and searches with `faiss.IndexFlatIP`, converting the similarity back
to a squared-Euclidean-equivalent distance (`2 - 2·cos θ`) so `vote()`,
margin, and `neighbor_distance` need no changes downstream.

Checked why, rather than leaving it as "cosine happened to score lower":
the same `leiden_representations.py:196` line builds the reference's own
neighbour graph with `metric='euclidean'`. The ground-truth labels were
produced from an explicitly Euclidean graph, so evaluating them under cosine
— which discards vector magnitude — structurally disagrees with that graph
more often. A property of this reference's construction, not a defect in the
cosine implementation.

**Settled:** `distance_power=2` — `power=3` measured no further gain (§10),
so the curve has plateaued. Cosine stays deprioritised given the root cause
above.

### 12. Margin-gated adaptive k — best result of the session
> **Deleted 2026-08-24.** `adaptive_k_experiment.py` and its test were removed:
> the question below was answered, `tune_classifier.py` now sweeps adaptive k as
> one knob among many, and `assign_hpc_clusters.py` ships it as
> `--adaptive-margin` / `--adaptive-k`. Nothing imported it but its own test. The
> section is kept as the record of how the answer was reached.

New script `backend/adaptive_k_experiment.py` (uncommitted) tests a specific,
motivated idea rather than "bigger k everywhere": item 2's own k-sweep already
proved a *global* larger k hurts (250→92.41%, 10→96.13%), so the only version
worth trying is conditional — leave the confident majority alone and only
re-query the tiles whose `k=10` vote came back ambiguous, at a larger k.

Needed one small addition to make the comparison fair: `leave_one_out()` in
`validate_reference.py` gained an optional `query_index` override, so a rerun
at a different k hits the *exact same* reference rows as the first pass
rather than a fresh random sample — otherwise a delta could just be sampling
noise. `compare()` runs pass 1 at `--k-base`, masks tiles below
`--margin-threshold`, reruns only those at `--k-expand`, and reports both the
subset-only delta and the blended whole-sample accuracy.

Measured on the same exploratory reference, `k_base=10`, `k_expand=25`,
`distance_power=2` throughout, 20,000-tile sample, `--seed 0`:

| margin_threshold | tiles re-queried | subset: before → after | overall accuracy | overall delta |
|---|---|---|---|---|
| 0.50 | 4,125 | 87.25% → 89.77% | 97.86% | +0.52 |
| 0.25 | 1,980 | 76.67% → 82.47% | 97.91% | +0.57 |
| 0.10 |   805 | 62.48% → 77.52% | **97.95%** | **+0.60** |

**Narrower is both better and cheaper — not a tradeoff.** Isolating the band
between margin 0.1 and 0.25 (1,175 tiles not touched by the 0.10 run but
touched by the 0.25 run): already ~86.5% correct at `k=10`, came back ~85.9%
at `k=25` — a small step backward. Same effect as item 2's global k-sweep,
just localised to a narrower band once the genuinely ambiguous tiles are
pulled out. The zone where a bigger k actually helps is concentrated below
margin ≈0.1; above that, more neighbours add noise, not signal.

**Working recommendation: `margin_threshold≈0.1`.** Touches the fewest tiles
(805 of 20,000, ~4%) for the best measured overall gain. `margin_threshold=0.05`
was proposed as the next narrowing check — **not yet run**. This is also a
concrete instance of the "hybrid rule" item below: the "second classifier" for
low-margin tiles turned out to be the same k-NN engine at a larger k, not a
separate model.

**Not yet done, and matters most:** none of §9–12 has been validated against
the real production `leiden_2.5` reference — everything above used the
exploratory `leiden_5.0`/109-cluster one.

---

### 13. Error diagnostics — why the weak clusters are weak (built, not yet run on a real reference)

Up to here every experiment changed a knob and re-measured the headline number.
This one adds no knob: it makes `validate_reference.py`'s report say **which
kind of error** each weak cluster is suffering, so the next knob can be chosen
rather than guessed. Four things were added to `report()`:

**1. The error budget (the headline).** For every wrong tile, whether the true
cluster appeared *at all* among the k neighbours.

- **Present but outvoted** → recoverable by a vote rule (`distance_power`,
  `class_weighted`, adaptive k). These are the addressable errors.
- **Never present** → no vote rule can fix it. The tile's neighbourhood simply
  does not contain its own cluster; that needs a larger k, a different
  embedding space, or a merge.

The report now prints a **ceiling**: accuracy if every recoverable error were
won. That is the honest upper bound on what any further vote tuning can buy —
if the ceiling is 97.6% and we are at 97.34%, there is 0.26 points left in the
vote and the rest of the gap is structural.

**2. Wrong-tile vs correct-tile margin, per cluster.** Distinguishes two failure
modes that look identical in an accuracy column:
- *low-confidence / boundary* — wrong tiles have visibly lower margins. Adaptive
  k (§12) already targets exactly these.
- *confidently wrong* — wrong tiles vote with about the same margin as correct
  ones. **Adaptive k cannot help here by construction**, because it only
  re-queries low-margin tiles. This is the cluster-95-style case from §10.

**3. Symmetric vs one-way confusion.** `[mutual]` marks a cluster whose top
confusion also names it back. Symmetric confusion means the two clusters are
mutually indistinguishable — a Leiden resolution artefact (one phenotype split
in two), not a classifier defect. One-way confusion (A→B heavily, B→A rarely)
means A is being *absorbed* by B, which is the class-frequency-weighting case.

**4. Errors ranked by count, not rate.** The weakest-cluster list ranks by
recovery *rate*, which systematically over-weights small clusters: a 60%
cluster of 40 tiles is 16 errors, a 97% cluster of 3,000 is 90. A new table
ranks confusion **pairs** by total error count — that is the list to work down
if the goal is the overall number. Also reports the size-vs-recovery
correlation, so "small clusters just recover worse" can be confirmed or ruled
out instead of assumed.

Validated on a synthetic reference built so the answer is known: a small cluster
placed as a near-duplicate of a much larger one was correctly reported as fully
absorbed, confidently wrong (margin 0.90), one-way — i.e. exactly the
class-weighting case and *not* an adaptive-k case.

Four new tests in `test_validate_reference.py`, all written so the diagnostic
can come out bad: `truth_neighbours` must be ~10/10 on separable data and ~k/n
on random labels; unreachable errors must be counted on a random-label
reference; a correct tile can never be unreachable; the ceiling must sit
strictly between measured accuracy and 1.0; and the pair table must rank the
large overlapping pair above the small bad ones. Suite: **145 → 149 passing.**

**Not yet run against any real reference.** The command:

```bash
python validate_reference.py --reference ref_raw128.npz --sample 20000 --k 10 --distance-weighted --distance-power 2 --worst 15
```

---

## In progress / not yet built

Requested as an ordered list of next experiments; work started but was interrupted before testing:

- **`distance_power` (stronger distance weighting, e.g. `1/d²`)** — **done, see §10.** `distance_power=2` measured a real improvement (96.75% → 97.34%); `distance_power=3` measured no further gain (97.27%, within noise of `power=2`) — settled on `power=2`.
- **Class-frequency weighting re-added** — see item 5 above. **Code written, not yet re-validated.** Still the one open item from the original list.
- **Cosine similarity instead of L2** — **done, see §11.** Measured a real regression (96.75% → 95.51%); root cause confirmed in `leiden_representations.py:196` (`metric='euclidean'` on the reference's own neighbour graph). Deprioritised.
- **Raw 128-D embeddings instead of PCA 127-D** — **done, see §9.** No meaningful difference (96.74% vs 96.75%), because this reference's own `n_pcs=adata.X.shape[1] - 1` only ever drops one dimension.
- **Hybrid rule (confident tiles via weighted k-NN, low-margin tiles to a second classifier)** — **partly answered, see §12.** Margin-gated adaptive k (rerun ambiguous tiles at a larger k) is a concrete, measured instance of this: +0.60 points overall, touching only ~4% of tiles. Still open: whether an actual different classifier (not just a bigger k) would do better on the tiles that remain wrong even at k=25.

---

## Files touched

```
backend/assign_hpc_clusters.py     — Searcher (faiss-only, metric={l2,cosine}),
                                      vote() (distance_weighted, distance_power,
                                      class_weights), CLI flags
backend/validate_reference.py      — leave_one_out() same params + metric + query_index
                                      override (§12), truth_neighbours (§13),
                                      confusion-matrix + error-budget reporting in
                                      report(), CLI flags
backend/build_hpc_reference.py     — --embedding-space {pca,raw}, identity components
                                      for the raw path
backend/tests/test_searcher_metric.py       — new: l2/cosine Searcher unit tests
backend/tests/test_build_hpc_reference.py   — new: pca/raw build() unit tests
backend/adaptive_k_experiment.py            — new: margin-gated adaptive-k comparison (§12)
backend/tests/test_adaptive_k_experiment.py — new: query_index + compare() unit tests
backend/submit_cluster_assignment.py — faiss in _REQUIRED_MODULES, backend param removed
backend/tile_server_v2_.py          — Stage 5 endpoints (min_margin, csv_path), package/
                                      package-test scope="tiled" manifest fix
backend/load_hpc_assignments.py    — min_margin param on inspect()/compute_profiles()
backend/requirements.txt            — faiss-cpu added
backend/migrate_dataset_runs_kb_load.sql — new: kb_load_done/at/rows/reference columns
app/api_client.py                   — min_margin, csv_path on preview/commit_kb_load
app/app_v28.py                      — Stage 5 UI (min_margin, source picker), backend
                                      selectbox removed from Stage 4 form
CLAUDE.md                           — faiss-only note, Stage 5 automation note
```

Two flowchart artifacts published (Mermaid, click-to-edit labels, no persistence):
- **How a tile becomes a Knowledge Bank row** — pipeline + classifier steps + all 26 KB relations + automation plan per group.
- **The classifier's architecture** — reference asset, projection, faiss k-NN, the two vote modes, three measured-constraint callouts.

---

## Outstanding

| item | notes |
|---|---|
| Real column-level schema for `tile_coordinates`, `h_latent_vectors` | Only table *names* confirmed (`\dt`), not column definitions (`\d`) — blocks writing any real `INSERT`. |
| Re-added `class_weighted` | Code written, not yet re-run against a real reference. The one open experiment left from the original list. |
| Validate §9–12 against the production `leiden_2.5` reference | Everything in the 2026-08-15 continuation used the exploratory `leiden_5.0`/109-cluster reference. Not yet checked against what `assign_hpc_clusters.py` actually assigns against in production. |
| `margin_threshold=0.05` for adaptive k | Proposed narrowing check after 0.10 came out both cheaper and better than 0.25/0.50 (§12) — not yet run. |
| Run the §13 error diagnostics on a real reference | Built and unit-tested, never run on real data. Its output decides whether `class_weighted` is worth re-validating (one-way absorption) or a dead end (symmetric splits), and how much headroom is left in the vote at all. |
| Everything in this log is uncommitted | See the exact state below. |

## Working-tree state (as of 2026-08-15)

Branch `perf/encode-throughput`. Last commit **`297085a`** "Make k-NN search faiss-only and
give Stage 5 a UI" — which also committed `backend/requirements.txt` and
`backend/migrate_dataset_runs_kb_load.sql` (both now tracked). **Nothing since then is committed.**

Modified, uncommitted — all part of this work:
```
app/api_client.py                 app/app_v28.py
backend/assign_hpc_clusters.py    backend/build_hpc_reference.py
backend/load_hpc_assignments.py   backend/tile_server_v2_.py
backend/validate_reference.py
```

Untracked, uncommitted — all part of this work:
```
CLASSIFIER_TUNING_2026-08-13.md   (this file)
backend/tests/test_searcher_metric.py
backend/tests/test_build_hpc_reference.py
backend/adaptive_k_experiment.py
backend/tests/test_adaptive_k_experiment.py
```

⚠️ **`backend/make_hpl_hdf5.py` and `backend/submit_mask_tile_slurm.py` are also modified
but were already modified before this work began — they are NOT part of it.** Don't sweep
them into a commit for this work without checking what they contain first.

Tests: **145 passing** (`python3 -m pytest backend/tests -q`) — up from 126, the three new
test files adding 19. Still no `.gitignore`; `hpl_kb_dump.pgsql` (3.8 MB DB dump) and
several large CSVs sit untracked in the repo root, one `git add -A` from permanent history.

---

# 2026-08-18 — everything above was measured on the wrong reference

## §14 The two references

Sections 9–13 were all measured against `ref_raw128.npz`: **LATTICeA**, leiden
**5.0**, fold0, 250k-tile subsample, **109 clusters**. Production assigns against
`hpc_reference_leiden_2p5_fold2.npz`: **TCGA LUAD**, leiden **2.5**, fold2, 2.5M
tiles, 127 comps, **71 clusters**. Verified locally: the repo-root CSV
`TCGA_LUAD_5x_he_train_filtered_leiden_2p5__fold2.csv` has exactly 71 distinct
clusters, 0–70, over 360,667 rows, matching `hpc_dictionary`.

Worse than "untested on production": leiden_5.0 is the **QC pass** whose only job
is isolating junk for deletion (`HPL-LATTICeA/README.md`, §Background and artefact
removal). 27 of its 108 reviewed clusters are flagged `remove=1` — background ×13,
edge ×7, ink, out-of-focus, air bubble, artefact — and `remove_indexes_h5.py`
physically deletes those tiles before production's leiden_2.5 runs on the
`_filtered` data. So it is deliberately over-split, with duplicate descriptions
(`52`/`61` both "solid retraction artefact", `46`/`79` both "edge", `31`/`38` both
"infiltrative tumour with collageous stroma"). Errors between such pairs are not
errors.

Now guarded in code: all five experiment/validation tools default `--reference` to
`HPC_REFERENCE_PATH` and print a provenance banner, shouting to stderr on anything
that is not 71-cluster leiden_2.5. `--reference` had been `required=True`, which is
why the correct default already in `build_hpc_reference.py` was never used.

## §15 Production numbers, and what changed

Leave-one-out, production reference. `tune_classifier.py` searches once at k_max
and re-votes, so k / distance_power / class-weighting / adaptive-k all sweep off
one search; `--neighbours-cache` makes re-sweeps instant. Reports **McNemar** on
discordant pairs, not marginal CIs — every config scores the same tiles, and a
marginal CI at n=20,000 is ±0.24pp, which would call a real +0.4pp gain noise.

Baseline (`k=10 --distance-power 2`): **96.78%** at n=200,000 (96.79% at 20,000).

| config | Δ | net | σ |
|---|---|---|---|
| `pow 3` alone | +0.02 | +5 | 0.5 |
| `pow 2` + adapt 0.1 | +0.34 | +676 | 12.0 |
| **`pow 3` + adapt k=25 below 0.1** | **+0.45** | **+895** | **16.5** |
| `k=15 pow 3`, no adapt | +0.28 | +558 | 9.0 |

**Settled: `--k 10 --distance-weighted --distance-power 3` + adaptive k=25 for
`vote_margin < 0.1`. 96.78% → 97.23%.**

Two corrections to §10 and §12. **`distance_power=3` beats 2 on production**,
where the exploratory reference had it as a plateau and this doc recommended 2.
And **threshold 0.1 still beats 0.25** (+895 vs +820) despite production's
0.10–0.25 band being worse (79.8% vs 86.4% correct) — precision, not coverage.

Sample size changed the conclusion, not just the error bar: at n=20,000 the top
two configs were 18 tiles apart and the simpler one looked equivalent; at 200,000
they separate by ~337 tiles.

## §16 Why it is not 100%, and where the remaining error is

Every structural finding from §13 **transferred**: 100% of errors had the true
cluster among the k neighbours (642/642), 93% had it as runner-up, size explains
nothing (r = −0.16), the biggest confusion pair is 7 of 642, and there are **zero
errors above margin 0.75** — which is 68% of tiles.

So retrieval is perfect and the vote is near its limit. 100% is also the wrong
target: a Leiden cluster is a cut through a continuous density, and a tile on that
cut has no unambiguous label.

`cluster_diagnostics.py` classified **all 71 production clusters as `boundary`** —
no duplicates, no engulfed clusters. Per-cluster work is the wrong frame here: 642
errors over 71 clusters is ~9 each, and the error sits at boundaries everywhere.

**The reframe worth more than another 0.4pp:** `supercluster_dictionary.py`
collapses HPCs into four immune/architecture classes and `hpc_annotations.py` takes
its majority *at that level*, so an HPC error inside one supercluster changes no
downstream number. The weakest cluster's main confusion (69 → 67) is exactly that,
as is 34 → 47. `validate_reference.py --superclusters` measures it; the mapping
covers 25 of 71 HPCs and prints its coverage.

## §17 Outstanding

| item | state |
|---|---|
| ~~`assign_hpc_clusters.py` has no adaptive-k flag~~ | **done, `0855f20`** — and see §18: the submitter forwarded no vote knob at all |
| Acceptance test at `pow 3` | not run. `--validate-against` is the gate; <99% is a defect, not drift. **Read §19 before reading its number** |
| `class_weighted` | flag is in `tune_classifier.py`; the cluster's copy of that file predates it, which is why the sweep would not parse |
| `--local-scaling` | untested. 19.5 h for all 2.5M rows; budgeted and skipped by default |
| Confirm winner on `--seed 1` | not run; best-of-many on one sample is upward-biased. Use a **separate** `--neighbours-cache` file: the cache is keyed on the query set and a different seed overwrites it |
| Stage 4's UI/endpoint path | `tile_server_v2_.py:3908,3958` call the submitter without vote arguments, so the UI still queues the default vote. CLI only, for now |

## §18 The vote settings never reached a real assignment

Two separate gaps, and the second is the larger one.

**Adaptive k existed only in the experiment scripts** — `adaptive_k_experiment.py`
and `tune_classifier.py`. `assign()` had `--k`, `--distance-weighted`,
`--distance-power`, `--class-weighted` and `--local-scaling`, but no gate. So the
best result of the session was unreachable from the production path.

**And `submit_cluster_assignment.py` forwarded none of the vote knobs.** The
Slurm command carried `--reference`, `--h5`, `--out`, `--rep-key`, `--batch-size`,
`--progress`, and conditionally `--k`, `--validate-against`, `--query-mean`,
`--row-start/--row-stop`. That is all
(`_build_assignment_command`). `--distance-weighted` is a `store_true` defaulting
to `False`, so **every Stage 4 job submitted through Slurm has run the plain
unweighted vote**, whatever was measured offline — and nothing in the output says
so. Wiring adaptive k without noticing this would have produced a flag that still
could not be used.

Fixed in `0855f20`:

- `assign()` takes `--adaptive-margin` and `--adaptive-k`. It searches once at
  `max(k, adaptive_k)` and votes on a prefix, rather than searching twice: the
  flat-L2 scan over the reference is the whole cost and does not depend on `k` —
  only the top-`k` selection does — so widening is nearly free, and the base vote
  and the re-vote see the same neighbours by construction. ~9% of tiles re-vote.
- Re-voted rows take the wide vote's **margin and mean distance**, not just its
  label. A `vote_margin` describing a discarded vote would make Stage 5's
  `--min-margin` drop precisely the tiles adaptive k exists to rescue.
- The submitter forwards the configuration as **one unit** through `vote_flags()`,
  not knob by knob. A partial vote configuration is this codebase's standard
  failure mode: a complete, well-formed CSV of different cluster IDs with nothing
  to say it was not what was asked for.
- `vote_flags()` refuses, before the queue: a distance power with no weighting; an
  `adaptive_k` no wider than `k`; an `adaptive_k` with no margin to gate it; and
  `--adaptive-margin` with no explicit `--k`, because `k` otherwise falls back to
  the reference's own `n_neighbors` *inside the container*, so whether the re-vote
  does anything could not be checked until the job was already running.
- The flags go into the returned run record. That is currently the only place two
  CSVs from one reference but different votes can be told apart — the CSV carries
  `hpc_reference` but nothing about the vote.

Defaults are unchanged. The measured configuration is opt-in, for the reason in §19.

## §19 Leave-one-out accuracy and the acceptance test can move in opposite directions

They measure different things, and the settled configuration is expected to help
one and hurt the other.

- **`validate_reference.py`** asks: does the k-NN vote recover the reference's own
  Leiden labels? That is what the sweep optimised, 96.78% → 97.23%.
- **`--validate-against`** asks: do we reproduce Kai's TCGA cluster transfer? Those
  labels came from `sc.tl.ingest`, which is an **unweighted** k-nearest-neighbour
  majority vote at the reference's own `n_neighbors`.

So `--distance-weighted --distance-power 3` makes the classifier *less* like
`ingest` by construction, and adaptive k more so again. A drop in
`--validate-against` agreement is therefore not automatically a defect the way
`CLAUDE.md`'s "<99% is a defect" rule reads for the default configuration — it is
the expected consequence of deliberately no longer imitating `ingest`.

That leaves a real decision, which is why the defaults were not changed:

- The 99% gate is what protects against projection and centering bugs, which
  leave-one-out does not cover at all. Losing it means losing that cover.
- So run the acceptance test **twice**: once at the default to confirm the ≥99%
  gate still holds (that validates projection and centering), and once at the
  settled configuration to see the size of the deliberate divergence. If the
  default still passes and the settled config lands a little below, the pipeline is
  sound and the difference is the intended change. If the *default* drops below 99%,
  that is a real defect and has nothing to do with any of this.

## §20 The runner-up idea, measured to a conclusion

Asked: 93% of errors have the true cluster as runner-up, so why not assign the
runner-up for those? Measured on the production reference at 200,000 tiles, the
answer went through three stages and ended somewhere useful.

### The blanket version cannot work, and the bar is exact

That 93% is conditioned on already knowing the tile is wrong, which at assignment
time is the one thing unknown. Swapping a band of tiles to their runner-up fixes
the wrong ones whose truth is the runner-up and breaks every tile already right:

    net > 0  <=>  p > 1/(1 + r)     p = fraction of the band wrong
                                    r = P(runner-up | wrong) = 0.93 measured

So the band must be **more than 51.9% wrong**. No band is:

| margin < | tiles | wrong | fixed | broke | net |
|---|---|---|---|---|---|
| 0.01 | 729 | 50.6% | 342 | 360 | −18 |
| 0.05 | 3,857 | 45.7% | 1,609 | 2,093 | −484 |
| 0.10 | 7,459 | 41.6% | 2,831 | 4,359 | −1,528 |
| 0.25 | 18,433 | 28.6% | 4,861 | 13,152 | −8,291 |
| 0.75 | 61,701 | 10.3% | 5,893 | 55,331 | −49,438 |

`fixed` saturates while `broke` explodes: from margin < 0.25 to < 0.75 adds 1,032
fixable errors and 42,179 breakable correct tiles. 76% of all errors already sit
below margin 0.25.

### Deciding *which* tiles to swap does work — and only one rule does

`tiebreak_experiment.py` had existed unrun. Of its four rules, exactly one has
signal:

| rule | net at flip 0 |
|---|---|
| `restricted` | **+796** |
| `mean-dist` | −1,137 |
| `nearest` | −1,150 |
| `centroid` | −1,640 |

`restricted` is the only rule that uses **more evidence** — it re-votes at k=50
while counting only the two candidates. `nearest` and `mean-dist` re-read the same
k=10; `centroid` uses a global summary. **The gain comes from more neighbours, not
from a cleverer read of the ones already there.**

### The band is a parameter, and the old grid missed its optimum

| band | best rule | net | adaptive k | hybrid | oracle |
|---|---|---|---|---|---|
| 0.05 | `restricted` @0.02 | +562 | +569 | +597 | 97.27% |
| 0.10 | `restricted` @0.05 | +828 | +840 | +904 | 97.56% |
| **0.15** | `restricted` @0.05 | **+911** | **+910** | **+992** | 97.72% |
| 0.25 | `restricted` @0.10 | +826 | +765 | +916 | 97.88% |
| 0.50 | `restricted` @0.10 | +684 | +564 | +799 | 97.92% |

**0.15 is the optimum for every policy, and the sweep never tested it** — its
grid was `0 / 0.10 / 0.25`. Plain adaptive k at 0.15 gives **97.27%** against
97.23% at 0.10 and 97.19% at 0.25. `tune_classifier.py`'s default `--adaptive`
grid is now `0 / 0.05 / 0.10 / 0.15 / 0.20 / 0.25`; thresholds are re-votes of a
subset, so a coarse grid buys nothing and can hide the answer.

The oracle column keeps rising with the band only because a wider band contains
more reachable tiles. It is not achievable — see below.

### They are complementary, and the hybrid is the best measured policy

At band 0.15, `restricted` (+911) and adaptive k (+910) score the same but agree
on only 82.9% of the band:

| | tiles |
|---|---|
| right after adaptive only | 897 |
| right after the rule only | 898 |
| right after both | 6,990 |
| right after neither | 2,264 |

Not one mechanism. `restricted` adds neighbours *while narrowing to two
candidates*; adaptive k adds neighbours *keeping all 71 open*. The narrowing does
independent work. The hybrid — adaptive's answer, overruled by the rule where
adaptive's own re-vote is still a near-tie — reaches **+992, 97.31%**.

### The selector is real, already exploited, and closed

Where they disagree it is a coin flip overall: 50.9% adaptive on the 341 tiles the
A/B rule structurally cannot reach, 49.8% on the 1,544 head-to-head. Stratified:

| adaptive margin | adaptive wins | | rule advantage | adaptive wins |
|---|---|---|---|---|
| 0.000–0.018 | 39.2% | | 0.000–0.028 | 52.4% |
| 0.018–0.039 | 50.1% | | 0.028–0.058 | 51.8% |
| 0.039–0.072 | 54.7% | | 0.058–0.103 | 50.9% |
| 0.072–0.272 | 54.8% | | 0.103–0.333 | 44.0% |
| | 15.6 pts, 4.3σ | | | 8.4 pts, 2.3σ |

Both monotone and beyond noise, so a selector exists. But:

- **Signal 1 is already fully exploited.** The hybrid's best gate (0.02) sits
  exactly on the first quartile boundary (0.018), which is the only lopsided
  stratum. Q2 is 50.1% — a coin flip — so widening the gate adds nothing, and the
  gate sweep confirms it (+992 at 0.02, declining after).
- **Signal 2 is worth ~35 tiles.** Only its top quartile deviates: 386 tiles at a
  12-point edge, less the overlap with Q1 already taken. **+0.017pp.**

The +0.41pp oracle gap needs a selector that is *right* on 1,795 near-coin-flip
decisions. The best stratum observed is 55/45. A selector at 58% — optimistic,
using both signals — recovers +144 tiles, **+0.07pp**, against the oracle's +898.
Most of that gap is irreducible.

**So this line of work is finished.** Not because nothing was found, but because
what was found has been measured to its limit.

### What to actually do

| | accuracy | cost |
|---|---|---|
| shipped today (`--adaptive-margin 0.1`) | 97.23% | — |
| **`--adaptive-margin 0.15`** | **97.27%** | **one flag, code already shipped** |
| hybrid (band 0.15, gate 0.02) | 97.31% | implement `restricted` in production |

The one-flag change is the recommendation. The hybrid's extra +0.04pp needs a
second mechanism inside `assign_hpc_clusters.py`, and every figure above is
best-of-sweep on **seed 0** — confirm on `--seed 1`, with its own cache file,
before adopting either.

Two independent tools agreeing is worth noting: `tune_classifier`'s sweep and
`tiebreak_experiment`'s adaptive arm both give **97.23%** for `k=10 / pow 3 /
adapt 0.1 / k=25`, computed through separately written code paths from the
baseline vote down.

## §21 The acceptance test, finally run — and what 99.619% does and does not say

Run 2026-08-22 at the **default (legacy) vote**, against
`hdf5_TCGA_LUAD_5x_he_train_reps.h5` (545,185 tiles) and the production reference:

```
Validation: 499,108 of 545,185 tiles matched by (slides, tiles)
  agreement 99.619%  (497,204 / 499,108)
  disagreements 1,904; vote_margin median 0.004 vs 0.632 for agreements
  highest-margin disagreements: 0.048, 0.044, 0.044, 0.040, 0.040
Agreement acceptable.
```

**The gate is closed.** `CLAUDE.md` calls below 99% a defect, and this is the
first time the number has existed. It had never run before for two reasons found
in getting here, both fixed: the truth file lists 100 tiles twice and 96 with
conflicting labels, which the `one_to_one` merge correctly refused (`f8d6f67`);
and passing `--validate-against` as a bare filename produced `--bind .:.`, which
Singularity refuses, killing the job in seconds with an error naming an absolute
path and complaining a path was not absolute (`5adf46b`).

**What it validates: the plumbing.** This is the only check that exercises the PCA
projection, `--centering query`, the `varm/PCs` basis, and the `(slides, tiles)`
join end to end on real embeddings. Leave-one-out covers none of it — it starts
from vectors already in the reference's space. So 99.619% confirms that the path
from encoder output to cluster ID is wired correctly across 499,108 tiles. Given
this pipeline's failure mode, that is the single most valuable thing it could
have told us.

**What it does not say: how accurate the classifier is.** The margin split gives
it away. Disagreements have a median `vote_margin` of **0.004** against **0.632**
for agreements, and the *highest*-margin disagreement in 545,185 tiles is
**0.048**. Every single disagreement is a dead tie.

That is not what a genuine transfer looks like. Leave-one-out errors spread across
the whole margin range — 4,861 of 6,375 below margin 0.25, but with errors present
up to 0.75 (§20). Here they are confined to a band 15x narrower. The explanation
is that these tiles are in the reference: for any tile that is not exactly tied,
the answer is already determined before the vote does any work. So the honest
accuracy estimate remains leave-one-out's ~97%, and that is still optimistic for
a new cohort because a tile's own slide-mates stay in the reference.

**Also learned:** 499,108 tiles matched, not the ~360,000 predicted from the local
copy of the label CSV (360,471 unique keys after dedup). So the cluster's
`TCGA_LUAD_5x_he_train_filtered_leiden_2p5__fold2.csv` is a **third** distinct file
with that name — there are already two locally, one carrying `leiden_2.5` and one
carrying `hpc_id`. Any statement about "the label CSV" has to name which machine.

**Still outstanding**, unchanged by this: the same test at `--vote-preset tuned`
(expected to come out *lower*, per §19 — the tuned vote deliberately stops
imitating `sc.tl.ingest`), and the slide-level holdout, which is the only design
here that would produce an accuracy figure predictive of a new cohort.

## §22 The production reference is LATTICeA, not TCGA — and §21 reads differently

Established 2026-08-22 by reading the reference's own provenance:

```
$ python -c "...json.loads(str(np.load('hpc_reference_leiden_2p5_fold2.npz')['meta']))['source']"
/mnt/.../Vaidehi/cluster reference/
  LATTICeA_5x_he_complete_surv_sex_filtered_leiden_2p5__fold2_subsample.h5ad
```

`hpc_reference_leiden_2p5_fold2.npz` — 2.5M tiles, 127 comps, 71 clusters,
leiden_2.5 — is a **LATTICeA** subsample. §14 and this document elsewhere
described it as TCGA LUAD. That was wrong, and it changes three readings.

**The TCGA acceptance test is not tautological.** §21 argued that 99.619% was
near-tautological because those tiles were probably inside the reference. They
are not: the reference is LATTICeA and the queries are TCGA. So the run was a
genuine cross-cohort transfer of 499,108 tiles, and 99.619% says our pipeline
reproduces `sc.tl.ingest`'s transfer of LATTICeA-defined clusters onto TCGA. That
is a much stronger result than §21 credited.

The margin evidence still holds and now makes better sense. Disagreements sat at
median margin 0.004 with a maximum of 0.048 — because both sides are exact k-NN
in the same space, so they can only differ on near-ties, plus whatever float
precision the projection introduces. Confinement to dead ties is what agreement
between two implementations of the same function looks like, not what a
self-match looks like.

**But it is still agreement, not accuracy.** Kai's TCGA labels were themselves
produced by transferring LATTICeA clusters, so there is no TCGA ground truth
anywhere. 99.619% means "we put TCGA tiles where the reference implementation puts
them", which is the operationally right target — those are the labels the KB
holds — and says nothing about whether either is biologically correct.

**The 97.27% is within-LATTICeA.** Leave-one-out asks whether k-NN recovers a
LATTICeA tile's own label given the rest of LATTICeA. §21 called it a within-TCGA
number; wrong cohort, same conclusion — it is still an upper bound for a third
cohort, and still optimistic because a tile's slide-mates stay in the reference.

**Consequences for the slide holdout.** It holds out **LATTICeA** slides, which is
correct and unchanged in value: it measures a slide the reference has never seen.
Note the h5ad is a `_subsample`, so confirm `obs` carries a slide column and check
how many distinct slides survived subsampling before choosing N.

**What we now have, honestly labelled:**

| measurement | what it is | number |
|---|---|---|
| leave-one-out on the reference | LATTICeA tile vs rest of LATTICeA, slide-mates included | 97.27% |
| slide holdout (to run) | LATTICeA slide the reference never saw | unknown |
| TCGA acceptance test | agreement with `ingest`'s cross-cohort transfer | 99.619% |
| Radiogenomics | no ground truth exists | unmeasured |

**Why this went unnoticed:** the reference file is named
`hpc_reference_leiden_2p5_fold2.npz` and Kai's label file
`TCGA_LUAD_5x_he_train_filtered_leiden_2p5__fold2.csv`. Same `leiden_2p5__fold2`
stem, because it is the same clustering config — one is where the clusters were
defined, the other is where they were transferred. Nothing in either name says
which. `build_hpc_reference.save()` records `meta["source"]` precisely so this is
answerable, and reading it is a one-liner; the code comment in
`validate_reference.describe_reference` asserted TCGA from memory instead, and has
been corrected to say where to look.
