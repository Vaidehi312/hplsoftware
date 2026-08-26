# JS frontend port — what was built and why (2026-08-24)

Goal: build a JavaScript/React UI as an alternative to the Streamlit UI
(`app/app_v28.py`), without touching `app_v28.py` or the FastAPI backend, so
both UIs can run side by side against the same backend.

## 1. Mapped the existing Streamlit app before writing anything

Before touching code, a background agent read `app/app_v28.py` (4,946 lines)
and `app/api_client.py` (566 lines) and reported back:

- The page layout: sidebar (upload panel + the 5-stage pipeline wizard) and
  main body (chat + slide viewer).
- Every `api_client.py` method actually called, and from where.
- `st.session_state` usage that matters for state design (which keys persist
  across reruns and why).
- Polling behavior: `@st.fragment(run_every=...)` on the dataset workspace
  (10s), packaging live progress (30s), and test-packaging progress (15s) —
  everything else is manual refresh-on-click.
- One important finding: **two separate backend-access paths** exist in
  `app_v28.py`. Most of the UI goes through `api_client.TileServerClient`
  (HTTP, to FastAPI). But the chat feature and two specific tile-viewer
  helpers (`load_hpc_titles`, `load_survival_coefficients`,
  `slides_for_hpc_from_kb`) query Postgres directly via SQLAlchemy, bypassing
  the API entirely.

Reason for doing this first: a 5,000-line app can't be ported by guessing at
structure. Getting a factual map up front is what let later steps scope the
work accurately instead of discovering gaps mid-build.

## 2. Scoped the work with you before building

Asked (and got answers on):
- **Stack**: React + Vite — a component-based SPA is the natural fit for a
  5-stage wizard with polling and a slide viewer; no build-step-free option
  was going to hold up as the app grew.
- **Rollout**: a separate dev server (Vite on :5173) hitting the same FastAPI
  backend (:8000), rather than folding the new UI into the FastAPI process.
  Keeps the two UIs fully independent — nothing about running one affects the
  other.
- **Scope for this pass**: the 5-stage pipeline wizard *and* the slide
  viewer, explicitly deferring the NL chatbot and the side-by-side two-slide
  "HPC explorer" compare view. Reason: those two features are the parts of
  `app_v28.py` that bypass the FastAPI backend and hit Postgres directly —
  including them well would mean either adding backend endpoints (against
  the "little to no changes" ask) or having the browser talk to Postgres
  directly (a real security problem, browsers should never hold DB
  credentials). Keeping them out kept the "no backend changes" constraint
  honest instead of quietly breaking it for two screens.

## 3. Confirmed zero backend changes were actually needed

Checked `backend/tile_server_v2_.py` directly: it already has
`CORSMiddleware` with `allow_origins=["*"]`. A browser-based frontend on a
different port can call it as-is. This was verified by reading the file, not
assumed — CORS is exactly the kind of thing that silently blocks a new
frontend if missed.

## 4. Built the shared foundation directly (not delegated)

These needed to be right before anything else could be built on top of them,
so they were written directly rather than handed to a subagent:

- **`frontend/src/api.js`** — a JS port of `api_client.py`'s
  `TileServerClient`, one method per endpoint, same semantics (timeouts,
  dry-run behavior, `min_tissue`/`tiling_params` omission rules, etc.) as the
  Python client's own docstrings describe. Reason: every other component
  needed one shared, correct place to call the backend from — duplicating
  fetch calls per-component would have reintroduced the exact
  "duplicate upload logic" bug the map found in `app_v28.py` itself
  (`_post_upload_slide` reimplementing `client.upload_slide`).
- **`frontend/src/colors.js`** — ports `color_for_hpc` (MD5 hash → HSV hue),
  `color_for_inflammation`, `color_for_necrosis`, and `risk_to_rgba` from
  Python to JS. Reason: colors need to match `app_v28.py`'s exactly (same HPC
  id → same color) so the two UIs are describing the same thing, not just
  visually similar. This required installing `blueimp-md5` since browsers
  don't expose MD5 in the Web Crypto API and Python's `hashlib.md5` had to be
  matched bit-for-bit.
- **`frontend/src/hooks/usePolling.js`** — the JS analogue of
  `@st.fragment(run_every="Ns")`: an interval-based re-fetch hook. Reason:
  three different screens need the same polling shape (interval, pause when
  not visible, expose loading/error/data) — one hook, used three ways,
  instead of three ad hoc `setInterval` calls.
- **`frontend/src/App.jsx` / `App.css` / `index.css`** — the page shell
  (sidebar for the pipeline panel, main area for a slide picker + viewer),
  replacing the Vite starter template. Reason: something had to define where
  the two big halves (built next) would actually mount, and what contract
  they had to satisfy (`PipelinePanel` with no required props;
  `SlideViewer({ slideId })`).
- **`.claude/launch.json`** — registered the frontend as a previewable dev
  server (`npm --prefix frontend run dev`), so it can be opened for a visual
  check without touching the backend's own launch config.

## 5. Delegated the two large, independent halves to background agents

The two remaining chunks — the pipeline wizard and the slide viewer — are
each substantial (thousands of lines of source logic to port) but don't
depend on each other, so they ran as two parallel background agents rather
than serially. Reason: no shared state between them beyond the already-built
foundation, so parallelizing cost nothing in correctness and roughly halved
wall-clock time.

**Agent A — pipeline wizard** (`frontend/src/components/pipeline/`)
Ported: WSI upload (with the 409-conflict/overwrite flow), the 10s-polled
dataset workspace, the new-run submission form, and all 5 stages (tiling,
packaging incl. test-on-a-subset mode, feature extraction, cluster
assignment with vote presets, KB load with dry-run preview + guarded
commit), plus job history and the cancel-run button. Told explicitly to
preserve every guard/refusal condition found in the source (e.g. the KB-load
commit button must stay disabled when the preview reports
`would_refuse_low_match`) rather than loosen anything for convenience — this
matches `CLAUDE.md`'s description of the pipeline being written against
"plausible-but-wrong" failures, where a guard exists specifically to stop bad
output before it's queued.

**Agent B — slide viewer** (`frontend/src/components/viewer/`)
Ported: `show_wsi()` in full — both viewer modes (OpenSeadragon pyramid zoom
with an SVG tile overlay, and click-tile-inspector), all 7 highlight modes
(HPC clusters, Inflammation, Necrosis, Malignant, Adjacency, Heatmap,
Survival Risk Heatmap), the HPC annotation panel, tile info panel, and
adjacency controls. Told explicitly how to handle the two DB-bypass
functions found in step 1 (`load_hpc_title_map`, `load_survival_coefficients`)
without adding backend endpoints: call the existing per-HPC endpoints
(`/hpc/{id}/info`, `/hpc/{id}/survival`) for just the HPC ids present on the
current slide, and cache results client-side (`hpcReferenceCache.js`) since
the HPC dictionary is small, static reference data. This is what let the
viewer keep full functional parity for those two features while genuinely
touching zero backend code.

Both agents were told to skip the chat pipeline and the two-viewer compare
mode (per the scoping decision in step 2), and to run `npm run build`
themselves before reporting back, so build errors were caught before
integration rather than after.

## 6. Verified the two halves against each other, not just against their own reports

An agent's summary of its own work is a claim, not a fact, so before treating
either half as done:

- Ran `npm run build` on the whole app (not just each half in isolation) —
  passed clean.
- Read `components/pipeline/index.jsx` and `components/viewer/index.jsx`
  directly and checked their actual exports against the contract `App.jsx`
  expects (`export default function PipelinePanel()`, no required props;
  `export default function SlideViewer({ slideId })`) — both matched.
- Attempted a live check: started `backend/tile_server_v2_.py` locally to
  hit it from a running frontend. It failed at startup with
  `connection to server at "127.0.0.1", port 5432 failed: Connection
  refused` — expected, since the backend needs an SSH tunnel to the
  cluster's Postgres on :5433 that only exists in your environment, not this
  one. Confirmed no process was left listening afterward. This means the
  port has been verified to *build* correctly but not yet verified against
  *live data* — that step still needs to happen in your environment.

## What's deliberately not built

- The NL chatbot / query pipeline (`query_planner_v25`, `llm_layer_v25`,
  etc.) — bypasses the FastAPI backend entirely in the original; including
  it here would have meant either new backend endpoints or the browser
  talking to Postgres directly.
- The side-by-side two-slide "HPC explorer" compare view — same reasoning,
  it's the other feature built on the direct-DB path.

## What's a known, intentional simplification (not a bug)

A few small behavior differences the build agents flagged and left in place
rather than "fixing" beyond what a faithful port calls for:
- `st.toast()` → a dismissing banner (no toast host exists outside
  Streamlit).
- The heatmap intensity slider affects the click-inspector raster overlay
  only, not the pyramid-zoom overlay — this matches a quirk already present
  in `app_v28.py`, not something introduced by the port.
- The malignant-status legend buckets by canonical category rather than by
  raw lowercased string first — functionally equivalent, slightly more
  robust to messy data than the original.

## How to run it

```bash
cd backend && python tile_server_v2_.py       # FastAPI on :8000 (needs your DB tunnel)
cd frontend && npm install && npm run dev     # Vite on :5173
```

`app_v28.py` keeps running unchanged, on its own Streamlit port, against the
same backend.
