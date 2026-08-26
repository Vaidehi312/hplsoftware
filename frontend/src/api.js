// JS port of app/api_client.py's TileServerClient.
// Every method maps 1:1 to a method there — see that file for the "why"
// behind each endpoint's shape (timeouts, dry-run semantics, etc.).

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

async function request(path, { method = "GET", params, json, timeoutMs = 30000, ...rest } = {}) {
  const url = new URL(BASE_URL + path);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) url.searchParams.set(k, v);
    }
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method,
      headers: json !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: json !== undefined ? JSON.stringify(json) : undefined,
      signal: controller.signal,
      ...rest,
    });
    if (!res.ok) {
      let body = null;
      try {
        body = await res.json();
      } catch {
        /* not JSON */
      }
      throw new ApiError(`${method} ${path} failed: ${res.status}`, res.status, body);
    }
    return res;
  } finally {
    clearTimeout(timer);
  }
}

async function getJson(path, opts) {
  return (await request(path, opts)).json();
}

async function postJson(path, jsonBody, opts) {
  return (await request(path, { method: "POST", json: jsonBody, ...opts })).json();
}

function imageUrl(path, params) {
  const url = new URL(BASE_URL + path);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) url.searchParams.set(k, v);
    }
  }
  return url.toString();
}

export const api = {
  baseUrl: BASE_URL,
  ApiError,

  // -- Health / slide list --------------------------------------------
  health: () => getJson("/health"),
  listSlides: async () => (await getJson("/slides")).slides,

  // -- Upload -----------------------------------------------------------
  async uploadSlide(file, slideId, confirmOverwrite = false) {
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("slide_id", (slideId || "").trim());
    form.append("confirm_overwrite", confirmOverwrite ? "true" : "false");
    return postFormData("/upload-slide", form, { timeoutMs: 600000 });
  },
  getProcessingStatus: (slideId) => getJson(`/slide/${slideId}/processing-status`),

  // -- Dataset-wide Slurm jobs -------------------------------------------
  getDatasetRoots: async () => (await getJson("/dataset-roots")).datasets,
  getTileDatasetNames: async () => (await getJson("/tile-dataset-names")).dataset_names,

  submitDatasetJob: (body) => postJson("/dataset-jobs", cleanBody(body)),

  listDatasetJobs: (withState = false) =>
    withState
      ? getJson("/dataset-jobs", { params: { with_state: "true" }, timeoutMs: 60000 })
      : getJson("/dataset-jobs"),

  listDatasets: () => getJson("/datasets", { timeoutMs: 60000 }),

  getDatasetCoverage: async (datasetName, rawDir) =>
    (
      await getJson("/datasets", {
        params: { dataset_name: datasetName, raw_dir: rawDir, coverage: "true" },
        timeoutMs: 300000,
      })
    ).datasets,

  getDatasetJobHistory: (submissionId) =>
    getJson(`/dataset-jobs/${submissionId}/jobs`, { timeoutMs: 60000 }),

  getDatasetJobStatus: (submissionId) =>
    getJson(`/dataset-jobs/${submissionId}/status`, { timeoutMs: 60000 }),

  resumeDatasetJob: (submissionId) => postJson(`/dataset-jobs/${submissionId}/resume`, {}),
  cancelDatasetJob: (submissionId) => postJson(`/dataset-jobs/${submissionId}/cancel`, {}),

  getTiledCoverage: (submissionId) => getJson(`/dataset-jobs/${submissionId}/tiled-coverage`),

  startPackagingJob: (submissionId, { allowIncomplete = false, scope = "run", resume } = {}) => {
    const params = { allow_incomplete: allowIncomplete ? "true" : "false", scope };
    if (resume !== undefined && resume !== null) params.resume = resume ? "true" : "false";
    return postJson(`/dataset-jobs/${submissionId}/package`, {}, { params });
  },

  getPackagingProgress: (submissionId, exact = true) =>
    getJson(`/dataset-jobs/${submissionId}/packaging-progress`, {
      params: { exact: exact ? "true" : "false" },
      timeoutMs: 60000,
    }),

  startFeatureExtraction: (submissionId, { checkpoint, model = "BarlowTwins_3", marker = "he" }) =>
    postJson(`/dataset-jobs/${submissionId}/extract-features`, { checkpoint, model, marker }),

  votePresets: () => getJson("/vote-presets"),

  startClusterAssignment: (
    submissionId,
    { reference = null, k = null, overwrite = false, votePreset = null, voteOverrides = {} } = {}
  ) =>
    postJson(`/dataset-jobs/${submissionId}/assign-clusters`, {
      reference,
      k,
      overwrite,
      ...(votePreset ? { vote_preset: votePreset } : {}),
      ...voteOverrides,
    }),

  startTestClusterAssignment: (
    submissionId,
    { projectionsH5, reference = null, k = null, votePreset = null, voteOverrides = {} }
  ) =>
    postJson(`/dataset-jobs/${submissionId}/assign-clusters-test`, {
      projections_h5: projectionsH5,
      reference,
      k,
      ...(votePreset ? { vote_preset: votePreset } : {}),
      ...voteOverrides,
    }),

  cohortShiftReadiness: (submissionId) =>
    getJson(`/dataset-jobs/${submissionId}/cohort-shift-readiness`),

  checkCohortShift: (submissionId, { csvPath = null, topSlides = 10 } = {}) =>
    postJson(`/dataset-jobs/${submissionId}/cohort-shift`, { csv_path: csvPath, top_slides: topSlides }),

  // Registration — the identity rows Stage 6's UPDATE needs to exist. Every
  // path the endpoint needs is already on the run record, so only the cohort
  // key and the two opt-ins are sent.
  previewRegistration: (
    submissionId,
    { datasetId = null, slideMetadata = false, writeDatasetConfig = true, replace = false } = {}
  ) =>
    postJson(`/dataset-jobs/${submissionId}/register-preview`, {
      dataset_id: datasetId,
      slide_metadata: slideMetadata,
      write_dataset_config: writeDatasetConfig,
      replace,
    }),

  commitRegistration: (
    submissionId,
    { datasetId = null, slideMetadata = false, writeDatasetConfig = true, replace = false } = {}
  ) =>
    postJson(`/dataset-jobs/${submissionId}/register`, {
      dataset_id: datasetId,
      slide_metadata: slideMetadata,
      write_dataset_config: writeDatasetConfig,
      replace,
    }),

  previewKbLoad: (submissionId, { minMargin = 0.0, csvPath = null } = {}) =>
    postJson(`/dataset-jobs/${submissionId}/kb-load-preview`, {
      min_margin: minMargin,
      csv_path: csvPath,
    }),

  commitKbLoad: (
    submissionId,
    { cancerType = null, allowUnknownClusters = false, skipProfiles = false, minMargin = 0.0, csvPath = null } = {}
  ) =>
    postJson(`/dataset-jobs/${submissionId}/kb-load`, {
      cancer_type: cancerType,
      allow_unknown_clusters: allowUnknownClusters,
      skip_profiles: skipProfiles,
      csv_path: csvPath,
      min_margin: minMargin,
    }),

  startTestPackaging: (
    submissionId,
    { sampleSize = null, slideNames = null, randomSeed = null, scope = "run" } = {}
  ) =>
    postJson(`/dataset-jobs/${submissionId}/package-test`, {
      sample_size: sampleSize,
      slide_names: slideNames,
      random_seed: randomSeed,
      scope,
    }),

  getTestPackagingStatus: (submissionId, jobId, outputPath) =>
    getJson(`/dataset-jobs/${submissionId}/package-test-status`, {
      params: { job_id: jobId, output_path: outputPath },
    }),

  startTestFeatureExtraction: (
    submissionId,
    { h5Path, checkpoint, model = "BarlowTwins_3", marker = "he" }
  ) =>
    postJson(`/dataset-jobs/${submissionId}/extract-features-test`, {
      h5_path: h5Path,
      checkpoint,
      model,
      marker,
    }),

  getTestFeatureExtractionStatus: (submissionId, jobId, outputPath) =>
    getJson(`/dataset-jobs/${submissionId}/extract-features-test-status`, {
      params: { job_id: jobId, output_path: outputPath },
    }),

  // -- Slide-level --------------------------------------------------------
  getSlideInfo: (slideId) => getJson(`/slide/${slideId}/info`),
  thumbnailUrl: (slideId, maxWidth = 3000, quality = 85) =>
    imageUrl(`/slide/${slideId}/thumbnail`, { max_width: maxWidth, quality }),
  tileUrl: (slideId, level, x, y, w = 256, h = 256, quality = 85) =>
    imageUrl(`/slide/${slideId}/tile`, { level, x, y, w, h, quality }),
  regionUrl: (slideId, x, y, w, h, level = 0, quality = 85) =>
    imageUrl(`/slide/${slideId}/region`, { x, y, w, h, level, quality }),

  // -- Tile metadata --------------------------------------------------------
  getTilesMeta: (slideId) => getJson(`/slide/${slideId}/tiles_meta`),
  getAdjacency: (slideId) => getJson(`/slide/${slideId}/adjacency`),

  // -- HPC -------------------------------------------------------------------
  getHpcInfo: (hpcId) => getJson(`/hpc/${hpcId}/info`),
  getHpcSurvival: (hpcId) => getJson(`/hpc/${hpcId}/survival`),

  // -- H5 tile image by slide_tile key ---------------------------------------
  tileImageUrl: (slideTile, quality = 85) => imageUrl(`/tile_image/${slideTile}`, { quality }),

  // -- DZI (OpenSeadragon reads this directly, but exposed for convenience) --
  dziUrl: (slideId) => `${BASE_URL}/dzi/${slideId}.dzi`,
};

async function postFormData(path, form, { timeoutMs = 30000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(BASE_URL + path, { method: "POST", body: form, signal: controller.signal });
    if (!res.ok) {
      let body = null;
      try {
        body = await res.json();
      } catch {
        /* not JSON */
      }
      throw new ApiError(`POST ${path} failed: ${res.status}`, res.status, body);
    }
    return res.json();
  } finally {
    clearTimeout(timer);
  }
}

// Drop null/undefined keys so the server doesn't get an explicit null where
// omission has different meaning (see submit_dataset_job's min_tissue note
// in api_client.py).
function cleanBody(body) {
  const out = {};
  for (const [k, v] of Object.entries(body)) {
    if (v !== undefined && v !== null) out[k] = v;
  }
  return out;
}
