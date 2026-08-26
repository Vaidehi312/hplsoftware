// Port of _render_new_dataset_submission_form() in app_v28.py (~lines
// 893-1042): dataset folder choice, Slurm options, slide selection, submit.
import { useEffect, useState } from "react";
import { api } from "../../api";
import { Alert, Button, Field, RadioGroup } from "./widgets";

export default function NewRunForm({ datasetPath, onSubmitted }) {
  const [existingNames, setExistingNames] = useState([]);
  const [namesError, setNamesError] = useState(null);
  const [namesLoaded, setNamesLoaded] = useState(false);

  const [folderMode, setFolderMode] = useState("Existing dataset folder");
  const [existingFolder, setExistingFolder] = useState("");
  const [newFolder, setNewFolder] = useState("");

  const [partition, setPartition] = useState("");
  const [notifyEmail, setNotifyEmail] = useState("");
  const [maxConcurrent, setMaxConcurrent] = useState(10);
  const [minTissue, setMinTissue] = useState(30);

  const [selectionMode, setSelectionMode] = useState("All slides");
  const [sampleSize, setSampleSize] = useState(10);
  const [slideNamesRaw, setSlideNamesRaw] = useState("");

  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState(null); // {type, text}

  useEffect(() => {
    let cancelled = false;
    api
      .getTileDatasetNames()
      .then((names) => {
        if (cancelled) return;
        setExistingNames(names || []);
        setNamesLoaded(true);
      })
      .catch((e) => {
        if (cancelled) return;
        // Tile server unreachable, route 404s, etc. — fall back to "always
        // create new" rather than blocking the whole form, but keep the
        // error so it isn't confused with "genuinely no folders exist yet".
        setNamesError(String(e.message || e));
        setNamesLoaded(true);
      });
    return () => {
      cancelled = true;
    };
    // datasetPath isn't used by this fetch, but re-checking when the parent
    // reuses this form for a different path costs nothing and stays fresh.
  }, [datasetPath]);

  useEffect(() => {
    if (folderMode === "Existing dataset folder" && !existingFolder && existingNames.length) {
      setExistingFolder(existingNames[0]);
    }
  }, [existingNames, folderMode, existingFolder]);

  const datasetName = folderMode === "Existing dataset folder" ? existingFolder : newFolder;

  const slideNames = selectionMode === "Specific slides"
    ? slideNamesRaw.split("\n").map((s) => s.trim()).filter(Boolean)
    : null;

  async function handleSubmit() {
    setMessage(null);
    if (!datasetPath.trim()) {
      setMessage({ type: "error", text: "Enter a dataset path first." });
      return;
    }
    if (selectionMode === "Specific slides" && (!slideNames || !slideNames.length)) {
      setMessage({ type: "error", text: "Enter at least one slide filename or ID." });
      return;
    }
    if (!datasetName || !datasetName.trim()) {
      setMessage({ type: "error", text: "Enter a name for the new dataset folder, or switch to an existing one." });
      return;
    }
    setSubmitting(true);
    try {
      const result = await api.submitDatasetJob({
        dataset_path: datasetPath.trim(),
        max_concurrent: Number(maxConcurrent),
        min_tissue: Number(minTissue),
        sample_size: selectionMode === "Random subset" ? Number(sampleSize) : null,
        slide_names: slideNames,
        partition: partition.trim() || null,
        notify_email: notifyEmail.trim() || null,
        dataset_name: datasetName.trim() || null,
      });
      setMessage({
        type: "success",
        text:
          `Queued — submission ${result.submission_id} into dataset folder ` +
          `'${result.dataset_name || datasetName.trim()}'. Discovering slides and submitting ` +
          `to Slurm in the background; check it under the dataset workspace above.`,
      });
      onSubmitted && onSubmitted(result);
    } catch (e) {
      const detail = e instanceof api.ApiError ? (e.body ? JSON.stringify(e.body) : e.message) : String(e);
      setMessage({ type: "error", text: `Submission failed: ${detail}` });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div>
      <RadioGroup
        label="Tiles go into"
        name="pipeline-new-run-folder-mode"
        options={["Existing dataset folder", "New dataset folder"]}
        value={folderMode}
        onChange={setFolderMode}
        help="Which folder under processed_tiles this run's tiles land in, e.g. TCGA or Radiogenomics — keeps different datasets from mixing together on disk."
      />

      {folderMode === "Existing dataset folder" ? (
        existingNames.length ? (
          <Field label="Dataset folder">
            <select value={existingFolder} onChange={(e) => setExistingFolder(e.target.value)}>
              {existingNames.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </Field>
        ) : namesError ? (
          <Alert type="warning">Couldn&apos;t load existing dataset folders: {namesError}</Alert>
        ) : namesLoaded ? (
          <Alert type="info">No dataset folders exist yet — switch to &quot;New dataset folder&quot; to create one.</Alert>
        ) : (
          <div className="pipeline-caption">Loading dataset folders…</div>
        )
      ) : (
        <Field label="New dataset folder name" help="Letters, numbers, '.', '_', '-' only, e.g. TCGA or Radiogenomics.">
          <input type="text" value={newFolder} onChange={(e) => setNewFolder(e.target.value)} />
        </Field>
      )}

      <Field
        label="Slurm partition (optional)"
        help="Leave blank to use the cluster's default partition. Only set this to override it (e.g. a low-priority queue for a large non-urgent run) — masking/tiling is CPU-only, no GPU partition needed."
      >
        <input type="text" value={partition} onChange={(e) => setPartition(e.target.value)} />
      </Field>

      <Field
        label="Email me when the job finishes (optional)"
        help="Uses Slurm's own end-of-job notification (one email for the whole array, on completion or failure) — not a summary or the log files themselves. Only works if your cluster has a mail relay configured; test with a throwaway job first if you're not sure."
      >
        <input type="text" value={notifyEmail} onChange={(e) => setNotifyEmail(e.target.value)} />
      </Field>

      <Field label="Max concurrent Slurm tasks">
        <input
          type="number"
          min={1}
          max={200}
          value={maxConcurrent}
          onChange={(e) => setMaxConcurrent(e.target.value)}
        />
      </Field>

      <Field
        label={`Minimum tissue % per tile: ${minTissue}`}
        help="Tiles below this tissue coverage are skipped. Lower this if a dataset comes back with mostly zero-tile slides."
      >
        <input
          type="range"
          min={0}
          max={100}
          step={5}
          value={minTissue}
          onChange={(e) => setMinTissue(Number(e.target.value))}
        />
      </Field>

      <RadioGroup
        label="Slides to run"
        name="pipeline-new-run-selection-mode"
        options={["All slides", "Random subset", "Specific slides"]}
        value={selectionMode}
        onChange={setSelectionMode}
      />

      {selectionMode === "Random subset" && (
        <Field label="Number of random slides">
          <input type="number" min={1} value={sampleSize} onChange={(e) => setSampleSize(e.target.value)} />
        </Field>
      )}
      {selectionMode === "Specific slides" && (
        <Field label="Slides to run (one per line)" help="Match by original filename (e.g. slide1.ndpi) or slide ID.">
          <textarea rows={4} value={slideNamesRaw} onChange={(e) => setSlideNamesRaw(e.target.value)} />
        </Field>
      )}

      <Button kind="primary" onClick={handleSubmit} disabled={submitting}>
        {submitting ? "Submitting…" : "Submit dataset job"}
      </Button>

      {message && <Alert type={message.type}>{message.text}</Alert>}
    </div>
  );
}
