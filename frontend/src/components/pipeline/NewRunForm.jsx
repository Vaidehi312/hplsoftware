// Port of _render_new_dataset_submission_form() in app_v28.py: dataset folder
// choice, Slurm options, slide selection, the Stage 3/4 inputs, and the one
// "Run pipeline" button — Stages 1-4 as a single Nextflow run, as ANORAK runs.
import { useEffect, useState } from "react";
import { api } from "../../api";
import { Alert, Button, Caption, CodeBlock, Expander, Field, RadioGroup } from "./widgets";
import { VotePicker } from "./stages/AssignmentStage.jsx";
import { httpDetail } from "./utils";

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

  const [seed, setSeed] = useState("");

  // Stages 3 and 4, asked for once here rather than at their own buttons.
  const [checkpoint, setCheckpoint] = useState("");
  const [extractionShards, setExtractionShards] = useState(1);
  const [reference, setReference] = useState("");
  const [votePreset, setVotePreset] = useState("");
  const [voteOverrides, setVoteOverrides] = useState({});
  const [assignmentShards, setAssignmentShards] = useState(1);
  const [device, setDevice] = useState("auto");
  const [chain, setChain] = useState(3);
  const [timeLimit, setTimeLimit] = useState("");
  const [allowIncomplete, setAllowIncomplete] = useState(false);
  const [outputs, setOutputs] = useState(null);

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
    if (!checkpoint.trim()) {
      setMessage({ type: "error", text: "A model checkpoint is required — feature extraction cannot run without one." });
      return;
    }
    setSubmitting(true);
    setOutputs(null);
    try {
      const result = await api.startPipelineRun({
        dataset_path: datasetPath.trim(),
        max_concurrent: Number(maxConcurrent),
        min_tissue: Number(minTissue),
        sample_size: selectionMode === "Random subset" ? Number(sampleSize) : null,
        slide_names: slideNames,
        seed: selectionMode === "Random subset" && /^\d+$/.test(seed.trim()) ? Number(seed.trim()) : null,
        partition: partition.trim() || null,
        notify_email: notifyEmail.trim() || null,
        dataset_name: datasetName.trim() || null,
        checkpoint: checkpoint.trim(),
        extraction_shards: Number(extractionShards),
        reference: reference.trim() || null,
        vote_preset: votePreset || null,
        ...voteOverrides,
        assignment_shards: Number(assignmentShards),
        device,
        chain: Number(chain),
        time_limit: timeLimit.trim() || null,
        allow_incomplete: allowIncomplete,
      });
      setMessage({
        type: "success",
        text:
          `Pipeline queued — run ${result.submission_id} into dataset folder ` +
          `'${result.dataset_name || datasetName.trim()}'. Slides are being found and the head job ` +
          `submitted in the background; follow each stage in the dataset workspace above. ` +
          `GPU: ${result.gpu_gres} (${result.gpu_gres_reason}) · search: ${result.device} ` +
          `(${result.device_reason}) · vote: ${result.vote}`,
      });
      setOutputs([result.h5_output_path, result.extraction_output_path, result.assignment_output_path].filter(Boolean));
      onSubmitted && onSubmitted(result);
    } catch (e) {
      setMessage({ type: "error", text: `Refused: ${httpDetail(e)}` });
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

      {selectionMode === "Random subset" && (
        <Field
          label="Sampling seed (optional)"
          help="Leave blank and one is chosen and recorded with the run, so the same slides can be asked for again — a subset is how a test run is done, exactly as for ANORAK."
        >
          <input type="text" value={seed} onChange={(e) => setSeed(e.target.value)} />
        </Field>
      )}

      {/* Stages 3 and 4. Each stage is still checked on its own: the pipeline
          marks a stage done only after the validator the server's gate uses. */}
      <h4 className="pipeline-subheading">Feature extraction (Stage 3)</h4>
      <Field
        label="Model checkpoint path"
        help="Absolute path to the frozen encoder checkpoint, e.g. .../weights/BarlowTwins_3.ckt (a TensorFlow prefix)."
      >
        <input type="text" value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)} />
      </Field>
      <Field
        label="GPU shards"
        help="Split the encode across this many GPU tasks. Extraction is read-bound, and separate processes are the only lever that scales it."
      >
        <input type="number" min={1} max={64} value={extractionShards} onChange={(e) => setExtractionShards(e.target.value)} />
      </Field>

      <h4 className="pipeline-subheading">Cluster classification (Stage 4)</h4>
      <VotePicker
        submissionId="new_run"
        onChange={(name, overrides) => {
          setVotePreset(name);
          setVoteOverrides(overrides || {});
        }}
      />
      <Field
        label="Reference .npz (optional)"
        help="Leave blank for the configured reference. Cluster IDs only mean anything relative to one reference."
      >
        <input type="text" value={reference} onChange={(e) => setReference(e.target.value)} />
      </Field>
      <Field label="Assignment shards" help="More than one computes a shared query mean first, so every shard centres identically.">
        <input type="number" min={1} max={64} value={assignmentShards} onChange={(e) => setAssignmentShards(e.target.value)} />
      </Field>
      <RadioGroup
        label="Search device"
        name="pipeline-new-run-device"
        options={["auto", "cpu", "gpu"]}
        value={device}
        onChange={setDevice}
        help="auto uses a GPU only when the GPU faiss extras are installed. The GPU search is the same exact scan, verified against the CPU one."
      />

      <Field
        label="Package without slides that fail to tile"
        help="Off: one slide that cannot be tiled (after retries) stops the run before packaging, so the .h5 is never missing slides silently. On: those slides are left out and named on the tiling step."
      >
        <input type="checkbox" checked={allowIncomplete} onChange={(e) => setAllowIncomplete(e.target.checked)} />
      </Field>

      <Expander title="Head job">
        <Caption>
          The pipeline runs as one Slurm head job that submits every task itself, supervised by the same stall
          watchdog as ANORAK.
        </Caption>
        <Field
          label="Head jobs"
          help="The head job plus standbys. A standby starts only if the one before it ended without finishing, and resumes it."
        >
          <input type="number" min={1} max={10} value={chain} onChange={(e) => setChain(e.target.value)} />
        </Field>
        <Field label="Head job walltime (optional)" help="Slurm format, e.g. 2-00:00:00. Blank uses the server default.">
          <input type="text" value={timeLimit} onChange={(e) => setTimeLimit(e.target.value)} />
        </Field>
      </Expander>

      <Button kind="primary" onClick={handleSubmit} disabled={submitting}>
        {submitting ? "Checking and submitting…" : "Run pipeline (Stages 1-4)"}
      </Button>

      {message && <Alert type={message.type}>{message.text}</Alert>}
      {outputs && outputs.length > 0 && (
        <>
          <Caption>Outputs will land at:</Caption>
          <CodeBlock>{outputs.join("\n")}</CodeBlock>
        </>
      )}
    </div>
  );
}
