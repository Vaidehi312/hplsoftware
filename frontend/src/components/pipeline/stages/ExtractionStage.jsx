// Port of the feature-extraction stage in app_v28.py: _render_extraction_step,
// _render_extract_features_form and _render_test_extraction (~lines
// 1191-1217, 2230-2329).
import { useState } from "react";
import { api } from "../../../api";
import { errorDetail, SLURM_IN_FLIGHT } from "../utils";
import { Alert, Button, CodeBlock, Field, RadioGroup } from "../widgets";

// Checkpoint input + submit, used both for a first attempt and a retry — a
// failed attempt is not a permanent dead end, the server only blocks a
// genuinely still-running or already-succeeded attempt.
function ExtractFeaturesForm({ submissionId, buttonLabel = "Start feature extraction", onQueued }) {
  const [checkpoint, setCheckpoint] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState(null);

  async function handleSubmit() {
    if (!checkpoint.trim()) {
      setMessage({ type: "error", text: "Enter a checkpoint path first." });
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      await api.startFeatureExtraction(submissionId, { checkpoint: checkpoint.trim() });
      setMessage({ type: "success", text: "Feature extraction job queued." });
      onQueued && onQueued();
    } catch (e) {
      setMessage({ type: "error", text: `Failed to start feature extraction: ${errorDetail(e).message}` });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <Field label="Model checkpoint path">
        <input
          type="text"
          placeholder="/path/to/BarlowTwins_3.ckt"
          value={checkpoint}
          onChange={(e) => setCheckpoint(e.target.value)}
        />
      </Field>
      <Button kind="primary" onClick={handleSubmit} disabled={busy}>
        {busy ? "Starting…" : buttonLabel}
      </Button>
      {message && <Alert type={message.type}>{message.text}</Alert>}
    </div>
  );
}

// Validate a checkpoint against a small .h5 (typically from test packaging)
// before running it over the full dataset. Not recorded against the run.
function TestExtractionPanel({ submissionId }) {
  const [h5Path, setH5Path] = useState("");
  const [checkpoint, setCheckpoint] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState(null);
  const [tracked, setTracked] = useState(null); // { job_id, output_path }
  const [testStatus, setTestStatus] = useState(null);
  const [statusError, setStatusError] = useState(null);

  async function handleSubmit() {
    if (!h5Path.trim() || !checkpoint.trim()) {
      setMessage({ type: "error", text: "Both a .h5 path and a checkpoint are required." });
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const result = await api.startTestFeatureExtraction(submissionId, {
        h5Path: h5Path.trim(),
        checkpoint: checkpoint.trim(),
      });
      const job = { job_id: result.extraction_job_id, output_path: result.expected_output_path };
      setTracked(job);
      setMessage({ type: "success", text: `Test extraction queued (job ${result.extraction_job_id}).` });
      refreshStatus(job);
    } catch (e) {
      setMessage({ type: "error", text: `Test extraction failed: ${errorDetail(e).message}` });
    } finally {
      setBusy(false);
    }
  }

  async function refreshStatus(job) {
    const target = job || tracked;
    if (!target || !target.job_id) return;
    try {
      const status = await api.getTestFeatureExtractionStatus(submissionId, target.job_id, target.output_path);
      setTestStatus(status);
      setStatusError(null);
    } catch (e) {
      setStatusError(errorDetail(e).message);
    }
  }

  return (
    <div>
      <div className="pipeline-caption">
        Validate a checkpoint against a small .h5 (typically one from test packaging above) before
        running it over the full dataset.
      </div>
      <Field label="Test .h5 path">
        <input
          type="text"
          placeholder="/path/to/hdf5_..._test_sample_....h5"
          value={h5Path}
          onChange={(e) => setH5Path(e.target.value)}
        />
      </Field>
      <Field label="Model checkpoint path">
        <input
          type="text"
          placeholder="/path/to/BarlowTwins_3.ckt"
          value={checkpoint}
          onChange={(e) => setCheckpoint(e.target.value)}
        />
      </Field>
      <Button kind="primary" onClick={handleSubmit} disabled={busy}>
        {busy ? "Starting…" : "Start test extraction"}
      </Button>
      {message && <Alert type={message.type}>{message.text}</Alert>}

      {tracked && (
        <div>
          <div className="pipeline-caption">Test extraction job:</div>
          <CodeBlock>{tracked.job_id}</CodeBlock>
          {statusError ? (
            <div className="pipeline-caption">Couldn&apos;t read test status: {statusError}</div>
          ) : testStatus && testStatus.ready ? (
            <>
              <Alert type="success">Test features ready:</Alert>
              <CodeBlock>{testStatus.output_path}</CodeBlock>
            </>
          ) : (
            <>
              <Alert type="info">State: {(testStatus && testStatus.slurm_state) || "waiting"}</Alert>
              <Button onClick={() => refreshStatus()}>Refresh status</Button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default function ExtractionStage({ status, submissionId, state, onChanged }) {
  const [mode, setMode] = useState("Full dataset");

  return (
    <div>
      <RadioGroup
        name={`ext-mode-${submissionId}`}
        label="Run"
        options={["Full dataset", "Test on a sample .h5"]}
        value={mode}
        onChange={setMode}
      />

      {mode === "Test on a sample .h5" ? (
        <TestExtractionPanel submissionId={submissionId} />
      ) : state === "blocked" ? (
        <div className="pipeline-caption">
          Waiting on packaging. The server requires the .h5 to be complete and readable before
          extraction can start.
        </div>
      ) : status.extraction_ready ? (
        <>
          <Alert type="success">Features ready:</Alert>
          <CodeBlock>{status.extraction_output_path}</CodeBlock>
        </>
      ) : !status.extraction_job_id ? (
        <>
          <div className="pipeline-caption">Run the packaged .h5 through the model:</div>
          <ExtractFeaturesForm submissionId={submissionId} onQueued={onChanged} />
        </>
      ) : (
        <>
          <div className="pipeline-caption">Feature extraction job:</div>
          <CodeBlock>{status.extraction_job_id}</CodeBlock>
          {SLURM_IN_FLIGHT.has(status.extraction_slurm_state) ? (
            <Alert type="info">Running (Slurm state: {status.extraction_slurm_state}).</Alert>
          ) : (
            <>
              <Alert type="warning">This attempt ended as: {status.extraction_slurm_state || "no Slurm record"}</Alert>
              {status.extraction_invalid_reason && (
                <>
                  <Alert type="error">
                    An output file exists but is not usable: {status.extraction_invalid_reason}
                  </Alert>
                  <div className="pipeline-caption">
                    Retrying clears that file first — the encoder skips its work when an output is
                    already in place, so it has to go before a rerun can succeed.
                  </div>
                </>
              )}
              <div className="pipeline-caption">Retry with the same or a different checkpoint:</div>
              <ExtractFeaturesForm submissionId={submissionId} buttonLabel="Retry feature extraction" onQueued={onChanged} />
            </>
          )}
        </>
      )}
    </div>
  );
}
