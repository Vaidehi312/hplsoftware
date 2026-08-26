// Port of _render_kb_load_step() in app_v28.py (~lines 2695-2870).
//
// Mirrors load_hpc_assignments.py's own shape — dry-run preview, then an
// explicit commit — rather than one button, because this is the one stage
// that mutates the shared Knowledge Bank every other view reads from. It
// never auto-commits, and the commit button is disabled outright when the
// preview reports would_refuse_low_match — the UI must not allow committing
// when the CLI's own --commit guard would refuse it (95% match rate).
import { useState } from "react";
import { api } from "../../../api";
import { errorDetail } from "../utils";
import { Alert, Button, Expander, Field } from "../widgets";

export default function KbLoadStage({ status, submissionId, state }) {
  // "blocked" only means this run's own tracked assignment has no usable
  // output — loading from an explicit CSV path doesn't depend on that (it's
  // how output from Stage 4's "Test on a sample .h5" mode gets into the KB,
  // since that mode never records a path against any run).
  const blockedByDefault = state === "blocked";
  const [sourceChoice, setSourceChoice] = useState("This run's tracked assignment");
  const useManualPath = blockedByDefault || sourceChoice !== "This run's tracked assignment";

  const [manualCsvPath, setManualCsvPath] = useState("");
  const [minMargin, setMinMargin] = useState(0);
  const [previewing, setPreviewing] = useState(false);
  const [previewError, setPreviewError] = useState(null);
  const [report, setReport] = useState(null);

  const [cancerType, setCancerType] = useState("");
  const [allowUnknown, setAllowUnknown] = useState(false);
  const [skipProfiles, setSkipProfiles] = useState(false);
  const [committing, setCommitting] = useState(false);
  const [commitMessage, setCommitMessage] = useState(null);

  async function handlePreview() {
    setPreviewError(null);
    if (useManualPath && !manualCsvPath.trim()) {
      setPreviewError("Enter the assignment CSV path.");
      return;
    }
    setPreviewing(true);
    try {
      const r = await api.previewKbLoad(submissionId, {
        minMargin: Number(minMargin),
        csvPath: useManualPath ? manualCsvPath.trim() || null : null,
      });
      setReport(r);
      setCommitMessage(null);
    } catch (e) {
      setPreviewError(`Preview failed: ${errorDetail(e).message}`);
      setReport(null);
    } finally {
      setPreviewing(false);
    }
  }

  async function handleCommit() {
    setCommitMessage(null);
    setCommitting(true);
    try {
      const result = await api.commitKbLoad(submissionId, {
        cancerType: cancerType.trim() || null,
        allowUnknownClusters: allowUnknown,
        skipProfiles,
        minMargin: Number(minMargin),
        csvPath: useManualPath ? manualCsvPath.trim() || null : null,
      });
      let msg = `Committed ${result.updated_rows.toLocaleString()} tile(s) to the Knowledge Bank.`;
      if (result.excluded_from_aggregates) {
        msg += ` ${result.excluded_from_aggregates.toLocaleString()} tile(s) below margin ${result.min_margin} were excluded from the aggregates.`;
      }
      if (result.recorded_on_run === false) {
        msg += " Not recorded against this run's own KB-load status, since the CSV wasn't this run's tracked output.";
      }
      setCommitMessage({ type: "success", text: msg });
      setReport(null);
    } catch (e) {
      setCommitMessage({ type: "error", text: `Load failed: ${errorDetail(e).message}` });
    } finally {
      setCommitting(false);
    }
  }

  return (
    <div>
      {status.kb_load_done && (
        <>
          <Alert type="success">
            Loaded into the Knowledge Bank
            {status.kb_load_rows != null ? `: ${status.kb_load_rows.toLocaleString()} tiles` : ""}
            {status.kb_load_reference ? ` (reference \`${status.kb_load_reference}\`)` : ""}
          </Alert>
          {status.kb_load_at && <div className="pipeline-caption">Last loaded: {status.kb_load_at}</div>}
          <div className="pipeline-caption">If Stage 4 has been re-run since, preview below and reload.</div>
        </>
      )}

      {!blockedByDefault && (
        <div className="pipeline-field">
          <span className="pipeline-field-label">Source</span>
          <div className="pipeline-radio-group">
            {["This run's tracked assignment", "A specific CSV path"].map((opt) => (
              <label className="pipeline-radio-option" key={opt}>
                <input
                  type="radio"
                  name={`kb-load-source-${submissionId}`}
                  checked={sourceChoice === opt}
                  onChange={() => setSourceChoice(opt)}
                />
                {opt}
              </label>
            ))}
          </div>
          <span className="pipeline-field-help">
            Use &quot;A specific CSV path&quot; for output from Stage 4&apos;s &quot;Test on a
            sample .h5&quot; mode — that mode never records a path against this run, so
            there&apos;s nothing tracked to load from automatically.
          </span>
        </div>
      )}

      {useManualPath && (
        <>
          {blockedByDefault && (
            <Alert type="info">
              This run has no tracked assignment yet. Load from a specific CSV instead —
              typically output from Stage 4&apos;s &quot;Test on a sample .h5&quot; mode — or
              finish Stage 4&apos;s &quot;Full dataset&quot; option first.
            </Alert>
          )}
          <Field
            label="Assignment CSV path"
            help="This still writes to the Knowledge Bank for real — it just isn't recorded against this run's own KB-load status, since the CSV may not be this run's tracked output."
          >
            <input
              type="text"
              placeholder="/path/to/DS_hpc_assignments.csv"
              value={manualCsvPath}
              onChange={(e) => setManualCsvPath(e.target.value)}
            />
          </Field>
        </>
      )}

      <Field
        label={`Exclude tiles below this vote_margin from the aggregates: ${minMargin}`}
        help="Leave-one-out validation against the reference: margin below 0.1 was 57% correct, 0.1-0.25 was 76%, 0.25+ was 92%+. tile_registry keeps every tile's own hpc_id and margin regardless of this — it only changes what counts toward the per-slide composition the chatbot and HPC panels read. 0 (default) excludes nothing."
      >
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={minMargin}
          onChange={(e) => setMinMargin(Number(e.target.value))}
        />
      </Field>

      <Button kind="primary" onClick={handlePreview} disabled={previewing}>
        {previewing ? "Checking…" : "Preview Knowledge Bank load"}
      </Button>
      {previewError && <Alert type="error">{previewError}</Alert>}

      {report && (
        <div>
          <div className="pipeline-caption pipeline-caption-strong">
            <strong>{report.rows.toLocaleString()}</strong> rows in the CSV · cluster column{" "}
            <code>{report.cluster_column}</code> · reference <code>{report.reference}</code>
          </div>
          <div className="pipeline-caption pipeline-caption-strong">
            Matched <strong>{report.matched.toLocaleString()}/{report.rows.toLocaleString()}</strong>{" "}
            ({(report.match_rate * 100).toFixed(1)}%) tiles in <code>tile_registry</code>
          </div>
          {report.unmatched > 0 && (
            <Alert type="warning">
              {report.unmatched.toLocaleString()} unmatched, e.g. {JSON.stringify(report.unmatched_examples)}
            </Alert>
          )}
          {report.overwriting > 0 && (
            <Alert type="info">
              Will overwrite {report.overwriting.toLocaleString()} tile(s) that already carry a cluster ID
              {report.overwriting_other_reference
                ? ` (${report.overwriting_other_reference.toLocaleString()} from a different reference)`
                : ""}
            </Alert>
          )}
          {report.unknown_clusters && report.unknown_clusters.length > 0 && (
            <Alert type="warning">
              {report.unknown_clusters.length} cluster ID(s) have no <code>hpc_dictionary</code> row:{" "}
              {JSON.stringify(report.unknown_clusters.slice(0, 10))}. Those tiles would show a
              cluster with no pattern or malignancy annotation unless allowed below.
            </Alert>
          )}
          {report.low_margin > 0 && (
            <div className="pipeline-caption">{report.low_margin.toLocaleString()} tile(s) have vote_margin below 0.1</div>
          )}
          {report.min_margin > 0 && (
            <Alert type="info">
              At the {report.min_margin} threshold above,{" "}
              <strong>{report.excluded_from_aggregates.toLocaleString()}</strong> tile(s) would be
              excluded from the per-slide aggregates (tile_registry keeps them regardless).
            </Alert>
          )}
          {report.would_refuse_low_match && (
            <Alert type="error">
              Match rate {(report.match_rate * 100).toFixed(1)}% is below the required{" "}
              {(report.min_match_rate * 100).toFixed(0)}% — committing would be refused the same
              way the CLI refuses it.
            </Alert>
          )}

          <Expander title="Commit options">
            <Field
              label="Cancer type (optional — fills hpl_profile_summary.cancer_type)"
              help="Left blank leaves it unset rather than guessing, same as the CLI."
            >
              <input type="text" placeholder="LUAD" value={cancerType} onChange={(e) => setCancerType(e.target.value)} />
            </Field>
            <div className="pipeline-checkbox-row">
              <input
                type="checkbox"
                id={`kb-allow-unknown-${submissionId}`}
                checked={allowUnknown}
                disabled={!(report.unknown_clusters && report.unknown_clusters.length)}
                onChange={(e) => setAllowUnknown(e.target.checked)}
              />
              <label htmlFor={`kb-allow-unknown-${submissionId}`}>
                Load cluster IDs with no hpc_dictionary row anyway
              </label>
            </div>
            <div className="pipeline-checkbox-row">
              <input
                type="checkbox"
                id={`kb-skip-profiles-${submissionId}`}
                checked={skipProfiles}
                onChange={(e) => setSkipProfiles(e.target.checked)}
              />
              <label htmlFor={`kb-skip-profiles-${submissionId}`}>
                Skip refreshing the per-slide aggregates (tile_registry only)
              </label>
            </div>
            <div className="pipeline-caption">
              Leaves hpl_profile_proportion/summary disagreeing with the new tile_registry values.
              Off unless you have a specific reason.
            </div>
          </Expander>

          <Button kind="primary" onClick={handleCommit} disabled={committing || report.would_refuse_low_match}>
            {committing ? "Committing…" : "Commit to Knowledge Bank"}
          </Button>
        </div>
      )}

      {commitMessage && <Alert type={commitMessage.type}>{commitMessage.text}</Alert>}
    </div>
  );
}
