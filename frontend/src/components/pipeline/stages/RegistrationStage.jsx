// Port of _render_registration_step() in app_v28.py.
//
// This step exists because Stage 6 could not work without it and said so only
// obliquely. load_hpc_assignments.py exclusively UPDATEs tile_registry, so a
// cohort that has never been in the Knowledge Bank has nothing to update and
// the load refuses at a 0% match rate — a number that reads like a naming bug
// and is in fact a missing step.
//
// Same preview-then-commit shape as KbLoadStage, and for the same reason: it
// writes to the shared Knowledge Bank, and its failure mode is a registration
// that succeeds against the wrong cohort. The commit button is disabled
// outright whenever the preview reports a state the CLI's own guard would
// refuse — this UI must never offer a write register_dataset.py would reject.
import { useState } from "react";
import { api } from "../../../api";
import { errorDetail, fmtInt } from "../utils";
import { Alert, Button, Caption, Expander, Field, Metric } from "../widgets";

export default function RegistrationStage({ status, submissionId, state, onChanged }) {
  const defaultId =
    status.registration_dataset_id || (status.dataset_name || "").toUpperCase();

  const [datasetId, setDatasetId] = useState(defaultId);
  const [slideMetadata, setSlideMetadata] = useState(false);
  const [writeDatasetConfig, setWriteDatasetConfig] = useState(true);
  const [replace, setReplace] = useState(false);

  const [previewing, setPreviewing] = useState(false);
  const [previewError, setPreviewError] = useState(null);
  const [report, setReport] = useState(null);

  const [committing, setCommitting] = useState(false);
  const [commitMessage, setCommitMessage] = useState(null);

  async function handlePreview() {
    setPreviewError(null);
    if (!datasetId.trim()) {
      setPreviewError("Enter a dataset_id — every row written is scoped to it.");
      return;
    }
    setPreviewing(true);
    try {
      const r = await api.previewRegistration(submissionId, {
        datasetId: datasetId.trim(),
        slideMetadata,
        writeDatasetConfig,
        replace,
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
      const r = await api.commitRegistration(submissionId, {
        datasetId: datasetId.trim(),
        slideMetadata,
        writeDatasetConfig,
        replace,
      });
      const written = r.written || {};
      setCommitMessage({
        type: "success",
        text:
          "Registered: " +
          Object.entries(written)
            .map(([t, n]) => `${t} +${fmtInt(n)}`)
            .join(", "),
      });
      setReport(null);
      if (onChanged) onChanged();
    } catch (e) {
      setCommitMessage({ type: "error", text: `Registration refused: ${errorDetail(e).message}` });
    } finally {
      setCommitting(false);
    }
  }

  if (state === "blocked") {
    return (
      <Alert type="info">
        Registration reads tile identity out of the packaged .h5, so it needs Stage 2 to have
        finished. It does not need Stages 3 or 4 — as soon as the .h5 is ready this can run, and
        Stage 6 will be waiting only on the assignment.
      </Alert>
    );
  }

  // Every reason the commit must not be offered, computed by the server so the
  // rule lives next to the guard that enforces it rather than being
  // reimplemented once per frontend.
  const blocked =
    !!report &&
    (report.would_refuse_collision ||
      report.needs_replace ||
      (report.missing_tables || []).length > 0);

  const problemLists = [
    ["ambiguous_slides", "slide id(s) match more than one file — neither is registered"],
    ["slides_without_files", "slide(s) have no raw file; their tiles register, the slide will not open"],
    ["unreadable_slides", "slide(s) could not be opened for metadata"],
    ["conflicting_samples", "slide(s) carry more than one sample_id in the .h5"],
    ["missing_slides", "slide(s) have no usable Stage 1 metadata; no coordinates for their tiles"],
  ];

  return (
    <div>
      {status.registration_done && (
        <Alert type="success">
          Registered in the Knowledge Bank
          {status.registration_dataset_id ? ` as ${status.registration_dataset_id}` : ""}
          {status.registration_at ? ` — last registered ${status.registration_at}` : ""}
        </Alert>
      )}

      <Field
        label="Knowledge Bank cohort (dataset_id)"
        help={
          "Every row this writes is scoped to this key, and a replace only ever touches its own. " +
          "It defaults to the run's dataset_name but is a different thing: dataset_name is the " +
          "folder of slides on scratch, this is the cohort the KB groups by."
        }
      >
        <input
          type="text"
          value={datasetId}
          onChange={(e) => setDatasetId(e.target.value)}
        />
      </Field>

      <div className="pipeline-checkbox-row">
        <label>
          <input
            type="checkbox"
            checked={slideMetadata}
            onChange={(e) => setSlideMetadata(e.target.checked)}
          />
          Also read slide headers into wsi_metadata
        </label>
        <span className="pipeline-field-help">
          Opens every slide file to record mpp, objective power and level dimensions. Minutes, not
          seconds, on a large cohort.
        </span>
      </div>

      <div className="pipeline-checkbox-row">
        <label>
          <input
            type="checkbox"
            checked={writeDatasetConfig}
            onChange={(e) => setWriteDatasetConfig(e.target.checked)}
          />
          Write dataset_config
        </label>
        <span className="pipeline-field-help">
          Records this cohort's target_mpp and tile size from the run's own tiling_params. Skipped
          automatically for a run that predates that column.
        </span>
      </div>

      <div className="pipeline-checkbox-row">
        <label>
          <input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} />
          Replace this cohort's existing rows
        </label>
        <span className="pipeline-field-help">
          Required to re-register a dataset_id that already has rows. Only ever deletes WHERE
          dataset_id = the key above; a tile or slide claimed by a different cohort is refused
          outright, not reassigned.
        </span>
      </div>

      <Button onClick={handlePreview} disabled={previewing}>
        {previewing ? "Previewing…" : "Preview registration"}
      </Button>
      {previewError && <Alert type="error">{previewError}</Alert>}
      {commitMessage && <Alert type={commitMessage.type}>{commitMessage.text}</Alert>}

      {!report && !previewError && (
        <Caption>
          Preview first — this writes to the shared Knowledge Bank, so it never commits without
          showing you the numbers.
        </Caption>
      )}

      {report && (
        <div>
          <div className="pipeline-metrics-row">
            <Metric label="Slides" value={fmtInt(report.slides || 0)} />
            <Metric label="Tiles in .h5" value={fmtInt(report.tiles_in_h5 || 0)} />
            <Metric label="With coordinates" value={fmtInt(report.tiles_with_coordinates || 0)} />
            <Metric label="Slides registered" value={fmtInt(report.slides_registered || 0)} />
          </div>

          {!report.slides_registered && (
            <Alert type="warning">
              No wsi_registry rows would be written — the run's raw slide directory could not be
              read. The tiles would register and Stage 6 would load, and the viewer would still 404
              on every slide in this cohort, because it resolves slide paths from wsi_registry
              alone.
            </Alert>
          )}

          {problemLists.map(([field, label]) => {
            const items = report[field] || [];
            if (!items.length) return null;
            return (
              <Expander key={field} title={`⚠️ ${fmtInt(items.length)} ${label}`}>
                {items.slice(0, 200).map((item, i) => (
                  <div key={i} className="pipeline-code">
                    {item}
                  </div>
                ))}
                {items.length > 200 && (
                  <Caption>…and {fmtInt(items.length - 200)} more</Caption>
                )}
              </Expander>
            );
          })}

          {(report.missing_tables || []).length > 0 && (
            <Alert type="error">
              This database has no {report.missing_tables.join(", ")}. Run{" "}
              <code>psql … -f backend/migrate_kb_base_tables.sql</code> first — eight of the
              Knowledge Bank's tables had no CREATE TABLE in git until that file existed.
            </Alert>
          )}
          {report.would_refuse_collision && (
            <Alert type="error">
              Refusing: some of these tiles or slides already belong to a different dataset_id. Two
              cohorts cannot claim the same tile, and overwriting would repoint the viewer at
              another cohort's files. This needs investigating, not overwriting.
            </Alert>
          )}
          {report.needs_replace && (
            <Alert type="warning">
              This dataset_id already has rows. Tick “Replace this cohort's existing rows” and
              preview again to overwrite them.
            </Alert>
          )}

          <Button kind="primary" onClick={handleCommit} disabled={blocked || committing}>
            {committing ? "Registering…" : "Register in the Knowledge Bank"}
          </Button>
        </div>
      )}
    </div>
  );
}
