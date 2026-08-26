// Port of _render_tiling_step() in app_v28.py (~lines 1396-1446).
import { useState } from "react";
import { api } from "../../../api";
import { errorDetail, fmtInt, resumeResultMessage } from "../utils";
import { Alert, Button, CodeBlock } from "../widgets";

export default function TilingStage({ status, submissionId, onChanged }) {
  const [showZeroTile, setShowZeroTile] = useState(false);
  const [resumeMsg, setResumeMsg] = useState(null);
  const [resuming, setResuming] = useState(false);

  const tilingJobIds = (status.job_id || "").split(",").filter(Boolean);
  const counts = status.slurm_state_counts || {};
  const total = status.total_slides;
  const succeeded = status.succeeded;
  const zeroTile = status.zero_tile_slides || [];
  const notAttempted = status.not_yet_attempted || 0;

  async function handleResume() {
    setResuming(true);
    setResumeMsg(null);
    try {
      const result = await api.resumeDatasetJob(submissionId);
      const described = resumeResultMessage(result);
      setResumeMsg(described);
      if (described.queued) onChanged && onChanged();
    } catch (e) {
      setResumeMsg({ queued: false, alert: { type: "error", text: `Resume failed: ${errorDetail(e).message}` } });
    } finally {
      setResuming(false);
    }
  }

  return (
    <div>
      {tilingJobIds.length > 0 && (
        <>
          <div className="pipeline-caption">
            Slurm job{tilingJobIds.length > 1 ? "s" : ""} ({tilingJobIds.length} batch
            {tilingJobIds.length > 1 ? "es" : ""}):
          </div>
          <CodeBlock>{tilingJobIds.join("\n")}</CodeBlock>
        </>
      )}

      {Object.keys(counts).length > 0 && (
        <div className="pipeline-caption">
          Task states:{" "}
          {Object.entries(counts)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([s, n]) => `${fmtInt(n)}x ${s}`)
            .join(", ")}
        </div>
      )}

      {status.tiling_complete && succeeded != null && (
        <>
          {notAttempted > 0 && (
            <Alert type="warning">
              {notAttempted} of {total} slides produced no usable output. Resume them before
              packaging, or the .h5 will be missing them.
            </Alert>
          )}
          {zeroTile.length > 0 && (
            <>
              <Alert type="warning">
                {zeroTile.length} of {total} slides ran but saved zero tiles (no tissue above the
                minimum-tissue threshold). Resubmitting won&apos;t change this — check the masking /
                min-tissue settings if they should have tissue.
              </Alert>
              <div className="pipeline-checkbox-row">
                <input
                  type="checkbox"
                  id={`zero-tile-${submissionId}`}
                  checked={showZeroTile}
                  onChange={(e) => setShowZeroTile(e.target.checked)}
                />
                <label htmlFor={`zero-tile-${submissionId}`}>
                  Show the {zeroTile.length} zero-tile slide ID(s)
                </label>
              </div>
              {showZeroTile && <CodeBlock>{zeroTile.join("\n")}</CodeBlock>}
            </>
          )}
          {notAttempted === 0 && zeroTile.length === 0 && (
            <Alert type="success">All {succeeded} slides have tiles.</Alert>
          )}
        </>
      )}

      <Button onClick={handleResume} disabled={resuming}>
        {resuming ? "Resuming…" : "Resume missing slides"}
      </Button>

      {resumeMsg && (
        <>
          <Alert type={resumeMsg.alert.type}>{resumeMsg.alert.text}</Alert>
          {resumeMsg.note && <Alert type={resumeMsg.note.type}>{resumeMsg.note.text}</Alert>}
          {resumeMsg.params && Object.keys(resumeMsg.params).length > 0 && (
            <CodeBlock>
              {Object.entries(resumeMsg.params)
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([k, v]) => `${k} = ${v}`)
                .join("\n")}
            </CodeBlock>
          )}
        </>
      )}
    </div>
  );
}
