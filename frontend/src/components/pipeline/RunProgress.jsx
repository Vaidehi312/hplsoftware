// Port of _render_job_progress() in app_v28.py (~lines 2872-2954): one run
// rendered as five stage steps (each expandable, with its own state icon and
// summary), the Slurm job history, and the "Stop run" button.
//
// Polls /dataset-jobs/{id}/status every 10s — in the original this whole
// tree sits inside _render_dataset_workspace's own 10s @st.fragment, so a
// stage advancing shows up without any other interaction.
import { useEffect, useRef, useState } from "react";
import { api } from "../../api";
import { usePolling } from "../../hooks/usePolling";
import {
  describeJobParams,
  displayStage,
  errorDetail,
  jobLabel,
  pipelineSteps,
  RUN_STATE_ICONS,
  STAGE_LABELS,
  STEP_ICON,
} from "./utils";
import { Alert, Button, Expander } from "./widgets";
import TilingStage from "./stages/TilingStage";
import PackagingStage from "./stages/PackagingStage";
import ExtractionStage from "./stages/ExtractionStage";
import AssignmentStage from "./stages/AssignmentStage";
import RegistrationStage from "./stages/RegistrationStage";
import KbLoadStage from "./stages/KbLoadStage";

// Port of _render_job_history(): every Slurm job this run has submitted,
// across every stage — the run's own status fields only hold one attempt
// per stage, so this is the only place an earlier attempt (and the .h5 path
// it produced) is still visible.
function JobHistory({ submissionId }) {
  const [history, setHistory] = useState(undefined);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getDatasetJobHistory(submissionId)
      .then((h) => {
        if (!cancelled) setHistory(h.jobs || []);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e.message || e));
      });
    return () => {
      cancelled = true;
    };
  }, [submissionId]);

  if (error) return <div className="pipeline-caption">Couldn&apos;t load job history: {error}</div>;
  if (history === undefined) return <div className="pipeline-caption">Loading…</div>;
  if (!history.length) {
    return (
      <div className="pipeline-caption">
        No Slurm jobs recorded for this run yet. Runs submitted before the job-history migration
        have none — their current stage still shows above.
      </div>
    );
  }

  return (
    <div>
      {history.map((job, i) => {
        const stage = job.stage || "?";
        const stateStr = String(job.slurm_state || "unknown").toLowerCase();
        const icon = RUN_STATE_ICONS[stateStr] || "❔";
        const when = (job.submitted_at || "").slice(0, 16).replace("T", " ");
        const batches = job.batch_count || 1;
        let header = `${icon} ${STAGE_LABELS[stage] || stage} · ${stateStr}`;
        if (batches > 1) header += ` · ${batches} batches`;
        if (when) header += ` · ${when}`;
        const detail = describeJobParams(stage, job.params);
        return (
          <div key={i} style={{ marginBottom: 8 }}>
            <div className="pipeline-caption pipeline-caption-strong">
              <strong>{header}</strong>
            </div>
            {detail && <div className="pipeline-caption">{detail}</div>}
            {job.output_path && (
              <div className="pipeline-caption">
                → <code>{job.output_path}</code>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default function RunProgress({ submissionId, job }) {
  const { data: status, error, loading, refresh } = usePolling(
    () => api.getDatasetJobStatus(submissionId),
    { intervalMs: 10000, deps: [submissionId] }
  );

  // Toast-on-transition: a short-lived banner rather than Streamlit's
  // st.toast, since there's no global toast host here.
  const [toast, setToast] = useState(null);
  const prevStageRef = useRef(null);
  useEffect(() => {
    if (!status) return;
    const stage = displayStage(status);
    if (prevStageRef.current !== null && prevStageRef.current !== stage) {
      setToast(`${prevStageRef.current} → ${stage}`);
      const t = setTimeout(() => setToast(null), 6000);
      prevStageRef.current = stage;
      return () => clearTimeout(t);
    }
    prevStageRef.current = stage;
  }, [status]);

  const [cancelBusy, setCancelBusy] = useState(false);
  const [cancelMessage, setCancelMessage] = useState(null);

  async function handleCancel() {
    setCancelBusy(true);
    setCancelMessage(null);
    try {
      const result = await api.cancelDatasetJob(submissionId);
      if (result.cancelled_job_ids && result.cancelled_job_ids.length) {
        setCancelMessage({ type: "success", text: `Cancelled: ${result.cancelled_job_ids.join(", ")}` });
      } else {
        setCancelMessage({ type: "info", text: "Marked cancelled (no Slurm jobs had been queued yet)." });
      }
      refresh();
    } catch (e) {
      setCancelMessage({ type: "error", text: `Stop failed: ${errorDetail(e).message}` });
    } finally {
      setCancelBusy(false);
    }
  }

  if (error && !status) {
    return <Alert type="warning">Could not load status: {error.message}</Alert>;
  }
  if (loading && !status) {
    return <div className="pipeline-caption">Loading run status…</div>;
  }
  if (!status) return null;

  const totalSlides = job && job.total_slides != null ? job.total_slides : status.total_slides;
  const submittedAt = (job && job.submitted_at) || status.submitted_at || "";
  const stage = status.status;
  const steps = pipelineSteps(status);

  const renderers = {
    tiling: () => <TilingStage status={status} submissionId={submissionId} onChanged={refresh} />,
    packaging: (s) => <PackagingStage status={status} submissionId={submissionId} state={s.state} onChanged={refresh} />,
    extraction: (s) => <ExtractionStage status={status} submissionId={submissionId} state={s.state} onChanged={refresh} />,
    assignment: (s) => <AssignmentStage status={status} submissionId={submissionId} state={s.state} onChanged={refresh} />,
    registration: (s) => <RegistrationStage status={status} submissionId={submissionId} state={s.state} onChanged={refresh} />,
    kb_load: (s) => <KbLoadStage status={status} submissionId={submissionId} state={s.state} />,
  };

  return (
    <div className="pipeline-run">
      <div className="pipeline-caption">
        Submitted: {submittedAt} ·{" "}
        {totalSlides != null ? `${Number(totalSlides).toLocaleString()} slides` : "discovering slides"}
      </div>

      {toast && (
        <div className="pipeline-toast">
          Job {job ? jobLabel(job) : submissionId.slice(0, 8)}: {toast}
        </div>
      )}

      {stage === "error" ? (
        <Alert type="error">Run failed: {status.error}</Alert>
      ) : status.error ? (
        <Alert type="warning">{status.error}</Alert>
      ) : null}
      {stage === "cancelled" && (
        <Alert type="info">This run was cancelled. Stages already finished on disk are still usable.</Alert>
      )}

      {steps.map((step) => (
        <Expander
          key={step.key}
          defaultOpen={["action", "attention", "failed", "blocked"].includes(step.state)}
          resetKey={`${step.state}|${step.summary}`}
          title={
            <span className="pipeline-step-header">
              <span className="pipeline-step-icon">{STEP_ICON[step.state] || "•"}</span>
              <span className="pipeline-step-title">{step.title}</span>
              <span className="pipeline-step-summary">— {step.summary}</span>
            </span>
          }
        >
          {renderers[step.key](step)}
        </Expander>
      ))}

      <Expander title="Slurm job history">
        <JobHistory submissionId={submissionId} />
      </Expander>

      {stage !== "error" && stage !== "cancelled" && (
        <Button kind="danger" onClick={handleCancel} disabled={cancelBusy}>
          {cancelBusy ? "Stopping…" : "Stop run"}
        </Button>
      )}
      {cancelMessage && <Alert type={cancelMessage.type}>{cancelMessage.text}</Alert>}
    </div>
  );
}
