// Port of render_dataset_job_panel() plus the sidebar wiring at the bottom of
// app_v28.py (render_wsi_upload_panel() + render_dataset_job_panel()).
//
// PipelinePanel is the only export the rest of the app needs — it is fully
// self-contained: upload panel, the dataset path box, the 10s-polled
// dataset workspace, and the "start a new run" form (shown directly, or
// tucked behind an explicit opt-in expander when this path already has
// runs — see _render_dataset_submit_section).
import { useEffect, useState } from "react";
import { findExistingJobForPath } from "./utils";
import { Expander } from "./widgets";
import UploadPanel from "./UploadPanel";
import DatasetWorkspace from "./DatasetWorkspace";
import NewRunForm from "./NewRunForm";
import "./pipeline.css";

function SubmitSection({ datasetPath }) {
  const [existingJob, setExistingJob] = useState(undefined); // undefined = loading
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setExistingJob(undefined);
    findExistingJobForPath(datasetPath).then((job) => {
      if (!cancelled) setExistingJob(job);
    });
    return () => {
      cancelled = true;
    };
  }, [datasetPath, refreshTick]);

  if (!datasetPath) {
    return <div className="pipeline-caption">Enter a dataset path above to start a new run.</div>;
  }

  if (existingJob === undefined) {
    return <div className="pipeline-caption">Checking for existing runs…</div>;
  }

  if (!existingJob) {
    return (
      <div>
        <div className="pipeline-caption">Start a new dataset run</div>
        <NewRunForm datasetPath={datasetPath} onSubmitted={() => setRefreshTick((t) => t + 1)} />
      </div>
    );
  }

  return (
    <div>
      <div className="pipeline-caption">
        This path already has runs — the newest went in {existingJob.submitted_at || ""}. Their
        progress is above; resume from there rather than starting again.
      </div>
      <Expander title="Submit a separate NEW run for this same path anyway">
        <div className="pipeline-caption">
          Only use this if you deliberately want a second, independent run (e.g. a different
          tissue threshold) — it will NOT resume or affect the runs shown above.
        </div>
        <NewRunForm datasetPath={datasetPath} onSubmitted={() => setRefreshTick((t) => t + 1)} />
      </Expander>
    </div>
  );
}

export default function PipelinePanel() {
  const [datasetPath, setDatasetPath] = useState("");

  return (
    <div className="pipeline-panel">
      <UploadPanel />

      <Expander title="Process a dataset" defaultOpen={false}>
        <div className="pipeline-field">
          <span className="pipeline-field-label">Dataset path</span>
          <input
            type="text"
            value={datasetPath}
            onChange={(e) => setDatasetPath(e.target.value)}
            placeholder="/mnt/cephfs-lts/long-term-scratch/users/vpandya/Radiogenomics"
          />
          <span className="pipeline-field-help">
            Full absolute path on the HPC filesystem. Enter one to see everything already done to
            it; leave empty to browse what&apos;s on record.
          </span>
        </div>

        <DatasetWorkspace path={datasetPath.trim()} />

        <hr className="pipeline-divider" />

        <SubmitSection datasetPath={datasetPath.trim()} />
      </Expander>
    </div>
  );
}
