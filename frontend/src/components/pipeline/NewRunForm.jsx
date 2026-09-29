// Port of _render_new_dataset_submission_form() in app_v28.py: one path, one
// click — Stages 1-4 as a Nextflow run on the server's own settings.
//
// Nothing is asked for but the path. The tile folder is the directory's own
// name; checkpoint, reference, vote, output locations and concurrency are the
// server's (GET /pipeline-defaults) and shown beside the button so a click is
// never a guess about what it will use. The server still checks every one of
// them before queueing anything.
import { useEffect, useState } from "react";
import { api } from "../../api";
import { httpDetail } from "./utils";
import { Alert, Button, Caption } from "./widgets";

export default function NewRunForm({ datasetPath, onSubmitted }) {
  const [defaults, setDefaults] = useState(null);
  const [defaultsError, setDefaultsError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState(null); // {type, text}
  const [superseded, setSuperseded] = useState([]);

  useEffect(() => {
    let cancelled = false;
    api
      .getPipelineDefaults()
      .then((d) => !cancelled && setDefaults(d))
      .catch((e) => !cancelled && setDefaultsError(String(e.message || e)));
    return () => {
      cancelled = true;
    };
  }, []);

  const name = (datasetPath || "").replace(/\/+$/, "").split("/").pop() || "?";

  async function handleSubmit() {
    setMessage(null);
    setSuperseded([]);
    if (!datasetPath.trim()) {
      setMessage({ type: "error", text: "Enter a dataset path first." });
      return;
    }
    setSubmitting(true);
    try {
      const result = await api.startPipelineRun({ dataset_path: datasetPath.trim() });
      setMessage({
        type: "success",
        text:
          `Pipeline queued — run ${result.submission_id} for '${result.dataset_name}'. Slides are ` +
          `being found and the head job submitted; its progress appears below.`,
      });
      setSuperseded(result.superseded || []);
      if (onSubmitted) onSubmitted(result);
    } catch (e) {
      setMessage({ type: "error", text: `Refused: ${httpDetail(e)}` });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div>
      {defaults ? (
        <Caption>
          Tiles into <code>{defaults.tile_root}/{name}</code>, the .h5 into{" "}
          <code>{defaults.h5_root}/{name}</code> · checkpoint <code>{defaults.checkpoint}</code> · reference{" "}
          <code>{defaults.reference}</code> · {defaults.vote_preset} vote · {defaults.max_tiling} slides at a
          time · {defaults.min_tissue}% minimum tissue. Slides already tiled are reused; earlier outputs in the
          way are moved aside, never deleted.
        </Caption>
      ) : defaultsError ? (
        <Caption>Could not load the pipeline&apos;s settings ({defaultsError}); the server applies them anyway.</Caption>
      ) : null}

      <Button kind="primary" onClick={handleSubmit} disabled={submitting}>
        {submitting ? "Checking and submitting…" : "Run pipeline"}
      </Button>

      {message && <Alert type={message.type}>{message.text}</Alert>}
      {superseded.length > 0 && (
        <Caption>Moved aside (nothing deleted): {superseded.map((m) => `${m.from} → ${m.to}`).join("; ")}</Caption>
      )}
    </div>
  );
}
