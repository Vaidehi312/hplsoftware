// Port of render_wsi_upload_panel() in app/app_v28.py (lines ~115-208).
//
// Uploads queue tissue masking + tiling as a background job right after the
// file is saved; this panel polls /processing-status for progress via a
// manual "Refresh status" button — deliberately no timer here, matching the
// original (only the dataset workspace list is on an interval).
//
// Re-using a slide_id from a previous upload is allowed, but the backend
// hard-stops the first attempt with a 409 ("slide_id_exists") rather than
// silently overwriting that upload's file/mask/tiles/.h5 — this renders that
// as a warning with an explicit "Yes, overwrite" action instead of just
// letting the overwrite happen.
import { useState } from "react";
import { api } from "../../api";
import { errorDetail } from "./utils";
import { Alert, Button, Expander, Field } from "./widgets";

export default function UploadPanel() {
  const [file, setFile] = useState(null);
  const [slideId, setSlideId] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null); // { kind: "success"|"error", message, payload }
  const [conflict, setConflict] = useState(null); // { message }
  const [pending, setPending] = useState(null); // { file, slideId } stashed for a confirmed overwrite

  const [processingSlideId, setProcessingSlideId] = useState(null);
  const [statusPayload, setStatusPayload] = useState(null);
  const [statusError, setStatusError] = useState(null);
  const [statusLoading, setStatusLoading] = useState(false);

  async function doUpload(uploadFile, uploadSlideId, confirmOverwrite) {
    setBusy(true);
    try {
      const payload = await api.uploadSlide(uploadFile, uploadSlideId, confirmOverwrite);
      setResult({ kind: "success", payload });
      const returned = String(payload.slide_id || "").trim().toUpperCase();
      if (returned) {
        setProcessingSlideId(returned);
        setStatusPayload(null);
        refreshStatus(returned);
      }
      setPending(null);
      setConflict(null);
    } catch (e) {
      if (e instanceof api.ApiError && e.status === 409) {
        const detail = e.body && e.body.detail;
        const message = (detail && typeof detail === "object" && detail.message) || (typeof detail === "string" ? detail : null) || e.message;
        if (detail && typeof detail === "object" && detail.error === "slide_id_exists") {
          setConflict({ message });
          // pending file/slideId were already stashed by the caller before
          // this request went out — see handleSubmit below.
        } else {
          setResult({ kind: "error", message });
          setPending(null);
          setConflict(null);
        }
      } else {
        setResult({ kind: "error", message: errorDetail(e).message });
        setPending(null);
        setConflict(null);
      }
    } finally {
      setBusy(false);
    }
  }

  function handleSubmit() {
    if (!file) {
      setResult({ kind: "error", message: "Please choose a WSI file first." });
      return;
    }
    setPending({ file, slideId });
    doUpload(file, slideId, false);
  }

  function handleConfirmOverwrite() {
    if (!pending) {
      setResult({ kind: "error", message: "Original file is no longer available — please re-select it and try again." });
      setConflict(null);
      return;
    }
    doUpload(pending.file, pending.slideId, true);
  }

  function handleCancelConflict() {
    setConflict(null);
    setPending(null);
  }

  async function refreshStatus(slide) {
    const target = slide || processingSlideId;
    if (!target) return;
    setStatusLoading(true);
    try {
      const payload = await api.getProcessingStatus(target);
      setStatusPayload(payload);
      setStatusError(null);
    } catch (e) {
      setStatusError(errorDetail(e).message);
    } finally {
      setStatusLoading(false);
    }
  }

  const stage = (statusPayload && statusPayload.status) || "unknown";

  return (
    <Expander title="Upload new WSI" defaultOpen={false}>
      <div className="pipeline-caption">
        Step 1: Click &quot;Browse files&quot; and select a WSI from your computer
      </div>

      <Field label="Browse for WSI file">
        <input
          type="file"
          accept=".svs,.ndpi,.tif,.tiff,.isyntax"
          onChange={(e) => setFile(e.target.files && e.target.files[0] ? e.target.files[0] : null)}
        />
      </Field>

      <Field label="Slide ID (optional)">
        <input
          type="text"
          placeholder="Example: TCGA-XX-XXXX-DX1"
          value={slideId}
          onChange={(e) => setSlideId(e.target.value)}
        />
      </Field>

      <Button kind="primary" onClick={handleSubmit} disabled={busy}>
        {busy ? "Uploading…" : "Send selected file to server"}
      </Button>

      {result && result.kind === "success" && (
        <>
          <Alert type="success">Slide uploaded successfully.</Alert>
          <pre className="pipeline-code">{JSON.stringify(result.payload, null, 2)}</pre>
        </>
      )}
      {result && result.kind === "error" && <Alert type="error">{result.message}</Alert>}

      {conflict && (
        <>
          <Alert type="warning">{conflict.message}</Alert>
          <div className="pipeline-btn-row">
            <Button kind="primary" onClick={handleConfirmOverwrite} disabled={busy}>
              Yes, overwrite
            </Button>
            <Button onClick={handleCancelConflict} disabled={busy}>
              Cancel
            </Button>
          </div>
        </>
      )}

      {processingSlideId && (
        <>
          <hr className="pipeline-divider" />
          <div className="pipeline-caption pipeline-caption-strong">
            Background processing: <strong>{processingSlideId}</strong>
          </div>

          <img
            className="pipeline-upload-thumb"
            src={api.thumbnailUrl(processingSlideId, 600)}
            alt={`${processingSlideId} thumbnail`}
            onError={(e) => {
              // Slide isn't openable yet or thumbnail generation hiccuped —
              // not worth blocking on, matches the bare `except: pass` in
              // the original.
              e.target.style.display = "none";
            }}
          />

          {statusError && <Alert type="warning">Could not fetch processing status: {statusError}</Alert>}

          {stage === "done" && (
            <>
              <Alert type="success">Tissue masking + tiling complete. Tiles are ready.</Alert>
              {statusPayload && statusPayload.error && <Alert type="warning">{statusPayload.error}</Alert>}
              <Button
                onClick={() => {
                  setProcessingSlideId(null);
                  setStatusPayload(null);
                }}
              >
                Dismiss
              </Button>
            </>
          )}
          {stage === "error" && (
            <>
              <Alert type="error">Processing failed: {statusPayload && statusPayload.error}</Alert>
              <Button
                onClick={() => {
                  setProcessingSlideId(null);
                  setStatusPayload(null);
                }}
              >
                Dismiss
              </Button>
            </>
          )}
          {stage !== "done" && stage !== "error" && (
            <>
              <Alert type="info">
                Status: <strong>{stage}</strong> (queued → masking → tiling → packaging → done)
              </Alert>
              <Button onClick={() => refreshStatus()} disabled={statusLoading}>
                {statusLoading ? "Refreshing…" : "Refresh status"}
              </Button>
            </>
          )}
        </>
      )}
    </Expander>
  );
}
