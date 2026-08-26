// Port of the packaging stage in app_v28.py: _render_packaging_step and
// everything it delegates to (~lines 1115-2228) — full-dataset packaging,
// "Test on a subset", the is_subset "package the whole directory" detour,
// the resumable-checkpoint radio+confirm flow, and the two dedicated
// progress polls (test packaging at 15s, live packaging at 30s).
import { useEffect, useState } from "react";
import { api } from "../../../api";
import { usePolling } from "../../../hooks/usePolling";
import { errorDetail, fmtInt, fullPackagingScopeCaption, humanBytes, SLURM_IN_FLIGHT } from "../utils";
import { Alert, Button, CodeBlock, Expander, Field, Metric, ProgressBar, RadioGroup } from "../widgets";

// -- Start/retry button, shared by every packaging path ---------------------
// Covers a first attempt, a retry, and a resume identically (the server
// decides which it is). A refusal because tiling didn't fully succeed
// ("tiling_incomplete") is caught specially and turned into the two real
// choices — fix tiling, or package without those slides — instead of a
// raw 400.
function StartPackagingButton({ submissionId, label = "Start packaging (.h5)", allowIncomplete = false, resume, disabled, onQueued }) {
  const [busy, setBusy] = useState(false);
  const [blocked, setBlocked] = useState(null);
  const [error, setError] = useState(null);

  async function start(allowInc) {
    setBusy(true);
    setError(null);
    try {
      await api.startPackagingJob(submissionId, { allowIncomplete: allowInc, resume });
      setBlocked(null);
      onQueued && onQueued();
    } catch (e) {
      const { detail, message } = errorDetail(e);
      if (detail && detail.error === "tiling_incomplete") {
        setBlocked(detail);
      } else {
        setError(`Failed to start packaging: ${message}`);
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <Button kind="primary" onClick={() => start(allowIncomplete)} disabled={disabled || busy}>
        {busy ? "Starting…" : label}
      </Button>
      {error && <Alert type="error">{error}</Alert>}
      {blocked && (
        <>
          <Alert type="warning">{blocked.message || "Some tiling tasks did not complete."}</Alert>
          {blocked.failed_task_states && Object.keys(blocked.failed_task_states).length > 0 && (
            <div className="pipeline-caption">
              Failed tiling tasks:{" "}
              {Object.entries(blocked.failed_task_states)
                .map(([s, n]) => `${n}x ${s}`)
                .join(", ")}
            </div>
          )}
          <div className="pipeline-caption">
            Either resume the tiling step above to retry those slides, or package without them —
            the .h5 will be missing their tiles.
          </div>
          <Button onClick={() => start(true)} disabled={busy}>
            Package anyway (accept missing slides)
          </Button>
        </>
      )}
    </div>
  );
}

// -- Resumable-checkpoint decision -----------------------------------------
// A checkpoint on disk no longer silently continues — resuming keeps
// whatever an earlier attempt wrote, starting fresh throws it away, and
// neither belongs in a button label. Both are now a deliberate, confirmed
// choice; the destructive one needs a second confirmation.
function ResumeChoice({ progress, submissionId, onQueued }) {
  const done = progress.tiles_done || 0;
  const totalTiles = progress.tiles_total;
  const percent = progress.percent;
  const partial = humanBytes(progress.partial_bytes || 0);
  const resumeLabel = `Resume — keep the ${fmtInt(done)} tiles already written`;
  const freshLabel = "Start fresh — discard them and repackage everything";
  const [choice, setChoice] = useState(resumeLabel);
  const [confirmFresh, setConfirmFresh] = useState(false);
  const resuming = choice === resumeLabel;

  return (
    <div>
      <Alert type="warning">
        <strong>An unfinished attempt is on disk.</strong> {fmtInt(done)} tiles were already
        written ({partial}). Choose what to do with it — nothing is resumed automatically.
      </Alert>
      {totalTiles ? (
        <ProgressBar fraction={(percent || 0) / 100} label={`${fmtInt(done)} / ${fmtInt(totalTiles)} tiles (${percent}%)`} />
      ) : null}
      <RadioGroup
        name={`pkg-resume-choice-${submissionId}`}
        label="This attempt"
        options={[resumeLabel, freshLabel]}
        value={choice}
        onChange={setChoice}
      />
      {resuming ? (
        <>
          <div className="pipeline-caption">
            Packaging continues from tile {fmtInt(done)}. Tiles already written are not re-decoded,
            so this is much faster — but it also keeps whatever that attempt wrote. Start fresh
            instead if the tiles on disk have changed since it ran.
          </div>
          <StartPackagingButton
            submissionId={submissionId}
            label={`Resume packaging (${fmtInt(done)} tiles already done)`}
            resume
            onQueued={onQueued}
          />
        </>
      ) : (
        <>
          <Alert type="error">
            Starting fresh deletes the checkpoint and the {partial} `.partial`, and re-decodes all{" "}
            {fmtInt(done)} tiles already done. This cannot be undone.
          </Alert>
          <div className="pipeline-checkbox-row">
            <input
              type="checkbox"
              id={`pkg-fresh-confirm-${submissionId}`}
              checked={confirmFresh}
              onChange={(e) => setConfirmFresh(e.target.checked)}
            />
            <label htmlFor={`pkg-fresh-confirm-${submissionId}`}>
              Yes, discard {fmtInt(done)} tiles and repackage from scratch
            </label>
          </div>
          <StartPackagingButton
            submissionId={submissionId}
            label="Discard and repackage from scratch"
            resume={false}
            disabled={!confirmFresh}
            onQueued={onQueued}
          />
        </>
      )}
    </div>
  );
}

// -- Live progress while a packaging job is running (30s poll) --------------
// exact=false so this is affordable on a timer: it estimates tiles written
// from the .partial's size rather than counting a completed-tiles checkpoint
// that can reach hundreds of MB.
function PackagingLiveProgress({ submissionId }) {
  const { data: progress, error } = usePolling(() => api.getPackagingProgress(submissionId, false), {
    intervalMs: 30000,
    deps: [submissionId],
  });
  if (error) return <div className="pipeline-caption">Couldn&apos;t read packaging progress: {error.message}</div>;
  if (!progress) return <div className="pipeline-caption">Loading progress…</div>;

  const done = progress.tiles_done;
  const total = progress.tiles_total;
  const percent = progress.percent;
  const estimated = progress.tiles_done_is_estimate;
  const approx = estimated ? "~" : "";

  return (
    <div>
      {done != null && total ? (
        <ProgressBar fraction={(percent || 0) / 100} label={`${approx}${fmtInt(done)} / ${fmtInt(total)} tiles (${percent}%)`} />
      ) : done != null ? (
        <div className="pipeline-caption">
          {approx}
          {fmtInt(done)} tiles written (total unknown).
        </div>
      ) : null}

      <div className="pipeline-metrics-row">
        <Metric
          label=".h5 written so far"
          value={humanBytes(progress.partial_bytes || 0)}
          caption={total && progress.bytes_per_tile ? `Projected final size: ${humanBytes(total * progress.bytes_per_tile)}` : null}
        />
        <Metric
          label="Last write"
          value={
            progress.seconds_since_write == null || progress.seconds_since_write < 60
              ? "just now"
              : `${Math.floor(progress.seconds_since_write / 60)} min ago`
          }
          caption={progress.skipped_tiles ? `${fmtInt(progress.skipped_tiles)} tiles skipped (unreadable JPEGs).` : null}
        />
      </div>

      {estimated && (
        <div className="pipeline-caption">
          Tile count is estimated from the file&apos;s size — every tile occupies exactly the same
          number of bytes, so it&apos;s close, but the exact figure comes from the checkpoint and is
          only read when deciding a resume.
        </div>
      )}
      {!progress.writing_now && (
        <Alert type="warning">
          Nothing has been written for a while — the job may have been killed. Slurm state:{" "}
          {progress.slurm_state || "unavailable"}.
        </Alert>
      )}
      <div className="pipeline-caption">
        Output: <code>{progress.output_path}</code>
      </div>
    </div>
  );
}

// -- Test packaging progress (15s poll) -------------------------------------
function TestPackagingProgress({ submissionId, currentParams }) {
  const { data: status, error } = usePolling(() => api.getDatasetJobStatus(submissionId), {
    intervalMs: 15000,
    deps: [submissionId],
  });
  if (error) return <div className="pipeline-caption">Couldn&apos;t read test packaging status: {error.message}</div>;
  if (!status || !status.test_h5_job_id) return null;

  const params = status.test_h5_params || {};
  const scopeLabel = params.scope === "tiled" ? "everything tiled on disk" : "this run's manifest";

  let stale = false;
  if (currentParams && Object.keys(params).length) {
    const recordedMode = params.sample_size ? "Random N" : "Specific slides";
    const recordedScope = params.scope || "run";
    const recordedSample = params.sample_size || null;
    const recordedNames = JSON.stringify(params.slide_names || []);
    stale =
      recordedMode !== currentParams.mode ||
      recordedScope !== currentParams.scope ||
      recordedSample !== currentParams.sampleSize ||
      recordedNames !== JSON.stringify(currentParams.slideNames || []);
  }

  return (
    <div>
      {stale && (
        <Alert type="warning">
          The job below is from an earlier setup, not the selection currently in the form. Click{" "}
          <strong>Start test packaging</strong> to run the current one.
        </Alert>
      )}

      <div className="pipeline-caption">Test packaging job:</div>
      <CodeBlock>{status.test_h5_job_id}</CodeBlock>

      {status.test_h5_ready ? (
        <>
          <Alert type="success">Test packaging finished — .h5 is readable and ready to use.</Alert>
          <CodeBlock>{status.test_h5_output_path}</CodeBlock>
          <div className="pipeline-caption">
            {[
              params.sample_size ? `${params.sample_size} slides drawn from ${scopeLabel}` : null,
              params.pool_size ? `pool was ${params.pool_size}` : null,
              params.random_seed != null ? `seed ${params.random_seed}` : null,
            ]
              .filter(Boolean)
              .join(" · ")}
          </div>
          <div className="pipeline-caption">
            Use this path as the &quot;Test .h5 path&quot; in feature extraction&apos;s &quot;Test on a
            sample .h5&quot; step.
          </div>
        </>
      ) : status.test_h5_invalid_reason ? (
        <Alert type="error">Test .h5 is not usable: {status.test_h5_invalid_reason}</Alert>
      ) : status.test_writing_now ? (
        <Alert type="info">
          Packaging in progress ({status.test_h5_slurm_state || "no Slurm record yet"}) — the .h5
          is being written to right now.
        </Alert>
      ) : !status.test_h5_slurm_state ? (
        <Alert type="info">Queued — no Slurm accounting record yet.</Alert>
      ) : (
        <Alert type="info">State: {status.test_h5_slurm_state}</Alert>
      )}
    </div>
  );
}

// -- "Test on a subset" mode -------------------------------------------------
function TestPackagingPanel({ submissionId, status, onChanged }) {
  const [poolChoice, setPoolChoice] = useState("This run's slides");
  const [mode, setMode] = useState("Random N");
  const [sampleSize, setSampleSize] = useState(3);
  const [seedInput, setSeedInput] = useState("");
  const [namesRaw, setNamesRaw] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState(null);
  const [infoNote, setInfoNote] = useState(null);
  const [justSubmittedJobId, setJustSubmittedJobId] = useState(null);

  const scope = poolChoice === "This run's slides" ? "run" : "tiled";
  const slideNames = mode === "Specific slides" ? namesRaw.split("\n").map((s) => s.trim()).filter(Boolean) : null;
  const currentParams = {
    mode,
    scope,
    sampleSize: mode === "Random N" && sampleSize ? Number(sampleSize) : null,
    slideNames: slideNames || [],
  };

  async function handleStart() {
    setMessage(null);
    setInfoNote(null);
    if (mode === "Specific slides" && (!slideNames || !slideNames.length)) {
      setMessage({ type: "error", text: "Enter at least one slide." });
      return;
    }
    if (seedInput.trim() && !/^-?\d+$/.test(seedInput.trim())) {
      setMessage({ type: "error", text: "Random seed must be a whole number, or left blank." });
      return;
    }
    setSubmitting(true);
    try {
      const result = await api.startTestPackaging(submissionId, {
        sampleSize: currentParams.sampleSize,
        slideNames,
        randomSeed: seedInput.trim() ? Number(seedInput.trim()) : null,
        scope,
      });
      setJustSubmittedJobId(result.h5_job_id);
      setMessage({ type: "success", text: `Test packaging queued (job ${result.h5_job_id}).` });
      const bits = [];
      if (result.pool_size != null) {
        const poolLabel = result.scope === "tiled" ? "everything tiled on disk" : "this run's manifest";
        bits.push(`Drawn from ${result.pool_size} slides (${poolLabel}).`);
      }
      if (result.random_seed != null) {
        bits.push(`Random seed ${result.random_seed} — keep this to repackage the same slides.`);
      }
      setInfoNote(bits.join(" "));
      onChanged && onChanged();
    } catch (e) {
      const { detail, message: msg } = errorDetail(e);
      setMessage({ type: "error", text: `Test packaging failed: ${msg}` });
      if (detail && detail.error === "sample_larger_than_pool" && detail.scope === "run") {
        setInfoNote('Switch "Draw from" to "Everything tiled on disk" to sample beyond this run\'s own slides.');
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div>
      <div className="pipeline-caption">
        Package a handful of slides into a separate .h5 to check the pipeline before committing to
        a multi-hour full run. Tracked separately — it neither blocks nor is blocked by the real
        packaging job.
      </div>
      <div className="pipeline-caption">
        A test run does <strong>not</strong> advance the pipeline — stage 3 still waits on{" "}
        <strong>Full dataset</strong> packaging. Use it to validate a checkpoint, not to produce
        the dataset&apos;s .h5.
      </div>

      <RadioGroup
        name={`pkg-test-scope-${submissionId}`}
        label="Draw from"
        options={["This run's slides", "Everything tiled on disk"]}
        value={poolChoice}
        onChange={setPoolChoice}
        help="This run's slides is limited to the manifest fixed when it was submitted. Everything tiled on disk covers the whole raw directory, whichever run (or resume, or hand-run job) produced the tiles."
      />

      <RadioGroup
        name={`pkg-test-mode-${submissionId}`}
        label="Pick slides"
        options={["Random N", "Specific slides"]}
        value={mode}
        onChange={setMode}
      />

      {mode === "Random N" ? (
        <>
          <Field label="How many slides">
            <input type="number" min={1} value={sampleSize} onChange={(e) => setSampleSize(e.target.value)} />
          </Field>
          <Field
            label="Random seed (optional)"
            help="Leave blank for a fresh draw each time. Paste a seed reported by an earlier run to package that exact set of slides again — though an identical seed and count counts as the same attempt and will be refused as a duplicate."
          >
            <input type="text" value={seedInput} onChange={(e) => setSeedInput(e.target.value)} />
          </Field>
        </>
      ) : (
        <Field label="Slide IDs / filenames (one per line)">
          <textarea rows={3} value={namesRaw} onChange={(e) => setNamesRaw(e.target.value)} />
        </Field>
      )}

      <Button kind="primary" onClick={handleStart} disabled={submitting}>
        {submitting ? "Starting…" : "Start test packaging"}
      </Button>
      {message && <Alert type={message.type}>{message.text}</Alert>}
      {infoNote && <div className="pipeline-caption">{infoNote}</div>}

      {status.test_h5_job_id ? (
        <TestPackagingProgress submissionId={submissionId} currentParams={currentParams} />
      ) : justSubmittedJobId ? (
        <div>
          <div className="pipeline-caption">Test packaging job:</div>
          <CodeBlock>{justSubmittedJobId}</CodeBlock>
        </div>
      ) : null}
    </div>
  );
}

// -- Plain (re)package this run's own manifest -------------------------------
function RunScopedPackaging({ status, submissionId, covered, onQueued }) {
  if (status.h5_ready) {
    return (
      <div>
        <Alert type="success">Already packaged:</Alert>
        <CodeBlock>{status.h5_output_path}</CodeBlock>
        <div className="pipeline-caption">
          Repackaging would rebuild this .h5 from the same slides. Use the full-coverage option
          above if you want more of them.
        </div>
      </div>
    );
  }
  return (
    <div>
      <div className="pipeline-caption">{fullPackagingScopeCaption(status)}</div>
      <StartPackagingButton submissionId={submissionId} label={`Package this run's ${covered} slides`} onQueued={onQueued} />
    </div>
  );
}

// -- "Full dataset" for a run that was submitted as a subset -----------------
// Packaging can only ever cover slides this run tiled, so honouring the
// "Full dataset" label means starting a *new* tiling run over the whole
// directory (submit_mask_tile_slurm.py skips slides already tiled, so only
// the untiled remainder actually runs) rather than re-packaging the subset.
function FullDatasetRun({ status, submissionId, onQueued }) {
  const rawDir = status.raw_dir || "";
  const datasetName = status.dataset_name || rawDir.split("/").filter(Boolean).pop() || "";
  const total = status.total_slides;
  const covered = total != null ? fmtInt(total) : "some";

  const [coverage, setCoverage] = useState(undefined); // undefined = loading, null = failed to load
  useEffect(() => {
    let cancelled = false;
    setCoverage(undefined);
    api
      .getTiledCoverage(submissionId)
      .then((c) => {
        if (!cancelled) setCoverage(c);
      })
      .catch(() => {
        if (!cancelled) setCoverage(null);
      });
    return () => {
      cancelled = true;
    };
  }, [submissionId]);

  const [prior, setPrior] = useState(null);
  useEffect(() => {
    let cancelled = false;
    api
      .getPackagingProgress(submissionId)
      .then((p) => {
        if (!cancelled) setPrior(p);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [submissionId]);

  const [resumeChoiceTiled, setResumeChoiceTiled] = useState("resume");
  const [confirmFreshTiled, setConfirmFreshTiled] = useState(false);
  const [tiledBusy, setTiledBusy] = useState(false);
  const [tiledMsg, setTiledMsg] = useState(null);

  const [confirmTileRest, setConfirmTileRest] = useState(false);
  const [minTissueRest, setMinTissueRest] = useState(30);
  const [tileRestBusy, setTileRestBusy] = useState(false);
  const [tileRestMsg, setTileRestMsg] = useState(null);

  async function submitTiledScopePackaging(allowIncomplete, resume) {
    setTiledBusy(true);
    setTiledMsg(null);
    try {
      const result = await api.startPackagingJob(submissionId, { allowIncomplete, scope: "tiled", resume });
      setTiledMsg({ type: "success", text: `Packaging queued (job ${result.h5_job_id}).`, path: result.h5_output_path });
      onQueued && onQueued();
    } catch (e) {
      const { detail, message } = errorDetail(e);
      if (detail && detail.error === "tiling_incomplete") {
        setTiledMsg({ type: "error", text: detail.message || message });
      } else {
        setTiledMsg({ type: "error", text: `Failed to start packaging: ${message}` });
      }
    } finally {
      setTiledBusy(false);
    }
  }

  if (coverage === undefined) {
    return <div className="pipeline-caption">Checking which slides have tiles on disk…</div>;
  }

  if (coverage === null) {
    return (
      <div>
        <Alert type="warning">
          Couldn&apos;t read tile coverage from disk, so there&apos;s no way to tell here how much
          of the directory is tiled. Package this run&apos;s {covered} slides below, or check the
          dataset folder directly.
        </Alert>
        <RunScopedPackaging status={status} submissionId={submissionId} covered={covered} onQueued={onQueued} />
      </div>
    );
  }

  const inDir = coverage.slides_in_directory || 0;
  const tiled = coverage.slides_tiled || 0;
  const untiled = coverage.slides_untiled || 0;
  const freshTiledChosen = resumeChoiceTiled === "fresh";
  const tiledButtonDisabled = tiledBusy || (prior && prior.resumable && freshTiledChosen && !confirmFreshTiled);
  const tiledButtonLabel =
    prior && prior.resumable && !freshTiledChosen
      ? `Resume packaging (${fmtInt(prior.tiles_done || 0)} done)`
      : `Package all ${fmtInt(tiled)} tiled slides`;

  return (
    <div>
      <div className="pipeline-caption">
        Directory: <code>{rawDir}</code> · dataset folder <code>{datasetName}</code>
      </div>

      {untiled === 0 && tiled ? (
        <Alert type="success">
          All <strong>{fmtInt(tiled)} slides</strong> in this directory are tiled on disk. No
          tiling needed — package them all now.
        </Alert>
      ) : (
        <>
          <Alert type="warning">
            <strong>
              {fmtInt(tiled)} of {fmtInt(inDir)}
            </strong>{" "}
            slides in this directory have tiles on disk. The remaining{" "}
            <strong>{fmtInt(untiled)}</strong> need tiling before they can be packaged.
          </Alert>
          {coverage.untiled_sample && coverage.untiled_sample.length > 0 && (
            <Expander title={`Show untiled slides (${fmtInt(untiled)})`}>
              <CodeBlock>{coverage.untiled_sample.join("\n")}</CodeBlock>
              {untiled > coverage.untiled_sample.length && (
                <div className="pipeline-caption">
                  Showing the first {coverage.untiled_sample.length} of {fmtInt(untiled)}.
                </div>
              )}
            </Expander>
          )}
        </>
      )}

      {tiled > 0 && (
        <div>
          <div className="pipeline-caption">
            Packages all {fmtInt(tiled)} tiled slides into <code>{datasetName}</code> — a separate
            output from this run&apos;s own subset .h5, which is left untouched.
          </div>

          {prior && prior.resumable && (
            <>
              <Alert type="warning">
                An unfinished attempt at this output is already on disk — {fmtInt(prior.tiles_done || 0)}{" "}
                tiles ({humanBytes(prior.partial_bytes || 0)}). Choose what happens to it; nothing is
                resumed automatically.
              </Alert>
              <div className="pipeline-caption">
                <code>{prior.output_path}</code>
              </div>
              <RadioGroup
                name={`pkg-tiled-resume-${submissionId}`}
                label="That attempt"
                options={[
                  { value: "resume", label: `Resume — keep the ${fmtInt(prior.tiles_done || 0)} tiles already written` },
                  { value: "fresh", label: "Start fresh — discard them and repackage everything" },
                ]}
                value={resumeChoiceTiled}
                onChange={setResumeChoiceTiled}
              />
              {freshTiledChosen && (
                <>
                  <Alert type="error">
                    This deletes the checkpoint and the {humanBytes(prior.partial_bytes || 0)}{" "}
                    `.partial`, and re-decodes all {fmtInt(prior.tiles_done || 0)} tiles. It cannot
                    be undone.
                  </Alert>
                  <div className="pipeline-checkbox-row">
                    <input
                      type="checkbox"
                      id={`pkg-tiled-fresh-${submissionId}`}
                      checked={confirmFreshTiled}
                      onChange={(e) => setConfirmFreshTiled(e.target.checked)}
                    />
                    <label htmlFor={`pkg-tiled-fresh-${submissionId}`}>
                      Yes, discard {fmtInt(prior.tiles_done || 0)} tiles and start over
                    </label>
                  </div>
                </>
              )}
            </>
          )}

          <Button
            kind="primary"
            disabled={tiledButtonDisabled}
            onClick={() =>
              submitTiledScopePackaging(untiled > 0, prior && prior.resumable ? !freshTiledChosen : false)
            }
          >
            {tiledBusy ? "Starting…" : tiledButtonLabel}
          </Button>
          {tiledMsg && <Alert type={tiledMsg.type}>{tiledMsg.text}</Alert>}
          {tiledMsg && tiledMsg.path && <CodeBlock>{tiledMsg.path}</CodeBlock>}

          {untiled > 0 && (
            <div className="pipeline-caption">
              This packages only the {fmtInt(tiled)} that are ready; the {fmtInt(untiled)} untiled
              slides are left out of the .h5.
            </div>
          )}
        </div>
      )}

      {untiled === 0 ? (
        <Expander title={`Or just (re)package this run's ${covered} slides`}>
          <RunScopedPackaging status={status} submissionId={submissionId} covered={covered} onQueued={onQueued} />
        </Expander>
      ) : (
        <>
          <hr className="pipeline-divider" />
          <div className="pipeline-caption">
            <strong>Tile the {fmtInt(untiled)} missing slides</strong> — starts a new tiling run:
          </div>

          {status.tiling_params ? (
            <>
              <div className="pipeline-caption">
                Reusing this run&apos;s own tiling settings, so the new slides match the ones
                already tiled:
              </div>
              <CodeBlock>
                {Object.entries(status.tiling_params)
                  .sort(([a], [b]) => a.localeCompare(b))
                  .map(([k, v]) => `${k} = ${v}`)
                  .join("\n")}
              </CodeBlock>
            </>
          ) : (
            <>
              <Alert type="warning">
                This run has no recorded tiling settings (it predates them being saved), so they
                can&apos;t be inherited. Set the minimum tissue % to whatever the original run
                used — a different value here leaves the dataset tiled two different ways.
              </Alert>
              <Field label={`Minimum tissue % per tile (for the slides not yet tiled): ${minTissueRest}`}>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={5}
                  value={minTissueRest}
                  onChange={(e) => setMinTissueRest(Number(e.target.value))}
                />
              </Field>
            </>
          )}

          <div className="pipeline-checkbox-row">
            <input
              type="checkbox"
              id={`tile-rest-confirm-${submissionId}`}
              checked={confirmTileRest}
              onChange={(e) => setConfirmTileRest(e.target.checked)}
            />
            <label htmlFor={`tile-rest-confirm-${submissionId}`}>
              Yes — queue tiling for the rest of the directory
            </label>
          </div>
          <div className="pipeline-caption">
            Submits a new tiling run on the HPC. It does not touch this run or its .h5, and it does
            not package anything.
          </div>

          <Button
            kind="primary"
            disabled={!confirmTileRest || tileRestBusy}
            onClick={async () => {
              setTileRestBusy(true);
              setTileRestMsg(null);
              try {
                const result = await api.submitDatasetJob({
                  dataset_path: rawDir,
                  min_tissue: status.tiling_params ? null : Number(minTissueRest),
                  tiling_params: status.tiling_params || null,
                  sample_size: null,
                  slide_names: null,
                  partition: status.partition || null,
                  notify_email: status.notify_email || null,
                  dataset_name: datasetName || null,
                });
                setTileRestMsg({
                  type: "success",
                  text:
                    `Tiling queued for the whole directory — submission ${result.submission_id}. ` +
                    `Find it in the dataset workspace; package it from that run's step 2 once its ` +
                    `tiling finishes. This run and its .h5 are untouched.`,
                });
              } catch (e) {
                setTileRestMsg({ type: "error", text: `Couldn't start the full run: ${errorDetail(e).message}` });
              } finally {
                setTileRestBusy(false);
              }
            }}
          >
            {tileRestBusy ? "Submitting…" : "Tile the rest of the directory (new run)"}
          </Button>
          {tileRestMsg && <Alert type={tileRestMsg.type}>{tileRestMsg.text}</Alert>}

          <Expander title={`Or just (re)package this run's ${covered} slides`}>
            <RunScopedPackaging status={status} submissionId={submissionId} covered={covered} onQueued={onQueued} />
          </Expander>
        </>
      )}
    </div>
  );
}

// -- In-flight / interrupted branch for a normal (non-subset) run -----------
function PackagingInFlightOrRetry({ status, submissionId, onChanged }) {
  const h5JobId = status.h5_job_id;
  const h5State = status.h5_slurm_state;
  const running = SLURM_IN_FLIGHT.has(h5State) || status.h5_packaging_active;
  const [progress, setProgress] = useState(undefined);

  useEffect(() => {
    if (running) return;
    let cancelled = false;
    setProgress(undefined);
    api
      .getPackagingProgress(submissionId)
      .then((p) => {
        if (!cancelled) setProgress(p);
      })
      .catch(() => {
        if (!cancelled) setProgress(null);
      });
    return () => {
      cancelled = true;
    };
  }, [submissionId, running]);

  return (
    <div>
      <div className="pipeline-caption">Packaging job:</div>
      <CodeBlock>{h5JobId}</CodeBlock>

      {running ? (
        <>
          {SLURM_IN_FLIGHT.has(h5State) ? (
            <Alert type="info">Running (Slurm state: {h5State}).</Alert>
          ) : (
            <Alert type="info">
              Running — the .h5 is being written to right now (Slurm state unavailable, judged
              from the output file).
            </Alert>
          )}
          <PackagingLiveProgress submissionId={submissionId} />
        </>
      ) : (
        <>
          {status.h5_invalid_reason ? (
            <Alert type="error">The .h5 exists but is not usable: {status.h5_invalid_reason}</Alert>
          ) : (
            <Alert type="warning">Previous attempt ended as: {h5State || "no Slurm record"}</Alert>
          )}
          {progress === undefined ? (
            <div className="pipeline-caption">Checking for a resumable checkpoint…</div>
          ) : progress && progress.resumable ? (
            <ResumeChoice progress={progress} submissionId={submissionId} onQueued={onChanged} />
          ) : (
            <>
              <div className="pipeline-caption">No resumable checkpoint on disk — this will start from the beginning.</div>
              <StartPackagingButton submissionId={submissionId} label="Retry packaging (.h5)" resume={false} onQueued={onChanged} />
            </>
          )}
        </>
      )}
    </div>
  );
}

// -- Top level ----------------------------------------------------------
export default function PackagingStage({ status, submissionId, state, onChanged }) {
  // Mode selector is rendered before any "blocked" check: test packaging is
  // explicitly designed to run while tiling is still in flight
  // (allow_incomplete=true), so hiding the whole selector on "blocked" made
  // neither option reachable during most of a long run's lifetime.
  const [mode, setMode] = useState(() => (status.test_h5_job_id && !status.h5_job_id ? "Test on a subset" : "Full dataset"));

  return (
    <div>
      <RadioGroup
        name={`pkg-mode-${submissionId}`}
        label="Run"
        options={["Full dataset", "Test on a subset"]}
        value={mode}
        onChange={setMode}
      />

      {mode === "Test on a subset" ? (
        <TestPackagingPanel submissionId={submissionId} status={status} onChanged={onChanged} />
      ) : status.is_subset ? (
        <FullDatasetRun status={status} submissionId={submissionId} onQueued={onChanged} />
      ) : state === "blocked" ? (
        <div className="pipeline-caption">
          Waiting on tiling. Full-dataset packaging can start once every tiling task has reached a
          terminal state — or use &quot;Test on a subset&quot; above to check a few slides now.
        </div>
      ) : status.h5_ready ? (
        <>
          <Alert type="success">Packaged .h5 ready:</Alert>
          <CodeBlock>{status.h5_output_path}</CodeBlock>
        </>
      ) : !status.h5_job_id ? (
        <>
          <div className="pipeline-caption">{fullPackagingScopeCaption(status)}</div>
          {status.slurm_unreachable && (
            <Alert type="warning">
              The server can&apos;t reach Slurm (sacct/squeue), so tiling&apos;s state couldn&apos;t
              be verified — starting packaging is allowed, but check tiling really has finished
              first. The submitted job carries its own Slurm dependency as a backstop.
            </Alert>
          )}
          <StartPackagingButton submissionId={submissionId} onQueued={onChanged} />
        </>
      ) : (
        <PackagingInFlightOrRetry status={status} submissionId={submissionId} onChanged={onChanged} />
      )}
    </div>
  );
}
